from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from unicvr.core.schemas import (
    APIUsage,
    BackendCallRecord,
    GenerationConfig,
    VisualInput,
)

T = TypeVar("T", bound=BaseModel)
_SENSITIVE = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\"'\s,}]+|"
    r"\b(?:sk|hf)_[A-Za-z0-9_-]{12,}\b"
)


class StructuredOutputError(RuntimeError):
    """Raised when bounded JSON extraction and repair cannot validate output."""


def redact(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix = next((group for group in match.groups() if group), "")
        return prefix + "[REDACTED]"

    return _SENSITIVE.sub(replace, value)


def _prompt_texts(
    messages: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """Extract system + user text prompts for distillation collection.

    Visual content (image entries) is omitted — the LLM never sees it and
    it would bloat every record. Returns (system_prompt, user_prompt).
    """
    system: str | None = None
    user_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system" and isinstance(content, str):
            if system is None:
                system = content
        elif role == "user":
            if isinstance(content, str):
                user_parts.append(content)
            elif isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if texts:
                    user_parts.append("\n".join(texts))
    return system, ("\n".join(user_parts) if user_parts else None)


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise StructuredOutputError("model response contains no valid JSON object")
    if not isinstance(parsed, dict):
        raise StructuredOutputError("structured model response must be a JSON object")
    return parsed


class _OpenAICompatibleBase:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        extra_headers: dict[str, str] | None = None,
        record_raw_responses: bool = True,
        max_raw_response_chars: int = 8000,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("API key is empty")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra_headers = extra_headers or {}
        self.record_raw_responses = record_raw_responses
        self.max_raw_response_chars = max_raw_response_chars
        self.transport = transport
        self.calls: list[BackendCallRecord] = []

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _post(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        output_schema: type[T],
        generation_config: GenerationConfig,
        visual_input_count: int,
    ) -> T:
        schema = output_schema.model_json_schema()
        schema_instruction = {
            "role": "system",
            "content": (
                "Return one JSON object matching this JSON Schema exactly. "
                "Do not wrap it in markdown.\n" + json.dumps(schema, separators=(",", ":"))
            ),
        }
        request = {
            "model": self.model,
            "messages": [messages[0], schema_instruction, *messages[1:]],
            "temperature": generation_config.temperature,
            "max_tokens": generation_config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if generation_config.enable_thinking is not None:
            request["enable_thinking"] = generation_config.enable_thinking
        retry_count = 0
        raw = ""
        usage = APIUsage()
        validation_error = ""
        system_prompt, user_prompt = _prompt_texts(request["messages"])
        try:
            with httpx.Client(
                timeout=generation_config.timeout_seconds,
                transport=self.transport,
            ) as client:
                response, retries = self._request_with_retry(
                    client, request, generation_config.max_retries
                )
                retry_count += retries
                raw, usage = self._response_content(response)
                try:
                    result = output_schema.model_validate(extract_json(raw))
                except (StructuredOutputError, ValidationError) as exc:
                    validation_error = self._redact(str(exc))
                    for _repair_attempt in range(1, generation_config.repair_retries + 1):
                        repair_messages = [
                            {
                                "role": "system",
                                "content": (
                                    "Repair the candidate into valid JSON matching the supplied "
                                    "schema. Return JSON only; do not add facts."
                                ),
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "schema": schema,
                                        "validation_error": validation_error,
                                        "candidate": self._redact(raw)[
                                            : self.max_raw_response_chars
                                        ],
                                    }
                                ),
                            },
                        ]
                        repair_request = {**request, "messages": repair_messages}
                        response, retries = self._request_with_retry(
                            client,
                            repair_request,
                            generation_config.max_retries,
                        )
                        retry_count += retries + 1
                        raw, repair_usage = self._response_content(response)
                        usage = _merge_usage(usage, repair_usage)
                        try:
                            result = output_schema.model_validate(extract_json(raw))
                            break
                        except (StructuredOutputError, ValidationError) as exc:
                            validation_error = self._redact(str(exc))
                    else:
                        raise StructuredOutputError(
                            f"{output_schema.__name__} validation failed after "
                            f"{generation_config.repair_retries} repair attempt(s): "
                            f"{validation_error}"
                        )
        except Exception as exc:
            self.calls.append(
                BackendCallRecord(
                    role=role,
                    backend="openai_compatible",
                    model=self.model,
                    visual_input_count=visual_input_count,
                    usage=usage,
                    retry_count=retry_count,
                    raw_response=self._safe_raw(raw),
                    error=self._redact(str(exc)),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            )
            raise
        self.calls.append(
            BackendCallRecord(
                role=role,
                backend="openai_compatible",
                model=self.model,
                visual_input_count=visual_input_count,
                usage=usage,
                retry_count=retry_count,
                raw_response=self._safe_raw(raw),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )
        return result

    def _post_text(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        generation_config: GenerationConfig,
        visual_input_count: int,
    ) -> str:
        """Send a text-only request (no JSON schema enforcement)."""
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": generation_config.temperature,
            "max_tokens": generation_config.max_output_tokens,
        }
        if generation_config.enable_thinking is not None:
            request["enable_thinking"] = generation_config.enable_thinking
        retry_count = 0
        raw = ""
        usage = APIUsage()
        system_prompt, user_prompt = _prompt_texts(request["messages"])
        try:
            with httpx.Client(
                timeout=generation_config.timeout_seconds,
                transport=self.transport,
            ) as client:
                response, retries = self._request_with_retry(
                    client, request, generation_config.max_retries
                )
                retry_count += retries
                raw, usage = self._response_content(response)
        except Exception as exc:
            self.calls.append(
                BackendCallRecord(
                    role=role,
                    backend="openai_compatible",
                    model=self.model,
                    visual_input_count=visual_input_count,
                    usage=usage,
                    retry_count=retry_count,
                    raw_response=self._safe_raw(raw),
                    error=self._redact(str(exc)),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            )
            raise
        self.calls.append(
            BackendCallRecord(
                role=role,
                backend="openai_compatible",
                model=self.model,
                visual_input_count=visual_input_count,
                usage=usage,
                retry_count=retry_count,
                raw_response=self._safe_raw(raw),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        )
        return raw

    def _request_with_retry(
        self,
        client: httpx.Client,
        payload: dict[str, Any],
        max_retries: int,
    ) -> tuple[httpx.Response, int]:
        for attempt in range(max_retries + 1):
            try:
                response = client.post(self.endpoint, headers=self._headers(), json=payload)
                if response.status_code < 400:
                    return response, attempt
                if response.status_code not in {408, 409, 429} and response.status_code < 500:
                    raise RuntimeError(
                        f"API returned HTTP {response.status_code}: "
                        f"{self._redact(response.text[:1000])}"
                    )
                if attempt == max_retries:
                    raise RuntimeError(f"API retry limit reached after HTTP {response.status_code}")
            except httpx.TransportError as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"API transport failure after {attempt + 1} attempt(s)"
                    ) from exc
            time.sleep(min(2**attempt, 8))
        raise RuntimeError("unreachable API retry state")

    def _response_content(self, response: httpx.Response) -> tuple[str, APIUsage]:
        try:
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("API response lacks choices[0].message.content") from exc
        if isinstance(raw, list):
            raw = "".join(item.get("text", "") for item in raw if isinstance(item, dict))
        if not isinstance(raw, str):
            raise RuntimeError("API message content is not text")
        raw_usage = body.get("usage") or {}
        details = raw_usage.get("prompt_tokens_details") or {}
        usage = APIUsage(
            input_tokens=raw_usage.get("prompt_tokens"),
            output_tokens=raw_usage.get("completion_tokens"),
            visual_tokens=details.get("image_tokens"),
        )
        return raw, usage

    def _safe_raw(self, raw: str) -> str | None:
        if not self.record_raw_responses:
            return None
        return self._redact(raw)[: self.max_raw_response_chars]

    def _redact(self, value: str) -> str:
        result = value.replace(self.api_key, "[REDACTED]")
        for header_value in self.extra_headers.values():
            if len(header_value) >= 4:
                result = result.replace(header_value, "[REDACTED]")
        return redact(result)


def _merge_usage(left: APIUsage, right: APIUsage) -> APIUsage:
    def add(a: int | None, b: int | None) -> int | None:
        return None if a is None and b is None else (a or 0) + (b or 0)

    return APIUsage(
        input_tokens=add(left.input_tokens, right.input_tokens),
        output_tokens=add(left.output_tokens, right.output_tokens),
        visual_tokens=add(left.visual_tokens, right.visual_tokens),
    )


class OpenAICompatibleLLMBackend(_OpenAICompatibleBase):
    def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        generation_config: GenerationConfig,
    ) -> T:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._post(
            role=role,
            messages=messages,
            output_schema=output_schema,
            generation_config=generation_config,
            visual_input_count=0,
        )

    def generate_text(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        generation_config: GenerationConfig,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._post_text(
            role=role,
            messages=messages,
            generation_config=generation_config,
            visual_input_count=0,
        )


class OpenAICompatibleVLMBackend(_OpenAICompatibleBase):
    def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        visual_inputs: Sequence[VisualInput],
        output_schema: type[T],
        generation_config: GenerationConfig,
    ) -> T:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for item in visual_inputs:
            label = item.label or (
                f"video_id={item.video_id} | time={item.timestamp_seconds:.3f}s "
                f"| frame={item.frame_index}"
            )
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(item.local_path, item.mime_type),
                        "detail": "high",
                    },
                }
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return self._post(
            role=role,
            messages=messages,
            output_schema=output_schema,
            generation_config=generation_config,
            visual_input_count=len(visual_inputs),
        )

    def generate_text(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        visual_inputs: Sequence[VisualInput],
        generation_config: GenerationConfig,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for item in visual_inputs:
            label = item.label or (
                f"video_id={item.video_id} | time={item.timestamp_seconds:.3f}s "
                f"| frame={item.frame_index}"
            )
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(item.local_path, item.mime_type),
                        "detail": "high",
                    },
                }
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return self._post_text(
            role=role,
            messages=messages,
            generation_config=generation_config,
            visual_input_count=len(visual_inputs),
        )


def _data_url(path: Path, mime_type: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"visual input does not exist: {path}")
    mime = mime_type or mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
