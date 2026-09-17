from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from unicvr.agents import ReasonerAgent
from unicvr.config import load_config
from unicvr.core.pipeline import Pipeline
from unicvr.core.schemas import GenerationConfig, VisualInput
from unicvr.core.state import ObserverReport, PipelineState
from unicvr.data.schema import load_sample
from unicvr.models.openai_compatible import (
    OpenAICompatibleLLMBackend,
    OpenAICompatibleVLMBackend,
    StructuredOutputError,
    extract_json,
)

ROOT = Path(__file__).resolve().parents[1]


class TinyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


def completion(content: str, *, prompt_tokens: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 2,
            },
        },
    )


def test_json_extraction_accepts_fenced_or_embedded_object() -> None:
    assert extract_json('```json\n{"value": 3}\n```') == {"value": 3}
    assert extract_json('prefix {"value": 4} suffix') == {"value": 4}


def test_structured_repair_is_bounded_and_recorded_once() -> None:
    responses = iter(
        [
            completion('{"value":"invalid"}', prompt_tokens=3),
            completion('{"value":7}', prompt_tokens=4),
        ]
    )
    backend = OpenAICompatibleLLMBackend(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(lambda request: next(responses)),
    )
    result = backend.generate_structured(
        role="test",
        system_prompt="system",
        user_prompt="user",
        output_schema=TinyResponse,
        generation_config=GenerationConfig(repair_retries=1, max_retries=0),
    )
    assert result.value == 7
    assert len(backend.calls) == 1
    assert backend.calls[0].retry_count == 1
    assert backend.calls[0].usage.input_tokens == 7


def test_json_extraction_from_model_output_is_robust() -> None:
    """Test extract_json handles various model output formats."""
    # Markdown-fenced JSON
    assert extract_json('```json\n{"value": 3}\n```') == {"value": 3}
    # JSON embedded in text
    assert extract_json('prefix {"value": 4} suffix') == {"value": 4}
    # Direct JSON
    assert extract_json('{"value": 5}') == {"value": 5}


def test_repeated_same_role_calls_each_get_a_record() -> None:
    backend = OpenAICompatibleLLMBackend(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(lambda request: completion('{"value":1}')),
    )
    for _ in range(2):
        backend.generate_structured(
            role="same-role",
            system_prompt="system",
            user_prompt="user",
            output_schema=TinyResponse,
            generation_config=GenerationConfig(),
        )
    assert len(backend.calls) == 2


def test_malformed_output_fails_without_accepting_or_leaking_secret() -> None:
    secret = "provider-secret-value"
    backend = OpenAICompatibleLLMBackend(
        base_url="https://example.invalid/v1",
        api_key=secret,
        model="test-model",
        transport=httpx.MockTransport(lambda request: completion(f"malformed {secret}")),
    )
    with pytest.raises(StructuredOutputError, match="validation failed"):
        backend.generate_structured(
            role="test",
            system_prompt="system",
            user_prompt="user",
            output_schema=TinyResponse,
            generation_config=GenerationConfig(repair_retries=0),
        )
    assert len(backend.calls) == 1
    serialized = backend.calls[0].model_dump_json()
    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_reasoner_handles_malformed_output_gracefully(
    completed_run: tuple[Pipeline, PipelineState],
) -> None:
    pipeline, _ = completed_run
    backend = OpenAICompatibleLLMBackend(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(lambda request: completion("not valid JSON")),
    )
    reasoner = ReasonerAgent(
        backend,
        GenerationConfig(repair_retries=0, max_retries=0),
    )
    # Reasoner uses generate_text + extract_json — malformed output
    # should produce a parse-error plan with confidence=0, not crash
    state = reasoner.form(
        question="test",
        answer_type="choice",
        reports=[],
    )
    assert state.decision == "UNCERTAIN"
    assert backend.calls[-1].error is None  # text calls don't error on malformed JSON


def test_vlm_serializes_explicit_labels_and_images(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg-placeholder")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return completion('{"value":9}')

    backend = OpenAICompatibleVLMBackend(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="vision-model",
        transport=httpx.MockTransport(handler),
    )
    result = backend.generate_structured(
        role="verify",
        system_prompt="system",
        user_prompt="user",
        visual_inputs=[
            VisualInput(
                video_id="v1",
                timestamp_seconds=1.25,
                frame_index=10,
                local_path=image,
                label="VIDEO_A | original_video_id=v1 | time=1.25s",
            )
        ],
        output_schema=TinyResponse,
        generation_config=GenerationConfig(),
    )
    content = captured["messages"][2]["content"]
    assert result.value == 9
    assert any(
        item.get("text", "").startswith("VIDEO_A") for item in content if item["type"] == "text"
    )
    assert any(
        item.get("image_url", {}).get("url", "").startswith("data:image/jpeg;base64,")
        for item in content
        if item["type"] == "image_url"
    )


@pytest.mark.api_smoke
def test_real_api_end_to_end_when_explicitly_enabled(tmp_path: Path) -> None:
    if os.environ.get("UNICVR_RUN_API_SMOKE") != "1":
        pytest.skip("set UNICVR_RUN_API_SMOKE=1 to authorize external API calls")
    required = {
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "VLM_BASE_URL",
        "VLM_API_KEY",
        "VLM_MODEL",
    }
    missing = sorted(name for name in required if not os.environ.get(name))
    if missing:
        pytest.skip(f"API smoke credentials are absent: {', '.join(missing)}")
    config = load_config(ROOT / "configs" / "api_budgeted.example.yaml")
    config = config.model_copy(
        update={"video": config.video.model_copy(update={"cache_dir": tmp_path / "frames"})}
    )
    state = Pipeline(config).run(
        load_sample(ROOT / "tests" / "fixtures" / "mock_sample.json")
    )
    assert state.final_answer is not None
    assert len(state.api_calls) > 0
