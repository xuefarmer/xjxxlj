from __future__ import annotations

import os

from unicvr.config import AppConfig, BackendConfig
from unicvr.models.base import LLMBackend, VLMBackend
from unicvr.models.mock import MockLLMBackend, MockVLMBackend
from unicvr.models.openai_compatible import (
    OpenAICompatibleLLMBackend,
    OpenAICompatibleVLMBackend,
)


def build_llm(config: AppConfig) -> LLMBackend:
    backend = _build(config.llm, config, multimodal=False)
    if not isinstance(backend, (MockLLMBackend, OpenAICompatibleLLMBackend)):
        raise TypeError("configured LLM backend does not implement the text interface")
    return backend


def build_vlm(config: AppConfig) -> VLMBackend:
    backend = _build(config.vlm, config, multimodal=True)
    if not isinstance(backend, (MockVLMBackend, OpenAICompatibleVLMBackend)):
        raise TypeError("configured VLM backend does not implement the visual interface")
    return backend


def _build(backend: BackendConfig, app: AppConfig, *, multimodal: bool) -> LLMBackend | VLMBackend:
    if backend.backend == "mock":
        return MockVLMBackend(backend.model) if multimodal else MockLLMBackend(backend.model)
    assert backend.api_key_env is not None
    key = os.environ.get(backend.api_key_env)
    if not key:
        raise ValueError(f"missing API credential in environment variable {backend.api_key_env}")
    if multimodal:
        return OpenAICompatibleVLMBackend(
            base_url=str(backend.base_url),
            api_key=key,
            model=backend.model,
            extra_headers=backend.extra_headers,
            record_raw_responses=app.trace.record_raw_responses,
            max_raw_response_chars=app.trace.max_raw_response_chars,
        )
    return OpenAICompatibleLLMBackend(
        base_url=str(backend.base_url),
        api_key=key,
        model=backend.model,
        extra_headers=backend.extra_headers,
        record_raw_responses=app.trace.record_raw_responses,
        max_raw_response_chars=app.trace.max_raw_response_chars,
    )
