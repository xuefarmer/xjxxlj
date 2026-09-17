from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel

from unicvr.core.schemas import (
    BackendCallRecord,
    GenerationConfig,
    VisualInput,
)

T = TypeVar("T", bound=BaseModel)


class LLMBackend(Protocol):
    calls: list[BackendCallRecord]

    def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        generation_config: GenerationConfig,
    ) -> T: ...

    def generate_text(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        generation_config: GenerationConfig,
    ) -> str: ...


class VLMBackend(Protocol):
    calls: list[BackendCallRecord]

    def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        visual_inputs: Sequence[VisualInput],
        output_schema: type[T],
        generation_config: GenerationConfig,
    ) -> T: ...

    def generate_text(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        visual_inputs: Sequence[VisualInput],
        generation_config: GenerationConfig,
    ) -> str: ...
