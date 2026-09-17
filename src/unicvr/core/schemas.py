"""Minimal schemas shared across the pipeline.

Only models needed by the video layer, model backends, and data layer
are kept here.  Agent-specific request/response types live in core/state.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── video identity ──────────────────────────────────────────────────


class VideoRef(StrictModel):
    video_id: str = Field(min_length=1)
    path: Path
    duration_seconds: float | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, ge=0)
    frame_count: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── visual data passed to VLMs ──────────────────────────────────────


class VisualInput(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    video_id: str = Field(min_length=1)
    timestamp_seconds: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    local_path: Path
    mime_type: str = "image/jpeg"
    label: str | None = None


# ── generation parameters ───────────────────────────────────────────


class GenerationConfig(StrictModel):
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=1200, gt=0)
    timeout_seconds: float = Field(default=90.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    repair_retries: int = Field(default=1, ge=0, le=3)
    # Qwen3-style thinking switch. None = let the API decide (defaults on for
    # thinking models). False skips reasoning_content -> content-only output,
    # which is what the compact-reasoner protocol wants to distill.
    enable_thinking: bool | None = None


# ── API telemetry ───────────────────────────────────────────────────


class APIUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    visual_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)


class BackendCallRecord(StrictModel):
    role: str
    backend: str
    model: str
    visual_input_count: int = Field(default=0, ge=0)
    usage: APIUsage = Field(default_factory=APIUsage)
    retry_count: int = Field(default=0, ge=0)
    raw_response: str | None = None
    error: str | None = None
    # Full text prompts (system/user) captured for distillation collection.
    # Text-only; visual inputs are omitted. Filled by OpenAICompatibleBackend.
    system_prompt: str | None = None
    user_prompt: str | None = None
