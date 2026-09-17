from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from unicvr.core.schemas import GenerationConfig

_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["mock", "openai_compatible"]
    base_url: str | None = None
    api_key_env: str | None = None
    model: str
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_api_fields(self) -> BackendConfig:
        if self.backend == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible backend requires base_url")
        if self.backend == "openai_compatible" and not self.api_key_env:
            raise ValueError("openai_compatible backend requires api_key_env")
        return self


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_frames: int = Field(gt=0)
    max_estimated_visual_tokens: int = Field(gt=0)
    max_visual_calls: int = Field(gt=0)
    min_warm_frames_per_video: int = Field(gt=0)
    scan_frames_per_video: int = Field(gt=0)
    default_focus_frames: int = Field(gt=0)
    single_timestamp_half_window_seconds: float = Field(default=16.0, gt=0)
    no_window_focus_frames_per_round: int = Field(default=16, gt=0)
    max_comparisons_per_round: int = Field(default=6, gt=0)
    max_comparison_rounds: int = Field(default=3, gt=0)
    estimated_tokens_per_frame: int = Field(default=256, gt=0)

    @model_validator(mode="after")
    def validate_partitions(self) -> BudgetConfig:
        if self.scan_frames_per_video < self.min_warm_frames_per_video:
            raise ValueError("scan_frames_per_video must satisfy minimum warm coverage")
        return self


class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_dir: Path = Path(".unicvr_cache/frames")
    resize_long_edge: int = Field(default=768, gt=0)
    jpeg_quality: int = Field(default=88, ge=1, le=100)
    candidate_multiplier: int = Field(default=3, ge=1)
    scan_seconds_per_frame: float | None = Field(default=None, gt=0)
    scan_frames_per_call: int = Field(default=12, ge=2)
    scan_overlap_frames: int = Field(default=1, ge=0)
    coverage_weight: float = 1.0
    change_weight: float = 0.35
    question_weight: float = 0.15
    redundancy_weight: float = 0.25
    pre_context_ratio: float = Field(default=0.25, ge=0, le=1)
    post_context_ratio: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def validate_scan_chunks(self) -> VideoConfig:
        if self.scan_overlap_frames >= self.scan_frames_per_call:
            raise ValueError("scan_overlap_frames must be smaller than scan_frames_per_call")
        return self


class TraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("outputs")
    record_raw_responses: bool = True
    max_raw_response_chars: int = Field(default=8000, gt=0)


class TimingGuardConfig(BaseModel):
    """Extreme-only duration-feedback plugin for interval (FSA) answers.

    Only intervals outside the dataset's normal band are touched, so the
    model keeps its sense of the real duration distribution: overlong
    (>= extreme_overlong_seconds) gets "compress" feedback, implausibly
    short (<= extreme_undershort_seconds) gets "re-derive boundaries"
    feedback. FSA GT band: p10=6s .. p90=45s, median 18s.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    extreme_overlong_seconds: float = Field(default=45.0, gt=0)
    extreme_undershort_seconds: float = Field(default=6.0, gt=0)
    max_feedback_rounds: int = Field(default=2, ge=1, le=3)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: BackendConfig
    vlm: BackendConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    budget: BudgetConfig
    video: VideoConfig = Field(default_factory=VideoConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    timing_guard: TimingGuardConfig = Field(default_factory=TimingGuardConfig)
    field_mapping: dict[str, str] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        match = _ENV_PATTERN.match(value)
        if match:
            return os.environ.get(match.group(1), value)
    return value


def load_config(path: Path) -> AppConfig:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    load_dotenv(path.parent / ".env")
    load_dotenv(Path(".env"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return AppConfig.model_validate(_expand_env(raw))


def validate_credentials(config: AppConfig) -> list[str]:
    missing = []
    for name, backend in (("llm", config.llm), ("vlm", config.vlm)):
        if backend.backend == "openai_compatible":
            assert backend.api_key_env is not None
            if not os.environ.get(backend.api_key_env):
                missing.append(f"{name}: environment variable {backend.api_key_env}")
            if backend.model.startswith("${"):
                missing.append(f"{name}: unresolved model environment variable")
            if backend.base_url and backend.base_url.startswith("${"):
                missing.append(f"{name}: unresolved base_url environment variable")
    return missing
