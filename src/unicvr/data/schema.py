from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MultiVideoSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    video_paths: list[Path] = Field(min_length=2)
    question: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    answer: str | None = None
    task_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("video_paths")
    @classmethod
    def existing_unique_videos(cls, paths: list[Path]) -> list[Path]:
        resolved = [path.expanduser().resolve() for path in paths]
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise ValueError(f"video files do not exist: {missing}")
        if len(set(resolved)) != len(resolved):
            raise ValueError("video_paths must identify distinct files")
        return resolved

    def inference_payload(self) -> dict[str, Any]:
        """Intentionally excludes ground truth."""
        return self.model_dump(exclude={"answer"})


def load_sample(path: Path, field_mapping: dict[str, str] | None = None) -> MultiVideoSample:
    if not path.is_file():
        raise FileNotFoundError(f"sample file does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sample JSON root must be an object")
    mapping = field_mapping or {}
    normalized: dict[str, Any] = {}
    for canonical in MultiVideoSample.model_fields:
        source = mapping.get(canonical, canonical)
        if source in raw:
            normalized[canonical] = raw[source]
    raw_paths = normalized.get("video_paths")
    if isinstance(raw_paths, list):
        normalized["video_paths"] = [
            _resolve_video_path(Path(item), sample_path=path) for item in raw_paths
        ]
    return MultiVideoSample.model_validate(normalized)


def _resolve_video_path(video_path: Path, *, sample_path: Path) -> Path:
    expanded = video_path.expanduser()
    if expanded.is_absolute() or expanded.is_file():
        return expanded
    beside_sample = sample_path.resolve().parent / expanded
    return beside_sample if beside_sample.is_file() else expanded
