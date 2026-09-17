from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from unicvr.data.schema import MultiVideoSample


def load_field_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"field mapping file does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise ValueError("field mapping must be a string-to-string YAML mapping")
    unknown = set(raw) - set(MultiVideoSample.model_fields)
    if unknown:
        raise ValueError(f"field mapping has unknown canonical fields: {sorted(unknown)}")
    return raw


class JSONDatasetAdapter:
    """Normalize JSON rows without selecting an inference pipeline."""

    def __init__(
        self,
        *,
        video_root: Path,
        field_mapping: dict[str, str],
    ) -> None:
        if not video_root.is_dir():
            raise FileNotFoundError(f"video root does not exist: {video_root}")
        self.video_root = video_root.resolve()
        self.field_mapping = field_mapping

    def iter_samples(
        self,
        dataset_path: Path,
        *,
        start: int = 0,
        limit: int | None = None,
    ) -> Iterator[MultiVideoSample]:
        if start < 0:
            raise ValueError("dataset start must be nonnegative")
        if limit is not None and limit <= 0:
            raise ValueError("dataset limit must be positive")
        if not dataset_path.is_file():
            raise FileNotFoundError(f"dataset file does not exist: {dataset_path}")
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("dataset JSON root must be a list")
        stop = len(raw) if limit is None else min(len(raw), start + limit)
        for index in range(start, stop):
            row = raw[index]
            if not isinstance(row, dict):
                raise ValueError(f"dataset row {index} must be an object")
            yield self._normalize_row(
                row,
                source_path=dataset_path,
                source_index=index,
            )

    def _normalize_row(
        self,
        row: dict[str, Any],
        *,
        source_path: Path,
        source_index: int,
    ) -> MultiVideoSample:
        normalized: dict[str, Any] = {}
        for canonical, source in self.field_mapping.items():
            if source in row:
                normalized[canonical] = row[source]
        if "sample_id" in normalized:
            normalized["sample_id"] = str(normalized["sample_id"])
        raw_paths = normalized.get("video_paths")
        if not isinstance(raw_paths, list):
            raise ValueError(f"dataset row {source_index} requires a list-valued video field")
        normalized["video_paths"] = [
            self._resolve_video_path(item, source_index=source_index) for item in raw_paths
        ]
        answer = normalized.get("answer")
        if isinstance(answer, list):
            normalized["answer"] = ",".join(str(item) for item in answer)
        metadata = normalized.get("task_metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"dataset row {source_index} task_metadata must be an object")
        normalized["task_metadata"] = {
            **metadata,
            "source_dataset": source_path.name,
            "source_index": source_index,
        }
        try:
            return MultiVideoSample.model_validate(normalized)
        except ValueError as exc:
            raise ValueError(f"invalid dataset row {source_index}: {exc}") from exc

    def _resolve_video_path(self, value: Any, *, source_index: int) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"dataset row {source_index} contains a non-string video path")
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.video_root / path
