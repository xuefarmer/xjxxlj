from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, cast, get_args

from unicvr.data.materialize import CrossVidMediaMaterializer
from unicvr.data.schema import MultiVideoSample

CrossVidTask = Literal["CC", "CCQA", "FSA", "MOC", "MSR", "NC", "PEA", "PI", "PSS"]
SUPPORTED_CROSSVID_TASKS: tuple[str, ...] = get_args(CrossVidTask)


class CrossVidDatasetAdapter:
    """Normalize task files before the shared inference engine sees a sample."""

    def __init__(
        self,
        *,
        crossvid_root: Path,
        task: CrossVidTask,
        materializer: CrossVidMediaMaterializer,
        frame_dir_fps: float | None = None,
    ) -> None:
        self.root = crossvid_root.resolve()
        self.task = task
        self.materializer = materializer
        # self-built Epic mirrors store frames as directories (30fps by
        # convention: sec = frame/30); official CrossVid videos are mp4 files.
        self.frame_dir_fps = frame_dir_fps
        self.video_root = self.root / "videos"
        self.uav_root = self.root / "uav"
        self.dataset_path = self.root / "QA" / f"{task}.json"
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"CrossVid task file does not exist: {self.dataset_path}")

    def iter_samples(
        self,
        *,
        start: int = 0,
        limit: int | None = None,
    ) -> Iterator[MultiVideoSample]:
        if start < 0:
            raise ValueError("dataset start must be nonnegative")
        if limit is not None and limit <= 0:
            raise ValueError("dataset limit must be positive")
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("CrossVid task JSON root must be a list")
        stop = len(raw) if limit is None else min(len(raw), start + limit)
        for index in range(start, stop):
            row = raw[index]
            if not isinstance(row, dict):
                raise ValueError(f"CrossVid row {index} must be an object")
            yield self._normalize(row, source_index=index)

    def _normalize(
        self,
        row: dict[str, Any],
        *,
        source_index: int,
    ) -> MultiVideoSample:
        sample_id = f"{self.task}-{row.get('id', source_index)}"
        if self.task in {"CC", "NC"}:
            payload = self._multiple_choice(row, sample_id)
        elif self.task == "CCQA":
            payload = self._ccqa(row, sample_id)
        elif self.task == "FSA":
            payload = self._fsa(row, sample_id)
        elif self.task == "PEA":
            payload = self._pea(row, sample_id)
        elif self.task == "PI":
            payload = self._pi(row, sample_id)
        elif self.task == "PSS":
            payload = self._pss(row, sample_id)
        elif self.task in {"MOC", "MSR"}:
            payload = self._uav(row, sample_id)
        else:
            raise ValueError(f"unsupported CrossVid task: {self.task}")
        metadata = cast(dict[str, Any], payload.setdefault("task_metadata", {}))
        metadata.update(
            {
                "crossvid_task": self.task,
                "source_dataset": self.dataset_path.name,
                "source_index": source_index,
            }
        )
        try:
            return MultiVideoSample.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"invalid {self.task} row {source_index}: {exc}") from exc

    def _multiple_choice(self, row: dict[str, Any], sample_id: str) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "video_paths": self._video_paths(_string_list(row.get("videos"), "videos")),
            "question": _required_string(row, "question"),
            "options": _string_list(row.get("options"), "options"),
            "answer": _answer_string(row.get("answer")),
        }

    def _ccqa(self, row: dict[str, Any], sample_id: str) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "video_paths": self._video_paths(
                [_required_string(row, "video A"), _required_string(row, "video B")]
            ),
            "question": _required_string(row, "question"),
            "options": [],
            "answer": _required_string(row, "answer"),
            "task_metadata": {
                "scoring_points": _string_list(row.get("scoring_points"), "scoring_points")
            },
        }

    def _fsa(self, row: dict[str, Any], sample_id: str) -> dict[str, Any]:
        source_a = self._video_path(_required_string(row, "video A"))
        source_b = self._video_path(_required_string(row, "video B"))
        reference = _interval(row.get("ref_segment"), "ref_segment")
        reference_clip = self.materializer.clip(
            source_a,
            [reference],
            cache_key=f"{sample_id}-reference",
        )
        return {
            "sample_id": sample_id,
            "video_paths": [reference_clip, source_b],
            "question": (
                "Video 1 is a reference cooking step. Find the functionally equivalent step in "
                "Video 2. Return only its Video 2 interval as [start_seconds, end_seconds]."
            ),
            "options": [],
            "answer": json.dumps(row.get("answer")),
            "task_metadata": {"reference_interval": list(reference)},
        }

    def _pea(self, row: dict[str, Any], sample_id: str) -> dict[str, Any]:
        videos = _string_list(row.get("videos"), "videos")
        begins = _number_list(row.get("begin"), "begin")
        ends = _number_list(row.get("end"), "end")
        if not (len(videos) == len(begins) == len(ends)):
            raise ValueError("PEA videos, begin, and end must have equal lengths")
        assert len(videos) == len(begins) == len(ends), "video/begin/end length mismatch"
        clips = [
            self.materializer.clip(
                self._video_path(video),
                [(begin, end)],
                cache_key=f"{sample_id}-video-{index}",
            )
            # explicit length check above (raise + assert) replaces
            # zip(strict=True), which needs Python 3.10+.
            for index, (video, begin, end) in enumerate(
                zip(videos, begins, ends),
                start=1,
            )
        ]
        return {
            "sample_id": sample_id,
            "video_paths": clips,
            "question": _required_string(row, "question"),
            "options": _string_list(row.get("options"), "options"),
            "answer": _answer_string(row.get("answer")),
        }

    def _pi(self, row: dict[str, Any], sample_id: str) -> dict[str, Any]:
        source = self._video_path(_required_string(row, "video"))
        beginning = _interval(row.get("beginning"), "beginning")
        ending = _interval(row.get("ending"), "ending")
        clips = [
            self.materializer.clip(source, [beginning], cache_key=f"{sample_id}-beginning"),
            self.materializer.clip(source, [ending], cache_key=f"{sample_id}-ending"),
        ]
        return {
            "sample_id": sample_id,
            "video_paths": clips,
            "question": (
                "Video 1 shows the beginning and Video 2 shows the ending of one story. "
                "Which option best explains what happened in the missing interval?"
            ),
            "options": _string_list(row.get("options"), "options"),
            "answer": _answer_string(row.get("answer")),
        }

    def _pss(self, row: dict[str, Any], sample_id: str) -> dict[str, Any]:
        source = self._video_path(_required_string(row, "video"))
        raw_segments = row.get("segments")
        if not isinstance(raw_segments, dict) or len(raw_segments) < 2:
            raise ValueError("PSS segments must be an object with at least two clips")
        clips = []
        for key in sorted(raw_segments, key=lambda value: int(value)):
            raw_intervals = raw_segments[key]
            # self-built mirrors wrap intervals as {"interval": [[s, e], ...]}
            if isinstance(raw_intervals, dict) and isinstance(
                raw_intervals.get("interval"), list
            ):
                raw_intervals = raw_intervals["interval"]
            if not isinstance(raw_intervals, list):
                raise ValueError(f"PSS segment {key} must contain intervals")
            intervals = [_interval(item, f"segments.{key}") for item in raw_intervals]
            clips.append(
                self.materializer.clip(
                    source,
                    intervals,
                    cache_key=f"{sample_id}-segment-{key}",
                    frame_dir_fps=self.frame_dir_fps,
                )
            )
        return {
            "sample_id": sample_id,
            "video_paths": clips,
            "question": (
                "These numbered clips are shuffled segments from one cooking video. Determine "
                "their chronological order. Return only clip numbers joined by ->."
            ),
            "options": [],
            "answer": _required_string(row, "answer"),
        }

    def _uav(self, row: dict[str, Any], sample_id: str) -> dict[str, Any]:
        scene_id = row.get("vid")
        objects = row.get("objects")
        if not isinstance(scene_id, int):
            raise ValueError("UAV vid must be an integer")
        if not isinstance(objects, list) or not all(isinstance(item, dict) for item in objects):
            raise ValueError("UAV objects must be a list of objects")
        paths = self.materializer.uav_views(
            uav_root=self.uav_root,
            scene_id=scene_id,
            objects=objects,
            cache_key=sample_id,
        )
        return {
            "sample_id": sample_id,
            "video_paths": paths,
            "question": _required_string(row, "question"),
            "options": _string_list(row.get("options"), "options"),
            "answer": _answer_string(row.get("answer")),
            "task_metadata": {
                "scene_id": scene_id,
                "reference_track_evidence": self._reference_track_evidence(
                    scene_id=scene_id,
                    objects=objects,
                ),
            },
        }

    def _reference_track_evidence(
        self,
        *,
        scene_id: int,
        objects: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expose only the reference tracks already rendered into UAV views.

        The full bbox files contain many more scene annotations.  Deliberately
        restrict this sidecar to ``row.objects`` so inference receives no
        hidden count-target annotations beyond the five reference boxes that
        are already visible in the proxy videos.
        """
        sources: dict[int, tuple[int, dict[int, dict[str, Any]]]] = {}
        paths = [self.uav_root / "bbox" / str(view) / f"{scene_id}.json" for view in (1, 2)]
        if not any(path.is_file() for path in paths):
            # Test/custom materializers may synthesize UAV media without a
            # bbox sidecar.  Optional evidence is a no-op in that case.
            return []
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete UAV reference-track sidecar: {missing}")

        # ``paths`` is derived from a two-element literal above, so plain zip
        # is safe here (zip(strict=True) needs Python 3.10+).
        for view, path in zip((1, 2), paths):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError(f"UAV bbox JSON must contain a list: {path}")
            by_id = {
                item["id"]: item
                for item in raw
                if isinstance(item, dict) and isinstance(item.get("id"), int)
            }
            frame_dir = self.uav_root / "frames" / str(view) / f"{scene_id}-{view}"
            frame_count = len(
                [
                    item
                    for item in frame_dir.iterdir()
                    if item.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            if frame_count == 0:
                raise ValueError(f"UAV frame directory is empty: {frame_dir}")
            sources[view] = (frame_count, by_id)

        evidence = []
        for index, selected in enumerate(objects, start=1):
            object_id = selected.get("id")
            if not isinstance(object_id, int):
                continue
            record: dict[str, Any] = {
                "reference": f"A{index}/B{index}",
                "raw_track_id": object_id,
                "declared_label": str(selected.get("label", "object")),
                "views": {},
            }
            views = cast(dict[str, Any], record["views"])
            for view, prefix in ((1, "A"), (2, "B")):
                frame_count, by_id = sources[view]
                entity = by_id.get(object_id)
                ranges = _visible_frame_ranges(entity.get("bbox") if entity else None)
                view_record: dict[str, Any] = {
                    "alias": f"{prefix}{index}",
                    "source_frame_count": frame_count,
                    "visible_frame_ranges": [list(span) for span in ranges],
                }
                if entity is not None:
                    if isinstance(entity.get("label"), str):
                        view_record["annotation_label"] = entity["label"]
                    if isinstance(entity.get("moving"), bool):
                        view_record["moving"] = entity["moving"]
                views[prefix] = view_record
            evidence.append(record)
        return evidence

    def _video_paths(self, values: Sequence[str]) -> list[Path]:
        return [self._video_path(value) for value in values]

    def _video_path(self, value: str) -> Path:
        path = self.video_root / value
        # either a video file (official CrossVid) or a frame directory
        # (self-built mirrors such as Epic_Kitchen new_frames)
        if not path.exists():
            raise FileNotFoundError(f"CrossVid video does not exist: {path}")
        return path


def _required_string(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return value


def _number_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{field} must be a list of numbers")
    return [float(item) for item in value]


def _interval(value: Any, field: str) -> tuple[float, float]:
    numbers = _number_list(value, field)
    if len(numbers) != 2 or numbers[0] < 0 or numbers[1] <= numbers[0]:
        raise ValueError(f"{field} must be an ordered [start, end] interval")
    return numbers[0], numbers[1]


def _answer_string(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, str):
        return value
    raise ValueError("answer must be a string or list")


def _visible_frame_ranges(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, dict):
        return []
    visible = sorted(
        int(frame)
        for frame, box in value.items()
        if isinstance(box, dict) and box.get("outside") is not True
    )
    ranges: list[tuple[int, int]] = []
    for frame in visible:
        if not ranges or frame > ranges[-1][1] + 1:
            ranges.append((frame, frame))
        else:
            ranges[-1] = (ranges[-1][0], frame)
    return ranges
