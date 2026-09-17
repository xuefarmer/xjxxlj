from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

import cv2
import numpy as np

from unicvr.core.schemas import VideoRef

T = TypeVar("T")


def _uniform_points(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1 or end <= start:
        return [(start + end) / 2]
    return [start + index * (end - start) / (count - 1) for index in range(count)]


class WarmStartSampler:
    """Coverage-first deterministic sampler.

    Coverage is primary. Deterministic visual-change scores add shot/action-transition
    candidates, while the question remains a weak signal in the Scout prompt rather
    than biasing frame selection before any visual evidence exists.
    """

    def __init__(
        self,
        *,
        coverage_weight: float = 1.0,
        change_weight: float = 0.35,
        question_weight: float = 0.15,
        redundancy_weight: float = 0.25,
        candidate_multiplier: int = 3,
    ) -> None:
        self.coverage_weight = coverage_weight
        self.change_weight = change_weight
        self.question_weight = question_weight
        self.redundancy_weight = redundancy_weight
        self.candidate_multiplier = candidate_multiplier

    def sample(self, video: VideoRef, *, frame_budget: int, question: str) -> list[float]:
        del question
        if frame_budget <= 0:
            raise ValueError("warm-start frame budget must be positive")
        duration = video.duration_seconds or 0.0
        candidates = _uniform_points(
            0.0, duration, max(frame_budget, frame_budget * self.candidate_multiplier)
        )
        change_scores = self._change_scores(video, candidates)
        if frame_budget <= 2:
            return _uniform_points(0.0, duration, frame_budget)
        selected = [0.0, duration]
        candidates = [item for item in candidates if item not in selected]
        while candidates and len(selected) < frame_budget:

            def score(timestamp: float) -> tuple[float, float]:
                if not selected:
                    coverage_gain = 1.0
                    redundancy = 0.0
                else:
                    nearest = min(abs(timestamp - item) for item in selected)
                    coverage_gain = nearest / max(duration, 1e-6)
                    redundancy = math.exp(-nearest / max(duration / frame_budget, 1e-6))
                warm_score = (
                    self.coverage_weight * coverage_gain
                    + self.change_weight * change_scores.get(timestamp, 0.0)
                    + self.question_weight * 0.0
                    - self.redundancy_weight * redundancy
                )
                return warm_score, -timestamp

            chosen = max(candidates, key=score)
            selected.append(chosen)
            candidates.remove(chosen)
        return sorted(selected)

    @staticmethod
    def sample_interval(start: float, end: float, *, frame_budget: int) -> list[float]:
        if start > end:
            raise ValueError("sampling interval must be ordered")
        if frame_budget < 2:
            raise ValueError("interval sampling requires at least two frames")
        return _uniform_points(start, end, frame_budget)

    @staticmethod
    def frame_count(
        video: VideoRef,
        *,
        minimum: int,
        maximum: int,
        seconds_per_frame: float | None,
    ) -> int:
        if minimum <= 0 or maximum < minimum:
            raise ValueError("invalid warm-start frame bounds")
        if seconds_per_frame is None:
            return maximum
        duration = video.duration_seconds or 0.0
        duration_count = math.ceil(duration / seconds_per_frame) + 1
        return min(maximum, max(minimum, duration_count))

    @staticmethod
    def chunk(
        items: Sequence[T],
        *,
        frames_per_call: int,
        overlap_frames: int,
    ) -> list[list[T]]:
        if frames_per_call < 2:
            raise ValueError("frames_per_call must be at least two")
        if not 0 <= overlap_frames < frames_per_call:
            raise ValueError("overlap_frames must lie in [0, frames_per_call)")
        if not items:
            return []
        windows: list[list[T]] = []
        start = 0
        while start < len(items):
            window = list(items[start : start + frames_per_call])
            windows.append(window)
            if start + frames_per_call >= len(items):
                break
            start += frames_per_call - overlap_frames
        return windows

    @staticmethod
    def chunk_cost(
        frame_count: int,
        *,
        frames_per_call: int,
        overlap_frames: int,
    ) -> tuple[int, int]:
        if frame_count <= 0:
            return 0, 0
        if frame_count <= frames_per_call:
            return frame_count, 1
        stride = frames_per_call - overlap_frames
        call_count = 1 + math.ceil((frame_count - frames_per_call) / stride)
        transmitted_frames = frame_count + overlap_frames * (call_count - 1)
        return transmitted_frames, call_count

    @staticmethod
    def _change_scores(video: VideoRef, timestamps: Sequence[float]) -> dict[float, float]:
        if not timestamps or video.fps is None or video.frame_count is None:
            return {}
        capture = cv2.VideoCapture(str(video.path))
        if not capture.isOpened():
            return {}
        raw = [0.0] * len(timestamps)
        if len(timestamps) < 64:
            previous: np.ndarray | None = None
            try:
                for position, timestamp in enumerate(timestamps):
                    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        previous = None
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    thumbnail = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
                    if previous is not None:
                        raw[position] = float(np.mean(cv2.absdiff(previous, thumbnail))) / 255.0
                    previous = thumbnail
            finally:
                capture.release()
            return dict(zip(timestamps, _normalize_scores(raw)))

        targets: dict[int, list[int]] = {}
        for position, timestamp in enumerate(timestamps):
            frame_index = min(
                video.frame_count - 1,
                max(0, round(timestamp * video.fps)),
            )
            targets.setdefault(frame_index, []).append(position)
        sequential_previous: np.ndarray | None = None
        try:
            for frame_index in range(max(targets) + 1):
                if not capture.grab():
                    break
                positions = targets.get(frame_index)
                if positions is None:
                    continue
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    sequential_previous = None
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thumbnail = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
                score = 0.0
                if sequential_previous is not None:
                    score = float(np.mean(cv2.absdiff(sequential_previous, thumbnail))) / 255.0
                for position in positions:
                    raw[position] = score
                sequential_previous = thumbnail
        finally:
            capture.release()
        return dict(zip(timestamps, _normalize_scores(raw)))


def _normalize_scores(scores: list[float]) -> list[float]:
    peak = max(scores, default=0.0)
    if peak <= 0:
        return scores
    return [item / peak for item in scores]


class FocusSampler:
    def __init__(
        self, *, pre_context_ratio: float = 0.25, post_context_ratio: float = 0.25
    ) -> None:
        self.pre_context_ratio = pre_context_ratio
        self.post_context_ratio = post_context_ratio

    def sample(
        self,
        video: VideoRef,
        *,
        start_seconds: float,
        end_seconds: float,
        frame_budget: int,
    ) -> list[float]:
        if frame_budget < 3:
            raise ValueError("focus sampling requires at least 3 frames")
        duration = video.duration_seconds or 0.0
        target_start = min(max(0.0, start_seconds), duration)
        target_end = min(max(target_start, end_seconds), duration)
        span = max(target_end - target_start, 1.0 / max(video.fps or 1.0, 1.0))
        expanded_start = max(0.0, target_start - span * self.pre_context_ratio)
        expanded_end = min(duration, target_end + span * self.post_context_ratio)
        midpoint = (target_start + target_end) / 2
        if frame_budget == 3:
            anchors = [target_start, midpoint, target_end]
        elif frame_budget == 4:
            anchors = [expanded_start, target_start, target_end, expanded_end]
        else:
            anchors = [
                expanded_start,
                target_start,
                midpoint,
                target_end,
                expanded_end,
            ]
        selected = list(dict.fromkeys(anchors))
        candidates = _uniform_points(
            expanded_start,
            expanded_end,
            max(frame_budget, frame_budget * 3),
        )
        while len(selected) < frame_budget:
            remaining = [item for item in candidates if item not in selected]
            if not remaining:
                break
            chosen = max(
                remaining,
                key=lambda item: (
                    min(abs(item - existing) for existing in selected),
                    -item,
                ),
            )
            selected.append(chosen)
        return sorted(selected)
