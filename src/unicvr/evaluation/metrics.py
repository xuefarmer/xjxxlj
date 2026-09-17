from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from unicvr.data.schema import MultiVideoSample
from unicvr.evaluation.normalization import normalize_answer

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_TOKEN = re.compile(r"[a-z0-9]+")


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_prediction: str
    normalized_ground_truth: str | None
    score: float | None = Field(default=None, ge=0, le=1)
    correct: bool | None = None
    details: dict[str, object] = Field(default_factory=dict)


class EvaluationMetric(Protocol):
    def score(self, prediction: str, sample: MultiVideoSample) -> ScoreResult: ...


class ExactMatchMetric:
    def score(self, prediction: str, sample: MultiVideoSample) -> ScoreResult:
        normalized_prediction = normalize_answer(prediction, sample.options)
        normalized_ground_truth = (
            normalize_answer(sample.answer, sample.options) if sample.answer is not None else None
        )
        correct = (
            normalized_ground_truth is not None and normalized_prediction == normalized_ground_truth
        )
        return ScoreResult(
            normalized_prediction=normalized_prediction,
            normalized_ground_truth=normalized_ground_truth,
            score=float(correct) if normalized_ground_truth is not None else None,
            correct=correct if normalized_ground_truth is not None else None,
        )


class SequenceExactMetric:
    def score(self, prediction: str, sample: MultiVideoSample) -> ScoreResult:
        normalized_prediction = _normalize_sequence(prediction)
        normalized_ground_truth = (
            _normalize_sequence(sample.answer) if sample.answer is not None else None
        )
        correct = (
            normalized_ground_truth is not None and normalized_prediction == normalized_ground_truth
        )
        return ScoreResult(
            normalized_prediction=normalized_prediction,
            normalized_ground_truth=normalized_ground_truth,
            score=float(correct) if normalized_ground_truth is not None else None,
            correct=correct if normalized_ground_truth is not None else None,
        )


class IntervalIoUMetric:
    """FSA interval IoU with optional duration-anchor postprocessing.

    `min_duration_seconds` (default None = disabled) expands predicted
    intervals shorter than the anchor to that duration, centered on the
    original prediction. The anchor should be the dataset GT median
    interval duration (FSA: 18s). The raw (unadjusted) interval and IoU
    are always reported in `details` for transparency.
    """

    def __init__(
        self, *, correct_threshold: float = 0.5,
        min_duration_seconds: float | None = None,
    ) -> None:
        self.correct_threshold = correct_threshold
        self.min_duration_seconds = min_duration_seconds

    def score(self, prediction: str, sample: MultiVideoSample) -> ScoreResult:
        predicted = _parse_interval(prediction)
        ground_truth = _parse_interval(sample.answer or "")
        raw_interval = predicted
        if (
            predicted is not None
            and ground_truth is not None
            and self.min_duration_seconds is not None
        ):
            predicted = _expand_to_min_duration(predicted, self.min_duration_seconds)
        score = _interval_iou(predicted, ground_truth) if predicted and ground_truth else 0.0
        details: dict[str, object] = {"iou_threshold": self.correct_threshold}
        if self.min_duration_seconds is not None:
            details["min_duration_seconds"] = self.min_duration_seconds
            details["raw_interval"] = _format_interval(raw_interval)
            details["adjusted_interval"] = _format_interval(predicted)
            details["raw_iou"] = (
                _interval_iou(raw_interval, ground_truth) if raw_interval and ground_truth else 0.0
            )
        return ScoreResult(
            normalized_prediction=_format_interval(predicted),
            normalized_ground_truth=_format_interval(ground_truth),
            score=score,
            correct=score >= self.correct_threshold,
            details=details,
        )


class ScoringPointCoverageMetric:
    """Deterministic CCQA proxy metric; not the benchmark's external model judge."""

    def __init__(self, *, point_threshold: float = 0.3) -> None:
        self.point_threshold = point_threshold

    def score(self, prediction: str, sample: MultiVideoSample) -> ScoreResult:
        points = sample.task_metadata.get("scoring_points", [])
        if not isinstance(points, list):
            points = []
        prediction_tokens = _tokens(prediction)
        point_scores = [_token_recall(prediction_tokens, _tokens(str(point))) for point in points]
        matched = [score >= self.point_threshold for score in point_scores]
        coverage = sum(matched) / len(matched) if matched else 0.0
        return ScoreResult(
            normalized_prediction=" ".join(prediction.strip().split()).casefold(),
            normalized_ground_truth=(
                " ".join(sample.answer.strip().split()).casefold()
                if sample.answer is not None
                else None
            ),
            score=coverage,
            correct=coverage >= 0.5,
            details={
                "matched_points": sum(matched),
                "total_points": len(matched),
                "point_scores": point_scores,
                "proxy_metric": True,
            },
        )


def crossvid_metric(task: str, fsa_min_duration: float | None = None) -> EvaluationMetric:
    if task == "FSA":
        return IntervalIoUMetric(min_duration_seconds=fsa_min_duration)
    if task == "PSS":
        return SequenceExactMetric()
    if task == "CCQA":
        return ScoringPointCoverageMetric()
    return ExactMatchMetric()


def _normalize_sequence(value: str) -> str:
    return "->".join(re.findall(r"\d+", value))


def _parse_interval(value: str) -> tuple[float, float] | None:
    numbers = [float(item) for item in _NUMBER.findall(value)]
    if len(numbers) < 2:
        return None
    start, end = numbers[-2], numbers[-1]
    return (start, end) if 0 <= start < end else None


def _expand_to_min_duration(
    interval: tuple[float, float], min_duration: float
) -> tuple[float, float]:
    """Expand a too-short interval to `min_duration`, anchored at its center."""
    start, end = interval
    if end - start >= min_duration:
        return interval
    center = (start + end) / 2.0
    half = min_duration / 2.0
    return (center - half, center + half)


def _format_interval(value: tuple[float, float] | None) -> str:
    if value is None:
        return ""
    return f"[{value[0]:g},{value[1]:g}]"


def _interval_iou(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(value.casefold()))


def _token_recall(prediction: set[str], point: set[str]) -> float:
    return len(prediction & point) / len(point) if point else 0.0
