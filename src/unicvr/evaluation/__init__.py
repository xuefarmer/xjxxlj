"""Task-agnostic batch inference and answer normalization."""

from unicvr.evaluation.evaluator import EvaluationSummary, evaluate_samples
from unicvr.evaluation.metrics import (
    EvaluationMetric,
    ExactMatchMetric,
    IntervalIoUMetric,
    ScoreResult,
    ScoringPointCoverageMetric,
    SequenceExactMetric,
    crossvid_metric,
)
from unicvr.evaluation.normalization import normalize_answer

__all__ = [
    "EvaluationMetric",
    "EvaluationSummary",
    "ExactMatchMetric",
    "IntervalIoUMetric",
    "ScoreResult",
    "ScoringPointCoverageMetric",
    "SequenceExactMetric",
    "crossvid_metric",
    "evaluate_samples",
    "normalize_answer",
]
