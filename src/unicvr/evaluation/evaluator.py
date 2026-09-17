from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from unicvr.config import AppConfig
from unicvr.core.pipeline import Pipeline, PipelineRunError
from unicvr.data.schema import MultiVideoSample
from unicvr.evaluation.metrics import EvaluationMetric, ExactMatchMetric


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempted: int = Field(ge=0)
    completed: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    average_score: float = Field(ge=0, le=1)
    failed: int = Field(ge=0)
    average_frames: float = Field(ge=0)
    average_visual_calls: float = Field(ge=0)
    average_llm_calls: float = Field(ge=0)
    average_rounds: float = Field(ge=0)


def evaluate_samples(
    config: AppConfig,
    samples: Iterable[MultiVideoSample],
    *,
    output_dir: Path,
    metric: EvaluationMetric | None = None,
    resume: bool = False,
) -> EvaluationSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.jsonl"
    selected_metric = metric or ExactMatchMetric()
    records: list[dict[str, object]] = []
    if resume and result_path.is_file():
        records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_ids = {str(record.get("sample_id")) for record in records}

    with result_path.open("a" if resume else "w", encoding="utf-8") as result_file:
        for sample in samples:
            if sample.sample_id in completed_ids:
                print(f"[{sample.sample_id}] skip: already present in results.jsonl")
                continue
            pipeline = Pipeline(config)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id)
            trace_path = trace_dir / f"{safe_id}.trace.json"

            try:
                state = pipeline.run(sample)
                _write_trace(trace_path, state, config)

                prediction = state.final_answer or ""
                score = selected_metric.score(prediction, sample)

                visual_calls = sum(
                    1 for c in state.api_calls if c.visual_input_count > 0
                )
                llm_calls = sum(
                    1 for c in state.api_calls if c.visual_input_count == 0
                )
                total_frames = sum(
                    c.visual_input_count for c in state.api_calls
                )

                record = {
                    "sample_id": sample.sample_id,
                    "prediction": prediction,
                    "normalized_prediction": score.normalized_prediction,
                    "ground_truth": sample.answer,
                    "normalized_ground_truth": score.normalized_ground_truth,
                    "score": score.score,
                    "score_details": score.details,
                    "correct": score.correct,
                    "action_trace": state.action_trace,
                    "observer_reports": [
                        {"video_id": r.video_id, "frame_count": r.frame_count}
                        for r in state.observer_reports
                    ],
                    "evidence_count": len(state.evidence_results) if hasattr(state, 'evidence_results') else 0,
                    "api_calls": [c.model_dump(mode="json") for c in state.api_calls],
                    "budget_usage": {
                        "used_frames": total_frames,
                        "used_visual_calls": visual_calls,
                        "used_llm_calls": llm_calls,
                        "current_round": 1,
                    },
                    "stop_reason": state.stop_reason,
                    "trace_path": str(trace_path.resolve()),
                    "error": None,
                }

                print(
                    # Show the pipeline's literal final answer first.  The
                    # normalized version remains useful for understanding
                    # scoring (notably PSS and FSA), but is not a substitute
                    # for inspecting what the model actually returned.
                    f"[{sample.sample_id}] answer={prediction!r} "
                    f"gt={sample.answer!r} "
                    f"normalized=({score.normalized_prediction!r}, "
                    f"{score.normalized_ground_truth!r}) "
                    f"score={score.score} correct={score.correct}"
                )

            except PipelineRunError as exc:
                _write_trace(trace_path, exc.state, config)
                failed_score = (
                    selected_metric.score("", sample)
                    if sample.answer is not None
                    else None
                )
                record = {
                    "sample_id": sample.sample_id,
                    "prediction": None,
                    "normalized_prediction": None,
                    "ground_truth": sample.answer,
                    "normalized_ground_truth": (
                        failed_score.normalized_ground_truth
                        if failed_score
                        else None
                    ),
                    "score": None,
                    "score_details": {},
                    "correct": False,
                    "action_trace": [],
                    "observer_reports": [],
                    "evidence_count": 0,
                    "api_calls": [],
                    "budget_usage": None,
                    "stop_reason": "error",
                    "trace_path": str(trace_path.resolve()),
                    "error": str(exc),
                }
                print(f"[{sample.sample_id}] failed: {exc}")

            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                failed_score = (
                    selected_metric.score("", sample)
                    if sample.answer is not None
                    else None
                )
                record = {
                    "sample_id": sample.sample_id,
                    "prediction": None,
                    "normalized_prediction": None,
                    "ground_truth": sample.answer,
                    "normalized_ground_truth": (
                        failed_score.normalized_ground_truth
                        if failed_score
                        else None
                    ),
                    "score": None,
                    "score_details": {},
                    "correct": False,
                    "action_trace": [],
                    "observer_reports": [],
                    "evidence_count": 0,
                    "api_calls": [],
                    "budget_usage": None,
                    "stop_reason": "error",
                    "trace_path": None,
                    "error": str(exc),
                }
                print(f"[{sample.sample_id}] failed before trace initialization: {exc}")

            records.append(record)
            completed_ids.add(str(record["sample_id"]))
            result_file.write(json.dumps(record, ensure_ascii=True) + "\n")
            result_file.flush()

    summary = _summarize(records)
    (output_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8",
    )
    return summary


# ------------------------------------------------------------------
# trace writing
# ------------------------------------------------------------------


def _write_trace(path: Path, state: object, config: AppConfig) -> Path:
    """Write a pipeline trace to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    observer_reports = [
        {
            "video_id": r.video_id,
            "frame_count": r.frame_count,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "report_text": _truncate(r.report_text, 3000),
        }
        for r in getattr(state, "observer_reports", [])
    ]
    evidence = [
        {
            "witness": c.ambiguity.witness[:200],
            "videos": [t.video for t in c.ambiguity.targets],
            "scope": c.scope,
            "result_text": _truncate(c.result_text, 2000),
        }
        for c in getattr(state, "evidence_results", [])
    ]

    path.write_text(
        json.dumps({
            "sample_id": getattr(state, "sample_id", "unknown"),
            "configuration_fingerprint": config.fingerprint(),
            "final_answer": getattr(state, "final_answer", None),
            "stop_reason": getattr(state, "stop_reason", None),
            "action_trace": getattr(state, "action_trace", []),
            "observer_reports": observer_reports,
            "evidence": evidence,
            "api_calls": [
                c.model_dump(mode="json")
                for c in getattr(state, "api_calls", [])
            ],
            "errors": getattr(state, "errors", []),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "...[truncated]"


# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------


def _summarize(records: list[dict[str, object]]) -> EvaluationSummary:
    completed = [item for item in records if item["error"] is None]
    scored = [item for item in completed if isinstance(item["score"], (int, float))]
    correct = sum(item["correct"] is True for item in scored)
    score_values: list[float] = []
    for item in scored:
        value = item["score"]
        if isinstance(value, (int, float)):
            score_values.append(float(value))

    def average(field: str) -> float:
        if not completed:
            return 0.0
        values: list[float] = []
        for item in completed:
            budget_usage = item["budget_usage"]
            if not isinstance(budget_usage, dict):
                values.append(0.0)
                continue
            value = budget_usage.get(field)
            if not isinstance(value, (int, float)):
                values.append(0.0)
                continue
            values.append(float(value))
        return sum(values) / len(values) if values else 0.0

    return EvaluationSummary(
        attempted=len(records),
        completed=len(completed),
        correct=correct,
        accuracy=correct / len(scored) if scored else 0.0,
        average_score=sum(score_values) / len(score_values) if score_values else 0.0,
        failed=len(records) - len(completed),
        average_frames=average("used_frames"),
        average_visual_calls=average("used_visual_calls"),
        average_llm_calls=average("used_llm_calls"),
        average_rounds=average("current_round"),
    )
