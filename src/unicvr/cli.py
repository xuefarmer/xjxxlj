from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from unicvr.config import AppConfig, load_config, validate_credentials
from unicvr.core.pipeline import Pipeline, PipelineRunError
from unicvr.data import (
    SUPPORTED_CROSSVID_TASKS,
    CrossVidDatasetAdapter,
    CrossVidTask,
    JSONDatasetAdapter,
    OpenCVMediaMaterializer,
    load_field_mapping,
)
from unicvr.data.schema import load_sample
from unicvr.evaluation import crossvid_metric, evaluate_samples
from unicvr.video import VideoProbe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unicvr.cli",
        description="Data-axis three-agent cross-video reasoning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="run three-agent inference on one sample")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--sample", type=Path, required=True)
    infer.add_argument("--output", type=Path)

    inspect = subparsers.add_parser("inspect-video", help="probe a local video")
    inspect.add_argument("--video", type=Path, required=True)
    inspect.add_argument("--video-id", default="v1")

    validate = subparsers.add_parser(
        "validate-config", help="validate schema and report credential readiness"
    )
    validate.add_argument("--config", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate normalized JSON dataset rows")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--video-root", type=Path, required=True)
    evaluate.add_argument("--mapping", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--start", type=int, default=0)
    evaluate.add_argument("--limit", type=int)

    crossvid = subparsers.add_parser(
        "evaluate-crossvid",
        help="normalize and evaluate a supported CrossVid task",
    )
    crossvid.add_argument("--config", type=Path, required=True)
    crossvid.add_argument("--crossvid-root", type=Path, required=True)
    crossvid.add_argument(
        "--task",
        choices=("ALL", *SUPPORTED_CROSSVID_TASKS),
        required=True,
    )
    crossvid.add_argument("--output-dir", type=Path, required=True)
    crossvid.add_argument(
        "--media-cache-dir",
        type=Path,
        default=Path(".unicvr_cache/crossvid_media"),
    )
    crossvid.add_argument("--start", type=int, default=0)
    crossvid.add_argument("--limit", type=int)
    crossvid.add_argument(
        "--fsa-min-duration",
        type=float,
        default=18.0,
        help=(
            "FSA interval duration anchor in seconds. Predicted intervals "
            "shorter than this are expanded to it (centered) before IoU is "
            "computed. Set 0 to disable. Default 18.0 = FSA GT median."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "infer":
            return _infer(args.config, args.sample, args.output)
        if args.command == "inspect-video":
            return _inspect_video(args.video, args.video_id)
        if args.command == "validate-config":
            return _validate_config(args.config)
        if args.command == "evaluate":
            return _evaluate(
                args.config, args.dataset, args.video_root,
                args.mapping, args.output_dir, args.start, args.limit,
            )
        if args.command == "evaluate-crossvid":
            return _evaluate_crossvid(
                args.config, args.crossvid_root, args.task,
                args.output_dir, args.media_cache_dir,
                args.start, args.limit,
                args.fsa_min_duration,
            )
    except (FileNotFoundError, ValueError, RuntimeError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise RuntimeError(f"unhandled command: {args.command}")


# ------------------------------------------------------------------
# infer
# ------------------------------------------------------------------


def _infer(config_path: Path, sample_path: Path, output: Path | None) -> int:
    config = load_config(config_path)
    sample = load_sample(sample_path, config.field_mapping)
    pipeline = Pipeline(config)

    trace_path = output or (config.trace.output_dir / f"{sample.sample_id}.trace.json")
    try:
        state = pipeline.run(sample)
    except PipelineRunError as exc:
        _write_trace(trace_path, exc.state, config)
        print(f"failure trace: {trace_path.resolve()}", file=sys.stderr)
        raise

    _write_trace(trace_path, state, config)
    print(f"sample: {sample.sample_id}")
    print("flow: " + " -> ".join(
        item["action"] for item in state.action_trace if "action" in item
    ))
    print(f"prediction: {state.final_answer}")
    print(f"observer reports: {len(state.observer_reports)}")
    print(f"comparisons: {len(state.comparison_results)}")
    print(f"trace: {trace_path.resolve()}")
    return 0


# ------------------------------------------------------------------
# inspect-video
# ------------------------------------------------------------------


def _inspect_video(path: Path, video_id: str) -> int:
    video = VideoProbe().probe(path, video_id=video_id)
    print(json.dumps(video.model_dump(mode="json"), indent=2))
    return 0


# ------------------------------------------------------------------
# validate-config
# ------------------------------------------------------------------


def _validate_config(path: Path) -> int:
    config = load_config(path)
    missing = validate_credentials(config)
    print(f"configuration valid: {path}")
    print(f"fingerprint: {config.fingerprint()}")
    print(f"LLM: {config.llm.backend} / {config.llm.model}; "
          f"VLM: {config.vlm.backend} / {config.vlm.model}")
    if missing:
        print("credential readiness: not ready (" + "; ".join(missing) + ")")
    else:
        print("credential readiness: ready")
    return 0


# ------------------------------------------------------------------
# evaluate
# ------------------------------------------------------------------


def _evaluate(
    config_path: Path,
    dataset_path: Path,
    video_root: Path,
    mapping_path: Path,
    output_dir: Path,
    start: int,
    limit: int | None,
) -> int:
    config = load_config(config_path)
    adapter = JSONDatasetAdapter(
        video_root=video_root,
        field_mapping=load_field_mapping(mapping_path),
    )
    samples = adapter.iter_samples(dataset_path, start=start, limit=limit)
    summary = evaluate_samples(config, samples, output_dir=output_dir)
    print(summary.model_dump_json(indent=2))
    print(f"results: {(output_dir / 'results.jsonl').resolve()}")
    print(f"summary: {(output_dir / 'summary.json').resolve()}")
    return 0


# ------------------------------------------------------------------
# evaluate-crossvid
# ------------------------------------------------------------------


def _evaluate_crossvid(
    config_path: Path,
    crossvid_root: Path,
    task: str,
    output_dir: Path,
    media_cache_dir: Path,
    start: int,
    limit: int | None,
    fsa_min_duration: float = 18.0,
) -> int:
    config = load_config(config_path)
    if task == "ALL":
        for selected in SUPPORTED_CROSSVID_TASKS:
            print(f"===== CrossVid {selected} =====")
            _evaluate_crossvid_task(
                config, crossvid_root, cast(CrossVidTask, selected),
                output_dir / selected, media_cache_dir, start, limit,
                fsa_min_duration,
            )
        print("completed tasks: " + ", ".join(SUPPORTED_CROSSVID_TASKS))
        return 0
    return _evaluate_crossvid_task(
        config, crossvid_root, cast(CrossVidTask, task),
        output_dir, media_cache_dir, start, limit,
        fsa_min_duration,
    )


def _evaluate_crossvid_task(
    config: AppConfig,
    crossvid_root: Path,
    task: CrossVidTask,
    output_dir: Path,
    media_cache_dir: Path,
    start: int,
    limit: int | None,
    fsa_min_duration: float = 18.0,
) -> int:
    adapter = CrossVidDatasetAdapter(
        crossvid_root=crossvid_root,
        task=task,
        materializer=OpenCVMediaMaterializer(media_cache_dir),
    )
    samples = adapter.iter_samples(start=start, limit=limit)
    summary = evaluate_samples(
        config, samples, output_dir=output_dir,
        metric=crossvid_metric(task, fsa_min_duration),
    )
    print(summary.model_dump_json(indent=2))
    print(f"results: {(output_dir / 'results.jsonl').resolve()}")
    print(f"summary: {(output_dir / 'summary.json').resolve()}")
    return 0


# ------------------------------------------------------------------
# trace writing
# ------------------------------------------------------------------


def _write_trace(path: Path, state: Any, config: AppConfig) -> Path:
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
    comparisons = [
        {
            "video_a": c.request.video_a,
            "video_b": c.request.video_b,
            "question": c.request.question,
            "result_text": _truncate(c.result_text, 2000),
        }
        for c in getattr(state, "comparison_results", [])
    ]

    path.write_text(
        json.dumps({
            "sample_id": getattr(state, "sample_id", "unknown"),
            "configuration_fingerprint": config.fingerprint(),
            "final_answer": getattr(state, "final_answer", None),
            "stop_reason": getattr(state, "stop_reason", None),
            "action_trace": getattr(state, "action_trace", []),
            "observer_reports": observer_reports,
            "comparisons": comparisons,
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


if __name__ == "__main__":
    raise SystemExit(main())
