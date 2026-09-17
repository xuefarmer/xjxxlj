#!/usr/bin/env python3
"""Resume staged evaluation from warmed Event reports; no HTTP servers.

Phase 1 (warm Observer) is read from the preheat cache written by
``scripts/preheat_event_local_vlm.py``.  Phase 2 FORM, dynamic
Focus/Comparer evidence and REVIEW then run unchanged through in-process
LLM/VLM LoRAs, exactly like ``eval_local_dual_lora_direct.py``.

One process owns one (LLM, VLM) GPU pair and one deterministic shard:

  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_cached_local_direct.py \
      --llm-device cuda:0 --vlm-device cuda:1 --shard-index 0 --shard-count 2 ...
  CUDA_VISIBLE_DEVICES=2,7 python scripts/eval_cached_local_direct.py \
      --llm-device cuda:0 --vlm-device cuda:1 --shard-index 1 --shard-count 2 ...

Each shard writes its own ``--output-dir`` with one ``<task>/results.jsonl``,
``<task>/traces/`` and ``<task>/summary.json`` per task, matching the layout
of ``scripts/merge_sharded_eval.py`` (shards named ``part0``, ``part1``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from unicvr.config import load_config
from unicvr.core.pipeline import Pipeline, PipelineRunError
from unicvr.core.schemas import BackendCallRecord
from unicvr.core.state import ObserverReport
from unicvr.data.crossvid import CrossVidDatasetAdapter
from unicvr.data.materialize import OpenCVMediaMaterializer
from unicvr.evaluation import crossvid_metric
from unicvr.evaluation.evaluator import _summarize, _write_trace

import eval_local_dual_lora_direct as direct
from run_staged_local_lora_eval import cache_path, cache_signature


class CachedPhaseOnePipeline(Pipeline):
    """Production FORM -> Focus -> REVIEW pipeline with a cache-backed Observer."""

    def __init__(self, *args: Any, cache_root: Path, signature_config: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cache_root = cache_root
        self._signature_config = signature_config
        self._sample: Any = None
        self._cached: dict[str, Any] | None = None

    def run(self, sample: Any):
        self._sample = sample
        state = super().run(sample)
        cached = self._cached or {}
        state.api_calls = [
            BackendCallRecord.model_validate(item) for item in cached.get("preheat_calls", [])
        ] + state.api_calls
        return state

    def _phase1_observe(self, videos: list[Any], question: str, **_kwargs: Any) -> list[ObserverReport]:
        sample = self._sample
        task = str(sample.task_metadata.get("crossvid_task", ""))
        path = cache_path(self._cache_root, task, sample.sample_id)
        if not path.is_file():
            raise RuntimeError(f"missing preheat cache: {path}")
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("signature") != cache_signature(sample, self._signature_config):
            # A changed sampling stride or local endpoint can alter the stored
            # signature without changing the warmed report's semantics.  Keep
            # the strict identity checks below and warn instead of discarding
            # a valid preheat.
            if cached.get("sample_id") != sample.sample_id or cached.get("task") != task:
                raise RuntimeError(f"stale preheat cache: {path}")
            print(
                f"[{task}:{sample.sample_id}] cache signature differs from current config; "
                "accepting reports by sample/task identity",
                flush=True,
            )
        reports = [ObserverReport(**item) for item in cached.get("reports", [])]
        expected = [video.video_id for video in videos]
        if [report.video_id for report in reports] != expected:
            raise RuntimeError(f"preheat report/video mismatch: {path}")
        self._cached = cached
        return reports


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/local_dual_lora_eval_longvideo_compact.yaml")
    ap.add_argument("--crossvid-root", type=Path, default=Path("/media/data6/xuejj/CrossVid"))
    ap.add_argument("--tasks", default="PSS,CC,FSA,NC,PI,MSR,MOC")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="per-task row limit; default runs every row")
    ap.add_argument("--cache-dir", type=Path, required=True, help="preheat cache root")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing output directory and skip sample_ids already recorded")
    ap.add_argument("--sample-ids", default=None,
                    help="comma-separated sample ids; run only these (targeted re-checks)")
    ap.add_argument("--llm-model", default="/media/data6/xuejj/Qwen3-8B")
    ap.add_argument("--llm-adapter", type=Path, required=True)
    ap.add_argument("--vlm-model", default="/media/data6/xuejj/Qwen-VL")
    ap.add_argument("--vlm-adapter", type=Path, required=True)
    ap.add_argument("--llm-device", default="cuda:0")
    ap.add_argument("--vlm-device", default="cuda:1")
    ap.add_argument("--max-llm-tokens", type=int, default=0, help="0 uses config generation max")
    ap.add_argument("--max-vlm-tokens", type=int, default=0, help="0 uses config generation max")
    return ap


def _load_completed(result_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if result_path.is_file():
        for raw in result_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    return records


def _failure_record(sample_id: str, error: str, trace: Path | None) -> dict[str, Any]:
    return {"sample_id": sample_id, "prediction": None, "normalized_prediction": None,
            "ground_truth": None, "normalized_ground_truth": None, "score": None, "score_details": {},
            "correct": False, "action_trace": [], "observer_reports": [], "evidence_count": 0,
            "api_calls": [], "budget_usage": None, "stop_reason": "error",
            "trace_path": str(trace.resolve()) if trace is not None else None,
            "elapsed_seconds": 0.0, "error": error}


def main() -> int:
    args = _parser().parse_args()
    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The compact YAML names HTTP endpoints via environment variables.  Direct
    # mode never contacts them, but AppConfig still validates those fields.
    os.environ.setdefault("LLM_BASE_URL", "http://local-direct.invalid/v1")
    os.environ.setdefault("LLM_API_KEY", "local")
    os.environ.setdefault("LLM_MODEL", "local-llm-lora")
    os.environ.setdefault("VLM_BASE_URL", "http://local-direct.invalid/v1")
    os.environ.setdefault("VLM_API_KEY", "local")
    os.environ.setdefault("VLM_MODEL", "local-vlm-lora")
    config = load_config(args.config)
    cache_root = args.cache_dir.resolve()
    if not cache_root.is_dir():
        raise SystemExit(f"preheat cache directory does not exist: {cache_root}")

    llm, vlm = direct.load_backends(args)
    tasks = [item.strip().upper() for item in args.tasks.split(",") if item.strip()]
    wanted_ids = (
        {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        if args.sample_ids else None
    )
    ephemeral_root = args.output_dir / ".ephemeral"

    print(
        f"[shard {args.shard_index}/{args.shard_count}] llm={args.llm_device} vlm={args.vlm_device} "
        f"cache={cache_root} tasks={','.join(tasks)}",
        flush=True,
    )
    for task in tasks:
        metric = crossvid_metric(task, 0)
        task_out = args.output_dir / task
        task_out.mkdir(parents=True, exist_ok=True)
        trace_dir = task_out / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        result_path = task_out / "results.jsonl"
        records = _load_completed(result_path) if args.resume else []
        completed_ids = {str(item.get("sample_id")) for item in records}

        adapter = CrossVidDatasetAdapter(
            crossvid_root=args.crossvid_root, task=task,
            materializer=OpenCVMediaMaterializer(ephemeral_root / "_probe" / task),
        )
        rows = json.loads(adapter.dataset_path.read_text(encoding="utf-8"))
        stop = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
        print(f"=== [{task}] cached Phase 1 -> FORM/Focus/REVIEW "
              f"({stop - args.start} rows, {len(completed_ids)} already recorded) ===", flush=True)
        done = skipped = failed = 0
        with result_path.open("a" if args.resume else "w", encoding="utf-8") as results:
            for source_index in range(args.start, stop):
                if source_index % args.shard_count != args.shard_index:
                    continue
                temporary = ephemeral_root / task / str(source_index)
                run_config = config.model_copy(deep=True)
                run_config.video.cache_dir = temporary / "frames"
                started = time.perf_counter()
                sample_id = f"{task}-{source_index}"
                trace_path: Path | None = None
                try:
                    if wanted_ids is not None:
                        # Same id formula as the adapter; checked before
                        # normalization so the proxy/video work is never paid
                        # for rows a targeted run does not want.
                        if f"{task}-{rows[source_index].get('id', source_index)}" not in wanted_ids:
                            continue
                    # Normalization can materialize an FSA/PSS proxy video.  Do
                    # it inside the per-index guard so one corrupt row loses
                    # only itself.
                    sample = adapter._normalize(rows[source_index], source_index=source_index)
                    sample_id = sample.sample_id
                    if wanted_ids is not None and sample_id not in wanted_ids:
                        continue
                    if sample_id in completed_ids:
                        print(f"[{sample_id}] skip: already present in results.jsonl", flush=True)
                        skipped += 1
                        continue
                    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
                    trace_path = trace_dir / f"{safe_id}.trace.json"
                    pipeline = CachedPhaseOnePipeline(
                        run_config, llm_backend=llm, vlm_backend=vlm,
                        cache_root=cache_root, signature_config=config,
                    )
                    state = pipeline.run(sample)
                    record = direct._record_success(
                        sample, state, metric, run_config, trace_path, time.perf_counter() - started,
                    )
                    done += 1
                except PipelineRunError as exc:
                    trace_path = trace_path or (trace_dir / f"{task}-{source_index}.trace.json")
                    _write_trace(trace_path, exc.state, run_config)
                    record = _failure_record(sample_id, str(exc), trace_path)
                    failed += 1
                    print(f"[{sample_id}] failed: {exc}", flush=True)
                except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
                    record = _failure_record(sample_id, f"{type(exc).__name__}: {exc}", None)
                    failed += 1
                    print(f"[{sample_id}] failed before trace: {exc}", flush=True)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
                    shutil.rmtree(ephemeral_root / "_probe" / task, ignore_errors=True)
                    llm.calls.clear()
                    vlm.calls.clear()
                    torch.cuda.empty_cache()
                records.append(record)
                completed_ids.add(str(record["sample_id"]))
                results.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.flush()
        summary = _summarize(records)
        (task_out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        print(f"[{task}] shard done: completed={done} failed={failed} skipped={skipped}", flush=True)
        print(summary.model_dump_json(indent=2), flush=True)

    shutil.rmtree(ephemeral_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
