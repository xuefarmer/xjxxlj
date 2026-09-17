#!/usr/bin/env python3
"""Run cached UniCVR evaluation with the LLM and VLM on one physical GPU.

This is a thin, result-compatible entry point around
``eval_cached_local_direct.py``.  It exposes exactly one physical GPU before
PyTorch is imported, maps both model backends to process-local ``cuda:0``, and
uses a separate output directory by default so dual-GPU results are untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/eval-event-rl200-staged/focus-fix1-cc-single-a100"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--physical-gpu",
        default="0",
        help="single physical CUDA index or GPU UUID to expose",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/local_dual_lora_eval_longvideo_compact.yaml",
    )
    parser.add_argument("--tasks", default="CC")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs/eval-event-rl200-staged/preheat",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--llm-adapter",
        type=Path,
        default=ROOT / "outputs/hc-magrpo-event-v1-stage1/step-000200/llm",
    )
    parser.add_argument(
        "--vlm-adapter",
        type=Path,
        default=ROOT / "outputs/hc-magrpo-event-v1-stage1/step-000200/vlm",
    )
    parser.add_argument("--llm-model", default="/media/data6/xuejj/Qwen3-8B")
    parser.add_argument("--vlm-model", default="/media/data6/xuejj/Qwen-VL")
    parser.add_argument("--crossvid-root", type=Path, default=Path("/media/data6/xuejj/CrossVid"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids", default=None)
    parser.add_argument("--max-llm-tokens", type=int, default=0)
    parser.add_argument("--max-vlm-tokens", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def _forwarded_arguments(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--config",
        str(args.config),
        "--crossvid-root",
        str(args.crossvid_root),
        "--tasks",
        args.tasks,
        "--start",
        str(args.start),
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(args.output_dir),
        "--resume",
        "--llm-model",
        args.llm_model,
        "--llm-adapter",
        str(args.llm_adapter),
        "--vlm-model",
        args.vlm_model,
        "--vlm-adapter",
        str(args.vlm_adapter),
        "--llm-device",
        "cuda:0",
        "--vlm-device",
        "cuda:0",
        "--max-llm-tokens",
        str(args.max_llm_tokens),
        "--max-vlm-tokens",
        str(args.max_vlm_tokens),
        "--shard-index",
        str(args.shard_index),
        "--shard-count",
        str(args.shard_count),
    ]
    if args.limit is not None:
        forwarded.extend(("--limit", str(args.limit)))
    if args.sample_ids:
        forwarded.extend(("--sample-ids", args.sample_ids))
    return forwarded


def main() -> int:
    args = _parser().parse_args()
    physical_gpu = args.physical_gpu.strip()
    if not physical_gpu or "," in physical_gpu:
        raise SystemExit("--physical-gpu must identify exactly one GPU")

    # These must be set before importing the evaluator, which imports PyTorch.
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from eval_cached_local_direct import main as cached_main

    sys.argv = [str(Path(__file__).resolve()), *_forwarded_arguments(args)]
    print(
        f"[single-gpu] physical={physical_gpu} llm=cuda:0 vlm=cuda:0 output={args.output_dir}",
        flush=True,
    )
    return cached_main()


if __name__ == "__main__":
    raise SystemExit(main())
