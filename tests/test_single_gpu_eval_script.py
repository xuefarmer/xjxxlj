"""Single-GPU cached-evaluation wrapper tests without loading model weights."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval_cached_single_gpu.py"
SPEC = importlib.util.spec_from_file_location("eval_cached_single_gpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DEFAULT_OUTPUT = MODULE.DEFAULT_OUTPUT
_forwarded_arguments = MODULE._forwarded_arguments
_parser = MODULE._parser


def test_single_gpu_wrapper_uses_distinct_output_and_one_cuda_device() -> None:
    args = _parser().parse_args([])
    forwarded = _forwarded_arguments(args)

    assert args.output_dir == DEFAULT_OUTPUT
    assert args.output_dir.name == "focus-fix1-cc-single-a100"
    assert forwarded.count("cuda:0") == 2
    assert "cuda:1" not in forwarded
    assert "--resume" in forwarded


def test_single_gpu_wrapper_forwards_optional_selection_arguments() -> None:
    args = _parser().parse_args(["--limit", "3", "--sample-ids", "CC-0,CC-2"])
    forwarded = _forwarded_arguments(args)

    assert forwarded[forwarded.index("--limit") + 1] == "3"
    assert forwarded[forwarded.index("--sample-ids") + 1] == "CC-0,CC-2"
