#!/usr/bin/env python3
"""Direct multi-GPU Event-Observer preheat for staged local evaluation.

Unlike ``run_staged_local_lora_eval.py preheat``, this utility has no HTTP
servers or OpenAI-compatible routing layer.  It spawns one process per GPU;
each process loads the VLM LoRA locally, owns a deterministic shard of the QA
rows, and retains only the Event Observer text packets under ``--cache-dir``.
Decoded JPEGs and temporary FSA proxies are removed after every sample.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


class _UnusedLLM:
    """Phase A never calls the Reasoner, but Pipeline construction requires it."""

    calls: list[Any]

    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, **_kwargs: Any) -> str:
        raise RuntimeError("LLM must not be called during Event preheat")


def _warm_settings(args: argparse.Namespace, task: str, config: Any) -> tuple[float, int, int]:
    high = {item.strip().upper() for item in args.high_density_tasks.split(",") if item.strip()}
    compact = {item.strip().upper() for item in args.compact_observer_tasks.split(",") if item.strip()}
    if task in high:
        return args.high_density_seconds_per_frame, args.high_density_min_warm_frames, args.high_density_scan_frames
    if task in compact:
        return args.compact_scan_seconds_per_frame, args.compact_min_warm_frames, args.compact_scan_frames
    return config.video.scan_seconds_per_frame, config.budget.min_warm_frames_per_video, config.budget.scan_frames_per_video


def _resolve_gpus(requested: str) -> list[str]:
    """Resolve ``auto`` without creating a CUDA context in the parent.

    If the caller exports ``CUDA_VISIBLE_DEVICES=0,5``, its entries are used
    as physical selectors for the spawned children.  Otherwise query the
    number of locally visible devices and use the conventional 0..N-1 ids.
    Each child subsequently exposes exactly one selector to torch.
    """
    if requested.strip().lower() != "auto":
        return [item.strip() for item in requested.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"all", "-1"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    import torch
    return [str(index) for index in range(torch.cuda.device_count())]


def _worker(args_dict: dict[str, Any], worker_index: int, gpu: str, world_size: int) -> None:
    # Set this before importing torch/transformers in the child process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from experiments.event_level_observer.event_pipeline import EventLevelObserverAgent
    from scripts.eval_local_dual_lora_direct import DirectBackend
    from scripts.run_staged_local_lora_eval import answer_type_for, cache_path, cache_signature, valid_cache
    from unicvr.config import load_config
    from unicvr.core.pipeline import Pipeline, VideoProbe
    from unicvr.data.crossvid import CrossVidDatasetAdapter
    from unicvr.data.materialize import OpenCVMediaMaterializer
    from unicvr.plugins.registry import blocks_for_sample, visual_question_for_sample

    args = argparse.Namespace(**args_dict)
    cache_root = Path(args.cache_dir).resolve()
    transient_root = cache_root / f"_transient_worker{worker_index}"
    frame_root = transient_root / "frames"
    media_root = transient_root / "media"
    prompt_path = Path(args.event_observer_prompt).resolve()
    config = load_config(Path(args.config))
    config.video.cache_dir = frame_root

    print(f"[worker {worker_index}] loading VLM {args.vlm_model} + {args.vlm_adapter} on physical GPU {gpu}", flush=True)
    processor = AutoProcessor.from_pretrained(args.vlm_model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.vlm_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0").eval()
    model = PeftModel.from_pretrained(model, args.vlm_adapter).to("cuda:0").eval()
    vlm = DirectBackend(name=f"event-vlm-lora-gpu{gpu}", model=model, tokenizer=processor.tokenizer,
                        processor=processor, device=torch.device("cuda:0"), max_tokens=args.max_vlm_tokens)
    pipe = Pipeline(config, llm_backend=_UnusedLLM(), vlm_backend=vlm)
    observer = EventLevelObserverAgent(vlm, config.generation, prompt_path)
    materializer = OpenCVMediaMaterializer(media_root)

    completed = skipped = failed = 0
    try:
        for task in [item.strip().upper() for item in args.tasks.split(",") if item.strip()]:
            adapter = CrossVidDatasetAdapter(crossvid_root=Path(args.crossvid_root), task=task, materializer=materializer)
            seconds_per_frame, minimum_frames, maximum_frames = _warm_settings(args, task, config)
            print(f"[worker {worker_index}] [{task}] {seconds_per_frame}s/frame min={minimum_frames} cap={maximum_frames}", flush=True)
            # Normalization can materialize an FSA/PSS proxy video.  Do it
            # inside the per-index guard: a corrupt source or an intermittent
            # OpenCV writer failure must lose only this one QA row, never the
            # complete GPU worker/generation shard.
            rows = json.loads(adapter.dataset_path.read_text(encoding="utf-8"))
            stop = min(len(rows), args.start + args.limit)
            for source_index in range(args.start, stop):
                if source_index % world_size != worker_index:
                    continue
                sample_label = f"{task}-{source_index}"
                try:
                    row = rows[source_index]
                    if not isinstance(row, dict):
                        raise ValueError(f"CrossVid row {source_index} must be an object")
                    sample = adapter._normalize(row, source_index=source_index)
                    sample_label = sample.sample_id
                    output = cache_path(cache_root, task, sample.sample_id)
                    if valid_cache(output, sample, config):
                        skipped += 1
                        continue
                    videos = [VideoProbe().probe(path, video_id=f"v{i + 1}") for i, path in enumerate(sample.video_paths)]
                    reports: list[dict[str, Any]] = []
                    for video in videos:
                        visual_question = visual_question_for_sample(sample, sample.question, video.video_id)
                        prompt_blocks = blocks_for_sample(
                            sample, "observer", question=visual_question, answer_type=answer_type_for(task),
                            options=sample.options, video_id=video.video_id,
                        )
                        frame_budget = pipe.warm_sampler.frame_count(
                            video, minimum=minimum_frames, maximum=maximum_frames,
                            seconds_per_frame=seconds_per_frame,
                        )
                        visuals = pipe.decoder.extract(video, pipe.warm_sampler.sample(
                            video, frame_budget=frame_budget, question=sample.question,
                        ))
                        chunks: list[dict[str, Any]] = []
                        for chunk_index, inputs in enumerate(pipe.warm_sampler.chunk(
                            visuals, frames_per_call=config.video.scan_frames_per_call,
                            overlap_frames=config.video.scan_overlap_frames,
                        )):
                            report = observer.observe(
                                video_id=video.video_id, visual_inputs=inputs,
                                task_context=("Observe all answer-relevant events, states and temporal cues. "
                                              "A later Reasoner compares videos to answer: " + visual_question),
                                prompt_blocks=prompt_blocks,
                            )
                            chunks.append({"index": chunk_index, "report_text": report,
                                           "frame_count": len(inputs),
                                           "start_time": inputs[0].timestamp_seconds,
                                           "end_time": inputs[-1].timestamp_seconds})
                        reports.append({
                            "video_id": video.video_id,
                            "report_text": "\n\n".join(item["report_text"] for item in chunks),
                            "frame_count": sum(item["frame_count"] for item in chunks),
                            "start_time": min(item["start_time"] for item in chunks),
                            "end_time": max(item["end_time"] for item in chunks),
                        })
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps({
                        "sample_id": sample.sample_id, "task": task,
                        "signature": cache_signature(sample, config),
                        "config_fingerprint": config.fingerprint(), "reports": reports,
                        "preheat_calls": [
                            {"role": "ObserverAgent.EVENT", "backend": "local_direct_lora",
                             "model": f"event-vlm-lora-gpu{gpu}", "visual_input_count": report["frame_count"],
                             "usage": {}, "retry_count": 0}
                            for report in reports
                        ],
                    }, ensure_ascii=False), encoding="utf-8")
                    completed += 1
                    print(f"=== [PREHEAT DONE] {task}:{sample.sample_id} | {len(reports)} video report(s) cached | worker={worker_index} ===", flush=True)
                except Exception as exc:  # retain the remainder of the full benchmark
                    failed += 1
                    print(f"[worker {worker_index}] [{task}:{sample_label}] failed: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    # Reports above are the only persistent artifact of Phase A.
                    shutil.rmtree(frame_root, ignore_errors=True)
                    shutil.rmtree(media_root, ignore_errors=True)
                    vlm.calls.clear()
                    torch.cuda.empty_cache()
    finally:
        shutil.rmtree(transient_root, ignore_errors=True)
    print(f"[worker {worker_index}] finished: cached={completed} skipped={skipped} failed={failed}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="configs/local_dual_lora_eval_longvideo_compact.yaml")
    parser.add_argument("--crossvid-root", default="/media/data6/xuejj/CrossVid")
    parser.add_argument("--tasks", default="PSS,CC,FSA,NC,PI,MSR,MOC")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gpus", default="auto",
                        help="physical GPU ids, e.g. 0,1,2,3; auto uses CUDA_VISIBLE_DEVICES or all visible GPUs")
    parser.add_argument("--vlm-model", default="/media/data6/xuejj/Qwen-VL")
    parser.add_argument("--vlm-adapter", required=True)
    parser.add_argument("--max-vlm-tokens", type=int, default=0, help="0 uses config generation max")
    parser.add_argument("--event-observer-prompt", default="experiments/event_level_observer/observer_event.v1.txt")
    parser.add_argument("--high-density-tasks", default="PSS,FSA,PI,MSR,MOC")
    parser.add_argument("--high-density-seconds-per-frame", type=float, default=2.0)
    parser.add_argument("--high-density-min-warm-frames", type=int, default=16)
    parser.add_argument("--high-density-scan-frames", type=int, default=192)
    parser.add_argument("--compact-observer-tasks", default="CC,NC")
    parser.add_argument("--compact-scan-seconds-per-frame", type=float, default=8.0)
    parser.add_argument("--compact-min-warm-frames", type=int, default=12)
    parser.add_argument("--compact-scan-frames", type=int, default=32)
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1:
        raise SystemExit("--start must be >=0 and --limit must be positive")
    gpus = _resolve_gpus(args.gpus)
    if not gpus:
        raise SystemExit("no CUDA GPUs selected; set CUDA_VISIBLE_DEVICES or pass --gpus")
    if not Path(args.event_observer_prompt).resolve().is_file():
        raise SystemExit(f"event Observer prompt does not exist: {args.event_observer_prompt}")
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"[launcher] workers={len(gpus)} physical GPU selector(s): {','.join(gpus)}", flush=True)
    context = mp.get_context("spawn")
    workers = [context.Process(target=_worker, args=(vars(args), index, gpu, len(gpus)), daemon=False)
               for index, gpu in enumerate(gpus)]
    for process in workers:
        process.start()
    for process in workers:
        process.join()
    failed = [process.pid for process in workers if process.exitcode != 0]
    if failed:
        raise SystemExit(f"preheat worker process(es) failed: {failed}")
    print(f"preheat cache: {Path(args.cache_dir).resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
