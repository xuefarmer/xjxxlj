#!/usr/bin/env python3
"""Direct two-GPU local-LoRA evaluation for the complete UniCVR pipeline.

Unlike the HTTP/staged diagnostics, this process loads the LLM and VLM LoRAs
once and calls them in-process.  A question still runs the production protocol
unchanged: Observer -> FORM -> optional Focus/Comparer -> REVIEW.  The config
controls video sampling; use the compact long-video config only when evaluating
CC/NC, so PSS/FSA retain their normal temporal density.

The source videos and QA are never written to.  Each sample's proxy media and
decoded frames live under ``OUTPUT/.ephemeral`` and are deleted immediately
after its trace/result has been saved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence, TypeVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from unicvr.config import load_config
from unicvr.core.pipeline import Pipeline, PipelineRunError
from unicvr.core.schemas import APIUsage, BackendCallRecord, GenerationConfig, VisualInput
from unicvr.data.crossvid import CrossVidDatasetAdapter
from unicvr.data.materialize import OpenCVMediaMaterializer
from unicvr.evaluation import crossvid_metric
from unicvr.evaluation.evaluator import _summarize, _write_trace

T = TypeVar("T", bound=BaseModel)


class DirectBackend:
    """Thread-safe direct Transformers backend matching the pipeline protocol."""

    def __init__(self, *, name: str, model: torch.nn.Module, tokenizer: Any,
                 processor: Any | None, device: torch.device, max_tokens: int) -> None:
        self.name = name
        self.model, self.tokenizer, self.processor, self.device = model, tokenizer, processor, device
        self.max_tokens = max_tokens
        self.calls: list[BackendCallRecord] = []
        self._lock = threading.Lock()

    def _messages(self, system_prompt: str, user_prompt: str,
                  visuals: Sequence[VisualInput]) -> list[dict[str, Any]]:
        if self.processor is None:
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for item in visuals:
            label = item.label or f"video_id={item.video_id} | time={item.timestamp_seconds:.3f}s | frame={item.frame_index}"
            content.extend((
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": f"file://{item.local_path}"}},
            ))
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

    def _generate(self, *, role: str, system_prompt: str, user_prompt: str,
                  generation_config: GenerationConfig, visual_inputs: Sequence[VisualInput]) -> str:
        messages = self._messages(system_prompt, user_prompt, visual_inputs)
        # A single model replica is not re-entrant.  Observer tasks are still
        # submitted in parallel by Pipeline, but are safely serialized here.
        with self._lock:
            started = time.perf_counter()
            print(f"[local {self.name}] {role} start frames={len(visual_inputs)}", flush=True)
            images: list[Image.Image] = []
            try:
                if self.processor is None:
                    text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                else:
                    text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    for item in visual_inputs:
                        with Image.open(item.local_path) as raw:
                            images.append(raw.convert("RGB"))
                    inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens or generation_config.max_output_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                ids = output[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(ids, skip_special_tokens=True)
            finally:
                for image in images:
                    image.close()
            elapsed = time.perf_counter() - started
            print(f"[local {self.name}] {role} done {elapsed:.1f}s output_tokens={len(ids)}", flush=True)
        self.calls.append(BackendCallRecord(
            role=role, backend="local_direct_lora", model=self.name,
            visual_input_count=len(visual_inputs), usage=APIUsage(output_tokens=len(ids)),
            raw_response=response, system_prompt=system_prompt, user_prompt=user_prompt,
        ))
        return response

    def generate_text(self, *, role: str, system_prompt: str, user_prompt: str,
                      generation_config: GenerationConfig,
                      visual_inputs: Sequence[VisualInput] = ()) -> str:
        return self._generate(role=role, system_prompt=system_prompt, user_prompt=user_prompt,
                              generation_config=generation_config, visual_inputs=visual_inputs)

    def generate_structured(self, *, role: str, system_prompt: str, user_prompt: str,
                            output_schema: type[T], generation_config: GenerationConfig,
                            visual_inputs: Sequence[VisualInput] = ()) -> T:
        raw = self._generate(role=role, system_prompt=system_prompt, user_prompt=user_prompt,
                             generation_config=generation_config, visual_inputs=visual_inputs)
        return output_schema.model_validate_json(raw)


def load_backends(args: argparse.Namespace) -> tuple[DirectBackend, DirectBackend]:
    llm_device, vlm_device = torch.device(args.llm_device), torch.device(args.vlm_device)
    print(f"[load] LLM {args.llm_model} + {args.llm_adapter} -> {llm_device}", flush=True)
    # HC-MA-GRPO checkpoints deliberately contain only PEFT adapter weights;
    # their ``config.json`` is an adapter config, not a HuggingFace model
    # config.  The tokenizer must therefore always come from the immutable
    # base model (same rule as the RL trainer itself).
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(llm_device).eval()
    llm_model = PeftModel.from_pretrained(llm_model, args.llm_adapter).to(llm_device).eval()

    print(f"[load] VLM {args.vlm_model} + {args.vlm_adapter} -> {vlm_device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.vlm_model, trust_remote_code=True)
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.vlm_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(vlm_device).eval()
    vlm_model = PeftModel.from_pretrained(vlm_model, args.vlm_adapter).to(vlm_device).eval()
    print("[load] direct local LoRAs ready", flush=True)
    return (
        DirectBackend(name="llm-lora", model=llm_model, tokenizer=llm_tokenizer,
                      processor=None, device=llm_device, max_tokens=args.max_llm_tokens),
        DirectBackend(name="vlm-lora", model=vlm_model, tokenizer=processor.tokenizer,
                      processor=processor, device=vlm_device, max_tokens=args.max_vlm_tokens),
    )


def _record_success(sample: Any, state: Any, metric: Any, config: Any, trace_path: Path, elapsed: float) -> dict[str, Any]:
    _write_trace(trace_path, state, config)
    prediction = state.final_answer or ""
    score = metric.score(prediction, sample)
    visual_calls = sum(1 for call in state.api_calls if call.visual_input_count > 0)
    llm_calls = sum(1 for call in state.api_calls if call.visual_input_count == 0)
    frames = sum(call.visual_input_count for call in state.api_calls)
    print(f"[{sample.sample_id}] answer={prediction!r} gt={sample.answer!r} "
          f"score={score.score} correct={score.correct} elapsed={elapsed:.1f}s", flush=True)
    return {
        "sample_id": sample.sample_id, "prediction": prediction,
        "normalized_prediction": score.normalized_prediction, "ground_truth": sample.answer,
        "normalized_ground_truth": score.normalized_ground_truth, "score": score.score,
        "score_details": score.details, "correct": score.correct,
        "action_trace": state.action_trace,
        "observer_reports": [{"video_id": item.video_id, "frame_count": item.frame_count} for item in state.observer_reports],
        "evidence_count": len(state.evidence_results),
        "api_calls": [item.model_dump(mode="json") for item in state.api_calls],
        "budget_usage": {"used_frames": frames, "used_visual_calls": visual_calls,
                         "used_llm_calls": llm_calls, "current_round": 1},
        "stop_reason": state.stop_reason, "trace_path": str(trace_path.resolve()),
        "elapsed_seconds": elapsed, "error": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/local_dual_lora_eval.yaml")
    ap.add_argument("--crossvid-root", type=Path, default=Path("/media/data6/xuejj/CrossVid"))
    ap.add_argument("--tasks", default="CC,NC")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing direct-eval directory and skip sample_ids already in results.jsonl")
    ap.add_argument("--llm-model", default="/media/data6/xuejj/Qwen3-8B")
    ap.add_argument("--llm-adapter", type=Path, required=True)
    ap.add_argument("--vlm-model", default="/media/data6/xuejj/Qwen-VL")
    ap.add_argument("--vlm-adapter", type=Path, required=True)
    ap.add_argument("--llm-device", default="cuda:0"); ap.add_argument("--vlm-device", default="cuda:1")
    ap.add_argument("--max-llm-tokens", type=int, default=0, help="0 uses config generation max")
    ap.add_argument("--max-vlm-tokens", type=int, default=0, help="0 uses config generation max")
    ap.add_argument("--scan-seconds-per-frame", type=float, default=None,
                    help="override only the Phase-1 sampling stride; smaller means denser warm observation")
    ap.add_argument("--scan-frames-per-video", type=int, default=None,
                    help="override the Phase-1 per-video frame cap")
    ap.add_argument("--min-warm-frames-per-video", type=int, default=None,
                    help="override the Phase-1 minimum number of frames, useful for short clips")
    ap.add_argument("--event-observer-prompt", type=Path, default=None,
                    help="opt in to experiments/event_level_observer; replaces only Phase-1 Observer")
    ap.add_argument("--high-density-tasks", default="PSS,FSA,PI,MSR,MOC",
                    help="tasks whose short/UAV clips use Event-FORM training density")
    ap.add_argument("--high-density-seconds-per-frame", type=float, default=2.0)
    ap.add_argument("--high-density-min-warm-frames", type=int, default=16)
    ap.add_argument("--high-density-scan-frames", type=int, default=192)
    args = ap.parse_args()
    if args.start < 0 or args.limit <= 0: raise SystemExit("--start must be >=0 and --limit must be >0")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The normal YAMLs name HTTP endpoints via environment variables.  Direct
    # mode never contacts them, but AppConfig still validates those fields.
    os.environ.setdefault("LLM_BASE_URL", "http://local-direct.invalid/v1")
    os.environ.setdefault("LLM_API_KEY", "local")
    os.environ.setdefault("LLM_MODEL", "local-llm-lora")
    os.environ.setdefault("VLM_BASE_URL", "http://local-direct.invalid/v1")
    os.environ.setdefault("VLM_API_KEY", "local")
    os.environ.setdefault("VLM_MODEL", "local-vlm-lora")
    config = load_config(args.config)
    if args.scan_seconds_per_frame is not None:
        if args.scan_seconds_per_frame <= 0:
            raise SystemExit("--scan-seconds-per-frame must be > 0")
        config.video.scan_seconds_per_frame = args.scan_seconds_per_frame
    if args.scan_frames_per_video is not None:
        if args.scan_frames_per_video <= 0:
            raise SystemExit("--scan-frames-per-video must be > 0")
        config.budget.scan_frames_per_video = args.scan_frames_per_video
    if args.min_warm_frames_per_video is not None:
        if args.min_warm_frames_per_video <= 0:
            raise SystemExit("--min-warm-frames-per-video must be > 0")
        config.budget.min_warm_frames_per_video = args.min_warm_frames_per_video
    print(
        "[sampling base; used by non-high-density tasks such as CC/NC] "
        "warm stride={}s, min_frames/video={}, cap/video={}".format(
            config.video.scan_seconds_per_frame,
            config.budget.min_warm_frames_per_video,
            config.budget.scan_frames_per_video,
        ),
        flush=True,
    )
    llm, vlm = load_backends(args)
    event_prompt = args.event_observer_prompt.resolve() if args.event_observer_prompt else None
    if event_prompt is not None and not event_prompt.is_file():
        raise SystemExit(f"event Observer prompt does not exist: {event_prompt}")
    if event_prompt is not None:
        from experiments.event_level_observer.event_pipeline import EventLevelObserverPipeline
        print(f"[mode] event-level warm Observer: {event_prompt}", flush=True)
    tasks = [item.strip().upper() for item in args.tasks.split(",") if item.strip()]
    high_density = {item.strip().upper() for item in args.high_density_tasks.split(",") if item.strip()}
    records: list[dict[str, Any]] = []
    result_path = args.output_dir / "results.jsonl"
    if args.resume and result_path.is_file():
        for raw in result_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                records.append(json.loads(raw))
    completed_ids = {str(item.get("sample_id")) for item in records}
    trace_dir = args.output_dir / "traces"; trace_dir.mkdir(parents=True, exist_ok=True)
    ephemeral_root = args.output_dir / ".ephemeral"

    with result_path.open("a" if args.resume else "w", encoding="utf-8") as results:
        for task in tasks:
            metric = crossvid_metric(task, 0)
            if task in high_density:
                print(f"=== [{task}] high-density Event sampling: "
                      f"{args.high_density_seconds_per_frame}s/frame, "
                      f"min={args.high_density_min_warm_frames}, "
                      f"cap={args.high_density_scan_frames} ===", flush=True)
            else:
                print(f"=== [{task}] config Event sampling: "
                      f"{config.video.scan_seconds_per_frame}s/frame, "
                      f"min={config.budget.min_warm_frames_per_video}, "
                      f"cap={config.budget.scan_frames_per_video} ===", flush=True)
            print(f"=== [{task}] direct local full pipeline ===", flush=True)
            for source_index in range(args.start, args.start + args.limit):
                temporary = ephemeral_root / task / str(source_index)
                run_config = config.model_copy(deep=True)
                if task in high_density:
                    run_config.video.scan_seconds_per_frame = args.high_density_seconds_per_frame
                    run_config.budget.min_warm_frames_per_video = args.high_density_min_warm_frames
                    run_config.budget.scan_frames_per_video = args.high_density_scan_frames
                run_config.video.cache_dir = temporary / "frames"
                materializer = OpenCVMediaMaterializer(temporary / "media")
                try:
                    adapter = CrossVidDatasetAdapter(crossvid_root=args.crossvid_root, task=task, materializer=materializer)
                    sample = next(adapter.iter_samples(start=source_index, limit=1), None)
                    if sample is None:
                        break
                    if sample.sample_id in completed_ids:
                        print(f"[{sample.sample_id}] skip: already present in results.jsonl", flush=True)
                        continue
                    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id)
                    trace_path = trace_dir / f"{safe_id}.trace.json"
                    started = time.perf_counter()
                    if event_prompt is None:
                        pipeline = Pipeline(run_config, llm_backend=llm, vlm_backend=vlm)
                    else:
                        pipeline = EventLevelObserverPipeline(
                            run_config, llm_backend=llm, vlm_backend=vlm,
                            event_prompt_path=event_prompt,
                        )
                    state = pipeline.run(sample)
                    record = _record_success(sample, state, metric, run_config, trace_path, time.perf_counter() - started)
                except PipelineRunError as exc:
                    trace_path = trace_dir / f"{task}-{source_index}.trace.json"
                    _write_trace(trace_path, exc.state, run_config)
                    record = {"sample_id": f"{task}-{source_index}", "prediction": None, "normalized_prediction": None,
                              "ground_truth": None, "normalized_ground_truth": None, "score": None, "score_details": {},
                              "correct": False, "action_trace": [], "observer_reports": [], "evidence_count": 0,
                              "api_calls": [], "budget_usage": None, "stop_reason": "error", "trace_path": str(trace_path.resolve()),
                              "elapsed_seconds": 0.0, "error": str(exc)}
                    print(f"[{task}-{source_index}] failed: {exc}", flush=True)
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    record = {"sample_id": f"{task}-{source_index}", "prediction": None, "normalized_prediction": None,
                              "ground_truth": None, "normalized_ground_truth": None, "score": None, "score_details": {},
                              "correct": False, "action_trace": [], "observer_reports": [], "evidence_count": 0,
                              "api_calls": [], "budget_usage": None, "stop_reason": "error", "trace_path": None,
                              "elapsed_seconds": 0.0, "error": f"{type(exc).__name__}: {exc}"}
                    print(f"[{task}-{source_index}] failed before trace: {exc}", flush=True)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
                    llm.calls.clear(); vlm.calls.clear()
                    torch.cuda.empty_cache()
                records.append(record)
                completed_ids.add(str(record["sample_id"]))
                results.write(json.dumps(record, ensure_ascii=False) + "\n"); results.flush()

    shutil.rmtree(ephemeral_root, ignore_errors=True)
    summary = _summarize(records)
    (args.output_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(summary.model_dump_json(indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
