#!/usr/bin/env python3
"""Minimal local Qwen3 smoke test.

This script is intentionally independent from the agent loop. It prints each
stage before running it, so a segfault tells you which stage was reached.
"""

import argparse
import os
import sys
import time
import traceback

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata


def log(message):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Test local Qwen3 loading and generation.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("LOCAL_MASTER_MODEL_PATH", "/media/data6/xuejj/Qwen3-8B"),
        help="Local HuggingFace model directory.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("LOCAL_MASTER_DEVICE", ""),
        help="Explicit device, e.g. cuda:2. Overrides --device-map when set.",
    )
    parser.add_argument(
        "--device-map",
        default=os.environ.get("LOCAL_MASTER_DEVICE_MAP", "auto"),
        help='Device map: auto, cpu, cuda:0, 0, or "none".',
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("LOCAL_MASTER_TORCH_DTYPE", "float16"),
        help="Torch dtype: float16, bfloat16, float32, or auto.",
    )
    parser.add_argument(
        "--backend",
        choices=["automodel", "pipeline"],
        default="automodel",
        help="automodel separates model load from generation; pipeline matches qwen_agent.py more closely.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt",
        default="Please answer in one short sentence: what is 2 plus 2?",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only load tokenizer/model, then exit.",
    )
    parser.add_argument(
        "--import-video-stack",
        action="store_true",
        help="Import cv2, decord, and utils.video_processor before loading Qwen3.",
    )
    parser.add_argument(
        "--probe-video",
        default="",
        help="Run video_processor.process_video on this video before loading Qwen3.",
    )
    return parser.parse_args()


def maybe_touch_video_stack(args):
    if not args.import_video_stack and not args.probe_video:
        return

    log("=== Video stack precheck ===")
    log("Importing cv2...")
    import cv2

    log(f"cv2: {cv2.__version__}")
    log("Importing decord...")
    import decord

    log(f"decord: {getattr(decord, '__version__', 'unknown')}")
    log("Importing utils.video_processor...")
    from utils import video_processor

    log("utils.video_processor imported.")
    if args.probe_video:
        log(f"Running video_processor.process_video on: {args.probe_video}")
        result = video_processor.process_video(
            input_path=args.probe_video,
            n_frames=1,
            intervals=[(0, 1)],
            max_length=360,
            encode=False,
        )
        _, _, _, original_fps, total_frames, duration = result
        log(f"video probe result: fps={original_fps}, total_frames={total_frames}, duration={duration}")
    log("")


def show_runtime(torch):
    log("=== Runtime ===")
    log(f"python: {sys.executable}")
    log(f"python_version: {sys.version.split()[0]}")
    log(f"torch: {getattr(torch, '__version__', 'unknown')}")
    log(f"torch_cuda: {getattr(torch.version, 'cuda', None)}")
    for package_name in ("transformers", "accelerate", "safetensors"):
        try:
            log(f"{package_name}: {importlib_metadata.version(package_name)}")
        except Exception:
            log(f"{package_name}: not installed")
    log(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"cuda_device_count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / 1024**3
            free_text = ""
            mem_get_info = getattr(torch.cuda, "mem_get_info", None)
            if mem_get_info is not None:
                try:
                    free_bytes, total_bytes = mem_get_info(idx)
                    free_text = f", free={free_bytes / 1024**3:.1f}GB"
                    total_gb = total_bytes / 1024**3
                except Exception:
                    pass
            log(f"gpu[{idx}]: {props.name}, capability={props.major}.{props.minor}, memory={total_gb:.1f}GB{free_text}")
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if bf16_supported is not None:
            try:
                log(f"bf16_supported_current_device: {bf16_supported()}")
            except Exception as exc:
                log(f"bf16_supported_current_device: check failed: {exc}")
    log("")


def resolve_dtype(torch, dtype_name):
    if dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown dtype {dtype_name!r}. Try float16, bfloat16, float32, or auto.")
    return getattr(torch, dtype_name)


def resolve_device_map(device, device_map):
    device = (device or "").strip()
    if device:
        return {"": device}

    device_map = (device_map or "").strip()
    if not device_map or device_map.lower() == "none":
        return None
    if device_map.isdigit():
        return {"": f"cuda:{device_map}"}
    if device_map.startswith(("cuda", "cpu", "mps")):
        return {"": device_map}
    return device_map


def build_prompt(tokenizer, user_prompt):
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_automodel(args, torch, dtype, device_map):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("=== Config ===")
    log(f"model_path: {args.model_path}")
    log(f"backend: automodel")
    log(f"dtype: {args.dtype} -> {dtype}")
    log(f"device_map: {device_map}")
    log("")

    log("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    log("[1/4] Tokenizer loaded.")

    log("[2/4] Loading model with AutoModelForCausalLM.from_pretrained...")
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()
    log("[2/4] Model loaded.")

    if args.skip_generate:
        log("skip_generate is set; exiting after model load.")
        return

    log("[3/4] Building prompt and tokenizing...")
    prompt_text = build_prompt(tokenizer, args.prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    first_param = next(model.parameters())
    inputs = {key: value.to(first_param.device) for key, value in inputs.items()}
    log(f"[3/4] Input tokens: {inputs['input_ids'].shape[-1]}, input_device: {inputs['input_ids'].device}")

    log("[4/4] Generating...")
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    start = time.time()
    inference_context = getattr(torch, "inference_mode", None)
    context = inference_context() if inference_context is not None else torch.no_grad()
    with context:
        output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start
    new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    log(f"[4/4] Done in {elapsed:.2f}s.")
    log("=== Output ===")
    log(text.strip())


def run_pipeline(args, torch, dtype, device_map):
    from transformers import AutoTokenizer, pipeline

    log("=== Config ===")
    log(f"model_path: {args.model_path}")
    log(f"backend: pipeline")
    log(f"dtype: {args.dtype} -> {dtype}")
    log(f"device_map: {device_map}")
    log("")

    log("[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    log("[1/3] Tokenizer loaded.")

    log("[2/3] Creating transformers.pipeline...")
    pipe_kwargs = {
        "task": "text-generation",
        "model": args.model_path,
        "tokenizer": tokenizer,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if dtype != "auto":
        pipe_kwargs["torch_dtype"] = dtype
    else:
        pipe_kwargs["torch_dtype"] = "auto"
    text_generator = pipeline(**pipe_kwargs)
    log("[2/3] Pipeline created.")

    if args.skip_generate:
        log("skip_generate is set; exiting after pipeline creation.")
        return

    log("[3/3] Generating...")
    prompt_text = build_prompt(tokenizer, args.prompt)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "return_full_text": False,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    start = time.time()
    outputs = text_generator(prompt_text, **gen_kwargs)
    elapsed = time.time() - start
    log(f"[3/3] Done in {elapsed:.2f}s.")
    log("=== Output ===")
    log(outputs[0].get("generated_text", "").strip() if outputs else "")


def main():
    args = parse_args()

    try:
        import faulthandler

        faulthandler.enable()
    except Exception:
        pass

    log("Importing torch...")
    import torch

    show_runtime(torch)
    maybe_touch_video_stack(args)
    dtype = resolve_dtype(torch, args.dtype)
    device_map = resolve_device_map(args.device, args.device_map)

    if args.backend == "pipeline":
        run_pipeline(args, torch, dtype, device_map)
    else:
        run_automodel(args, torch, dtype, device_map)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("=== Python exception ===")
        traceback.print_exc()
        sys.exit(1)
