#!/usr/bin/env python3
"""
DistillAgent — Entry Point
===========================
Distills correct multi-turn agent trajectories from the CrossVid training set
using a strong API model + diagnosis-based self-correction.

Usage:
    # Single task
    python run_distill.py --task PSS

    # Multiple tasks
    python run_distill.py --task PSS,FSA,CC

    # All tasks
    python run_distill.py --all

    # With custom settings
    python run_distill.py --task PSS --max-diagnosis 5 --max-turns 25 --start 1 --end 50

    # CCQA scoring (requires scoring API)
    python run_distill.py --task CCQA --scoring-api-url https://...

Output:
    output/
    ├── sft_data.jsonl              # SFT training data (OpenAI messages format)
    ├── raw_trajectories/            # Full trajectories for debugging
    │   ├── all_trajectories.jsonl
    │   └── trajectory_summary.json
    ├── failed_samples/              # Questions that couldn't be corrected
    │   └── failed_samples.json
    └── distillation_report.json     # Final report
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add agent_system to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_AGENT_ROOT = os.path.join(_PROJECT_ROOT, "agent_system")
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

# Load .env from agent_system/ (where all API keys are configured)
_DOTENV_PATH = os.path.join(_AGENT_ROOT, ".env")
if os.path.exists(_DOTENV_PATH):
    try:
        from dotenv import load_dotenv
        load_dotenv(_DOTENV_PATH)
    except ImportError:
        pass  # python-dotenv not installed, assume env vars are already set

from runtime_options import apply_runtime_overrides, resolve_prompt_config, runtime_summary

from task_config import TASK_CODES
from distill_core import DistillationOrchestrator
from trajectory_converter import (
    convert_samples_to_sft,
    save_raw_trajectories,
    save_failed_samples,
)

# Default paths
DEFAULT_QUESTION_DIR = os.path.join(_AGENT_ROOT, "question", "train")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")
DEFAULT_PROMPT_DIR = "myprompts"


def setup_logging(output_dir: str, task_filter: str) -> "logging.Logger":
    """Configure logging for ALL components (DistillAgent + agent_system internals)."""
    import logging

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"distill_{task_filter}_{timestamp}.log")

    # Console formatter: compact with module name
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    # File formatter: full detail
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    # ── Console handler (capture ALL loggers) ──
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)

    # ── File handler ──
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)

    # ── Configure root logger so agent_system modules are visible ──
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Remove any default handlers to avoid duplicate output
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(ch)
    root.addHandler(fh)

    # ── Ensure agent_system loggers are at INFO level ──
    for name in ["AgentWorkflowLogger", "qwen_agent", "agent_executor",
                 "DistillAgent", "DistillAgent.Core", "DistillAgent.Diagnosis",
                 "DistillAgent.TaskRunner", "DistillAgent.Converter",
                 "utils", "root"]:
        logging.getLogger(name).setLevel(logging.INFO)

    logger = logging.getLogger("DistillAgent")
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="DistillAgent: Distill correct agent trajectories with diagnosis-based self-correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_distill.py --task PSS
  python run_distill.py --task PSS,FSA,CC --max-diagnosis 5
  python run_distill.py --all --start 1 --end 100
  python run_distill.py --task PSS --output-dir ./my_distill_output
        """,
    )

    # Task selection
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", type=str, help=f"Comma-separated task codes: {','.join(TASK_CODES)}")
    task_group.add_argument("--all", action="store_true", help="Run all tasks")

    # Data paths
    parser.add_argument("--question-dir", type=str, default=DEFAULT_QUESTION_DIR,
                        help="Directory containing train question JSONs")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for SFT data and reports")

    # Distillation settings
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max agent turns per question")
    parser.add_argument("--max-diagnosis", type=int, default=3,
                        help="Max diagnosis+correction attempts per question")
    parser.add_argument("--start", type=int, default=1,
                        help="Start question index (1-based)")
    parser.add_argument("--end", type=int, default=None,
                        help="End question index (inclusive)")

    # Prompt
    parser.add_argument("--prompt-dir", type=str, default=DEFAULT_PROMPT_DIR,
                        help="Prompt directory under agent_system/")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Specific master prompt file path")

    # SFT output options
    parser.add_argument("--keep-incorrect", action="store_true",
                        help="Include incorrect samples in SFT data (for error-correction training)")
    parser.add_argument("--sft-format", choices=["jsonl", "json"], default="jsonl",
                        help="SFT output format")

    # API overrides (for running with different models per task)
    parser.add_argument("--master-model-name", type=str, help="Override master API model name")
    parser.add_argument("--diagnosis-model-name", type=str, help="Override diagnosis API model name")
    parser.add_argument("--tool-model-name", type=str, help="Override tool/VLM API model name")

    args = parser.parse_args()

    # ── Resolve tasks ──
    if args.all:
        tasks = list(TASK_CODES)
    else:
        tasks = [t.strip() for t in args.task.split(",")]
        for t in tasks:
            if t not in TASK_CODES:
                print(f"❌ Unknown task: {t}. Supported: {TASK_CODES}")
                sys.exit(1)

    # ── Setup output ──
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Setup logging ──
    task_filter = "all" if args.all else "-".join(tasks)
    logger = setup_logging(output_dir, task_filter)
    logger.info("DistillAgent starting | tasks=%s | output=%s", tasks, output_dir)

    # ── Pre-flight: API connectivity check ──
    import requests as _requests
    master_url = os.environ.get("MASTER_API_BASE_URL", "")
    master_model = os.environ.get("MASTER_MODEL_NAME", "")
    master_backend = os.environ.get("MASTER_BACKEND", "local_hf")
    logger.info("Master backend: %s | model: %s | url: %s", master_backend, master_model, master_url[:60])

    if master_backend in ("api",):
        try:
            _resp = _requests.post(
                master_url,
                headers={
                    "Authorization": f"Bearer {os.environ.get('MASTER_API_KEY', '')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": master_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 10,
                },
                timeout=15,
            )
            if _resp.status_code == 200:
                logger.info("✅ Master API OK (%d, %.1fs)", _resp.status_code, _resp.elapsed.total_seconds())
            else:
                logger.warning("⚠️  Master API returned %d: %s", _resp.status_code, _resp.text[:200])
        except Exception as e:
            logger.error("❌ Master API unreachable: %s", e)

    tool_url = os.environ.get("TOOL_API_BASE_URL", "")
    tool_model = os.environ.get("TOOL_MODEL_NAME", "")
    tool_backend = os.environ.get("TOOL_BACKEND", "local_qwenvl")
    logger.info("Tool backend: %s | model: %s | url: %s", tool_backend, tool_model, tool_url[:60])

    if tool_backend in ("api",):
        try:
            _resp = _requests.post(
                tool_url,
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOOL_API_KEY', '')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": tool_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 10,
                },
                timeout=15,
            )
            if _resp.status_code == 200:
                logger.info("✅ Tool API OK (%d, %.1fs)", _resp.status_code, _resp.elapsed.total_seconds())
            else:
                logger.warning("⚠️  Tool API returned %d: %s", _resp.status_code, _resp.text[:200])
        except Exception as e:
            logger.error("❌ Tool API unreachable: %s", e)

    # ── Apply API overrides ──
    if args.master_model_name:
        os.environ["MASTER_MODEL_NAME"] = args.master_model_name
    if args.diagnosis_model_name:
        os.environ["DIAGNOSIS_MODEL_NAME"] = args.diagnosis_model_name
    if args.tool_model_name:
        os.environ["TOOL_MODEL_NAME"] = args.tool_model_name

    # ── Build prompt config (reuse agent_system logic) ──
    # Create a minimal args object for resolve_prompt_config
    class PromptArgs:
        prompt_file = args.prompt_file
        prompt_dir = args.prompt_dir

    # Determine task code for prompt resolution (use first task's config)
    # For multi-task, each task's contract is loaded separately in the orchestrator
    first_task = tasks[0]

    # Build default config
    default_config = {
        "master_base": os.path.join(_AGENT_ROOT, args.prompt_dir, "_unified_crossvid_base_compress.prompt"),
        "task_contract": os.path.join(_AGENT_ROOT, args.prompt_dir, "task_contracts", f"{first_task}.prompt"),
        "task_code": first_task,
        "master_name": f"master_{first_task}.prompt",
    }

    # Verify files exist
    if not os.path.exists(default_config["master_base"]):
        logger.error(f"Base prompt not found: {default_config['master_base']}")
        logger.info("Make sure myprompts/_unified_crossvid_base.prompt exists under agent_system/")
        sys.exit(1)

    logger.info("Prompt config: base=%s", default_config["master_base"])

    # ── Run distillation per task ──
    all_samples = []
    all_failed = []
    total_start_time = time.time()

    for task in tasks:
        logger.info("\n" + "█" * 60)
        logger.info(f"  Starting Task: {task}")
        logger.info("█" * 60)

        # Update task contract for current task
        task_contract_path = os.path.join(
            _AGENT_ROOT, args.prompt_dir, "task_contracts", f"{task}.prompt"
        )
        if not os.path.exists(task_contract_path):
            logger.error(f"Task contract not found: {task_contract_path}, skipping {task}")
            continue

        prompt_config = dict(default_config)
        prompt_config["task_contract"] = task_contract_path
        prompt_config["task_code"] = task
        prompt_config["master_name"] = f"master_{task}.prompt"
        # Remove single-file master if base+contract are used
        prompt_config.pop("master", None)

        # Load question data
        question_file = os.path.join(args.question_dir, f"{task}.json")
        if not os.path.exists(question_file):
            logger.error(f"Question file not found: {question_file}, skipping {task}")
            continue

        with open(question_file, "r", encoding="utf-8") as f:
            all_questions = json.load(f)

        total_questions = len(all_questions)
        start_idx = max(0, args.start - 1)
        end_idx = min(total_questions, args.end) if args.end else total_questions
        questions = all_questions[start_idx:end_idx]

        logger.info(
            "Task %s: %d questions (Q%d-Q%d of %d total)",
            task, len(questions), start_idx + 1, end_idx, total_questions,
        )

        # Initialize orchestrator
        orchestrator = DistillationOrchestrator(
            prompt_config=prompt_config,
            max_turns=args.max_turns,
            max_diagnosis_attempts=args.max_diagnosis,
            output_dir=output_dir,
        )

        # Run
        task_samples, task_failed = orchestrator.run_distillation(
            task=task,
            questions=questions,
            start_index=start_idx,
        )

        all_samples.extend(task_samples)
        all_failed.extend(task_failed)

        # Per-task intermediate save
        if task_samples:
            task_sft_dir = os.path.join(output_dir, "per_task_sft")
            os.makedirs(task_sft_dir, exist_ok=True)
            convert_samples_to_sft(
                task_samples,
                os.path.join(task_sft_dir, f"{task}_sft.jsonl"),
                only_correct=not args.keep_incorrect,
                format_type="jsonl",
            )

        orchestrator.print_summary()

    total_elapsed = time.time() - total_start_time

    # ── Final output ──
    logger.info("\n" + "=" * 60)
    logger.info("  Generating final outputs...")

    # SFT data (all tasks merged, correct-only by default)
    sft_output_path = os.path.join(output_dir, "sft_data.jsonl")
    convert_samples_to_sft(
        all_samples,
        sft_output_path,
        only_correct=not args.keep_incorrect,
        format_type=args.sft_format,
    )

    # Raw trajectories
    raw_dir = os.path.join(output_dir, "raw_trajectories")
    save_raw_trajectories(all_samples, raw_dir)

    # Failed samples
    failed_dir = os.path.join(output_dir, "failed_samples")
    save_failed_samples(all_failed, failed_dir)

    # Final report
    correct_count = sum(1 for s in all_samples if s.is_correct)
    incorrect_count = sum(1 for s in all_samples if not s.is_correct)
    with_diagnosis = sum(1 for s in all_samples if len(s.diagnosis_history) > 0)

    report = {
        "timestamp": datetime.now().isoformat(),
        "tasks": tasks,
        "total_questions": len(all_samples),
        "correct": correct_count,
        "incorrect": incorrect_count,
        "with_diagnosis_correction": with_diagnosis,
        "failed_samples": len(all_failed),
        "total_time_seconds": total_elapsed,
        "settings": {
            "max_turns": args.max_turns,
            "max_diagnosis_attempts": args.max_diagnosis,
            "question_range": f"Q{args.start}-Q{args.end or 'end'}",
            "keep_incorrect": args.keep_incorrect,
        },
    }
    report_path = os.path.join(output_dir, "distillation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  🎉 Distillation Complete!")
    print("=" * 60)
    print(f"  Total time:     {total_elapsed/60:.1f} min")
    print(f"  Total samples:  {len(all_samples)}")
    print(f"  ✅ Correct:     {correct_count}")
    print(f"  ❌ Incorrect:   {incorrect_count}")
    print(f"  🩺 With diagnosis: {with_diagnosis}")
    print(f"\n  Output files:")
    print(f"    SFT data:      {sft_output_path}")
    print(f"    Raw trajectories: {raw_dir}")
    print(f"    Failed samples:   {failed_dir}")
    print(f"    Report:        {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
