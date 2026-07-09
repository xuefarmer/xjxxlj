"""
Convert distilled trajectories to SFT training format.

Output formats:
  1. JSONL with "messages" field (OpenAI-compatible, for verl SFT)
  2. Raw trajectories preserved for debugging/analysis

For diagnosis-corrected trajectories, injected diagnosis messages are stripped
so the SFT data looks like a clean, naturally-correct trajectory.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

import logging
logger = logging.getLogger("DistillAgent.Converter")

# ── Diagnosis message markers (stripped from SFT output) ──
DIAGNOSIS_MARKERS = [
    "[Diagnosis Correction",
    "[FULL RESTART WITH DIAGNOSIS",
    "[DIAGNOSIS GUIDANCE",
]


def _sanitize_content_for_sft(content: Any) -> str:
    """Convert message content to a clean string for SFT."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif "text" in item:
                    text_parts.append(str(item["text"]))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    if content is None:
        return ""
    return str(content)


def _is_diagnosis_message(content: str) -> bool:
    """Check if a message is an injected diagnosis correction."""
    for marker in DIAGNOSIS_MARKERS:
        if marker in content:
            return True
    return False


def strip_diagnosis_messages(
    master_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove injected diagnosis messages and the wrong turns they corrected,
    producing a clean trajectory suitable for SFT training.

    For each diagnosis injection found:
      - Remove the injection message itself
      - Remove the wrong answer (last assistant turn before injection)
    Result: correct early turns -> corrected later turns -> answer.
    Looks exactly like a naturally correct trajectory.
    """
    if not master_history:
        return master_history

    clean = []
    for msg in master_history:
        role = msg.get("role", "")
        content = _sanitize_content_for_sft(msg.get("content", ""))

        if role == "user" and _is_diagnosis_message(content):
            # Remove the last assistant turn (wrong answer that triggered diagnosis)
            while clean and clean[-1].get("role") == "assistant":
                clean.pop()
            # Skip the diagnosis injection itself
            continue

        clean.append(msg)

    return clean


def trajectory_to_messages(
    master_history: List[Dict[str, Any]],
    tool_history: Optional[List[Dict[str, Any]]] = None,
    strip_tool_role: bool = True,
    clean_diagnosis: bool = True,
) -> List[Dict[str, str]]:
    """
    Convert a master_history to OpenAI-compatible messages format.

    Args:
        master_history: The full master dialogue from AgentExecutor
        tool_history: Optional tool execution history (for metadata)
        strip_tool_role: If True, convert "tool" role to "user" for Qwen3 compatibility
        clean_diagnosis: If True, strip diagnosis injection messages

    Returns:
        List of {"role": "...", "content": "..."} messages
    """
    if clean_diagnosis:
        master_history = strip_diagnosis_messages(master_history)

    messages = []

    for msg in master_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        content_str = _sanitize_content_for_sft(content)

        if role == "system":
            messages.append({"role": "system", "content": content_str})
        elif role == "user":
            messages.append({"role": "user", "content": content_str})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content_str})
        elif role == "tool":
            if strip_tool_role:
                messages.append({
                    "role": "user",
                    "content": f"[TOOL OBSERVATION]\n{content_str}",
                })
            else:
                messages.append({"role": "tool", "content": content_str})

    return messages


def convert_samples_to_sft(
    samples: List[Any],
    output_path: str,
    strip_tool_role: bool = True,
    only_correct: bool = True,
    clean_diagnosis: bool = True,
    format_type: str = "jsonl",
) -> str:
    """
    Convert distilled samples to SFT format and save.

    Args:
        samples: List of DistilledSample objects
        output_path: Path to save the SFT data
        strip_tool_role: Convert tool role to user for Qwen3 compatibility
        only_correct: Only include samples with correct final answers
        clean_diagnosis: Strip diagnosis injection messages for clean trajectories
        format_type: "jsonl" (one JSON per line) or "json" (single array)

    Returns:
        Path to the saved file
    """
    sft_data = []
    skipped_incorrect = 0
    skipped_empty = 0

    for sample in samples:
        if only_correct and not sample.is_correct:
            skipped_incorrect += 1
            continue

        messages = trajectory_to_messages(
            master_history=sample.master_history,
            tool_history=sample.tool_history,
            strip_tool_role=strip_tool_role,
            clean_diagnosis=clean_diagnosis,
        )

        # Skip empty trajectories
        if not messages:
            skipped_empty += 1
            continue

        record = {
            "messages": messages,
            "metadata": {
                "task": sample.task,
                "question_id": str(sample.question_id),
                "num_attempts": sample.num_attempts,
                "has_diagnosis": len(sample.diagnosis_history) > 0,
            },
        }
        sft_data.append(record)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if format_type == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for record in sft_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

    logger.info(
        "Saved %d SFT samples to %s (skipped: %d incorrect, %d empty)",
        len(sft_data), output_path, skipped_incorrect, skipped_empty,
    )
    return output_path


def save_raw_trajectories(
    samples: List[Any],
    output_dir: str,
    keep_diagnosis: bool = True,
):
    """
    Save raw trajectories for debugging and analysis.

    Args:
        samples: List of DistilledSample objects
        output_dir: Output directory
        keep_diagnosis: If True, keep diagnosis messages in raw output (for debugging)
    """
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "all_trajectories.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in samples:
            record = sample.to_dict() if hasattr(sample, "to_dict") else sample

            # Optionally strip diagnosis from raw too for clean export
            if not keep_diagnosis and "master_history" in record:
                record["master_history"] = strip_diagnosis_messages(record["master_history"])

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "total_samples": len(samples),
        "correct": sum(1 for s in samples if s.is_correct),
        "incorrect": sum(1 for s in samples if not s.is_correct),
        "with_diagnosis": sum(1 for s in samples if len(s.diagnosis_history) > 0),
        "per_task": {},
    }
    for sample in samples:
        task = sample.task
        if task not in summary["per_task"]:
            summary["per_task"][task] = {"total": 0, "correct": 0, "incorrect": 0, "with_diagnosis": 0}
        summary["per_task"][task]["total"] += 1
        if sample.is_correct:
            summary["per_task"][task]["correct"] += 1
        else:
            summary["per_task"][task]["incorrect"] += 1
        if len(sample.diagnosis_history) > 0:
            summary["per_task"][task]["with_diagnosis"] += 1

    summary_path = os.path.join(output_dir, "trajectory_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Raw trajectories saved to %s", output_dir)
    return output_dir


def save_failed_samples(
    failed: List[Dict[str, Any]],
    output_dir: str,
):
    """Save failed samples for analysis."""
    if not failed:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "failed_samples.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d failed samples to %s", len(failed), path)
