#!/usr/bin/env python3
"""Extract Type-B single-video precise VQA candidates from agent logs.

The script converts active_perception calls into JSONL records that can be
reviewed and then adapted to a VLM SFT format. It is intentionally conservative:
by default it keeps only single-video observations whose prompts look like
closed or targeted questions, while filtering out broad Stage-1 survey prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_ROOT = REPO_ROOT / "agent_system" / "logs" / "CC_task"
DEFAULT_QUESTION_FILE = REPO_ROOT / "agent_system" / "question" / "CC.json"
DEFAULT_VIDEO_ROOT = Path(os.environ.get("CROSSVID_VIDEO_ROOT", "/media/data6/xuejj/CrossVid/videos"))
DEFAULT_OUTPUT = REPO_ROOT / "data_synthesis" / "type_b_vqa_candidates.jsonl"


TARGET_ANSWER_INSTRUCTION = (
    "Rewrite the raw visual answer into 1-3 numbered answers. Use exactly this "
    "style for each item: '(1) direct answer - brief visual evidence'. If the "
    "frames do not support a confident answer, use 'UNCERTAIN - reason'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Type-B single-video VQA candidates from CC agent logs."
    )
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--question-file", type=Path, default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-stage1",
        action="store_true",
        help="Keep broad Stage-1 survey prompts too. Default keeps targeted/closed prompts only.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick inspection.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional directory where sampled JPEG frames will be materialized with ffmpeg.",
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=None,
        help="Override each logged num_frames when materializing frames.",
    )
    return parser.parse_args()


def load_questions(question_file: Path) -> Dict[int, Dict[str, Any]]:
    with question_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    questions: Dict[int, Dict[str, Any]] = {}
    for pos, row in enumerate(rows):
        try:
            qid = int(row.get("id", pos))
        except (TypeError, ValueError):
            qid = pos
        questions[qid] = row
    return questions


def task_id_from_log_dir(path: Path) -> Optional[int]:
    match = re.match(r"(\d+)", path.name)
    return int(match.group(1)) if match else None


def prompt_kind(prompt: str) -> str:
    lower = prompt.lower()
    if "answer only" in lower:
        return "answer_only"
    if "look again" in lower or "answer:" in lower:
        return "targeted_answer"
    if "?" in prompt:
        return "closed_question"
    return "other"


def is_stage1_survey(prompt: str) -> bool:
    lower = prompt.lower()
    survey_markers = [
        "describe only what is relevant",
        "look at this single video. the question is:",
        "look at this video. the question is:",
    ]
    return any(marker in lower for marker in survey_markers)


def is_type_b_candidate(prompt: str, include_stage1: bool) -> bool:
    if not prompt.strip():
        return False
    if not include_stage1 and is_stage1_survey(prompt):
        return False

    kind = prompt_kind(prompt)
    if kind in {"answer_only", "targeted_answer"}:
        return True

    lower = prompt.lower()
    closed_markers = [
        " is ",
        " are ",
        " which ",
        " whether ",
        " visible",
        "exact state",
        "how many",
        "count",
    ]
    return "?" in prompt and any(marker in lower for marker in closed_markers)


def clean_prompt(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


def concise_target_answer(raw_answer: str) -> Tuple[str, bool]:
    """Make a light, review-friendly target answer without inventing content."""
    text = raw_answer.strip()
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    needs_review = False
    if not text:
        return "(1) UNCERTAIN - Empty visual answer.", True

    if re.match(r"^\(\d+\)\s+", text):
        return text, False

    sentence_parts = re.split(r"(?<=[.!?])\s+", text)
    short_text = " ".join(sentence_parts[:5]).strip()
    if len(sentence_parts) > 5 or len(short_text) > 900:
        short_text = short_text[:900].rsplit(" ", 1)[0].rstrip(".") + "."
        needs_review = True

    needs_review = True
    return f"(1) {short_text}", needs_review


def question_video_info(
    question: Optional[Dict[str, Any]],
    video_index: int,
    video_root: Path,
) -> Dict[str, Any]:
    if not question:
        return {}

    videos = question.get("videos") or []
    begins = question.get("begin") or [None] * len(videos)
    ends = question.get("end") or [None] * len(videos)
    durations = question.get("duration") or [None] * len(videos)

    zero_idx = video_index - 1
    if zero_idx < 0 or zero_idx >= len(videos):
        return {}

    video_name = videos[zero_idx]
    clip_begin = begins[zero_idx] if zero_idx < len(begins) and begins[zero_idx] is not None else 0.0
    clip_end = ends[zero_idx] if zero_idx < len(ends) and ends[zero_idx] is not None else None
    duration = durations[zero_idx] if zero_idx < len(durations) else None
    if clip_end is None and duration is not None:
        clip_end = float(clip_begin) + float(duration)

    return {
        "video_name": video_name,
        "video_path": str(video_root / video_name),
        "clip_begin": float(clip_begin),
        "clip_end": float(clip_end) if clip_end is not None else None,
        "duration_seconds": float(duration) if duration is not None else None,
    }


def sample_frame_times(start: float, end: float, count: int) -> List[float]:
    if count <= 1 or end <= start:
        return [start]
    step = (end - start) / (count - 1)
    return [start + step * i for i in range(count)]


def materialize_frames(
    video_path: Path,
    start: float,
    end: float,
    count: int,
    sample_dir: Path,
) -> List[str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[str] = []
    for idx, timestamp in enumerate(sample_frame_times(start, end, count), start=1):
        out_path = sample_dir / f"frame_{idx:03d}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        frame_paths.append(str(out_path))
    return frame_paths


def iter_tool_entries(logs_root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for log_dir in sorted(logs_root.iterdir(), key=lambda p: p.name):
        if not log_dir.is_dir():
            continue
        tool_path = log_dir / "tools_execution.json"
        if not tool_path.exists():
            continue
        try:
            with tool_path.open("r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as exc:
            print(f"[WARN] Skip unreadable {tool_path}: {exc}")
            continue
        for entry in entries:
            yield log_dir, entry


def build_record(
    log_dir: Path,
    entry: Dict[str, Any],
    questions: Dict[int, Dict[str, Any]],
    video_root: Path,
    frames_dir: Optional[Path],
    frames_per_sample: Optional[int],
) -> Optional[Dict[str, Any]]:
    task_id = task_id_from_log_dir(log_dir)
    inputs = entry.get("inputs") or {}
    targets = inputs.get("observation_targets") or []
    if len(targets) != 1:
        return None

    target = targets[0]
    try:
        video_index = int(target["video_index"])
    except (KeyError, TypeError, ValueError):
        return None

    question = questions.get(task_id) if task_id is not None else None
    video_info = question_video_info(question, video_index, video_root)

    start_time = float(target.get("start_time", 0.0))
    end_time = float(target.get("end_time", video_info.get("duration_seconds") or start_time))
    num_frames = int(frames_per_sample or target.get("num_frames", 16))
    clip_begin = float(video_info.get("clip_begin", 0.0))
    absolute_start = clip_begin + start_time
    absolute_end = clip_begin + end_time

    raw_answer = entry.get("output_raw") or ""
    target_answer, needs_review = concise_target_answer(raw_answer)
    prompt = clean_prompt(inputs.get("focus_prompt") or "")

    sample_id = f"cc_{task_id}_{entry.get('call_id', 'tool')}" if task_id is not None else str(entry.get("call_id", "tool"))
    frame_paths: List[str] = []
    frame_error = None
    if frames_dir is not None:
        try:
            frame_paths = materialize_frames(
                Path(video_info["video_path"]),
                absolute_start,
                absolute_end,
                num_frames,
                frames_dir / sample_id,
            )
        except Exception as exc:
            frame_error = str(exc)

    user_content: List[Dict[str, str]] = []
    for frame_path in frame_paths:
        user_content.append({"type": "image", "image": frame_path})
    user_content.append({"type": "text", "text": prompt})

    record = {
        "id": sample_id,
        "task_id": task_id,
        "source_log_dir": str(log_dir),
        "source_call_id": entry.get("call_id"),
        "source_task_question": question.get("question") if question else None,
        "source_options": question.get("options") if question else None,
        "video_index": video_index,
        **video_info,
        "segment_start": start_time,
        "segment_end": end_time,
        "absolute_start": absolute_start,
        "absolute_end": absolute_end,
        "num_frames": num_frames,
        "focus_prompt": prompt,
        "raw_answer": raw_answer,
        "target_answer": target_answer,
        "target_answer_instruction": TARGET_ANSWER_INSTRUCTION,
        "messages": [
            {"role": "user", "content": user_content if frame_paths else prompt},
            {"role": "assistant", "content": target_answer},
        ],
        "quality_flags": {
            "single_video": True,
            "prompt_kind": prompt_kind(prompt),
            "stage1_survey": is_stage1_survey(prompt),
            "answer_needs_review": needs_review,
            "frame_materialization_error": frame_error,
        },
    }
    if frame_paths:
        record["frame_paths"] = frame_paths
    return record


def main() -> None:
    args = parse_args()
    questions = load_questions(args.question_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0
    with args.output.open("w", encoding="utf-8") as out:
        for log_dir, entry in iter_tool_entries(args.logs_root):
            seen += 1
            inputs = entry.get("inputs") or {}
            prompt = inputs.get("focus_prompt") or ""
            targets = inputs.get("observation_targets") or []
            if len(targets) != 1:
                continue
            if not is_type_b_candidate(prompt, args.include_stage1):
                continue

            record = build_record(
                log_dir=log_dir,
                entry=entry,
                questions=questions,
                video_root=args.video_root,
                frames_dir=args.frames_dir,
                frames_per_sample=args.frames_per_sample,
            )
            if record is None:
                continue

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_samples is not None and kept >= args.max_samples:
                break

    print(f"Scanned tool entries: {seen}")
    print(f"Wrote Type-B candidates: {kept}")
    print(f"Output: {args.output}")
    if args.frames_dir:
        print(f"Frames directory: {args.frames_dir}")


if __name__ == "__main__":
    main()
