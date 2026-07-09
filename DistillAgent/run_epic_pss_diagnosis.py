#!/usr/bin/env python3
"""
EPIC PSS diagnosis runner.

Runs EPIC-Kitchens-derived PSS samples with the same agent stack used for
CrossVid, then applies up to N DiagnosisAgent corrections to wrong answers.

Example:
  cd /media/data6/xuejj
  MASTER_BACKEND=api TOOL_BACKEND=api \
  python3 CVRAgent/DistillAgent/run_epic_pss_diagnosis.py \
    --prompt-dir myprompts --start 1 --end 20 --max-turns 20 --max-diagnosis 3
"""

import argparse
import json
import logging
import os
import sys
import time
import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AGENT_ROOT = PROJECT_ROOT / "agent_system"
REPO_ROOT = PROJECT_ROOT.parent

for p in (str(AGENT_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DOTENV_PATH = AGENT_ROOT / ".env"
if DOTENV_PATH.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(DOTENV_PATH)
    except ImportError:
        pass

from agent_executor import AgentExecutor
from diagnosis_agent import DiagnosisAgent, build_correction_injection
from qwen_agent import QwenModel
from runtime_options import apply_runtime_overrides, runtime_summary
from task_config import check_correct, get_ground_truth_text
from trajectory_converter import convert_samples_to_sft, save_failed_samples, save_raw_trajectories
from utils.log_utils import RedactingFormatter


DEFAULT_QUESTION_FILE = REPO_ROOT / "Epic_Kitchen" / "PSS_train.json"
DEFAULT_AUDIT_FILE = REPO_ROOT / "Epic_Kitchen" / "PSS_train_audit.jsonl"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "Epic_Kitchen" / "new_frames"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_epic_pss_diagnosis"
DEFAULT_PROMPT_CONFIG = {"master": str(AGENT_ROOT / "prompts" / "master_PSS.prompt")}
DEFAULT_MASTER_BASE = "_unified_crossvid_base_compress.prompt"
FPS = 30

QUERY = (
    "Review these video clips. They are shuffled segments from a single cooking video. "
    "Please determine their correct chronological order."
)

logger = logging.getLogger("EPIC_PSS_Diagnosis")


@dataclass
class EpicDistilledSample:
    task: str
    question_id: Any
    question_data: Dict[str, Any]
    master_history: List[Dict[str, Any]]
    tool_history: List[Dict[str, Any]]
    final_answer: Any
    ground_truth: Any
    is_correct: bool
    num_attempts: int
    diagnosis_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "question_id": self.question_id,
            "question_data": self.question_data,
            "master_history": self.master_history,
            "tool_history": self.tool_history,
            "final_answer": str(self.final_answer),
            "ground_truth": str(self.ground_truth),
            "is_correct": self.is_correct,
            "num_attempts": self.num_attempts,
            "diagnosis_history": self.diagnosis_history,
            "metadata": self.metadata,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EPIC PSS with API agent + up to N diagnosis corrections."
    )
    parser.add_argument("--question-file", default=str(DEFAULT_QUESTION_FILE))
    parser.add_argument("--audit-file", default=str(DEFAULT_AUDIT_FILE))
    parser.add_argument("--frames-root", default=str(DEFAULT_FRAMES_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", type=int, default=1, help="1-based start index")
    parser.add_argument("--end", type=int, default=20, help="1-based inclusive end index")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-diagnosis", type=int, default=3)
    parser.add_argument("--prompt-dir", default="myprompts")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument(
        "--master-base",
        default=DEFAULT_MASTER_BASE,
        help="Unified base prompt under --prompt-dir. Defaults to the CrossVid DistillAgent compress base.",
    )
    parser.add_argument("--keep-incorrect", action="store_true")

    parser.add_argument("--master-backend", choices=["qwen3", "local_hf", "local", "hf", "api"])
    parser.add_argument("--master-model-path")
    parser.add_argument("--master-api-base-url")
    parser.add_argument("--master-api-key")
    parser.add_argument("--master-model-name")
    parser.add_argument("--master-max-tokens", type=int)
    parser.add_argument("--local-master-max-new-tokens", type=int)
    parser.add_argument("--local-master-device")
    parser.add_argument("--local-master-device-map")
    parser.add_argument("--local-master-dtype")
    parser.add_argument("--local-master-enable-thinking", action="store_true")
    parser.add_argument("--oracle-answer-hint", action="store_true")

    parser.add_argument("--tool-backend", choices=["qwenvl", "local_qwenvl", "local_vlm", "api"])
    parser.add_argument("--tool-model-path")
    parser.add_argument("--tool-api-base-url")
    parser.add_argument("--tool-api-key")
    parser.add_argument("--tool-model-name")
    parser.add_argument("--tool-max-tokens", type=int)
    parser.add_argument("--local-tool-max-new-tokens", type=int)
    parser.add_argument("--local-tool-device")
    parser.add_argument("--local-tool-device-map")
    parser.add_argument("--local-tool-dtype")
    return parser.parse_args()


def setup_logging(output_dir: str, start: int, end: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(output_dir) / f"log_EPIC_PSS_diagnosis_main_q{start}_to_{end}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(RedactingFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(RedactingFormatter("%(asctime)s - %(levelname)s - %(message)s"))

    root.addHandler(console)
    root.addHandler(file_handler)
    logger.info("Log file: %s", log_path)


def build_prompt_config(args: argparse.Namespace) -> Dict[str, str]:
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = AGENT_ROOT / prompt_path
        return {"master": str(prompt_path)}

    prompt_dir = Path(args.prompt_dir)
    if not prompt_dir.is_absolute():
        prompt_dir = AGENT_ROOT / prompt_dir

    master_base = prompt_dir / args.master_base
    task_contract = prompt_dir / "task_contracts" / "PSS.prompt"
    if master_base.exists() and task_contract.exists():
        return {
            "master_base": str(master_base),
            "task_contract": str(task_contract),
            "task_code": "PSS",
            "master_name": "master_PSS.prompt",
        }

    return dict(DEFAULT_PROMPT_CONFIG)


def load_audit(audit_file: str) -> Dict[str, Dict[str, Any]]:
    path = Path(audit_file)
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[str(row.get("id"))] = row
    return out


def _segment_intervals(seg: Any) -> List[List[float]]:
    if isinstance(seg, dict):
        intervals = seg.get("interval", [])
    else:
        intervals = seg
    if not isinstance(intervals, list):
        return []
    clean = []
    for item in intervals:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                start, end = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                continue
            if start < end:
                clean.append([start, end])
    return clean


def build_epic_video_contexts(question_data: Dict[str, Any], frames_root: str) -> List[Dict[str, Any]]:
    video_id = question_data.get("video")
    segments = question_data.get("segments")
    if not video_id or not isinstance(segments, dict):
        raise ValueError("EPIC PSS question missing video or segments")

    frames_dir = Path(frames_root) / str(video_id)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    all_frames = sorted(
        p.name for p in frames_dir.iterdir()
        if p.name.startswith("frame_") and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not all_frames:
        raise FileNotFoundError(f"No frame images found in {frames_dir}")

    contexts = []
    seg_keys = sorted(segments.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    for key in seg_keys:
        intervals = _segment_intervals(segments[key])
        if not intervals:
            continue
        seg_start = intervals[0][0]
        seg_end = intervals[-1][1]
        seg_dur = max(0.0, seg_end - seg_start)

        start_idx = max(0, min(int(seg_start * FPS), len(all_frames) - 1))
        end_idx = max(0, min(int(seg_end * FPS), len(all_frames) - 1))
        if end_idx < start_idx:
            continue
        seg_frames = all_frames[start_idx:end_idx + 1]
        if not seg_frames:
            continue

        contexts.append({
            "path": str(frames_dir),
            # Keep this non-image_folder on purpose: the master prompt and
            # AgentExecutor should expose this exactly like CrossVid clips
            # (duration_seconds + start_time/end_time). The tool below maps
            # seconds to extracted EPIC frame files internally.
            "type": "epic_frame_segment",
            "fps": FPS,
            "total_frames": len(seg_frames),
            "image_files": seg_frames,
            "duration_seconds": seg_dur,
            "clip_begin": 0.0,
            "clip_end": seg_dur,
            "name_for_log": f"{video_id}_seg_{key}",
            "segment_key": str(key),
            "source_interval": intervals,
        })

    if not contexts:
        raise ValueError("No valid EPIC segment contexts built")
    return contexts


def patch_epic_active_perception(executor: AgentExecutor):
    original_tool = executor.tool_mapping.get("active_perception")

    def epic_active_perception(observation_targets: List[Dict[str, Any]], focus_prompt: str) -> Dict[str, Any]:
        if not observation_targets:
            return {"result": "Error: No observation targets provided."}

        observation_targets, focus_prompt_local = executor._normalize_pss_frame_counts(
            observation_targets, focus_prompt
        )
        focus_prompt_local = executor._augment_focus_prompt_with_cross_context(focus_prompt_local)

        all_frames: List[str] = []
        mapping_texts: List[str] = []
        current_frame_index = 0
        skim_used = False

        try:
            for target in observation_targets:
                v_idx = target.get("video_index")
                if v_idx is None:
                    return {"result": "Error: Missing video_index."}
                real_idx = int(v_idx) - 1
                if real_idx < 0 or real_idx >= len(executor.video_contexts):
                    return {"result": f"Error: Video index {v_idx} out of range."}

                context = executor.video_contexts[real_idx]
                if context.get("type") != "epic_frame_segment":
                    if original_tool is None:
                        return {"result": "Error: original active_perception tool missing."}
                    return original_tool(observation_targets, focus_prompt_local)

                image_files = context.get("image_files", [])
                folder_path = context.get("path")
                duration = float(context.get("duration_seconds", 0.0) or 0.0)
                total_frames = len(image_files)
                if total_frames <= 0:
                    continue

                start_t = float(target.get("start_time", 0.0) or 0.0)
                end_t = target.get("end_time", duration)
                end_t = duration if end_t is None else float(end_t)
                start_t = max(0.0, min(start_t, duration))
                end_t = max(start_t, min(end_t, duration))

                num_f = int(target.get("num_frames", 32) or 32)
                num_f = max(1, min(num_f, total_frames))

                if duration > 0 and end_t > start_t:
                    # Match utils.video_processor.process_video(): sample at
                    # the midpoint of each temporal bin, not at interval ends.
                    window = end_t - start_t
                    sample_times = (
                        np.linspace(0, window, num=num_f, endpoint=False)
                        + (window / (2 * num_f))
                        + start_t
                    )
                    effective_fps = max(1e-6, (total_frames - 1) / duration)
                    indices = [
                        min(int(t * effective_fps), total_frames - 1)
                        for t in sample_times
                    ]
                else:
                    indices = [0]
                indices = sorted(set(int(i) for i in indices))

                use_skim = executor._layer1_skim_enabled(focus_prompt_local)
                raw_frames = []
                raw_mapping = {}
                extracted = 0
                for idx in indices:
                    path = os.path.join(folder_path, image_files[idx])
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    max_len = 360
                    if max(h, w) > max_len:
                        scale = max_len / max(h, w)
                        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    if use_skim:
                        raw_mapping[len(raw_frames)] = idx
                        raw_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        continue
                    ok, buffer = cv2.imencode(".jpg", img)
                    if not ok:
                        continue
                    all_frames.append(base64.b64encode(buffer).decode("utf-8"))
                    extracted += 1

                selected_positions = None
                selected_source_frames = None
                selected_time_text = ""
                if use_skim and raw_frames:
                    selected, skim_strategy = executor._select_layer1_skim_frames(raw_frames, raw_mapping)
                    frames = [encoded for _, _, encoded in selected]
                    if not frames:
                        continue
                    all_frames.extend(frames)
                    skim_used = True
                    extracted = len(frames)
                    selected_positions = [idx + 1 for idx, _, _ in selected]
                    selected_source_frames = [source_idx for _, source_idx, _ in selected]
                    if duration > 0 and total_frames > 1:
                        selected_times = [
                            (source_idx / max(1e-6, (total_frames - 1) / duration))
                            for source_idx in selected_source_frames
                        ]
                        selected_time_text = ", local_times=" + ",".join(f"{t:.1f}s" for t in selected_times)
                    logger.info(
                        "    [LAYER1-SKIM] Video %d %.1f-%.1fs: selected %d/%d representative frames (%s), positions=%s",
                        v_idx,
                        start_t,
                        end_t,
                        extracted,
                        len(raw_frames),
                        skim_strategy,
                        selected_positions,
                    )

                if extracted:
                    s_idx = current_frame_index + 1
                    e_idx = current_frame_index + extracted
                    if use_skim:
                        mapping_texts.append(
                            f"Frames {s_idx}-{e_idx} are Layer1 skim representatives from Video {v_idx} "
                            f"({start_t:.1f}s to {end_t:.1f}s); original sampled positions={selected_positions}, "
                            f"source_frame_ids={selected_source_frames}{selected_time_text}"
                        )
                    else:
                        mapping_texts.append(
                            f"Frames {s_idx}-{e_idx} are from Video {v_idx} ({start_t:.1f}s to {end_t:.1f}s)"
                        )
                    current_frame_index += extracted

            if skim_used:
                focus_prompt_local = (
                    f"{focus_prompt_local}\n\n[Layer1 Skim Notice]\n"
                    "The frames are representative skim frames selected inside each chunk (centroid/anomaly style), "
                    "not dense temporal coverage. Use them only for coarse scene/plot relevance, candidate chunk selection, "
                    "and obvious visible clues. Mark details as UNCERTAIN rather than absent when not visible in the skim frames. "
                    "Return compact rows per mapped chunk and suggest which chunk windows need Layer2 detail."
                )
            return executor._call_vlm_for_perception(all_frames, mapping_texts, focus_prompt_local)
        except Exception as exc:
            logger.error("EPIC active perception failed: %s", exc, exc_info=True)
            return {"result": f"Error during EPIC frame observation: {exc}"}

    executor.tool_mapping["active_perception"] = epic_active_perception
    return original_tool


def run_attempt(
    question_data: Dict[str, Any],
    agent: QwenModel,
    prompt_config: Dict[str, str],
    frames_root: str,
    max_turns: int,
    log_dir: str,
    resume_history: Optional[List[Dict[str, Any]]] = None,
    previous_tool_history: Optional[List[Dict[str, Any]]] = None,
    keep_tool_turns_before: Optional[int] = None,
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    contexts = build_epic_video_contexts(question_data, frames_root)
    executor = AgentExecutor(agent_instance=agent, prompt_config=prompt_config)
    original_tool = patch_epic_active_perception(executor)

    try:
        executor.init_agent_state(query=QUERY, possible_answers=[], video_contexts=contexts)
        if resume_history is not None:
            executor._resume_history = list(resume_history)
            if previous_tool_history and keep_tool_turns_before is not None:
                executor._tool_history = [
                    entry for entry in previous_tool_history
                    if int(entry.get("turn", 0) or 0) < keep_tool_turns_before
                ]

        final_answer, master_history = executor.run_agent_loop(max_turns=max_turns)
        tool_history = list(executor._tool_history)
        return final_answer, master_history, tool_history, contexts
    finally:
        if original_tool is not None:
            executor.tool_mapping["active_perception"] = original_tool
        try:
            os.makedirs(log_dir, exist_ok=True)
            executor.save_logs(log_dir)
        finally:
            executor.cleanup_agent_state()


def make_resume_history(
    previous_history: List[Dict[str, Any]],
    correction_message: str,
    error_turn: int,
) -> Tuple[List[Dict[str, Any]], int]:
    error_turn = max(1, int(error_turn or 1))
    error_msg_idx = 2 + (error_turn - 1) * 2
    if error_msg_idx >= len(previous_history):
        error_msg_idx = max(2, len(previous_history) - 1)

    safe_history = list(previous_history[:error_msg_idx])
    safe_history.append({"role": "user", "content": correction_message})
    return safe_history, error_turn


def run_one_question(
    question_data: Dict[str, Any],
    audit_row: Dict[str, Any],
    agent: QwenModel,
    diagnosis_agent: DiagnosisAgent,
    prompt_config: Dict[str, str],
    frames_root: str,
    output_dir: str,
    max_turns: int,
    max_diagnosis: int,
) -> Tuple[EpicDistilledSample, Optional[Dict[str, Any]]]:
    qid = question_data.get("id", "?")
    ground_truth = question_data.get("answer")
    ground_truth_text = get_ground_truth_text("PSS", question_data)

    diagnosis_history = []
    q_dir = Path(output_dir) / "detailed_logs" / "EPIC_PSS" / str(qid)

    logger.info("Query: %s", QUERY)
    logger.info("Correct Answer: [%s]", ground_truth_text)
    t0 = time.time()
    final_answer, master_history, tool_history, _ = run_attempt(
        question_data, agent, prompt_config, frames_root, max_turns,
        str(q_dir / "attempt_1"),
    )
    is_correct = check_correct("PSS", final_answer, ground_truth)
    logger.info("Evaluating result...")
    logger.info("  - Predicted: %s", final_answer)
    logger.info("  - Ground Truth: %s", ground_truth)
    logger.info("  - Result: %s", "CORRECT" if is_correct else "WRONG")
    logger.info("Attempt 1 elapsed: %.1fs", time.time() - t0)

    if is_correct:
        logger.info("  - Final Result: CORRECT")
        return EpicDistilledSample(
            task="EPIC_PSS", question_id=qid, question_data=question_data,
            master_history=master_history, tool_history=tool_history,
            final_answer=final_answer, ground_truth=ground_truth, is_correct=True,
            num_attempts=1, diagnosis_history=[],
            metadata={"segments": len(question_data.get("segments", {})), **audit_row},
        ), None

    current_answer = final_answer
    current_history = master_history
    current_tool_history = tool_history

    for diag_idx in range(1, max_diagnosis + 1):
        logger.info("Diagnosis attempt %d/%d", diag_idx, max_diagnosis)
        diagnosis = diagnosis_agent.diagnose(
            task="PSS",
            question_data=question_data,
            ground_truth_text=ground_truth_text,
            master_dialogue=current_history,
            tool_history=current_tool_history,
        )
        diagnosis_history.append({
            "attempt": diag_idx,
            "error_turn": diagnosis.error_turn,
            "error_type": diagnosis.error_type,
            "correction_scope": diagnosis.correction_scope,
            "error_description": diagnosis.error_description,
            "correction_message": diagnosis.correction_message,
            "confidence": diagnosis.confidence,
        })

        scope = diagnosis.correction_scope or "full_restart"
        effective_error_turn = 1 if scope == "full_restart" else max(1, diagnosis.error_turn or 1)
        correction_message = build_correction_injection(diagnosis, diag_idx, max_diagnosis)
        resume_history, keep_tool_turns_before = make_resume_history(
            current_history, correction_message, effective_error_turn
        )

        t0 = time.time()
        current_answer, current_history, current_tool_history, _ = run_attempt(
            question_data, agent, prompt_config, frames_root,
            max(1, max_turns - effective_error_turn + 1),
            str(q_dir / f"attempt_{diag_idx + 1}"),
            resume_history=resume_history,
            previous_tool_history=current_tool_history,
            keep_tool_turns_before=keep_tool_turns_before,
        )
        is_correct = check_correct("PSS", current_answer, ground_truth)
        logger.info("Evaluating diagnosis attempt %d result...", diag_idx)
        logger.info("  - Predicted: %s", current_answer)
        logger.info("  - Ground Truth: %s", ground_truth)
        logger.info("  - Result: %s", "CORRECTED" if is_correct else "STILL WRONG")
        logger.info(
            "Attempt %d elapsed: %.1fs",
            diag_idx + 1, time.time() - t0
        )
        if is_correct:
            logger.info("  - Final Result: CORRECTED_BY_DIAGNOSIS_%d", diag_idx)
            return EpicDistilledSample(
                task="EPIC_PSS", question_id=qid, question_data=question_data,
                master_history=current_history, tool_history=current_tool_history,
                final_answer=current_answer, ground_truth=ground_truth, is_correct=True,
                num_attempts=diag_idx + 1, diagnosis_history=diagnosis_history,
                metadata={"segments": len(question_data.get("segments", {})), **audit_row},
            ), None

    logger.info("  - Final Result: FAILED_AFTER_%d_DIAGNOSIS", max_diagnosis)
    failed = {
        "task": "EPIC_PSS",
        "question_id": qid,
        "question_data": question_data,
        "ground_truth": ground_truth_text,
        "final_answer": str(current_answer),
        "diagnosis_history": diagnosis_history,
        "metadata": {"segments": len(question_data.get("segments", {})), **audit_row},
    }
    return EpicDistilledSample(
        task="EPIC_PSS", question_id=qid, question_data=question_data,
        master_history=current_history, tool_history=current_tool_history,
        final_answer=current_answer, ground_truth=ground_truth, is_correct=False,
        num_attempts=max_diagnosis + 1, diagnosis_history=diagnosis_history,
        metadata={"segments": len(question_data.get("segments", {})), **audit_row},
    ), failed


def summarize(samples: List[EpicDistilledSample], failed: List[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    total = len(samples)
    correct_first = sum(1 for s in samples if s.is_correct and s.num_attempts == 1)
    corrected = sum(1 for s in samples if s.is_correct and s.num_attempts > 1)
    incorrect = sum(1 for s in samples if not s.is_correct)
    by_seg: Dict[str, Dict[str, int]] = {}
    by_source: Dict[str, Dict[str, int]] = {}

    for sample in samples:
        seg_key = str(sample.metadata.get("segments", len(sample.question_data.get("segments", {}))))
        src_key = str(sample.metadata.get("source", "unknown"))
        for table, key in ((by_seg, seg_key), (by_source, src_key)):
            table.setdefault(key, {"total": 0, "first_try": 0, "corrected": 0, "failed": 0})
            table[key]["total"] += 1
            if sample.is_correct and sample.num_attempts == 1:
                table[key]["first_try"] += 1
            elif sample.is_correct:
                table[key]["corrected"] += 1
            else:
                table[key]["failed"] += 1

    return {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "correct_first_try": correct_first,
        "corrected_by_diagnosis": corrected,
        "failed_after_max_diagnosis": incorrect,
        "failed_samples": len(failed),
        "first_try_accuracy": correct_first / max(total, 1),
        "success_after_diagnosis": (correct_first + corrected) / max(total, 1),
        "failed_rate_after_diagnosis": incorrect / max(total, 1),
        "elapsed_seconds": elapsed,
        "by_segment_count": by_seg,
        "by_source": by_source,
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir, args.start, args.end)
    apply_runtime_overrides(args)

    prompt_config = build_prompt_config(args)
    logger.info("Runtime: %s", runtime_summary(prompt_config))
    logger.info("Questions: %s | frames: %s", args.question_file, args.frames_root)

    with open(args.question_file, "r", encoding="utf-8") as f:
        questions_all = json.load(f)

    start_idx = max(0, args.start - 1)
    end_idx = min(len(questions_all), args.end if args.end else len(questions_all))
    questions = questions_all[start_idx:end_idx]
    audit = load_audit(args.audit_file)

    logger.info("Running EPIC_PSS Q%d-Q%d (%d samples)", start_idx + 1, end_idx, len(questions))

    agent = QwenModel(prompt_config=prompt_config)
    diagnosis_agent = DiagnosisAgent()

    samples: List[EpicDistilledSample] = []
    failed: List[Dict[str, Any]] = []
    t_start = time.time()

    for offset, question in enumerate(questions):
        abs_num = start_idx + offset + 1
        qid = question.get("id", abs_num)
        logger.info("=" * 70)
        logger.info("  Testing Question %d/%d (ID: %s)", abs_num, len(questions_all), qid)
        logger.info("  Task: EPIC_PSS | segments=%d", len(question.get("segments", {})))
        try:
            sample, failed_record = run_one_question(
                question_data=question,
                audit_row=audit.get(str(qid), {}),
                agent=agent,
                diagnosis_agent=diagnosis_agent,
                prompt_config=prompt_config,
                frames_root=args.frames_root,
                output_dir=args.output_dir,
                max_turns=args.max_turns,
                max_diagnosis=args.max_diagnosis,
            )
            samples.append(sample)
            if failed_record:
                failed.append(failed_record)
            logger.info("--- End of Q%s ---", qid)
        except Exception as exc:
            logger.error("Error on EPIC_PSS id=%s: %s", qid, exc, exc_info=True)
            failed.append({
                "task": "EPIC_PSS",
                "question_id": qid,
                "error": str(exc),
                "question_data": question,
            })
            logger.info("  - Final Result: SKIPPED_ERROR")
            logger.info("--- End of Q%s ---", qid)

    elapsed = time.time() - t_start

    raw_dir = Path(args.output_dir) / "raw_trajectories"
    failed_dir = Path(args.output_dir) / "failed_samples"
    save_raw_trajectories(samples, str(raw_dir))
    save_failed_samples(failed, str(failed_dir))
    convert_samples_to_sft(
        samples,
        str(Path(args.output_dir) / "sft_data.jsonl"),
        only_correct=not args.keep_incorrect,
        format_type="jsonl",
    )

    report = summarize(samples, failed, elapsed)
    report["settings"] = {
        "question_file": args.question_file,
        "audit_file": args.audit_file,
        "frames_root": args.frames_root,
        "start": args.start,
        "end": args.end,
        "max_turns": args.max_turns,
        "max_diagnosis": args.max_diagnosis,
        "prompt_dir": args.prompt_dir,
    }
    report_path = Path(args.output_dir) / "epic_pss_diagnosis_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("*" * 30)
    logger.info("   Final Summary   ")
    logger.info("*" * 30)
    logger.info("Questions Run:    %d", report["total_questions"])
    logger.info("Correct First Try: %d", report["correct_first_try"])
    logger.info("Corrected by Diagnosis: %d", report["corrected_by_diagnosis"])
    logger.info("Failed after %d Diagnosis: %d", args.max_diagnosis, report["failed_after_max_diagnosis"])
    logger.info("First Try Accuracy: %.2f%%", report["first_try_accuracy"] * 100)
    logger.info("Success after Diagnosis: %.2f%%", report["success_after_diagnosis"] * 100)
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
