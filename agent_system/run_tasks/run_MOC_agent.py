# MOC task entry. Run from agent_system: python run_tasks/run_MOC_agent.py

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qwen_agent import QwenModel
from runtime_options import apply_runtime_overrides, parse_runtime_args, resolve_prompt_config, runtime_summary

QUESTION_FILE = os.path.join(_ROOT, "question", "MOC.json")
MAX_TURNS = 20
START_FROM_QUESTION_NUM = 1
END_AT_QUESTION_NUM = None
TASK_NAME = "MOC_task"

# Checkpoint file for resume support
CHECKPOINT_FILE = os.path.join(_ROOT, "logs", "MOC_checkpoint.txt")

DEFAULT_PROMPT_CONFIG = {"master": os.path.join(_ROOT, "prompts", "master_MOC.prompt")}


def _load_moc_checkpoint():
    """Return (correct_count, last_completed_abs_num) or (0, 0) if no checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                parts = f.read().strip().split(',')
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
    return 0, 0


def _save_moc_checkpoint(correct_count, last_abs_num):
    """Save progress so we can resume after interruption."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(f"{correct_count},{last_abs_num}")


if __name__ == "__main__":
    runtime_args = parse_runtime_args("Run MOC agent with selectable master/tool backends.")
    apply_runtime_overrides(runtime_args)
    PROMPT_CONFIG = resolve_prompt_config(_ROOT, "MOC", DEFAULT_PROMPT_CONFIG, runtime_args)
    max_turns = runtime_args.max_turns or MAX_TURNS

    # Reset checkpoint if explicitly requested
    if os.environ.get("MOC_RESET_CHECKPOINT", "").strip().lower() in {"1", "true", "yes", "on"}:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("🧹 MOC checkpoint reset.")

    # ── Preload Master on its GPU BEFORE importing torch-heavy modules ──
    qwen_agent_instance = QwenModel(prompt_config=PROMPT_CONFIG)
    if qwen_agent_instance.master_backend in {"local_hf", "local", "hf"}:
        print("Preloading local MASTER before other task imports...", flush=True)
        qwen_agent_instance._load_local_master()

    import json
    import re
    import logging
    from datetime import datetime
    import traceback
    from typing import List, Dict, Any, Optional
    from agent_executor import AgentExecutor
    from utils.config import get_uav_data_dir
    from utils.log_utils import setup_logger
    from utils.answer_parse import parse_agent_output_json_list

    base_data_dir = get_uav_data_dir()
    frames_dir = os.path.join(base_data_dir, "frames")
    jsons_dir = os.path.join(base_data_dir, "bbox")

    def extract_objects(s):
        return re.findall(r'\{(.*?)\}', s)

    def resolve_placeholder_object(placeholder, objects):
        """Resolve {A1}/{B4} style placeholders to the indexed object entry."""
        match = re.fullmatch(r'[AB](\d+)', placeholder)
        if not match:
            return None
        obj_index = int(match.group(1)) - 1
        if obj_index < 0 or obj_index >= len(objects):
            return None
        return objects[obj_index]

    def prepare_metadata(class_id, target_object_ids, view):
        """Load frames and bbox data for a given vid and view."""
        frame_folder_path = f"{frames_dir}/{view}/{class_id}-{view}"
        if not os.path.exists(frame_folder_path):
            return None, 0, {}, []
        try:
            image_files = sorted([
                f for f in os.listdir(frame_folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        except Exception:
            return None, 0, {}, []
        total_frames = len(image_files)
        if total_frames == 0:
            return None, 0, {}, []

        bbox_lookup = {}
        try:
            with open(f"{jsons_dir}/{view}/{class_id}.json", "r") as f:
                all_bbox_data = json.load(f)
        except Exception as e:
            print(f"Error loading bbox JSON for vid={class_id} view={view}: {e}")
            return frame_folder_path, total_frames, bbox_lookup, image_files

        for entity in all_bbox_data:
            eid = entity["id"]
            if eid not in target_object_ids:
                continue
            tag = f"obj_{eid}"
            bbox_container = entity.get("bbox")
            if not bbox_container:
                continue
            if isinstance(bbox_container, dict):
                for f_key, box_item in bbox_container.items():
                    try:
                        f_idx = int(f_key)
                        if isinstance(box_item, dict) and all(k in box_item for k in ['xtl', 'ytl', 'xbr', 'ybr']):
                            if box_item['xtl'] is not None:
                                standard_box = [int(box_item['xtl']), int(box_item['ytl']),
                                                int(box_item['xbr']), int(box_item['ybr'])]
                                bbox_lookup.setdefault(f_idx, {})[tag] = standard_box
                    except Exception:
                        pass

        return frame_folder_path, total_frames, bbox_lookup, image_files

    def is_truthy_env(name):
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

    def get_option_text(options, answer_letter):
        prefix = f"{str(answer_letter).strip().upper()}."
        for option in options:
            if str(option).strip().upper().startswith(prefix):
                return option
        return str(answer_letter)

    if not os.path.exists(QUESTION_FILE):
        print(f"FATAL: Question file not found: {QUESTION_FILE}")
        raise SystemExit(1)
    with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)

    original_total_questions = len(all_questions)

    s_num = max(1, runtime_args.start if runtime_args.start is not None else START_FROM_QUESTION_NUM)
    end_num = runtime_args.end if runtime_args.end is not None else END_AT_QUESTION_NUM
    e_num = end_num if end_num else original_total_questions

    # Resume from checkpoint (skip if MOC_IGNORE_CHECKPOINT is set)
    saved_correct, last_done = (0, 0) if os.environ.get("MOC_IGNORE_CHECKPOINT", "").strip().lower() in {"1", "true", "yes", "on"} else _load_moc_checkpoint()
    if last_done > 0:
        resume_from = last_done + 1
        logger_early = setup_logger("MOC", resume_from, e_num)
        logger_early.info(f"🔄 Resuming from checkpoint: completed Q1-Q{last_done}, {saved_correct} correct so far.")
    else:
        resume_from = s_num

    if resume_from > e_num:
        print(f"All {e_num} questions already completed. Done.")
        raise SystemExit(0)

    start_index = resume_from - 1
    questions_to_run = all_questions[start_index:e_num]
    num_questions_to_run = len(questions_to_run)

    logger = setup_logger("MOC", resume_from, e_num)
    logger.info(f"▶️  Run Config: Task={TASK_NAME}, Q{resume_from}-Q{e_num}")
    logger.info("⚙️  Runtime: %s", runtime_summary(PROMPT_CONFIG))
    oracle_answer_hint = is_truthy_env("AGENT_ORACLE_ANSWER_HINT")
    if oracle_answer_hint:
        logger.warning("🧪 ORACLE DEBUG MODE enabled: ground-truth answers will be provided to the master for diagnostic runs only.")

    executor = AgentExecutor(agent_instance=qwen_agent_instance, prompt_config=PROMPT_CONFIG)
    correct_count = saved_correct

    for i, q_data in enumerate(questions_to_run):
        abs_num = i + resume_from
        q_id = q_data.get('id', str(abs_num))
        log_suffix = os.environ.get("MOC_LOG_SUFFIX", "").strip()
        if oracle_answer_hint and "oracle" not in log_suffix.lower():
            log_suffix = f"{log_suffix}_oracle" if log_suffix else "_oracle"
        log_dir = os.path.join(_ROOT, "logs", TASK_NAME, str(q_id) + log_suffix)
        os.makedirs(log_dir, exist_ok=True)

        logger.info(f"=== Testing Q{abs_num} (ID: {q_id}) ===")
        try:
            class_id = q_data["vid"]
            query_raw = q_data['question']
            objects = q_data.get('objects', [])
            placeholders = extract_objects(query_raw)
            obj_map = {}
            target_ids = set()
            for ph in placeholders:
                obj_entry = resolve_placeholder_object(ph, objects)
                if obj_entry is not None:
                    obj_id = obj_entry['id']
                    obj_map[ph] = f"obj_{obj_id}"
                    target_ids.add(obj_id)
                else:
                    obj_map[ph] = ph
            fmt_query = query_raw.format(**obj_map)
            if obj_map:
                logger.info(
                    "  Object mapping: %s",
                    ", ".join(f"{{{ph}}}->{obj}" for ph, obj in sorted(obj_map.items()))
                )

            contexts = []
            bbox_hints = []
            for view_idx, view_id in enumerate([1, 2]):
                path, total, bbox_data, img_files = prepare_metadata(class_id, target_ids, view_id)
                view_char = "A" if view_id == 1 else "B"
                hint_count = 0
                for frame_idx in sorted(bbox_data.keys()):
                    if hint_count >= 3:
                        break
                    for tag, box in bbox_data[frame_idx].items():
                        hint = f"{tag} appears in Video {view_idx+1} (View {view_char}) at Frame {frame_idx}"
                        if hint not in bbox_hints:
                            bbox_hints.append(hint)
                            hint_count += 1
                contexts.append({
                    "type": "image_folder", "path": path, "total_frames": total,
                    "bbox_data": bbox_data, "image_files": img_files, "obj_map": obj_map,
                    "description": f"UAV View {view_char}"
                })

            if contexts[0]['total_frames'] == 0 and contexts[1]['total_frames'] == 0:
                logger.error("❌ No frames found for either view. Skipping.")
                _save_moc_checkpoint(correct_count, abs_num)
                continue

            executor.init_agent_state(
                query=fmt_query, possible_answers=q_data['options'], video_contexts=contexts,
                formatted_query=fmt_query, bbox_info="\n".join(bbox_hints),
                oracle_debug=(
                    {
                        "enabled": True,
                        "correct_answer": q_data.get("answer"),
                        "correct_option": get_option_text(q_data.get("options", []), q_data.get("answer")),
                        "instruction": (
                            "Diagnostic oracle mode only. You are given the ground-truth answer to help generate "
                            "a visual evidence log for mechanism analysis. Still use visual tools; do not answer "
                            "from the oracle alone. Explore which observations support the correct answer and "
                            "which misleading observations would lead to wrong alternatives."
                        ),
                    }
                    if oracle_answer_hint else None
                )
            )
            final_ans, _ = executor.run_agent_loop(max_turns)
            pred = parse_agent_output_json_list(final_ans, 4)
            truth = [q_data['answer']]
            is_correct = (len(truth) > 0 and set(pred) == set(truth))
            logger.info(f"  Pred: {pred} | Truth: {truth} | {'✅' if is_correct else '❌'}")
            if is_correct:
                correct_count += 1
            executor.save_logs(log_dir)
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
        finally:
            executor.cleanup_agent_state()
            _save_moc_checkpoint(correct_count, abs_num)
            logger.info(f"--- End of Q{q_id} ---")

    # Clean up checkpoint on full completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    logger.info(f"🏁 Accuracy: {correct_count}/{num_questions_to_run} ({correct_count/num_questions_to_run*100:.2f}%)")
