# MSR task entry. Run from agent_system: python run_tasks/run_MSR_agent.py

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import re
import logging
from typing import List, Dict, Any, Optional

from qwen_agent import QwenModel
from agent_executor import AgentExecutor
from runtime_options import apply_runtime_overrides, parse_runtime_args, resolve_prompt_config, runtime_summary
from utils.config import get_uav_data_dir
from utils.log_utils import setup_logger
from utils.answer_parse import parse_agent_output_json_list

QUESTION_FILE = os.path.join(_ROOT, "question", "MSR.json")
MAX_TURNS = 20
START_FROM_QUESTION_NUM = 1
END_AT_QUESTION_NUM = None
TASK_NAME = "MSR_ondemand"

base_data_dir = get_uav_data_dir()
frames_dir = os.path.join(base_data_dir, "frames")
jsons_dir = os.path.join(base_data_dir, "bbox")

DEFAULT_PROMPT_CONFIG = {"master": os.path.join(_ROOT, "prompts", "master_MSR.prompt")}

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
    try:
        with open(f"{jsons_dir}/{view}/{class_id}.json", "r") as f:
            all_bbox_data = json.load(f)
    except Exception as e:
        print(f"Error loading bbox JSON for vid={class_id} view={view}: {e}")
        return frame_folder_path, total_frames, {}, image_files

    bbox_lookup = {}
    for entity in all_bbox_data:
        eid = entity["id"]
        if eid in target_object_ids:
            tag = f"obj_{eid}"
            bbox_container = entity.get("bbox")
            if not bbox_container: continue
            if isinstance(bbox_container, dict):
                for f_key, box_item in bbox_container.items():
                    try:
                        f_idx = int(f_key)
                        standard_box = None
                        if isinstance(box_item, dict):
                            if all(k in box_item for k in ['xtl', 'ytl', 'xbr', 'ybr']):
                                if box_item['xtl'] is not None:
                                    standard_box = [
                                        int(box_item['xtl']), int(box_item['ytl']),
                                        int(box_item['xbr']), int(box_item['ybr'])
                                    ]
                        elif isinstance(box_item, list):
                            standard_box = box_item
                        if standard_box:
                            if f_idx not in bbox_lookup: bbox_lookup[f_idx] = {}
                            bbox_lookup[f_idx][tag] = standard_box
                    except Exception: pass
            elif isinstance(bbox_container, list):
                for f_idx, box_item in enumerate(bbox_container):
                    standard_box = None
                    if isinstance(box_item, dict):
                        if all(k in box_item for k in ['xtl', 'ytl', 'xbr', 'ybr']):
                            if box_item['xtl'] is not None:
                                standard_box = [
                                    int(box_item['xtl']), int(box_item['ytl']),
                                    int(box_item['xbr']), int(box_item['ybr'])
                                ]
                    elif isinstance(box_item, list):
                        standard_box = box_item
                    if standard_box:
                        if f_idx not in bbox_lookup: bbox_lookup[f_idx] = {}
                        bbox_lookup[f_idx][tag] = standard_box
    return frame_folder_path, total_frames, bbox_lookup, image_files

if __name__ == "__main__":
    runtime_args = parse_runtime_args("Run MSR agent with selectable master/tool backends.")
    apply_runtime_overrides(runtime_args)
    PROMPT_CONFIG = resolve_prompt_config(_ROOT, "MSR", DEFAULT_PROMPT_CONFIG, runtime_args)
    max_turns = runtime_args.max_turns or MAX_TURNS

    if not os.path.exists(QUESTION_FILE):
        print(f"FATAL: Question file not found: {QUESTION_FILE}")
        raise SystemExit(1)
    with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)

    s_num = max(1, runtime_args.start if runtime_args.start is not None else START_FROM_QUESTION_NUM)
    end_num = runtime_args.end if runtime_args.end is not None else END_AT_QUESTION_NUM
    e_num = end_num if end_num else len(all_questions)
    questions_to_run = all_questions[s_num-1 : e_num]

    logger = setup_logger("MSR", s_num, e_num)
    logger.info(f"▶️  Run Config: Task={TASK_NAME}, Q{s_num}-Q{e_num}")
    logger.info("⚙️  Runtime: %s", runtime_summary(PROMPT_CONFIG))

    agent = QwenModel(prompt_config=PROMPT_CONFIG)
    executor = AgentExecutor(agent_instance=agent, prompt_config=PROMPT_CONFIG)
    correct_count = 0

    for i, q_data in enumerate(questions_to_run):
        abs_num = i + s_num
        q_id = q_data.get('id', str(abs_num))
        log_suffix = os.environ.get("MSR_LOG_SUFFIX", "").strip()
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
                    if hint_count >= 3: break
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
                continue

            executor.init_agent_state(
                query=fmt_query, possible_answers=q_data['options'], video_contexts=contexts,
                formatted_query=fmt_query, bbox_info="\n".join(bbox_hints)
            )
            final_ans, _ = executor.run_agent_loop(max_turns)
            pred = parse_agent_output_json_list(final_ans, 4)
            truth = [q_data['answer']]
            is_correct = (len(truth) > 0 and set(pred) == set(truth))
            logger.info(f"  Pred: {pred} | Truth: {truth} | {'✅' if is_correct else '❌'}")
            if is_correct: correct_count += 1
            executor.save_logs(log_dir)
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
        finally:
            executor.cleanup_agent_state()

    logger.info(f"🏁 Accuracy: {correct_count}/{len(questions_to_run)} ({correct_count/len(questions_to_run)*100:.2f}%)")
