"""
Task Runner: adapts each CrossVid task to the AgentExecutor.
Handles video context setup, query construction, and answer evaluation.

Imports from agent_system/ but stays read-only — no modifications to existing code.
"""

import json
import os
import re
import sys
import traceback
from typing import List, Dict, Any, Optional, Tuple

# Ensure agent_system is importable
_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..", "agent_system")
_AGENT_ROOT = os.path.abspath(_AGENT_ROOT)
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from qwen_agent import QwenModel
from agent_executor import AgentExecutor
from runtime_options import resolve_prompt_config
from utils.config import get_local_video_root, get_remote_video_base_url, get_uav_data_dir

from task_config import (
    TASK_CODES, build_query, get_ground_truth_text, check_correct, get_num_options,
)

import logging
logger = logging.getLogger("DistillAgent.TaskRunner")

MAX_FRAME_LENGTH = 360
MAX_TURNS_DEFAULT = 20


def _get_video_path(video_name: str, subdir: str = "") -> Tuple[str, str]:
    """Return (local_path, remote_url) for a video."""
    local_root = get_local_video_root()
    remote_base = get_remote_video_base_url().rstrip("/")
    if subdir:
        local_path = os.path.join(local_root, subdir, video_name)
        remote_url = f"{remote_base}/{subdir}/{video_name}"
    else:
        local_path = os.path.join(local_root, video_name)
        remote_url = f"{remote_base}/{video_name}"
    return local_path, remote_url


def _resolve_path(local_path: str, remote_url: str) -> str:
    """Use local path if it exists, otherwise remote URL."""
    if os.path.exists(local_path):
        return local_path
    return remote_url


# ═══════════════════════════════════════════════════════════════
# Per-task video context builders
# ═══════════════════════════════════════════════════════════════

def _setup_pss(question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """PSS: shuffled segments from one cooking video."""
    from utils import video_processor

    video_name = question_data.get("video")
    segments_dict = question_data.get("segments")
    if not video_name or not segments_dict:
        raise ValueError("PSS: missing 'video' or 'segments'")

    local_path, remote_url = _get_video_path(video_name)
    video_path = _resolve_path(local_path, remote_url)

    # Probe video metadata
    _, _, _, original_fps, total_f, total_d = video_processor.process_video(
        input_path=video_path, n_frames=1, intervals=[(0, 1)],
        max_length=MAX_FRAME_LENGTH, encode=False,
    )

    contexts = []
    sorted_keys = sorted(segments_dict.keys(), key=lambda x: int(re.sub(r'\D', '', x) or 0))
    for key in sorted_keys:
        raw_intervals = segments_dict[key]
        if not raw_intervals or not isinstance(raw_intervals[0], list):
            continue
        seg_start = float(raw_intervals[0][0])
        seg_end = float(raw_intervals[-1][1])
        contexts.append({
            "type": "video_file",
            "path": video_path,
            "fps": original_fps,
            "total_frames": total_f,
            "duration_seconds": max(0.0, seg_end - seg_start),
            "clip_begin": seg_start,
            "clip_end": seg_end,
            "name_for_log": f"{video_name}_seg_{key}",
        })

    return contexts


def _setup_fsa(question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """FSA: two videos, find equivalent segment."""
    from utils import video_processor

    video_a = question_data.get("video A")
    video_b = question_data.get("video B")
    if not video_a or not video_b:
        raise ValueError("FSA: missing 'video A' or 'video B'")

    contexts = []
    for vname in [video_a, video_b]:
        local_path, remote_url = _get_video_path(vname)
        video_path = _resolve_path(local_path, remote_url)
        _, _, _, fps, total_f, total_d = video_processor.process_video(
            input_path=video_path, n_frames=1, intervals=[(0, 1)],
            max_length=MAX_FRAME_LENGTH, encode=False,
        )
        contexts.append({
            "type": "video_file",
            "path": video_path,
            "fps": fps,
            "total_frames": total_f,
            "duration_seconds": max(0.0, total_d),
            "clip_begin": 0.0,
            "clip_end": total_d,
        })

    return contexts


def _setup_standard_videos(
    question_data: Dict[str, Any],
    subdir: str = "",
) -> List[Dict[str, Any]]:
    """Standard multi-video tasks: CC, NC, PEA, BU, PI."""
    from utils import video_processor

    # Try 'videos' key (list) or 'video' key (single string)
    video_names = question_data.get("videos")
    if not video_names:
        video_names = question_data.get("video")
    if not video_names:
        raise ValueError(f"Missing video field in question data: {list(question_data.keys())}")
    if isinstance(video_names, str):
        video_names = [video_names]

    contexts = []
    for vname in video_names:
        local_path, remote_url = _get_video_path(vname, subdir=subdir)
        video_path = _resolve_path(local_path, remote_url)
        _, _, _, fps, total_f, total_d = video_processor.process_video(
            input_path=video_path, n_frames=1, intervals=[(0, 1)],
            max_length=MAX_FRAME_LENGTH, encode=False,
        )
        contexts.append({
            "type": "video_file",
            "path": video_path,
            "fps": fps,
            "total_frames": total_f,
            "duration_seconds": max(0.0, total_d),
            "clip_begin": 0.0,
            "clip_end": total_d,
        })

    return contexts


def _setup_ccqa(question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """CCQA: two videos (video A, video B)."""
    from utils import video_processor

    contexts = []
    for key in ["video A", "video B"]:
        vname = question_data.get(key)
        if not vname:
            raise ValueError(f"CCQA: missing '{key}'")
        local_path, remote_url = _get_video_path(vname)
        video_path = _resolve_path(local_path, remote_url)
        _, _, _, fps, total_f, total_d = video_processor.process_video(
            input_path=video_path, n_frames=1, intervals=[(0, 1)],
            max_length=MAX_FRAME_LENGTH, encode=False,
        )
        contexts.append({
            "type": "video_file",
            "path": video_path,
            "fps": fps,
            "total_frames": total_f,
            "duration_seconds": max(0.0, total_d),
            "clip_begin": 0.0,
            "clip_end": total_d,
        })

    return contexts


def _setup_uav(question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """MOC/MSR: UAV image folders with bbox data."""
    base_data_dir = get_uav_data_dir()
    frames_dir = os.path.join(base_data_dir, "frames")
    jsons_dir = os.path.join(base_data_dir, "bbox")

    class_id = question_data.get("vid")
    objects = question_data.get("objects", [])

    # Resolve target object IDs
    query_raw = question_data.get("question", "")
    placeholders = re.findall(r'\{(.*?)\}', query_raw)
    obj_map = {}
    target_ids = set()
    for ph in placeholders:
        match = re.fullmatch(r'[AB](\d+)', ph)
        if match:
            obj_index = int(match.group(1)) - 1
            if 0 <= obj_index < len(objects):
                obj_id = objects[obj_index]['id']
                obj_map[ph] = f"obj_{obj_id}"
                target_ids.add(obj_id)
            else:
                obj_map[ph] = ph
        else:
            obj_map[ph] = ph

    contexts = []
    for view_id in [1, 2]:
        frame_folder = os.path.join(frames_dir, str(view_id), f"{class_id}-{view_id}")
        if not os.path.exists(frame_folder):
            contexts.append({
                "type": "image_folder", "path": frame_folder,
                "total_frames": 0, "bbox_data": {}, "image_files": [],
                "obj_map": obj_map,
                "description": f"UAV View {chr(64+view_id)}",
            })
            continue

        image_files = sorted([
            f for f in os.listdir(frame_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        total_frames = len(image_files)

        # Load bbox
        bbox_lookup = {}
        bbox_file = os.path.join(jsons_dir, str(view_id), f"{class_id}.json")
        if os.path.exists(bbox_file):
            try:
                with open(bbox_file, "r") as f:
                    all_bbox = json.load(f)
                for entity in all_bbox:
                    eid = entity["id"]
                    if eid not in target_ids:
                        continue
                    tag = f"obj_{eid}"
                    bbox_container = entity.get("bbox", {})
                    if isinstance(bbox_container, dict):
                        for f_key, box_item in bbox_container.items():
                            try:
                                f_idx = int(f_key)
                                if isinstance(box_item, dict) and all(
                                    k in box_item for k in ['xtl', 'ytl', 'xbr', 'ybr']
                                ):
                                    if box_item['xtl'] is not None:
                                        bbox_lookup.setdefault(f_idx, {})[tag] = [
                                            int(box_item['xtl']), int(box_item['ytl']),
                                            int(box_item['xbr']), int(box_item['ybr']),
                                        ]
                            except Exception:
                                pass
            except Exception as e:
                logger.warning("Failed to load bbox for %s view %d: %s", class_id, view_id, e)

        contexts.append({
            "type": "image_folder",
            "path": frame_folder,
            "total_frames": total_frames,
            "bbox_data": bbox_lookup,
            "image_files": image_files,
            "obj_map": obj_map,
            "description": f"UAV View {chr(64+view_id)}",
        })

    return contexts


# Registry mapping task code → context builder
CONTEXT_BUILDERS = {
    "PSS": _setup_pss,
    "FSA": _setup_fsa,
    "PEA": _setup_standard_videos,
    "CC": _setup_standard_videos,
    "NC": _setup_standard_videos,
    "BU": _setup_standard_videos,
    "PI": lambda qd: _setup_standard_videos(qd, subdir="movie"),
    "CCQA": _setup_ccqa,
    "MOC": _setup_uav,
    "MSR": _setup_uav,
}


def build_video_contexts(task: str, question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build video_contexts list for AgentExecutor.init_agent_state()."""
    builder = CONTEXT_BUILDERS.get(task)
    if builder is None:
        raise ValueError(f"Unknown task: {task}. Supported: {TASK_CODES}")
    return builder(question_data)


def run_single_question(
    task: str,
    question_data: Dict[str, Any],
    agent_instance: QwenModel,
    prompt_config: Dict[str, str],
    max_turns: int = MAX_TURNS_DEFAULT,
    oracle_debug: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None,
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run one question through the agent loop.

    Args:
        log_dir: If set, save detailed logs (master_dialogue.json, tools_execution.json)
                 to this directory after the run.

    Returns:
        (final_answer, master_history, tool_history, video_contexts)
    """
    query = build_query(task, question_data)
    possible_answers = question_data.get("options", [])

    # Build video contexts
    video_contexts = build_video_contexts(task, question_data)

    if not video_contexts:
        raise RuntimeError(f"No video contexts built for task {task}")

    # Check for empty image folders (UAV tasks)
    if task in ("MOC", "MSR"):
        has_frames = any(ctx.get("total_frames", 0) > 0 for ctx in video_contexts)
        if not has_frames:
            raise RuntimeError("No frames found for UAV views")

    executor = AgentExecutor(
        agent_instance=agent_instance,
        prompt_config=prompt_config,
    )

    executor.init_agent_state(
        query=query,
        possible_answers=possible_answers,
        video_contexts=video_contexts,
        oracle_debug=oracle_debug,
    )

    try:
        final_answer, master_history = executor.run_agent_loop(max_turns=max_turns)
        tool_history = list(executor._tool_history)
        return final_answer, master_history, tool_history, video_contexts
    finally:
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                executor.save_logs(log_dir)
            except Exception as e:
                logger.warning("Failed to save logs to %s: %s", log_dir, e)
        executor.cleanup_agent_state()
