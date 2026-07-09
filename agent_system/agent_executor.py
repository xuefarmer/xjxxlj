# agent_executor.py
# Orchestrates the agent loop: master dialogue, active_perception (video/image).
# get_caption (Whisper) disabled — GPU driver mismatch causes hang on load.
# Supports standard video and UAV image-folder contexts.

import re
import logging
import time
import json
import base64
import os
import copy
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Callable, List, Any, Optional, Tuple, Union
from datetime import datetime

import cv2
import numpy as np

from utils import video_processor
from qwen_agent import QwenModel
# DISABLED: GPU driver mismatch causes Whisper load_model to hang
# from utils.caption_generator import generate_caption_for_segment
from utils.sanitize import safe_truncate, sanitize_for_log
from utils.text_utils import clean_thought_content

logger = logging.getLogger("AgentWorkflowLogger")

MAX_FRAME_LENGTH = 360
MAX_UAV_FRAME_LENGTH = int(os.environ.get("MAX_UAV_FRAME_LENGTH", "768"))


class AgentExecutor:
    """Runs ReAct loop: master LLM + tools (active_perception)."""

    def __init__(self, agent_instance: QwenModel, prompt_config: Optional[Dict[str, str]] = None, tool_frames_per_clip: int = 128):
        self.agent = agent_instance
        self.tool_frames_per_clip = tool_frames_per_clip
        self.video_contexts: List[Dict[str, Any]] = []
        self.tool_mapping = self._create_tool_mapping()

        default_prompts = {"master": "prompts/master_autonomous.prompt"}
        self.prompt_config = default_prompts
        if prompt_config: self.prompt_config.update(prompt_config)
        
        master_prompt_path = self.prompt_config.get("master")
        master_base_path = self.prompt_config.get("master_base")
        task_contract_path = self.prompt_config.get("task_contract")
        if master_base_path and task_contract_path:
            self._master_prompt_name = self.prompt_config.get(
                "master_name",
                f"master_{self.prompt_config.get('task_code', 'task')}.prompt",
            )
            with open(master_base_path, 'r', encoding='utf-8') as f:
                base_prompt = f.read().strip()
            with open(task_contract_path, 'r', encoding='utf-8') as f:
                task_contract = f.read().strip()
            self.master_prompt_template = (
                base_prompt
                + "\n\n"
                + task_contract
                + "\n"
            )
        else:
            if not master_prompt_path: raise ValueError("Prompt path for 'master' not configured.")
            self._master_prompt_name = os.path.basename(master_prompt_path)
            with open(master_prompt_path, 'r', encoding='utf-8') as f:
                self.master_prompt_template = f.read()

        self._possible_answers: List[str] = []
        self._initial_query: str = ""
        self._master_history: List[Dict[str, Any]] = []
        self._tool_history: List[Dict[str, Any]] = [] 
        self.cross_context: Dict[str, Any] = self._new_cross_context()
        self._pss_layer1_num_frames: Optional[int] = None
        self._oracle_debug: Optional[Dict[str, Any]] = None
        self.current_turn = 0

    def _load_prompt_template(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"FATAL: Master Prompt not found at {path}.")
            raise

    def _create_tool_mapping(self) -> Dict[str, Callable]:
        """Maps tool names to handlers: active_perception."""
        return {
            "active_perception": self._proxy_active_perception,
            # DISABLED: GPU driver mismatch causes Whisper load_model to hang
            # "get_caption": self._proxy_get_caption,
        }

    def save_logs(self, folder_path: str):
        """Persist master dialogue and tool execution to JSON and readable TXT (Base64 stripped)."""
        try:
            sanitized_master_history = sanitize_for_log(self._master_history)
            sanitized_tool_history = sanitize_for_log(self._tool_history)

            master_log_path = os.path.join(folder_path, "master_dialogue.json")
            with open(master_log_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_master_history, f, ensure_ascii=False, indent=2)
            
            tool_log_path = os.path.join(folder_path, "tools_execution.json")
            with open(tool_log_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_tool_history, f, ensure_ascii=False, indent=2)

            context_log_path = os.path.join(folder_path, "cross_context.json")
            with open(context_log_path, 'w', encoding='utf-8') as f:
                json.dump(sanitize_for_log(self.cross_context), f, ensure_ascii=False, indent=2)

            readable_master_path = os.path.join(folder_path, "master_dialogue_readable.txt")
            with open(readable_master_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Master Dialogue Log (Readable) ===\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, msg in enumerate(sanitized_master_history):
                    role = msg.get("role", "UNKNOWN").upper()
                    content = msg.get("content", "")
                    
                    f.write(f"--- [Step {i}] {role} ---\n")
                    
                    if isinstance(content, (dict, list)):
                        f.write(json.dumps(content, ensure_ascii=False, indent=2))
                    else:
                        f.write(str(content))
                    
                    f.write("\n\n" + "="*60 + "\n\n")

            readable_tool_path = os.path.join(folder_path, "tools_execution_readable.txt")
            with open(readable_tool_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Tools Execution Log (Readable) ===\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for entry in sanitized_tool_history:
                    turn = entry.get("turn", "?")
                    tool = entry.get("tool_name", "unknown")
                    call_id = entry.get("call_id", "unknown")
                    
                    f.write(f"### [Turn {turn}] Tool: {tool} (ID: {call_id}) ###\n")
                    f.write(f"Time: {entry.get('timestamp_start')} -> {entry.get('timestamp_end')}\n")
                    
                    f.write("\n[Inputs]:\n")
                    inputs = entry.get("inputs")
                    if isinstance(inputs, (dict, list)):
                        f.write(json.dumps(inputs, ensure_ascii=False, indent=2))
                    else:
                        f.write(str(inputs))
                        
                    f.write("\n\n[Output]:\n")
                    output = entry.get("output_raw")
                    if isinstance(output, (dict, list)):
                        f.write(json.dumps(output, ensure_ascii=False, indent=2))
                    else:
                        f.write(str(output))
                        
                    f.write("\n\n" + "-"*60 + "\n\n")

            logger.info(f"    💾 Logs saved to: {folder_path} (JSON + Readable TXT)")
            
        except Exception as e:
            logger.error(f"    ❌ Failed to save logs: {e}", exc_info=True)


    @staticmethod
    def _new_cross_context() -> Dict[str, Any]:
        """Creates the per-question cross-video context buffer."""
        return {
            "confirmed": [],
            "uncertain": [],
            "uncertain_visual_states": [],
            "next_vlm_attention": [],
            "next_object_focus": [],
            "event_memory": [],
            "event_state_links": [],
            "object_memory": {},
            "object_state_memory": [],
            "object_state_chains": [],
            "dimensions_to_compare": [],
            "resolved_uncertain": [],
            "notes": [],
            "updates": [],
        }

    @staticmethod
    def _normalize_context_items(value: Any) -> List[str]:
        """Normalizes LLM-supplied context values to short, deduplicable strings."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = [value]

        items: List[str] = []
        for item in raw_items:
            if item is None:
                continue
            if isinstance(item, (dict, list)):
                try:
                    text = json.dumps(item, ensure_ascii=False)
                except Exception:
                    text = str(item)
            else:
                text = str(item)
            text = text.strip()
            if text:
                items.append(text)
        return items

    def _append_cross_context_items(self, key: str, value: Any):
        if key not in self.cross_context or not isinstance(self.cross_context.get(key), list):
            self.cross_context[key] = []

        for item in self._normalize_context_items(value):
            if item not in self.cross_context[key]:
                self.cross_context[key].append(item)

    def _merge_object_memory(self, value: Any):
        """Merges shared-key object memory: object -> video -> visible state."""
        if not isinstance(self.cross_context.get("object_memory"), dict):
            self.cross_context["object_memory"] = {}
        if not isinstance(value, dict):
            self._append_cross_context_items("object_state_memory", value)
            return

        for object_key, per_video_state in value.items():
            object_name = str(object_key).strip()
            if not object_name:
                continue
            if object_name not in self.cross_context["object_memory"] or not isinstance(self.cross_context["object_memory"].get(object_name), dict):
                self.cross_context["object_memory"][object_name] = {}

            if isinstance(per_video_state, dict):
                for video_key, state in per_video_state.items():
                    video_name = str(video_key).strip()
                    if not video_name:
                        continue
                    self.cross_context["object_memory"][object_name][video_name] = state
            else:
                self.cross_context["object_memory"][object_name]["note"] = per_video_state

    def _remove_cross_context_items(self, key: str, value: Any):
        if key not in self.cross_context or not isinstance(self.cross_context.get(key), list):
            return
        items_to_remove = set(self._normalize_context_items(value))
        if not items_to_remove:
            return
        self.cross_context[key] = [
            item for item in self.cross_context[key]
            if item not in items_to_remove
        ]

    def _merge_cross_context_update(self, context_update: Any):
        """Merges optional LLM context_update into the cross-video context buffer."""
        if context_update is None:
            return

        if not isinstance(context_update, dict):
            context_update = {"notes": context_update}

        if context_update.get("reset"):
            self.cross_context = self._new_cross_context()

        if context_update.get("clear_uncertain"):
            self.cross_context["uncertain"] = []
            self.cross_context["uncertain_visual_states"] = []

        if "resolved_uncertain" in context_update:
            self._append_cross_context_items("resolved_uncertain", context_update.get("resolved_uncertain"))
            self._remove_cross_context_items("uncertain", context_update.get("resolved_uncertain"))
            self._remove_cross_context_items("uncertain_visual_states", context_update.get("resolved_uncertain"))

        control_keys = {"reset", "clear_uncertain", "resolved_uncertain"}
        replace_keys = {"uncertain", "uncertain_visual_states", "next_vlm_attention", "next_object_focus"}
        for key, value in context_update.items():
            if key in control_keys:
                continue
            if key == "object_memory":
                self._merge_object_memory(value)
                continue
            if key in replace_keys:
                self.cross_context[key] = self._normalize_context_items(value)
                continue
            self._append_cross_context_items(key, value)

        update_record = {
            "turn": self.current_turn,
            "update": context_update,
        }
        self.cross_context.setdefault("updates", []).append(update_record)
        logger.info(
            "  🧠 Cross-context updated: confirmed=%d, uncertain=%d, dimensions=%d",
            len(self.cross_context.get("confirmed", [])),
            len(self.cross_context.get("uncertain", [])),
            len(self.cross_context.get("dimensions_to_compare", [])),
        )

    def _build_cross_context_message(self) -> Optional[Dict[str, Any]]:
        """Builds an ephemeral message so the next focus prompt inherits prior findings."""
        if not self.cross_context:
            return None

        visible_context = {
            key: value
            for key, value in self.cross_context.items()
            if key != "updates" and value
        }
        if not visible_context:
            return None

        context_text = (
            "[Cross-Video Context Buffer]\n"
            "This buffer contains findings accumulated from previous turns. "
            "When you write the next observe focus_prompt, explicitly inherit the confirmed context, "
            "address the uncertain items, and compare the listed dimensions. "
            "If a new observation contradicts earlier understanding, backtrack and update this buffer.\n"
            f"{json.dumps(visible_context, ensure_ascii=False, indent=2)}"
        )
        return {"role": "user", "content": context_text}

    def _augment_focus_prompt_with_cross_context(self, focus_prompt: str) -> str:
        if focus_prompt is None:
            focus_prompt = ""
        elif not isinstance(focus_prompt, str):
            focus_prompt = str(focus_prompt)

        is_pss_visual_call = self._is_pss_sort_task() and (
            "MODE=layer2_state_scan" in focus_prompt or "MODE=focus" in focus_prompt
        )

        if self._is_pss_sort_task() and "MODE=layer2_state_scan" in focus_prompt:
            evidence_note = (
                "\n\n[PSS Evidence Notice]\n"
                "This is a local visual state extraction call. Do not decide chronology, do not confirm first/last, "
                "and do not say that any video happens before/after another video. "
                "Report only visible object-state cells for the current frames."
            )
            focus_prompt = f"{focus_prompt}{evidence_note}"
        elif self._is_pss_sort_task() and "MODE=focus" in focus_prompt:
            neutral_focus_note = (
                "\n\n[PSS Neutral Focus Notice]\n"
                "Even if the prompt or accumulated context mentions a leading edge/order, do not decide which segment is earlier "
                "and do not choose between hypotheses. Report only per-target visible object-state cells: object state, "
                "container fill/empty state, source/destination, residue/depletion, background object state, confidence, and uncertainty. "
                "The master will compare these cells after the tool returns."
            )
            focus_prompt = f"{focus_prompt}{neutral_focus_note}"

        visible_context = {
            key: value
            for key, value in self.cross_context.items()
            if key != "updates" and value
        }
        if not visible_context:
            return focus_prompt

        if is_pss_visual_call:
            pss_local_context = {
                key: visible_context[key]
                for key in [
                    "object_memory",
                    "object_state_memory",
                    "dimensions_to_compare",
                    "uncertain_visual_states",
                    "next_object_focus",
                    "uncertain",
                ]
                if key in visible_context
            }
            if not pss_local_context:
                return focus_prompt
            context_block = (
                "\n\n[PSS Local-State Context]\n"
                "Use this only to know which objects/states to inspect. Ignore any implied chronological order. "
                "Return compact local evidence and uncertainty, not a final sequence.\n"
                f"{json.dumps(pss_local_context, ensure_ascii=False, indent=2)}"
            )
            return f"{focus_prompt}{context_block}"

        context_block = (
            "\n\n[Accumulated Cross-Video Context]\n"
            "Carry these findings into the observation. Focus especially on unresolved uncertainty "
            "and the dimensions that must be compared.\n"
            f"{json.dumps(visible_context, ensure_ascii=False, indent=2)}"
        )
        return f"{focus_prompt}{context_block}"

    def _normalize_pss_frame_counts(
        self,
        observation_targets: List[Dict[str, Any]],
        focus_prompt: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Keeps PSS comparison layers fair by using one frame count per observe call."""
        if getattr(self, "_master_prompt_name", "") != "master_PSS.prompt" or len(observation_targets) <= 1:
            return observation_targets, focus_prompt

        frame_counts = [
            target.get("num_frames")
            for target in observation_targets
            if target.get("num_frames") is not None
        ]
        if not frame_counts:
            return observation_targets, focus_prompt

        first_num_frames = int(frame_counts[0])
        is_layer1 = "MODE=layer1_caption" in focus_prompt
        is_layer2 = "MODE=layer2_state_scan" in focus_prompt

        if is_layer1 and len(set(frame_counts)) == 1:
            self._pss_layer1_num_frames = first_num_frames

        if is_layer2:
            allowed_num_frames = self._pss_layer1_num_frames or 8
            if any(int(count) != allowed_num_frames for count in frame_counts):
                normalized_targets = copy.deepcopy(observation_targets)
                for target in normalized_targets:
                    target["num_frames"] = allowed_num_frames
                budget_note = (
                    "\n\n[Layer2 Frame Budget Notice]\n"
                    f"Layer 2 must not increase visual sampling over Layer 1. "
                    f"The executor normalized every target to num_frames={allowed_num_frames}."
                )
                logger.info(
                    "    [EXECUTOR] PSS Layer2 frame budget normalized num_frames to %d for %d targets.",
                    allowed_num_frames,
                    len(normalized_targets),
                )
                return normalized_targets, f"{focus_prompt}{budget_note}"

        if len(set(frame_counts)) <= 1:
            return observation_targets, focus_prompt

        uniform_num_frames = first_num_frames
        normalized_targets = copy.deepcopy(observation_targets)
        for target in normalized_targets:
            target["num_frames"] = uniform_num_frames

        fairness_note = (
            "\n\n[Frame Fairness Notice]\n"
            f"For fair cross-video comparison, the executor normalized every target in this observe call "
            f"to num_frames={uniform_num_frames}."
        )
        logger.info(
            "    [EXECUTOR] PSS frame fairness normalized num_frames to %d for %d targets.",
            uniform_num_frames,
            len(normalized_targets),
        )
        return normalized_targets, f"{focus_prompt}{fairness_note}"

    @staticmethod
    def _env_flag(name: str, default: str = "false") -> bool:
        return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}

    def _task_code(self) -> str:
        return str(self.prompt_config.get("task_code", "")).strip().upper()

    def _layer1_skim_enabled(self, focus_prompt: str) -> bool:
        if not self._env_flag("AGENT_LAYER1_SKIM", "false"):
            return False
        if "MODE=layer1" not in str(focus_prompt) and "MODE=layer1_caption" not in str(focus_prompt):
            return False

        task_code = self._task_code()
        raw_tasks = os.environ.get("AGENT_LAYER1_SKIM_TASKS", "NC")
        allowed_tasks = {task.strip().upper() for task in raw_tasks.split(",") if task.strip()}
        return "*" in allowed_tasks or task_code in allowed_tasks

    @staticmethod
    def _safe_int_env(name: str, default: int, minimum: int = 1) -> int:
        try:
            return max(minimum, int(os.environ.get(name, str(default))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _encode_rgb_frame(frame: np.ndarray) -> Optional[str]:
        if frame is None:
            return None
        try:
            success, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not success:
                return None
            return base64.b64encode(buffer).decode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _frame_skim_embedding(frame: np.ndarray) -> Optional[np.ndarray]:
        """Lightweight spatial-color vector used to pick chunk representatives."""
        if frame is None:
            return None
        try:
            rgb_small = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
            rgb_vec = rgb_small.astype(np.float32).reshape(-1) / 255.0

            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist(
                [hsv],
                [0, 1, 2],
                None,
                [8, 4, 4],
                [0, 180, 0, 256, 0, 256],
            ).astype(np.float32).reshape(-1)
            hist_sum = float(hist.sum())
            if hist_sum > 0:
                hist /= hist_sum

            vec = np.concatenate([rgb_vec, hist], axis=0)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            return vec
        except Exception:
            return None

    def _select_layer1_skim_frames(
        self,
        raw_frames: List[np.ndarray],
        mapping_table: Dict[int, int],
    ) -> Tuple[List[Tuple[int, int, str]], str]:
        """Select centroid/anomaly representatives from one chunk."""
        reps_per_chunk = self._safe_int_env("AGENT_LAYER1_SKIM_FRAMES_PER_CHUNK", 2, minimum=1)
        if len(raw_frames) <= reps_per_chunk:
            selected = []
            for idx, frame in enumerate(raw_frames):
                encoded = self._encode_rgb_frame(frame)
                if encoded:
                    selected.append((idx, int(mapping_table.get(idx, idx)), encoded))
            return selected, "all"

        embeddings = [self._frame_skim_embedding(frame) for frame in raw_frames]
        valid = [(idx, emb) for idx, emb in enumerate(embeddings) if emb is not None]
        if not valid:
            fallback_indices = sorted(set(np.linspace(0, len(raw_frames) - 1, reps_per_chunk).astype(int)))
            selected = []
            for idx in fallback_indices:
                encoded = self._encode_rgb_frame(raw_frames[idx])
                if encoded:
                    selected.append((idx, int(mapping_table.get(idx, idx)), encoded))
            return selected, "uniform_fallback"

        valid_indices = [idx for idx, _ in valid]
        matrix = np.stack([emb for _, emb in valid], axis=0)
        centroid = matrix.mean(axis=0)
        distances = np.linalg.norm(matrix - centroid, axis=1)

        chosen: List[int] = []
        strategy_parts: List[str] = []

        centroid_idx = valid_indices[int(np.argmin(distances))]
        chosen.append(centroid_idx)
        strategy_parts.append("centroid")

        if reps_per_chunk >= 2:
            anomaly_idx = valid_indices[int(np.argmax(distances))]
            chosen.append(anomaly_idx)
            strategy_parts.append("anomaly")

        if reps_per_chunk >= 3:
            chosen.append(0)
            strategy_parts.append("start")

        if reps_per_chunk >= 4:
            chosen.append(len(raw_frames) - 1)
            strategy_parts.append("end")

        if len(set(chosen)) < reps_per_chunk:
            for idx in np.linspace(0, len(raw_frames) - 1, reps_per_chunk).astype(int):
                chosen.append(int(idx))
                if len(set(chosen)) >= reps_per_chunk:
                    break

        selected_indices = sorted(set(chosen))[:reps_per_chunk]
        selected = []
        for idx in selected_indices:
            encoded = self._encode_rgb_frame(raw_frames[idx])
            if encoded:
                selected.append((idx, int(mapping_table.get(idx, idx)), encoded))

        return selected, "+".join(strategy_parts)

    def _proxy_active_perception(self, observation_targets: List[Dict[str, Any]], focus_prompt: str) -> Dict[str, Any]:
        """Routes to UAV (image_folder) or standard (video) perception handler."""
        if not observation_targets: return {"result": "Error: No observation targets provided."}
        observation_targets, focus_prompt = self._normalize_pss_frame_counts(observation_targets, focus_prompt)
        focus_prompt = self._augment_focus_prompt_with_cross_context(focus_prompt)

        first_idx = observation_targets[0].get("video_index")
        if first_idx is None: return {"result": "Error: Missing video_index."}
        
        real_idx = first_idx - 1
        if real_idx < 0 or real_idx >= len(self.video_contexts):
            return {"result": f"Error: Video index {first_idx} out of range."}
        
        context = self.video_contexts[real_idx]
        ctx_type = context.get("type", "video_file")

        if ctx_type == "image_folder":
            logger.info("    [EXECUTOR] 🔀 Dispatching to UAV Handler (will use Tool API)")
            return self._active_perception_uav(observation_targets, focus_prompt)
        else:
            logger.info("    [EXECUTOR] 🔀 Dispatching to Standard Handler (will use Tool API)")
            return self._active_perception_standard(observation_targets, focus_prompt)

    def _active_perception_uav(self, observation_targets: List[Dict[str, Any]], focus_prompt: str) -> Dict[str, Any]:
        """UAV path: frame-index based, draws bboxes, uses image_files list for lookup."""
        logger.info(f"    [TOOL-UAV] 🚁 Processing {len(observation_targets)} targets (Frame-based)")
        
        all_frames = []
        frame_mapping_text_parts = []
        
        try:
            for target in observation_targets:
                v_idx = target.get("video_index")
                if v_idx is None: return {"result": "Error: Missing video_index"}

                real_idx = v_idx - 1
                if real_idx < 0 or real_idx >= len(self.video_contexts):
                     return {"result": f"Error: Video index {v_idx} out of range."}

                context = self.video_contexts[real_idx]
                
                folder_path = context.get("path")
                bbox_lookup = context.get("bbox_data", {})
                obj_map = context.get("obj_map", {})
                total_f = context.get("total_frames", 0)
                image_files_list = context.get("image_files", [])

                # Accept both start_frame/end_frame (UAV-native) and start_time/end_time
                # (general-prompt fallback). For image_folder tasks, both are treated as
                # frame indices so Master can use a unified general prompt.
                start_f = target.get("start_frame")
                if start_f is None:
                    start_f = target.get("start_time")  # general prompt fallback
                end_f = target.get("end_frame")
                if end_f is None:
                    end_f = target.get("end_time")      # general prompt fallback
                n_frames = target.get("num_frames", 16)

                if start_f is None:
                    start_f = 0
                if end_f is None:
                    end_f = total_f - 1
                start_f = max(0, int(start_f))
                end_f = min(total_f - 1, int(end_f))
                
                if start_f > end_f:
                    return {"result": f"Error: start_frame {start_f} is greater than end_frame {end_f}"}

                indices = np.linspace(start_f, end_f, n_frames).astype(int)
                indices = sorted(list(set(indices)))
                
                extracted_count = 0
                for idx in indices:
                    if idx >= len(image_files_list):
                        logger.warning(f"Frame index {idx} out of bounds (Total: {len(image_files_list)})")
                        continue

                    file_name = image_files_list[idx] 
                    full_path = os.path.join(folder_path, file_name)
                    
                    if not os.path.exists(full_path):
                        continue
                    img = cv2.imread(full_path)
                    if img is None: continue
                    
                    if idx in bbox_lookup:
                        objs_in_frame = bbox_lookup[idx]
                        for tag, box in objs_in_frame.items():
                            display_name = obj_map.get(tag, tag)
                            xtl, ytl, xbr, ybr = map(int, box)
                            cv2.rectangle(img, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
                            (w, h), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(img, (xtl, ytl - 20), (xtl + w, ytl), (0, 255, 0), -1)
                            cv2.putText(img, display_name, (xtl, ytl - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    h, w = img.shape[:2]
                    if MAX_UAV_FRAME_LENGTH and max(h, w) > MAX_UAV_FRAME_LENGTH:
                        scale = MAX_UAV_FRAME_LENGTH / max(h, w)
                        img = cv2.resize(
                            img,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    _, buffer = cv2.imencode('.jpg', img)
                    b64_str = base64.b64encode(buffer).decode('utf-8')
                    all_frames.append(b64_str)
                    extracted_count += 1
                
                mapping_desc = f"Frames {start_f}-{end_f} from Video {v_idx} (sampled {extracted_count} images)"
                frame_mapping_text_parts.append(mapping_desc)

            return self._call_vlm_for_perception(all_frames, frame_mapping_text_parts, focus_prompt)

        except Exception as e:
            logger.error(f"    [TOOL-UAV] ❌ Failed: {e}", exc_info=True)
            return {"result": f"Error during UAV observation: {e}"}

    def _active_perception_standard(self, observation_targets: List[Dict[str, Any]], focus_prompt: str) -> Dict[str, Any]:
        """Standard video path: time-based segments via video_processor, no bbox."""
        logger.info(f"    [TOOL-STD] 🎬 Processing {len(observation_targets)} targets (Time-based)")
        
        all_frames = []
        frame_mapping_text_parts = []
        current_frame_index = 0
        skim_used = False
        
        try:
            for target in observation_targets:
                v_idx = target.get("video_index")
                if v_idx is None: return {"result": "Error: Missing video_index"}
                
                real_idx = v_idx - 1
                if real_idx < 0 or real_idx >= len(self.video_contexts):
                    return {"result": f"Error: Video index {v_idx} out of range."}
                
                context = self.video_contexts[real_idx]
                video_path = context.get("path")
                clip_begin_offset = context.get("clip_begin", 0.0)
                duration = context.get("duration_seconds", 0)

                start_t = target.get("start_time", 0)
                end_t = target.get("end_time", duration)
                # Safety net: if model blindly copied a template default (e.g. 14s)
                # for a long video, auto-expand to full duration. This only fires
                # when the requested window covers <10% of the clip and starts at 0.
                is_precise_recheck = (
                    "MODE=layer2" in str(focus_prompt)
                    or "MODE=focus" in str(focus_prompt)
                    or self._is_fsa_task()
                )
                is_executor_chunk = bool(target.get("_executor_chunk"))
                if (start_t == 0 and duration > 0
                        and end_t is not None and end_t < duration * 0.1
                        and not is_executor_chunk
                        and not is_precise_recheck):
                    logger.warning(
                        "    [SAFETY] end_time=%.1fs covers only %.1f%% of %.1fs clip — "
                        "auto-expanding to full duration (likely template copy-paste)",
                        end_t, (end_t / duration) * 100, duration,
                    )
                    end_t = duration
                num_f = target.get("num_frames", 32)
                
                abs_start = start_t + clip_begin_offset
                abs_end = end_t + clip_begin_offset
                use_skim = self._layer1_skim_enabled(focus_prompt)
                if use_skim:
                    frame_groups, mapping_tables, _, _, _, _ = video_processor.process_video(
                        input_path=video_path,
                        n_frames=num_f,
                        intervals=[(abs_start, abs_end)],
                        max_length=MAX_FRAME_LENGTH,
                        encode=False
                    )
                else:
                    frame_groups, mapping_tables, _, _, _, _ = video_processor.process_video(
                        input_path=video_path,
                        n_frames=num_f,
                        intervals=[(abs_start, abs_end)],
                        max_length=MAX_FRAME_LENGTH,
                        encode=True
                    )
                
                if frame_groups and frame_groups[0]:
                    mapping_table = mapping_tables[0] if mapping_tables else {}
                    if use_skim:
                        raw_frames = list(frame_groups[0])
                        selected, skim_strategy = self._select_layer1_skim_frames(raw_frames, mapping_table)
                        frames = [encoded for _, _, encoded in selected]
                        if not frames:
                            continue
                        skim_used = True
                        selected_positions = [idx + 1 for idx, _, _ in selected]
                        selected_source_frames = [source_idx for _, source_idx, _ in selected]
                        fps = float(context.get("fps", 0) or 0)
                        if fps > 0:
                            selected_times = [
                                max(0.0, (source_idx / fps) - clip_begin_offset)
                                for source_idx in selected_source_frames
                            ]
                            selected_time_text = ", local_times=" + ",".join(f"{t:.1f}s" for t in selected_times)
                        else:
                            selected_time_text = ""
                        logger.info(
                            "    [LAYER1-SKIM] Video %d %.1f-%.1fs: selected %d/%d representative frames (%s), positions=%s",
                            v_idx,
                            start_t,
                            end_t,
                            len(frames),
                            len(raw_frames),
                            skim_strategy,
                            selected_positions,
                        )
                    else:
                        frames = frame_groups[0]
                    all_frames.extend(frames)
                    
                    s_idx = current_frame_index + 1
                    e_idx = current_frame_index + len(frames)
                    if use_skim:
                        mapping_line = (
                            f"Frames {s_idx}-{e_idx} are Layer1 skim representatives from Video {v_idx} "
                            f"({start_t:.1f}s to {end_t:.1f}s); original sampled positions={selected_positions}, "
                            f"source_frame_ids={selected_source_frames}{selected_time_text}"
                        )
                    else:
                        mapping_line = f"Frames {s_idx}-{e_idx} are from Video {v_idx} ({start_t:.1f}s to {end_t:.1f}s)"
                    frame_mapping_text_parts.append(mapping_line)
                    current_frame_index += len(frames)
	            
            if skim_used:
                focus_prompt = (
                    f"{focus_prompt}\n\n[Layer1 Skim Notice]\n"
                    "The frames are representative skim frames selected inside each chunk (centroid/anomaly style), "
                    "not dense temporal coverage. Use them only for coarse scene/plot relevance, candidate chunk selection, "
                    "and obvious visible clues. Mark details as UNCERTAIN rather than absent when not visible in the skim frames. "
                    "Return compact rows per mapped chunk and suggest which chunk windows need Layer2 detail."
                )
            return self._call_vlm_for_perception(all_frames, frame_mapping_text_parts, focus_prompt)

        except Exception as e:
            logger.error(f"    [TOOL-STD] ❌ Failed: {e}", exc_info=True)
            return {"result": f"Error during video observation: {e}"}

    def _call_vlm_for_perception(self, frames: List[str], mapping_texts: List[str], prompt: str) -> Dict[str, Any]:
        """Calls tool VLM with frames and mapping info; returns observation text."""
        if not frames:
            return {"result": "Error: No frames extracted."}

        frame_mapping_text = "\n".join(mapping_texts)
        full_visual_prompt = f"""[Frame Mapping Information]
{frame_mapping_text}

Important: frame numbers in this message are artificial positions in the combined input sent to you. When frames come from different video_index values, their order here does NOT imply chronological order between those videos. Use the mapping only to identify which frames belong to which video and local window. Cross-video ordering must be inferred only from visible object states, actions, transfers, residues, and other visual evidence inside each mapped video.

[Visual Question]
{prompt}

Please carefully observe the frames and answer the question."""

        logger.info(f"    [TOOL] 👁️ Calling VLM with {len(frames)} frames...")
        
        vlm_response = self.agent.forward(
            task='active_perception',
            frames=frames,
            prompt=full_visual_prompt,
            return_prompt=True
        )
        
        obs_text = vlm_response.get("result", "")
        obs_text = self._strip_think_from_result(obs_text)
        
        return {"result": obs_text, "prompt": vlm_response.get("prompt")}

    # DISABLED: GPU driver mismatch causes Whisper load_model to hang
    # def _proxy_get_caption(self, video_index: int, start_time: float = None, end_time: float = None) -> Dict[str, Any]:
    #     """Generates captions for the given video segment via Whisper (on-demand)."""
    #     logger.info(f"    [TOOL] 📝 Get Caption (on-demand Whisper): Video {video_index}, Time: {start_time}-{end_time}")
    #     try:
    #         if video_index is None:
    #             return {"result": "Error: video_index is required."}
    #         real_video_index = video_index - 1
    #         if real_video_index < 0 or real_video_index >= len(self.video_contexts):
    #             return {"result": f"Error: Video index {video_index} out of range."}
    #         context = self.video_contexts[real_video_index]
    #         ctx_type = context.get("type", "video_file")
    #         if ctx_type == "image_folder":
    #             return {"result": "Error: Caption generation is not available for image folder input (no video file)."}
    #         video_path = context.get("path")
    #         clip_begin = context.get("clip_begin", 0.0)
    #         duration_seconds = context.get("duration_seconds", 0.0)
    #         if not video_path:
    #             return {"result": f"Error: No video path for Video {video_index}."}
    #         abs_start = clip_begin + (start_time if start_time is not None else 0.0)
    #         abs_end = clip_begin + (end_time if end_time is not None else duration_seconds)
    #         abs_end = min(abs_end, clip_begin + duration_seconds)
    #         if abs_start >= abs_end:
    #             return {"result": f"Error: Invalid time range for Video {video_index}."}
    #         caption_data = generate_caption_for_segment(video_path, abs_start, abs_end)
    #         if not caption_data:
    #             return {"result": f"No speech detected in Video {video_index} between {abs_start:.1f}s and {abs_end:.1f}s (or generation failed)."}
    #         result_text = f"Caption for Video {video_index} ({abs_start:.1f}s - {abs_end:.1f}s):\n"
    #         for seg in caption_data:
    #             result_text += f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}\n"
    #         return {"result": result_text}
    #     except Exception as e:
    #         logger.error(f"    [TOOL] ❌ Get caption failed: {e}", exc_info=True)
    #         return {"result": f"Error: {e}"}
    
    @staticmethod
    def _strip_think_from_result(result_str: str) -> str:
        if not isinstance(result_str, str): return result_str
        if '</think>' in result_str:
            parts = result_str.rsplit('</think>', 1)
            if len(parts) == 2: return parts[1].strip()
        return result_str

    def _execute_tool_call(
        self,
        call: Dict[str, Any],
        record_history: bool = True,
        return_log: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Runs one tool call and appends result to tool history."""
        tool_name = call.get("tool_name")
        tool_args = call.get("arguments", {}) or {}
        call_id = call.get("id", f"c_{datetime.now().microsecond}")

        tool_log_entry = {
            "turn": self.current_turn,
            "call_id": call_id,
            "tool_name": tool_name,
            "inputs": tool_args,
            "timestamp_start": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if tool_name not in self.tool_mapping:
            error_msg = f"Tool '{tool_name}' does not exist."
            tool_log_entry["error"] = error_msg
            if record_history:
                self._tool_history.append(tool_log_entry)
            result = {"call_id": call_id, "tool_name": tool_name, "error": error_msg}
            return (result, tool_log_entry) if return_log else result

        try:
            tool_function = self.tool_mapping[tool_name]
            tool_result_obj = tool_function(**tool_args)

            if isinstance(tool_result_obj, dict) and "result" in tool_result_obj:
                result_payload = tool_result_obj.get("result")
                call_prompt = tool_result_obj.get("prompt")
            else:
                result_payload = tool_result_obj
                call_prompt = getattr(self.agent, "last_tool_prompt", None)

            tool_log_entry["actual_prompt_sent"] = call_prompt
            tool_log_entry["output_raw"] = result_payload
            tool_log_entry["timestamp_end"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if record_history:
                self._tool_history.append(tool_log_entry)

            if isinstance(result_payload, str):
                result_str = result_payload
            else:
                result_str = json.dumps(result_payload, ensure_ascii=False)

            result = {
                "call_id": call_id,
                "tool_name": tool_name,
                "result": result_str
            }
            return (result, tool_log_entry) if return_log else result

        except Exception as e:
            logger.error(f"    [EXECUTOR] Tool execution failed ({tool_name}): {e}", exc_info=True)
            tool_log_entry["error"] = str(e)
            tool_log_entry["timestamp_end"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if record_history:
                self._tool_history.append(tool_log_entry)
            result = {
                "call_id": call_id,
                "tool_name": tool_name,
                "error": f"{e}"
            }
            return (result, tool_log_entry) if return_log else result

    def _parse_llm_response(self, response_str: str) -> Dict[str, Any]:
        """Parses master LLM response into JSON (handles markdown/think tags)."""
        response_str = response_str.strip()
        try:
            data = json.loads(response_str)
            if 'action' in data: return data
        except json.JSONDecodeError: pass

        match_md = re.search(r'```(?:json)?\s*(.*?)\s*```', response_str, re.DOTALL | re.IGNORECASE)
        if match_md:
            try:
                data = json.loads(match_md.group(1).strip())
                if 'action' in data: return data
            except: pass

        try:
            parts = response_str.rsplit('</think>', 1)
            if len(parts) == 2:
                json_part = parts[1].strip()
                match_md_2 = re.search(r'```(?:json)?\s*(.*?)\s*```', json_part, re.DOTALL | re.IGNORECASE)
                if match_md_2: json_part = match_md_2.group(1).strip()
                idx = json_part.find('{')
                if idx != -1:
                    data = json.loads(json_part[idx:])
                    if 'action' in data: return data
        except: pass
        
        try:
            p1 = response_str.find('{')
            p2 = response_str.rfind('}')
            if p1 != -1 and p2 != -1:
                data = json.loads(response_str[p1:p2+1])
                if 'action' in data: return data
        except: pass

        # Fallback: model output multiple concatenated JSONs like {...}{...}{...}
        # Extract first complete JSON by brace counting
        try:
            start = response_str.find('{')
            if start != -1:
                depth = 0
                for i in range(start, len(response_str)):
                    if response_str[i] == '{':
                        depth += 1
                    elif response_str[i] == '}':
                        depth -= 1
                        if depth == 0:
                            first_json = response_str[start:i+1]
                            data = json.loads(first_json)
                            if 'action' in data:
                                return data
                            break
        except: pass

        raise ValueError(f"LLM response is not valid JSON.")

    def _is_pss_sort_task(self) -> bool:
        prompt_name = getattr(self, "_master_prompt_name", "").lower()
        query = (self._initial_query or "").lower()
        return (
            "pss" in prompt_name
            or "chronological order" in query
            or "shuffled segments" in query
        )

    def _is_fsa_task(self) -> bool:
        prompt_name = getattr(self, "_master_prompt_name", "").lower()
        query = (self._initial_query or "").lower()
        return (
            "fsa" in prompt_name
            or "functionally equivalent" in query
            or "reference segment" in query
        )

    def _is_cc_task(self) -> bool:
        prompt_name = getattr(self, "_master_prompt_name", "").lower()
        query = (self._initial_query or "").lower()
        return (
            prompt_name in {"cc", "master_cc.prompt"}
            or self.prompt_config.get("task_code", "").lower() == "cc"
            or "which video" in query
            or "possible_answers" in (self._initial_query or "")
        )

    def _is_unified_prompt_task(self) -> bool:
        return bool(self.prompt_config.get("master_base") and self.prompt_config.get("task_contract"))

    def _observation_stage_state(self) -> Tuple[bool, bool, bool]:
        """Returns whether actual visual tool calls include Layer1, Layer2, and focus."""
        has_layer1 = False
        has_layer2 = False
        has_focus = False

        for entry in self._tool_history:
            if entry.get("tool_name") != "active_perception":
                continue
            inputs = entry.get("inputs")
            if not isinstance(inputs, dict):
                continue
            focus_prompt = str(inputs.get("focus_prompt", ""))
            if "MODE=layer1_caption" in focus_prompt or "MODE=layer1" in focus_prompt:
                has_layer1 = True
            if "MODE=layer2_state_scan" in focus_prompt or "MODE=layer2" in focus_prompt:
                has_layer2 = True
            if "MODE=focus" in focus_prompt:
                has_focus = True

        return has_layer1, has_layer2, has_focus

    def _pss_layer2_covered_indices(self):
        covered = set()
        for entry in self._tool_history:
            if entry.get("tool_name") != "active_perception":
                continue
            inputs = entry.get("inputs")
            if not isinstance(inputs, dict):
                continue
            focus_prompt = str(inputs.get("focus_prompt", ""))
            if "MODE=layer2_state_scan" not in focus_prompt:
                continue
            targets = inputs.get("observation_targets", [])
            if isinstance(targets, dict):
                targets = [targets]
            if not isinstance(targets, list):
                continue
            for target in targets:
                if not isinstance(target, dict):
                    continue
                try:
                    covered.add(int(target.get("video_index")))
                except (TypeError, ValueError):
                    continue
        return covered

    @staticmethod
    def _extract_numeric_confidence(*values: Any) -> Optional[float]:
        def normalize_score(raw: Any) -> Optional[float]:
            try:
                score = float(raw)
            except (TypeError, ValueError):
                return None
            if 0 <= score <= 1:
                score *= 10
            if 1 <= score <= 10:
                return score
            return None

        def from_text(text: str) -> List[float]:
            scores: List[float] = []
            patterns = [
                r"\b(?:confidence_score|answer_confidence|final_confidence|preliminary_confidence|order_confidence|confidence)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/10)?",
                r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b.{0,30}\bconfidence\b",
                r"\bconfidence\b.{0,30}\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                    score = normalize_score(match.group(1))
                    if score is not None:
                        scores.append(score)
            return scores

        def walk(value: Any) -> List[float]:
            if value is None:
                return []
            if isinstance(value, (int, float)):
                score = normalize_score(value)
                return [score] if score is not None else []
            if isinstance(value, str):
                return from_text(value)
            if isinstance(value, dict):
                scores: List[float] = []
                confidence_key = re.compile(
                    r"(?:^|_)(?:confidence|confidence_score|answer_confidence|final_confidence|preliminary_confidence|order_confidence)(?:$|_)",
                    re.IGNORECASE,
                )
                for key, child in value.items():
                    if confidence_key.search(str(key)):
                        if isinstance(child, (int, float)):
                            score = normalize_score(child)
                            if score is not None:
                                scores.append(score)
                        elif isinstance(child, str):
                            direct = normalize_score(child.strip().split("/")[0])
                            if direct is not None:
                                scores.append(direct)
                            scores.extend(from_text(child))
                    scores.extend(walk(child))
                return scores
            if isinstance(value, (list, tuple, set)):
                scores: List[float] = []
                for child in value:
                    scores.extend(walk(child))
                return scores
            return []

        scores: List[float] = []
        for value in values:
            scores.extend(walk(value))
        return max(scores) if scores else None

    def _validate_minimum_stages_before_answer(self, thought: Any = None, context_update: Any = None) -> Tuple[bool, str]:
        if not (self._is_unified_prompt_task() or self._is_pss_sort_task()):
            return True, ""

        has_layer1, has_layer2, has_focus = self._observation_stage_state()
        if self._is_pss_sort_task():
            # Require Layer1 as before
            if not has_layer1:
                return False, (
                    "Do not answer yet. PSS requires Layer1 before `answer`. "
                    "Missing: Layer1 segment scan. "
                    "Call `batch_observe` with `MODE=layer1` for every segment."
                )

            # Layer2 must cover every segment (full state-transition graph verification).
            # But each segment should be observed only at its most informative chunk,
            # with a narrow start_time/end_time and a concrete per-object checklist.
            expected_indices = set(range(1, len(self.video_contexts) + 1))
            covered_layer2 = self._pss_layer2_covered_indices()
            missing_layer2_indices = sorted(expected_indices - covered_layer2)

            if not has_layer2:
                return False, (
                    "Do not answer yet. PSS requires Layer2 before `answer`. "
                    "Missing: Layer2 discriminative evidence reinspection. "
                    "Layer2 re-observes shared evidence (objects/regions/events) that Layer1 "
                    "found recurring across segments. For each segment, ask the VLM to report "
                    "the actual visible state of these evidence items. "
                    "Do NOT pass your hypothesized order to the VLM. "
                    "Use `batch_observe` with `MODE=layer2_state_scan`."
                )

            if missing_layer2_indices:
                return False, (
                    "Do not answer yet. Layer2 must cover every segment. "
                    f"Missing Layer2 for segment(s): {missing_layer2_indices}. "
                    "For each missing segment, re-observe the linked evidence items discovered "
                    "in Layer1. Ask the VLM to report actual visible states — do NOT ask it "
                    "to confirm or verify your hypothesis."
                )

            # All segments covered by Layer2. Check confidence.
            confidence = self._extract_numeric_confidence(thought, context_update)
            if confidence is not None and confidence >= 7:
                return True, ""

            confidence_text = f"{confidence:.1f}/10" if confidence is not None else "missing (include `order_confidence: N` as an integer key in `context_update`, e.g. `\"order_confidence\": 8`)"
            return False, (
                "All segments are covered by Layer2, but "
                f"order_confidence ({confidence_text}) is still too low (need >=7). "
                "Put `order_confidence` (integer 1-10) as a key in `context_update`. "
                "Check `contradictions` — if any Layer2 evidence conflicts "
                "with your hypothesized state-transition graph, downgrade confidence and "
                "run focus on the conflicting relation. "
                # DISABLED: GPU driver mismatch causes Whisper load_model to hang
                # "`get_caption` if narration may help."
            )

        if has_layer1:
            if not has_layer2 and not has_focus:
                confidence = self._extract_numeric_confidence(thought, context_update)
                if confidence is None or confidence < 8:
                    confidence_text = "missing" if confidence is None else f"{confidence:.1f}/10"
                    return (
                        False,
                        "Do not answer yet. A direct answer after Layer1 requires explicit `confidence_score >= 8/10` "
                        "plus decisive evidence and rejected close alternatives. "
                        f"Current confidence: {confidence_text}. "
                        "Next, first analyze the Layer1 chunk evidence, then call `batch_observe` or `observe` with `MODE=layer2` only on the plausible candidates, close negatives, or decisive windows. "
                        "Do not repeat the Layer1 full scan for convenience; if every video/chunk truly remains answer-relevant, state a per-video/chunk reason in `thought` and use a specific checklist. "
                        "Carry forward `preliminary_answer`, `preliminary_confidence`, and `blocking_uncertainties` in `context_update`."
                    )
            return True, ""

        next_action = (
            "First call `batch_observe` or `observe` with `MODE=layer1` to collect visual evidence. "
            "After Layer1, decide yourself whether the evidence is sufficient to answer or whether Layer2/focus is needed."
        )

        message = (
            "Do not answer yet. This prompt requires at least one Layer1 visual stage before `answer`. "
            "Missing: Layer1 visual scan. "
            f"{next_action}"
        )
        return False, message

    @staticmethod
    def _extract_numeric_order(answer: Any) -> List[int]:
        if isinstance(answer, dict):
            answer = answer.get("final_answer") or answer.get("content") or ""
        if isinstance(answer, list):
            answer = "->".join(str(item) for item in answer)
        return [int(num) for num in re.findall(r"\d+", str(answer))]

    def _validate_pss_final_answer(self, answer: Any, response: Any = None) -> Tuple[bool, str]:
        expected_count = len(self.video_contexts)
        if expected_count <= 0:
            return True, ""

        order = self._extract_numeric_order(answer)
        expected = list(range(1, expected_count + 1))
        missing = [idx for idx in expected if idx not in order]
        duplicates = sorted({idx for idx in order if order.count(idx) > 1})
        out_of_range = [idx for idx in order if idx < 1 or idx > expected_count]

        if len(order) == expected_count and not missing and not duplicates and not out_of_range:
            confidence = self._extract_numeric_confidence(response)
            if confidence is None or confidence < 8:
                confidence_text = "missing" if confidence is None else f"{confidence:.1f}/10"
                return (
                    False,
                    "Do not answer yet. PSS final answers require one overall `confidence_score >= 8/10` "
                    "for the complete order. This is not per-edge scoring; it is the model's conservative "
                    "confidence that the whole sequence is correct after Layer1 and full Layer2. "
                    f"Current confidence: {confidence_text}. "
                    "If any critical relation, conflicting object chain, or same-role tie remains unresolved, "
                    # DISABLED: GPU driver mismatch causes Whisper load_model to hang
                    # "run `get_caption` or neutral `MODE=focus` on the smallest ambiguous set, then answer only "
                    "run neutral `MODE=focus` on the smallest ambiguous set, then answer only "
                    "when the overall confidence reaches 8/10."
                )
            return True, ""

        issues = []
        if not order:
            issues.append("no video indices were found")
        if missing:
            issues.append(f"missing video indices: {missing}")
        if duplicates:
            issues.append(f"duplicate video indices: {duplicates}")
        if out_of_range:
            issues.append(f"out-of-range video indices: {out_of_range}")
        if len(order) != expected_count:
            issues.append(f"found {len(order)} indices, expected {expected_count}")

        message = (
            f"Invalid PSS final_answer {answer!r}: {'; '.join(issues)}. "
            f"The answer must contain exactly these video indices once: {expected}. "
            "Re-read the prior tool results, especially any dependency graph, orphan clip, "
            "or parallel preparation step. Produce a corrected complete linear order, or call "
            "observe if a missing video's placement is still unsupported."
        )
        return False, message

    @staticmethod
    def _extract_interval_answer(answer: Any) -> List[float]:
        if isinstance(answer, dict):
            answer = answer.get("final_answer") or answer.get("content") or ""
        if isinstance(answer, (list, tuple)):
            nums = []
            for item in answer:
                try:
                    nums.append(float(item))
                except (TypeError, ValueError):
                    pass
            return nums
        return [float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", str(answer))]

    def _validate_fsa_final_answer(self, answer: Any) -> Tuple[bool, str]:
        nums = self._extract_interval_answer(answer)
        duration = 0.0
        if len(self.video_contexts) >= 2:
            try:
                duration = float(self.video_contexts[1].get("duration_seconds", 0) or 0)
            except (TypeError, ValueError):
                duration = 0.0

        if len(nums) == 2:
            start, end = nums
            if start > end:
                start, end = end, start
            if start < 0:
                return False, f"start time {start} is negative"
            if end <= start:
                return False, f"end time {end} must be greater than start time {start}"
            if duration > 0 and start > duration:
                return False, f"start time {start} exceeds Video 2 duration {duration:.1f}s"
            if duration > 0 and end > duration + 1.0:
                return False, f"end time {end} exceeds Video 2 duration {duration:.1f}s"
            return True, ""

        message = (
            f"Invalid FSA final_answer {answer!r}. FSA always requires exactly one numeric interval "
            "`[start, end]` in Video 2, even if the visual match is imperfect. Do not output `[]`, "
            "`None`, option text, or a prose-only answer. Re-read the Layer1/Layer2 candidate windows, "
            "choose the best functional match in Video 2, and return only a JSON list like `[82, 95]`. "
            "Functional equivalence means the same role/boundary in the procedure, not necessarily the same ingredient."
        )
        return False, message

    def _validate_cc_final_answer(self, answer: Any, thought: Any) -> Tuple[bool, str]:
        text_answer = str(answer).strip().upper()
        option_count = 0
        query = self._initial_query or ""
        match = re.search(r'"possible_answers"\s*:\s*\[(.*?)\]', query, re.DOTALL)
        if match:
            option_count = len(re.findall(r'"[A-Z]\.', match.group(1)))
        if option_count <= 0:
            option_count = 4

        valid_letters = [chr(ord("A") + i) for i in range(option_count)]
        if text_answer not in valid_letters:
            return (
                False,
                f"Invalid CC final_answer {answer!r}. Return exactly one uppercase option letter from {valid_letters}.",
            )

        thought_text = str(thought or "").lower()
        ambiguous_unique = (
            "also unique" in thought_text
            or re.search(r"\b(other|both|multiple|several|more than one)\b.{0,80}\bunique\b", thought_text)
            or re.search(r"\bunique\b.{0,80}\b(other|both|multiple|several|more than one)\b", thought_text)
        )
        if ambiguous_unique:
            return (
                False,
                "Do not answer yet. Your reasoning says multiple candidate options/tools are unique. "
                "For CC, first resolve the conflict with an option-by-video presence matrix and, if needed, "
                "a focused re-check of the conflicting candidates. Single-video observations may confirm "
                "only that video's frames; do not infer other-video absence from them."
            )

        return True, ""

    def _normalize_chunk_reference_targets(self, targets: Any) -> Any:
        """Correct off-by-one windows when Master cites a prior chunk label.

        Chunk labels printed by the executor are 1-based, e.g. [Chunk 2/7].
        Only targets whose focus_prompt explicitly mentions "Chunk k/n" are
        touched; ordinary hand-computed time windows remain unchanged.
        """
        if not isinstance(targets, list):
            return targets

        normalized = copy.deepcopy(targets)
        for target in normalized:
            if not isinstance(target, dict):
                continue
            focus_prompt = str(target.get("focus_prompt", ""))
            match = re.search(r"\bChunk\s+(\d+)\s*/\s*(\d+)\b(?:\s*\(([^)]*)\))?", focus_prompt, re.IGNORECASE)
            if not match:
                continue

            try:
                chunk_idx = int(match.group(1))
                chunk_count = int(match.group(2))
                video_idx = int(target.get("video_index"))
            except (TypeError, ValueError):
                continue
            if chunk_idx < 1 or chunk_count < 1 or chunk_idx > chunk_count:
                continue
            if video_idx < 1 or video_idx > len(self.video_contexts):
                continue

            explicit_range = match.group(3) or ""
            range_nums = [float(num) for num in re.findall(r"\d+(?:\.\d+)?", explicit_range)]
            if len(range_nums) >= 2:
                expected_start, expected_end = range_nums[0], range_nums[1]
            else:
                try:
                    duration = float(self.video_contexts[video_idx - 1].get("duration_seconds", 0) or 0)
                except (TypeError, ValueError):
                    duration = 0.0
                if duration <= 0:
                    continue
                chunk_size = duration / chunk_count
                expected_start = (chunk_idx - 1) * chunk_size
                expected_end = chunk_idx * chunk_size

            try:
                current_start = float(target.get("start_time", expected_start) or 0)
                current_end = float(target.get("end_time", expected_end) or expected_end)
            except (TypeError, ValueError):
                current_start, current_end = expected_start, expected_end

            tolerance = max(2.0, (expected_end - expected_start) * 0.05)
            current_center = (current_start + current_end) / 2.0
            center_inside_expected = expected_start - tolerance <= current_center <= expected_end + tolerance
            starts_near_expected = abs(current_start - expected_start) <= tolerance
            ends_near_expected = abs(current_end - expected_end) <= tolerance
            if center_inside_expected or (starts_near_expected and ends_near_expected):
                continue

            logger.warning(
                "    [CHUNK REF] Correcting Video %d target %.1f-%.1fs to Chunk %d/%d window %.1f-%.1fs",
                video_idx,
                current_start,
                current_end,
                chunk_idx,
                chunk_count,
                expected_start,
                expected_end,
            )
            target["start_time"] = expected_start
            target["end_time"] = expected_end
            target["focus_prompt"] = (
                f"{focus_prompt}\n\n[Executor Chunk Reference Correction]\n"
                f"The focus_prompt references Chunk {chunk_idx}/{chunk_count}. Chunk labels are 1-based, "
                f"so this target was corrected to {expected_start:.1f}s-{expected_end:.1f}s for Video {video_idx}."
            )

        return normalized

    def init_agent_state(self, query: str, possible_answers: List[str], video_contexts: List[Dict[str, Any]], **kwargs):
        """Reset state for a new question; kwargs e.g. formatted_query, bbox_info for UAV."""
        self.video_contexts = video_contexts
        self._possible_answers = possible_answers
        self._initial_query = query
        self._master_history = []
        self._tool_history = []
        self.cross_context = self._new_cross_context()
        self._pss_layer1_num_frames = None
        self._oracle_debug = kwargs.get("oracle_debug")
        self.current_turn = 0

    def cleanup_agent_state(self):
        self.video_contexts = []
        self._master_history = []
        self._tool_history = []
        self._possible_answers = []
        self._initial_query = ""
        self.cross_context = self._new_cross_context()
        self._pss_layer1_num_frames = None
        self._oracle_debug = None
        self.current_turn = 0

    def run_agent_loop(self, max_turns=100) -> Tuple[str, List[Dict[str, Any]]]:
        """Runs ReAct loop until answer or max_turns."""
        logger.info("🚀 Starting Master Agent Loop...")
        final_answer = "Error: Loop ended unexpectedly."
        
        try:
            video_info_list = []
            for i, ctx in enumerate(self.video_contexts):
                info = {"video_index": i + 1}
                if ctx.get("type") == "image_folder":
                    info["total_frames"] = ctx.get("total_frames")
                else:
                    info["duration_seconds"] = ctx.get("duration_seconds", 0)
                    
                video_info_list.append(info)
            
            system_content = self.master_prompt_template
            messages = [{"role": "system", "content": system_content}]
            
            has_image_folders = any(ctx.get("type") == "image_folder" for ctx in self.video_contexts)
            hard_rule = (
                "For image-folder/UAV observations, use frame indices: set start_frame=0 and "
                "end_frame=total_frames-1 for a full-folder scan. Do not use seconds, and do not "
                "treat 0-10 as a full scan unless total_frames is 11."
                if has_image_folders else
                "For any full-video observation, set start_time=0 and end_time to the video's "
                "duration_seconds value (provided above). Do NOT copy end_time from prompt "
                "examples — use the actual duration of each video."
            )

            user_payload = {
                "query": self._initial_query,
                "videos": video_info_list,
                "hard_rule": hard_rule
            }
            if self._possible_answers:
                user_payload["possible_answers"] = self._possible_answers
            if self._oracle_debug:
                user_payload["oracle_debug"] = self._oracle_debug
                logger.warning(
                    "🧪 ORACLE DEBUG MODE is enabled for this question. Ground-truth answer is present in master input."
                )
            
            user_content_items = [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]
            messages.append({"role": "user", "content": user_content_items})
            
            self._master_history = list(messages)

            # Resume mode: if history was pre-loaded (e.g. by DistillAgent correction),
            # keep it instead of rebuilding from scratch.
            if hasattr(self, '_resume_history') and self._resume_history:
                logger.info("📋 Resuming from pre-loaded history (%d messages)", len(self._resume_history))
                self._master_history = list(self._resume_history)
                del self._resume_history
            
            loop_start = time.time()
            for turn in range(1, max_turns + 1):
                self.current_turn = turn
                turn_start = time.time()
                logger.info(f"🔄 [TURN {turn}] Master Thinking... (elapsed since loop start: {turn_start - loop_start:.1f}s)")
                
                max_retries = 3
                valid_response_dict = None
                raw_response_str_for_history = None
                last_failed_response = None

                for attempt in range(max_retries):
                    messages_to_send = list(self._master_history)
                    cross_context_message = self._build_cross_context_message()
                    if cross_context_message:
                        messages_to_send.append(cross_context_message)
                    if attempt > 0:
                        logger.info(f"  Attempt {attempt+1}: Sending format reminder...")
                        gentle_reminder = {
                            "role": "user",
                            "content": (
                                "Please ensure your reply is valid JSON and output only one JSON object per turn. "
                                "The top-level object must include an `action` field with one of: "
                                "`batch_observe`, `observe`, or `answer`. For `batch_observe`, put items under `batch`; "
                                "for other tool use, put tool arguments under `params`; "
                                "put memory fields only under `context_update`.\n"
                            )
                        }
                        messages_to_send.append(gentle_reminder)

                    try:
                        llm_start = time.time()
                        llm_response_str = self.agent.forward(task='master', messages=messages_to_send)
                        last_failed_response = llm_response_str
                        llm_elapsed = time.time() - llm_start
                    except Exception as e:
                        logger.error(f"  ❌ API Network/Server Error on attempt {attempt+1}: {e}")
                        time.sleep(2)
                        continue
                    
                    try:
                        parsed_response = self._parse_llm_response(llm_response_str)
                        if "action" not in parsed_response:
                            raise ValueError("JSON missing 'action' field")

                        valid_response_dict = parsed_response
                        raw_response_str_for_history = llm_response_str
                        break 
                    except Exception as e:
                        logger.warning(f"  ⚠️ Format issue on attempt {attempt+1}. Retrying cleanly... ({e})")
                        continue

                if not valid_response_dict:
                    logger.error(f"❌ Max retries reached on Turn {turn}. Moving to termination.")
                    if last_failed_response:
                        self._master_history.append({
                            "role": "assistant", 
                            "content": f"[FAILED RESPONSE - Invalid JSON]\n{last_failed_response}"
                        })
                    break 

                cleaned_response = clean_thought_content(raw_response_str_for_history)
                self._master_history.append({"role": "assistant", "content": cleaned_response})
                
                action = valid_response_dict.get("action")
                thought = valid_response_dict.get("thought", "N/A")
                logger.info(f"  ⏱️  LLM: {llm_elapsed:.1f}s")
                logger.info(f"  🤖 Thought: {thought[:100]}...")
                logger.info(f"  🤖 Action:  [{action.upper()}]")

                self._merge_cross_context_update(valid_response_dict.get("context_update"))

                if action == "answer":
                    final_answer = valid_response_dict.get("final_answer", "No answer provided")
                    is_ready, readiness_message = self._validate_minimum_stages_before_answer(
                        thought,
                        valid_response_dict.get("context_update"),
                    )
                    if not is_ready:
                        logger.warning(f"  ⚠️ Rejected premature final answer: {readiness_message}")
                        self._master_history.append({
                            "role": "user",
                            "content": readiness_message,
                        })
                        continue
                    if self._is_fsa_task():
                        is_valid, validation_message = self._validate_fsa_final_answer(final_answer)
                        if not is_valid:
                            logger.warning(f"  ⚠️ Rejected invalid FSA final answer: {validation_message}")
                            self._master_history.append({
                                "role": "user",
                                "content": validation_message,
                            })
                            continue
                    if self._is_cc_task():
                        is_valid, validation_message = self._validate_cc_final_answer(final_answer, thought)
                        if not is_valid:
                            logger.warning(f"  ⚠️ Rejected invalid/ambiguous CC final answer: {validation_message}")
                            self._master_history.append({
                                "role": "user",
                                "content": validation_message,
                            })
                            continue
                    if self._is_pss_sort_task():
                        is_valid, validation_message = self._validate_pss_final_answer(final_answer, valid_response_dict)
                        if not is_valid:
                            logger.warning(f"  ⚠️ Rejected invalid PSS final answer: {validation_message}")
                            self._master_history.append({
                                "role": "user",
                                "content": validation_message,
                            })
                            continue
                    logger.info(f"🎯 Final Answer: {final_answer} (Turn total: {time.time() - turn_start:.1f}s)")
                    break
                
                elif action == "batch_observe":
                    # Master plans multiple single-video observes at once.
                    # Executor can run independent API VLM calls in parallel.
                    # All results are collected and returned as one tool response.
                    batch = valid_response_dict.get("batch", [])
                    # Fallback: try params.batch
                    if not batch:
                        batch = valid_response_dict.get("params", {}).get("batch", [])
                    if not batch:
                        err = "Missing batch in batch_observe"
                        self._master_history.append({"role": "tool", "content": [{"error": err}]})
                        continue
                    batch = self._normalize_chunk_reference_targets(batch)
                    video_occurrences = {}
                    for i_item, item in enumerate(batch):
                        vi_key = item.get("video_index", i_item + 1)
                        video_occurrences[vi_key] = video_occurrences.get(vi_key, 0) + 1

                    MAX_SECS_PER_FRAME = 6       # split if standard-video frame interval exceeds this
                    MAX_FRAMES_PER_CHUNK = 16     # max frames per VLM call
                    MAX_UAV_FRAMES_PER_SAMPLE = 20  # broad UAV scans: at least 1 image per N frames
                    MAX_UAV_CHUNK_OVERLAP_FRAMES = MAX_UAV_FRAMES_PER_SAMPLE * 2

                    tool_start = time.time()
                    batch_results = {}
                    # Pre-count total chunks for progress display
                    total_chunks = 0
                    chunk_plan = []  # (item_idx, vi, num_chunks)
                    for i, item in enumerate(batch):
                        vi = item.get("video_index", i + 1)
                        real_idx = vi - 1
                        is_image_folder = (
                            0 <= real_idx < len(self.video_contexts)
                            and self.video_contexts[real_idx].get("type") == "image_folder"
                        )
                        if is_image_folder:
                            total_f = int(self.video_contexts[real_idx].get("total_frames", 0) or 0)
                            st_f = item.get("start_frame", item.get("start_time", 0))
                            et_f = item.get("end_frame", item.get("end_time", max(0, total_f - 1)))
                            st_f = max(0, int(st_f))
                            et_f = min(max(0, total_f - 1), int(et_f))
                            window_frames = max(1, et_f - st_f + 1)
                            requested_frames = int(item.get("num_frames", 0) or 0)
                            needed_frames = int(math.ceil(window_frames / MAX_UAV_FRAMES_PER_SAMPLE))
                            total_sample_frames = max(requested_frames, needed_frames)
                            nc = max(1, int(math.ceil(total_sample_frames / MAX_FRAMES_PER_CHUNK)))
                            chunk_plan.append((i, vi, nc))
                            total_chunks += nc
                            continue

                        window_s = (item.get("end_time", item.get("duration_seconds", 10))
                                    - item.get("start_time", 0))
                        nf = item.get("num_frames", 12)
                        chunk_frames = min(nf, MAX_FRAMES_PER_CHUNK)
                        max_win = chunk_frames * MAX_SECS_PER_FRAME
                        nc = max(1, int(math.ceil(window_s / max_win))) if window_s > 0 else 1
                        chunk_plan.append((i, vi, nc))
                        total_chunks += nc

                    chunk_offsets = []
                    running_chunks = 0
                    for _, _, nc in chunk_plan:
                        chunk_offsets.append(running_chunks)
                        running_chunks += nc

                    def _run_batch_item(i: int, item: Dict[str, Any]) -> Tuple[int, str, str, List[Dict[str, Any]], int]:
                        vi = item.get("video_index", i + 1)
                        result_key = str(vi) if video_occurrences.get(vi, 0) == 1 else f"{vi}_item_{i + 1}"
                        real_idx = vi - 1
                        is_image_folder = (
                            0 <= real_idx < len(self.video_contexts)
                            and self.video_contexts[real_idx].get("type") == "image_folder"
                        )
                        if is_image_folder:
                            total_f = int(self.video_contexts[real_idx].get("total_frames", 0) or 0)
                            st_f = item.get("start_frame", item.get("start_time", 0))
                            et_f = item.get("end_frame", item.get("end_time", max(0, total_f - 1)))
                            st_f = max(0, int(st_f))
                            et_f = min(max(0, total_f - 1), int(et_f))
                            fp = item.get("focus_prompt", "")

                            window_frames = max(1, et_f - st_f + 1)
                            requested_frames = int(item.get("num_frames", 0) or 0)
                            needed_frames = int(math.ceil(window_frames / MAX_UAV_FRAMES_PER_SAMPLE))
                            total_sample_frames = max(requested_frames, needed_frames)
                            num_chunks = max(1, int(math.ceil(total_sample_frames / MAX_FRAMES_PER_CHUNK)))

                            if num_chunks > 1:
                                logger.info(
                                    "    [BATCH %d/%d] Video %d: frames %d-%d require %d samples "
                                    "(~1 per %d frames) → splitting into %d chunks with boundary overlap",
                                    i + 1, len(batch), vi, st_f, et_f, total_sample_frames,
                                    MAX_UAV_FRAMES_PER_SAMPLE, num_chunks,
                                )

                            chunk_results = []
                            chunk_specs = []
                            chunk_logs = []
                            core_len = max(1, int(math.ceil(window_frames / num_chunks)))
                            overlap_frames = min(MAX_UAV_CHUNK_OVERLAP_FRAMES, max(0, core_len // 2))
                            for c in range(num_chunks):
                                core_st = st_f + int(round(c * window_frames / num_chunks))
                                core_et = st_f + int(round((c + 1) * window_frames / num_chunks)) - 1
                                core_et = min(et_f, max(core_st, core_et))
                                chunk_st = max(st_f, core_st - overlap_frames if c > 0 else core_st)
                                chunk_et = core_et
                                chunk_et = min(et_f, max(chunk_st, chunk_et))
                                chunk_specs.append((chunk_st, chunk_et, core_st, core_et))
                                chunk_len = max(1, chunk_et - chunk_st + 1)
                                chunk_frames = min(
                                    MAX_FRAMES_PER_CHUNK,
                                    max(1, int(math.ceil(chunk_len / MAX_UAV_FRAMES_PER_SAMPLE))),
                                )
                                if num_chunks == 1:
                                    chunk_prompt = fp
                                else:
                                    overlap_note = ""
                                    if c > 0 and chunk_st < core_st:
                                        overlap_note = (
                                            f" Frames {chunk_st}-{core_st - 1} are overlap context from the previous part. "
                                            "Use them only to link identities/trajectories across the boundary; do not start a new count merely because an object appears in this overlap."
                                        )
                                    previous_note = ""
                                    if c > 0 and chunk_results:
                                        prev_summary = str(chunk_results[-1])
                                        if len(prev_summary) > 1600:
                                            prev_summary = "..." + prev_summary[-1600:]
                                        previous_note = (
                                            "\n\n[Previous chunk tentative summary]\n"
                                            f"{prev_summary}\n"
                                            "Treat this as tentative continuity context; correct it if the current visual evidence disagrees."
                                        )
                                    chunk_prompt = (
                                        f"{fp}\n\n[Chunk stitching instruction]\n"
                                        f"This is part {c+1}/{num_chunks}. Core interval: frames {core_st}-{core_et}. "
                                        f"Observed window: frames {chunk_st}-{chunk_et}.{overlap_note} "
                                        "For counting tasks, output local candidates as a ledger and include boundary_state_start/boundary_state_end: "
                                        "subject location, nearby candidate objects, continuing objects from adjacent chunks, and duplicate-risk candidates."
                                        f"{previous_note}"
                                    )
                                call_idx = chunk_offsets[i] + c + 1
                                logger.info(
                                    "    [CHUNK %d/%d] Video %d frame window %d-%d (core %d-%d), %d sampled frames",
                                    call_idx, total_chunks, vi, chunk_st, chunk_et, core_st, core_et, chunk_frames,
                                )
                                single_call = {
                                    "id": f"batch_{turn}_{call_idx}",
                                    "tool_name": "active_perception",
                                    "arguments": {
                                        "observation_targets": [{
                                            "video_index": vi,
                                            "start_frame": chunk_st,
                                            "end_frame": chunk_et,
                                            "num_frames": chunk_frames,
                                        }],
                                        "focus_prompt": chunk_prompt,
                                    },
                                }
                                single_result, single_log = self._execute_tool_call(
                                    single_call,
                                    record_history=False,
                                    return_log=True,
                                )
                                chunk_results.append(single_result.get("result", str(single_result)))
                                chunk_logs.append(single_log)

                            if num_chunks > 1:
                                merged_parts = []
                                for c, cr in enumerate(chunk_results):
                                    chunk_st, chunk_et, core_st, core_et = chunk_specs[c]
                                    merged_parts.append(
                                        f"[Chunk {c+1}/{num_chunks} "
                                        f"(observed frames {chunk_st}-{chunk_et}, core frames {core_st}-{core_et})]:\n{cr}"
                                    )
                                item_result = "\n\n".join(merged_parts)
                            else:
                                item_result = chunk_results[0] if chunk_results else ""
                            logger.info(
                                "    [BATCH %d/%d] Video %d done in %.1fs (%d frame chunks)",
                                i + 1, len(batch), vi, time.time() - tool_start, num_chunks,
                            )
                            return i, result_key, item_result, chunk_logs, num_chunks

                        st = item.get("start_time", 0)
                        et = item.get("end_time", item.get("duration_seconds", 10))
                        nf = item.get("num_frames", 12)
                        fp = item.get("focus_prompt", "")

                        window_s = et - st
                        chunk_frames = min(nf, MAX_FRAMES_PER_CHUNK)
                        max_window_per_chunk = chunk_frames * MAX_SECS_PER_FRAME
                        num_chunks = max(1, int(math.ceil(window_s / max_window_per_chunk))) if window_s > 0 else 1

                        if num_chunks > 1:
                            logger.info(
                                "    [BATCH %d/%d] Video %d: %.0fs → splitting into %d chunks "
                                "(%.0fs per chunk, %d frames each)",
                                i + 1, len(batch), vi, window_s, num_chunks,
                                window_s / num_chunks, chunk_frames,
                            )

                        if self._layer1_skim_enabled(fp):
                            targets = []
                            chunk_specs = []
                            for c in range(num_chunks):
                                chunk_st = st + c * (window_s / num_chunks)
                                chunk_et = st + (c + 1) * (window_s / num_chunks)
                                targets.append({
                                    "video_index": vi,
                                    "start_time": chunk_st,
                                    "end_time": chunk_et,
                                    "num_frames": chunk_frames,
                                    "_executor_chunk": True,
                                })
                                chunk_specs.append((chunk_st, chunk_et))

                            call_idx = chunk_offsets[i] + 1
                            reps_per_chunk = self._safe_int_env("AGENT_LAYER1_SKIM_FRAMES_PER_CHUNK", 2, minimum=1)
                            logger.info(
                                "    [LAYER1-SKIM] Video %d: merging %d chunks into one skim call "
                                "(sample %d frames/chunk -> up to %d representatives/chunk)",
                                vi,
                                num_chunks,
                                chunk_frames,
                                reps_per_chunk,
                            )
                            for c, (chunk_st, chunk_et) in enumerate(chunk_specs):
                                logger.info(
                                    "    [SKIM-CHUNK %d/%d] Video %d chunk %d/%d: %.0f-%.0fs, sample %d frames",
                                    call_idx + c,
                                    total_chunks,
                                    vi,
                                    c + 1,
                                    num_chunks,
                                    chunk_st,
                                    chunk_et,
                                    chunk_frames,
                                )

                            chunk_prompt = (
                                f"{fp}\n\n[Layer1 skim chunking instruction]\n"
                                f"This call contains representative frames from {num_chunks} chunks of Video {vi}. "
                                "For each mapped chunk, output compact coarse evidence and whether it is likely relevant, "
                                "uncertain, or low-value for the query/options. Do not claim a detail is absent unless it is visible enough in these skim frames. "
                                "Recommend exact chunk windows for Layer2 detail when needed."
                            )
                            single_call = {
                                "id": f"batch_{turn}_{call_idx}_skim",
                                "tool_name": "active_perception",
                                "arguments": {
                                    "observation_targets": targets,
                                    "focus_prompt": chunk_prompt,
                                },
                            }
                            single_result, single_log = self._execute_tool_call(
                                single_call,
                                record_history=False,
                                return_log=True,
                            )
                            item_result = (
                                f"[Layer1 skim over {num_chunks} chunks of Video {vi}; "
                                f"representative frames only, use Layer2 for details]:\n"
                                f"{single_result.get('result', str(single_result))}"
                            )
                            logger.info(
                                "    [BATCH %d/%d] Video %d done in %.1fs (%d chunks skimmed in one call)",
                                i + 1, len(batch), vi, time.time() - tool_start, num_chunks,
                            )
                            return i, result_key, item_result, [single_log], num_chunks

                        chunk_results = []
                        chunk_logs = []
                        for c in range(num_chunks):
                            chunk_st = st + c * (window_s / num_chunks)
                            chunk_et = st + (c + 1) * (window_s / num_chunks)
                            chunk_prompt = fp if num_chunks == 1 else (
                                f"{fp} (part {c+1}/{num_chunks}, "
                                f"time {chunk_st:.0f}s–{chunk_et:.0f}s)"
                            )
                            targets = [{
                                "video_index": vi,
                                "start_time": chunk_st,
                                "end_time": chunk_et,
                                "num_frames": chunk_frames,
                                "_executor_chunk": True,
                            }]
                            call_idx = chunk_offsets[i] + c + 1
                            logger.info(
                                "    [CHUNK %d/%d] Video %d chunk %d/%d: %.0f-%.0fs, %d frames",
                                call_idx, total_chunks,
                                vi, c + 1, num_chunks, chunk_st, chunk_et, chunk_frames,
                            )
                            single_call = {
                                "id": f"batch_{turn}_{call_idx}",
                                "tool_name": "active_perception",
                                "arguments": {
                                    "observation_targets": targets,
                                    "focus_prompt": chunk_prompt,
                                },
                            }
                            single_result, single_log = self._execute_tool_call(
                                single_call,
                                record_history=False,
                                return_log=True,
                            )
                            chunk_results.append(single_result.get("result", str(single_result)))
                            chunk_logs.append(single_log)

                        if num_chunks > 1:
                            # Merge chunk results: prefix each chunk and combine
                            merged_parts = []
                            for c, cr in enumerate(chunk_results):
                                merged_parts.append(
                                    f"[Chunk {c+1}/{num_chunks} "
                                    f"({st + c*(window_s/num_chunks):.0f}s–"
                                    f"{st + (c+1)*(window_s/num_chunks):.0f}s)]:\n{cr}"
                                )
                            item_result = "\n\n".join(merged_parts)
                        else:
                            item_result = chunk_results[0] if chunk_results else ""

                        logger.info(
                            "    [BATCH %d/%d] Video %d done in %.1fs (%d chunks)",
                            i + 1, len(batch), vi, time.time() - tool_start, num_chunks,
                        )
                        return i, result_key, item_result, chunk_logs, num_chunks

                    tool_backend = getattr(self.agent, "tool_backend", "api")
                    default_parallel = "false" if tool_backend in {"local_qwenvl", "qwenvl", "local_vlm"} else "true"
                    is_layer1_batch = any(
                        "MODE=layer1" in str(item.get("focus_prompt", ""))
                        or "MODE=layer1_caption" in str(item.get("focus_prompt", ""))
                        for item in batch
                    )
                    parallel_all_batches = (
                        os.environ.get("AGENT_BATCH_OBSERVE_PARALLEL_ALL", "false").strip().lower()
                        in {"1", "true", "yes", "on"}
                    )
                    parallel_enabled = (
                        len(batch) > 1
                        and (is_layer1_batch or parallel_all_batches)
                        and os.environ.get("AGENT_BATCH_OBSERVE_PARALLEL", default_parallel).strip().lower()
                        in {"1", "true", "yes", "on"}
                    )
                    max_workers = max(1, int(os.environ.get("AGENT_BATCH_OBSERVE_WORKERS", "6")))
                    max_workers = min(max_workers, len(batch))

                    item_outputs: List[Optional[Tuple[int, str, str, List[Dict[str, Any]], int]]] = [None] * len(batch)
                    if parallel_enabled and max_workers > 1:
                        logger.info(
                            "  ⚡ Batch observe parallel enabled: %d items, %d workers (tool_backend=%s, layer1=%s)",
                            len(batch), max_workers, tool_backend, is_layer1_batch,
                        )
                        with ThreadPoolExecutor(max_workers=max_workers) as pool:
                            futures = [pool.submit(_run_batch_item, i, item) for i, item in enumerate(batch)]
                            for fut in as_completed(futures):
                                item_idx, result_key, item_result, chunk_logs, num_chunks = fut.result()
                                item_outputs[item_idx] = (item_idx, result_key, item_result, chunk_logs, num_chunks)
                    else:
                        for i, item in enumerate(batch):
                            item_outputs[i] = _run_batch_item(i, item)

                    for output in item_outputs:
                        if output is None:
                            continue
                        _, result_key, item_result, chunk_logs, _ = output
                        batch_results[result_key] = item_result
                        self._tool_history.extend(chunk_logs)

                    tool_elapsed = time.time() - tool_start
                    logger.info(f"  ⏱️  Batch Tool ({len(batch)} videos): {tool_elapsed:.1f}s, Turn total: {time.time() - turn_start:.1f}s")
                    self._master_history.append({
                        "role": "tool",
                        "content": [{"type": "text", "text": json.dumps({
                            "call_id": f"batch_{turn}",
                            "tool_name": "active_perception",
                            "result": json.dumps(batch_results, ensure_ascii=False)
                        }, ensure_ascii=False)}]
                    })

                elif action == "observe":
                    params = valid_response_dict.get("params", {})
                    targets = params.get("observation_targets", [])
                    prompt = params.get("focus_prompt", "")
                    # Fallback: if params nesting is missing, try top-level fields (local models often flatten JSON)
                    if not targets:
                        targets = valid_response_dict.get("observation_targets", [])
                    if not prompt:
                        prompt = valid_response_dict.get("focus_prompt", "")

                    if not targets:
                        err = "Missing observation_targets"
                        self._master_history.append({"role": "tool", "content": [{"error": err}]})
                        continue
                    if isinstance(targets, list):
                        normalized_targets = []
                        for target in targets:
                            if isinstance(target, dict):
                                target = dict(target)
                                target.setdefault("focus_prompt", prompt)
                            normalized_targets.append(target)
                        targets = self._normalize_chunk_reference_targets(normalized_targets)
                        prompt = targets[0].pop("focus_prompt", prompt) if len(targets) == 1 and isinstance(targets[0], dict) else prompt
                        
                    tool_call = {
                        "id": f"obs_{turn}",
                        "tool_name": "active_perception",
                        "arguments": {"observation_targets": targets, "focus_prompt": prompt}
                    }
                    tool_start = time.time()
                    obs_result = self._execute_tool_call(tool_call)
                    tool_elapsed = time.time() - tool_start
                    logger.info(f"  ⏱️  Tool: {tool_elapsed:.1f}s, Turn total: {time.time() - turn_start:.1f}s")
                    self._master_history.append({
                        "role": "tool", 
                        "content": [{"type": "text", "text": json.dumps(obs_result, ensure_ascii=False)}]
                    })
                
                # DISABLED: GPU driver mismatch causes Whisper load_model to hang
                # elif action == "get_caption":
                #     tool_start = time.time()
                #     params = valid_response_dict.get("params", {})
                #     # Fallback: model may mimic batch_observe and put targets in "batch"
                #     if "video_index" not in params:
                #         batch = valid_response_dict.get("batch", [])
                #         if batch:
                #             captions = []
                #             for item in batch:
                #                 single_params = {
                #                     "video_index": item.get("video_index"),
                #                     "start_time": item.get("start_time"),
                #                     "end_time": item.get("end_time"),
                #                 }
                #                 single_call = {
                #                     "id": f"cap_{turn}_{item.get('video_index', '?')}",
                #                     "tool_name": "get_caption",
                #                     "arguments": single_params,
                #                 }
                #                 cap_result = self._execute_tool_call(single_call)
                #                 captions.append(cap_result)
                #             cap_result = {"results": captions} if captions else {"error": "No valid batch items"}
                #         else:
                #             cap_result = {"error": "get_caption requires video_index in params or batch"}
                #             logger.warning("    [PARSER] get_caption called without video_index in params or batch")
                #     else:
                #         tool_call = {
                #             "id": f"cap_{turn}",
                #             "tool_name": "get_caption",
                #             "arguments": params
                #         }
                #         cap_result = self._execute_tool_call(tool_call)
                #     tool_elapsed = time.time() - tool_start
                #     logger.info(f"  ⏱️  Tool (caption): {tool_elapsed:.1f}s, Turn total: {time.time() - turn_start:.1f}s")
                #     self._master_history.append({
                #         "role": "tool",
                #         "content": [{"type": "text", "text": json.dumps(cap_result, ensure_ascii=False)}]
                #     })
                
                else:
                    logger.warning(f"Unknown action: {action}")
                    error_obs = {"call_id": f"err_{turn}", "tool_name": "master", "error": f"Unknown action: {action}"}
                    self._master_history.append({
                        "role": "tool", 
                        "content": [{"type": "text", "text": json.dumps(error_obs, ensure_ascii=False)}]
                    })

            return final_answer, self._master_history

        except Exception as e:
            logger.error(f"❌ Critical Error in Loop: {e}", exc_info=True)
            return f"Error: {e}", self._master_history
