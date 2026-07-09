# EPIC PSS task entry. Run from agent_system:
#   EPIC_PSS_LOG_SUFFIX=_qwenapi_myprompts \
#   python run_tasks/run_EPIC_PSS_agent.py \
#     --master-backend api --tool-backend api --prompt-dir myprompts \
#     --start 1 --end 10 --max-turns 12

import sys
import os
import base64
import cv2
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qwen_agent import QwenModel
from runtime_options import (
    apply_runtime_overrides,
    parse_runtime_args,
    resolve_prompt_config,
    runtime_summary,
)

# --- Config ---
QUESTION_FILE = "/media/data6/xuejj/Epic_Kitchen/PSS_train.json"
EPIC_FRAMES_ROOT = "/media/data6/xuejj/Epic_Kitchen/new_frames"
FPS = 30
MAX_TURNS = 20
START_FROM_QUESTION_NUM = int(os.environ.get("EPIC_PSS_START", "1"))
END_AT_QUESTION_NUM = int(os.environ.get("EPIC_PSS_END", "10"))
TASK_NAME = "EPIC_PSS_task"
DEFAULT_PROMPT_CONFIG = {"master": os.path.join(_ROOT, "prompts", "master_PSS.prompt")}
DEFAULT_MASTER_BASE = "_unified_crossvid_base_compress.prompt"

CHECKPOINT_FILE = os.path.join(_ROOT, "logs", "EPIC_PSS_checkpoint.txt")


def build_prompt_config(runtime_args):
    if runtime_args.prompt_file:
        prompt_path = runtime_args.prompt_file
        if not os.path.isabs(prompt_path):
            prompt_path = os.path.join(_ROOT, prompt_path)
        return {"master": prompt_path}

    prompt_dir = runtime_args.prompt_dir or "myprompts"
    prompt_dir_path = prompt_dir if os.path.isabs(prompt_dir) else os.path.join(_ROOT, prompt_dir)
    master_base = os.path.join(prompt_dir_path, DEFAULT_MASTER_BASE)
    task_contract = os.path.join(prompt_dir_path, "task_contracts", "PSS.prompt")
    if os.path.exists(master_base) and os.path.exists(task_contract):
        return {
            "master_base": master_base,
            "task_contract": task_contract,
            "task_code": "PSS",
            "master_name": "master_PSS.prompt",
        }
    return dict(DEFAULT_PROMPT_CONFIG)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                parts = f.read().strip().split(',')
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
    return 0, 0


def save_checkpoint(correct_count, last_abs_num):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(f"{correct_count},{last_abs_num}")


class EpicPSSExecutor:
    """
    Thin wrapper that exposes EPIC frame folders as CrossVid-like clips.
    Master and VLM both see duration_seconds + start_time/end_time; only this
    wrapper maps seconds to extracted jpg frames internally.
    """
    def __init__(self, real_executor):
        self._real = real_executor

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run_agent_loop(self, max_turns=100):
        real = self._real
        orig_tool = real.tool_mapping.get("active_perception")

        def epic_active_perception(observation_targets, focus_prompt):
            if not observation_targets:
                return {"result": "Error: No observation targets provided."}

            observation_targets, focus_prompt_local = real._normalize_pss_frame_counts(
                observation_targets, focus_prompt
            )
            focus_prompt_local = real._augment_focus_prompt_with_cross_context(focus_prompt_local)

            all_frames = []
            mapping_texts = []
            current_frame_index = 0
            skim_used = False
            try:
                for target in observation_targets:
                    v_idx = target.get("video_index")
                    if v_idx is None:
                        return {"result": "Error: Missing video_index."}
                    ri = int(v_idx) - 1
                    if ri < 0 or ri >= len(real.video_contexts):
                        return {"result": f"Error: Video index {v_idx} out of range."}
                    ctx = real.video_contexts[ri]
                    if ctx.get("type") != "epic_frame_segment":
                        if orig_tool is None:
                            return {"result": "Error: original active_perception tool missing."}
                        return orig_tool(observation_targets, focus_prompt_local)

                    image_files = ctx.get("image_files", [])
                    folder_path = ctx.get("path")
                    total_frames = len(image_files)
                    duration = float(ctx.get("duration_seconds", 0.0) or 0.0)
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
                        # Match utils.video_processor.process_video(): sample
                        # at temporal-bin midpoints, not at interval endpoints.
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
                    use_skim = real._layer1_skim_enabled(focus_prompt_local)
                    raw_frames = []
                    raw_mapping = {}
                    extracted = 0
                    for idx in indices:
                        img = cv2.imread(os.path.join(folder_path, image_files[idx]))
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
                        selected, skim_strategy = real._select_layer1_skim_frames(raw_frames, raw_mapping)
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
                        import logging
                        logging.getLogger("AgentWorkflowLogger").info(
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
                return real._call_vlm_for_perception(all_frames, mapping_texts, focus_prompt_local)
            except Exception as e:
                return {"result": f"Error during EPIC frame observation: {e}"}

        real.tool_mapping["active_perception"] = epic_active_perception
        try:
            return real.run_agent_loop(max_turns=max_turns)
        finally:
            if orig_tool is not None:
                real.tool_mapping["active_perception"] = orig_tool


if __name__ == "__main__":
    runtime_args = parse_runtime_args("Run EPIC PSS agent with selectable master/tool backends.")
    apply_runtime_overrides(runtime_args)
    PROMPT_CONFIG = build_prompt_config(runtime_args)
    max_turns = runtime_args.max_turns or MAX_TURNS

    if os.environ.get("EPIC_PSS_RESET_CHECKPOINT", "").strip().lower() in {"1", "true", "yes", "on"}:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    qwen_agent_instance = QwenModel(prompt_config=PROMPT_CONFIG)
    if qwen_agent_instance.master_backend in {"local_hf", "local", "hf"}:
        print("Preloading local MASTER before other task imports...", flush=True)
        qwen_agent_instance._load_local_master()

    import json
    import traceback
    from agent_executor import AgentExecutor
    from utils.log_utils import setup_logger
    from utils.answer_parse import parse_agent_output_sort

    if not os.path.exists(QUESTION_FILE):
        print(f"FATAL: Question file not found: {QUESTION_FILE}")
        raise SystemExit(1)
    with open(QUESTION_FILE, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)

    original_total_questions = len(all_questions)

    start_question_num = START_FROM_QUESTION_NUM
    if runtime_args.start is not None:
        start_question_num = runtime_args.start
    if start_question_num < 1:
        start_question_num = 1

    end_question_num = END_AT_QUESTION_NUM
    if runtime_args.end is not None:
        end_question_num = runtime_args.end
    if end_question_num is None:
        end_index = original_total_questions
    else:
        end_index = min(end_question_num, original_total_questions)

    saved_correct, last_done = (0, 0) if os.environ.get(
        "EPIC_PSS_IGNORE_CHECKPOINT", "").strip().lower() in {"1", "true", "yes", "on"} else load_checkpoint()
    if last_done > 0:
        resume_from = last_done + 1
        logger_early = setup_logger(TASK_NAME.replace("_task", ""), resume_from, end_index)
        logger_early.info(f"Resuming from checkpoint: completed Q1-Q{last_done}, {saved_correct} correct so far.")
    else:
        resume_from = start_question_num

    if resume_from > end_index:
        print(f"All {end_index} questions already completed. Done.")
        raise SystemExit(0)

    start_index = resume_from - 1
    questions_to_run = all_questions[start_index:end_index]
    num_questions_to_run = len(questions_to_run)

    logger = setup_logger(TASK_NAME.replace("_task", ""), resume_from, end_index)
    logger.info(f"Run Config: Task={TASK_NAME}, Q{resume_from}-Q{end_index}")
    logger.info("Runtime: %s", runtime_summary(PROMPT_CONFIG))

    real_executor = AgentExecutor(agent_instance=qwen_agent_instance, prompt_config=PROMPT_CONFIG)
    executor = EpicPSSExecutor(real_executor)
    correct_answers_count = saved_correct

    for i, question_data in enumerate(questions_to_run):
        current_question_abs_num = i + resume_from
        question_id = question_data.get('id', str(current_question_abs_num))
        log_suffix = os.environ.get("EPIC_PSS_LOG_SUFFIX", "").strip()
        log_dir = os.path.join(_ROOT, "logs", TASK_NAME, str(question_id) + log_suffix)
        os.makedirs(log_dir, exist_ok=True)

        logger.info("=" * 50)
        logger.info(f"  Testing Question {current_question_abs_num}/{original_total_questions} (ID: {question_id})")

        query = ("Review these video clips. They are shuffled segments from a single cooking video. "
                 "Please determine their correct chronological order.")
        gt_answer = question_data['answer']
        logger.info(f"Query: {query}")
        logger.info(f"Correct Answer: [{gt_answer}]")

        video_id = question_data.get("video")
        segments_dict = question_data.get("segments")
        if not video_id or not segments_dict:
            logger.error("Missing 'video' or 'segments' data.")
            save_checkpoint(correct_answers_count, current_question_abs_num)
            continue

        frames_dir = os.path.join(EPIC_FRAMES_ROOT, video_id)
        if not os.path.isdir(frames_dir):
            logger.error(f"Frames directory not found: {frames_dir}")
            save_checkpoint(correct_answers_count, current_question_abs_num)
            continue

        try:
            all_frames = sorted([
                f for f in os.listdir(frames_dir)
                if f.endswith('.jpg') and f.startswith('frame_')
            ])
            if len(all_frames) == 0:
                logger.error(f"No frames in {frames_dir}")
                save_checkpoint(correct_answers_count, current_question_abs_num)
                continue
        except Exception as e:
            logger.error(f"Error listing frames: {e}")
            save_checkpoint(correct_answers_count, current_question_abs_num)
            continue

        video_contexts = []
        seg_keys = sorted(segments_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)

        for key in seg_keys:
            seg = segments_dict[key]
            intervals = seg.get('interval', [])
            if not intervals:
                continue

            seg_start = float(intervals[0][0])
            seg_end = float(intervals[-1][1])
            seg_dur = max(0.0, seg_end - seg_start)

            # Slice frames to only this segment's range
            start_idx = int(seg_start * FPS)
            end_idx = int(seg_end * FPS)
            start_idx = max(0, min(start_idx, len(all_frames) - 1))
            end_idx = max(0, min(end_idx, len(all_frames) - 1))
            seg_frames = all_frames[start_idx:end_idx + 1]

            video_contexts.append({
                "path": frames_dir,
                "type": "epic_frame_segment",
                "fps": FPS,
                "total_frames": len(seg_frames),
                "image_files": seg_frames,
                "duration_seconds": seg_dur,
                "clip_begin": 0.0,     # segment-local: frame 0 = segment start
                "clip_end": seg_dur,
                "name_for_log": f"{video_id}_seg_{key}",
            })
            logger.info(f"  Seg {key}: [{seg_start:.1f}s-{seg_end:.1f}s] → {len(seg_frames)} frames")

        if not video_contexts:
            logger.error("Skipping due to missing segments.")
            save_checkpoint(correct_answers_count, current_question_abs_num)
            continue

        try:
            executor.init_agent_state(
                query=query,
                possible_answers=[],
                video_contexts=video_contexts
            )
            final_answer_text, final_history = executor.run_agent_loop(max_turns=max_turns)
            logger.info("Evaluating result...")
            predicted_str = parse_agent_output_sort(final_answer_text)
            logger.info(f"  - Predicted: {predicted_str}")
            logger.info(f"  - Ground Truth: {gt_answer}")
            is_correct = (predicted_str is not None and predicted_str == gt_answer)
            if is_correct:
                logger.info("  - Result: CORRECT")
                correct_answers_count += 1
            else:
                logger.info("  - Result: WRONG")
            executor.save_logs(log_dir)
        except Exception as e:
            logger.error(f"Loop Error: {e}", exc_info=True)
        finally:
            executor.cleanup_agent_state()
            save_checkpoint(correct_answers_count, current_question_abs_num)
            logger.info(f"--- End of Q{question_id} ---")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    logger.info("*" * 30)
    logger.info("   Final Summary   ")
    logger.info("*" * 30)
    accuracy = (correct_answers_count / num_questions_to_run) * 100 if num_questions_to_run > 0 else 0
    logger.info(f"Questions Run:    {num_questions_to_run}")
    logger.info(f"Correct Answers:  {correct_answers_count}")
    logger.info(f"Accuracy:         {accuracy:.2f}%")
