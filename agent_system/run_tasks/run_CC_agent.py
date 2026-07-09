# CC task entry. Run from agent_system: python run_tasks/run_CC_agent.py

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from qwen_agent import QwenModel
from runtime_options import apply_runtime_overrides, parse_runtime_args, resolve_prompt_config, runtime_summary

QUESTION_FILE = os.path.join(_ROOT, "question", "CC.json")
MAX_FRAME_LENGTH = 360
MAX_TURNS = 20
START_FROM_QUESTION_NUM = 1
END_AT_QUESTION_NUM = None
TASK_NAME = "CC_task"
DEFAULT_PROMPT_CONFIG = {"master": os.path.join(_ROOT, "prompts", "master_CC.prompt")}

# Checkpoint file for resume support
CHECKPOINT_FILE = os.path.join(_ROOT, "logs", "CC_checkpoint.txt")


def _load_cc_checkpoint():
    """Return (correct_count, last_completed_abs_num) or (0, 0) if no checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                parts = f.read().strip().split(',')
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
    return 0, 0


def _save_cc_checkpoint(correct_count, last_abs_num):
    """Save progress so we can resume after interruption."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(f"{correct_count},{last_abs_num}")


if __name__ == "__main__":
    runtime_args = parse_runtime_args("Run CC agent with selectable master/tool backends.")
    apply_runtime_overrides(runtime_args)
    PROMPT_CONFIG = resolve_prompt_config(_ROOT, "CC", DEFAULT_PROMPT_CONFIG, runtime_args)
    max_turns = runtime_args.max_turns or MAX_TURNS

    # Reset checkpoint if explicitly requested
    if os.environ.get("CC_RESET_CHECKPOINT", "").strip().lower() in {"1", "true", "yes", "on"}:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("🧹 CC checkpoint reset.")

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
    from utils import video_processor
    from utils.config import get_local_video_root, get_remote_video_base_url
    from utils.log_utils import setup_logger
    from utils.answer_parse import extract_correct_list, parse_agent_output_multi

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

    # Resume from checkpoint (skip if CC_IGNORE_CHECKPOINT is set)
    saved_correct, last_done = (0, 0) if os.environ.get("CC_IGNORE_CHECKPOINT", "").strip().lower() in {"1", "true", "yes", "on"} else _load_cc_checkpoint()
    if last_done > 0:
        resume_from = last_done + 1
        logger_early = setup_logger(TASK_NAME.replace("_task", ""), resume_from, end_index)
        logger_early.info(f"🔄 Resuming from checkpoint: completed Q1-Q{last_done}, {saved_correct} correct so far.")
    else:
        resume_from = start_question_num

    if resume_from > end_index:
        print(f"All {end_index} questions already completed. Done.")
        raise SystemExit(0)

    start_index = resume_from - 1
    questions_to_run = all_questions[start_index:end_index]
    num_questions_to_run = len(questions_to_run)

    logger = setup_logger(TASK_NAME.replace("_task", ""), resume_from, end_index)
    logger.info(f"▶️  Run Config: Task={TASK_NAME}, Q{resume_from}-Q{end_index}")
    logger.info("⚙️  Runtime: %s", runtime_summary(PROMPT_CONFIG))

    executor = AgentExecutor(agent_instance=qwen_agent_instance, prompt_config=PROMPT_CONFIG)
    correct_answers_count = saved_correct

    for i, question_data in enumerate(questions_to_run):
        current_question_abs_num = i + resume_from
        question_id = question_data.get('id', str(current_question_abs_num))
        log_suffix = os.environ.get("TASK_LOG_SUFFIX", "").strip()
        log_dir = os.path.join(_ROOT, "logs", TASK_NAME, str(question_id) + log_suffix)
        os.makedirs(log_dir, exist_ok=True)

        logger.info("==================================")
        logger.info(f"   Testing Question {current_question_abs_num}/{original_total_questions} (ID: {question_id})   ")
        query = question_data['question']
        possible_answers = question_data['options']
        logger.info(f"🎯 Query: {query}")

        logger.info("📹 Preparing video metadata for the current question...")
        video_contexts = []
        videos = question_data.get('videos', [])
        begins = question_data.get('begin', [None] * len(videos))
        ends = question_data.get('end', [None] * len(videos))
        for idx, v_name in enumerate(videos):
            begin_sec = begins[idx]
            end_sec = ends[idx]
            local_path = os.path.join(get_local_video_root(), v_name)
            remote_url = f"{get_remote_video_base_url().rstrip('/')}/{v_name}"
            video_path = local_path if os.path.exists(local_path) else remote_url
            try:
                _, _, _, original_fps, total_f, total_d = video_processor.process_video(
                    input_path=video_path, n_frames=1, intervals=[(0, 1)], max_length=MAX_FRAME_LENGTH, encode=False
                )
                actual_begin = begin_sec if begin_sec is not None else 0.0
                actual_end = end_sec if end_sec is not None else total_d
                video_contexts.append({
                    "path": video_path, "fps": original_fps, "total_frames": total_f,
                    "duration_seconds": max(0.0, actual_end - actual_begin),
                    "clip_begin": actual_begin, "clip_end": actual_end,
                })
                logger.info(f"    ✅ Video {idx+1}: [{actual_begin}s - {actual_end}s], FPS={original_fps:.2f}")
            except Exception as e:
                logger.error(f"    ❌ Error getting metadata for Video {idx+1}: {e}")
                video_contexts = []
                break
        if not video_contexts:
            logger.error("❌ Skipping due to video failure.")
            _save_cc_checkpoint(correct_answers_count, current_question_abs_num)
            continue

        try:
            executor.init_agent_state(query=query, possible_answers=possible_answers, video_contexts=video_contexts)
            final_answer_text, final_history = executor.run_agent_loop(max_turns=max_turns)
            logger.info("📊 Evaluating result...")
            predicted_list = parse_agent_output_multi(final_answer_text, len(possible_answers))
            correct_list = extract_correct_list(question_data.get('answer'), len(possible_answers))
            is_correct = (len(correct_list) > 0 and set(predicted_list) == set(correct_list))
            logger.info(f"  - Predicted: {predicted_list}")
            logger.info(f"  - Ground Truth: {correct_list}")
            if is_correct:
                logger.info("  - Result: 🎉 CORRECT! 🎉")
                correct_answers_count += 1
            else:
                logger.info("  - Result: 🔴 WRONG. 🔴")
            executor.save_logs(log_dir)
        except Exception as e:
            logger.error(f"❌ Loop Error: {e}", exc_info=True)
        finally:
            executor.cleanup_agent_state()
            _save_cc_checkpoint(correct_answers_count, current_question_abs_num)
            logger.info(f"--- End of Q{question_id} ---")

    # Clean up checkpoint on full completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    logger.info("*******************")
    logger.info("   Final Summary   ")
    logger.info("*******************")
    accuracy = (correct_answers_count / num_questions_to_run) * 100 if num_questions_to_run > 0 else 0
    logger.info(f"✅ Questions Run:    {num_questions_to_run}")
    logger.info(f"🏆 Correct Answers:  {correct_answers_count}")
    logger.info(f"📊 Accuracy:         {accuracy:.2f}%")
