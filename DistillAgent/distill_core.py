"""
Distillation Core: orchestrates the full distillation pipeline.

Flow for each question:
  1. Run master agent (API backend) → trajectory + answer
  2. Check correctness against ground truth
  3. If wrong → diagnosis agent analyzes → correction injected → retry
  4. Repeat until correct or max attempts exhausted
  5. Save final trajectory (original if correct, corrected if needed)
"""

import json
import os
import sys
import time
import copy
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Ensure agent_system is importable
_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..", "agent_system")
_AGENT_ROOT = os.path.abspath(_AGENT_ROOT)
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from qwen_agent import QwenModel
from runtime_options import resolve_prompt_config
from agent_executor import AgentExecutor

from task_config import (
    TASK_CODES, build_query, check_correct, get_ground_truth_text, get_num_options,
)
from task_runner import run_single_question
from diagnosis_agent import DiagnosisAgent, DiagnosisResult, build_correction_injection

import logging
logger = logging.getLogger("DistillAgent.Core")


@dataclass
class DistilledSample:
    """One successfully distilled training sample."""
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
        }


class DistillationOrchestrator:
    """Main orchestrator for the distillation pipeline."""

    def __init__(
        self,
        prompt_config: Dict[str, str],
        max_turns: int = 20,
        max_diagnosis_attempts: int = 3,
        output_dir: str = "",
    ):
        self.prompt_config = prompt_config
        self.max_turns = max_turns
        self.max_diagnosis_attempts = max_diagnosis_attempts
        self.output_dir = output_dir

        # Initialize components
        self.agent = QwenModel(prompt_config=prompt_config)
        self.diagnosis_agent = DiagnosisAgent()

        # Statistics
        self.stats = defaultdict(int)
        self.stats.update({
            "total": 0, "correct_first_try": 0,
            "corrected_by_diagnosis": 0, "failed_after_max_attempts": 0,
            "skipped": 0,
        })
        self.per_task_stats = defaultdict(lambda: defaultdict(int))

        # Collected samples
        self.samples: List[DistilledSample] = []
        self.failed_samples: List[Dict[str, Any]] = []

    def run_distillation(
        self,
        task: str,
        questions: List[Dict[str, Any]],
        start_index: int = 0,
    ) -> Tuple[List[DistilledSample], List[Dict[str, Any]]]:
        """Run distillation on a list of questions for one task."""

        task_samples = []
        task_failed = []

        total_in_batch = len(questions)
        for i, question_data in enumerate(questions):
            qid = question_data.get("id", f"{task}_{i}")
            abs_idx = start_index + i

            logger.info(
                "═" * 60 + f"\n  [{task}] Q{abs_idx+1}/{start_index+total_in_batch} (ID: {qid})\n" + "═" * 60
            )

            self.stats["total"] += 1
            self.per_task_stats[task]["total"] += 1

            try:
                sample = self._distill_single(task, question_data)
                if sample is not None:
                    task_samples.append(sample)
                    self.samples.append(sample)
                    if sample.is_correct:
                        if sample.num_attempts == 1:
                            self.stats["correct_first_try"] += 1
                            self.per_task_stats[task]["correct_first_try"] += 1
                        else:
                            self.stats["corrected_by_diagnosis"] += 1
                            self.per_task_stats[task]["corrected_by_diagnosis"] += 1
                    else:
                        self.stats["failed_after_max_attempts"] += 1
                        self.per_task_stats[task]["failed_after_max_attempts"] += 1
                else:
                    self.stats["skipped"] += 1
                    self.per_task_stats[task]["skipped"] += 1

            except Exception as e:
                logger.error(f"❌ Error on Q{qid}: {e}", exc_info=True)
                self.stats["skipped"] += 1
                self.per_task_stats[task]["skipped"] += 1
                task_failed.append({
                    "task": task, "question_id": qid,
                    "error": str(e),
                    "question_data": question_data,
                })

            # Progress report every 10 questions
            if (i + 1) % 10 == 0:
                self._log_progress(task)

        return task_samples, task_failed

    def _distill_single(
        self,
        task: str,
        question_data: Dict[str, Any],
    ) -> Optional[DistilledSample]:
        """Distill a single question, with diagnosis-based correction if needed."""

        ground_truth = question_data.get("answer")
        ground_truth_text = get_ground_truth_text(task, question_data)
        qid = question_data.get("id", "?")

        logger.info(f"  🎯 Ground Truth: {ground_truth_text[:200]}")

        diagnosis_history = []

        # ── Attempt 1: Standard run ──
        logger.info(f"  ▶️  Attempt 1/{self.max_diagnosis_attempts + 1}: Standard run")
        t_start = time.time()

        q_log_dir = os.path.join(self.output_dir, "detailed_logs", task, str(qid), "attempt_1")

        final_answer, master_history, tool_history, video_contexts = run_single_question(
            task=task,
            question_data=question_data,
            agent_instance=self.agent,
            prompt_config=self.prompt_config,
            max_turns=self.max_turns,
            log_dir=q_log_dir,
        )

        t_elapsed = time.time() - t_start
        is_correct = check_correct(task, final_answer, ground_truth)

        if is_correct is None:
            # CCQA: can't auto-check; accept as-is
            logger.info(f"  ⏱️  {t_elapsed:.1f}s → CCQA (requires scoring API)")
            return DistilledSample(
                task=task, question_id=qid, question_data=question_data,
                master_history=master_history, tool_history=tool_history,
                final_answer=final_answer, ground_truth=ground_truth,
                is_correct=True, num_attempts=1,
                diagnosis_history=diagnosis_history,
            )

        logger.info(
            f"  ⏱️  {t_elapsed:.1f}s → {'✅ CORRECT' if is_correct else '❌ WRONG'}"
        )

        if is_correct:
            return DistilledSample(
                task=task, question_id=qid, question_data=question_data,
                master_history=master_history, tool_history=tool_history,
                final_answer=final_answer, ground_truth=ground_truth,
                is_correct=True, num_attempts=1,
                diagnosis_history=diagnosis_history,
            )

        # ── Diagnosis + Correction Loop ──
        current_history = master_history
        current_tool_history = tool_history
        current_answer = final_answer

        for attempt in range(1, self.max_diagnosis_attempts + 1):
            logger.info(f"  🩺 Diagnosis attempt {attempt}/{self.max_diagnosis_attempts}...")

            # 1. Run diagnosis on the failed trajectory
            diagnosis = self.diagnosis_agent.diagnose(
                task=task,
                question_data=question_data,
                ground_truth_text=ground_truth_text,
                master_dialogue=current_history,
                tool_history=current_tool_history,
            )
            diagnosis_history.append({
                "attempt": attempt,
                "error_turn": diagnosis.error_turn,
                "error_type": diagnosis.error_type,
                "correction_scope": diagnosis.correction_scope,
                "error_description": diagnosis.error_description,
                "correction_message": diagnosis.correction_message,
                "confidence": diagnosis.confidence,
            })

            scope = diagnosis.correction_scope or "full_restart"
            logger.info(
                f"  💉 Correction: scope={scope} | turn={diagnosis.error_turn} | "
                f"type={diagnosis.error_type} | confidence={diagnosis.confidence:.2f}"
            )

            t_start = time.time()
            correction_msg = build_correction_injection(
                diagnosis, attempt, self.max_diagnosis_attempts
            )

            # Determine truncation point based on correction scope
            if scope == "full_restart":
                effective_error_turn = 1  # Keep only system + user query
            elif scope == "layer2_reobserve":
                # Keep all Layer1 results, discard Layer2 onwards
                # error_turn should point to the first Layer2 turn
                effective_error_turn = diagnosis.error_turn
            else:
                # answer_only / focus_reobserve: keep ALL observations
                # Truncate right before the wrong answer, preserving Layer1+Layer2
                effective_error_turn = diagnosis.error_turn

            correction_log_dir = os.path.join(
                self.output_dir, "detailed_logs", task, str(qid), f"attempt_{attempt+1}"
            )

            current_answer, current_history, current_tool_history = _run_with_correction(
                task=task,
                question_data=question_data,
                agent_instance=self.agent,
                prompt_config=self.prompt_config,
                previous_history=current_history,
                previous_tool_history=current_tool_history,
                correction_message=correction_msg,
                error_turn=effective_error_turn,
                video_contexts=None,
                max_turns=self.max_turns,
                log_dir=correction_log_dir,
            )
            t_elapsed = time.time() - t_start

            is_correct = check_correct(task, current_answer, ground_truth)
            if is_correct is None:
                is_correct = True  # CCQA accept

            logger.info(
                f"  ⏱️  {t_elapsed:.1f}s → {'✅ CORRECTED' if is_correct else '❌ STILL WRONG'}"
            )

            if is_correct:
                return DistilledSample(
                    task=task, question_id=qid, question_data=question_data,
                    master_history=current_history, tool_history=current_tool_history,
                    final_answer=current_answer, ground_truth=ground_truth,
                    is_correct=True, num_attempts=attempt + 1,
                    diagnosis_history=diagnosis_history,
                )

        # Exhausted all attempts
        logger.warning(f"  ❌ Failed after {self.max_diagnosis_attempts} diagnosis attempts")
        self.failed_samples.append({
            "task": task,
            "question_id": qid,
            "question_data": question_data,
            "ground_truth": ground_truth_text,
            "final_answer": str(current_answer),
            "diagnosis_history": diagnosis_history,
        })

        # Return the last trajectory anyway (for analysis, marked as incorrect)
        return DistilledSample(
            task=task, question_id=qid, question_data=question_data,
            master_history=current_history, tool_history=current_tool_history,
            final_answer=current_answer, ground_truth=ground_truth,
            is_correct=False, num_attempts=self.max_diagnosis_attempts + 1,
            diagnosis_history=diagnosis_history,
        )

    def _log_progress(self, current_task: str):
        """Log distillation progress."""
        s = self.stats
        logger.info(
            f"  📊 Progress: {s['total']} total | "
            f"✅ {s['correct_first_try']} first-try | "
            f"🩺 {s['corrected_by_diagnosis']} corrected | "
            f"❌ {s['failed_after_max_attempts']} failed | "
            f"⏭️  {s['skipped']} skipped"
        )

    def print_summary(self):
        """Print final distillation summary."""
        s = self.stats
        total_attempted = s["total"] - s["skipped"]
        correct_total = s["correct_first_try"] + s["corrected_by_diagnosis"]
        success_rate = correct_total / max(total_attempted, 1) * 100
        first_try_rate = s["correct_first_try"] / max(total_attempted, 1) * 100

        print("\n" + "=" * 60)
        print("  Distillation Summary")
        print("=" * 60)
        print(f"  Total questions:        {s['total']}")
        print(f"  ✅ Correct on 1st try:  {s['correct_first_try']} ({first_try_rate:.1f}%)")
        print(f"  🩺 Corrected by diagnosis: {s['corrected_by_diagnosis']}")
        print(f"  ❌ Failed after max:    {s['failed_after_max_attempts']}")
        print(f"  ⏭️  Skipped (error):     {s['skipped']}")
        print(f"  ─────────────────────")
        print(f"  📊 Success rate:        {correct_total}/{total_attempted} ({success_rate:.1f}%)")
        print(f"  💾 Collectable samples: {s['correct_first_try'] + s['corrected_by_diagnosis']}")
        print(f"  🗑️  Discarded samples:   {s['failed_after_max_attempts']}")

        print("\n  Per-task breakdown:")
        for task in sorted(self.per_task_stats.keys()):
            ts = self.per_task_stats[task]
            t_total = ts["total"] - ts.get("skipped", 0)
            t_correct = ts.get("correct_first_try", 0) + ts.get("corrected_by_diagnosis", 0)
            t_rate = t_correct / max(t_total, 1) * 100
            print(
                f"    {task:6s}: {t_correct:4d}/{t_total:4d} ({t_rate:5.1f}%)  "
                f"1st-try={ts.get('correct_first_try',0)}  corrected={ts.get('corrected_by_diagnosis',0)}  "
                f"failed={ts.get('failed_after_max_attempts',0)}"
            )
        print("=" * 60)


def _run_with_correction(
    task: str,
    question_data: Dict[str, Any],
    agent_instance: QwenModel,
    prompt_config: Dict[str, str],
    previous_history: List[Dict[str, Any]],
    previous_tool_history: List[Dict[str, Any]],
    correction_message: str,
    error_turn: int,
    video_contexts: Optional[List[Dict[str, Any]]],
    max_turns: int = 20,
    log_dir: Optional[str] = None,
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Resume a failed trajectory from before the error turn, injecting correction.
    We truncate history before the error and add the correction message.
    """
    from task_runner import build_video_contexts, build_query

    query = build_query(task, question_data)
    possible_answers = question_data.get("options", [])

    if video_contexts is None:
        video_contexts = build_video_contexts(task, question_data)

    executor = AgentExecutor(
        agent_instance=agent_instance,
        prompt_config=prompt_config,
    )

    executor.init_agent_state(
        query=query,
        possible_answers=possible_answers,
        video_contexts=video_contexts,
    )

    # Truncate master history to before the error turn.
    # error_turn=1 → keep only system + user query (full restart)
    # error_turn=N → keep everything before turn N's assistant message
    # Message layout: [0]=system, [1]=user_query, [2]=asst_t1, [3]=tool_t1, [4]=asst_t2, ...
    error_msg_idx = 2 + (error_turn - 1) * 2
    if error_msg_idx >= len(previous_history):
        error_msg_idx = max(2, len(previous_history) - 1)

    # Build resume history: safe prefix + correction injection
    safe_history = list(previous_history[:error_msg_idx])
    safe_history.append({
        "role": "user",
        "content": correction_message,
    })

    # Set resume flag so run_agent_loop uses this instead of rebuilding
    executor._resume_history = safe_history

    # Copy tool history up to the error turn
    safe_tool_count = 0
    for entry in previous_tool_history:
        if entry.get("turn", 0) < error_turn:
            safe_tool_count += 1
    executor._tool_history = previous_tool_history[:safe_tool_count]

    # Continue the agent loop (will use _resume_history)
    try:
        remaining_turns = max(1, max_turns - error_turn + 1)
        final_answer, master_history = executor.run_agent_loop(max_turns=remaining_turns)
        tool_history = list(executor._tool_history)
        return final_answer, master_history, tool_history
    finally:
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                executor.save_logs(log_dir)
            except Exception as e:
                import logging
                logging.getLogger("DistillAgent.Core").warning(
                    "Failed to save correction logs: %s", e
                )
        executor.cleanup_agent_state()
