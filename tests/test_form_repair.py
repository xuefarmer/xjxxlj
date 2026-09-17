"""Phase-2 repair loop: structurally impossible sequence answers (PSS).

A dropped video id can never match the ground truth and no later evidence
round notices it, so FORM is re-asked with the concrete validation errors.
These tests pin the control flow with a scripted backend (no GPU, no video).
"""

from __future__ import annotations

import json
from typing import Any

from unicvr.config import AppConfig
from unicvr.core.pipeline import MAX_FORM_REPAIRS, Pipeline
from unicvr.core.schemas import BackendCallRecord
from unicvr.core.state import PipelineState


class _ScriptedLLM:
    """Replays scripted FORM answers; the final entry repeats once exhausted."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = answers
        self.calls: list[BackendCallRecord] = []
        self.prompts: list[str] = []

    def generate_text(self, *, role: str, system_prompt: str, user_prompt: str,
                      generation_config: Any, visual_inputs: Any = ()) -> str:
        raw = self._answers[min(len(self.calls), len(self._answers) - 1)]
        self.prompts.append(user_prompt)
        self.calls.append(BackendCallRecord(role=role, backend="scripted", model="scripted",
                                            raw_response=raw[:500], user_prompt=user_prompt))
        return raw

    def generate_structured(self, **_kwargs: Any) -> Any:
        raise AssertionError("FORM must go through generate_text")

    def generate_json(self, **_kwargs: Any) -> Any:
        raise AssertionError("FORM must go through generate_text")


def _sequence(*value: str) -> str:
    return json.dumps({
        "best_answer": {"type": "sequence", "value": list(value)},
        "alternatives": [], "state_factors": [], "ambiguities": [], "decision": "DONE",
    })


def _interval(start: float, end: float) -> str:
    return json.dumps({
        "best_answer": {"type": "interval", "value": [start, end]},
        "alternatives": [], "state_factors": [], "ambiguities": [], "decision": "DONE",
    })


def _run(app_config: AppConfig, answers: list[str], *, answer_type: str = "sequence",
         video_ids: tuple[str, ...] = ("v1", "v2", "v3")):
    llm = _ScriptedLLM(answers)
    pipeline = Pipeline(app_config, llm_backend=llm)
    state = PipelineState(sample_id="unit", question="Which order?", options=[],
                          answer_format=answer_type)
    rs = pipeline._form_checked(question="Which order?", answer_type=answer_type,
                                reports=[], prompt_blocks=None,
                                video_ids=list(video_ids), state=state)
    return llm, state, rs


def test_complete_sequence_is_not_re_asked(app_config: AppConfig) -> None:
    llm, state, rs = _run(app_config, [_sequence("v1", "v2", "v3")])

    assert len(llm.calls) == 1
    assert state.action_trace == []
    assert rs.best_answer is not None and rs.best_answer.value == ["v1", "v2", "v3"]


def test_missing_video_re_asks_with_validation_errors(app_config: AppConfig) -> None:
    llm, state, rs = _run(app_config, [_sequence("v1", "v3"),
                                       _sequence("v3", "v1", "v2")])

    assert len(llm.calls) == 2
    assert [entry["action"] for entry in state.action_trace] == ["FORM_INVALID", "FORM_REPAIRED"]
    assert state.action_trace[0]["errors"] == [
        "best_answer: expected 3 videos, got 2",
        "best_answer: missing ['v2']",
    ]
    # The rejected answer would reproduce itself under greedy decoding, so the
    # errors must be part of the new prompt, not just a retry.
    assert "missing ['v2']" in llm.prompts[1]
    assert "## Your Previous Output Was Rejected" in llm.prompts[1]
    assert "## Your Previous Output Was Rejected" not in llm.prompts[0]
    assert rs.best_answer is not None and rs.best_answer.value == ["v3", "v1", "v2"]


def test_duplicate_video_re_asks(app_config: AppConfig) -> None:
    llm, state, rs = _run(app_config, [_sequence("v1", "v1", "v3"),
                                       _sequence("v1", "v2", "v3")])

    assert len(llm.calls) == 2
    assert state.action_trace[0]["errors"] == [
        "best_answer: duplicate videos",
        "best_answer: missing ['v2']",
    ]
    assert rs.best_answer is not None and rs.best_answer.value == ["v1", "v2", "v3"]


def test_unrepairable_sequence_falls_back_deterministically(app_config: AppConfig) -> None:
    llm, state, rs = _run(app_config, [_sequence("v1", "v3")])

    assert len(llm.calls) == 1 + MAX_FORM_REPAIRS
    assert [entry["action"] for entry in state.action_trace] == (
        ["FORM_INVALID"] * (MAX_FORM_REPAIRS + 1) + ["FORM_REPAIR_FALLBACK"]
    )
    assert state.action_trace[-1]["missing"] == ["v2"]
    # A wrong-length sequence already scores zero, so completing it cannot lose.
    assert rs.best_answer is not None and rs.best_answer.value == ["v1", "v3", "v2"]


def test_fallback_keeps_the_surviving_order(app_config: AppConfig) -> None:
    _, _, rs = _run(app_config, [_sequence("v3", "v1")], video_ids=("v1", "v2", "v3"))

    assert rs.best_answer is not None and rs.best_answer.value == ["v3", "v1", "v2"]


def test_non_sequence_answers_are_never_re_asked(app_config: AppConfig) -> None:
    # FSA intervals stay exactly as before: the repair loop is sequence-only.
    llm, state, rs = _run(app_config, [_interval(20.0, 10.0)], answer_type="interval")

    assert len(llm.calls) == 1
    assert state.action_trace == []
    assert rs.best_answer is not None and rs.best_answer.value == [20.0, 10.0]
