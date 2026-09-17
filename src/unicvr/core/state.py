"""Compact Reasoner Protocol v1 — state types.

Reasoner outputs a fixed-schema Decision State (400-900 tokens).
No free-text reasoning, no narrative essays, no claim graphs.
Same schema for FORM and REVIEW.  Task-agnostic across PSS/FSA/CC/etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from unicvr.core.schemas import BackendCallRecord, VideoRef


# ── observation (unchanged) ──────────────────────────────────────────


@dataclass
class ObserverReport:
    video_id: str
    report_text: str
    frame_count: int
    start_time: float
    end_time: float


# ── answer types (unified algebra) ───────────────────────────────────

AnswerType = Literal["sequence", "interval", "choice", "matching", "set", "free_text"]


@dataclass
class Answer:
    type: AnswerType = "free_text"
    value: Any = None                          # list[str] | [float,float] | str | list[list[str]]
    rationale: str | None = None               # compact reasoning summary (1-2 sentences per candidate)


# ── state factors (compact external scratchpad) ──────────────────────

BasisKind = Literal["visual", "cross_visual", "inferred", "prior", "uncertain"]


@dataclass
class StateObservation:
    video: str = ""
    span: tuple[float, float] | None = None    # time window, null if unknown
    state: str = ""                             # short label, 1-5 words
    basis: BasisKind = "visual"


@dataclass
class StateFactor:
    factor: str = ""                            # object | event | functional_step | ...
    observations: list[StateObservation] = field(default_factory=list)


# ── ambiguities (evidence needs) ─────────────────────────────────────

ScopeKind = Literal["single_video", "multi_video_independent", "joint_compare", "not_observable"]


@dataclass
class AmbiguityTarget:
    video: str = ""
    span: tuple[float, float] | None = None


@dataclass
class Ambiguity:
    contrast: str = ""                          # short label, e.g. "v3_vs_v5_order"
    witness: str = ""                           # specific observable question, <= 60 chars CN / 35 words EN
    scope: ScopeKind = "not_observable"
    targets: list[AmbiguityTarget] = field(default_factory=list)
    pair: list[str] = field(default_factory=list)  # [vA, vB] — the disputed pair the witness MUST adjudicate


# ── decision ─────────────────────────────────────────────────────────

DecisionKind = Literal["DONE", "NEED_EVIDENCE", "UNCERTAIN"]


# ── Reasoner state (the ONLY LLM output) ─────────────────────────────

@dataclass
class ReasonerState:
    """Compact decision state.  5 top-level fields only.  No reasoning essay."""

    best_answer: Answer | None = None
    alternatives: list[Answer] = field(default_factory=list)     # 0-2
    state_factors: list[StateFactor] = field(default_factory=list)  # 0-6
    ambiguities: list[Ambiguity] = field(default_factory=list)     # 0-2
    decision: DecisionKind = "DONE"

    # ── metadata (not part of LLM output, filled by pipeline) ──
    input_tokens: int = 0
    output_tokens: int = 0
    round_index: int = 0


# ── evidence results ─────────────────────────────────────────────────


@dataclass
class EvidenceResult:
    ambiguity: Ambiguity
    scope: ScopeKind = "not_observable"
    result_text: str = ""


# ── pipeline state ───────────────────────────────────────────────────


@dataclass
class PipelineState:
    sample_id: str
    question: str
    options: list[str]
    answer_format: str
    videos: list[VideoRef] = field(default_factory=list)

    observer_reports: list[ObserverReport] = field(default_factory=list)
    reasoner_state: ReasonerState | None = None
    evidence_results: list[EvidenceResult] = field(default_factory=list)
    final_answer: str | None = None

    action_trace: list[dict] = field(default_factory=list)
    api_calls: list[BackendCallRecord] = field(default_factory=list)
    stop_reason: str | None = None
    errors: list[str] = field(default_factory=list)


# ── validators ───────────────────────────────────────────────────────


def validate_reasoner_state(state: ReasonerState, video_ids: list[str]) -> list[str]:
    """Deterministic schema + cross-field validation. Returns errors."""
    errs: list[str] = []

    # best_answer
    if state.best_answer is None or state.best_answer.value is None:
        errs.append("best_answer missing")
    else:
        errs.extend(_validate_answer(state.best_answer, video_ids))

    # alternatives
    if len(state.alternatives) > 2:
        errs.append(f"alternatives count {len(state.alternatives)} > 2")
    for i, alt in enumerate(state.alternatives):
        errs.extend(_validate_answer(alt, video_ids, prefix=f"alt[{i}]"))

    # state_factors
    if len(state.state_factors) > 6:
        errs.append(f"state_factors count {len(state.state_factors)} > 6")

    # ambiguities
    if len(state.ambiguities) > 2:
        errs.append(f"ambiguities count {len(state.ambiguities)} > 2")

    known_vids = set(video_ids)
    for i, amb in enumerate(state.ambiguities):
        if amb.scope not in ("single_video", "multi_video_independent", "joint_compare", "not_observable"):
            errs.append(f"amb[{i}]: invalid scope '{amb.scope}'")
        for j, t in enumerate(amb.targets):
            if t.video not in known_vids:
                errs.append(f"amb[{i}].target[{j}]: unknown video '{t.video}'")
        if amb.pair:
            if len(amb.pair) != 2:
                errs.append(f"amb[{i}]: pair must have exactly 2 videos, got {len(amb.pair)}")
            for pid in amb.pair:
                if pid not in known_vids:
                    errs.append(f"amb[{i}].pair: unknown video '{pid}'")

    # cross-field
    if state.decision == "DONE" and len(state.ambiguities) > 0:
        errs.append("decision=DONE but ambiguities non-empty")
    if state.decision == "NEED_EVIDENCE":
        if len(state.ambiguities) == 0:
            errs.append("decision=NEED_EVIDENCE but no ambiguities")
        elif all(a.scope == "not_observable" for a in state.ambiguities):
            errs.append("decision=NEED_EVIDENCE but all ambiguities are not_observable")

    return errs


def validate_answer_payload(ans: Answer, video_ids: list[str],
                            prefix: str = "best_answer") -> list[str]:
    """Structural checks for one answer payload.

    ``validate_reasoner_state`` wraps the whole decision state; the pipeline's
    FORM repair loop needs the same checks for ``best_answer`` alone so the
    messages can be fed back into a re-ask prompt.
    """
    return _validate_answer(ans, video_ids, prefix=prefix)


def _validate_answer(ans: Answer, video_ids: list[str], prefix: str = "best_answer") -> list[str]:
    errs: list[str] = []
    v = ans.value
    if ans.type == "sequence":
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            errs.append(f"{prefix}: sequence value must be list[str]")
        else:
            s = set(v)
            e = set(video_ids)
            if len(v) != len(video_ids):
                errs.append(f"{prefix}: expected {len(video_ids)} videos, got {len(v)}")
            if len(s) != len(v):
                errs.append(f"{prefix}: duplicate videos")
            if m := e - s:
                errs.append(f"{prefix}: missing {sorted(m)}")
            if x := s - e:
                errs.append(f"{prefix}: unknown {sorted(x)}")
    elif ans.type == "interval":
        if not isinstance(v, list) or len(v) != 2:
            errs.append(f"{prefix}: interval must be [start, end]")
        else:
            try:
                a, b = float(v[0]), float(v[1])
                if a >= b:
                    errs.append(f"{prefix}: start({a}) >= end({b})")
                if a < 0:
                    errs.append(f"{prefix}: negative start")
            except (TypeError, ValueError):
                errs.append(f"{prefix}: non-numeric interval")
    elif ans.type == "choice":
        if not isinstance(v, str):
            errs.append(f"{prefix}: choice must be a string")
    elif ans.type == "matching":
        if not isinstance(v, list):
            errs.append(f"{prefix}: matching must be list of pairs")
    return errs


# ── P0#2: covered-pair gating (REVIEW may only reorder witnessed pairs) ──


def flipped_pairs(before: list[str], after: list[str]) -> list[tuple[str, str]]:
    """Pairs (a, b) whose relative order changed from `before` to `after`.

    (a, b) is flipped if a comes before b in `after` but came after b in
    `before`. Unknown ids are treated as not reordered. `after` is the
    candidate new order, `before` the previously accepted one.
    """
    pos = {v: i for i, v in enumerate(before)}
    flips: list[tuple[str, str]] = []
    for i in range(len(after)):
        for j in range(i + 1, len(after)):
            a, b = after[i], after[j]
            pa = pos.get(a)
            pb = pos.get(b)
            if pa is not None and pb is not None and pa > pb:
                flips.append((a, b))
    return flips


def uncovered_flips(
    before: list[str], after: list[str], covered_pairs: list[list[str]]
) -> list[tuple[str, str]]:
    """Flipped pairs not covered by any witness pair.

    A witness pair [vA, vB] covers both orientations (vA before vB and
    vB before vA), since the witness only pins their relative order.
    """
    flips = flipped_pairs(before, after)
    if not flips:
        return []
    covered = set()
    for p in covered_pairs:
        if len(p) == 2:
            covered.add((p[0], p[1]))
            covered.add((p[1], p[0]))
    return [(a, b) for (a, b) in flips if (a, b) not in covered]


# ── deterministic renderer ────────────────────────────────────────────


def render_answer(ans: Answer) -> str:
    """Deterministic rendering — no LLM call."""
    v = ans.value
    if ans.type == "sequence" and isinstance(v, list):
        return "->".join(str(x) for x in v)
    if ans.type == "interval" and isinstance(v, list) and len(v) == 2:
        return f"[{float(v[0]):.1f}, {float(v[1]):.1f}]"
    return str(v)
