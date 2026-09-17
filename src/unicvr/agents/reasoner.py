"""Compact Reasoner Agent — outputs fixed-schema Decision State.

No free-text reasoning.  Same schema for FORM and REVIEW.
Target: 400-900 tokens output.  Designed for 8B/14B fine-tuning.
"""

from __future__ import annotations

import re

from unicvr.core.schemas import GenerationConfig
from unicvr.core.state import (
    Ambiguity,
    AmbiguityTarget,
    Answer,
    ObserverReport,
    ReasonerState,
    StateFactor,
    StateObservation,
    render_answer,
)
from unicvr.models.base import LLMBackend
from unicvr.prompts import prompt


class ReasonerAgent:
    def __init__(self, backend: LLMBackend, generation: GenerationConfig) -> None:
        self.backend = backend
        self.generation = generation

    # ── FORM (Phase 2) ───────────────────────────────────────────────

    def form(self, *, question: str, answer_type: str,
             reports: list[ObserverReport],
             prompt_blocks: list[str] | None = None,
             repair_feedback: list[str] | None = None) -> ReasonerState:
        """Form a decision state; ``repair_feedback`` re-asks a rejected answer.

        Decoding is greedy by default, so re-issuing an identical prompt
        reproduces the rejected output; the caller passes the concrete
        validation errors so the model has something new to act on.
        """
        raw = self.backend.generate_text(
            role="ReasonerAgent.FORM",
            system_prompt=prompt("reasoner_form"),
            user_prompt=_form_prompt(question, answer_type, reports,
                                     prompt_blocks=prompt_blocks,
                                     repair_feedback=repair_feedback),
            generation_config=self.generation,
        )
        return _parse(raw, source="FORM")

    # ── REVIEW (Phase 3 loop) ─────────────────────────────────────────

    def review(self, *, question: str, answer_type: str,
               reports: list[ObserverReport],
               previous: ReasonerState,
               new_evidence: list, round_index: int,
               video_ids: list[str] | None = None,
               timing_feedback: str | None = None,
               prompt_blocks: list[str] | None = None) -> ReasonerState:
        ev_text = _format_evidence(new_evidence) if new_evidence else "None."
        prev_json = _state_to_json(previous)
        raw = self.backend.generate_text(
            role="ReasonerAgent.REVIEW",
            system_prompt=prompt("reasoner_review"),
            user_prompt=_review_prompt(question, answer_type, reports, prev_json,
                                       ev_text, round_index, timing_feedback,
                                       new_evidence=new_evidence,
                                       prompt_blocks=prompt_blocks),
            generation_config=self.generation,
        )
        state = _parse(raw, source="REVIEW")
        if video_ids:
            errs = validate_reasoner_state(state, video_ids)
            if errs and state.decision == "DONE":
                from unicvr.core.state import validate_reasoner_state as _v
                pass  # validation errors are logged by pipeline
        return state


# ── prompt builders ──────────────────────────────────────────────────

def _form_prompt(q: str, atype: str, reports: list[ObserverReport],
                 prompt_blocks: list[str] | None = None,
                 repair_feedback: list[str] | None = None) -> str:
    parts = [
        f"## Question\n{q}\n\n",
        f"## Answer Type: {atype}\n\n",
    ]
    if prompt_blocks:
        for block in prompt_blocks:
            parts.append(f"{block}\n\n")
    parts.append(f"## Observer Reports\n{_fmt_reports(reports)}\n\n")
    parts.append("Produce compact decision state JSON. Do NOT output reasoning process.")
    if repair_feedback:
        parts.append(
            "\n\n## Your Previous Output Was Rejected\n"
            + "\n".join(f"- {item}" for item in repair_feedback)
            + "\nFix these problems and return the corrected JSON object only. "
              "A sequence answer must list every video id exactly once."
        )
    return "".join(parts)

def _review_prompt(q: str, atype: str, reports: list[ObserverReport],
                   prev_json: str, ev_text: str, rnd: int,
                   timing_feedback: str | None = None,
                   new_evidence: list | None = None,
                   prompt_blocks: list[str] | None = None) -> str:
    parts = [
        f"## Question\n{q}\n\n## Answer Type: {atype}\n\n",
        f"## Observer Reports\n{_fmt_reports(reports)}\n\n",
        f"## Previous State\n{prev_json}\n\n",
        f"## New Evidence (Round {rnd})\n{ev_text}\n\n",
    ]
    if prompt_blocks:
        for block in prompt_blocks:
            parts.append(f"{block}\n\n")
    if new_evidence:
        pairs = [
            list(item.ambiguity.pair)
            for item in new_evidence
            if len(item.ambiguity.pair) == 2
        ]
        if pairs:
            parts.append(
                "## Covered Pairs\n"
                "You may ONLY reorder the relative position of a pair listed here: "
                + ", ".join(f"[{p[0]},{p[1]}]" for p in pairs)
                + "\n\n"
            )
    if timing_feedback:
        parts.append(f"## Timing Feedback\n{timing_feedback}\n\n")
    parts.append("Update compact decision state. Return ONLY the updated JSON.")
    return "".join(parts)


# ── parse ────────────────────────────────────────────────────────────

def _parse(raw: str, *, source: str) -> ReasonerState:
    try:
        data = _extract_json(raw)
    except ValueError:
        return ReasonerState(decision="UNCERTAIN")

    # best_answer
    ba = data.get("best_answer")
    best = _parse_answer(ba) if isinstance(ba, dict) else None

    # alternatives
    alts = []
    for a in (data.get("alternatives") or [])[:2]:
        if isinstance(a, dict):
            parsed = _parse_answer(a)
            if parsed is not None:
                alts.append(parsed)

    # state_factors
    factors = []
    for f in (data.get("state_factors") or [])[:6]:
        if not isinstance(f, dict):
            continue
        obs_list = []
        for o in (f.get("observations") or []):
            if not isinstance(o, dict):
                continue
            obs_list.append(StateObservation(
                video=str(o.get("video", "")),
                span=_parse_span(o.get("span")),
                state=str(o.get("state", "")),
                basis=o.get("basis", "visual"),
            ))
        factors.append(StateFactor(
            factor=str(f.get("factor", "")),
            observations=obs_list,
        ))

    # ambiguities
    ambs = []
    for a in (data.get("ambiguities") or [])[:2]:
        if not isinstance(a, dict):
            continue
        targets = []
        for t in (a.get("targets") or []):
            if isinstance(t, dict):
                targets.append(AmbiguityTarget(
                    video=str(t.get("video", "")),
                    span=_parse_span(t.get("span")),
                ))
        witness = str(a.get("witness", ""))
        _recover_witness_spans(targets, witness)
        raw_pair = a.get("pair") or []
        pair = [str(x) for x in raw_pair[:2]] if isinstance(raw_pair, list) else []
        ambs.append(Ambiguity(
            contrast=str(a.get("contrast", "")),
            witness=witness,
            scope=a.get("scope", "not_observable"),
            targets=targets,
            pair=pair,
        ))

    decision = data.get("decision", "DONE")
    if decision not in ("DONE", "NEED_EVIDENCE", "UNCERTAIN"):
        decision = "UNCERTAIN"

    return ReasonerState(
        best_answer=best,
        alternatives=alts,
        state_factors=factors,
        ambiguities=ambs,
        decision=decision,
    )


# ── helpers ──────────────────────────────────────────────────────────

def _parse_answer(d: dict) -> Answer | None:
    v = d.get("value")
    if v is None:
        return None
    t = d.get("type", "free_text")
    rationale = d.get("rationale")
    return Answer(
        type=t,
        value=v,
        rationale=str(rationale).strip() if isinstance(rationale, str) else None,
    )

def _parse_span(v: object) -> tuple[float, float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            start, end = float(v[0]), float(v[1])
            return (start, end) if 0 <= start <= end else None
        except (TypeError, ValueError):
            return None
    # Local models frequently render an otherwise valid span as
    # ``"138.3s-163.5s"`` (or with an en dash) rather than JSON numbers.
    # Treat it as a recoverable schema variation: dropping the span makes a
    # later Focus inspect the entire long video instead of its stated witness.
    if isinstance(v, str):
        # Video timestamps are non-negative.  Delimiter ``-`` must not be
        # misread as the sign of the second timestamp in ``12.0s-18.0s``.
        numbers = re.findall(r"\d+(?:\.\d+)?", v)
        if len(numbers) == 1:
            timestamp = float(numbers[0])
            return (timestamp, timestamp)
        if len(numbers) == 2:
            try:
                start, end = float(numbers[0]), float(numbers[1])
                if 0 <= start <= end:
                    return (start, end)
            except ValueError:
                pass
    return None


_WITNESS_TIMESTAMP_RE = re.compile(
    r"(?<![\w.])(\d{1,4}(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b",
    re.IGNORECASE,
)


def _recover_witness_spans(targets: list[AmbiguityTarget], witness: str) -> None:
    """Recover unambiguous point timestamps omitted from target ``span`` fields.

    If the witness names videos, each timestamp belongs to its nearest video
    mention.  Without video mentions, recovery is safe only for one timestamp
    and one target.  Existing spans are never replaced.
    """
    timestamps = [
        (match.start(1), float(match.group(1)))
        for match in _WITNESS_TIMESTAMP_RE.finditer(witness)
    ]
    if not timestamps or not targets:
        return

    mentions = [
        (match.start(), f"v{match.group(1)}")
        for match in re.finditer(
            r"(?<!\w)(?:v\s*|video\s+)(\d+)(?:['’]s)?(?!\w)",
            witness,
            flags=re.IGNORECASE,
        )
    ]
    for target in targets:
        short = re.fullmatch(r"v(\d+)", target.video, flags=re.IGNORECASE)
        if short:
            continue
        elif target.video:
            pattern = re.compile(
                rf"(?<!\w){re.escape(target.video)}(?:['’]s)?(?!\w)",
                re.IGNORECASE,
            )
            mentions.extend((match.start(), target.video) for match in pattern.finditer(witness))
        else:
            continue

    if not mentions:
        if len(targets) == 1 and len(timestamps) == 1 and targets[0].span is None:
            timestamp = timestamps[0][1]
            targets[0].span = (timestamp, timestamp)
        return

    assigned: dict[str, list[float]] = {}
    for position, timestamp in timestamps:
        _, video_id = min(mentions, key=lambda item: abs(item[0] - position))
        assigned.setdefault(video_id, []).append(timestamp)
    for target in targets:
        short = re.fullmatch(r"v(\d+)", target.video, flags=re.IGNORECASE)
        lookup_video = f"v{short.group(1)}" if short else target.video
        candidates = assigned.get(lookup_video, [])
        if target.span is None and len(candidates) == 1:
            target.span = (candidates[0], candidates[0])

def _fmt_reports(reports: list[ObserverReport]) -> str:
    return "\n".join(
        f"### {r.video_id} ({r.start_time:.1f}s-{r.end_time:.1f}s, {r.frame_count}f)\n\n{r.report_text}\n"
        for r in reports
    )

def _format_evidence(results: list) -> str:
    parts = []
    for item in results:
        amb = item.ambiguity
        parts.append(
            f"### Witness: {amb.witness}\n"
            f"Scope: {amb.scope} | Targets: {[(t.video, t.span) for t in amb.targets]}\n"
            f"Result:\n{item.result_text}\n"
        )
    return "\n".join(parts)

def _state_to_json(state: ReasonerState) -> str:
    import json
    d = {"decision": state.decision}
    if state.best_answer:
        d["best_answer"] = {"type": state.best_answer.type, "value": state.best_answer.value}
        if state.best_answer.rationale:
            d["best_answer"]["rationale"] = state.best_answer.rationale
    if state.alternatives:
        d["alternatives"] = [
            {"type": a.type, "value": a.value, **({"rationale": a.rationale} if a.rationale else {})}
            for a in state.alternatives
        ]
    if state.state_factors:
        d["state_factors"] = [
            {"factor": f.factor, "observations": [
                {"video": o.video, "span": o.span, "state": o.state, "basis": o.basis}
                for o in f.observations
            ]} for f in state.state_factors
        ]
    if state.ambiguities:
        d["ambiguities"] = [
            {"contrast": a.contrast, "witness": a.witness, "scope": a.scope,
             "targets": [{"video": t.video, "span": t.span} for t in a.targets],
             **({"pair": a.pair} if a.pair else {})}
            for a in state.ambiguities
        ]
    return json.dumps(d, ensure_ascii=False)

def _extract_json(text: str) -> dict:
    import re, json as _json
    t = text.strip()
    if t.startswith("```"): t = re.sub(r"^```(?:json)?\s*", "", t, 1); t = re.sub(r"\s*```$", "", t, 1)
    try: return _json.loads(t)
    except Exception: pass
    d = __import__("json").JSONDecoder()
    for i, ch in enumerate(t):
        if ch == "{":
            try: return d.raw_decode(t[i:])[0]
            except Exception: continue
    raise ValueError(f"No JSON: {t[:200]}")

# Re-export
from unicvr.core.state import validate_reasoner_state
