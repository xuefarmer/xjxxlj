"""Compact Reasoner Protocol — unified pipeline with scope-based evidence routing.

Phase 1 [parallel VLM]: Observer x N -> N observation reports
Phase 2 [single LLM]:   Reasoner FORM -> compact ReasonerState
Phase 3 [loop, max 2]:  Route ambiguities by scope -> FOCUS|COMPARE
                        Reasoner REVIEW -> updated ReasonerState
Phase 4 [no LLM]:       Deterministic render from best_answer
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

from unicvr.agents import ComparerAgent, ObserverAgent, ReasonerAgent
from unicvr.config import AppConfig
from unicvr.core.schemas import VideoRef, VisualInput
from unicvr.core.state import (
    Ambiguity,
    Answer,
    EvidenceResult,
    ObserverReport,
    PipelineState,
    ReasonerState,
    render_answer,
    validate_answer_payload,
)
from unicvr.data.schema import MultiVideoSample
from unicvr.models.base import LLMBackend, VLMBackend
from unicvr.models.factory import build_llm, build_vlm
from unicvr.prompts import prompt
from unicvr.video import (
    FocusSampler,
    FrameCache,
    MicroClipBuilder,
    VideoDecoder,
    VideoProbe,
    WarmStartSampler,
)

MAX_EVIDENCE_ROUNDS = 2
# Bounded re-asks when FORM returns a structurally impossible answer.  Only
# sequence answers (PSS) are gated; see ``Pipeline._form_checked``.
MAX_FORM_REPAIRS = 2


class PipelineRunError(RuntimeError):
    def __init__(self, message: str, state: PipelineState) -> None:
        super().__init__(message)
        self.state = state


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        *,
        llm_backend: LLMBackend | None = None,
        vlm_backend: VLMBackend | None = None,
    ) -> None:
        """Create a pipeline.

        ``llm_backend`` / ``vlm_backend`` are intentionally injectable for
        local training and deterministic diagnostics.  Normal CLI/evaluation
        callers omit them and retain the configured backend factory path.
        """
        self.config = config
        llm = llm_backend or build_llm(config)
        vlm = vlm_backend or build_vlm(config)
        gen = config.generation
        self.observer = ObserverAgent(vlm, gen)
        self.reasoner = ReasonerAgent(llm, gen)
        self.comparer = ComparerAgent(vlm, gen)
        self._llm = llm
        self._vlm = vlm
        from unicvr.plugins import TimingGuard
        self.timing_guard = TimingGuard(config.timing_guard)

        cache = FrameCache(config.video.cache_dir)
        self.decoder = VideoDecoder(cache, resize_long_edge=config.video.resize_long_edge,
                                     jpeg_quality=config.video.jpeg_quality)
        self.warm_sampler = WarmStartSampler(
            coverage_weight=config.video.coverage_weight,
            change_weight=config.video.change_weight,
            question_weight=config.video.question_weight,
            redundancy_weight=config.video.redundancy_weight,
            candidate_multiplier=config.video.candidate_multiplier,
        )
        self.microclips = MicroClipBuilder(self.decoder, FocusSampler(
            pre_context_ratio=config.video.pre_context_ratio,
            post_context_ratio=config.video.post_context_ratio,
        ))

    # ── main ──────────────────────────────────────────────────────────

    def run(self, sample: MultiVideoSample) -> PipelineState:
        videos = [VideoProbe().probe(path, video_id=f"v{i+1}")
                  for i, path in enumerate(sample.video_paths)]
        answer_type = _derive_answer_type(sample)
        state = PipelineState(
            sample_id=sample.sample_id, question=sample.question,
            options=sample.options, answer_format=answer_type, videos=videos,
        )
        llm0, vlm0 = len(self._llm.calls), len(self._vlm.calls)
        video_ids = [v.video_id for v in videos]

        try:
            # Per-task plugin blocks (choice options, MOC counting, MSR force-choice...)
            from unicvr.plugins.registry import (
                blocks_for_sample,
                visual_question_for_sample,
            )
            form_blocks = blocks_for_sample(
                sample, "form",
                question=state.question, answer_type=answer_type,
                options=sample.options,
            )
            visual_questions = {
                video_id: visual_question_for_sample(sample, state.question, video_id)
                for video_id in video_ids
            }
            observer_blocks = {
                video_id: blocks_for_sample(
                    sample, "observer",
                    question=visual_questions[video_id], answer_type=answer_type,
                    options=sample.options, video_id=video_id,
                )
                for video_id in video_ids
            }
            focus_blocks = {
                video_id: blocks_for_sample(
                    sample, "focus",
                    question=visual_questions[video_id], answer_type=answer_type,
                    options=sample.options, video_id=video_id,
                )
                for video_id in video_ids
            }
            compare_blocks = blocks_for_sample(
                sample, "compare",
                question=state.question, answer_type=answer_type,
                options=sample.options,
            )

            # Phase 1
            state.observer_reports = self._phase1_observe(
                videos, state.question, prompt_blocks=observer_blocks,
                questions_by_video=visual_questions,
            )
            state.action_trace.append({"phase":1,"action":"OBSERVE_DONE","reports":len(state.observer_reports)})

            # Phase 2
            rs = self._form_checked(question=state.question, answer_type=answer_type,
                                    reports=state.observer_reports,
                                    prompt_blocks=form_blocks,
                                    video_ids=video_ids, state=state)
            state.reasoner_state = rs
            state.action_trace.append({"phase":2,"action":"FORM_DONE","decision":rs.decision,
                                        "candidates":1+len(rs.alternatives),
                                        "ambiguities":len(rs.ambiguities)})

            # Phase 3
            for rnd in range(1, MAX_EVIDENCE_ROUNDS + 1):
                if rs.decision != "NEED_EVIDENCE":
                    break
                executable = [a for a in rs.ambiguities if a.scope != "not_observable"]
                if not executable:
                    if rs.decision == "NEED_EVIDENCE":
                        rs.decision = "UNCERTAIN"
                    break

                new_evidence = self._execute_ambiguities(
                    executable,
                    videos,
                    state.question,
                    rnd,
                    focus_prompt_blocks=focus_blocks,
                    compare_prompt_blocks=compare_blocks,
                    visual_question_adapter=lambda value, video_id: (
                        visual_question_for_sample(sample, value, video_id)
                    ),
                )
                state.evidence_results.extend(new_evidence)
                state.action_trace.append({"phase":3,"round":rnd,"action":"EVIDENCE_DONE",
                                            "results":len(new_evidence)})

                review_blocks = blocks_for_sample(
                    sample, "review",
                    question=state.question, answer_type=answer_type,
                    options=sample.options, round_index=rnd,
                )
                prev_rs = rs
                rs = self.reasoner.review(
                    question=state.question, answer_type=answer_type,
                    reports=state.observer_reports, previous=rs,
                    new_evidence=new_evidence, round_index=rnd, video_ids=video_ids,
                    prompt_blocks=review_blocks,
                )
                # P0#2: deterministic gate — REVIEW may only reorder pairs
                # covered by a witness result; reject uncovered flips.
                flips = _gate_review_flips(prev_rs, rs, new_evidence)
                if flips:
                    state.action_trace.append({"phase":3,"round":rnd,
                                               "action":"REVIEW_FLIP_REJECTED",
                                               "uncovered": [list(f) for f in flips]})
                if _gate_inconclusive_choice_flip(prev_rs, rs, new_evidence):
                    state.action_trace.append({"phase":3,"round":rnd,
                                               "action":"REVIEW_CHOICE_FLIP_REJECTED",
                                               "reason":"all_new_evidence_inconclusive"})
                state.reasoner_state = rs
                state.action_trace.append({"phase":3,"round":rnd,"action":"REVIEW_DONE",
                                            "decision":rs.decision,
                                            "ambiguities":len(rs.ambiguities)})

            # Phase 3.5: timing guard (interval answers only)
            if answer_type == "interval" and self.timing_guard.config.enabled:
                guard_blocks = blocks_for_sample(
                    sample, "review",
                    question=state.question, answer_type=answer_type,
                    options=sample.options, round_index=10,
                )
                fb = self.timing_guard.first(rs)
                guard_round = 0
                while fb is not None and guard_round < self.config.timing_guard.max_feedback_rounds:
                    guard_round += 1
                    rs = self.reasoner.review(
                        question=state.question, answer_type=answer_type,
                        reports=state.observer_reports, previous=rs,
                        new_evidence=[], round_index=10 + guard_round,
                        video_ids=video_ids, timing_feedback=fb,
                        prompt_blocks=guard_blocks,
                    )
                    state.reasoner_state = rs
                    state.action_trace.append({"phase":3,"action":"TIMING_GUARD",
                                                "round":guard_round,
                                                "duration_seconds":_best_interval_duration(rs),
                                                "feedback_level":guard_round})
                    if rs.decision == "NEED_EVIDENCE":
                        executable = [a for a in rs.ambiguities if a.scope != "not_observable"]
                        if executable:
                            new_evidence = self._execute_ambiguities(
                                executable,
                                videos,
                                state.question,
                                10 + guard_round,
                                focus_prompt_blocks=focus_blocks,
                                compare_prompt_blocks=compare_blocks,
                                visual_question_adapter=lambda value, video_id: (
                                    visual_question_for_sample(sample, value, video_id)
                                ),
                            )
                            state.evidence_results.extend(new_evidence)
                            rs = self.reasoner.review(
                                question=state.question, answer_type=answer_type,
                                reports=state.observer_reports, previous=rs,
                                new_evidence=new_evidence, round_index=10 + guard_round,
                                video_ids=video_ids, timing_feedback=None,
                                prompt_blocks=guard_blocks,
                            )
                            state.reasoner_state = rs
                    fb = self.timing_guard.next(rs)

            # Phase 4
            if rs.best_answer and rs.best_answer.value is not None:
                state.final_answer = render_answer(rs.best_answer)
            state.stop_reason = "completed"
            state.action_trace.append({"phase":4,"action":"RENDER_DONE","prediction":str(state.final_answer)[:200]})

        except Exception as exc:
            state.stop_reason = "error"
            state.errors.append(f"{type(exc).__name__}: {exc}")
            state.api_calls = list(self._llm.calls[llm0:]) + list(self._vlm.calls[vlm0:])
            raise PipelineRunError(str(exc), state) from exc

        state.api_calls = list(self._llm.calls[llm0:]) + list(self._vlm.calls[vlm0:])
        return state

    # ── Phase 1 ───────────────────────────────────────────────────────

    def _phase1_observe(
        self,
        videos: list[VideoRef],
        question: str,
        *,
        prompt_blocks: list[str] | dict[str, list[str]] | None = None,
        questions_by_video: dict[str, str] | None = None,
    ) -> list[ObserverReport]:
        tasks = []
        for video in videos:
            budget = self.warm_sampler.frame_count(video, minimum=self.config.budget.min_warm_frames_per_video,
                                                    maximum=self.config.budget.scan_frames_per_video,
                                                    seconds_per_frame=self.config.video.scan_seconds_per_frame)
            ts = self.warm_sampler.sample(video, frame_budget=budget, question=question)
            vis = self.decoder.extract(video, ts)
            for w in self.warm_sampler.chunk(vis, frames_per_call=self.config.video.scan_frames_per_call,
                                              overlap_frames=self.config.video.scan_overlap_frames):
                tasks.append({"video_id": video.video_id, "inputs": w})

        def _run(task: dict[str, Any]) -> tuple[str, str, int, float, float]:
            visual_inputs = task["inputs"]
            video_id = task["video_id"]
            report = self.observer.observe(
                video_id=video_id,
                visual_inputs=visual_inputs,
                task_context=(
                    "Observe all objects, actions, states and temporal cues. "
                    "Later stages compare across videos to answer: "
                    + (
                        questions_by_video.get(video_id, question)
                        if questions_by_video
                        else question
                    )
                ),
                prompt_blocks=(
                    prompt_blocks.get(video_id)
                    if isinstance(prompt_blocks, dict)
                    else prompt_blocks
                ),
            )
            return (
                video_id,
                report,
                len(visual_inputs),
                visual_inputs[0].timestamp_seconds,
                visual_inputs[-1].timestamp_seconds,
            )

        raw: dict[str, list[tuple[str, int, float, float]]] = {}
        with ThreadPoolExecutor(max_workers=min(len(tasks),8)) as pool:
            for fut in as_completed([pool.submit(_run,t) for t in tasks]):
                vid,text,n,t0,t1 = fut.result()
                raw.setdefault(vid,[]).append((text,n,t0,t1))

        reports = []
        for v in videos:
            chunks = raw.get(v.video_id,[])
            if not chunks:
                reports.append(ObserverReport(video_id=v.video_id, report_text="(no observation)", frame_count=0, start_time=0.0, end_time=v.duration_seconds or 0.0))
            else:
                # Observer calls finish in arbitrary order (particularly with
                # a local VLM protected by a generation lock).  Evidence must
                # be rendered in temporal order rather than completion order:
                # otherwise a late-video uncertainty can precede the early
                # frame that actually identifies the object/action.
                ordered = sorted(chunks, key=lambda item: (item[2], item[3]))
                reports.append(ObserverReport(video_id=v.video_id, report_text="\n\n".join(t for t,_,_,_ in ordered),
                                               frame_count=sum(n for _,n,_,_ in ordered),
                                               start_time=min(t0 for _,_,t0,_ in ordered),
                                               end_time=max(t1 for _,_,_,t1 in ordered)))
        return reports

    # ── Phase 2 repair loop ───────────────────────────────────────────

    def _form_checked(
        self,
        *,
        question: str,
        answer_type: str,
        reports: list[ObserverReport],
        prompt_blocks: list[str] | None,
        video_ids: list[str],
        state: PipelineState,
    ) -> ReasonerState:
        """FORM plus a bounded repair loop for structurally impossible answers.

        A sequence answer that omits or duplicates a video id is already wrong
        before any evidence is gathered, and later rounds never notice the
        omission, so it is repaired here.  Decoding is greedy by default, so a
        blind re-ask would reproduce the rejected answer verbatim: the concrete
        validation errors go back into the prompt instead.  Every attempt lands
        in ``action_trace`` so the behaviour stays auditable.
        """
        def ask(feedback: list[str] | None) -> ReasonerState:
            return self.reasoner.form(
                question=question, answer_type=answer_type, reports=reports,
                prompt_blocks=prompt_blocks, repair_feedback=feedback,
            )

        rs = ask(None)
        if answer_type != "sequence":
            return rs
        errors = _answer_errors(rs.best_answer, video_ids)
        if not errors:
            return rs
        state.action_trace.append({"phase":2,"action":"FORM_INVALID","attempt":0,"errors":errors})
        for attempt in range(1, MAX_FORM_REPAIRS + 1):
            rs = ask(errors)
            errors = _answer_errors(rs.best_answer, video_ids)
            if not errors:
                state.action_trace.append({"phase":2,"action":"FORM_REPAIRED","attempt":attempt})
                return rs
            state.action_trace.append({"phase":2,"action":"FORM_INVALID","attempt":attempt,"errors":errors})
        missing = _fill_missing_videos(rs.best_answer, video_ids)
        state.action_trace.append({"phase":2,"action":"FORM_REPAIR_FALLBACK",
                                    "missing":missing,"errors":errors})
        return rs

    # ── Phase 3: scope-based routing ──────────────────────────────────

    def _execute_ambiguities(
        self,
        ambs: list[Ambiguity],
        videos: list[VideoRef],
        question: str,
        rnd: int,
        *,
        focus_prompt_blocks: list[str] | dict[str, list[str]] | None = None,
        compare_prompt_blocks: list[str] | None = None,
        visual_question_adapter: Callable[[str, str], str] | None = None,
    ) -> list[EvidenceResult]:
        results = []
        vmap = {v.video_id: v for v in videos}

        def _focus_prompt(value: str, video_id: str) -> str:
            blocks = (
                focus_prompt_blocks.get(video_id)
                if isinstance(focus_prompt_blocks, dict)
                else focus_prompt_blocks
            )
            if not blocks:
                return value
            return value + "\n\n" + "\n\n".join(blocks)

        def _localize(value: str, video_id: str) -> str:
            if visual_question_adapter is None:
                return value
            return visual_question_adapter(value, video_id)

        def _run_focus(
            amb: Ambiguity,
            target_video: str,
            span: tuple[float, float] | None,
            instruction: str,
        ) -> str | None:
            vref = vmap.get(target_video)
            if vref is None:
                return None
            expanded_span = self._expand_focus_span(vref, span)
            frames = self._extract_clip(
                vref,
                expanded_span,
                no_window_round_index=(rnd - 1) % MAX_EVIDENCE_ROUNDS,
            )
            witness = _localize(amb.witness, target_video)
            user_prompt = _focus_prompt(
                f"Video: {target_video}\nWitness: {witness}\n"
                f"Time window: {expanded_span}\n{instruction}",
                target_video,
            )
            return self._vlm.generate_text(
                role="ObserverAgent.FOCUS",
                system_prompt=prompt("observer"),
                user_prompt=user_prompt,
                visual_inputs=frames,
                generation_config=self.config.generation,
            ).strip()

        for amb in ambs:
            if amb.scope == "single_video":
                t = amb.targets[0] if amb.targets else None
                if t is None:
                    continue
                target_video = _explicit_witness_video_id(amb.witness, set(vmap)) or t.video
                routed_amb = amb
                if target_video != t.video:
                    routed_amb = replace(
                        amb,
                        targets=[replace(t, video=target_video), *amb.targets[1:]],
                    )
                raw = _run_focus(
                    routed_amb,
                    target_video,
                    t.span,
                    "Report only what you observe regarding this witness.",
                )
                if raw is None:
                    continue
                results.append(
                    EvidenceResult(
                        ambiguity=routed_amb,
                        scope=routed_amb.scope,
                        result_text=raw.strip(),
                    )
                )

            elif amb.scope == "multi_video_independent":
                for t in amb.targets:
                    raw = _run_focus(amb, t.video, t.span, "Report only what you observe.")
                    if raw is None:
                        continue
                    results.append(
                        EvidenceResult(
                            ambiguity=amb,
                            scope=amb.scope,
                            result_text=f"[{t.video}] {raw.strip()}",
                        )
                    )

            elif amb.scope == "joint_compare":
                if len(amb.targets) < 2:
                    continue
                va, vb = amb.targets[0].video, amb.targets[1].video
                vra, vrb = vmap.get(va), vmap.get(vb)
                if vra is None or vrb is None:
                    continue
                ca = self._extract_clip(vra, amb.targets[0].span)
                cb = self._extract_clip(vrb, amb.targets[1].span)
                raw = self.comparer.compare(
                    clip_a=ca,
                    clip_b=cb,
                    video_a=va,
                    video_b=vb,
                    question=amb.witness,
                    prompt_blocks=compare_prompt_blocks,
                )
                results.append(EvidenceResult(ambiguity=amb, scope=amb.scope, result_text=raw))

            # not_observable: skip
        return results

    def _expand_focus_span(
        self,
        video: VideoRef,
        span: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if span is None or span[0] != span[1]:
            return span
        timestamp = span[0]
        half_window = self.config.budget.single_timestamp_half_window_seconds
        duration = video.duration_seconds
        if duration is None:
            return (max(0.0, timestamp - half_window), timestamp + half_window)
        if timestamp < 0:
            return (0.0, min(duration, 2 * half_window))
        if timestamp > duration:
            return (max(0.0, duration - 2 * half_window), duration)
        return (
            max(0.0, timestamp - half_window),
            min(duration, timestamp + half_window),
        )

    def _extract_clip(
        self,
        video: VideoRef,
        span: tuple[float, float] | None,
        *,
        no_window_round_index: int | None = None,
    ) -> list[VisualInput]:
        if span is not None:
            s, e = span
            return self.microclips.build(video, start_seconds=max(0.0,s),
                                          end_seconds=min(video.duration_seconds or e, e),
                                          frame_budget=self.config.budget.default_focus_frames)
        if no_window_round_index is None:
            budget = min(
                self.config.budget.default_focus_frames,
                self.warm_sampler.frame_count(
                    video,
                    minimum=4,
                    maximum=self.config.budget.default_focus_frames,
                    seconds_per_frame=self.config.video.scan_seconds_per_frame,
                ),
            )
            timestamps = self.warm_sampler.sample(video, frame_budget=budget, question="")
        else:
            timestamps = self._no_window_focus_timestamps(video, no_window_round_index)
        return self.decoder.extract(video, timestamps)

    def _no_window_focus_timestamps(
        self,
        video: VideoRef,
        round_index: int,
    ) -> list[float]:
        """Return one chronological, interleaved whole-video fallback batch.

        The two evidence rounds use disjoint halves of a temporal grid.  Each
        half spans the full video, so REVIEW can stop after the first batch
        without paying for the second one.
        """
        if video.fps is None or video.fps <= 0 or video.frame_count is None:
            raise ValueError(f"video {video.video_id} lacks frame metadata for no-window FOCUS")
        if video.frame_count <= 0:
            raise ValueError(f"video {video.video_id} has no frames for no-window FOCUS")
        budget = min(
            self.config.budget.no_window_focus_frames_per_round,
            self.config.video.scan_frames_per_call,
        )
        candidate_count = min(video.frame_count, budget * MAX_EVIDENCE_ROUNDS)
        candidates = [
            round((index + 0.5) * video.frame_count / candidate_count - 0.5)
            for index in range(candidate_count)
        ]
        batch = candidates[round_index % MAX_EVIDENCE_ROUNDS :: MAX_EVIDENCE_ROUNDS]
        if not batch:
            batch = candidates
        return [frame_index / video.fps for frame_index in batch]


# ── helpers ──────────────────────────────────────────────────────────

def _answer_errors(ans: Answer | None, video_ids: list[str]) -> list[str]:
    """Structural errors that make the rendered answer unscoreable."""
    if ans is None or ans.value is None:
        return ["best_answer missing"]
    return validate_answer_payload(ans, video_ids)


def _fill_missing_videos(ans: Answer | None, video_ids: list[str]) -> list[str]:
    """Last resort: normalize a sequence so every video appears exactly once.

    Missing ids are appended in prompt order and duplicates collapsed.  A
    sequence of the wrong length already scores zero, so a deterministic
    completion cannot lose anything; the substitution is recorded as
    ``FORM_REPAIR_FALLBACK``.  Returns the ids that had to be added.
    """
    if ans is None or ans.type != "sequence" or not isinstance(ans.value, list):
        return []
    known = set(video_ids)
    order = list(dict.fromkeys(
        vid for vid in (str(item) for item in ans.value) if vid in known
    ))
    missing = [vid for vid in video_ids if vid not in set(order)]
    ans.value = order + missing
    return missing


def _gate_review_flips(
    previous: Any, candidate: Any, evidence: list[EvidenceResult]
) -> list[tuple[str, str]]:
    """P0#2 deterministic gate: reject reorders not covered by witness pairs.

    If the REVIEW candidate reordered any pair that no witness result
    covered, revert best_answer to the previous (pre-REVIEW) answer and
    return the rejected flip pairs. Sequence answers only.
    """
    from unicvr.core.state import uncovered_flips

    if previous.best_answer is None or candidate.best_answer is None:
        return []
    if previous.best_answer.type != "sequence" or candidate.best_answer.type != "sequence":
        return []
    before = previous.best_answer.value
    after = candidate.best_answer.value
    if not isinstance(before, list) or not isinstance(after, list):
        return []
    pairs = [e.ambiguity.pair for e in evidence]
    flips = uncovered_flips(before, after, pairs)
    if flips:
        candidate.best_answer = previous.best_answer
    return flips


_STRONG_INCONCLUSIVE_EVIDENCE_MARKERS = (
    "cannot determine",
    "cannot verify",
    "unable to determine",
    "unable to verify",
    "insufficient evidence",
    "witness: undetermined",
)
_WEAK_INCONCLUSIVE_EVIDENCE_MARKERS = ("not visible",)
_CONCLUSIVE_COUNT_RE = re.compile(
    r"(?:witness\s+)?qualifying\s+(?:instances/events\s+in\s+this\s+view:\s*|count:\s*)\d+",
    re.IGNORECASE,
)


def _gate_inconclusive_choice_flip(
    previous: Any, candidate: Any, evidence: list[EvidenceResult]
) -> bool:
    """Reject a choice change when every new visual result is inconclusive."""
    if previous.best_answer is None or candidate.best_answer is None or not evidence:
        return False
    if previous.best_answer.type != "choice" or candidate.best_answer.type != "choice":
        return False
    if previous.best_answer.value == candidate.best_answer.value:
        return False

    def inconclusive(item: EvidenceResult) -> bool:
        text = item.result_text.lower()
        if any(marker in text for marker in _STRONG_INCONCLUSIVE_EVIDENCE_MARKERS):
            return True
        return (any(marker in text for marker in _WEAK_INCONCLUSIVE_EVIDENCE_MARKERS)
                and _CONCLUSIVE_COUNT_RE.search(item.result_text) is None)

    if not all(inconclusive(item) for item in evidence):
        return False
    candidate.best_answer = previous.best_answer
    return True


_EXPLICIT_VIEW_RE = re.compile(r"\bview\s+([ab])\b", re.IGNORECASE)
_EXPLICIT_VIDEO_RE = re.compile(r"(?<![A-Za-z0-9_])(v\d+)(?![A-Za-z0-9_])", re.IGNORECASE)


def _explicit_witness_video_id(witness: str, available: set[str]) -> str | None:
    """Resolve one explicitly named witness view without guessing from context."""
    mentioned: set[str] = set()
    for match in _EXPLICIT_VIEW_RE.finditer(witness):
        mentioned.add("v1" if match.group(1).lower() == "a" else "v2")
    for match in _EXPLICIT_VIDEO_RE.finditer(witness):
        mentioned.add(match.group(1).lower())
    valid = mentioned & available
    return next(iter(valid)) if len(valid) == 1 else None


def _best_interval_duration(rs: Any) -> float | None:
    best = rs.best_answer
    if best is None or best.type != "interval":
        return None
    value = best.value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start, end = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
        if end > start:
            return round(end - start, 2)
    return None


def _derive_answer_type(sample: MultiVideoSample) -> str:
    meta = getattr(sample, "task_metadata", None) or {}
    task = meta.get("crossvid_task", "")
    if task in ("PSS",): return "sequence"
    if task in ("FSA",): return "interval"
    if task in ("CC", "NC", "MOC", "MSR", "PEA", "PI"): return "choice"
    if task in ("CCQA",): return "free_text"
    return "free_text"
