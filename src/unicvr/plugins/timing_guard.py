"""TimingGuard plugin — extreme-only interval-boundary refinement for FSA.

Feeds duration diagnostics back into the REVIEW prompt, but ONLY for
extreme intervals, so the model keeps its sense of the dataset's real
duration distribution instead of regressing everything to the median.

The FSA GT interval duration distribution is wide (median 18s, p10 6s,
p90 45s). The guard leaves the normal band [p10, p90] untouched:

1. OVERLONG  (duration >= p90, default 45s): remind the model to
   compress — keep the core action, drop prep/cleanup padding — while
   explicitly exempting genuinely long actions (slow cooking, long
   whisking), so it doesn't over-tighten real long intervals.
2. UNDERSHORT (duration <= p10, default 6s): the interval is implausibly
   short; ask the model to re-derive the start/end boundaries — the
   action was likely truncated. Optionally propose boundary-probing
   witnesses.

The plugin is answer-type gated (interval answers only) and config-gated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from unicvr.config import TimingGuardConfig as TimingGuardConfigModel
    from unicvr.core.state import ReasonerState

_EPSILON = 1e-6

FeedbackKind = Literal["OVERLONG", "UNDERSHORT"]


def _is_overlong(duration: float, threshold: float) -> bool:
    # strict: the band upper bound itself is NOT extreme
    return duration > threshold + _EPSILON


def _is_undershort(duration: float, threshold: float) -> bool:
    # strict: the band lower bound itself is NOT extreme
    return duration < threshold - _EPSILON


def _interval_of(state: ReasonerState) -> tuple[float, float] | None:
    best = state.best_answer
    if best is None or best.type != "interval":
        return None
    value = best.value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(x, (int, float)) for x in value)
    ):
        start, end = float(value[0]), float(value[1])
        if end > start:
            return (start, end)
    return None


def _compress_feedback(duration: float, threshold: float) -> str:
    return (
        f"Timing feedback: your predicted interval is {duration:.1f}s, "
        f"implausibly long for this task (typical range ends around "
        f"{threshold:g}s). Compress it: keep only the core action span, "
        f"exclude prep/cleanup padding, and tighten the start/end "
        f"boundaries. EXCEPTION: if the action genuinely lasts a long time "
        f"(slow cooking, continuous stirring), keep its true duration — "
        f"only trim what evidence shows is not part of the action."
    )


def _undershort_feedback(duration: float, threshold: float) -> str:
    return (
        f"Timing feedback: your predicted interval is only {duration:.1f}s, "
        f"implausibly short for this task (typical range starts around "
        f"{threshold:g}s). The action was probably truncated — the interval "
        f"covers only the most visible core moment. Re-derive the true "
        f"start and end: check the observer reports for when the action "
        f"actually begins (e.g. 'starts whisking', 'picks up the tool') and "
        f"when it ends (e.g. 'puts it down', 'turns away'). If the reports "
        f"cannot pin the edges, propose a new ambiguity witness that probes "
        f"the region just before/after your current interval."
    )


class TimingGuard:
    """Extreme-only duration-feedback state machine for interval answers.

    Usage:
        guard = TimingGuard(config.timing_guard)
        fb = guard.first(state)      # after FORM
        fb = guard.next(state)       # after each guarded REVIEW
    `fb` is a feedback string to inject into the REVIEW prompt, or None when
    the interval is inside the normal band or the feedback budget is spent.
    """

    def __init__(self, config: TimingGuardConfigModel) -> None:
        self.config = config
        self._round = 0

    def _diagnose(
        self, state: ReasonerState
    ) -> tuple[FeedbackKind, float] | None:
        interval = _interval_of(state)
        if interval is None:
            return None
        duration = interval[1] - interval[0]
        if _is_overlong(duration, self.config.extreme_overlong_seconds):
            return ("OVERLONG", duration)
        if _is_undershort(duration, self.config.extreme_undershort_seconds):
            return ("UNDERSHORT", duration)
        return None

    def _feedback_for(self, kind: FeedbackKind, duration: float) -> str:
        if kind == "OVERLONG":
            return _compress_feedback(duration, self.config.extreme_overlong_seconds)
        return _undershort_feedback(duration, self.config.extreme_undershort_seconds)

    def first(self, state: ReasonerState) -> str | None:
        self._round = 0
        if not self.config.enabled:
            return None
        diag = self._diagnose(state)
        if diag is None:
            return None
        self._round = 1
        return self._feedback_for(*diag)

    def next(self, state: ReasonerState) -> str | None:
        if not self.config.enabled or self._round == 0:
            return None
        if self._round >= self.config.max_feedback_rounds:
            return None
        diag = self._diagnose(state)
        if diag is None:
            return None
        self._round += 1
        return self._feedback_for(*diag)

    @property
    def round(self) -> int:
        return self._round
