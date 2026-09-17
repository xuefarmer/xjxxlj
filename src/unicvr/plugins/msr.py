"""MsrForceChoicePlugin — Multi-Scene Recognition adaptation.

MSR asks for a spatial-relation option between views. The probe run
showed the model answering "unknown" / "Not visible simultaneously"
instead of picking an option. This plugin forbids refusal and requires
a best-effort option choice grounded in the reports.

Regression finding (2026-08-14, 10-item real-API run): all 5 failures were
cross-view temporal-anchoring errors — the model treated view A and view B
as separate clips, or substituted a different event (e.g. "B4 leaves view
B") for the stated trigger ("A4 leaves view A"), reading the answer at the
wrong moment. The plugin therefore also injects the synchronized-camera
contract and the reference-track visibility sidecar (the same evidence
MOC already consumes via its reference_track_block).
"""

from __future__ import annotations

from typing import Any

from unicvr.plugins.base import TaskPlugin
from unicvr.plugins.uav import UAV_SYNCHRONIZATION_CONTRACT, render_reference_track_block

_MSR_BLOCK = """\
## Answer Requirement
- You MUST pick one option letter. "unknown", "not visible", and refusal
  answers are NEVER valid.
- If no option is fully confirmed, choose the option most consistent with
  the reports and say exactly what is missing in the rationale.
"""

_MSR_CONTRACT = """\
## MSR Cross-View Contract
- When the question asks where one reference is located RELATIVE TO
  another (e.g. "where is {B4} located relative to {B1}"), read BOTH drawn
  boxes at the trigger moment and report their mutual position
  (front/behind/left/right/above/below) — not either box's own position in
  the frame.
- For trajectory questions (e.g. "final driving path"), read the
  reference's drawn boxes across its whole visible range and describe the
  path they trace; do not answer from a single frame or a witness's
  summary of the path.
- The proxy videos draw every reference box at every sampled frame inside
  its visible range. Read the drawn box at the trigger moment: its
  on-screen position IS the answer.
- The answer is a FRAME REGION of the answer reference in the target view
  at the trigger moment. The observer's narrative position descriptions
  (lanes, road sides, distances, "following X") are too coarse to answer a
  frame-region question. To read the drawn box, request focused visual
  evidence on the ANSWER reference in the target view and SET THE TIME
  WINDOW to the trigger second plus/minus one second (e.g. [2.8, 3.4] for
  a 3.1s trigger) — without a narrow window the focus frames skip the
  trigger moment and the position is lost. Answer from what the box shows
  in the frame, never from lane descriptions.
- Witness/observer reports about a reference's position are hypotheses,
  not ground truth — a witness watches one view and can confuse the
  reference with a nearby box or misreport the axis. If a position claim
  conflicts with the drawn box at the trigger moment, the claim is
  mistaken: never flip an anchored reading to match it. A witness who
  reports a reference "not visible" at a moment inside its visible range
  has missed a small or occluded box; the box IS there.
"""


class MsrForceChoicePlugin(TaskPlugin):
    name = "msr_force_choice"

    def form_block(
        self,
        *,
        question: str,
        answer_type: str,
        options: list[str],
    ) -> str | None:
        if answer_type != "choice":
            return None
        return (_MSR_BLOCK + "\n\n" + UAV_SYNCHRONIZATION_CONTRACT
                + "\n\n" + _MSR_CONTRACT)

    def review_block(
        self,
        *,
        question: str,
        answer_type: str,
        options: list[str],
        round_index: int,
    ) -> str | None:
        if answer_type != "choice":
            return None
        return (_MSR_BLOCK + "\n\n" + UAV_SYNCHRONIZATION_CONTRACT
                + "\n\n" + _MSR_CONTRACT)

    _OBSERVER_BLOCK = """\
## MSR Observation Requirements
- The colored boxes drawn on the frames are the reference tracks. For EVERY
  drawn box, report its position within the VIDEO FRAME — top / bottom /
  left / right / center of the frame or a corner — with the seconds when it
  holds that position. "Top of the frame" means the upper part of the
  picture, NOT the leading edge of the road.
- A road LANE is not a frame region: a vehicle driving in the left lane can
  still appear in the center or right of the frame. Report where the box
  sits in the PICTURE, and only then (optionally) which lane it is in.
- Precisely report the moment each box completely leaves the frame (its box
  is no longer drawn) and the moment it first appears, in seconds.
- The two views are synchronized cameras of one scene; an event reported in
  this view at time t happens at the same time t in the other view.
"""

    def visual_block(
        self,
        *,
        phase: str,
        question: str,
        answer_type: str,
        options: list[str],
        video_id: str | None = None,
    ) -> str | None:
        """MSR-only observation requirements for Observer/Focus phases."""
        del answer_type, options, video_id, question
        if phase not in {"observer", "focus"}:
            return None
        return UAV_SYNCHRONIZATION_CONTRACT + "\n\n" + self._OBSERVER_BLOCK

    def reference_track_block(self, evidence: list[dict[str, Any]]) -> str | None:
        """Render a bounded sidecar for the five already-visible references."""
        return render_reference_track_block(evidence)
