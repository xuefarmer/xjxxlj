"""General multi-video conditional-counting guidance for MOC.

The shared pipeline and decision-state schema stay unchanged.  This plugin
only states the task contract that every MOC question follows: establish the
question's scope, count identity-preserving evidence, and map the resulting
integer to an option exactly.  It deliberately contains no question-template,
scene, answer-letter, object-category, or count-specific routing.
"""

from __future__ import annotations

import re
from typing import Any

from unicvr.plugins.base import TaskPlugin
from unicvr.plugins.uav import UAV_SYNCHRONIZATION_CONTRACT, render_reference_track_block

# ── Question parsing (lightweight, fallback-safe) ──────────────────────

_COLOR_WORDS = (
    "light blue", "blue-green", "yellow", "red", "blue", "black",
    "white", "green", "grey", "cyan",
)
_TYPE_WORDS = ("cycles", "sedans", "cars", "buses", "trucks", "vehicles")

# (verb phrase, normalized relation key) — first hit wins
_RELATION_PHRASES = (
    ("passes by", "pass-by"), ("pass by", "pass-by"),
    (" pass ", "pass-by"),
    ("overtakes", "overtake"), ("overtake", "overtake"),
    ("ahead of", "same-direction-ahead"), (" behind ", "same-direction-behind"),
    ("parked by the roadside", "parked"), ("parked in front of", "parked-front"),
    ("parked", "parked"),
    ("completely leaves", "leave-view"), ("leaves view", "leave-view"),
    ("level with", "level-with"), ("appear", "appear"),
    ("currently driving", "driving"), ("driving in", "driving"),
    ("show clear displacement", "moving"),
)

_RELATION_TEXT = {
    "pass-by": "passing by (approaching then leaving)",
    "overtake": "overtaking (same direction, behind to ahead)",
    "parked": "parked/stationary",
    "parked-front": "parked in front of",
    "leave-view": "leaving the view",
    "level-with": "being level with",
    "appear": "appearing in the scene",
    "driving": "driving in a lane",
    "same-direction-ahead": "travelling ahead in the same direction",
    "same-direction-behind": "travelling behind in the same direction",
    "moving": "showing clear displacement during the requested clip",
}

_CATEGORY_HINTS = {
    "cycles": (
        "`cycles` = two-wheeled vehicles/riders: bicycles, e-bikes, scooters, "
        "motorcycles, mopeds. Count vehicles, not riders. Some cycles are "
        "annotated 'person' — decide by appearance/motion, not the label."
    ),
}

def _parse_moc_question(question: str) -> dict[str, Any]:
    """Extract {category, color, reference, relation} from a MOC question.

    Deliberately shallow: every field is optional and every failure falls
    back to a generic focus, so a parsing mistake can never distort the
    observation contract itself.
    """
    semantic_question = _count_clause(question)
    lower = semantic_question.lower()
    # An enumerated target set may occur first ("Among {A2}, {A3} ...").
    # Resolve the *grammatical* reference before falling back to first alias.
    reference = _relation_reference(semantic_question)
    relation = "?"
    last_pos = -1
    for phrase, norm in _RELATION_PHRASES:
        # last hit in sentence wins: "red cars parked ... does {A1} pass"
        # -> pass-by, because the passing relation is what must be tracked
        pos = lower.find(phrase)
        if pos >= 0 and pos > last_pos:
            last_pos, relation = pos, norm
    phrase = ""
    m = re.search(
        r"how many (.+?) (?:does |are |is |pass |overtake |parked |in |"
        r"appear |leave |drive |currently |at |do you see |have |turn |"
        r"show |completely |fully |oncoming |traveling |moving |\?)",
        lower,
    )
    if m:
        phrase = m.group(1).strip()
    if not phrase:
        m = re.search(r"how many (.+?)$", lower)
        if m:
            phrase = m.group(1).strip()
    for tail in (" in total", " do you see", " have completely", " completely",
                 " traveling", " moving", " on the same side"):
        if phrase.endswith(tail):
            phrase = phrase[: -len(tail)].rstrip()
    color = next((c for c in _COLOR_WORDS if c in phrase), None)
    ctype = next((t for t in _TYPE_WORDS if t in phrase), None)
    category = " ".join(w for w in (color, ctype) if w) if ctype else phrase
    return {
        "_question": semantic_question,
        "category": category or "the target category in the question",
        "color": color,
        "reference": reference,
        "references": list(
            dict.fromkeys(re.findall(r"\{([AB]\d)\}", semantic_question))
        ),
        "relation": relation,
        "relation_clauses": _relation_clauses(semantic_question),
        "category_hint": _CATEGORY_HINTS.get(ctype or "", None),
    }


def _count_clause(question: str) -> str:
    """Return the answer-bearing clause rather than an earlier trigger clause."""
    match = re.search(r"\bhow\s+many\b", question, re.IGNORECASE)
    return question[match.start():] if match else question


def _conditional_counting_block(question: str) -> str:
    """Render the trigger/count split used by synchronized MOC questions."""
    count_clause = _count_clause(question).strip()
    trigger = question[: max(0, len(question) - len(count_clause))].strip(" ,:;")
    if trigger:
        trigger_line = f"- Trigger clause (time only): `{trigger}`."
    else:
        trigger_line = "- No separate trigger clause was detected; use the stated time boundary."
    return "\n".join(
        [
            "## Trigger / Count Separation",
            trigger_line,
            f"- Count clause (defines the answer set): `{count_clause}`.",
            "- A When/After/Once trigger establishes the synchronized timestamp only. "
            "Do not copy its actor, action, reference, color, category, or view into "
            "the count clause.",
            "- Freeze the count clause's view, category, relation, and timepoint through FORM, "
            "FOCUS, and REVIEW. Trigger anchors are not counted unless the count clause "
            "explicitly includes them.",
        ]
    )


def _relation_reference(question: str) -> str | None:
    patterns = (
        # "how many targets pass {A1}" / "overtake {A1}"
        r"\b(?:pass(?:es)?|pass\s+by|overtake(?:s)?)\s+\{([AB]\d)\}",
        # "how many targets travel ahead of {A1}" / "behind {A1}"
        r"\b(?:ahead\s+of|behind)\s+\{([AB]\d)\}",
        # "how many targets does {A1} overtake"
        r"\bdoes\s+\{([AB]\d)\}\s+(?:pass(?:es)?|pass\s+by|overtake(?:s)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1)
    match = re.search(r"\{([AB]\d)\}", question)
    return match.group(1) if match else None


def _relation_clauses(question: str) -> list[str]:
    """Render every anchor-bearing condition, retaining grammatical actors."""
    clauses = []
    patterns = (
        (r"\bdoes\s+\{([AB]\d)\}\s+overtake", "{0} overtakes each counted target"),
        (r"\bdoes\s+\{([AB]\d)\}\s+pass(?:es|\s+by)?", "{0} passes each counted target"),
        (r"\b(?:pass(?:es)?|pass\s+by)\s+\{([AB]\d)\}", "each counted target passes {0}"),
        (r"\bovertake(?:s)?\s+\{([AB]\d)\}", "each counted target overtakes {0}"),
        (
            r"\bahead\s+of\s+\{([AB]\d)\}",
            "each counted target is ahead of {0} in the same direction",
        ),
        (
            r"\bbehind\s+\{([AB]\d)\}",
            "each counted target is behind {0} in the same direction",
        ),
    )
    for pattern, template in patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            clause = template.format(match.group(1))
            if clause not in clauses:
                clauses.append(clause)
    return clauses


_DIRECTIONAL_VERBS = ("overtake", "pass", "overtakes", "passes", "passes by")
_VIEW_REFERENCE_RE = re.compile(r"(?<![A-Za-z0-9_])([AB])(\d+)(?![A-Za-z0-9_])")


def _localize_reference_aliases(question: str, video_id: str) -> str:
    """Rewrite every A_i/B_i mention to the alias visible in one UAV view."""
    prefix = {"v1": "A", "v2": "B"}.get(video_id)
    if prefix is None:
        return question
    return _VIEW_REFERENCE_RE.sub(
        lambda match: prefix + match.group(2),
        question,
    )


def _direction_line(question: str, reference: str | None,
                    relation: str) -> str | None:
    """Resolve who acts on whom from the question grammar.

    The reference appearing BEFORE the action verb makes it the acting
    subject ("how many cycles does {B5} overtake" -> B5 overtakes cycles);
    appearing AFTER the verb makes the target instances the actors
    ("how many cycles overtake {B4}" -> cycles overtake B4).
    """
    if not reference or relation not in ("overtake", "pass-by"):
        return None
    ref_m = re.search(re.escape(reference), question)
    # Keep this compatible with the Python parser used by the distillation
    # environment; some nodes still reject assignment expressions here.
    verb_positions = []
    for verb in _DIRECTIONAL_VERBS:
        match = re.search(re.escape(verb), question)
        if match:
            verb_positions.append(match.start())
    if not ref_m or not verb_positions:
        return None
    verb_at = min(verb_positions)
    if ref_m.start() < verb_at:
        return (f"- Direction: {reference} passes/overtakes the targets; do not "
                f"reverse this relation.")
    return (f"- Direction: the targets pass/overtake {reference}; do not reverse "
            f"this relation.")


def _focus_lines(parsed: dict[str, Any]) -> str:
    """One-shot focus instruction rendered from the parsed question."""
    lines = ["## MOC Observation Focus",
             f"- Target category: {parsed['category']}."]
    hint = parsed.get("category_hint")
    if hint:
        lines.append(f"- {hint}")
    ref = parsed.get("reference")
    refs = parsed.get("references") or []
    clauses = parsed.get("relation_clauses") or []
    if clauses:
        lines.append("- Required anchor conditions (logical AND when more than one):")
        lines.extend(f"  - {clause}." for clause in clauses)
    else:
        rel = _RELATION_TEXT.get(parsed["relation"])
        if rel:
            lines.append(f"- Relation/condition to track: {rel}.")
    if len(refs) > 1:
        lines.append(
            f"- All condition anchors in the original question: {', '.join(refs)}. "
            "Preserve their views and roles."
        )
    if ref and len(refs) <= 1:
        lines.append(f"- Reference in this view: {ref}.")
        direction = _direction_line(parsed["_question"], ref, parsed["relation"])
        if direction:
            lines.append(direction)
    return "\n".join(lines)

_MOC_BLOCK = """\
## MOC Rules
- `cycles` means bicycles, e-bikes, scooters, motorcycles, mopeds, and similar
  two-wheelers. Preserve the action direction stated by the question.
- View A and View B are synchronized views of one scene. A_i and B_i with the
  same suffix are one reference identity. Apply the explicit counting scope
  stated below before deciding whether to merge cross-view observations.
- Per-view reports separate `Visible target candidates in this view` (the
  pre-filter pool) from `Qualifying instances/events in this view`. Base the
  final integer on the qualifying count; never let the visible pool stand in
  for the answer count.
- Derive the qualifying integer from visual evidence, then map it exactly to
  the option. Never change a count merely to fit the available options.
"""

_SINGLE_VIEW_SCOPE_RE = re.compile(r"\bin\s+view\s+([ab])\b", re.IGNORECASE)
_ENUMERATED_TARGET_RE = re.compile(r"\bamong\s+\{[AB]\d+\}", re.IGNORECASE)
_JOINT_SCOPE_RE = re.compile(
    r"\b(?:across\s+both|in\s+both\s+view\s*[ab]\s+and\s+view\s*[ab])",
    re.IGNORECASE,
)


def _has_cross_view_roles(question: str) -> bool:
    """Return whether aliases and the answer scope deliberately name different views."""
    refs = set(re.findall(r"\{([AB]\d)\}", question))
    prefixes = {ref[0] for ref in refs}
    explicit_views = {
        match.group(1).upper() for match in _SINGLE_VIEW_SCOPE_RE.finditer(question)
    }
    return (
        len(prefixes) > 1
        or len(explicit_views) > 1
        or ("A" in prefixes and "B" in explicit_views)
        or ("B" in prefixes and "A" in explicit_views)
        or bool(_JOINT_SCOPE_RE.search(question))
    )


def _scope_block(question: str) -> str:
    """Resolve whether the question explicitly narrows the counting view.

    Legacy CrossVid MOC wording does not name one view and therefore keeps the
    union rule above.  Semantic-v1 MOC questions deliberately say ``in view
    A/B`` so their answer is a single-camera count; treating those as a union
    would add an unasked-for second set of vehicles.
    """
    count_clause = _count_clause(question)
    match = _SINGLE_VIEW_SCOPE_RE.search(count_clause)
    enumerated = bool(_ENUMERATED_TARGET_RE.search(question))
    if _JOINT_SCOPE_RE.search(question):
        scope = ("## Counting Scope\n"
                 "- This is a multi-view joint-count question. A_i and B_i with the same "
                 "suffix are one "
                 "physical identity and count at most once.\n"
                 "- Count an identity only if it satisfies the stated condition in BOTH "
                 "views; an instance "
                 "that qualifies in just one view is excluded, not unioned in.")
        if enumerated:
            scope += ("\n- The phrase `Among {...}` is a closed candidate set. Count only "
                      "those explicitly "
                      "listed matched identities; do not include any unlabelled vehicle.")
        return scope
    if match:
        view = match.group(1).upper()
        other = "B" if view == "A" else "A"
        scope = ("## Counting Scope\n"
                f"- This question explicitly asks for view {view}. Count only physical "
                "instances visible "
                f"in view {view}; do not add candidates seen only in view {other}.\n"
                "- The other synchronized view may be used as supporting visual evidence "
                "for identities, "
                f"but never expands the counted set.")
        if enumerated:
            scope += ("\n- The phrase `Among {...}` is a closed candidate set. Count only "
                      "those explicitly "
                      "listed aliases; do not include any unlabelled vehicle in the scene.")
        return scope
    return ("## Counting Scope\n"
            "- The question does not name a single view. Count the cross-view union, "
            "merging duplicate "
            "physical identities rather than adding per-view counts blindly.")

_MOC_FORM_PROTOCOL = """\
## Counting Protocol
- Build a stable candidate ledger before choosing an option. Give every row a
  persistent identity, counted/not-counted/undetermined verdict, and the exact
  count-clause predicate it satisfies or fails.
- Accounting: the qualifying integer equals the number of qualified rows in
  that view's instance table. Before choosing an integer, state per view
  `Visible=.., Qualifying=..` and which instances qualify.
- Same-suffix A_i/B_i are one identity: a qualifying event seen in both views
  counts once. If the two views' qualifying counts disagree, do not average
  or silently pick one — the disagreement is itself evidence of a missed look.
- `Visible != Qualifying` is normal when color/type/state/relation filters
  exclude objects; it is not by itself a reason to request more evidence.
- Request evidence instead of guessing when a candidate lacks a verdict, the
  instance-table arithmetic is incomplete, or a required cross-view identity/
  conjunction is unresolved. Prefer a `joint_compare` for a disputed shared
  identity and a `single_video` focus for a view-local unresolved candidate.
"""

_MOC_REVIEW_PROTOCOL = """\
## Review Protocol
- Preserve the previous candidate ledger. New local evidence may update only
  the explicitly inspected candidate rows; it must not replace the global
  ledger or be treated as a fresh full-scene count.
- Re-derive the integer from the updated ledger. State the previous total, the
  named row-level additions/removals, and the resulting total before mapping
  that integer to an option.
- A witness about the wrong view, category, relation, or timepoint is
  inapplicable. A witness that cannot determine its requested fact changes no
  row and must not change the answer.
- Never reinterpret a trigger action as a predicate on the counted objects.
"""

_MOC_VISUAL_BLOCK = """\
## MOC Visual Semantics
- `cycles` means two-wheeled road vehicles/riders, including bicycles,
  e-bikes, scooters, motorcycles, and mopeds.
- View A/v1 and View B/v2 are synchronized cameras of the same scene.
- Judge motion relative to the reference and stable scene landmarks, not raw
  screen coordinates alone; the UAV camera itself moves.
"""

_MOC_OBSERVE_CONTRACT = """\
Track every visible target candidate, including small or brief instances, and
state its approximate time and evidence for the count-clause predicate. After
listing the initially salient candidates, run a second explicit pass over
stationary, parked, small, distant, or partially occluded areas and append any
missed candidates — do not stop at the first pass.
For `At the start` questions, qualify objects only at the initial boundary;
for `At the end` questions, qualify them only at the final boundary. Use the
remaining frames to infer motion direction, never to expand the timepoint set.
Every candidate row must carry a count-clause verdict (qualifies/excluded/
undetermined). When the count clause requests a motion relation, also report
its relation verdict (passes/overtakes/unrelated/undetermined). Mark
undetermined explicitly rather than omitting the row.
End with:
- `Visible target candidates in this view: N`
- `Qualifying instances/events in this view: M`
Use the best-supported temporal estimate; a missing exact alignment frame or
different lane alone is not a reason to discard an otherwise supported event.
"""

_MOC_COLOR_BOUNDARY = """\
- Color boundary: the target color includes darker or washed-out variants of
  the same hue (e.g. dark red / burgundy / maroon count as `red`). Include
  them as candidates and mark color confidence per instance (e.g. `red (clear)`
  or `red? (dark variant)`). Never drop a candidate solely because its color
  is darker or less saturated.
"""

_MOC_FOCUS_CONTRACT = """\
For this MOC witness, report the relevant target instances, timestamps, and
target-reference motion. This is local delta evidence.
It is not a replacement for the full candidate ledger. State the observed view and repeat the count
clause's category/relation/timepoint before judging candidates. Identify each
changed candidate row; if none can be resolved, say `Witness: undetermined`
and do not invent a zero. End conclusive evidence with
`Witness qualifying count: M`.
"""

_MOC_COMPARE_ADDENDUM = """\
The views are synchronized. Merge matching physical instances across views;
do not add duplicate observations. A_i/B_i with the same suffix are already
one authoritative reference identity. Report the merged qualifying count in
the normal comparer JSON response.
"""


class MocCountingPlugin(TaskPlugin):
    """Inject the same general counting contract into every MOC question."""

    name = "moc_counting"

    def visual_block(
        self,
        *,
        phase: str,
        question: str,
        answer_type: str,
        options: list[str],
        video_id: str | None = None,
    ) -> str:
        """Provide MOC-only facts to Observer, Focus, and Compare."""
        del answer_type, options
        if phase != "compare" and video_id is not None:
            # Bind the reference alias to the view actually being watched:
            # v1 draws A_i labels, v2 draws B_i. Idempotent when the pipeline
            # already localized the question via visual_question().
            cross_view = _has_cross_view_roles(question)
            if not cross_view:
                question = _localize_reference_aliases(question, video_id)
        parsed = _parse_moc_question(question)
        focus = _focus_lines(parsed)
        if phase != "compare" and parsed.get("color"):
            focus += "\n\n" + _MOC_COLOR_BOUNDARY
        if phase == "compare":
            return (UAV_SYNCHRONIZATION_CONTRACT + "\n\n" + _MOC_VISUAL_BLOCK + "\n\n" + focus
                    + "\n\n" + _scope_block(question)
                    + "\n\n" + _conditional_counting_block(question)
                    + "\n\n" + _MOC_COMPARE_ADDENDUM)
        if phase == "focus":
            return (UAV_SYNCHRONIZATION_CONTRACT + "\n\n" + _MOC_VISUAL_BLOCK + "\n\n" + focus
                    + "\n\n" + _scope_block(question)
                    + "\n\n" + _conditional_counting_block(question)
                    + "\n\n" + _MOC_FOCUS_CONTRACT)
        return (UAV_SYNCHRONIZATION_CONTRACT + "\n\n" + _MOC_VISUAL_BLOCK + "\n\n" + focus
                + "\n\n" + _scope_block(question)
                + "\n\n" + _conditional_counting_block(question)
                + "\n\n" + _MOC_OBSERVE_CONTRACT)

    def visual_question(self, *, question: str, video_id: str) -> str:
        """Mechanically bind A_i/B_i references to the current visual view."""
        # Explicit cross-view questions use A_i and B_j for different logical
        # clauses.  Rewriting both to one prefix destroys the intersection or
        # trigger/count relationship.  Each observer can inspect the anchors
        # belonging to its own view while preserving the original grammar.
        if _has_cross_view_roles(question):
            return question
        return _localize_reference_aliases(question, video_id)

    def reference_track_block(self, evidence: list[dict[str, Any]]) -> str | None:
        """Render a bounded sidecar for the five already-visible references."""
        return render_reference_track_block(evidence)

    def form_block(
        self,
        *,
        question: str,
        answer_type: str,
        options: list[str],
    ) -> str | None:
        del answer_type, options
        return (UAV_SYNCHRONIZATION_CONTRACT + "\n\n" + _MOC_BLOCK
                + "\n\n" + _scope_block(question)
                + "\n\n" + _conditional_counting_block(question)
                + "\n\n" + _MOC_FORM_PROTOCOL)

    def review_block(
        self,
        *,
        question: str,
        answer_type: str,
        options: list[str],
        round_index: int,
    ) -> str | None:
        del answer_type, options, round_index
        return (UAV_SYNCHRONIZATION_CONTRACT + "\n\n" + _MOC_BLOCK
                + "\n\n" + _scope_block(question)
                + "\n\n" + _conditional_counting_block(question)
                + "\n\n" + _MOC_REVIEW_PROTOCOL)
