"""Shared synchronization and reference-track protocol for UAV tasks."""

from __future__ import annotations

from typing import Any

UAV_SYNCHRONIZATION_CONTRACT = """\
## Shared UAV Synchronization Contract
- View A and View B are synchronized cameras of one scene and share one
  timeline. An event at time t in one view occurs at the same time t in the
  other view. A_i and B_i with the same suffix are one physical reference;
  trust that identity and do not re-match it visually.
- For a conditional question, first locate the stated trigger in its original
  view and record one `t_trigger`; only then inspect the requested target view
  at that exact synchronized time.
- Never substitute a different event in the other view or use an object's
  general position across the clip in place of the stated trigger.
- Do not guess the trigger time. For a `parallel`/`side by side` trigger, use
  the proxy frame where the named boxes are abreast. For `passes`/`completely
  passes`, treat them as relative-motion events: use the relative-order flip
  that remains flipped, or the moment the named object has fully crossed the
  stated feature.
- `Completely leaves view` is the end of the reference's FIRST range (the
  first visible range).
- `Fully appears in`/`appears fully in` is the START of that object's visible range.
  The reference-track sidecar supplies source-frame
  ranges and their corresponding proxy seconds.
- Visible ranges are authoritative for presence of references. If `t_trigger`
  falls inside a reference's range, that reference is present even when an
  observer misses its small or occluded box.
- Focus the target view in a narrow window around `t_trigger`, normally
  plus/minus one second. Keep the established `t_trigger` fixed through
  REVIEW unless new direct trigger-view evidence disproves it.
"""


def _proxy_seconds(start: int, end: int, total: int) -> str:
    """Map a source-frame span onto the 12-second UAV proxy timeline."""
    duration = 12.0
    low = start / total * duration
    high = (end + 1) / total * duration
    return f"{low:.1f}-{high:.1f}s"


def _render_ranges(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    rendered = []
    for span in value[:3]:
        if not isinstance(span, list) or len(span) != 2:
            continue
        rendered.append(f"{span[0]}-{span[1]}")
    return ",".join(rendered) if rendered else "none"


def _render_proxy_ranges(value: Any, total: int) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    rendered = []
    for span in value[:3]:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(item, int) for item in span)
        ):
            continue
        rendered.append(_proxy_seconds(span[0], span[1], total))
    return ",".join(rendered) if rendered else "none"


def render_reference_track_block(evidence: list[dict[str, Any]]) -> str:
    """Render the same bounded, source-preserving sidecar for MOC and MSR."""
    lines = [
        "## Provided Reference-Track Evidence",
        "This sidecar describes only the annotated reference boxes already visible in",
        "the videos. It does NOT enumerate unlabelled target objects and is not the answer.",
        "View A and View B are synchronized cameras sharing the same timeline.",
        "A_i and B_i with the same suffix are the same physical reference.",
        "This identity is authoritative and MUST NOT be re-verified or overruled visually.",
    ]
    for item in evidence[:5]:
        reference = item.get("reference")
        track_id = item.get("raw_track_id")
        label = item.get("declared_label", "object")
        if not isinstance(reference, str) or not isinstance(track_id, int):
            continue
        lines.append(f"- {reference}: source_track={track_id}; declared={label}")
        views = item.get("views")
        if not isinstance(views, dict):
            continue
        for prefix in ("A", "B"):
            view = views.get(prefix)
            if not isinstance(view, dict):
                continue
            alias = view.get("alias", f"{prefix}?")
            total = view.get("source_frame_count", "?")
            spans = view.get("visible_frame_ranges")
            ranges = _render_ranges(spans)
            if isinstance(total, int) and total > 0:
                proxy_ranges = _render_proxy_ranges(spans, total)
                lines.append(
                    f"  - {alias}: visible_source_frames={ranges}/{total} "
                    f"(proxy {proxy_ranges})"
                )
            else:
                lines.append(f"  - {alias}: visible_source_frames={ranges}/{total}")
    lines.extend(
        [
            "Use these records only to bind reference identity, presence, and trigger time.",
            "For the first visible range, its end is `completely leaves`; the START of",
            "the first range is `fully appears`. The proxy videos draw each reference",
            "box throughout its visible range. These ranges are authoritative for presence:",
            "a range covering the trigger moment means the reference is present then.",
            "For position questions, read the drawn box at the trigger moment.",
        ]
    )
    return "\n".join(lines)
