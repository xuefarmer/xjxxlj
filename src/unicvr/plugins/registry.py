"""Task → plugin routing. Each CrossVid task composes its own plugin list.

Family-level adaptations (ChoiceAnswerPlugin) are shared by the 6 choice
tasks; task-specific adaptations (MOC counting, MSR force-choice) are
added only to their own task. PSS needs no plugin (covered-pair gate
handles its REVIEW correctness deterministically); FSA is served by the
existing TimingGuard.
"""

from __future__ import annotations

from typing import Any

from unicvr.plugins.base import TaskPlugin, collect_blocks
from unicvr.plugins.choice import ChoiceAnswerPlugin
from unicvr.plugins.moc import MocCountingPlugin
from unicvr.plugins.msr import MsrForceChoicePlugin

PLUGIN_REGISTRY: dict[str, list[TaskPlugin]] = {
    "CC": [ChoiceAnswerPlugin()],
    "NC": [ChoiceAnswerPlugin()],
    "PEA": [ChoiceAnswerPlugin()],
    "PI": [ChoiceAnswerPlugin()],
    "MOC": [ChoiceAnswerPlugin(), MocCountingPlugin()],
    "MSR": [ChoiceAnswerPlugin(), MsrForceChoicePlugin()],
    "FSA": [],          # served by TimingGuard (pipeline phase 3.5)
    "PSS": [],          # covered-pair gate handles REVIEW correctness
    "CCQA": [],         # free-text; evaluation-side matching (TBD)
}


def plugins_for_task(task: str) -> list[TaskPlugin]:
    """Return the registered plugin list for a task (empty if unknown)."""
    return list(PLUGIN_REGISTRY.get(task, []))


def blocks_for_sample(
    sample: Any,
    phase: str,
    *,
    question: str,
    answer_type: str,
    options: list[str],
    round_index: int = 0,
    video_id: str | None = None,
) -> list[str]:
    """Collect prompt blocks for a sample's task (task routing stays here).

    The pipeline calls this single entry point; it never sees task names.
    """
    metadata = getattr(sample, "task_metadata", None) or {}
    task = metadata.get("crossvid_task", "")
    plugins = plugins_for_task(task)
    if phase in {"form", "review"}:
        blocks = collect_blocks(
            plugins, phase=phase, question=question, answer_type=answer_type,
            options=options, round_index=round_index,
        )
    else:
        blocks = []
        for plugin in plugins:
            renderer = getattr(plugin, "visual_block", None)
            if not callable(renderer):
                continue
            block = renderer(
                phase=phase,
                question=question,
                answer_type=answer_type,
                options=options,
                video_id=video_id,
            )
            if isinstance(block, str) and block:
                blocks.append(block)
    evidence = metadata.get("reference_track_evidence")
    if phase in {"form", "review", "compare"} and isinstance(evidence, list) and evidence:
        for plugin in plugins:
            renderer = getattr(plugin, "reference_track_block", None)
            if not callable(renderer):
                continue
            block = renderer(evidence)
            if isinstance(block, str) and block:
                blocks.append(block)
    return blocks


def visual_question_for_sample(sample: Any, question: str, video_id: str) -> str:
    """Apply plugin-provided, view-local question normalization.

    This is generic plumbing: task plugins may normalize aliases for a visual
    call, while the shared pipeline remains unaware of task names.
    """
    metadata = getattr(sample, "task_metadata", None) or {}
    task = metadata.get("crossvid_task", "")
    localized = question
    for plugin in plugins_for_task(task):
        renderer = getattr(plugin, "visual_question", None)
        if callable(renderer):
            localized = renderer(question=localized, video_id=video_id)
    return localized


__all__ = [
    "PLUGIN_REGISTRY",
    "plugins_for_task",
    "blocks_for_sample",
    "visual_question_for_sample",
    "collect_blocks",
    "TaskPlugin",
]
