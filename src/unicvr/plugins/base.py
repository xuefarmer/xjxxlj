"""Task plugin protocol — per-task, decoupled prompt adaptations.

A plugin contributes prompt blocks that are injected into the Reasoner
FORM/REVIEW user prompt for one specific CrossVid task (or a task family).
Plugins never modify the protocol schema or the pipeline control flow;
they only add task-specific guidance text. Each task composes its own
plugin list in the registry, so adaptations stay decoupled per task.
"""

from __future__ import annotations

from typing import Protocol


class TaskPlugin(Protocol):
    """Protocol: a plugin may inject prompt blocks at FORM and/or REVIEW."""

    name: str

    def form_block(
        self,
        *,
        question: str,
        answer_type: str,
        options: list[str],
    ) -> str | None:
        """Extra section appended to the FORM user prompt, or None."""
        ...

    def review_block(
        self,
        *,
        question: str,
        answer_type: str,
        options: list[str],
        round_index: int,
    ) -> str | None:
        """Extra section appended to the REVIEW user prompt, or None."""
        ...


def collect_blocks(
    plugins: list[TaskPlugin],
    *,
    phase: str,
    question: str,
    answer_type: str,
    options: list[str],
    round_index: int = 0,
) -> list[str]:
    """Collect non-None prompt blocks from the given plugins in order."""
    blocks: list[str] = []
    for plugin in plugins:
        if phase == "form":
            block = plugin.form_block(
                question=question, answer_type=answer_type, options=options
            )
        else:
            block = plugin.review_block(
                question=question, answer_type=answer_type,
                options=options, round_index=round_index,
            )
        if block:
            blocks.append(block)
    return blocks
