"""ChoiceAnswerPlugin — shared by all multiple-choice tasks.

Root cause fix for the 6 choice tasks (CC/NC/PEA/PI/MOC/MSR): the option
list was never shown to the Reasoner, so it answered with video ids,
free text, or "unknown" instead of the option letter. This plugin
injects the option list into FORM and REVIEW and pins the output format
to the option letter (or exact option text).
"""

from __future__ import annotations

from unicvr.plugins.base import TaskPlugin


class ChoiceAnswerPlugin(TaskPlugin):
    name = "choice_answer"

    def _options_block(self, options: list[str]) -> str | None:
        if not options:
            return None
        lines = "\n".join(options)
        return (
            "## Options (choose exactly one)\n"
            f"{lines}\n\n"
            "best_answer.value MUST be the option letter (e.g. \"C\") — never a "
            "video id, never free text, never \"unknown\". If you cannot decide, "
            "pick the most likely letter and explain the uncertainty in the "
            "rationale instead."
        )

    def form_block(
        self,
        *,
        question: str,
        answer_type: str,
        options: list[str],
    ) -> str | None:
        if answer_type != "choice":
            return None
        return self._options_block(options)

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
        return self._options_block(options)
