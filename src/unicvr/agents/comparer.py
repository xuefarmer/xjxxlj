"""Comparer Agent — targeted pairwise visual comparison.

The Comparer directly watches TWO clips side-by-side and answers a
specific comparison question.  Multiple Comparers run in parallel
during Phase 3 of the pipeline.
"""

from __future__ import annotations

from unicvr.core.schemas import GenerationConfig, VisualInput
from unicvr.models.base import VLMBackend
from unicvr.prompts import prompt


class ComparerAgent:
    """Compares two specific video clips to answer one focused question."""

    def __init__(self, backend: VLMBackend, generation: GenerationConfig) -> None:
        self.backend = backend
        self.generation = generation

    def compare(
        self,
        *,
        clip_a: list[VisualInput],
        clip_b: list[VisualInput],
        video_a: str,
        video_b: str,
        question: str,
        prompt_blocks: list[str] | None = None,
    ) -> str:
        """Compare two clips and return a natural-language comparison result.

        Args:
            clip_a: Frames from the first video.
            clip_b: Frames from the second video.
            video_a: Video ID for clip A.
            video_b: Video ID for clip B.
            question: The specific comparison question to answer.

        Returns:
            Natural-language comparison result.
        """
        # Label all frames with their video ID
        for item in clip_a:
            object.__setattr__(
                item, "label",
                f"[{video_a}] t={item.timestamp_seconds:.1f}s frame={item.frame_index}",
            )
        for item in clip_b:
            object.__setattr__(
                item, "label",
                f"[{video_b}] t={item.timestamp_seconds:.1f}s frame={item.frame_index}",
            )

        all_inputs = list(clip_a) + list(clip_b)

        extra = ""
        if prompt_blocks:
            extra = "\n\n" + "\n\n".join(prompt_blocks)
        user_prompt = (
            f"## Clip A: {video_a}\n"
            f"Frames: {len(clip_a)} | "
            f"Time: {clip_a[0].timestamp_seconds:.1f}s"
            f" – {clip_a[-1].timestamp_seconds:.1f}s\n\n"
            f"## Clip B: {video_b}\n"
            f"Frames: {len(clip_b)} | "
            f"Time: {clip_b[0].timestamp_seconds:.1f}s"
            f" – {clip_b[-1].timestamp_seconds:.1f}s\n\n"
            f"## Comparison Question\n{question}\n\n"
            f"{extra}\n\n"
            "Compare the two clips and answer the question as JSON."
        )

        raw = self.backend.generate_text(
            role="ComparerAgent",
            system_prompt=prompt("comparer"),
            user_prompt=user_prompt,
            visual_inputs=all_inputs,
            generation_config=self.generation,
        )
        return raw.strip()
