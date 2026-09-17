"""Observer Agent — single-video natural-language observer.

Each Observer watches ONE video and writes a detailed, factual report.
Multiple Observers run in parallel during Phase 1 of the pipeline.
"""

from __future__ import annotations

from unicvr.core.schemas import GenerationConfig, VisualInput
from unicvr.models.base import VLMBackend
from unicvr.prompts import prompt


class ObserverAgent:
    """Watches one video and produces a natural-language observation report."""

    def __init__(self, backend: VLMBackend, generation: GenerationConfig) -> None:
        self.backend = backend
        self.generation = generation

    def observe(
        self,
        *,
        video_id: str,
        visual_inputs: list[VisualInput],
        task_context: str,
        prompt_blocks: list[str] | None = None,
    ) -> str:
        """Observe one video and return a detailed NL report.

        Args:
            video_id: Which video this is (e.g. "v1").
            visual_inputs: Frames extracted from the video.
            task_context: Brief description of what kind of information
                will be needed later (e.g. "observe objects and actions").
        """
        frame_timestamps = [
            f"  frame {item.frame_index}: t={item.timestamp_seconds:.1f}s"
            for item in visual_inputs
        ]
        extra = ""
        if prompt_blocks:
            extra = "\n\n" + "\n\n".join(prompt_blocks)
        user_prompt = (
            f"Video: {video_id}\n"
            f"Frame count: {len(visual_inputs)}\n"
            f"Time range: {visual_inputs[0].timestamp_seconds:.1f}s"
            f" – {visual_inputs[-1].timestamp_seconds:.1f}s\n"
            f"Context: {task_context}\n"
            f"\nFrame index:\n"
            + "\n".join(frame_timestamps)
            + extra
            + "\n\nWrite a detailed observation report for this video."
        )

        raw = self.backend.generate_text(
            role="ObserverAgent",
            system_prompt=prompt("observer"),
            user_prompt=user_prompt,
            visual_inputs=visual_inputs,
            generation_config=self.generation,
        )
        return raw.strip()
