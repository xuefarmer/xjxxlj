"""Phase-1-only event-level Observer replacement.

The production :class:`unicvr.core.pipeline.Pipeline` is left intact.  This
subclass changes only warm-start observation.  The existing Focus/Comparer
code continues to use the detailed production visual prompts, so the protocol
is deliberately coarse-to-fine rather than globally less visual.
"""

from __future__ import annotations

from pathlib import Path

from unicvr.core.pipeline import Pipeline
from unicvr.core.schemas import GenerationConfig, VisualInput


class EventLevelObserverAgent:
    """Question-conditioned, compact evidence reporter for warm-start chunks."""

    def __init__(self, backend: object, generation: GenerationConfig, prompt_path: Path) -> None:
        self.backend = backend
        self.generation = generation
        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    def observe(
        self,
        *,
        video_id: str,
        visual_inputs: list[VisualInput],
        task_context: str,
        prompt_blocks: list[str] | None = None,
    ) -> str:
        frame_timestamps = [
            f"  frame {item.frame_index}: t={item.timestamp_seconds:.1f}s"
            for item in visual_inputs
        ]
        extra = "\n\n" + "\n\n".join(prompt_blocks) if prompt_blocks else ""
        user_prompt = (
            f"Video: {video_id}\n"
            f"Chunk time range: {visual_inputs[0].timestamp_seconds:.1f}s"
            f" – {visual_inputs[-1].timestamp_seconds:.1f}s\n"
            f"Question context: {task_context}\n"
            "Frame index:\n" + "\n".join(frame_timestamps) + extra
            + "\n\nReturn the temporally indexed event-level evidence packet now."
        )
        raw = self.backend.generate_text(
            role="ObserverAgent.EVENT",
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            visual_inputs=visual_inputs,
            generation_config=self.generation,
        )
        return raw.strip()


class EventLevelObserverPipeline(Pipeline):
    """Pipeline whose initial Observer is compact and whose Focus remains detailed."""

    def __init__(self, *args: object, event_prompt_path: Path, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.observer = EventLevelObserverAgent(self._vlm, self.config.generation, event_prompt_path)
