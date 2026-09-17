"""Deterministic mock backends for testing the data-axis pipeline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from unicvr.core.schemas import (
    APIUsage,
    BackendCallRecord,
    GenerationConfig,
    VisualInput,
)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Mock LLM Backend
# ---------------------------------------------------------------------------


class MockLLMBackend:
    def __init__(self, model: str = "deterministic-mock-llm") -> None:
        self.model = model
        self.calls: list[BackendCallRecord] = []
        self.prompts: list[str] = []

    def generate_text(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        generation_config: GenerationConfig,
    ) -> str:
        del system_prompt, generation_config
        self.prompts.append(user_prompt)

        if "ObserverAgent" in role:
            result = '{"observation": "Mock observer report."}'
        elif "ReasonerAgent.FORM" in role:
            result = _MOCK_FORM_OUTPUT
        elif "ReasonerAgent.REVIEW" in role:
            result = _MOCK_REVIEW_OUTPUT
        elif "ComparerAgent" in role:
            result = _MOCK_COMPARER_OUTPUT
        else:
            result = '{"prediction": "mock"}'

        self.calls.append(
            BackendCallRecord(
                role=role,
                backend="mock",
                model=self.model,
                raw_response=result[:500],
            )
        )
        return result

    def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        generation_config: GenerationConfig,
    ) -> T:
        """Structured generation – parses mock text output into schema."""
        raw = self.generate_text(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            generation_config=generation_config,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"prediction": raw.strip(), "confidence": 0.5, "rationale": ""}
        return output_schema.model_validate(data)


# ---------------------------------------------------------------------------
# Mock VLM Backend
# ---------------------------------------------------------------------------


class MockVLMBackend:
    def __init__(self, model: str = "deterministic-mock-vlm") -> None:
        self.model = model
        self.calls: list[BackendCallRecord] = []
        self.prompts: list[str] = []
        self.visual_calls: list[list[VisualInput]] = []

    def generate_text(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        visual_inputs: Sequence[VisualInput],
        generation_config: GenerationConfig,
    ) -> str:
        del system_prompt, generation_config
        if not visual_inputs:
            raise ValueError("mock VLM requires visual inputs")
        self.prompts.append(user_prompt)
        inputs = list(visual_inputs)
        self.visual_calls.append(inputs)

        if "ObserverAgent" in role:
            video_ids = sorted({item.video_id for item in inputs})
            vid = video_ids[0] if len(video_ids) == 1 else ", ".join(video_ids)
            t0 = inputs[0].timestamp_seconds
            t1 = inputs[-1].timestamp_seconds
            result = _build_mock_observer_output(vid, t0, t1)
        elif "ComparerAgent" in role:
            result = _MOCK_COMPARER_OUTPUT
        else:
            result = '{"comparison_result": "cannot_determine", "confidence": 0.5, "evidence": "Mock VLM response."}'

        self.calls.append(
            BackendCallRecord(
                role=role,
                backend="mock",
                model=self.model,
                visual_input_count=len(inputs),
                raw_response=result[:500],
            )
        )
        return result

    def generate_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        visual_inputs: Sequence[VisualInput],
        output_schema: type[T],
        generation_config: GenerationConfig,
    ) -> T:
        raw = self.generate_text(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            visual_inputs=visual_inputs,
            generation_config=generation_config,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"prediction": raw.strip()}
        return output_schema.model_validate(data)


# ---------------------------------------------------------------------------
# Mock output constants (English)
# ---------------------------------------------------------------------------

def _build_mock_observer_output(vid: str, t0: float, t1: float) -> str:
    """Build a deterministic mock observer report."""
    tmid = (t0 + t1) / 2.0
    return f"""## Observation Report for {vid}

### Timeline of Events
- **{t0:.1f}s - {t1:.1f}s**: A person is visible throughout the clip, interacting with objects in a scene.

### Objects and Entities
- **Person**: Visible throughout, wearing casual clothing. Positioned in the center of the frame.
- **Object A**: A rectangular object, dark in color, present from the start. Moves from left to center by mid-clip.
- **Object B**: A smaller, lighter-colored item that appears around the {tmid:.1f}s mark.

### Actions
- The person reaches toward Object A at approximately {t0 + 0.5:.1f}s.
- Object A is manipulated (lifted/turned/moved) between {t0 + 1.0:.1f}s and {t1 - 1.0:.1f}s.
- Object B is introduced and placed near Object A.

### States and Transformations
- Object A: Initially stationary (directly observed). Changes position to center-right (directly observed).
- Object B: Absent at start (directly observed). Present from ~{tmid:.1f}s (directly observed). Placed adjacent to Object A (directly observed).
- The spatial relationship between Object A and Object B changes from separated to adjacent.

### Visual Details
- Lighting: Even, indoor/neutral lighting throughout.
- Background: Plain, non-distracting background.
- Colors: Object A is dark (black or dark blue). Object B is light (white or beige).

### Uncertainty and Gaps
- Cannot determine the exact nature of Object A and Object B from these frames alone.
- Fine details of the manipulation are partly occluded by the person's hands.
- Whether Object B is a tool, ingredient, or unrelated item cannot be definitively determined.
"""

_MOCK_FORM_OUTPUT = """{
  "best_answer": {"type": "sequence", "value": ["v1", "v2"]},
  "alternatives": [],
  "state_factors": [{"factor":"object","observations":[{"video":"v1","span":null,"state":"initial","basis":"visual"},{"video":"v2","span":null,"state":"final","basis":"visual"}]}],
  "ambiguities": [{"contrast":"v1_vs_v2","witness":"Does v2 show a later state of the same object?","scope":"joint_compare","targets":[{"video":"v1","span":null},{"video":"v2","span":null}],"pair":["v1","v2"]}],
  "decision": "NEED_EVIDENCE"
}"""

_MOCK_REVIEW_OUTPUT = """{
  "best_answer": {"type": "choice", "value": "B"},
  "alternatives": [],
  "state_factors": [],
  "ambiguities": [],
  "decision": "DONE"
}"""

_MOCK_COMPARER_OUTPUT = """{
  "clip_a_observation": "Object A is stationary on the left side. Object B is not present in the early frames.",
  "clip_b_observation": "Object A is in the center-right position. Object B is present and placed adjacent to Object A.",
  "comparison_result": "a_before_b",
  "confidence": 0.82,
  "evidence": "Clip A shows Object A in its initial left position without Object B. Clip B shows Object A in a later center-right position with Object B present. The movement of Object A from left to center-right and the introduction of Object B indicates clip A precedes clip B.",
  "uncertainty": "The exact time gap between the two clips cannot be determined. The manipulation action may have intermediate steps not visible in either clip."
}"""
