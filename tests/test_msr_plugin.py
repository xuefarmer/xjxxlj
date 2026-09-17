from __future__ import annotations

from types import SimpleNamespace

from unicvr.plugins.msr import MsrForceChoicePlugin
from unicvr.plugins.registry import blocks_for_sample, plugins_for_task

_EVIDENCE = [
    {
        "reference": "A5/B5",
        "raw_track_id": 65,
        "declared_label": "car",
        "views": {
            "A": {"alias": "A5", "source_frame_count": 450,
                  "visible_frame_ranges": [[0, 85]]},
            "B": {"alias": "B5", "source_frame_count": 450,
                  "visible_frame_ranges": [[0, 127]]},
        },
    },
]


def test_msr_form_contains_force_choice_and_cross_view_contract() -> None:
    plugin = MsrForceChoicePlugin()
    block = plugin.form_block(
        question="When {A4} completely leaves view A, where is {B1} located in view B?",
        answer_type="choice",
        options=["A. Bottom left of the frame", "B. Not shown in the frame"],
    )

    assert block is not None
    assert "Answer Requirement" in block
    assert "MUST pick one option letter" in block
    assert "MSR Cross-View Contract" in block
    assert "synchronized cameras" in block
    assert "same suffix are one physical reference" in block
    assert "Never substitute a different event" in block
    assert "Do not guess the trigger time" in block
    assert "abreast" in block
    assert "Completely leaves" in block
    assert "FIRST range" in block
    assert "Fully appears in" in block
    assert "START of that object's visible range" in block
    assert "relative-motion events" in block
    assert "RELATIVE TO" in block
    assert "final driving path" in block
    assert "FRAME REGION of the answer reference" in block
    assert "SET THE TIME" in block
    assert "plus/minus one second" in block
    assert "authoritative for presence" in block
    assert "on-screen position IS the answer" in block
    assert "position claim" in block
    assert "hypotheses" in block


def test_msr_form_ignores_non_choice_answers() -> None:
    block = MsrForceChoicePlugin().form_block(
        question="q",
        answer_type="free_text",
        options=[],
    )

    assert block is None


def test_msr_review_reuses_the_form_contract() -> None:
    plugin = MsrForceChoicePlugin()
    form = plugin.form_block(
        question="q1", answer_type="choice", options=["A. x"]
    )
    review = plugin.review_block(
        question="q2",
        answer_type="choice",
        options=["A. x"],
        round_index=2,
    )

    assert review == form


def test_msr_reference_track_block_renders_sidecar() -> None:
    block = MsrForceChoicePlugin().reference_track_block(_EVIDENCE)

    assert block is not None
    assert "Provided Reference-Track Evidence" in block
    assert "synchronized cameras" in block
    assert "MUST NOT be re-verified or overruled visually" in block
    assert "A5/B5: source_track=65; declared=car" in block
    assert "A5: visible_source_frames=0-85/450 (proxy 0.0-2.3s)" in block
    assert "B5: visible_source_frames=0-127/450 (proxy 0.0-3.4s)" in block
    assert "its end" in block
    assert "authoritative for presence" in block
    assert "read the drawn box at the trigger moment" in block


def test_msr_reference_track_block_handles_empty_evidence() -> None:
    block = MsrForceChoicePlugin().reference_track_block([])

    assert block is not None
    assert "Provided Reference-Track Evidence" in block


def test_registry_injects_msr_contract_and_sidecar_into_form() -> None:
    sample = SimpleNamespace(
        task_metadata={
            "crossvid_task": "MSR",
            "reference_track_evidence": _EVIDENCE,
        }
    )
    blocks = blocks_for_sample(
        sample,
        "form",
        question="When {A5} completely leaves view A, where is {B1} located in view B?",
        answer_type="choice",
        options=["A. Bottom left of the frame", "B. Not shown in the frame"],
    )

    assert any("MSR Cross-View Contract" in block for block in blocks)
    assert any("A5: visible_source_frames=0-85/450" in block for block in blocks)


def test_msr_visual_block_requires_frame_regions_for_observer() -> None:
    plugin = MsrForceChoicePlugin()
    block = plugin.visual_block(
        phase="observer",
        question="When {A4} completely leaves view A, where is {B1} located in view B?",
        answer_type="choice",
        options=[],
    )

    assert block is not None
    assert "MSR Observation Requirements" in block
    assert "position within the VIDEO FRAME" in block
    assert "left /" in block
    assert "right / center of the frame" in block
    assert "A road LANE is not a frame region" in block
    assert "completely leaves the frame" in block
    assert plugin.visual_block(
        phase="compare", question="q", answer_type="choice", options=[]
    ) is None


def test_msr_plugin_is_registered_for_msr_only() -> None:
    assert any(
        isinstance(plugin, MsrForceChoicePlugin)
        for plugin in plugins_for_task("MSR")
    )
    for task in ("CC", "NC", "PEA", "PI", "MOC", "FSA", "PSS"):
        assert not any(
            isinstance(plugin, MsrForceChoicePlugin)
            for plugin in plugins_for_task(task)
        )
