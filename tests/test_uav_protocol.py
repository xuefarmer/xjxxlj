from __future__ import annotations

from unicvr.plugins.moc import MocCountingPlugin
from unicvr.plugins.msr import MsrForceChoicePlugin
from unicvr.plugins.uav import UAV_SYNCHRONIZATION_CONTRACT

_EVIDENCE = [
    {
        "reference": "A5/B5",
        "raw_track_id": 65,
        "declared_label": "car",
        "views": {
            "A": {
                "alias": "A5",
                "source_frame_count": 450,
                "visible_frame_ranges": [[0, 85]],
            },
            "B": {
                "alias": "B5",
                "source_frame_count": 450,
                "visible_frame_ranges": [[90, 127]],
            },
        },
    }
]


def test_moc_and_msr_use_the_exact_same_synchronization_contract() -> None:
    moc = MocCountingPlugin().form_block(
        question="When {A5} leaves view A, how many cars are in view B?",
        answer_type="choice",
        options=[],
    )
    msr = MsrForceChoicePlugin().form_block(
        question="When {A5} leaves view A, where is {B1} in view B?",
        answer_type="choice",
        options=[],
    )

    assert moc is not None
    assert msr is not None
    assert UAV_SYNCHRONIZATION_CONTRACT in moc
    assert UAV_SYNCHRONIZATION_CONTRACT in msr


def test_moc_and_msr_use_the_exact_same_reference_sidecar() -> None:
    moc = MocCountingPlugin().reference_track_block(_EVIDENCE)
    msr = MsrForceChoicePlugin().reference_track_block(_EVIDENCE)

    assert moc == msr
    assert moc is not None
    assert "A5: visible_source_frames=0-85/450 (proxy 0.0-2.3s)" in moc
    assert "B5: visible_source_frames=90-127/450 (proxy 2.4-3.4s)" in moc
    assert "does NOT enumerate unlabelled target objects" in moc


def test_moc_keeps_counting_contract_after_shared_uav_contract() -> None:
    block = MocCountingPlugin().review_block(
        question="When {A5} leaves view A, how many red cars are in view B?",
        answer_type="choice",
        options=[],
        round_index=1,
    )

    assert block is not None
    assert "Shared UAV Synchronization Contract" in block
    assert "Preserve the previous candidate ledger" in block
    assert "Never reinterpret a trigger action as a predicate" in block
