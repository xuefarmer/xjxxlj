from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from unicvr.core.schemas import VideoRef
from unicvr.plugins.moc import MocCountingPlugin
from unicvr.plugins.registry import plugins_for_task, visual_question_for_sample
from unicvr.video import WarmStartSampler

ROOT = Path(__file__).resolve().parents[1]


def test_moc_uses_shared_rules_but_question_specific_scope() -> None:
    plugin = MocCountingPlugin()
    first = plugin.form_block(
        question="How many vehicles overtake the reference object?",
        answer_type="choice",
        options=["A. 1", "B. 2"],
    )
    second = plugin.form_block(
        question="At the synchronized moment, how many objects are in view B?",
        answer_type="choice",
        options=["A. 3", "B. 4"],
    )

    assert first is not None
    assert second is not None
    assert "MOC Rules" in first
    assert "MOC Rules" in second
    assert "two-wheelers" in first
    assert "synchronized views" in first
    assert "Count the cross-view union" in first
    assert "Count only physical instances visible in view B" in second
    assert "Never change a count" in first


def test_moc_review_reuses_the_general_form_contract() -> None:
    plugin = MocCountingPlugin()
    form = plugin.form_block(question="q1", answer_type="choice", options=[])
    review = plugin.review_block(
        question="q2",
        answer_type="choice",
        options=["A. 0"],
        round_index=2,
    )

    assert review is not None
    assert "MOC Rules" in review
    # Review keeps the general MOC contract and appends the review protocol.
    assert "MOC Rules" in form
    assert "Review Protocol" in review
    assert "Re-derive the integer" in review
    assert "Preserve the previous candidate ledger" in review
    assert "wrong view, category, relation, or timepoint" in review


def test_conditional_question_separates_trigger_from_count_clause() -> None:
    plugin = MocCountingPlugin()
    question = (
        "When {A4} completely passes {A5} in view A, "
        "how many red cars are visible in view B?"
    )
    block = plugin.form_block(question=question, answer_type="choice", options=[])

    assert block is not None
    assert "Trigger clause (time only)" in block
    assert "Count clause (defines the answer set)" in block
    assert "Count only physical instances visible in view B" in block
    assert "Do not copy its actor, action, reference, color, category, or view" in block
    assert "stable candidate ledger" in block

    visual = plugin.visual_block(
        phase="observer",
        question=question,
        answer_type="choice",
        options=[],
        video_id="v2",
    )
    assert "Target category: red cars" in visual
    assert "Relation/condition to track: passing by" not in visual
    assert "each counted target passes" not in visual
    assert "Trigger clause (time only): `When {A4} completely passes {A5} in view A`" in visual
    assert "state its approximate time and evidence for the count-clause predicate" in visual


def test_focus_is_local_delta_not_a_global_recount() -> None:
    block = MocCountingPlugin().visual_block(
        phase="focus",
        question="After {A1} leaves view A, how many cycles are in view B?",
        answer_type="choice",
        options=[],
        video_id="v2",
    )

    assert "local delta evidence" in block
    assert "not a replacement for the full candidate ledger" in block
    assert "Witness: undetermined" in block


def test_moc_visual_semantics_cover_observer_focus_and_compare() -> None:
    plugin = MocCountingPlugin()
    for phase in ("observer", "focus", "compare"):
        block = plugin.visual_block(
            phase=phase,
            question="How many cycles overtake A1?",
            answer_type="choice",
            options=["A. 1"],
        )
        assert "two-wheeled road vehicles/riders" in block
        assert "UAV camera itself moves" in block
    compare = plugin.visual_block(
        phase="compare",
        question="q",
        answer_type="choice",
        options=[],
    )
    assert "views are synchronized" in compare
    assert "A_i/B_i with the same suffix" in compare


def test_moc_visual_question_normalizes_reference_alias_in_code() -> None:
    plugin = MocCountingPlugin()

    assert plugin.visual_question(
        question="How many cycles does {B1} overtake near A3?",
        video_id="v1",
    ) == "How many cycles does {A1} overtake near A3?"
    assert plugin.visual_question(
        question="How many cycles overtake {A2} near B4?",
        video_id="v2",
    ) == "How many cycles overtake {B2} near B4?"
    assert plugin.visual_question(
        question="How many cycles overtake {A2}?",
        video_id="other",
    ) == "How many cycles overtake {A2}?"


def test_registry_applies_moc_visual_alias_hook_only_to_moc() -> None:
    moc = SimpleNamespace(task_metadata={"crossvid_task": "MOC"})
    pss = SimpleNamespace(task_metadata={"crossvid_task": "PSS"})

    assert visual_question_for_sample(
        moc,
        "How many cycles does {B5} overtake?",
        "v1",
    ) == "How many cycles does {A5} overtake?"
    assert visual_question_for_sample(
        pss,
        "Keep {B5} unchanged",
        "v1",
    ) == "Keep {B5} unchanged"


def test_moc_visual_block_receives_already_localized_reference() -> None:
    plugin = MocCountingPlugin()
    localized = plugin.visual_question(
        question="How many cycles does {B1} overtake?",
        video_id="v1",
    )
    block = plugin.visual_block(
        phase="observer",
        question=localized,
        answer_type="choice",
        options=[],
        video_id="v1",
    )

    assert "Reference in this view: A1" in block
    assert "A1 passes/overtakes the targets" in block
    assert "B1" not in block


def test_moc_observer_separates_visible_and_qualifying_counts() -> None:
    block = MocCountingPlugin().visual_block(
        phase="observer",
        question="How many cycles pass by {A2}?",
        answer_type="choice",
        options=[],
    )

    assert "Visible target candidates in this view: N" in block
    assert "Qualifying instances/events in this view: M" in block
    assert "best-supported temporal estimate" in block
    assert "conservative count" not in block


def test_open_world_difference_between_visible_and_qualifying_is_normal() -> None:
    block = MocCountingPlugin().form_block(
        question="How many red sedans pass {A2} in view A?",
        answer_type="choice",
        options=[],
    )

    assert block is not None
    assert "Visible != Qualifying` is normal" in block
    assert "not by itself a reason" in block


def test_joint_question_preserves_both_anchors_and_logical_and() -> None:
    plugin = MocCountingPlugin()
    question = (
        "Across both displayed views, how many vehicles pass {A2} in view A "
        "and are ahead of {B3} in the same direction in view B?"
    )

    assert plugin.visual_question(question=question, video_id="v1") == question
    assert plugin.visual_question(question=question, video_id="v2") == question
    block = plugin.visual_block(
        phase="observer", question=question, answer_type="choice", options=[], video_id="v1"
    )
    assert "each counted target passes A2" in block
    assert "each counted target is ahead of B3" in block
    assert "logical AND" in block
    assert "qualifies in just one view is excluded" in block


def test_timepoint_questions_restrict_count_to_boundary() -> None:
    block = MocCountingPlugin().visual_block(
        phase="observer",
        question="At the start of the displayed clip, how many vehicles are ahead of {A1}?",
        answer_type="choice",
        options=[],
        video_id="v1",
    )

    assert "qualify objects only at the initial boundary" in block
    assert "never to expand the timepoint set" in block


def test_joint_moving_semantics_are_parsed() -> None:
    block = MocCountingPlugin().visual_block(
        phase="observer",
        question=(
            "Across both displayed views, how many vehicles show clear displacement "
            "in both view A and view B?"
        ),
        answer_type="choice",
        options=[],
        video_id="v1",
    )

    assert "showing clear displacement during the requested clip" in block


def test_moc_plugin_is_not_registered_for_pss_or_fsa() -> None:
    assert any(isinstance(plugin, MocCountingPlugin) for plugin in plugins_for_task("MOC"))
    assert not any(isinstance(plugin, MocCountingPlugin) for plugin in plugins_for_task("PSS"))
    assert not any(isinstance(plugin, MocCountingPlugin) for plugin in plugins_for_task("FSA"))


def test_moc_config_observes_each_current_proxy_in_one_chunk() -> None:
    raw = yaml.safe_load((ROOT / "configs" / "moc_nothinking.yaml").read_text())
    video_config = raw["video"]
    frame_count = WarmStartSampler.frame_count(
        VideoRef(
            video_id="v1",
            path=ROOT / "tests" / "fixtures" / "video_a.mp4",
            duration_seconds=12.0,
        ),
        minimum=raw["budget"]["min_warm_frames_per_video"],
        maximum=raw["budget"]["scan_frames_per_video"],
        seconds_per_frame=video_config["scan_seconds_per_frame"],
    )
    _, call_count = WarmStartSampler.chunk_cost(
        frame_count,
        frames_per_call=video_config["scan_frames_per_call"],
        overlap_frames=video_config["scan_overlap_frames"],
    )

    assert video_config["scan_seconds_per_frame"] == 0.5
    assert video_config["scan_frames_per_call"] == 32
    assert video_config["scan_overlap_frames"] == 0
    assert call_count == 1
