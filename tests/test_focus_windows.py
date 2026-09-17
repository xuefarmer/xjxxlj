"""Regression tests for timestamp-localized and no-window FOCUS calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unicvr.agents.reasoner import _parse, _parse_span
from unicvr.config import AppConfig
from unicvr.core.pipeline import Pipeline
from unicvr.core.schemas import VideoRef
from unicvr.core.state import Ambiguity, AmbiguityTarget
from unicvr.models.mock import MockVLMBackend
from unicvr.video import VideoProbe

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _parse_ambiguity(witness: str, targets: list[dict[str, object]]):
    raw = json.dumps(
        {
            "decision": "NEED_EVIDENCE",
            "ambiguities": [
                {
                    "contrast": "unclear event",
                    "witness": witness,
                    "scope": "multi_video_independent",
                    "targets": targets,
                }
            ],
        }
    )
    return _parse(raw, source="FORM").ambiguities[0]


def test_parse_span_accepts_point_and_preserves_valid_intervals() -> None:
    assert _parse_span("257.1s") == (257.1, 257.1)
    assert _parse_span("138.3s-163.5s") == (138.3, 163.5)
    assert _parse_span([138.3, 163.5]) == (138.3, 163.5)
    assert _parse_span("unknown") is None
    assert _parse_span("163.5s-138.3s") is None
    assert _parse_span([163.5, 138.3]) is None


def test_witness_recovers_single_named_target_timestamp() -> None:
    ambiguity = _parse_ambiguity(
        "Inspect the action at 186.9s in v3.",
        [{"video": "v3", "span": None}],
    )

    assert ambiguity.targets[0].span == (186.9, 186.9)


def test_witness_recognizes_long_video_name_and_possessive() -> None:
    ambiguity = _parse_ambiguity(
        "Compare video 3 (81.2 seconds) with v1's pose (12s).",
        [{"video": "v3", "span": None}, {"video": "v1", "span": None}],
    )

    assert [target.span for target in ambiguity.targets] == [(81.2, 81.2), (12.0, 12.0)]


def test_witness_does_not_assign_timestamp_named_for_a_different_video() -> None:
    ambiguity = _parse_ambiguity(
        "Inspect v9 at 42.5s.",
        [{"video": "v3", "span": None}],
    )

    assert ambiguity.targets[0].span is None


def test_witness_recovers_timestamp_for_long_video_id() -> None:
    ambiguity = _parse_ambiguity(
        "Inspect cityflow_camera_07 at 42.5s.",
        [{"video": "cityflow_camera_07", "span": None}],
    )

    assert ambiguity.targets[0].span == (42.5, 42.5)


def test_witness_assigns_each_timestamp_to_nearest_video() -> None:
    ambiguity = _parse_ambiguity(
        "Does the character in v3 (324.4s) match the seated man in v1 (143.3s)?",
        [{"video": "v3", "span": None}, {"video": "v1", "span": None}],
    )

    assert [target.span for target in ambiguity.targets] == [
        (324.4, 324.4),
        (143.3, 143.3),
    ]


def test_witness_without_video_name_does_not_guess_among_targets() -> None:
    ambiguity = _parse_ambiguity(
        "Inspect what happens at 42.5s.",
        [{"video": "v1", "span": None}, {"video": "v2", "span": None}],
    )

    assert [target.span for target in ambiguity.targets] == [None, None]


def test_witness_without_video_name_recovers_one_timestamp_for_one_target() -> None:
    ambiguity = _parse_ambiguity(
        "Inspect what happens at 42.5s.",
        [{"video": "v2", "span": None}],
    )

    assert ambiguity.targets[0].span == (42.5, 42.5)


def test_witness_without_timestamp_keeps_span_unknown() -> None:
    ambiguity = _parse_ambiguity(
        "Inspect the uncertain action in v3.",
        [{"video": "v3", "span": None}],
    )

    assert ambiguity.targets[0].span is None


def test_point_window_expands_and_out_of_range_point_uses_video_tail(
    app_config: AppConfig,
) -> None:
    pipeline = Pipeline(app_config)
    video = VideoRef(
        video_id="v1",
        path=FIXTURES / "video_a.mp4",
        duration_seconds=450.0,
        fps=8.0,
    )

    assert pipeline._expand_focus_span(video, (257.1, 257.1)) == pytest.approx((241.1, 273.1))
    tail = pipeline._expand_focus_span(video, (479.9, 479.9))
    assert tail == (418.0, 450.0)
    assert tail is not None and 0 <= tail[0] < tail[1] <= 450.0


def test_point_window_prompt_and_frames_use_same_expanded_span(
    app_config: AppConfig,
) -> None:
    pipeline = Pipeline(app_config)
    video = VideoProbe().probe(FIXTURES / "video_a.mp4", video_id="v1")
    point = (video.duration_seconds or 0.0) + 10.0
    ambiguity = Ambiguity(
        contrast="unclear event",
        witness="What happens there?",
        scope="single_video",
        targets=[AmbiguityTarget(video="v1", span=(point, point))],
    )

    pipeline._execute_ambiguities([ambiguity], [video], "Question", 1)

    backend = pipeline.observer.backend
    assert isinstance(backend, MockVLMBackend)
    expected = pipeline._expand_focus_span(video, (point, point))
    assert f"Time window: {expected}" in backend.prompts[-1]
    assert all(
        expected is not None and expected[0] <= frame.timestamp_seconds <= expected[1]
        for frame in backend.visual_calls[-1]
    )


def test_no_window_uses_one_interleaved_focus_call_per_round(
    app_config: AppConfig,
) -> None:
    pipeline = Pipeline(app_config)
    video = VideoProbe().probe(FIXTURES / "video_a.mp4", video_id="v1")
    first_timestamps = pipeline._no_window_focus_timestamps(video, 0)
    second_timestamps = pipeline._no_window_focus_timestamps(video, 1)
    assert len(first_timestamps) <= app_config.video.scan_frames_per_call
    assert len(second_timestamps) <= app_config.video.scan_frames_per_call
    assert set(first_timestamps).isdisjoint(second_timestamps)

    ambiguity = Ambiguity(
        contrast="unclear event",
        witness="What happens?",
        scope="single_video",
        targets=[AmbiguityTarget(video="v1", span=None)],
    )
    results = pipeline._execute_ambiguities(
        [ambiguity],
        [video],
        "Question",
        1,
    )

    backend = pipeline.observer.backend
    assert isinstance(backend, MockVLMBackend)
    assert len(results) == 1
    assert len(backend.visual_calls) == 1
    assert len(backend.visual_calls[0]) <= app_config.video.scan_frames_per_call
    assert "Time window: None" in backend.prompts[0]

    pipeline._execute_ambiguities([ambiguity], [video], "Question", 2)
    assert len(backend.visual_calls) == 2
    assert {frame.frame_index for frame in backend.visual_calls[0]}.isdisjoint(
        frame.frame_index for frame in backend.visual_calls[1]
    )


def test_no_window_short_video_never_returns_an_empty_batch(app_config: AppConfig) -> None:
    pipeline = Pipeline(app_config)
    video = VideoRef(
        video_id="one-frame",
        path=FIXTURES / "video_a.mp4",
        duration_seconds=0.0,
        fps=8.0,
        frame_count=1,
    )

    assert pipeline._no_window_focus_timestamps(video, 0) == [0.0]
    assert pipeline._no_window_focus_timestamps(video, 1) == [0.0]
