from __future__ import annotations

from pathlib import Path

import pytest

from unicvr.core.schemas import VideoRef
from unicvr.video import (
    FocusSampler,
    FrameCache,
    TemporalSegmenter,
    VideoDecodeError,
    VideoDecoder,
    VideoProbe,
    WarmStartSampler,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_real_video_probe_decode_and_cache(tmp_path: Path) -> None:
    video = VideoProbe().probe(FIXTURES / "video_a.mp4", video_id="v1")
    assert video.duration_seconds is not None and video.duration_seconds > 0
    assert video.fps == 8.0
    decoder = VideoDecoder(FrameCache(tmp_path / "frames"), resize_long_edge=96)
    first = decoder.extract(video, [0.0, 1.0, video.duration_seconds])
    mtimes = [item.local_path.stat().st_mtime_ns for item in first]
    second = decoder.extract(video, [0.0, 1.0, video.duration_seconds])
    assert [item.local_path for item in second] == [item.local_path for item in first]
    assert [item.local_path.stat().st_mtime_ns for item in second] == mtimes
    assert all(item.local_path.suffix == ".jpg" for item in first)


def test_decoder_emits_each_physical_frame_once(tmp_path: Path) -> None:
    video = VideoProbe().probe(FIXTURES / "video_a.mp4", video_id="v1")
    decoder = VideoDecoder(FrameCache(tmp_path / "frames"), resize_long_edge=96)

    result = decoder.extract(video, [0.0, 0.01, 0.02, 0.03, 0.2])

    assert [item.frame_index for item in result] == [0, 2]


def test_decode_falls_back_from_phantom_terminal_frame(tmp_path: Path) -> None:
    probed = VideoProbe().probe(FIXTURES / "video_a.mp4", video_id="v1")
    assert probed.frame_count is not None
    assert probed.fps is not None
    reported_count = probed.frame_count + 1
    video = probed.model_copy(
        update={
            "frame_count": reported_count,
            "duration_seconds": (reported_count - 1) / probed.fps,
        }
    )
    decoder = VideoDecoder(FrameCache(tmp_path / "frames"), resize_long_edge=96)

    result = decoder.extract(video, [video.duration_seconds or 0.0])

    assert result[0].frame_index == probed.frame_count - 1
    assert result[0].timestamp_seconds == (probed.frame_count - 1) / probed.fps
    assert result[0].local_path.is_file()


def test_warm_start_is_deterministic_and_coverage_first() -> None:
    video = VideoProbe().probe(FIXTURES / "video_a.mp4", video_id="v1")
    sampler = WarmStartSampler()
    first = sampler.sample(video, frame_budget=4, question="where is the object?")
    second = sampler.sample(video, frame_budget=4, question="different wording")
    assert first == second
    assert first == sorted(first)
    assert len(first) == 4
    assert first[0] == 0.0
    assert first[-1] == video.duration_seconds


def test_duration_adaptive_scan_is_chunked_with_overlap() -> None:
    video = VideoRef(
        video_id="long",
        path=FIXTURES / "video_a.mp4",
        duration_seconds=300.0,
        fps=8.0,
    )
    sampler = WarmStartSampler()

    count = sampler.frame_count(
        video,
        minimum=6,
        maximum=80,
        seconds_per_frame=4.0,
    )
    timestamps = [float(index) for index in range(count)]
    windows = sampler.chunk(timestamps, frames_per_call=12, overlap_frames=1)

    assert count == 76
    assert len(windows) == 7
    assert all(len(window) <= 12 for window in windows)
    assert all(left[-1] == right[0] for left, right in zip(windows, windows[1:], strict=False))
    assert sampler.chunk_cost(count, frames_per_call=12, overlap_frames=1) == (82, 7)


def test_focus_sampling_preserves_before_center_after_context() -> None:
    video = VideoProbe().probe(FIXTURES / "video_a.mp4", video_id="v1")
    points = FocusSampler().sample(
        video,
        start_seconds=1.0,
        end_seconds=2.0,
        frame_budget=5,
    )
    assert len(points) == 5
    assert points[0] < 1.0
    assert points[-1] > 2.0
    assert {1.0, 1.5, 2.0}.issubset(points)


def test_uniform_temporal_segment_fallback_is_deterministic() -> None:
    video = VideoProbe().probe(FIXTURES / "video_a.mp4", video_id="v1")
    segmenter = TemporalSegmenter(target_segment_seconds=1.0)
    assert segmenter.segment(video) == segmenter.segment(video)
    assert all(segment.change_score == 0 for segment in segmenter.segment(video))


def test_corrupted_video_has_actionable_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_text("not a video", encoding="ascii")
    with pytest.raises(VideoDecodeError, match="cannot open"):
        VideoProbe().probe(corrupt, video_id="broken")
