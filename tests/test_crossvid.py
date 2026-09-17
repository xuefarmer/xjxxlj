from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from unicvr.config import AppConfig
from unicvr.core.pipeline import Pipeline
from unicvr.data.crossvid import SUPPORTED_CROSSVID_TASKS, CrossVidDatasetAdapter
from unicvr.data.materialize import OpenCVMediaMaterializer
from unicvr.evaluation.metrics import (
    IntervalIoUMetric,
    ScoringPointCoverageMetric,
    SequenceExactMetric,
)
from unicvr.plugins.registry import blocks_for_sample
from unicvr.video import VideoProbe

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class StubMaterializer:
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = iter(paths)

    def clip(
        self,
        source: Path,
        intervals: Sequence[tuple[float, float]],
        *,
        cache_key: str,
        frame_dir_fps: float | None = None,
    ) -> Path:
        del source, intervals, cache_key, frame_dir_fps
        return next(self.paths)

    def uav_views(
        self,
        *,
        uav_root: Path,
        scene_id: int,
        objects: Sequence[dict[str, Any]],
        cache_key: str,
    ) -> list[Path]:
        del uav_root, scene_id, objects, cache_key
        return [next(self.paths), next(self.paths)]


def test_all_crossvid_tasks_normalize_and_run_through_same_mock_engine(
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    root, media = _crossvid_fixture(tmp_path)
    config = app_config.model_copy(
        update={
            "budget": app_config.budget.model_copy(
                update={
                    "max_frames": 56,
                    "max_estimated_visual_tokens": 24576,
                    "max_visual_calls": 10,
                    "max_llm_calls": 14,
                    "max_rounds": 8,
                    "verification_fraction": 0.2,
                }
            )
        }
    )

    for task in SUPPORTED_CROSSVID_TASKS:
        adapter = CrossVidDatasetAdapter(
            crossvid_root=root,
            task=task,
            materializer=StubMaterializer(media),
        )
        sample = next(adapter.iter_samples(limit=1))
        state = Pipeline(config).run(sample)

        assert sample.sample_id == f"{task}-0"
        assert sample.task_metadata["crossvid_task"] == task
        assert "answer" not in sample.inference_payload()
        assert state.final_answer is not None
        assert len(state.observer_reports) > 0


def test_crossvid_task_metrics_cover_interval_sequence_and_open_answer() -> None:
    root = FIXTURES
    base = {
        "sample_id": "metric",
        "video_paths": [root / "video_a.mp4", root / "video_b.mp4"],
        "question": "q",
    }
    from unicvr.data.schema import MultiVideoSample

    interval_sample = MultiVideoSample(**base, answer="[82, 95]")
    interval = IntervalIoUMetric().score("The interval is [80, 94].", interval_sample)
    assert interval.score is not None and interval.score > 0.7
    assert interval.correct is True

    sequence_sample = MultiVideoSample(**base, answer="3->5->4->2->1")
    sequence = SequenceExactMetric().score(
        "Order: 3 -> 5 -> 4 -> 2 -> 1",
        sequence_sample,
    )
    assert sequence.correct is True

    open_sample = MultiVideoSample(
        **base,
        answer="Video B rinses potatoes and squeezes them in a cloth.",
        task_metadata={
            "scoring_points": [
                "rinsing stage",
                "use of dishcloth",
                "squeezing over sink",
            ]
        },
    )
    coverage = ScoringPointCoverageMetric().score(
        "Video B adds a rinsing stage and uses a dishcloth over the sink.",
        open_sample,
    )
    assert coverage.score is not None and coverage.score >= 2 / 3
    assert coverage.details["proxy_metric"] is True


def test_uav_reference_tracks_are_structured_and_rendered_for_moc_and_msr(
    tmp_path: Path,
) -> None:
    root, media = _crossvid_fixture(tmp_path)
    for view in (1, 2):
        frame_dir = root / "uav" / "frames" / str(view) / f"22-{view}"
        frame_dir.mkdir(parents=True)
        for index in range(4):
            frame_dir.joinpath(f"{index:08d}.jpg").touch()
        bbox_dir = root / "uav" / "bbox" / str(view)
        bbox_dir.mkdir(parents=True)
        visible = (0, 1, 3) if view == 1 else (1, 2)
        bbox_dir.joinpath("22.json").write_text(
            json.dumps(
                [
                    {
                        "id": 7,
                        "label": "car",
                        "moving": view == 1,
                        "bbox": {
                            str(index): {
                                "outside": index not in visible,
                                "xtl": 10,
                                "ytl": 10,
                                "xbr": 40,
                                "ybr": 40,
                            }
                            for index in range(4)
                        },
                    },
                    {
                        "id": 99,
                        "label": "car",
                        "moving": True,
                        "bbox": {},
                    },
                ]
            ),
            encoding="utf-8",
        )

    moc = next(
        CrossVidDatasetAdapter(
            crossvid_root=root,
            task="MOC",
            materializer=StubMaterializer(media),
        ).iter_samples(limit=1)
    )
    evidence = moc.task_metadata["reference_track_evidence"]
    assert len(evidence) == 1  # hidden full-scene track 99 is not exposed
    assert evidence[0]["reference"] == "A1/B1"
    assert evidence[0]["raw_track_id"] == 7
    assert evidence[0]["views"]["A"]["visible_frame_ranges"] == [[0, 1], [3, 3]]
    assert evidence[0]["views"]["B"]["visible_frame_ranges"] == [[1, 2]]

    moc_blocks = blocks_for_sample(
        moc,
        "form",
        question=moc.question,
        answer_type="choice",
        options=moc.options,
    )
    assert any("Provided Reference-Track Evidence" in block for block in moc_blocks)
    assert any("A1/B1: source_track=7" in block for block in moc_blocks)
    assert any("synchronized cameras" in block for block in moc_blocks)
    assert any("MUST NOT be re-verified" in block for block in moc_blocks)
    assert all("source_track=99" not in block for block in moc_blocks)
    for phase in ("observer", "focus", "compare"):
        visual_blocks = blocks_for_sample(
            moc,
            phase,
            question=moc.question,
            answer_type="choice",
            options=moc.options,
        )
        assert any("two-wheeled road vehicles/riders" in block for block in visual_blocks)
        assert any("synchronized cameras" in block for block in visual_blocks)
        if phase == "compare":
            assert any("source_track=7" in block for block in visual_blocks)

    msr = next(
        CrossVidDatasetAdapter(
            crossvid_root=root,
            task="MSR",
            materializer=StubMaterializer(media),
        ).iter_samples(limit=1)
    )
    msr_blocks = blocks_for_sample(
        msr,
        "form",
        question=msr.question,
        answer_type="choice",
        options=msr.options,
    )
    # MSR shares the UAV sidecar with MOC: the trigger event ("X completely
    # leaves view Y") is anchored via the reference tracks' visible ranges.
    assert any("Provided Reference-Track Evidence" in block for block in msr_blocks)
    assert any("A1/B1: source_track=7" in block for block in msr_blocks)
    assert any("synchronized cameras" in block for block in msr_blocks)
    assert any("MUST NOT be re-verified" in block for block in msr_blocks)
    assert any("MSR Cross-View Contract" in block for block in msr_blocks)
    assert all("source_track=99" not in block for block in msr_blocks)
    for phase in ("observer", "focus"):
        obs_blocks = blocks_for_sample(
            msr,
            phase,
            question=msr.question,
            answer_type="choice",
            options=msr.options,
        )
        assert any("MSR Observation Requirements" in block for block in obs_blocks)
    compare_blocks = blocks_for_sample(
        msr,
        "compare",
        question=msr.question,
        answer_type="choice",
        options=msr.options,
    )
    assert any("Provided Reference-Track Evidence" in block for block in compare_blocks)


def test_opencv_materializer_builds_clip_and_annotated_uav_views(
    tmp_path: Path,
) -> None:
    materializer = OpenCVMediaMaterializer(
        tmp_path / "cache",
        proxy_frames=8,
        resize_long_edge=96,
    )
    clip = materializer.clip(
        FIXTURES / "video_a.mp4",
        [(0.0, 1.0), (2.0, 3.0)],
        cache_key="clip",
    )
    assert VideoProbe().probe(clip, video_id="clip").frame_count == 4

    uav_root = tmp_path / "uav"
    for view in (1, 2):
        frame_dir = uav_root / "frames" / str(view) / f"22-{view}"
        frame_dir.mkdir(parents=True)
        for index in range(4):
            image = np.full((64, 96, 3), 220, dtype=np.uint8)
            assert cv2.imwrite(str(frame_dir / f"{index:08d}.jpg"), image)
        bbox_dir = uav_root / "bbox" / str(view)
        bbox_dir.mkdir(parents=True)
        bbox_dir.joinpath("22.json").write_text(
            json.dumps(
                [
                    {
                        "id": 7,
                        "label": "car",
                        "bbox": {
                            str(index): {
                                "outside": False,
                                "xtl": 10,
                                "ytl": 10,
                                "xbr": 40,
                                "ybr": 40,
                            }
                            for index in range(4)
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )
    views = materializer.uav_views(
        uav_root=uav_root,
        scene_id=22,
        objects=[{"id": 7, "label": "car"}],
        cache_key="uav",
    )
    assert len(views) == 2
    assert all(VideoProbe().probe(path, video_id="view").frame_count == 4 for path in views)


def _crossvid_fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    root = tmp_path / "CrossVid"
    qa = root / "QA"
    videos = root / "videos"
    qa.mkdir(parents=True)
    videos.mkdir()
    media = []
    for index in range(8):
        path = videos / f"v{index}.mp4"
        shutil.copyfile(FIXTURES / f"video_{'a' if index % 2 == 0 else 'b'}.mp4", path)
        media.append(path)

    rows = {
        "CC": {
            "id": 0,
            "question": "Which differs?",
            "videos": ["v0.mp4", "v1.mp4"],
            "options": ["A. one", "B. two"],
            "answer": "B",
        },
        "CCQA": {
            "id": 0,
            "video A": "v0.mp4",
            "video B": "v1.mp4",
            "question": "How do they differ?",
            "answer": "They differ.",
            "scoring_points": ["difference"],
        },
        "FSA": {
            "id": 0,
            "video A": "v0.mp4",
            "video B": "v1.mp4",
            "ref_segment": [0, 1],
            "answer": [1, 2],
        },
        "MOC": {
            "id": 0,
            "question": "How many cars does {A1} pass?",
            "options": ["A. 1", "B. 2"],
            "answer": "B",
            "objects": [{"id": 7, "label": "car"}],
            "vid": 22,
        },
        "MSR": {
            "id": 0,
            "question": "Where is {A1} when {B1} leaves?",
            "options": ["A. left", "B. right"],
            "answer": "B",
            "objects": [{"id": 7, "label": "car"}],
            "vid": 22,
        },
        "NC": {
            "id": 0,
            "question": "Which plot?",
            "videos": ["v0.mp4", "v1.mp4"],
            "options": ["A. one", "B. two"],
            "answer": "B",
        },
        "PEA": {
            "id": 0,
            "question": "Which assembly?",
            "videos": ["v0.mp4", "v1.mp4", "v2.mp4"],
            "begin": [0, 0, 0],
            "end": [1, 1, 1],
            "options": ["A. one", "B. two"],
            "answer": "B",
        },
        "PI": {
            "id": 0,
            "video": "v0.mp4",
            "beginning": [0, 1],
            "ending": [2, 3],
            "options": ["A. one", "B. two"],
            "answer": "B",
        },
        "PSS": {
            "id": 0,
            "video": "v0.mp4",
            "segments": {
                "1": [[0, 1]],
                "2": [[1, 2]],
                "3": [[2, 3]],
            },
            "answer": "1->2->3",
        },
    }
    for task, row in rows.items():
        (qa / f"{task}.json").write_text(json.dumps([row]), encoding="utf-8")
    return root, media


def test_materializer_clips_from_frame_directory(
    tmp_path: Path,
) -> None:
    """Self-built Epic mirrors store clips as 30fps frame directories; clip()
    must materialize interval proxies from them (frame_%010d.jpg, 1-based)."""
    frame_dir = tmp_path / "P30_05"
    frame_dir.mkdir()
    fps = 30.0
    for index in range(1, 901):  # 30 seconds of frames
        image = np.full((64, 96, 3), 180, dtype=np.uint8)
        assert cv2.imwrite(str(frame_dir / f"frame_{index:010d}.jpg"), image)
    materializer = OpenCVMediaMaterializer(
        tmp_path / "cache",
        proxy_frames=8,
        resize_long_edge=96,
    )
    clip = materializer.clip(
        frame_dir,
        [(10.0, 14.0), (20.0, 24.0)],
        cache_key="frame-dir-clip",
        frame_dir_fps=fps,
    )
    assert clip.is_file() and clip.stat().st_size > 0
    assert VideoProbe().probe(clip, video_id="clip").frame_count == 8
    # cache hit: same call returns the same proxy without regenerating
    again = materializer.clip(
        frame_dir,
        [(10.0, 14.0), (20.0, 24.0)],
        cache_key="frame-dir-clip",
        frame_dir_fps=fps,
    )
    assert again == clip
