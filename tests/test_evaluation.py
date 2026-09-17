from __future__ import annotations

import json
from pathlib import Path

from unicvr.config import AppConfig
from unicvr.data.adapter import JSONDatasetAdapter, load_field_mapping
from unicvr.evaluation import evaluate_samples, normalize_answer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_dataset_adapter_normalizes_paths_and_list_answer(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (video_root / name).symlink_to(FIXTURES / f"video_{name[0]}.mp4")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": 17,
                    "videos": ["a.mp4", "b.mp4"],
                    "question": "Which option is correct?",
                    "options": ["A. first", "B. second"],
                    "answer": ["B"],
                }
            ]
        ),
        encoding="utf-8",
    )
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        "\n".join(
            [
                "sample_id: id",
                "video_paths: videos",
                "question: question",
                "options: options",
                "answer: answer",
            ]
        ),
        encoding="utf-8",
    )

    adapter = JSONDatasetAdapter(
        video_root=video_root,
        field_mapping=load_field_mapping(mapping_path),
    )
    sample = next(adapter.iter_samples(dataset_path))

    assert sample.sample_id == "17"
    assert sample.video_paths == [
        (video_root / "a.mp4").resolve(),
        (video_root / "b.mp4").resolve(),
    ]
    assert sample.answer == "B"
    assert sample.task_metadata == {
        "source_dataset": "dataset.json",
        "source_index": 0,
    }


def test_answer_normalization_handles_letter_and_option_text() -> None:
    options = ["A. open the drawer", "B. close the drawer"]
    assert normalize_answer("Option B", options) == "B"
    assert normalize_answer("close the drawer", options) == "B"
    assert normalize_answer(["B"], options) == "B"


def test_mock_dataset_evaluation_writes_result_trace_and_summary(
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "mock-eval",
                    "videos": [
                        str(FIXTURES / "video_a.mp4"),
                        str(FIXTURES / "video_b.mp4"),
                    ],
                    "question": "Which option is supported by both videos?",
                    "options": ["A. first", "B. second"],
                    "answer": "B",
                }
            ]
        ),
        encoding="utf-8",
    )
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        "\n".join(
            [
                "sample_id: id",
                "video_paths: videos",
                "question: question",
                "options: options",
                "answer: answer",
            ]
        ),
        encoding="utf-8",
    )
    adapter = JSONDatasetAdapter(
        video_root=tmp_path,
        field_mapping=load_field_mapping(mapping_path),
    )
    output_dir = tmp_path / "evaluation"

    summary = evaluate_samples(
        app_config,
        adapter.iter_samples(dataset_path),
        output_dir=output_dir,
    )

    assert summary.attempted == summary.completed == summary.correct == 1
    assert summary.failed == 0
    assert summary.accuracy == 1.0
    record = json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8"))
    assert record["normalized_prediction"] == "B"
    assert record["action_trace"][-1]["action"] == "RENDER_DONE"
    assert Path(record["trace_path"]).is_file()
    assert json.loads((output_dir / "summary.json").read_text())["accuracy"] == 1.0
