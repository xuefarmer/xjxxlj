from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from unicvr.config import load_config, validate_credentials
from unicvr.data.schema import load_sample

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_configurable_crossvid_field_mapping(tmp_path: Path) -> None:
    sample_path = tmp_path / "mapped.json"
    sample_path.write_text(
        json.dumps(
            {
                "id": "mapped-1",
                "videos": [
                    str(FIXTURES / "video_a.mp4"),
                    str(FIXTURES / "video_b.mp4"),
                ],
                "query": "What relation holds?",
                "gold": "B",
            }
        ),
        encoding="utf-8",
    )
    sample = load_sample(
        sample_path,
        {
            "sample_id": "id",
            "video_paths": "videos",
            "question": "query",
            "answer": "gold",
        },
    )
    assert sample.sample_id == "mapped-1"
    assert sample.answer == "B"


def test_malformed_sample_reports_missing_video_paths(tmp_path: Path) -> None:
    sample_path = tmp_path / "bad.json"
    sample_path.write_text(
        json.dumps(
            {
                "sample_id": "bad",
                "video_paths": ["missing-a.mp4", "missing-b.mp4"],
                "question": "question",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="video files do not exist"):
        load_sample(sample_path)


def test_api_example_validates_without_exposing_credentials() -> None:
    config = load_config(ROOT / "configs" / "api_high_capability.example.yaml")
    missing = validate_credentials(config)
    assert config.llm.backend == "openai_compatible"
    assert config.vlm.backend == "openai_compatible"
    assert config.llm.model == "${LLM_MODEL}" or config.llm.model
    assert all("API_KEY=" not in item for item in missing)


def test_llm_and_vlm_backends_are_independently_configurable() -> None:
    config = load_config(ROOT / "configs" / "mock_debug.yaml")
    changed = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"model": "reasoning-model"}),
            "vlm": config.vlm.model_copy(update={"model": "vision-model"}),
        }
    )
    assert changed.llm.model == "reasoning-model"
    assert changed.vlm.model == "vision-model"
