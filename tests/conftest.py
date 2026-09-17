from __future__ import annotations

from pathlib import Path

import pytest

from unicvr.config import AppConfig, load_config
from unicvr.core.pipeline import Pipeline
from unicvr.core.state import PipelineState
from unicvr.data.schema import MultiVideoSample, load_sample

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    config = load_config(ROOT / "configs" / "mock_debug.yaml")
    return config.model_copy(
        update={
            "video": config.video.model_copy(update={"cache_dir": tmp_path / "frames"}),
            "trace": config.trace.model_copy(update={"output_dir": tmp_path / "outputs"}),
        }
    )


@pytest.fixture
def sample() -> MultiVideoSample:
    return load_sample(FIXTURES / "mock_sample.json")


@pytest.fixture
def completed_run(
    app_config: AppConfig,
    sample: MultiVideoSample,
) -> tuple[Pipeline, PipelineState]:
    pipeline = Pipeline(app_config)
    return pipeline, pipeline.run(sample)
