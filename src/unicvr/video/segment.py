from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from unicvr.core.schemas import VideoRef


class TemporalSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    change_score: float = Field(ge=0, le=1)


class TemporalSegmenter:
    """Deterministic low-cost segmentation with a documented uniform fallback."""

    def __init__(self, *, target_segment_seconds: float = 8.0) -> None:
        if target_segment_seconds <= 0:
            raise ValueError("target_segment_seconds must be positive")
        self.target_segment_seconds = target_segment_seconds

    def segment(self, video: VideoRef) -> list[TemporalSegment]:
        duration = video.duration_seconds or 0.0
        if duration == 0:
            return [TemporalSegment(start_seconds=0, end_seconds=0, change_score=0)]
        count = max(1, round(duration / self.target_segment_seconds))
        width = duration / count
        return [
            TemporalSegment(
                start_seconds=index * width,
                end_seconds=duration if index == count - 1 else (index + 1) * width,
                change_score=0.0,
            )
            for index in range(count)
        ]
