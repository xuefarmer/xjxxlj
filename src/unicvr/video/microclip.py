from __future__ import annotations

from unicvr.core.schemas import VideoRef, VisualInput
from unicvr.video.decode import VideoDecoder
from unicvr.video.sampling import FocusSampler


class MicroClipBuilder:
    def __init__(self, decoder: VideoDecoder, sampler: FocusSampler) -> None:
        self.decoder = decoder
        self.sampler = sampler

    def build(
        self,
        video: VideoRef,
        *,
        start_seconds: float,
        end_seconds: float,
        frame_budget: int,
    ) -> list[VisualInput]:
        timestamps = self.sampler.sample(
            video,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            frame_budget=frame_budget,
        )
        return self.decoder.extract(video, timestamps)
