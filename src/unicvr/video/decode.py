from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from unicvr.core.schemas import VideoRef, VisualInput
from unicvr.video.cache import FrameCache


class VideoDecodeError(RuntimeError):
    """Actionable video probe or decode failure."""


class VideoProbe:
    def probe(
        self,
        path: Path,
        *,
        video_id: str,
    ) -> VideoRef:
        if not path.is_file():
            raise VideoDecodeError(f"video does not exist: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VideoDecodeError(f"OpenCV cannot open video: {path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if fps <= 0 or frame_count <= 0:
            raise VideoDecodeError(
                f"invalid video metadata for {path}: fps={fps}, frames={frame_count}"
            )
        return VideoRef(
            video_id=video_id,
            path=path.resolve(),
            duration_seconds=max(0.0, (frame_count - 1) / fps),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )


class VideoDecoder:
    def __init__(
        self,
        cache: FrameCache,
        *,
        resize_long_edge: int = 768,
        jpeg_quality: int = 88,
    ) -> None:
        self.cache = cache
        self.resize_long_edge = resize_long_edge
        self.jpeg_quality = jpeg_quality

    def extract(self, video: VideoRef, timestamps_seconds: list[float]) -> list[VisualInput]:
        if video.fps is None or video.frame_count is None:
            raise VideoDecodeError("video must be probed before frame extraction")
        duration = video.duration_seconds or 0.0
        requested_indices = {
            min(video.frame_count - 1, max(0, round(min(max(0.0, item), duration) * video.fps)))
            for item in timestamps_seconds
        }
        missing_indices = {
            frame_index
            for frame_index in requested_indices
            if not self.cache.path_for(
                video.path,
                frame_index / video.fps,
                resize_long_edge=self.resize_long_edge,
                jpeg_quality=self.jpeg_quality,
            ).exists()
        }
        if len(missing_indices) >= 24:
            self._populate_cache_sequential(video, missing_indices)
        capture = cv2.VideoCapture(str(video.path))
        if not capture.isOpened():
            raise VideoDecodeError(f"cannot decode video: {video.path}")
        outputs = []
        emitted_indices: set[int] = set()
        try:
            for requested in timestamps_seconds:
                timestamp = min(max(0.0, requested), duration)
                frame_index = min(video.frame_count - 1, max(0, round(timestamp * video.fps)))
                actual_timestamp = frame_index / video.fps
                target = self.cache.path_for(
                    video.path,
                    actual_timestamp,
                    resize_long_edge=self.resize_long_edge,
                    jpeg_quality=self.jpeg_quality,
                )
                if not target.exists():
                    try:
                        actual_frame_index, frame = self._read_nearby(
                            capture,
                            video.path,
                            frame_index,
                        )
                    except VideoDecodeError:
                        # 源视频尾段损坏（NC 类整视频采样大面积出现）：
                        # 跳过该采样帧，其余帧继续（VLM 仅少看 0-2 帧）
                        continue
                    frame_index = actual_frame_index
                    actual_timestamp = frame_index / video.fps
                    target = self.cache.path_for(
                        video.path,
                        actual_timestamp,
                        resize_long_edge=self.resize_long_edge,
                        jpeg_quality=self.jpeg_quality,
                    )
                    frame_array = self._resize(frame)
                    if not target.exists() and not cv2.imwrite(
                        str(target), frame_array, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    ):
                        raise VideoDecodeError(f"failed to cache frame at {target}")
                if frame_index in emitted_indices:
                    continue
                emitted_indices.add(frame_index)
                outputs.append(
                    VisualInput(
                        video_id=video.video_id,
                        timestamp_seconds=actual_timestamp,
                        frame_index=frame_index,
                        local_path=target,
                        mime_type="image/jpeg",
                    )
                )
        finally:
            capture.release()
        return outputs

    def _populate_cache_sequential(
        self,
        video: VideoRef,
        frame_indices: set[int],
    ) -> None:
        if not frame_indices or video.fps is None:
            return
        capture = cv2.VideoCapture(str(video.path))
        if not capture.isOpened():
            raise VideoDecodeError(f"cannot decode video: {video.path}")
        try:
            for frame_index in range(max(frame_indices) + 1):
                if not capture.grab():
                    break
                if frame_index not in frame_indices:
                    continue
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    continue
                target = self.cache.path_for(
                    video.path,
                    frame_index / video.fps,
                    resize_long_edge=self.resize_long_edge,
                    jpeg_quality=self.jpeg_quality,
                )
                if target.exists():
                    continue
                if not cv2.imwrite(
                    str(target),
                    self._resize(np.asarray(frame, dtype=np.uint8)),
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                ):
                    raise VideoDecodeError(f"failed to cache frame at {target}")
        finally:
            capture.release()

    @staticmethod
    def _read_nearby(
        capture: cv2.VideoCapture,
        path: Path,
        frame_index: int,
    ) -> tuple[int, NDArray[np.uint8]]:
        # 双向搜索：长视频深帧处（NC 整视频采样，frame 10000+）ffmpeg 可能
        # 在目标帧附近连续解码失败；与 materialize._read_nearby 保持同策略
        for offset in (0, 1, 2, 5, 10, -1, -2, -5, -10):
            candidate = frame_index - offset
            if candidate < 0:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
            ok, frame = capture.read()
            if ok and frame is not None:
                return candidate, np.asarray(frame, dtype=np.uint8)
        raise VideoDecodeError(f"failed to decode {path} near frame {frame_index}")

    def _resize(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        height, width = frame.shape[:2]
        longest = max(width, height)
        if longest <= self.resize_long_edge:
            return frame
        scale = self.resize_long_edge / longest
        return cast(
            NDArray[np.uint8],
            cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            ),
        )
