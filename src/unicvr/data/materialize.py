from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

# OpenCV's ffmpeg video decoder is not safe under concurrent VideoCapture
# (h264 mmco errors, then hangs). Serialize video decode/encode globally.
_DECODE_LOCK = threading.Lock()


class CrossVidMediaMaterializer(Protocol):
    def clip(
        self,
        source: Path,
        intervals: Sequence[tuple[float, float]],
        *,
        cache_key: str,
        frame_dir_fps: float | None = None,
    ) -> Path: ...

    def uav_views(
        self,
        *,
        uav_root: Path,
        scene_id: int,
        objects: Sequence[dict[str, Any]],
        cache_key: str,
    ) -> list[Path]: ...


class OpenCVMediaMaterializer:
    """Build small deterministic proxy videos for interval and frame-folder inputs."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        proxy_frames: int = 96,
        output_fps: float = 8.0,
        resize_long_edge: int = 960,
    ) -> None:
        if proxy_frames < 2:
            raise ValueError("proxy_frames must be at least two")
        if output_fps <= 0 or resize_long_edge <= 0:
            raise ValueError("proxy video settings must be positive")
        self.cache_dir = cache_dir
        self.proxy_frames = proxy_frames
        self.output_fps = output_fps
        self.resize_long_edge = resize_long_edge
        # PSS repeatedly clips different temporal segments from a small set
        # of EPIC frame directories.  Listing and parsing a directory of
        # thousands of JPEGs for every sample dominated later SFT-data
        # preparation.  Cache the immutable directory index for the lifetime
        # of this materializer, invalidating it when the directory mtime moves.
        self._frame_dir_index_cache: dict[
            Path, tuple[int, list[Path], int, dict[int, int], list[int]]
        ] = {}
        self._frame_dir_index_lock = threading.Lock()

    def clip(
        self,
        source: Path,
        intervals: Sequence[tuple[float, float]],
        *,
        cache_key: str,
        frame_dir_fps: float | None = None,
    ) -> Path:
        if source.is_dir():
            return self._clip_frame_dir(
                source,
                intervals,
                cache_key=cache_key,
                fps=frame_dir_fps,
            )
        if not source.is_file():
            raise FileNotFoundError(f"clip source does not exist: {source}")
        normalized = [(float(start), float(end)) for start, end in intervals]
        if not normalized or any(start < 0 or end <= start for start, end in normalized):
            raise ValueError("clip intervals must be nonempty positive spans")
        identity = {
            "kind": "clip",
            "source": str(source.resolve()),
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "intervals": normalized,
            "proxy_frames": self.proxy_frames,
            "output_fps": self.output_fps,
            "resize_long_edge": self.resize_long_edge,
            "cache_key": cache_key,
        }
        output = self._output_path(identity)
        if output.is_file():
            return output

        # OpenCV's ffmpeg backend deadlocks when several threads decode h264
        # concurrently (mmco errors then hang at workers>1). Serialize the
        # decode+encode section; API calls from other threads stay parallel.
        with _DECODE_LOCK:
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError(f"OpenCV cannot open clip source: {source}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frame_count <= 0:
                capture.release()
                raise RuntimeError(f"invalid clip source metadata: {source}")
            duration = (frame_count - 1) / fps
            bounded = [
                (min(start, duration), min(end, duration))
                for start, end in normalized
                if start < duration
            ]
            bounded = [(start, end) for start, end in bounded if end > start]
            if not bounded:
                capture.release()
                raise ValueError(f"clip intervals lie outside source duration: {source}")
            timestamps = _sample_intervals(bounded, self.proxy_frames)
            frames: list[NDArray[np.uint8]] = []
            skipped = 0
            try:
                for timestamp in timestamps:
                    index = min(frame_count - 1, max(0, round(timestamp * fps)))
                    try:
                        frame = _read_nearby(capture, index)
                    except RuntimeError:
                        # 源视频尾段损坏（NC 类任务大面积出现）：跳过该采样帧。
                        # 每视频仅损失 0-2 帧（96 帧采样），视觉无差别；
                        # 结果确定性幂等，缓存 key 不变，已缓存 proxy 不受影响。
                        skipped += 1
                        continue
                    frames.append(self._resize(frame))
            finally:
                capture.release()
            if skipped:
                print(f"[materialize] {output.name}: skipped {skipped} corrupted "
                      f"frame(s) in {output.stem[:12]}…", flush=True)
            if not frames:
                raise ValueError(f"clip source unreadable throughout: {source}")
            self._write_video(output, frames)
        return output

    def _clip_frame_dir(
        self,
        source: Path,
        intervals: Sequence[tuple[float, float]],
        *,
        cache_key: str,
        fps: float | None,
    ) -> Path:
        """Materialize interval clips from a directory of sequentially numbered
        frame files (e.g. EPIC-Kitchens `frame_%010d.jpg` at 30fps)."""
        if fps is None:
            fps = 30.0
        if fps <= 0:
            raise ValueError("frame_dir_fps must be positive")
        source = source.resolve()
        mtime_ns = source.stat().st_mtime_ns
        with self._frame_dir_index_lock:
            cached = self._frame_dir_index_cache.get(source)
            if cached is None or cached[0] != mtime_ns:
                frames = sorted(
                    path for path in source.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if not frames:
                    raise ValueError(f"frame directory is empty: {source}")
                first_id = _frame_number(frames[0])
                id_to_index = {_frame_number(path): index for index, path in enumerate(frames)}
                ids = sorted(id_to_index)
                cached = (mtime_ns, frames, first_id, id_to_index, ids)
                self._frame_dir_index_cache[source] = cached
            _, frames, first_id, id_to_index, ids = cached
        identity = {
            "kind": "frame_dir",
            "source": str(source.resolve()),
            "frame_count": len(frames),
            "first": frames[0].name,
            "last": frames[-1].name,
            "mtime_ns": mtime_ns,
            "fps": float(fps),
            "intervals": [(float(start), float(end)) for start, end in intervals],
            "proxy_frames": self.proxy_frames,
            "output_fps": self.output_fps,
            "resize_long_edge": self.resize_long_edge,
            "cache_key": cache_key,
        }
        output = self._output_path(identity)
        if output.is_file():
            return output

        # frame id -> time offset in seconds (1-based file numbering).
        # ``first_id``, ``id_to_index`` and ``ids`` come from the cached
        # directory index above.
        duration = (ids[-1] - first_id + 1) / fps
        bounded = [
            (min(start, duration), min(end, duration))
            for start, end in identity["intervals"]
            if start < duration
        ]
        bounded = [(start, end) for start, end in bounded if end > start]
        if not bounded:
            raise ValueError(f"clip intervals lie outside source duration: {source}")
        timestamps = _sample_intervals(bounded, self.proxy_frames)
        proxies: list[NDArray[np.uint8]] = []
        for timestamp in timestamps:
            frame_id = first_id + int(round(timestamp * fps))
            target = _nearest_frame_id(ids, frame_id)
            path = frames[id_to_index[target]]
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"failed to read frame: {path}")
            proxies.append(self._resize(np.asarray(image, dtype=np.uint8)))
        if not proxies:
            raise ValueError(f"frame directory unreadable throughout: {source}")
        self._write_video(output, proxies)
        return output

    def uav_views(
        self,
        *,
        uav_root: Path,
        scene_id: int,
        objects: Sequence[dict[str, Any]],
        cache_key: str,
    ) -> list[Path]:
        outputs = []
        for view in (1, 2):
            frame_dir = uav_root / "frames" / str(view) / f"{scene_id}-{view}"
            bbox_path = uav_root / "bbox" / str(view) / f"{scene_id}.json"
            if not frame_dir.is_dir() or not bbox_path.is_file():
                raise FileNotFoundError(
                    f"missing UAV view assets for scene={scene_id}, view={view}"
                )
            image_paths = sorted(
                path
                for path in frame_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not image_paths:
                raise ValueError(f"UAV frame directory is empty: {frame_dir}")
            identity = {
                "kind": "uav",
                "scene_id": scene_id,
                "view": view,
                "objects": list(objects),
                "frame_count": len(image_paths),
                "bbox_size": bbox_path.stat().st_size,
                "bbox_mtime_ns": bbox_path.stat().st_mtime_ns,
                "proxy_frames": self.proxy_frames,
                "output_fps": self.output_fps,
                "resize_long_edge": self.resize_long_edge,
                "cache_key": cache_key,
            }
            output = self._output_path(identity)
            if not output.is_file():
                bbox_by_id = _load_bboxes(bbox_path)
                indices = _uniform_indices(len(image_paths), self.proxy_frames)
                frames = []
                prefix = "A" if view == 1 else "B"
                for index in indices:
                    frame = cv2.imread(str(image_paths[index]))
                    if frame is None:
                        raise RuntimeError(f"failed to read UAV frame: {image_paths[index]}")
                    frame_array = np.asarray(frame, dtype=np.uint8)
                    _draw_object_labels(
                        frame_array,
                        frame_index=index,
                        objects=objects,
                        bbox_by_id=bbox_by_id,
                        prefix=prefix,
                    )
                    frames.append(self._resize(frame_array))
                self._write_video(output, frames)
            outputs.append(output)
        return outputs

    def _output_path(self, identity: dict[str, Any]) -> Path:
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        output = self.cache_dir / digest[:2] / f"{digest}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    def _write_video(self, output: Path, frames: Sequence[NDArray[np.uint8]]) -> None:
        if not frames:
            raise ValueError("cannot materialize an empty proxy video")
        height, width = frames[0].shape[:2]
        # unique tmp name: concurrent workers may encode the same cache key
        temporary = output.parent / f"{output.stem}.{threading.get_ident()}.tmp.mp4"
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter.fourcc(*"mp4v"),
            self.output_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot create proxy video: {temporary}")
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    resized = np.asarray(
                        cv2.resize(
                            frame,
                            (width, height),
                            interpolation=cv2.INTER_AREA,
                        ),
                        dtype=np.uint8,
                    )
                else:
                    resized = frame
                writer.write(resized)
        finally:
            writer.release()
        if not temporary.is_file() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"proxy video encoding failed: {output}")
        temporary.replace(output)

    def _resize(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        height, width = frame.shape[:2]
        longest = max(height, width)
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


def _sample_intervals(
    intervals: Sequence[tuple[float, float]],
    limit: int,
) -> list[float]:
    lengths = [end - start for start, end in intervals]
    total = sum(lengths)
    count = min(limit, max(2, round(total * 2)))
    positions = np.linspace(0.0, total, num=count, endpoint=True)
    timestamps = []
    for position in positions:
        offset = float(position)
        consumed = 0.0
        assert len(intervals) == len(lengths), "intervals/lengths length mismatch"
        for index, ((start, end), length) in enumerate(zip(intervals, lengths)):
            if offset <= consumed + length or index == len(intervals) - 1:
                timestamps.append(min(end, start + max(0.0, offset - consumed)))
                break
            consumed += length
    return timestamps


def _frame_number(path: Path) -> int:
    """Extract the leading integer run from a frame filename (e.g.
    frame_0000001811.jpg -> 1811)."""
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def _nearest_frame_id(ids: Sequence[int], target: int) -> int:
    """Return the frame id closest to target (handles sparse/missing frames)."""
    import bisect

    position = bisect.bisect_left(ids, target)
    if position == 0:
        return ids[0]
    if position == len(ids):
        return ids[-1]
    before, after = ids[position - 1], ids[position]
    return before if target - before <= after - target else after


def _uniform_indices(frame_count: int, limit: int) -> list[int]:
    count = min(frame_count, limit)
    return sorted(
        {
            min(frame_count - 1, max(0, round(float(value))))
            for value in np.linspace(0, frame_count - 1, num=count)
        }
    )


def _read_nearby(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> NDArray[np.uint8]:
    # 双向搜索：长视频深帧处 ffmpeg 在目标帧附近可能连续解码失败
    # （NC 类任务 seek 到 frame 10000+ 曾大面积 failed），先往前找，
    # 失败再往后找最近的可用帧。
    offsets = (0, 1, 2, 5, 10, -1, -2, -5, -10)
    for offset in offsets:
        candidate = frame_index - offset
        if candidate < 0:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, frame = capture.read()
        if ok and frame is not None:
            return np.asarray(frame, dtype=np.uint8)
    raise RuntimeError(f"failed to decode source near frame {frame_index}")


def _load_bboxes(path: Path) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"UAV bbox JSON must contain a list: {path}")
    result: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    for entity in raw:
        if not isinstance(entity, dict) or not isinstance(entity.get("id"), int):
            continue
        boxes = entity.get("bbox")
        if not isinstance(boxes, dict):
            continue
        parsed: dict[int, tuple[int, int, int, int]] = {}
        for frame_key, box in boxes.items():
            if not isinstance(box, dict) or box.get("outside") is True:
                continue
            values = [box.get(key) for key in ("xtl", "ytl", "xbr", "ybr")]
            if not all(isinstance(value, (int, float)) for value in values):
                continue
            coordinates = tuple(
                round(float(value)) for value in values if isinstance(value, (int, float))
            )
            if len(coordinates) == 4:
                parsed[int(frame_key)] = (
                    coordinates[0],
                    coordinates[1],
                    coordinates[2],
                    coordinates[3],
                )
        result[entity["id"]] = parsed
    return result


def _draw_object_labels(
    frame: NDArray[np.uint8],
    *,
    frame_index: int,
    objects: Sequence[dict[str, Any]],
    bbox_by_id: dict[int, dict[int, tuple[int, int, int, int]]],
    prefix: str,
) -> None:
    colors = (
        (0, 255, 255),
        (255, 128, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 128, 255),
    )
    height, width = frame.shape[:2]
    for index, item in enumerate(objects, start=1):
        object_id = item.get("id")
        if not isinstance(object_id, int):
            continue
        box = bbox_by_id.get(object_id, {}).get(frame_index)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1, x2 = sorted((min(max(x1, 0), width - 1), min(max(x2, 0), width - 1)))
        y1, y2 = sorted((min(max(y1, 0), height - 1), min(max(y2, 0), height - 1)))
        color = colors[(index - 1) % len(colors)]
        label = f"{prefix}{index} {item.get('label', 'object')}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (min(width - 1, x1 + 180), y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
