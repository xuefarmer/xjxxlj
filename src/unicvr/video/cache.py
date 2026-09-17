from __future__ import annotations

import hashlib
from pathlib import Path


class FrameCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(
        self,
        video_path: Path,
        timestamp_seconds: float,
        *,
        resize_long_edge: int,
        jpeg_quality: int,
    ) -> Path:
        source = video_path.resolve()
        stat = source.stat()
        key = (
            f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{timestamp_seconds:.6f}|"
            f"{resize_long_edge}|{jpeg_quality}"
        )
        digest = hashlib.sha256(key.encode()).hexdigest()
        target = self.root / digest[:2] / f"{digest}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
