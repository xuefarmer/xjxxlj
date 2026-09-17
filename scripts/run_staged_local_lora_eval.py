#!/usr/bin/env python3
"""Two-stage local-LoRA evaluation, isolated from the API evaluation path.

``preheat`` globally keeps three VLM endpoints busy while collecting each
sample's initial Observer reports. ``evaluate`` reuses those reports and runs
the original per-sample FORM -> evidence -> REVIEW logic serially.

No files under src/unicvr and no API evaluation scripts are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_TASKS = "PSS,CC,FSA,NC,PI,MSR,MOC"


def dotenv_values(path: Path) -> dict[str, str]:
    """Read only the small set of credentials/config values needed here.

    This intentionally never prints the values.  Stage-B API-LLM mode needs
    the original API LLM endpoint while it replaces only VLM_* with a local
    router, so relying on the process environment alone is error-prone.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key.replace("_", "").isalnum():
            values[key] = value.strip().strip("\"'")
    return values


class VlmRouter:
    """Three-way local OpenAI-compatible VLM router; no queueing beyond workers."""

    def __init__(self, targets: list[str], port: int) -> None:
        self._cycle = itertools.cycle(item.rstrip("/") for item in targets)
        self._lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _target(self) -> str:
        with self._lock:
            return next(self._cycle)

    def _handler(self):
        router = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlsplit(self.path)
                url = router._target() + parsed.path.removeprefix("/v1")
                if parsed.query:
                    url += "?" + parsed.query
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                headers = {key: value for key, value in self.headers.items()
                           if key.lower() in {"authorization", "content-type"}}
                try:
                    with httpx.Client(timeout=900.0) as client:
                        response = client.post(url, content=body, headers=headers)
                    data, status = response.content, response.status_code
                    content_type = response.headers.get("content-type", "application/json")
                except httpx.HTTPError as exc:
                    data = json.dumps({"error": {"message": f"local VLM router: {exc}"}}).encode()
                    status, content_type = 503, "application/json"
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    return

        return Handler

    def __enter__(self) -> "VlmRouter":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def local_env(args: argparse.Namespace, targets: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    if args.llm_mode == "api":
        dotenv = dotenv_values(ROOT / ".env")
        llm_base_url = args.llm_url or env.get("LLM_BASE_URL") or dotenv.get("LLM_BASE_URL")
        llm_api_key = env.get("LLM_API_KEY") or dotenv.get("LLM_API_KEY")
        llm_model = args.llm_model or env.get("LLM_MODEL") or dotenv.get("LLM_MODEL")
        if not all((llm_base_url, llm_api_key, llm_model)):
            raise SystemExit("API-LLM mode needs LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env or environment")
    else:
        llm_base_url = args.llm_url or "http://127.0.0.1:8000/v1"
        llm_api_key = "local"
        llm_model = args.llm_model or "qwen3-8b-all7-lora"
    if args.vlm_mode == "api":
        dotenv = dotenv_values(ROOT / ".env")
        vlm_base_url = args.vlm_url or env.get("VLM_BASE_URL") or dotenv.get("VLM_BASE_URL")
        vlm_api_key = env.get("VLM_API_KEY") or dotenv.get("VLM_API_KEY")
        vlm_model = args.vlm_model or env.get("VLM_MODEL") or dotenv.get("VLM_MODEL")
        if not all((vlm_base_url, vlm_api_key, vlm_model)):
            raise SystemExit("API-VLM mode needs VLM_BASE_URL, VLM_API_KEY and VLM_MODEL in .env or environment")
    else:
        vlm_base_url = f"http://127.0.0.1:{args.router_port}/v1"
        vlm_api_key = "local"
        vlm_model = args.vlm_model or "qwen3-vl-8b-all7-lora"
    env.update({
        "LLM_BASE_URL": llm_base_url,
        "LLM_API_KEY": llm_api_key,
        "LLM_MODEL": llm_model,
        "VLM_BASE_URL": vlm_base_url,
        "VLM_API_KEY": vlm_api_key,
        "VLM_MODEL": vlm_model,
        "UNICVR_LOCAL_VLM_PARALLELISM": str(max(1, len(targets))),
        # Corrupt-but-decodable H.264 streams can emit repetitive MMCO
        # diagnostics directly from FFmpeg.  OpenCV still signals a genuine
        # failed read to Python; only its stderr chatter is muted locally.
        "OPENCV_FFMPEG_LOGLEVEL": "-8",
        "OPENCV_LOG_LEVEL": "SILENT",
    })
    return env


def cache_signature(sample: Any, config: Any) -> str:
    """Bind reports to semantic inputs, never temporary materialized paths.

    FSA creates a temporary reference clip under ``media-cache-dir``.  Its
    pathname may legitimately differ between stage A and stage B; the cached
    observer report remains valid because the official QA row/config is the
    same.  Dynamic Focus in stage B will decode its own source media.
    """
    payload = sample.inference_payload()
    payload.pop("video_paths", None)
    normalized = json.loads(json.dumps(payload, default=str, sort_keys=True))
    config_payload = config.model_dump(mode="json")
    # Storage location changes do not change the frames, prompts, or report.
    config_payload["video"]["cache_dir"] = "<transient-frame-cache>"
    # Stage A and Stage B deliberately use different local proxy ports (and
    # may use different LLM replicas).  Those transport addresses do not
    # change the Observer prompt, sampled frames, or the report's semantics.
    # Keeping them in the signature made a cache produced through router 8010
    # look stale when consumed through router 8011.
    for backend_name in ("llm", "vlm"):
        if backend_name in config_payload:
            config_payload[backend_name]["base_url"] = f"<{backend_name}-runtime-endpoint>"
    raw = json.dumps({"sample": normalized, "config": config_payload}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_path(cache_root: Path, task: str, sample_id: str) -> Path:
    return cache_root / task / f"{sample_id}.json"


def answer_type_for(task: str) -> str:
    if task == "PSS":
        return "sequence"
    if task == "FSA":
        return "interval"
    return "choice"


def apply_frame_cache(config: Any, value: str | None) -> None:
    """Use an explicitly disposable decoded-frame cache for stage A if asked."""
    if value:
        config.video.cache_dir = (ROOT / value).resolve()


def valid_cache(path: Path, sample: Any, config: Any) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("signature") == cache_signature(sample, config) and bool(data.get("reports"))
    except (OSError, json.JSONDecodeError):
        return False


def preheat(args: argparse.Namespace) -> int:
    if args.vlm_mode != "local":
        raise SystemExit("preheat only supports --vlm-mode local; this diagnostic reuses its local reports")
    targets = [item.strip() for item in args.vlm_urls.split(",") if item.strip()]
    if not targets:
        raise SystemExit("--vlm-urls is empty")
    env = local_env(args, targets)
    os.environ.update({key: value for key, value in env.items() if key.startswith(("LLM_", "VLM_", "UNICVR_"))})
    # Must happen before importing Pipeline/Materializer: those import cv2,
    # which initializes FFmpeg's log level once per process.
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = env["OPENCV_FFMPEG_LOGLEVEL"]
    os.environ["OPENCV_LOG_LEVEL"] = env["OPENCV_LOG_LEVEL"]

    from unicvr.config import load_config
    from unicvr.core.pipeline import Pipeline
    from unicvr.core.pipeline import VideoProbe
    from unicvr.data.crossvid import CrossVidDatasetAdapter
    from unicvr.data.materialize import OpenCVMediaMaterializer
    from unicvr.plugins.registry import blocks_for_sample, visual_question_for_sample
    cache_root = (ROOT / args.cache_dir).resolve()
    # Stage A has no need to retain decoded images/proxy clips after their
    # report has been received.  Default to cache-private transient locations
    # instead of the shared .unicvr_cache tree.
    frame_root = (ROOT / args.frame_cache_dir).resolve() if args.frame_cache_dir else cache_root / "_transient_frames"
    media_root = (ROOT / args.media_cache_dir).resolve() if args.media_cache_dir else cache_root / "_transient_media"
    config = load_config(ROOT / args.config)
    apply_frame_cache(config, str(frame_root))
    materializer = OpenCVMediaMaterializer(media_root)
    event_prompt = Path(args.event_observer_prompt).resolve() if args.event_observer_prompt else None
    if event_prompt is not None and not event_prompt.is_file():
        raise SystemExit(f"event Observer prompt does not exist: {event_prompt}")
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if not tasks:
        raise SystemExit("--tasks is empty")
    high_density_tasks = {item.strip().upper() for item in args.high_density_tasks.split(",") if item.strip()}
    compact_tasks = {item.strip().upper() for item in args.compact_observer_tasks.split(",") if item.strip()}

    def warm_settings(task: str) -> tuple[float, int, int]:
        """Keep the staged warm pass identical to the Event-SFT/RL policy."""
        if task.upper() in high_density_tasks:
            return (args.high_density_seconds_per_frame,
                    args.high_density_min_warm_frames,
                    args.high_density_scan_frames)
        if task.upper() in compact_tasks:
            return (args.compact_scan_seconds_per_frame,
                    args.compact_min_warm_frames,
                    args.compact_scan_frames)
        return (config.video.scan_seconds_per_frame,
                config.budget.min_warm_frames_per_video,
                config.budget.scan_frames_per_video)

    # One client object is safe here: at most len(targets) request threads
    # call it concurrently.  Crucially, jobs are submitted as soon as ONE
    # sample has been decoded; do not decode the whole benchmark before the
    # first VLM request starts.
    try:
        with VlmRouter(targets, args.router_port):
            pipe = Pipeline(config)
            if event_prompt is not None:
                from experiments.event_level_observer.event_pipeline import EventLevelObserverAgent
                observer = EventLevelObserverAgent(pipe._vlm, config.generation, event_prompt)
                print(f"[preheat] Event Observer enabled: {event_prompt}", flush=True)
            else:
                observer = pipe.observer
            pending_samples: dict[tuple[str, str], dict[str, Any]] = {}
            frame_refs: dict[Path, int] = {}
            complete = submitted = skipped = 0
            max_in_flight = max(len(targets), len(targets) * 2)
            print(f"[preheat] streaming queue: workers={len(targets)}, max_in_flight={max_in_flight}", flush=True)

            def observe(job: tuple[str, str, int, Any, Any, Any, str, list[str] | None]) -> tuple[tuple[str, str], str, int, str, int, float, float]:
                task, sample_id, chunk_index, video, inputs, sample, visual_question, prompt_blocks = job
                report = observer.observe(
                    video_id=video.video_id, visual_inputs=inputs,
                    task_context=("Observe all objects, actions, states and temporal cues. "
                                  "Later stages compare across videos to answer: " + visual_question),
                    prompt_blocks=prompt_blocks,
                )
                return ((task, sample_id), video.video_id, chunk_index, report, len(inputs),
                        inputs[0].timestamp_seconds, inputs[-1].timestamp_seconds)

            def finish(future: Any, job: tuple[str, str, int, Any, Any, Any, str, list[str] | None]) -> None:
                nonlocal complete
                try:
                    key, video_id, chunk_index, report, frame_count, start, end = future.result()
                finally:
                    # Chunks can overlap by one frame. Delete a decoded JPEG
                    # only after its final in-flight VLM request has returned.
                    for item in job[4]:
                        remaining = frame_refs[item.local_path] - 1
                        frame_refs[item.local_path] = remaining
                        if remaining == 0:
                            item.local_path.unlink(missing_ok=True)
                info = pending_samples[key]
                info["chunks"].setdefault(video_id, []).append(
                    {"index": chunk_index, "report_text": report, "frame_count": frame_count,
                     "start_time": start, "end_time": end}
                )
                complete += 1
                if sum(len(items) for items in info["chunks"].values()) != info["expected"]:
                    return
                reports = []
                for video_id2 in info["video_order"]:
                    chunks = info["chunks"][video_id2]
                    chunks.sort(key=lambda item: item["index"])
                    reports.append({
                        "video_id": video_id2,
                        "report_text": "\n\n".join(item["report_text"] for item in chunks),
                        "frame_count": sum(item["frame_count"] for item in chunks),
                        "start_time": min(item["start_time"] for item in chunks),
                        "end_time": max(item["end_time"] for item in chunks),
                    })
                sample = info["sample"]
                payload = {
                    "sample_id": sample.sample_id, "task": info["task"],
                    "signature": cache_signature(sample, config),
                    "config_fingerprint": config.fingerprint(), "reports": reports,
                    "preheat_calls": [
                        {"role": "ObserverAgent", "backend": "openai_compatible", "model": config.vlm.model,
                         "visual_input_count": chunk["frame_count"], "usage": {}, "retry_count": 0}
                        for values in info["chunks"].values() for chunk in values
                    ],
                }
                out = info["output"]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                print(
                    f"=== [PREHEAT DONE] {info['task']}:{sample.sample_id} | "
                    f"all {info['expected']} Observer call(s) returned | "
                    f"{len(reports)} video report(s) cached | "
                    f"completed {complete} VLM calls ===",
                    flush=True,
                )

            with ThreadPoolExecutor(max_workers=len(targets)) as pool:
                futures: dict[Any, tuple[str, str, int, Any, Any, Any, str, list[str] | None]] = {}

                def drain_one() -> None:
                    future = next(as_completed(futures))
                    job = futures.pop(future)
                    finish(future, job)

                for task in tasks:
                    adapter = CrossVidDatasetAdapter(
                        crossvid_root=Path(args.crossvid_root), task=task,
                        materializer=materializer,
                    )
                    for sample in adapter.iter_samples(start=args.start, limit=args.limit):
                        out = cache_path(cache_root, task, sample.sample_id)
                        if valid_cache(out, sample, config):
                            skipped += 1
                            continue
                        videos = [VideoProbe().probe(path, video_id=f"v{i + 1}")
                                  for i, path in enumerate(sample.video_paths)]
                        visual_questions = {
                            video.video_id: visual_question_for_sample(sample, sample.question, video.video_id)
                            for video in videos
                        }
                        blocks = {
                            video.video_id: blocks_for_sample(
                                sample, "observer", question=visual_questions[video.video_id],
                                answer_type=answer_type_for(task), options=sample.options,
                                video_id=video.video_id,
                            ) for video in videos
                        }
                        key = (task, sample.sample_id)
                        info = pending_samples[key] = {
                            "sample": sample, "chunks": {}, "expected": 0, "output": out,
                            "task": task, "video_order": [video.video_id for video in videos],
                        }
                        sample_jobs = []
                        seconds_per_frame, minimum_frames, maximum_frames = warm_settings(task)
                        for video in videos:
                            budget = pipe.warm_sampler.frame_count(
                                video, minimum=minimum_frames,
                                maximum=maximum_frames,
                                seconds_per_frame=seconds_per_frame,
                            )
                            vis = pipe.decoder.extract(video, pipe.warm_sampler.sample(
                                video, frame_budget=budget, question=sample.question,
                            ))
                            for chunk_index, inputs in enumerate(pipe.warm_sampler.chunk(
                                vis, frames_per_call=config.video.scan_frames_per_call,
                                overlap_frames=config.video.scan_overlap_frames,
                            )):
                                info["expected"] += 1
                                sample_jobs.append((task, sample.sample_id, chunk_index, video, inputs,
                                                    sample, visual_questions[video.video_id], blocks[video.video_id]))
                        for job in sample_jobs:
                            for item in job[4]:
                                frame_refs[item.local_path] = frame_refs.get(item.local_path, 0) + 1
                        for job in sample_jobs:
                            while len(futures) >= max_in_flight:
                                drain_one()
                            futures[pool.submit(observe, job)] = job
                            submitted += 1
                while futures:
                    drain_one()
            print(f"[preheat] submitted={submitted}; completed={complete}; cache hits={skipped}", flush=True)
    finally:
        # Only paths dedicated to this staged run are removed.  Text reports
        # under cache_root/preheat remain untouched.
        shutil.rmtree(frame_root, ignore_errors=True)
        shutil.rmtree(media_root, ignore_errors=True)
    print(f"preheat cache: {cache_root}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    """Run official evaluator unchanged except for an in-process cached Phase 1."""
    targets = [item.strip() for item in args.vlm_urls.split(",") if item.strip()]
    if args.vlm_mode == "local" and not targets:
        raise SystemExit("--vlm-urls is empty")
    if args.vlm_mode == "api":
        targets = []
    env = local_env(args, targets)
    command = [
        sys.executable, "-u", str(ROOT / "scripts/staged_local_lora_worker.py"),
        "--config", str(ROOT / args.config), "--crossvid-root", str(Path(args.crossvid_root)),
        "--tasks", args.tasks, "--start", str(args.start), "--limit", str(args.limit),
        "--cache-dir", str((ROOT / args.cache_dir).resolve()),
        "--output-dir", str((ROOT / args.output_dir).resolve()),
        "--media-cache-dir", str((ROOT / (args.media_cache_dir or ".unicvr_cache/local-lora-stageb")).resolve()),
        "--frame-cache-dir", args.frame_cache_dir or "",
    ]
    if args.sample_ids:
        command += ["--sample-ids", args.sample_ids]
    if args.resume:
        command.append("--resume")
    # Phase B only sends dynamic Focus/Comparer requests to the VLM router.
    if args.vlm_mode == "local":
        with VlmRouter(targets, args.router_port):
            return subprocess.call(command, cwd=ROOT, env=env)
    return subprocess.call(command, cwd=ROOT, env=env)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("preheat", "evaluate"):
        child = sub.add_parser(name)
        child.add_argument("--config", default="configs/local_dual_lora_eval.yaml")
        child.add_argument("--crossvid-root", default="/media/data6/xuejj/CrossVid")
        child.add_argument("--tasks", default=DEFAULT_TASKS)
        child.add_argument("--start", type=int, default=0)
        child.add_argument("--limit", type=int, default=25)
        child.add_argument("--cache-dir", default="outputs/local-lora-preheat")
        child.add_argument("--media-cache-dir", default=None,
                           help="optional temporary media location; preheat auto-deletes it")
        child.add_argument("--frame-cache-dir", default=None,
                           help="optional disposable decoded-frame cache (use for preheat only)")
        child.add_argument("--event-observer-prompt", default=None,
                           help="opt in to the Event-SFT warm Observer; Focus/REVIEW remain unchanged")
        child.add_argument("--high-density-tasks", default="PSS,FSA,PI,MSR,MOC",
                           help="short/UAV tasks using dense Event warm sampling")
        child.add_argument("--high-density-seconds-per-frame", type=float, default=2.0)
        child.add_argument("--high-density-min-warm-frames", type=int, default=16)
        child.add_argument("--high-density-scan-frames", type=int, default=192)
        child.add_argument("--compact-observer-tasks", default="CC,NC",
                           help="long-video tasks using compact Event warm sampling")
        child.add_argument("--compact-scan-seconds-per-frame", type=float, default=8.0)
        child.add_argument("--compact-min-warm-frames", type=int, default=12)
        child.add_argument("--compact-scan-frames", type=int, default=32)
        child.add_argument("--llm-mode", choices=("local", "api"), default="local",
                           help="local: serve_sft endpoint; api: use LLM_* values from .env")
        child.add_argument("--llm-url", default=None,
                           help="override the LLM endpoint (defaults depend on --llm-mode)")
        child.add_argument("--llm-model", default=None,
                           help="override model name (API mode otherwise uses .env LLM_MODEL)")
        child.add_argument("--vlm-mode", choices=("local", "api"), default="local",
                           help="local: route to --vlm-urls; api: use VLM_* values from .env")
        child.add_argument("--vlm-url", default=None,
                           help="override VLM endpoint in --vlm-mode api")
        child.add_argument("--vlm-model", default=None,
                           help="override VLM model name")
        child.add_argument("--vlm-urls", default="http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1")
        child.add_argument("--router-port", type=int, default=8010)
        if name == "evaluate":
            child.add_argument("--output-dir", default="outputs/eval-local-lora-staged")
            child.add_argument("--sample-ids", default="",
                               help="comma-separated exact IDs for controlled Stage-B diagnostics")
            child.add_argument("--resume", action="store_true",
                               help="skip completed result IDs and append safely after interruption")
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.start < 0 or args.limit <= 0:
        raise SystemExit("--start must be nonnegative and --limit must be positive")
    if args.command == "preheat":
        return preheat(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
