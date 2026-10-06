"""``calibrex check`` (plan, run and estimate) for callers that hold files but no command line.

The browser check page (``docs/app/check.html``) mounts the dropped files in
Pyodide and calls :func:`plan_bag_for_browser`; it is the same planner as
``calibrex check --plan`` (:func:`calibrex.check.runner.build_calibration_check`),
with the user's file names in place of the virtual mount paths, so the artifact
and the printed command line read like a local run. Nothing here imports ROS,
OpenCV or Open3D.
"""

from __future__ import annotations

import json
import math
import shlex
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from calibrex.check.progress import CheckProgress
from calibrex.check.roles import parse_topic_kinds
from calibrex.check.runner import (
    CheckRunOptions,
    build_calibration_check,
    format_check_table,
    parse_frame_map,
)
from calibrex.core.exceptions import CalibrexError, DatasetError

STORAGE_SUFFIXES = (".db3", ".mcap")

#: Pairs whose estimator reads only numpy, scipy and the bag, so they run in Pyodide.
#: camera-imu / camera-focal also need OpenCV (``opencv-python`` ships with Pyodide 0.27).
BROWSER_RUNNABLE_PAIRS = (
    "imu-lidar",
    "lidar-lidar",
    "lidar-vehicle",
    "imu-vehicle",
    "ins-lidar",
    "lidar-wheel_odometry",
    "gnss-lidar",
    "gnss-imu",
)
BROWSER_CAMERA_PAIRS = ("camera-imu", "camera-focal")
#: Decoded-scan reuse budget (MB). WebAssembly is 32-bit (4 GB), the native default is 2048.
BROWSER_SCAN_MEMORY_MB = 512
BROWSER_NOTE = (
    "plan computed in the browser (Pyodide); the bag and calibration files were read from "
    "local files and never uploaded. Paths are the names of the dropped files."
)


def locate_bag(directory: str | Path) -> Path:
    """Return the bag path inside a directory of dropped files.

    A directory with ``metadata.yaml`` is a rosbag2 directory (the metadata names the
    storage file). Without it, exactly one ``.db3`` or ``.mcap`` file is read as a
    bare storage file.
    """

    root = Path(directory)
    if (root / "metadata.yaml").is_file():
        return root
    storage = sorted(
        item
        for item in root.iterdir()
        if item.is_file() and item.suffix.lower() in STORAGE_SUFFIXES
    )
    if len(storage) == 1:
        return storage[0]
    if not storage:
        msg = "no rosbag2 data found: add metadata.yaml and the .db3/.mcap file, or one bare .mcap"
        raise DatasetError(msg)
    names = ", ".join(item.name for item in storage)
    msg = f"several storage files ({names}) without metadata.yaml: add the bag's metadata.yaml"
    raise DatasetError(msg)


def check_command(
    bag_label: str,
    *,
    tf_labels: Sequence[str] = (),
    vehicle_frame: str | None = None,
    frame_map: Sequence[str] = (),
    topic_kind: Sequence[str] = (),
    plan: bool = False,
    outputs: bool = False,
) -> list[str]:
    """The ``calibrex check`` argument vector (program first) that a local run would use."""

    argv = ["calibrex", "check", bag_label]
    for label in tf_labels:
        argv += ["--tf", label]
    if vehicle_frame:
        argv += ["--vehicle-frame", vehicle_frame]
    for item in frame_map:
        argv += ["--frame-map", item]
    for item in topic_kind:
        argv += ["--topic-kind", item]
    if plan:
        argv.append("--plan")
    if outputs:
        argv += ["--output", "check.json", "--html", "check.html"]
    return argv


def plan_bag_for_browser(
    bag: str | Path,
    *,
    tf_files: Sequence[str | Path] = (),
    vehicle_frame: str | None = None,
    frame_map: Sequence[str] = (),
    topic_kind: Sequence[str] = (),
    bag_label: str | None = None,
    tf_labels: Sequence[str] | None = None,
    command_bag: str | None = None,
) -> dict[str, Any]:
    """Plan a bag and return the artifact with the matching local command lines.

    ``bag_label`` and ``tf_labels`` are the names shown in the artifact and in the
    commands (the user's real paths or file names); they default to the paths given.
    ``command_bag`` replaces the bag in the printed ``command`` (for example a
    ``path/to/bag`` placeholder when only loose files were dropped).
    The result is JSON-serialisable: ``artifact`` is the ``slac.calibration_check/v0.1``
    mapping, ``command`` the full local run, ``plan_command`` the ``--plan`` run.
    """

    bag_path = Path(bag)
    label = bag_label or str(bag_path)
    labels = list(tf_labels) if tf_labels is not None else [str(path) for path in tf_files]
    if len(labels) != len(tf_files):
        msg = "tf_labels must have one entry per tf file"
        raise DatasetError(msg)
    frame_overrides = parse_frame_map(frame_map)
    topic_kinds = parse_topic_kinds(topic_kind)
    vehicle = vehicle_frame.strip() if vehicle_frame and vehicle_frame.strip() else None
    artifact = build_calibration_check(
        bag_path,
        tf_files=tf_files,
        vehicle_frame=vehicle,
        frame_overrides=frame_overrides,
        topic_kinds=topic_kinds,
        command=check_command(
            label,
            tf_labels=labels,
            vehicle_frame=vehicle,
            frame_map=frame_map,
            topic_kind=topic_kind,
            plan=True,
        ),
    )
    remaining = iter(labels)
    sources = [
        source.model_copy(
            update={"path": label if source.kind == "bag_tf_static" else next(remaining)}
        )
        for source in artifact.candidate_sources
    ]
    provenance = artifact.provenance.model_copy(
        update={"notes": [*artifact.provenance.notes, BROWSER_NOTE]}
    )
    artifact = artifact.model_copy(
        update={
            "bag": artifact.bag.model_copy(update={"path": label}),
            "candidate_sources": sources,
            "provenance": provenance,
        }
    )
    return {
        "artifact": artifact.model_dump(mode="json", exclude_none=True),
        "command": shlex.join(
            check_command(
                command_bag or label,
                tf_labels=labels,
                vehicle_frame=vehicle,
                frame_map=frame_map,
                topic_kind=topic_kind,
                outputs=True,
            )
        ),
        "plan_command": shlex.join(
            check_command(
                command_bag or label,
                tf_labels=labels,
                vehicle_frame=vehicle,
                frame_map=frame_map,
                topic_kind=topic_kind,
                plan=True,
            )
        ),
    }


def plan_request_json(request_json: str) -> str:
    """Run one plan request given as JSON and return the outcome as JSON.

    The request holds ``bag_dir`` (a directory of dropped files, see
    :func:`locate_bag`), ``tf_files`` (paths), and optionally ``tf_labels``,
    ``vehicle_frame``, ``frame_map``, ``bag_label`` and ``command_bag``. The reply is
    ``{"ok": true, "artifact": ..., "command": ..., "plan_command": ..., "bag": path}``
    or ``{"ok": false, "error": "..."}``: a bad bag or calibration file is a message
    for the page, not an exception across the JS boundary.
    """

    try:
        request = json.loads(request_json)
        bag = locate_bag(request["bag_dir"])
        result = plan_bag_for_browser(
            bag,
            tf_files=request.get("tf_files", []),
            tf_labels=request.get("tf_labels"),
            vehicle_frame=request.get("vehicle_frame"),
            frame_map=request.get("frame_map", []),
            topic_kind=request.get("topic_kind", []),
            bag_label=request.get("bag_label"),
            command_bag=request.get("command_bag"),
        )
    except CalibrexError as error:
        return json.dumps({"ok": False, "error": str(error)})
    except Exception as error:  # corrupt files raise sqlite, yaml or pydantic errors
        return json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"})
    return json.dumps(_finite_json({"ok": True, "bag": str(bag), **result}))


EventSink = Callable[[str], None]


class EventProgress(CheckProgress):
    """A :class:`CheckProgress` that posts JSON events (``{"event": ...}``) to a sink.

    The browser worker passes a function that ``postMessage``s each string to the page.
    Tick events are throttled to ``tick_interval_s`` so the message queue stays short.
    """

    def __init__(
        self,
        sink: EventSink,
        clock: Callable[[], float] = time.monotonic,
        tick_interval_s: float = 0.25,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._interval = tick_interval_s
        self._last_tick = float("-inf")
        self._scan_total: int | None = None
        self._pair_started = clock()

    def _send(self, event: str, **fields: Any) -> None:
        self._sink(json.dumps({"event": event, **fields}))

    def run_started(self, total_pairs: int) -> None:
        self._send("run_started", total_pairs=total_pairs)

    def pair_started(self, index: int, total: int, label: str) -> None:
        self._pair_started = self._clock()
        self._scan_total = None
        self._send("pair_started", index=index, total=total, label=label)

    def set_scan_total(self, total: int | None) -> None:
        self._scan_total = total

    def stage(self, text: str) -> None:
        self._last_tick = float("-inf")
        self._send("stage", text=text, elapsed_s=self._clock() - self._pair_started)

    def tick(self, done: int, total: int | None = None) -> None:
        now = self._clock()
        if now - self._last_tick < self._interval:
            return
        self._last_tick = now
        self._send(
            "tick",
            done=done,
            total=total if total is not None else self._scan_total,
            elapsed_s=now - self._pair_started,
        )

    def pair_skipped(self, label: str, code: str, reason: str) -> None:
        self._send("pair_skipped", label=label, code=code, reason=reason)

    def pair_failed(self, label: str, error: str) -> None:
        self._send("pair_failed", label=label, error=error)

    def pair_finished(
        self, label: str, status: str, runtime_s: float, from_cache: bool | None
    ) -> None:
        self._send("pair_finished", label=label, status=status, runtime_s=runtime_s)

    def run_finished(self) -> None:
        self._send("run_finished")


def run_options_from_request(request: dict[str, Any], work_dir: Path) -> CheckRunOptions:
    """The :class:`CheckRunOptions` of a browser run request (no cache; bounded scan memory)."""

    pairs = request.get("pairs")
    max_duration = request.get("max_duration_s")
    return CheckRunOptions(
        pairs=tuple(pairs) if pairs else None,
        max_duration_s=float(max_duration) if max_duration else None,
        camera=(request.get("camera") or "").strip() or None,
        evidence_dir=work_dir / "evidence",
        base_dir=work_dir,
        cache_dir=None,
        scan_memory_mb=int(request.get("scan_memory_mb", BROWSER_SCAN_MEMORY_MB)),
    )


def run_command(
    mode: Literal["check", "estimate"],
    label: str,
    *,
    tf_labels: Sequence[str] = (),
    vehicle_frame: str | None = None,
    frame_map: Sequence[str] = (),
    topic_kind: Sequence[str] = (),
    camera: str | None = None,
    pairs: Sequence[str] = (),
    max_duration_s: float | None = None,
) -> str:
    """The local ``calibrex check`` / ``calibrex estimate`` command equal to a browser run."""

    argv = ["calibrex", mode, label]
    for item in tf_labels:
        argv += ["--tf", item]
    if vehicle_frame:
        argv += ["--vehicle-frame", vehicle_frame]
    for item in frame_map:
        argv += ["--frame-map", item]
    for item in topic_kind:
        argv += ["--topic-kind", item]
    if camera:
        argv += ["--camera", camera]
    if pairs:
        argv += ["--pairs", ",".join(pairs)]
    if max_duration_s:
        argv += ["--max-duration-s", f"{max_duration_s:g}"]
    if mode == "check":
        argv += ["--output", "check.json", "--html", "check.html"]
    else:
        argv += ["--output", "estimate"]
    return shlex.join(argv)


def run_request_json(request_json: str, emit: EventSink | None = None) -> str:
    """Run the pair estimators of one request (``check`` or ``estimate``) and return JSON.

    The request is the plan request of :func:`plan_request_json` plus ``mode``
    (``"check"`` by default, or ``"estimate"``), ``pairs``, ``max_duration_s`` and
    ``camera``. ``emit`` receives the :class:`EventProgress` events as JSON strings.
    The reply is ``{"ok": true, "mode", "artifact", "text", "html"?, "files", "command",
    "seconds"}`` or ``{"ok": false, "error"}``. ``files`` maps an export name
    (``frames.yaml``) to its text. A pair that cannot run reports itself in the
    artifact (skipped or inconclusive); it does not fail the call.
    """

    try:
        request = json.loads(request_json)
        mode: Literal["check", "estimate"] = (
            "estimate" if request.get("mode") == "estimate" else "check"
        )
        bag = locate_bag(request["bag_dir"])
        label = request.get("bag_label") or str(bag)
        tf_files = request.get("tf_files", [])
        tf_labels = request.get("tf_labels")
        labels = list(tf_labels) if tf_labels is not None else [str(item) for item in tf_files]
        if len(labels) != len(tf_files):
            msg = "tf_labels must have one entry per tf file"
            raise DatasetError(msg)
        vehicle_text = request.get("vehicle_frame")
        vehicle = vehicle_text.strip() if vehicle_text and vehicle_text.strip() else None
        frame_map = request.get("frame_map", [])
        topic_kind = request.get("topic_kind", [])
        topic_kinds = parse_topic_kinds(topic_kind)
        camera = (request.get("camera") or "").strip() or None
        progress: CheckProgress = EventProgress(emit) if emit is not None else CheckProgress()
        max_duration = float(request["max_duration_s"]) if request.get("max_duration_s") else None
        argv_args: dict[str, Any] = {
            "tf_labels": labels,
            "vehicle_frame": vehicle,
            "frame_map": frame_map,
            "topic_kind": topic_kind,
            "camera": camera,
            "pairs": request.get("pairs") or (),
            "max_duration_s": max_duration,
        }
        recorded = shlex.split(run_command(mode, label, **argv_args))
        command = run_command(mode, request.get("command_bag") or label, **argv_args)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="calibrex_browser_") as work:
            work_dir = Path(work)
            run = run_options_from_request(request, work_dir)
            if mode == "check":
                artifact = build_calibration_check(
                    bag,
                    tf_files=tf_files,
                    vehicle_frame=vehicle,
                    frame_overrides=parse_frame_map(frame_map),
                    topic_kinds=topic_kinds,
                    command=recorded,
                    run=run,
                    progress=progress,
                )
                artifact = _with_labels(artifact, label, labels, "candidate_sources")
                from calibrex.visualization.check_report import render_check_html

                reply: dict[str, Any] = {
                    "artifact": artifact.model_dump(mode="json", exclude_none=True),
                    "text": format_check_table(artifact),
                    "html": render_check_html(artifact),
                    "files": {},
                }
            else:
                from calibrex.check.estimate import build_bag_estimate, format_estimate_text

                output_dir = work_dir / "estimate"
                estimate = build_bag_estimate(
                    bag,
                    output_dir=output_dir,
                    run=run,
                    tf_files=tf_files,
                    vehicle_frame=vehicle,
                    frame_overrides=parse_frame_map(frame_map),
                    topic_kinds=topic_kinds,
                    command=recorded,
                    progress=progress,
                )
                estimate = _with_labels(estimate, label, labels, "prior_sources")
                reply = {
                    "artifact": estimate.model_dump(mode="json", exclude_none=True),
                    "text": format_estimate_text(estimate, Path("estimate")),
                    "files": {
                        item.name: item.read_text(encoding="utf-8")
                        for item in sorted(output_dir.iterdir())
                        if item.is_file()
                    },
                }
    except CalibrexError as error:
        return json.dumps({"ok": False, "error": str(error)})
    except Exception as error:  # corrupt files raise sqlite, yaml or pydantic errors
        return json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"})
    return json.dumps(
        _finite_json(
            {
                "ok": True,
                "mode": mode,
                "bag": str(bag),
                "command": command,
                "seconds": time.monotonic() - started,
                **reply,
            }
        )
    )


def _finite_json(value: Any) -> Any:
    """``value`` with every non-finite float replaced by ``None``.

    Python's ``json.dumps`` writes ``Infinity`` / ``NaN`` (the CLI artifact keeps them, for
    example an unobservable ``rate_ratio_std``), which ``JSON.parse`` in the page rejects.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_json(item) for item in value]
    return value


def _with_labels(artifact: Any, label: str, tf_labels: Sequence[str], sources_field: str) -> Any:
    """The artifact with the user's file names in place of the virtual mount paths."""

    remaining = iter(tf_labels)
    sources = [
        source.model_copy(
            update={"path": label if source.kind == "bag_tf_static" else next(remaining)}
        )
        for source in getattr(artifact, sources_field)
    ]
    provenance = artifact.provenance.model_copy(
        update={"notes": [*artifact.provenance.notes, BROWSER_NOTE]}
    )
    return artifact.model_copy(
        update={
            "bag": artifact.bag.model_copy(update={"path": label}),
            sources_field: sources,
            "provenance": provenance,
        }
    )
