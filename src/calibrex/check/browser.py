"""``calibrex check --plan`` for callers that hold files but no command line.

The browser check page (``docs/app/check.html``) mounts the dropped files in
Pyodide and calls :func:`plan_bag_for_browser`; it is the same planner as
``calibrex check --plan`` (:func:`calibrex.check.runner.build_calibration_check`),
with the user's file names in place of the virtual mount paths, so the artifact
and the printed command line read like a local run. Nothing here imports ROS,
OpenCV or Open3D.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from calibrex.check.runner import build_calibration_check, parse_frame_map
from calibrex.core.exceptions import CalibrexError, DatasetError

STORAGE_SUFFIXES = (".db3", ".mcap")
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
    vehicle = vehicle_frame.strip() if vehicle_frame and vehicle_frame.strip() else None
    artifact = build_calibration_check(
        bag_path,
        tf_files=tf_files,
        vehicle_frame=vehicle,
        frame_overrides=frame_overrides,
        command=check_command(
            label, tf_labels=labels, vehicle_frame=vehicle, frame_map=frame_map, plan=True
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
                outputs=True,
            )
        ),
        "plan_command": shlex.join(
            check_command(
                command_bag or label,
                tf_labels=labels,
                vehicle_frame=vehicle,
                frame_map=frame_map,
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
            bag_label=request.get("bag_label"),
            command_bag=request.get("command_bag"),
        )
    except CalibrexError as error:
        return json.dumps({"ok": False, "error": str(error)})
    except Exception as error:  # corrupt files raise sqlite, yaml or pydantic errors
        return json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"})
    return json.dumps({"ok": True, "bag": str(bag), **result})
