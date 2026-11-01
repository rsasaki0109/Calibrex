#!/usr/bin/env python3
"""Multi-window camera-LiDAR check: one verdict from several disjoint 60 s windows.

Development only (``docs/benchmarks/kitti360_camera_lidar_multiwindow.md``). A single
window's block-jackknife std measures how well that minute pins its optimum, not a bias
the whole minute shares. Several windows of one recording see different scenes, so their
spread estimates that bias. This tool combines the published per-window check results of
the KITTI-360 development batch with a rule fixed before the 42 exposed segments ran.

Per rotation axis, over the windows that judge it (as ``calibrex check`` judges it):

* at least ``max(2, ceil(K / 2))`` of the recording's ``K`` windows must judge it,
  otherwise it is unchecked;
* ``m`` = median of the windows' candidate errors (candidate minus estimate);
* ``s`` = 1.4826 * MAD of those errors (between-window spread), ``w`` = median of the
  windows' reported stds, ``sigma = max(s, w)``;
* the median's std is ``1.2533 * sigma / sqrt(n)``; the tolerance is
  ``max(3 * that, 0.5 deg)``, the check's ``sigma_k`` and rotation floor;
* pass when ``|m| <= tol``, fail when ``|m| > 2 tol``, otherwise warn.

The recording's verdict is the worst axis status, or inconclusive when none is judged.

    python tools/kitti360_camera_lidar_multiwindow.py PUBLISHED_DIR OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrex.check.estimators import camera_lidar_axis_estimates, transform_matrix  # noqa: E402
from calibrex.check.verdict import VerdictOptions, judge_pair  # noqa: E402
from calibrex.core.camera_lidar_edge import load_camera_lidar_edge  # noqa: E402

SCHEMA = "calibrex.kitti360_camera_lidar_multiwindow/v0.1"
AXES = ("roll", "pitch", "yaw")
TURN_AXIS = {"x": "roll", "y": "pitch", "z": "yaw"}
RUNS = {
    "baseline": None,
    **{f"inject_{a}+{d}": (TURN_AXIS[a], float(d)) for d in (3, 1) for a in "xyz"},
}
SIGMA_K = 3.0
FLOOR_DEG = 0.5
MAD_SCALE = 1.4826
MEDIAN_EFFICIENCY = 1.2533
TRIPLE = 3


def _tool(name: str) -> ModuleType:
    key = f"_multiwindow_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def window_axes(path: Path) -> dict[str, Any]:
    """One window's judged rotation axes: candidate error, std and error to the reference.

    ``reference`` is unknown here; the caller subtracts it. ``estimate`` is the window's
    estimated rotation (``None`` when the solver gave none).
    """

    artifact = load_camera_lidar_edge(path)
    assert artifact.start_transform is not None
    candidate = transform_matrix(
        artifact.start_transform.translation_m, artifact.start_transform.rotation_quat_xyzw
    )
    pair = judge_pair(camera_lidar_axis_estimates(artifact, candidate), VerdictOptions())
    judged = {
        axis.name: {
            "candidate_error_deg": axis.candidate_error,
            "std_deg": axis.estimate_std,
            "status": axis.status,
        }
        for axis in pair.axes
        if axis.name in AXES
    }
    estimate = None
    if artifact.transform is not None:
        estimate = transform_matrix(
            artifact.transform.translation_m, artifact.transform.rotation_quat_xyzw
        )[:3, :3]
    return {"judged": judged, "candidate": candidate[:3, :3], "estimate": estimate}


def combine_axis(
    values: Sequence[tuple[float, float]], windows: int, single_status: str | None = None
) -> dict[str, Any] | None:
    """The rule for one axis from ``(candidate_error, std)`` of the windows judging it.

    A recording of one window keeps the shipped check's own status (``single_status``).
    """

    if windows == 1:
        if not values:
            return None
        return {"candidate_error_deg": values[0][0], "windows_judged": 1, "status": single_status}
    needed = max(2, math.ceil(windows / 2))
    if len(values) < needed:
        return None
    errors = [value for value, _ in values]
    center = median(errors)
    spread = MAD_SCALE * median(abs(e - center) for e in errors)
    within = median(std for _, std in values)
    sigma = max(spread, within)
    sigma_median = MEDIAN_EFFICIENCY * sigma / math.sqrt(len(values))
    tolerance = max(SIGMA_K * sigma_median, FLOOR_DEG)
    size = abs(center)
    status = "pass" if size <= tolerance else ("fail" if size > 2 * tolerance else "warn")
    return {
        "candidate_error_deg": center,
        "windows_judged": len(values),
        "between_window_sd_deg": spread,
        "within_window_std_deg": within,
        "tolerance_deg": tolerance,
        "status": status,
    }


def combine(windows: Sequence[dict[str, Any]], reference: np.ndarray) -> dict[str, Any]:
    """One recording's verdict from its windows (same candidate in every window)."""

    scorer = _tool("score_kitti_camera_lidar_v2")
    axes: dict[str, Any] = {}
    for axis in AXES:
        values = [
            (w["judged"][axis]["candidate_error_deg"], w["judged"][axis]["std_deg"])
            for w in windows
            if axis in w["judged"]
        ]
        single = windows[0]["judged"].get(axis, {}).get("status") if len(windows) == 1 else None
        result = combine_axis(values, len(windows), single)
        if result is not None:
            index = AXES.index(axis)
            result["error_to_reference_deg"] = median(
                float(scorer.axis_errors_deg(w["estimate"], reference)[index])
                for w in windows
                if axis in w["judged"]
            )
        axes[axis] = result
    statuses = [item["status"] for item in axes.values() if item is not None]
    verdict = (
        "inconclusive"
        if not statuses
        else "fail"
        if "fail" in statuses
        else "warn"
        if "warn" in statuses
        else "pass"
    )
    return {"windows": len(windows), "verdict": verdict, "axes": axes}


def score_recording(per_run: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Baseline verdict, judged-axis errors and the six turns, v2's trial definitions."""

    reference = per_run["baseline"][0]["candidate"]
    runs = {run: combine(windows, reference) for run, windows in per_run.items()}
    base = runs["baseline"]
    trials = {}
    for run, injection in RUNS.items():
        if injection is None:
            continue
        axis, degrees = injection
        turned = runs[run]["axes"][axis]
        status = None if turned is None else turned["status"]
        trials[run] = {
            "degrees": degrees,
            "counted": base["axes"][axis] is not None,
            "flagged": status in ("warn", "fail"),
            "judged_right": status == "pass",
            "reading_deg": None if turned is None else turned["candidate_error_deg"],
            "judged_errors_deg": {
                name: item["error_to_reference_deg"]
                for name, item in runs[run]["axes"].items()
                if item is not None
            },
        }
    return {"baseline": base, "trials": trials}


def single_window(windows: dict[str, dict[str, Any]], reference: np.ndarray) -> dict[str, Any]:
    """The same measures for one window judged alone (the shipped check), for comparison."""

    return score_recording({run: [window] for run, window in windows.items()}) | {
        "reference_check": bool(np.allclose(windows["baseline"]["candidate"], reference))
    }


def summarize(recordings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flagged = [r for r, s in recordings.items() if s["baseline"]["verdict"] in ("warn", "fail")]
    base_errors = [
        abs(item["error_to_reference_deg"])
        for s in recordings.values()
        for item in s["baseline"]["axes"].values()
        if item is not None
    ]
    turned_errors = [
        abs(value)
        for s in recordings.values()
        for trial in s["trials"].values()
        for value in trial["judged_errors_deg"].values()
    ]
    judged = sum(
        item is not None for s in recordings.values() for item in s["baseline"]["axes"].values()
    )
    detection = {}
    for degrees in (3.0, 1.0):
        counted = [
            t
            for s in recordings.values()
            for t in s["trials"].values()
            if t["degrees"] == degrees and t["counted"]
        ]
        detection[f"{degrees:g}deg"] = {
            "counted": len(counted),
            "missed": sum(not t["flagged"] for t in counted),
            "missed_fraction": (
                sum(not t["flagged"] for t in counted) / len(counted) if counted else None
            ),
            "judged_right": sum(t["judged_right"] for t in counted),
        }
    return {
        "recordings": len(recordings),
        "vendor_flagged": flagged,
        "max_baseline_judged_error_deg": max(base_errors, default=None),
        "max_turned_judged_error_deg": max(turned_errors, default=None),
        "judged_axes": judged,
        "judged_fraction": judged / (3 * len(recordings)) if recordings else None,
        "detection": detection,
    }


CRITERIA = {
    "no_vendor_flag": "no sequence- or triple-level recording flags the vendor calibration",
    "baseline_error": "every judged triple-level baseline axis within 1.0 deg",
    "turned_error": "every judged triple-level turned-run axis within 1.0 deg",
    "turn_3deg": "at most 0.15 of counted triple-level 3 deg turns missed, none judged right",
    "turn_1deg": "at most 0.25 of counted triple-level 1 deg turns missed",
    "coverage": "at least 0.5 of triple-level baseline axes judged",
}


def criteria(sequences: dict[str, Any], triples: dict[str, Any]) -> dict[str, bool | None]:
    seq, tri = summarize(sequences), summarize(triples)

    def at_most(value: float | None, bound: float) -> bool | None:
        return None if value is None else value <= bound + 1e-9

    return {
        "no_vendor_flag": not seq["vendor_flagged"] and not tri["vendor_flagged"],
        "baseline_error": at_most(tri["max_baseline_judged_error_deg"], 1.0),
        "turned_error": at_most(tri["max_turned_judged_error_deg"], 1.0),
        "turn_3deg": None
        if tri["detection"]["3deg"]["missed_fraction"] is None
        else at_most(tri["detection"]["3deg"]["missed_fraction"], 0.15)
        and tri["detection"]["3deg"]["judged_right"] == 0,
        "turn_1deg": at_most(tri["detection"]["1deg"]["missed_fraction"], 0.25),
        "coverage": None
        if tri["judged_fraction"] is None
        else tri["judged_fraction"] >= 0.5 - 1e-9,
    }


def load_windows(published: Path, group: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Each segment's latest passing attempt, as per-run window records."""

    batch = _tool("kitti360_camera_lidar_v3_batch")
    windows: dict[str, dict[str, dict[str, Any]]] = {}
    for segment in batch.segments(group):
        attempt, report = batch._latest(published, segment)
        if report is None or report["status"] != "PASS_DEVELOPMENT":
            continue
        root = published / segment / f"attempt_{attempt}" / "comparison" / segment / "current"
        windows[segment] = {run: window_axes(root / f"{run}.json") for run in RUNS}
    return windows


def evaluate(windows: dict[str, dict[str, dict[str, Any]]], segments: dict[str, Any]) -> dict:
    by_sequence: dict[str, list[str]] = {}
    for segment in sorted(
        windows, key=lambda s: (segments[s]["sequence"], segments[s]["start_frame"])
    ):
        by_sequence.setdefault(segments[segment]["sequence"], []).append(segment)
    sequences = {
        sequence: score_recording({run: [windows[s][run] for s in ids] for run in RUNS})
        for sequence, ids in by_sequence.items()
    }
    triples = {}
    for ids in by_sequence.values():
        for k in range(len(ids) // TRIPLE):
            chosen = ids[k * TRIPLE : (k + 1) * TRIPLE]
            triples["+".join(chosen)] = score_recording(
                {run: [windows[s][run] for s in chosen] for run in RUNS}
            )
    reference = {s: windows[s]["baseline"]["candidate"] for s in windows}
    singles = {s: single_window(windows[s], reference[s]) for s in windows}
    return {
        "single_window": summarize(singles),
        "sequence_level": summarize(sequences),
        "triple_level": summarize(triples),
        "criteria": criteria(sequences, triples),
        "recordings": {
            "sequences": {k: v["baseline"] for k, v in sequences.items()},
            "triples": {k: v["baseline"] for k, v in triples.items()},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("published", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--segments", choices=("all", "spent", "exposed"), default="exposed")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse overwrite")
    batch = _tool("kitti360_camera_lidar_v3_batch")
    windows = load_windows(args.published, args.segments)
    result = {
        "schema_version": SCHEMA,
        "analysis": "development; KITTI-360 is not unread data, no claim",
        "segments": args.segments,
        "windows_scored": len(windows),
        "windows_expected": len(batch.segments(args.segments)),
        **evaluate(windows, batch.segments("all")),
        "criteria_text": CRITERIA,
        "provenance": {
            "generator": "tools/kitti360_camera_lidar_multiwindow.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_commit": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    result["carried_forward"] = result["windows_scored"] == result["windows_expected"] and all(
        value is True for value in result["criteria"].values()
    )
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("criteria", "carried_forward")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
