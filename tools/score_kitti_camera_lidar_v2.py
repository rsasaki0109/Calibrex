#!/usr/bin/env python3
"""Score KITTI camera-lidar check runs for the v2 audit, on drives of any KITTI day.

An evaluation harness, not part of the Calibrex package. It implements exactly the
metrics, gates and verdict rules of
``docs/benchmarks/kitti_camera_lidar_v2_preregistration.yaml``; its thresholds are module
constants and cannot be changed from the command line::

    python tools/score_kitti_camera_lidar_v2.py RUNS_DIR CALIB_ROOT OUT_DIR \\
        --preregistration docs/benchmarks/kitti_camera_lidar_v2_preregistration.yaml
    python tools/score_kitti_camera_lidar_v2.py RUNS_DIR CALIB_ROOT OUT_DIR \\
        --drive 2011_09_26_0005 --drive 2011_09_30_0018 ...   # development runs

A drive is named ``DATE_DRIVE``. ``RUNS_DIR/<drive>`` holds what
``tools/kitti_camera_lidar_v2_evaluate.py`` writes for one drive: ``download.json`` (the
fetch of the drive's first ``MAX_FRAMES`` frames, every member's CRC-32 checked),
``manifest.json``, the ``calibrex check`` artifacts ``baseline.json`` and
``inject_<axis><+deg>.json``, and their evidence directories. ``CALIB_ROOT/<DATE>`` holds
that day's KITTI calibration files. With ``--preregistration`` the drives are its
evaluation drives, in its order; every day's calibration archive in
``RUNS_DIR/calibration_download.json`` must match its pinned SHA-256 (or nothing is
scored), and a drive whose fetched archive differs from its pinned size and ETag fails.

The reference is that day's KITTI calibration: the baseline run's candidate ``T_cam2_velo``,
which ``calibrex convert kitti-raw`` composes from the calibration files; its rotation is
checked against ``R_rect_00 R`` of the day's ``calib_cam_to_cam.txt`` and
``calib_velo_to_cam.txt``. An axis
(roll, pitch, yaw) is a component (deg) of ``rotvec(R_est R_ref^T)``: a small rotation about
the x, y or z axis of the camera's optical frame. It is *judged* on a run when ``calibrex
check`` judges it there: the pair lists it in ``axes`` (estimated, reported std at most
0.5 deg, its 2 deg known-bad control detected on held-out frames, not at the search bound).

* A *drive* is failed when its fetch failed or is incomplete (fewer frames of a stream
  than ``min(MAX_FRAMES, the archive's)``), its archive did not match the pre-registration,
  or its conversion or one of its checks did not finish (non-zero exit code, missing
  artifact or pair record). It is not usable when the baseline estimator refused
  (``solver_status`` ``insufficient_samples``: fewer usable frames than it needs). Every
  other drive is scored.
* A *turned trial* is one injection ``AXIS:DEG`` on a scored drive: the camera turned by
  ``DEG`` about its own ``AXIS`` (``T_camera_lidar' = Rot(AXIS, DEG) T_camera_lidar``), so a
  check that sees it reads ``+DEG`` on roll, pitch or yaw. A trial counts only when that
  axis is judged on the drive's baseline run. It is *flagged* when the turned run judges the
  axis with status ``warn`` or ``fail``, *judged right* when it judges the axis ``pass``
  (the check says the turned axis is right), and otherwise *unjudged* (the turned run does
  not judge the axis).
* ``no-failed-drives``.
* ``vendor-calibration-not-flagged``: no scored drive's baseline verdict is warn or fail.
* ``judged-axis-accuracy``: every judged axis of every scored baseline run within
  ``AXIS_ERROR_MAX_DEG`` of KITTI's calibration.
* ``judged-axis-coverage`` (coverage): at least ``JUDGED_FRACTION_MIN`` of the scored
  baseline runs' axes judged.
* ``turned-judged-axis-accuracy``: every axis judged in every turned run of a scored drive
  (counted or not) within ``TURNED_AXIS_ERROR_MAX_DEG`` of KITTI's calibration.
* ``turn-3deg-flagged``: at most ``MISSED_3DEG_MAX`` of the counted 3 deg trials not
  flagged (a fraction, compared exactly as the audit compares its failure rate).
* ``turn-3deg-never-judged-right``: no counted 3 deg trial judges the turned axis ``pass``.
* ``turn-1deg-flagged``: at most ``MISSED_1DEG_MAX`` of the counted 1 deg trials not flagged.
* ``scored-drives-coverage`` (coverage): at least ``SCORED_DRIVES_MIN`` drives scored.

A failed non-coverage requirement refutes the claim; all passing supports it; anything
else is inconclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

AXES = ("roll", "pitch", "yaw")
TURN_AXIS = {"x": "roll", "y": "pitch", "z": "yaw"}
INJECTIONS = ("x:3", "y:3", "z:3", "x:1", "y:1", "z:1")
VELODYNE_TIME = "azimuth"  # calibrex convert kitti-raw --velodyne-time
MAX_FRAMES = 600  # the frames fetched from the head of each drive
FETCHED_STREAMS = ("image_02", "velodyne_points", "oxts")
AXIS_ERROR_MAX_DEG = 0.75
TURNED_AXIS_ERROR_MAX_DEG = 1.5
JUDGED_FRACTION_MIN = 0.5
MISSED_3DEG_MAX = 0.1
MISSED_1DEG_MAX = 0.25
SCORED_DRIVES_MIN = 15
# The estimator options every artifact must carry (the defaults at the frozen commit).
RECIPE = {
    "edge_background_sigma_px": 20.0,
    "repeated_view_change": 0.05,
    "rotation_control_deg": 2.0,
    "observable_rotation_std_deg": 0.5,
    "min_control_t": 2.5,
    "start_scan_deg": 8.0,
    "start_scan_min_observed": 2,
    "estimate_translation": False,
}
TOLERANCE = 1e-9  # a value within 1e-9 of its threshold meets it (rotation round-off)
INJECTION_CHECK_DEG = 1e-6  # the turned candidate must be the reference turned exactly
REFERENCE_CHECK_DEG = 1e-6  # the reference must be KITTI's R_rect_00 R
COVERAGE = ("judged-axis-coverage", "scored-drives-coverage")
REQUIREMENTS = (
    "no-failed-drives",
    "vendor-calibration-not-flagged",
    "judged-axis-accuracy",
    "judged-axis-coverage",
    "turned-judged-axis-accuracy",
    "turn-3deg-flagged",
    "turn-3deg-never-judged-right",
    "turn-1deg-flagged",
    "scored-drives-coverage",
)


@dataclass(frozen=True)
class RunInput:
    """What the scorer needs from one ``calibrex check`` run of the camera-lidar pair."""

    error: str | None = None
    verdict: str | None = None
    solver_status: str = "converged"
    candidate: np.ndarray | None = None  # R_camera_lidar the check started from
    estimate: np.ndarray | None = None  # R_camera_lidar it estimated
    judged: dict[str, str] = field(default_factory=dict)  # axis -> pass | warn | fail


@dataclass(frozen=True)
class DriveInput:
    """One drive: its fetch, the baseline run and the turned runs."""

    error: str | None = None
    baseline: RunInput = RunInput()
    turned: dict[str, RunInput] = field(default_factory=dict)  # "x:3" -> run
    fetch: dict[str, Any] = field(default_factory=dict)  # archive size, ETag, frames


@dataclass
class DriveScore:
    """One scored (or not) drive."""

    state: str  # scored | inconclusive | failed
    reason: str = ""
    verdict: str | None = None
    axes: dict[str, dict[str, Any]] = field(default_factory=dict)
    trials: dict[str, dict[str, Any]] = field(default_factory=dict)
    fetch: dict[str, Any] = field(default_factory=dict)


def axis_errors_deg(rotation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``rotvec(R R_ref^T)`` in degrees: roll, pitch, yaw about the camera's axes."""

    return np.degrees(Rotation.from_matrix(rotation @ reference.T).as_rotvec())


def turn(spec: str) -> tuple[str, float]:
    """``"y:3"`` -> ``("pitch", 3.0)``."""

    axis, degrees = spec.split(":")
    return TURN_AXIS[axis], float(degrees)


def score_drive(drive: DriveInput) -> DriveScore:
    """Apply the failure and usable-drive rules and score the baseline and the trials."""

    score = _score_drive(drive)
    score.fetch = dict(drive.fetch)
    return score


def _score_drive(drive: DriveInput) -> DriveScore:
    if drive.error is not None:
        return DriveScore("failed", drive.error)
    runs = {"baseline": drive.baseline, **drive.turned}
    broken = {name: run.error for name, run in runs.items() if run.error is not None}
    if broken:
        return DriveScore("failed", "; ".join(f"{name}: {why}" for name, why in broken.items()))
    missing = [spec for spec in INJECTIONS if spec not in drive.turned]
    if missing:
        return DriveScore("failed", f"turned runs missing: {', '.join(missing)}")
    base = drive.baseline
    if base.solver_status == "insufficient_samples":
        return DriveScore(
            "inconclusive",
            "the baseline estimator refused: too few usable frames",
            verdict=base.verdict,
        )
    reference = base.candidate
    if reference is None or base.estimate is None:
        return DriveScore("failed", f"baseline solver_status {base.solver_status}, no estimate")
    errors = axis_errors_deg(base.estimate, reference)
    axes = {
        name: {
            "judged": name in base.judged,
            "status": base.judged.get(name),
            "error_deg": float(e),
        }
        for name, e in zip(AXES, errors, strict=True)
    }
    trials: dict[str, dict[str, Any]] = {}
    for spec in INJECTIONS:
        run = drive.turned[spec]
        name, degrees = turn(spec)
        assert run.candidate is not None
        expected = Rotation.from_rotvec(np.radians(degrees) * np.eye(3)[AXES.index(name)])
        applied = axis_errors_deg(run.candidate, expected.as_matrix() @ reference)
        if np.max(np.abs(applied)) > INJECTION_CHECK_DEG:
            raise ValueError(f"{spec}: the turned candidate is not KITTI's turned by {spec}")
        reading = None
        if run.estimate is not None and name in run.judged:
            # candidate minus estimate on the turned axis, the reading of calibrex check
            reading = float(axis_errors_deg(run.candidate, run.estimate)[AXES.index(name)])
        judged_errors = {}
        if run.estimate is not None:
            errors_to_reference = axis_errors_deg(run.estimate, reference)
            judged_errors = {
                axis: float(errors_to_reference[AXES.index(axis)]) for axis in run.judged
            }
        status = run.judged.get(name)
        flagged = status in ("warn", "fail")
        others = sorted(
            other for other, value in run.judged.items() if other != name and value != "pass"
        )
        trials[spec] = {
            "counted": axes[name]["judged"],
            "flagged": flagged,
            "judged_right": status == "pass",
            "outcome": "flagged"
            if flagged
            else ("judged right" if status == "pass" else "unjudged"),
            "verdict": run.verdict,
            "turned_axis_status": status,
            "reading_deg": reading,
            "other_axes_flagged": others,
            "judged_errors_deg": judged_errors,
        }
    return DriveScore("scored", verdict=base.verdict, axes=axes, trials=trials)


def _requirement(failed: bool | None, observed: float | None, threshold: float) -> dict[str, Any]:
    status = "not_evaluated" if failed is None else ("fail" if failed else "pass")
    return {"status": status, "observed": observed, "threshold": threshold}


def _missed_gate(trials: list[dict[str, Any]], maximum: float) -> dict[str, Any]:
    if not trials:
        return _requirement(None, None, maximum)
    missed = sum(1 for trial in trials if not trial["flagged"]) / len(trials)
    return _requirement(missed > maximum, missed, maximum)


def score_drives(drives: dict[str, DriveInput], order: list[str]) -> dict[str, Any]:
    """Score every drive of ``order``; apply the requirements and the verdict."""

    scores = {name: score_drive(drives[name]) for name in order}
    scored = {name: score for name, score in scores.items() if score.state == "scored"}
    failed = [name for name, score in scores.items() if score.state == "failed"]
    judged = [
        (name, axis, item["error_deg"])
        for name, score in scored.items()
        for axis, item in score.axes.items()
        if item["judged"]
    ]
    flagged_baselines = [
        name for name, score in scored.items() if score.verdict in ("warn", "fail")
    ]
    counted: dict[float, list[dict[str, Any]]] = {3.0: [], 1.0: []}
    for score in scored.values():
        for spec, trial in score.trials.items():
            if trial["counted"]:
                counted[turn(spec)[1]].append(trial)
    axis_count = 3 * len(scored)
    judged_fraction = len(judged) / axis_count if axis_count else None
    worst = max((abs(error) for _, _, error in judged), default=None)
    judged_right_3 = [trial for trial in counted[3.0] if trial["judged_right"]]
    turned_errors = [
        abs(error)
        for score in scored.values()
        for trial in score.trials.values()
        for error in trial["judged_errors_deg"].values()
    ]
    worst_turned = max(turned_errors, default=None)
    requirements = {
        "no-failed-drives": _requirement(bool(failed), float(len(failed)), 0.0),
        "vendor-calibration-not-flagged": _requirement(
            bool(flagged_baselines) if scored else None, float(len(flagged_baselines)), 0.0
        ),
        "judged-axis-accuracy": _requirement(
            None if worst is None else worst > AXIS_ERROR_MAX_DEG + TOLERANCE,
            worst,
            AXIS_ERROR_MAX_DEG,
        ),
        "judged-axis-coverage": _requirement(
            None if judged_fraction is None else judged_fraction < JUDGED_FRACTION_MIN - TOLERANCE,
            judged_fraction,
            JUDGED_FRACTION_MIN,
        ),
        "turned-judged-axis-accuracy": _requirement(
            None if worst_turned is None else worst_turned > TURNED_AXIS_ERROR_MAX_DEG + TOLERANCE,
            worst_turned,
            TURNED_AXIS_ERROR_MAX_DEG,
        ),
        "turn-3deg-flagged": _missed_gate(counted[3.0], MISSED_3DEG_MAX),
        "turn-3deg-never-judged-right": _requirement(
            bool(judged_right_3) if counted[3.0] else None, float(len(judged_right_3)), 0.0
        ),
        "turn-1deg-flagged": _missed_gate(counted[1.0], MISSED_1DEG_MAX),
        "scored-drives-coverage": _requirement(
            len(scored) < SCORED_DRIVES_MIN, float(len(scored)), float(SCORED_DRIVES_MIN)
        ),
    }
    statuses = {name: item["status"] for name, item in requirements.items()}
    if any(status == "fail" for name, status in statuses.items() if name not in COVERAGE):
        verdict = "refuted"
    elif all(status == "pass" for status in statuses.values()):
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {
        "drives": list(order),
        "scores": {name: asdict(score) for name, score in scores.items()},
        "scored_drives": sorted(scored),
        "judged_axes": len(judged),
        "scored_axes": axis_count,
        "judged_fraction": judged_fraction,
        "trials": {
            f"{degrees:g}deg": {
                "counted": len(items),
                "flagged": sum(1 for item in items if item["flagged"]),
                "judged_right": sum(1 for item in items if item["judged_right"]),
            }
            for degrees, items in counted.items()
        },
        "requirements": requirements,
        "verdict": verdict,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quat_matrix(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(Rotation.from_quat(record["rotation_quat_xyzw"]).as_matrix())


def load_run(drive_dir: Path, name: str, manifest: dict[str, Any]) -> RunInput:
    """Read one check run: its exit code, pair verdict, judged axes and evidence."""

    entry = manifest.get("runs", {}).get(name)
    artifact = drive_dir / f"{name}.json"
    if entry is None:
        return RunInput(error="not run")
    if entry.get("exit_code") != 0:
        return RunInput(error=f"exit code {entry.get('exit_code')}")
    if not artifact.is_file():
        return RunInput(error="artifact missing")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except ValueError as exc:  # a truncated artifact is a failed run
        return RunInput(error=f"artifact unreadable: {exc}")
    pairs = [item for item in payload.get("pairs", []) if item.get("pair") == "camera-lidar"]
    if len(pairs) != 1 or pairs[0].get("status") in ("planned", "skipped"):
        return RunInput(error="no camera-lidar pair record")
    record = pairs[0]
    evidence_path = drive_dir / str(record.get("evidence_artifact"))
    if not evidence_path.is_file():
        return RunInput(error="evidence missing")
    try:
        evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return RunInput(error=f"evidence unreadable: {exc}")
    for key, value in RECIPE.items():
        if evidence["options"].get(key) != value:
            raise ValueError(f"{evidence_path}: {key} is not the pre-registered {value}")
    transform = evidence.get("transform")
    return RunInput(
        verdict=record.get("verdict"),
        solver_status=str(evidence.get("solver_status")),
        candidate=_quat_matrix(record["candidate_transform"]),
        estimate=None if transform is None else _quat_matrix(transform),
        judged={
            item["name"]: item["status"] for item in record.get("axes", []) if item["name"] in AXES
        },
    )


def kitti_rotation(calib_dir: Path) -> np.ndarray:
    """``R_rect_00 R`` of KITTI's calibration: the rotation of ``T_cam2_velo``."""

    from calibrex.data.kitti import read_calibration_file

    cam = read_calibration_file(calib_dir / "calib_cam_to_cam.txt")
    velo = read_calibration_file(calib_dir / "calib_velo_to_cam.txt")
    rect = np.asarray(cam["R_rect_00"], dtype=np.float64).reshape(3, 3)
    return rect @ np.asarray(velo["R"], dtype=np.float64).reshape(3, 3)


def split_drive(name: str) -> tuple[str, str]:
    """``2011_09_30_0028`` -> ``("2011_09_30", "0028")``."""

    date, _, drive = name.rpartition("_")
    if len(date) != 10 or len(drive) != 4:
        raise ValueError(f"a drive is named DATE_DRIVE, e.g. 2011_09_30_0028: {name}")
    return date, drive


def load_drive(
    drive_dir: Path, reference: np.ndarray, pin: dict[str, Any] | None = None
) -> DriveInput:
    """Read a drive's fetch record and runs; the baseline candidate must be KITTI's.

    ``pin`` is the drive's pinned archive (``size_bytes``, ``etag``) when there is one.
    """

    manifest_path = drive_dir / "manifest.json"
    download_path = drive_dir / "download.json"
    if not download_path.is_file():
        return DriveInput(error="fetch record missing (the drive was not fetched)")
    download = json.loads(download_path.read_text(encoding="utf-8"))
    if not download.get("ok"):
        return DriveInput(error=f"fetch: {download.get('reason')}")
    if download.get("max_frames") != MAX_FRAMES or download.get("crc_verified") is not True:
        raise ValueError(f"{drive_dir}: not fetched as pre-registered ({MAX_FRAMES} frames)")
    fetch = {
        key: download.get(key)
        for key in ("size_bytes", "etag", "frames", "archive_frames", "members_sha256")
    }
    if pin is not None and (download.get("size_bytes"), download.get("etag")) != (
        pin["size_bytes"],
        pin["etag"],
    ):
        return DriveInput(error="fetch: the archive differs from its pinned size and ETag")
    frames, archive = download.get("frames", {}), download.get("archive_frames", {})
    short = [
        stream
        for stream in FETCHED_STREAMS
        if frames.get(stream) != min(MAX_FRAMES, archive.get(stream, 0)) or not frames[stream]
    ]
    if short:
        return DriveInput(error=f"fetch incomplete: {', '.join(short)}", fetch=fetch)
    if not manifest_path.is_file():
        return DriveInput(error="manifest missing (the drive was not run)", fetch=fetch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("convert", {}).get("exit_code") != 0:
        return DriveInput(error="conversion failed", fetch=fetch)
    conversion_path = drive_dir / "conversion.json"
    if not conversion_path.is_file():
        return DriveInput(error="conversion record missing", fetch=fetch)
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    if conversion.get("velodyne_time") != VELODYNE_TIME:
        raise ValueError(f"{drive_dir}: velodyne_time is not the pre-registered {VELODYNE_TIME}")
    baseline = load_run(drive_dir, "baseline", manifest)
    if baseline.candidate is not None:
        gap = np.max(np.abs(axis_errors_deg(baseline.candidate, reference)))
        if gap > REFERENCE_CHECK_DEG:
            raise ValueError(f"{drive_dir}: the baseline candidate is {gap:.3g} deg off KITTI's")
    turned = {}
    for spec in INJECTIONS:
        axis, degrees = spec.split(":")
        turned[spec] = load_run(drive_dir, f"inject_{axis}{float(degrees):+g}", manifest)
    return DriveInput(baseline=baseline, turned=turned, fetch=fetch)


def check_calibrations(runs_dir: Path, dates: list[str], pinned: dict[str, str]) -> None:
    """Every day's calibration archive must have been downloaded and match its pin."""

    path = runs_dir / "calibration_download.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for date in dates:
        entry = record.get(date, {})
        if not entry.get("ok") or entry.get("sha256") != pinned.get(date):
            raise ValueError(f"{path}: {date}'s calibration archive does not match its pin")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("calib_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preregistration", type=Path, default=None)
    source.add_argument("--drive", action="append", default=None, help="DATE_DRIVE")
    args = parser.parse_args(argv)
    from calibrex import __version__

    pins: dict[str, Any] = {}
    if args.preregistration is not None:
        plan = yaml.safe_load(args.preregistration.read_text(encoding="utf-8"))
        names = [str(name) for name in plan["recordings"]["evaluation"]]
        pins = {str(key): value for key, value in plan["recordings"]["archives"].items()}
        dates = sorted({split_drive(name)[0] for name in names})
        calibration_pins = plan["recordings"]["calibration_archive_sha256"]
        check_calibrations(args.runs_dir, dates, {str(k): v for k, v in calibration_pins.items()})
        if set(pins) != set(names):
            raise ValueError("the pre-registration must pin exactly its evaluation drives")
    else:
        names = list(args.drive)
        dates = sorted({split_drive(name)[0] for name in names})
    references = {date: kitti_rotation(args.calib_root / date) for date in dates}
    drives = {
        name: load_drive(args.runs_dir / name, references[split_drive(name)[0]], pins.get(name))
        for name in names
    }
    result = score_drives(drives, names)
    hashes = {
        f"{name}/{path.name}": _sha256(path)
        for name in names
        for path in sorted((args.runs_dir / name).glob("*.json"))
    }
    calibration_record = args.runs_dir / "calibration_download.json"
    result["provenance"] = {
        "generator": "tools/score_kitti_camera_lidar_v2.py",
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "calibrex_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": None
        if args.preregistration is None
        else {"path": str(args.preregistration), "sha256": _sha256(args.preregistration)},
        "drives": names,
        "artifact_sha256": hashes,
        "calibration_download": json.loads(calibration_record.read_text(encoding="utf-8"))
        if calibration_record.is_file()
        else None,
        "calibration_sha256": {
            date: {
                path.name: _sha256(path)
                for path in sorted((args.calib_root / date).glob("calib_*.txt"))
            }
            for date in dates
        },
        "thresholds": {
            "axis_error_max_deg": AXIS_ERROR_MAX_DEG,
            "turned_axis_error_max_deg": TURNED_AXIS_ERROR_MAX_DEG,
            "judged_fraction_min": JUDGED_FRACTION_MIN,
            "missed_3deg_max": MISSED_3DEG_MAX,
            "missed_1deg_max": MISSED_1DEG_MAX,
            "scored_drives_min": SCORED_DRIVES_MIN,
            "injections": list(INJECTIONS),
            "velodyne_time": VELODYNE_TIME,
            "max_frames": MAX_FRAMES,
            "recipe": RECIPE,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for name, score in result["scores"].items():
        if score["state"] != "scored":
            print(f"{name}: {score['state']} ({score['reason']})")
            continue
        axes = " ".join(
            f"{axis} {item['error_deg']:+.2f}{'' if item['judged'] else '(u)'}"
            for axis, item in score["axes"].items()
        )
        trials = " ".join(
            f"{spec}:{trial['outcome'][0]}{'' if trial['counted'] else '*'}"
            for spec, trial in score["trials"].items()
        )
        print(f"{name}: {score['verdict']}; {axes}; {trials}")
    for name, item in result["requirements"].items():
        observed, threshold = item["observed"], item["threshold"]
        print(f"  {name}: {item['status']} (observed {observed}, threshold {threshold})")
    print("verdict:", result["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
