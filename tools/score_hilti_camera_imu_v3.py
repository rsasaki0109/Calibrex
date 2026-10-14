#!/usr/bin/env python3
"""Score camera-IMU rotation artifacts per judged axis for the Hilti camera-IMU v3 audit.

An evaluation harness, not part of the Calibrex package. It implements exactly the
metrics, gates and verdict rules of ``docs/benchmarks/hilti_camera_imu_v3_preregistration.yaml``;
its thresholds are module constants and cannot be changed from the command line::

    python tools/score_hilti_camera_imu_v3.py RUNS_DIR CAMCHAIN OUT_DIR \\
        --recording exp05 --recording exp06 ...

``RUNS_DIR`` holds, per gated camera and recording, the artifact ``{camera}_{recording}.yaml``
of v1's ``calibrex camera-imu rotation`` command and the command's log
``{camera}_{recording}.log``. ``CAMCHAIN`` is Hilti's Kalibr camchain of cam0 and cam1, the
reference.

v3 judges the method by its own per-axis verdicts. A rotation axis (roll, pitch or yaw:
a small rotation about the camera's x, y or z axis that takes Kalibr's ``R_cam_imu`` to
the estimate) is *judged* when the artifact reports it ``estimated`` (reported std at
most 0.3 deg) and its 1 deg known-bad control is detected on held-out windows: the rule
``calibrex check`` applies before it judges an axis. The clock offset is judged by the
same rule with its 10 ms control.

* A *unit* is one camera on one recording. A missing artifact is a failed unit, except
  when the log shows the estimator's refusal for too few tracking windows (fewer than
  3): such a unit has fewer than 8 windows and is not usable, like any unit with fewer
  than 8 windows or more than 25 percent failed frame pairs. A usable unit whose solver
  did not converge or whose policy is ``fail`` is failed; every other usable unit is
  scored.
* ``judged-axis-accuracy``: every judged axis of every scored unit within 1.0 deg of
  Kalibr.
* ``judged-time-offset``: every judged clock offset within 0.75 ms of Kalibr's.
* ``judged-axis-coverage`` (coverage): at least half of the scored units' axes judged.
* ``no-failed-units``.
* ``judged-axis-consistency``: per camera and axis, the errors of the recordings that
  judge it span at most 1.0 deg.
* ``judged-axis-relative``: per recording scored for both cameras and per axis judged
  for both, the matching component of the relative cam0-to-cam1 rotation error to
  Kalibr (``rotvec((R0 R1^T)(R0ref R1ref^T)^T)``, cam0's axes) is at most 0.75 deg.
* ``scored-recordings-coverage`` (coverage): at least 3 recordings scored for both
  cameras.

A failed non-coverage requirement refutes the claim; all passing supports it; anything
else is inconclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

GATED_CAMERAS = ("cam0", "cam1")
AXES = ("roll", "pitch", "yaw")
AXIS_ERROR_MAX_DEG = 1.0
TIME_ERROR_MAX_MS = 0.75
JUDGED_FRACTION_MIN = 0.5
CONSISTENCY_MAX_DEG = 1.0
RELATIVE_MAX_DEG = 0.75
USABLE_WINDOWS_MIN = 8
USABLE_FAILED_PAIR_FRACTION_MAX = 0.25
SCORED_RECORDINGS_MIN = 3
RECIPE = {
    "min_window_rotation_deg": 1.0,
    "observable_rotation_std_deg": 0.3,
    "rotation_control_deg": 1.0,
    "frame_stride": 4,
}
TOLERANCE = 1e-9  # a value within 1e-9 of its threshold meets it (rotation round-off)
REFUSAL = re.compile(r"camera tracking produced (\d+) windows; at least 3 are needed")
COVERAGE = ("judged-axis-coverage", "scored-recordings-coverage")
REQUIREMENTS = (
    "judged-axis-accuracy",
    "judged-time-offset",
    "judged-axis-coverage",
    "no-failed-units",
    "judged-axis-consistency",
    "judged-axis-relative",
    "scored-recordings-coverage",
)


@dataclass(frozen=True)
class AxisInput:
    """One quantity's verdict evidence: status, reported std, and its control."""

    status: str = "estimated"
    std: float = 0.0
    control_detected: bool | None = True

    @property
    def judged(self) -> bool:
        return self.status == "estimated" and self.control_detected is True


@dataclass(frozen=True)
class UnitInput:
    """What the scorer needs from one rotation artifact and its Kalibr reference."""

    error: str | None = None
    refused_windows: int | None = None
    rotation: np.ndarray | None = None
    reference_rotation: np.ndarray | None = None
    time_offset_s: float | None = None
    reference_shift_s: float | None = None
    windows_used: int = 0
    pairs: int = 0
    failed_pairs: int = 0
    solver_status: str = "converged"
    policy_status: str = "pass"
    axes: tuple[AxisInput, AxisInput, AxisInput] = (AxisInput(),) * 3
    time: AxisInput = AxisInput()


@dataclass
class UnitScore:
    """One scored (or not) unit."""

    state: str  # scored | inconclusive | failed
    reason: str = ""
    windows_used: int = 0
    failed_pair_fraction: float | None = None
    axes: dict[str, dict[str, Any]] = field(default_factory=dict)
    time: dict[str, Any] = field(default_factory=dict)


def axis_errors_deg(rotation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``rotvec(R_est R_ref^T)`` in degrees: roll, pitch, yaw about the camera's axes."""

    return np.degrees(Rotation.from_matrix(rotation @ reference.T).as_rotvec())


def score_unit(unit: UnitInput) -> UnitScore:
    """Apply the usable-unit rule, the failure policy, and the per-axis verdicts."""

    if unit.refused_windows is not None:
        return UnitScore(
            "inconclusive",
            f"the estimator refused: {unit.refused_windows} tracking windows (< 3), so "
            f"fewer than {USABLE_WINDOWS_MIN} windows",
        )
    if unit.error is not None:
        return UnitScore("failed", f"estimator failure: {unit.error}")
    fraction = unit.failed_pairs / unit.pairs if unit.pairs > 0 else 1.0
    if unit.windows_used < USABLE_WINDOWS_MIN:
        return UnitScore(
            "inconclusive",
            f"{unit.windows_used} windows after static exclusion (< {USABLE_WINDOWS_MIN})",
            windows_used=unit.windows_used,
            failed_pair_fraction=fraction,
        )
    if fraction > USABLE_FAILED_PAIR_FRACTION_MAX:
        return UnitScore(
            "inconclusive",
            f"{fraction:.3f} of tracked frame pairs failed (> {USABLE_FAILED_PAIR_FRACTION_MAX})",
            windows_used=unit.windows_used,
            failed_pair_fraction=fraction,
        )
    if (
        unit.solver_status != "converged"
        or unit.policy_status == "fail"
        or unit.rotation is None
        or unit.reference_rotation is None
        or unit.time_offset_s is None
        or unit.reference_shift_s is None
    ):
        return UnitScore(
            "failed",
            f"solver_status {unit.solver_status}, policy_status {unit.policy_status}",
            windows_used=unit.windows_used,
            failed_pair_fraction=fraction,
        )
    errors = axis_errors_deg(unit.rotation, unit.reference_rotation)
    return UnitScore(
        "scored",
        windows_used=unit.windows_used,
        failed_pair_fraction=fraction,
        axes={
            name: {
                "judged": axis.judged,
                "error_deg": float(error),
                "status": axis.status,
                "std_deg": axis.std,
                "control_detected": axis.control_detected,
            }
            for name, axis, error in zip(AXES, unit.axes, errors, strict=True)
        },
        time={
            "judged": unit.time.judged,
            "error_ms": (unit.time_offset_s - unit.reference_shift_s) * 1000.0,
            "status": unit.time.status,
            "std_ms": unit.time.std * 1000.0,
            "control_detected": unit.time.control_detected,
        },
    )


def _requirement(failed: bool | None, observed: float | None, threshold: float) -> dict[str, Any]:
    status = "not_evaluated" if failed is None else ("fail" if failed else "pass")
    return {"status": status, "observed": observed, "threshold": threshold}


def _maximum_gate(values: list[float], threshold: float) -> dict[str, Any]:
    if not values:
        return _requirement(None, None, threshold)
    worst = max(values)
    return _requirement(worst > threshold + TOLERANCE, worst, threshold)


def score_units(units: dict[str, dict[str, UnitInput]], recordings: list[str]) -> dict[str, Any]:
    """Score ``units[camera][recording]``; apply the requirements and the verdict."""

    scores = {
        camera: {
            recording: score_unit(units[camera][recording])
            for recording in recordings
            if recording in units[camera]
        }
        for camera in units
    }
    gated = [camera for camera in GATED_CAMERAS if camera in units]
    gated_scores = [score for camera in gated for score in scores[camera].values()]
    scored = [score for score in gated_scores if score.state == "scored"]
    failed = [score for score in gated_scores if score.state == "failed"]
    judged_axes = [axis for score in scored for axis in score.axes.values() if axis["judged"]]
    judged_times = [score.time for score in scored if score.time["judged"]]

    consistency: dict[str, dict[str, Any]] = {}
    for camera in gated:
        for name in AXES:
            errors = {
                recording: score.axes[name]["error_deg"]
                for recording, score in scores[camera].items()
                if score.state == "scored" and score.axes[name]["judged"]
            }
            if len(errors) >= 2:
                consistency[f"{camera}-{name}"] = {
                    "range_deg": max(errors.values()) - min(errors.values()),
                    "recordings": sorted(errors),
                }

    relative: dict[str, dict[str, Any]] = {}
    both_scored: list[str] = []
    for recording in recordings:
        pair = [scores.get(camera, {}).get(recording) for camera in GATED_CAMERAS]
        if not all(score is not None and score.state == "scored" for score in pair):
            continue
        both_scored.append(recording)
        first, second = (units[camera][recording] for camera in GATED_CAMERAS)
        assert first.rotation is not None and second.rotation is not None
        assert first.reference_rotation is not None and second.reference_rotation is not None
        error = axis_errors_deg(
            first.rotation @ second.rotation.T,
            first.reference_rotation @ second.reference_rotation.T,
        )
        for index, name in enumerate(AXES):
            if all(score is not None and score.axes[name]["judged"] for score in pair):
                relative[f"{recording}-{name}"] = {"relative_deg": float(error[index])}

    axis_count = 3 * len(scored)
    judged_fraction = len(judged_axes) / axis_count if axis_count else None
    requirements = {
        "judged-axis-accuracy": _maximum_gate(
            [abs(axis["error_deg"]) for axis in judged_axes], AXIS_ERROR_MAX_DEG
        ),
        "judged-time-offset": _maximum_gate(
            [abs(time["error_ms"]) for time in judged_times], TIME_ERROR_MAX_MS
        ),
        "judged-axis-coverage": _requirement(
            judged_fraction < JUDGED_FRACTION_MIN - TOLERANCE
            if judged_fraction is not None
            else None,
            judged_fraction,
            JUDGED_FRACTION_MIN,
        ),
        "no-failed-units": _requirement(bool(failed), float(len(failed)), 0.0),
        "judged-axis-consistency": _maximum_gate(
            [item["range_deg"] for item in consistency.values()], CONSISTENCY_MAX_DEG
        ),
        "judged-axis-relative": _maximum_gate(
            [abs(item["relative_deg"]) for item in relative.values()], RELATIVE_MAX_DEG
        ),
        "scored-recordings-coverage": _requirement(
            len(both_scored) < SCORED_RECORDINGS_MIN,
            float(len(both_scored)),
            float(SCORED_RECORDINGS_MIN),
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
        "recordings": list(recordings),
        "units": {
            camera: {recording: asdict(score) for recording, score in per.items()}
            for camera, per in scores.items()
        },
        "judged_axes": len(judged_axes),
        "scored_axes": axis_count,
        "judged_fraction": judged_fraction,
        "consistency": consistency,
        "relative_cam0_cam1": relative,
        "recordings_scored_for_both_cameras": both_scored,
        "requirements": requirements,
        "verdict": verdict,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _axis(record: Any) -> AxisInput:
    control = record.known_bad_control
    return AxisInput(
        status=record.status,
        std=float(record.std_reported),
        control_detected=None if control is None else bool(control.detected),
    )


def load_unit(
    artifact_path: Path, log_path: Path, camchain: dict[str, Any], camera: str
) -> UnitInput:
    """Read one unit: its artifact, or the refusal its log shows, and Kalibr's reference."""

    from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation

    if not artifact_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        refusal = REFUSAL.search(log)
        if refusal is not None:
            return UnitInput(refused_windows=int(refusal.group(1)))
        return UnitInput(error="artifact missing")
    try:
        artifact = load_imu_lidar_rotation(artifact_path)
    except Exception as exc:  # an unreadable artifact is a failed unit
        return UnitInput(error=f"artifact unreadable: {exc}")
    for name, value in RECIPE.items():
        if artifact.options.get(name) != value:
            raise ValueError(f"{artifact_path}: {name} is not the pre-registered {value}")
    entry = camchain[camera]
    records = {record.name: record for record in artifact.dofs}
    visual = artifact.options.get("visual_rotation", [])
    return UnitInput(
        rotation=(
            Rotation.from_quat(artifact.rotation_quat_xyzw).as_matrix()
            if artifact.rotation_quat_xyzw is not None
            else None
        ),
        reference_rotation=np.asarray(entry["T_cam_imu"], dtype=np.float64)[:3, :3],
        time_offset_s=artifact.time_offset_s,
        reference_shift_s=float(entry["timeshift_cam_imu"]),
        windows_used=artifact.train_windows + artifact.holdout_windows,
        pairs=sum(int(item["pairs"]) for item in visual),
        failed_pairs=sum(int(item["failed"]) for item in visual),
        solver_status=artifact.solver_status,
        policy_status=artifact.policy_status,
        axes=tuple(_axis(records[name]) for name in AXES),  # type: ignore[arg-type]
        time=_axis(records["time_offset"]),
    )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("camchain", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--recording", action="append", required=True)
    args = parser.parse_args(argv)
    from calibrex import __version__
    from calibrex.core.io import read_mapping

    camchain = read_mapping(args.camchain)
    units: dict[str, dict[str, UnitInput]] = {camera: {} for camera in GATED_CAMERAS}
    hashes: dict[str, str | None] = {}
    for camera in GATED_CAMERAS:
        for recording in args.recording:
            artifact = args.runs_dir / f"{camera}_{recording}.yaml"
            log = args.runs_dir / f"{camera}_{recording}.log"
            units[camera][recording] = load_unit(artifact, log, camchain, camera)
            hashes[f"{camera}/{recording}"] = _sha256(artifact) if artifact.is_file() else None
    result = score_units(units, args.recording)
    result["provenance"] = {
        "generator": "tools/score_hilti_camera_imu_v3.py",
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "calibrex_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recordings": args.recording,
        "artifact_sha256": hashes,
        "camchain_sha256": _sha256(args.camchain),
        "thresholds": {
            "axis_error_max_deg": AXIS_ERROR_MAX_DEG,
            "time_error_max_ms": TIME_ERROR_MAX_MS,
            "judged_fraction_min": JUDGED_FRACTION_MIN,
            "consistency_max_deg": CONSISTENCY_MAX_DEG,
            "relative_max_deg": RELATIVE_MAX_DEG,
            "usable_windows_min": USABLE_WINDOWS_MIN,
            "usable_failed_pair_fraction_max": USABLE_FAILED_PAIR_FRACTION_MAX,
            "scored_recordings_min": SCORED_RECORDINGS_MIN,
            "recipe": RECIPE,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for camera in GATED_CAMERAS:
        for recording, unit in result["units"][camera].items():
            if unit["state"] != "scored":
                print(f"{camera} {recording}: {unit['state']} ({unit['reason']})")
                continue
            axes = " ".join(
                f"{name} {axis['error_deg']:+.3f}{'' if axis['judged'] else '(u)'}"
                for name, axis in unit["axes"].items()
            )
            print(f"{camera} {recording}: {axes}; dt {unit['time']['error_ms']:+.3f} ms")
    for name, item in result["requirements"].items():
        observed, threshold = item["observed"], item["threshold"]
        print(f"  {name}: {item['status']} (observed {observed}, threshold {threshold})")
    print("verdict:", result["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
