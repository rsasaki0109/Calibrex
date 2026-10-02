"""Score camera-IMU rotation artifacts against Kalibr for the Hilti 2022 audit.

The scorer implements exactly the metrics, gates, and verdict rules of
``docs/benchmarks/hilti_camera_imu_preregistration.yaml``; its thresholds are
module constants and cannot be changed from the command line.  Usage::

    python tools/score_hilti_camera_imu.py MANIFEST.json OUT_DIR

``MANIFEST.json``::

    {
      "recordings": ["exp01", "exp02", "exp03", "exp04"],
      "camchains": {"cam0": {"path": "calib_3_cam0-1-camchain-imucam.yaml", "key": "cam0"},
                    "cam1": {"path": "calib_3_cam0-1-camchain-imucam.yaml", "key": "cam1"}},
      "artifacts": {"cam0": {"exp01": "cam0_exp01.yaml", "exp02": null, ...}, "cam1": {...}}
    }

An artifact entry of ``null``, or a missing or unreadable file, is an estimator
failure and counts as a failed unit.  Cameras other than ``cam0`` and ``cam1``
may be listed; they are reported but never gated.  An artifact whose recorded
``min_window_rotation_deg`` or ``observable_rotation_std_deg`` differs from the
pre-registered recipe is rejected (the scorer aborts), so a run with other
options cannot be scored.

A unit is one camera on one recording.  ``scores.json`` holds every unit, the
per-camera consistency, the per-recording relative rotation, the requirements
(``pass``, ``fail``, or ``not_evaluated``), the verdict, and provenance.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

GATED_CAMERAS = ("cam0", "cam1")
ROTATION_ERROR_MAX_DEG = 1.0
TIME_ERROR_MAX_MS = 0.5
CONSTRAINED_STD_MAX_DEG = 0.3
CONSTRAINED_CONTROLS_MIN = 2
CONSISTENCY_MAX_DEG = 0.75
RELATIVE_MAX_DEG = 0.5
USABLE_WINDOWS_MIN = 8
USABLE_FAILED_PAIR_FRACTION_MAX = 0.25
SCORED_RECORDINGS_MIN = 3
RECIPE_MIN_WINDOW_ROTATION_DEG = 1.0
TOLERANCE = 1e-9  # a value within 1e-9 of its threshold meets it (rotation round-off)
REQUIREMENTS = (
    "rotation-vs-kalibr",
    "time-offset-vs-kalibr",
    "constrained",
    "no-failed-units",
    "cross-recording-consistency",
    "relative-cam0-cam1",
)


@dataclass(frozen=True)
class UnitInput:
    """What the scorer needs from one rotation artifact and its Kalibr reference."""

    error: str | None = None
    rotation: np.ndarray | None = None
    time_offset_s: float | None = None
    reference_rotation: np.ndarray | None = None
    reference_shift_s: float | None = None
    windows_used: int = 0
    pairs: int = 0
    failed_pairs: int = 0
    solver_status: str = "converged"
    policy_status: str = "pass"
    axis_status: tuple[str, str, str] = ("estimated",) * 3
    axis_std_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controls_detected: tuple[bool, bool, bool] = (True, True, True)


@dataclass
class UnitScore:
    """One scored (or not) unit."""

    state: str  # scored | inconclusive | failed
    reason: str = ""
    rotation_error_deg: float | None = None
    time_error_ms: float | None = None
    constrained: bool | None = None
    windows_used: int = 0
    failed_pair_fraction: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


def geodesic_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Geodesic angle (deg) of ``first @ second.T``."""

    return float(np.degrees(Rotation.from_matrix(first @ second.T).magnitude()))


def score_unit(unit: UnitInput) -> UnitScore:
    """Apply the usable-unit rule, the failure policy, and the per-unit metrics."""

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
    rotation_error = geodesic_deg(unit.rotation, unit.reference_rotation)
    time_error = (unit.time_offset_s - unit.reference_shift_s) * 1000.0
    constrained = (
        all(status == "estimated" for status in unit.axis_status)
        and all(std <= CONSTRAINED_STD_MAX_DEG + TOLERANCE for std in unit.axis_std_deg)
        and sum(unit.controls_detected) >= CONSTRAINED_CONTROLS_MIN
    )
    return UnitScore(
        "scored",
        windows_used=unit.windows_used,
        failed_pair_fraction=fraction,
        rotation_error_deg=rotation_error,
        time_error_ms=time_error,
        constrained=constrained,
        details={
            "axis_status": list(unit.axis_status),
            "axis_std_deg": list(unit.axis_std_deg),
            "controls_detected": list(unit.controls_detected),
            "policy_status": unit.policy_status,
        },
    )


def score_units(units: dict[str, dict[str, UnitInput]], recordings: list[str]) -> dict[str, Any]:
    """Score ``units[camera][recording]`` and apply the five requirements and the verdict."""

    scores = {
        camera: {
            recording: score_unit(units[camera][recording])
            for recording in recordings
            if recording in units[camera]
        }
        for camera in units
    }
    gated = [camera for camera in GATED_CAMERAS if camera in units]
    consistency: dict[str, float | None] = {}
    for camera in gated:
        rotations = [
            units[camera][recording].rotation
            for recording, score in scores[camera].items()
            if score.state == "scored"
        ]
        consistency[camera] = (
            max(geodesic_deg(a, b) for a, b in itertools.combinations(rotations, 2))  # type: ignore[arg-type]
            if len(rotations) >= 2
            else None
        )
    relative: dict[str, float | None] = {}
    both_scored: list[str] = []
    for recording in recordings:
        states = [
            scores[camera].get(recording, UnitScore("failed")).state == "scored"
            for camera in GATED_CAMERAS
            if camera in units
        ]
        if len(states) == len(GATED_CAMERAS) and all(states):
            both_scored.append(recording)
            first, second = (units[camera][recording] for camera in GATED_CAMERAS)
            assert first.rotation is not None and second.rotation is not None
            assert first.reference_rotation is not None and second.reference_rotation is not None
            relative[recording] = geodesic_deg(
                first.rotation @ second.rotation.T,
                first.reference_rotation @ second.reference_rotation.T,
            )
        else:
            relative[recording] = None
    gated_scores = [score for camera in gated for score in scores[camera].values()]
    scored = [score for score in gated_scores if score.state == "scored"]
    failed = [score for score in gated_scores if score.state == "failed"]

    def requirement(failed_gate: bool | None, observed: float | None, threshold: float | None):
        status = "not_evaluated" if failed_gate is None else ("fail" if failed_gate else "pass")
        return {"status": status, "observed": observed, "threshold": threshold}

    rotation_values = [s.rotation_error_deg for s in scored]
    time_values = [abs(s.time_error_ms) for s in scored]  # type: ignore[arg-type]
    consistency_values = [value for value in consistency.values() if value is not None]
    relative_values = [value for value in relative.values() if value is not None]
    requirements = {
        "rotation-vs-kalibr": requirement(
            max(rotation_values) > ROTATION_ERROR_MAX_DEG + TOLERANCE if scored else None,
            max(rotation_values) if scored else None,  # type: ignore[type-var]
            ROTATION_ERROR_MAX_DEG,
        ),
        "time-offset-vs-kalibr": requirement(
            max(time_values) > TIME_ERROR_MAX_MS + TOLERANCE if scored else None,
            max(time_values) if scored else None,
            TIME_ERROR_MAX_MS,
        ),
        "constrained": requirement(
            any(not s.constrained for s in scored) if scored else None,
            float(sum(not s.constrained for s in scored)) if scored else None,
            0.0,
        ),
        "no-failed-units": requirement(bool(failed), float(len(failed)), 0.0),
        "cross-recording-consistency": requirement(
            max(consistency_values) > CONSISTENCY_MAX_DEG + TOLERANCE
            if consistency_values
            else None,
            max(consistency_values) if consistency_values else None,
            CONSISTENCY_MAX_DEG,
        ),
        "relative-cam0-cam1": requirement(
            max(relative_values) > RELATIVE_MAX_DEG + TOLERANCE if relative_values else None,
            max(relative_values) if relative_values else None,
            RELATIVE_MAX_DEG,
        ),
    }
    statuses = [item["status"] for item in requirements.values()]
    if "fail" in statuses:
        verdict = "refuted"
    elif all(status == "pass" for status in statuses) and len(both_scored) >= SCORED_RECORDINGS_MIN:
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {
        "units": {
            camera: {recording: vars(score) for recording, score in per.items()}
            for camera, per in scores.items()
        },
        "consistency_deg": consistency,
        "relative_cam0_cam1_deg": relative,
        "recordings_scored_for_both_cameras": both_scored,
        "requirements": requirements,
        "verdict": verdict,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_unit(
    artifact_path: str | None, camchain: dict[str, Any], key: str
) -> tuple[UnitInput, str | None]:
    from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
    from calibrex.core.io import read_mapping

    if artifact_path is None or not Path(artifact_path).is_file():
        return UnitInput(error="artifact missing"), None
    try:
        artifact = load_imu_lidar_rotation(artifact_path)
    except Exception as exc:  # an unreadable artifact is a failed unit
        return UnitInput(error=f"artifact unreadable: {exc}"), _sha256(Path(artifact_path))
    options = artifact.options
    if options.get("min_window_rotation_deg") != RECIPE_MIN_WINDOW_ROTATION_DEG:
        raise ValueError(f"{artifact_path}: min_window_rotation_deg is not the pre-registered 1.0")
    if options.get("observable_rotation_std_deg") != CONSTRAINED_STD_MAX_DEG:
        raise ValueError(
            f"{artifact_path}: observable_rotation_std_deg is not the pre-registered 0.3"
        )
    entry = read_mapping(Path(camchain["path"]))[camchain["key"]]
    reference_rotation = np.asarray(entry["T_cam_imu"], dtype=np.float64)[:3, :3]
    records = {record.name: record for record in artifact.dofs}
    axes = [records[name] for name in ("roll", "pitch", "yaw")]
    visual = options.get("visual_rotation", [])
    unit = UnitInput(
        rotation=(
            Rotation.from_quat(artifact.rotation_quat_xyzw).as_matrix()
            if artifact.rotation_quat_xyzw is not None
            else None
        ),
        time_offset_s=artifact.time_offset_s,
        reference_rotation=reference_rotation,
        reference_shift_s=float(entry["timeshift_cam_imu"]),
        windows_used=artifact.train_windows + artifact.holdout_windows,
        pairs=sum(int(item["pairs"]) for item in visual),
        failed_pairs=sum(int(item["failed"]) for item in visual),
        solver_status=artifact.solver_status,
        policy_status=artifact.policy_status,
        axis_status=tuple(axis.status for axis in axes),  # type: ignore[arg-type]
        axis_std_deg=tuple(axis.std_reported for axis in axes),  # type: ignore[arg-type]
        controls_detected=tuple(  # type: ignore[arg-type]
            bool(axis.known_bad_control and axis.known_bad_control.detected) for axis in axes
        ),
    )
    return unit, _sha256(Path(artifact_path))


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def main() -> None:
    from calibrex import __version__

    manifest_path, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recordings: list[str] = manifest["recordings"]
    units: dict[str, dict[str, UnitInput]] = {}
    artifact_hashes: dict[str, str | None] = {}
    for camera, per_recording in manifest["artifacts"].items():
        units[camera] = {}
        for recording in recordings:
            unit, digest = _load_unit(
                per_recording.get(recording),
                manifest["camchains"][camera],
                manifest["camchains"][camera]["key"],
            )
            units[camera][recording] = unit
            artifact_hashes[f"{camera}/{recording}"] = digest
    result = score_units(units, recordings)
    result["provenance"] = {
        "generator": "tools/score_hilti_camera_imu.py",
        "generator_sha256": _sha256(Path(__file__)),
        "calibrex_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [
            "python",
            "tools/score_hilti_camera_imu.py",
            str(manifest_path),
            str(output_dir),
        ],
        "manifest_sha256": _sha256(manifest_path),
        "artifact_sha256": artifact_hashes,
        "camchain_sha256": {
            camera: _sha256(Path(entry["path"])) for camera, entry in manifest["camchains"].items()
        },
        "recordings": recordings,
        "gated_cameras": [c for c in GATED_CAMERAS if c in units],
        "reported_cameras": [c for c in units if c not in GATED_CAMERAS],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "rotation_error_max_deg": ROTATION_ERROR_MAX_DEG,
            "time_error_max_ms": TIME_ERROR_MAX_MS,
            "constrained_std_max_deg": CONSTRAINED_STD_MAX_DEG,
            "constrained_controls_min": CONSTRAINED_CONTROLS_MIN,
            "consistency_max_deg": CONSISTENCY_MAX_DEG,
            "relative_max_deg": RELATIVE_MAX_DEG,
            "usable_windows_min": USABLE_WINDOWS_MIN,
            "usable_failed_pair_fraction_max": USABLE_FAILED_PAIR_FRACTION_MAX,
            "scored_recordings_min": SCORED_RECORDINGS_MIN,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for camera, per in result["units"].items():
        tag = "" if camera in GATED_CAMERAS else " (not gated)"
        for recording, unit in per.items():
            print(
                f"{camera} {recording}{tag}: {unit['state']}"
                + (
                    f" rot {unit['rotation_error_deg']:.3f} deg,"
                    f" dt {unit['time_error_ms']:+.2f} ms,"
                    f" constrained {unit['constrained']}"
                    if unit["state"] == "scored"
                    else f" ({unit['reason']})"
                )
            )
    print("consistency deg:", result["consistency_deg"])
    print("relative cam0-cam1 deg:", result["relative_cam0_cam1_deg"])
    for name, item in result["requirements"].items():
        print(
            f"  {name}: {item['status']} "
            f"(observed {item['observed']}, threshold {item['threshold']})"
        )
    print("verdict:", result["verdict"])


if __name__ == "__main__":
    main()
