"""Evaluate a LiDAR-to-wheel-odometry calibration on held-out blocks.

LiDAR motions (from odometry or a trajectory) are grouped into
``block_duration_s`` blocks and every third block is held out.  The reported
std is the larger of the analytic std and an 8-group block jackknife.  The
held-out chi-square must rise by at least 9 when the rotation is turned by
1 deg about each vehicle axis, the clock offset shifted by 20 ms, or the
speed scale changed by 1 %.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.lidar_wheel_odometry import (
    LidarWheelControl,
    LidarWheelOdometryArtifact,
    LidarWheelProvenance,
    LidarWheelRecord,
    LidarWheelReference,
)
from calibrex.core.provenance import git_commit
from calibrex.data.imu_trajectory import parse_tum_trajectory
from calibrex.data.kitti import read_timestamps
from calibrex.data.kitti_ins_lidar import load_kitti_ins_lidar_drive
from calibrex.evaluation.ins_lidar_hand_eye import _velodyne_loader
from calibrex.solvers.lidar_wheel_solver import (
    WHEEL_PARAMETERS,
    LidarWheelOptions,
    LidarWheelResult,
    WheelOdometry,
    lidar_wheel_residuals,
    parse_wheel_csv,
    solve_lidar_wheel,
    wheel_std,
)
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions, odometry_from_loader
from calibrex.solvers.vehicle_frame_solver import VehicleMotion, motions_from_poses

FloatArray: TypeAlias = NDArray[np.float64]
_UNITS = {
    "roll": "deg",
    "pitch": "deg",
    "yaw": "deg",
    "lever": "m",
    "speed_scale": "ratio",
    "time_offset": "s",
}
LIMITATIONS: tuple[str, ...] = (
    "The vehicle frame is the wheel odometry's: x along the reported speed, z about "
    "the reported yaw rate; no side slip is assumed.",
    "Roll is seen only through the tilt of the turning axis and needs turns.",
)


@dataclass(frozen=True)
class LidarWheelRunOptions:
    """Motion construction, blocking, evidence, and observability."""

    step_frames: int = 2
    block_duration_s: float = 10.0
    holdout_every: int = 3
    jackknife_groups: int = 8
    rotation_control_deg: float = 1.0
    time_control_s: float = 0.02
    scale_control: float = 0.01
    detection_delta_chi2: float = 9.0
    observable: dict[str, float] = field(
        default_factory=lambda: {
            "roll": 0.1,
            "pitch": 0.1,
            "yaw": 0.1,
            "lever": 0.2,
            "speed_scale": 0.005,
            "time_offset": 0.005,
        }
    )
    solver: LidarWheelOptions = field(default_factory=LidarWheelOptions)
    odometry: ScanOdometryOptions = field(default_factory=ScanOdometryOptions)


@dataclass(frozen=True)
class LidarWheelEvaluation:
    """Dataset-independent evaluation output."""

    result: LidarWheelResult
    records: tuple[LidarWheelRecord, ...]
    train_motions: int
    holdout_motions: int
    jackknife_fits: int
    train_median: float | None
    holdout_median: float | None
    policy_status: str
    policy_reasons: tuple[str, ...]


def evaluate_lidar_wheel(
    motions: Sequence[VehicleMotion],
    wheel: WheelOdometry,
    options: LidarWheelRunOptions | None = None,
) -> LidarWheelEvaluation:
    """Fit on train blocks; collect jackknife, control, and holdout evidence."""

    opts = options or LidarWheelRunOptions()
    blocks = sorted({item.block for item in motions})
    holdout_blocks = {
        block for position, block in enumerate(blocks) if position % opts.holdout_every == 1
    }
    train = [item for item in motions if item.block not in holdout_blocks]
    holdout = [item for item in motions if item.block in holdout_blocks]
    result = solve_lidar_wheel(train, wheel, opts.solver)
    if result.rotation is None or result.speed_scale is None or result.time_offset_s is None:
        return LidarWheelEvaluation(
            result, (), len(train), len(holdout), 0, None, None, "fail",
            ("too few moving intervals to solve",),
        )  # fmt: skip
    fixed = np.array(
        [0.0, 0.0, 0.0, result.lever_m or 0.0, result.speed_scale, result.time_offset_s]
    )
    jackknife, fits = _jackknife(train, wheel, sorted(set(blocks) - holdout_blocks), result, opts)
    analytic = wheel_std(result)
    base = _chi2(holdout, wheel, fixed, result.rotation, opts)
    euler = Rotation.from_matrix(result.rotation).as_euler("xyz")
    values = [*euler, fixed[3], fixed[4], fixed[5]]
    records = []
    for index, name in enumerate(WHEEL_PARAMETERS):
        rotation = index < 3
        reported = max(
            float(analytic[index]), 0.0 if jackknife is None else float(jackknife[index])
        )
        convert = math.degrees if rotation else (lambda value: value)
        threshold = opts.observable[name]
        threshold_native = math.radians(threshold) if rotation else threshold
        records.append(
            LidarWheelRecord(
                name=name,  # type: ignore[arg-type]
                unit=_UNITS[name],  # type: ignore[arg-type]
                value=convert(float(values[index])),
                std_analytic=convert(float(analytic[index])),
                std_jackknife=None if jackknife is None else convert(float(jackknife[index])),
                std_reported=convert(reported),
                status="estimated" if reported <= threshold_native else "unobservable",
                known_bad_control=_control(name, index, holdout, wheel, fixed, result, base, opts),
            )
        )
    train_median = _median(train, wheel, fixed, result.rotation, opts)
    holdout_median = _median(holdout, wheel, fixed, result.rotation, opts)
    status, reasons = _policy(records, train_median, holdout_median)
    return LidarWheelEvaluation(
        result, tuple(records), len(train), len(holdout), fits, train_median, holdout_median,
        status, reasons,
    )  # fmt: skip


def build_artifact(
    evaluation: LidarWheelEvaluation,
    options: LidarWheelRunOptions,
    *,
    motions: int,
    references: list[LidarWheelReference],
    provenance: LidarWheelProvenance,
    limitations: Sequence[str],
) -> LidarWheelOdometryArtifact:
    """Assemble the schema-valid artifact."""

    result = evaluation.result
    return LidarWheelOdometryArtifact(
        solver_status=result.status,
        policy_status=evaluation.policy_status,  # type: ignore[arg-type]
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_parameters=[
            item.name for item in evaluation.records if item.status == "estimated"
        ],
        rotation_quat_xyzw=None
        if result.rotation is None
        else [float(value) for value in Rotation.from_matrix(result.rotation).as_quat()],
        lever_m=result.lever_m,
        speed_scale=result.speed_scale,
        time_offset_s=result.time_offset_s,
        parameters=list(evaluation.records),
        references=references,
        motions=motions,
        train_motions=evaluation.train_motions,
        holdout_motions=evaluation.holdout_motions,
        jackknife_fits=evaluation.jackknife_fits,
        train_median_normalized_residual=evaluation.train_median,
        holdout_median_normalized_residual=evaluation.holdout_median,
        options={
            "step_frames": options.step_frames,
            "block_duration_s": options.block_duration_s,
            "holdout_every": options.holdout_every,
            "jackknife_groups": options.jackknife_groups,
            "rotation_control_deg": options.rotation_control_deg,
            "time_control_s": options.time_control_s,
            "scale_control": options.scale_control,
            "detection_delta_chi2": options.detection_delta_chi2,
            "observable": dict(options.observable),
            "time_offset_bound_s": options.solver.time_offset_bound_s,
        },
        limitations=list(limitations),
        provenance=provenance,
    )


def run_trajectory_lidar_wheel(
    trajectory_text: str,
    wheel_text: str,
    options: LidarWheelRunOptions | None = None,
    *,
    sequence_id: str = "trajectory",
    dataset_family: str = "user",
    dataset_license: str = "user-provided",
    command: list[str] | None = None,
) -> LidarWheelOdometryArtifact:
    """Calibrate against wheel odometry from a sensor trajectory (TUM) and a wheel CSV."""

    opts = options or LidarWheelRunOptions()
    times, poses = parse_tum_trajectory(trajectory_text)
    wheel = parse_wheel_csv(wheel_text)
    motions = motions_from_poses(
        times, list(poses), step=opts.step_frames, block_duration_s=opts.block_duration_s
    )
    evaluation = evaluate_lidar_wheel(motions, wheel, opts)
    digest = hashlib.sha256((trajectory_text + "\0" + wheel_text).encode("utf-8")).hexdigest()
    return build_artifact(
        evaluation,
        opts,
        motions=len(motions),
        references=[],
        provenance=LidarWheelProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family=dataset_family,
            sequence_ids=[sequence_id],
            wheel_source="wheel CSV (t, speed, yaw_rate)",
            input_sha256=digest,
            dataset_license=dataset_license,
        ),
        limitations=LIMITATIONS,
    )


def kitti_oxts_wheel_proxy(drive_dir: str | Path) -> WheelOdometry:
    """Forward speed ``vf`` and yaw rate ``wu`` of the OXTS level frame, as a wheel stand-in."""

    root = Path(drive_dir)
    times = np.array(
        [stamp.timestamp_ns * 1.0e-9 for stamp in read_timestamps(root / "oxts" / "timestamps.txt")]
    )
    values = np.array([np.loadtxt(path) for path in sorted((root / "oxts" / "data").glob("*.txt"))])
    order = np.argsort(times, kind="stable")
    return WheelOdometry(times[order], values[order, 8], values[order, 22])


def run_kitti_lidar_wheel(
    drive_dirs: Sequence[str | Path],
    options: LidarWheelRunOptions | None = None,
    *,
    lidar_vehicle_rotation: FloatArray | None = None,
    command: list[str] | None = None,
) -> LidarWheelOdometryArtifact:
    """KITTI raw: LiDAR odometry against the OXTS forward speed and yaw rate (a wheel proxy)."""

    opts = options or LidarWheelRunOptions()
    if not drive_dirs:
        raise ValueError("at least one KITTI drive is required")
    motions: list[VehicleMotion] = []
    wheels: list[WheelOdometry] = []
    digests = []
    for position, path in enumerate(drive_dirs):
        drive = load_kitti_ins_lidar_drive(path)
        frames = drive.velodyne_frames
        times = [frame.time_s for frame in frames]
        odometry = odometry_from_loader(
            _velodyne_loader(frames), len(frames), opts.odometry, times_s=times
        )
        motions += motions_from_poses(
            times,
            list(odometry.poses),
            step=opts.step_frames,
            block_duration_s=opts.block_duration_s,
            block_offset=position * 100_000,
        )
        wheels.append(kitti_oxts_wheel_proxy(path))
        digests.append(drive.input_sha256)
    order = np.argsort([float(item.times_s[0]) for item in wheels])
    wheel = WheelOdometry(
        np.concatenate([wheels[index].times_s for index in order]),
        np.concatenate([wheels[index].speed_mps for index in order]),
        np.concatenate([wheels[index].yaw_rate_rps for index in order]),
    )
    evaluation = evaluate_lidar_wheel(motions, wheel, opts)
    references = []
    if lidar_vehicle_rotation is not None and evaluation.result.rotation is not None:
        difference = Rotation.from_matrix(
            evaluation.result.rotation @ lidar_vehicle_rotation.T
        ).as_rotvec()
        references.append(
            LidarWheelReference(
                name="LiDAR-to-vehicle estimate from motion alone (lidar-vehicle)",
                rotation_quat_xyzw=[
                    float(value) for value in Rotation.from_matrix(lidar_vehicle_rotation).as_quat()
                ],
                difference_deg=[math.degrees(float(value)) for value in difference],
            )
        )
    return build_artifact(
        evaluation,
        opts,
        motions=len(motions),
        references=references,
        provenance=LidarWheelProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family="kitti_raw",
            sequence_ids=[Path(path).name for path in drive_dirs],
            wheel_source=(
                "proxy: OXTS level-frame forward speed vf and yaw rate wu "
                "(KITTI has no wheel odometry)"
            ),
            input_sha256=hashlib.sha256("".join(digests).encode("ascii")).hexdigest(),
            dataset_license="CC BY-NC-SA 3.0",
        ),
        limitations=(
            *LIMITATIONS,
            "KITTI has no wheel odometry: the OXTS forward speed and yaw rate stand in for it, "
            "so the speed scale and clock offset describe the OXTS, not wheels.",
        ),
    )


def _chi2(
    motions: Sequence[VehicleMotion],
    wheel: WheelOdometry,
    params: FloatArray,
    rotation: FloatArray,
    opts: LidarWheelRunOptions,
) -> float:
    residuals = lidar_wheel_residuals(motions, wheel, params, rotation, opts.solver)
    return float(np.sum(np.minimum(residuals**2, 25.0)))


def _median(
    motions: Sequence[VehicleMotion],
    wheel: WheelOdometry,
    params: FloatArray,
    rotation: FloatArray,
    opts: LidarWheelRunOptions,
) -> float | None:
    residuals = lidar_wheel_residuals(motions, wheel, params, rotation, opts.solver)
    return float(np.median(np.abs(residuals))) if len(residuals) else None


def _control(
    name: str,
    index: int,
    holdout: Sequence[VehicleMotion],
    wheel: WheelOdometry,
    fixed: FloatArray,
    result: LidarWheelResult,
    base: float,
    opts: LidarWheelRunOptions,
) -> LidarWheelControl | None:
    if not holdout or name == "lever" or result.rotation is None:
        return None
    moved = fixed.copy()
    if index < 3:
        amount, unit = opts.rotation_control_deg, "deg"
        moved[index] = math.radians(amount)
    elif name == "speed_scale":
        amount, unit = opts.scale_control, "ratio"
        moved[4] += amount
    else:
        amount, unit = opts.time_control_s, "s"
        moved[5] += amount
    delta = _chi2(holdout, wheel, moved, result.rotation, opts) - base
    return LidarWheelControl(
        amount=amount,
        unit=unit,  # type: ignore[arg-type]
        holdout_delta_chi2=float(delta),
        detected=bool(delta >= opts.detection_delta_chi2),
    )


def _jackknife(
    train: Sequence[VehicleMotion],
    wheel: WheelOdometry,
    train_blocks: Sequence[int],
    result: LidarWheelResult,
    opts: LidarWheelRunOptions,
) -> tuple[FloatArray | None, int]:
    groups = min(opts.jackknife_groups, len(train_blocks))
    if groups < 3 or result.rotation is None:
        return None, 0
    samples = []
    for members in np.array_split(np.asarray(train_blocks), groups):
        left_out = set(members.tolist())
        fit = solve_lidar_wheel(
            [item for item in train if item.block not in left_out],
            wheel,
            opts.solver,
            initial=result.rotation,
        )
        if fit.rotation is None or fit.speed_scale is None or fit.time_offset_s is None:
            return None, 0
        samples.append(
            np.r_[
                Rotation.from_matrix(fit.rotation @ result.rotation.T).as_rotvec(),
                fit.lever_m or 0.0,
                fit.speed_scale,
                fit.time_offset_s,
            ]
        )
    array = np.asarray(samples)
    spread = np.sqrt((groups - 1) / groups * np.sum((array - array.mean(axis=0)) ** 2, axis=0))
    return np.asarray(spread, dtype=np.float64), groups


def _policy(
    records: Sequence[LidarWheelRecord], train_median: float | None, holdout_median: float | None
) -> tuple[str, tuple[str, ...]]:
    if holdout_median is None:
        return "inconclusive", ("no held-out moving intervals; holdout evidence is missing",)
    if train_median is not None and holdout_median > 2.0 * max(train_median, 1e-9):
        return "fail", (
            f"held-out median normalized residual {holdout_median:.2f} exceeds twice the "
            f"training residual {train_median:.2f}",
        )
    warnings = [
        f"{record.name} is reported as estimated but its known-bad shift is not detected "
        f"(delta chi-square {record.known_bad_control.holdout_delta_chi2:.1f})"
        for record in records
        if record.status == "estimated"
        and record.known_bad_control is not None
        and not record.known_bad_control.detected
    ]
    unobservable = [record.name for record in records if record.status == "unobservable"]
    notes = (
        ["not constrained by the data, so not calibrated: " + ", ".join(unobservable)]
        if unobservable
        else []
    )
    if warnings:
        return "warn", tuple(warnings + notes)
    needed = {"roll", "pitch", "yaw", "speed_scale", "time_offset"}
    if needed & set(unobservable):
        return "inconclusive", tuple(notes)
    summary = (
        "rotation, speed scale, and clock offset are estimated and held-out blocks "
        "detect every control"
    )
    return "pass", (summary, *notes)
