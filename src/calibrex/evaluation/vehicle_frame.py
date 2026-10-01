"""Evaluate a sensor-to-vehicle rotation from non-holonomic motion on held-out blocks.

Short-interval motions of the sensor (``step`` frames of its odometry) are
grouped into ``block_duration_s`` blocks and every third block is held out.
The reported std of each axis is the larger of the analytic std and an
8-group block jackknife, both as small rotations about the vehicle axes.
Rotating the estimate by 1 deg about each vehicle axis must raise the
held-out chi-square by at least 9.  Reference rotations are compared only
after the fit.

For KITTI raw the run also derives an independent vehicle frame from the
OXTS unit's own velocities and angular rates (the same non-holonomic solver
applied to the INS), because KITTI's ``calib_imu_to_velo`` gives the OXTS
frame, which need not be aligned with the vehicle's motion.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.provenance import git_commit
from calibrex.core.vehicle_frame_rotation import (
    VehicleFrameControl,
    VehicleFrameDofRecord,
    VehicleFrameProvenance,
    VehicleFrameReference,
    VehicleFrameRotationArtifact,
    VehiclePolicyStatus,
)
from calibrex.data.kitti_ins_lidar import load_kitti_ins_lidar_drive
from calibrex.evaluation.ins_lidar_hand_eye import _velodyne_loader
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions, odometry_from_loader
from calibrex.solvers.vehicle_frame_solver import (
    VEHICLE_DOFS,
    VehicleFrameOptions,
    VehicleFrameResult,
    VehicleMotion,
    motions_from_poses,
    solve_vehicle_frame,
    vehicle_residuals,
)

FloatArray: TypeAlias = NDArray[np.float64]
KITTI_LIMITATIONS: tuple[str, ...] = (
    "The vehicle frame is defined by the motion: it assumes no side slip and no "
    "vertical velocity, and roll needs turns.",
    "KITTI's calib_imu_to_velo gives the OXTS frame; the OXTS unit need not be aligned "
    "with the vehicle, so a second reference derives the vehicle frame from the OXTS "
    "velocities with the same solver.",
    "The translation of T_vehicle_sensor is not estimated.",
)


@dataclass(frozen=True)
class VehicleFrameRunOptions:
    """Motion construction, blocking, evidence, and observability."""

    step_frames: int = 2
    block_duration_s: float = 10.0
    holdout_every: int = 3
    jackknife_groups: int = 8
    control_deg: float = 1.0
    detection_delta_chi2: float = 9.0
    observable_rotation_std_deg: float = 0.1
    holdout_residual_ratio: float = 2.0
    solver: VehicleFrameOptions = field(default_factory=VehicleFrameOptions)
    odometry: ScanOdometryOptions = field(default_factory=ScanOdometryOptions)


@dataclass(frozen=True)
class VehicleFrameEvaluation:
    """Dataset-independent evaluation output."""

    result: VehicleFrameResult
    records: tuple[VehicleFrameDofRecord, ...]
    train_blocks: tuple[int, ...]
    holdout_blocks: tuple[int, ...]
    train_motions: int
    holdout_motions: int
    jackknife_fits: int
    train_median: float | None
    holdout_median: float | None
    holdout_chi2: float | None
    policy_status: VehiclePolicyStatus
    policy_reasons: tuple[str, ...]


def evaluate_vehicle_frame(
    motions: Sequence[VehicleMotion], options: VehicleFrameRunOptions | None = None
) -> VehicleFrameEvaluation:
    """Fit on train blocks; collect jackknife, control, and holdout evidence."""

    opts = options or VehicleFrameRunOptions()
    blocks = sorted({item.block for item in motions})
    holdout_blocks = tuple(
        block for position, block in enumerate(blocks) if position % opts.holdout_every == 1
    )
    train_blocks = tuple(block for block in blocks if block not in holdout_blocks)
    train = [item for item in motions if item.block in train_blocks]
    holdout = [item for item in motions if item.block in holdout_blocks]
    result = solve_vehicle_frame(train, opts.solver)
    if result.rotation is None or result.lever_m is None or result.covariance is None:
        return VehicleFrameEvaluation(
            result, (), train_blocks, holdout_blocks, len(train), len(holdout), 0, None, None,
            None, "fail", ("too few moving intervals to solve",),
        )  # fmt: skip
    rotation, lever = result.rotation, result.lever_m
    jackknife, fits = _jackknife(train, train_blocks, rotation, opts)
    base = _chi2(holdout, rotation, lever, opts)
    analytic = np.sqrt(np.clip(np.diag(result.covariance)[:3], 0.0, None))
    euler = Rotation.from_matrix(rotation).as_euler("xyz")
    records = []
    for index, name in enumerate(VEHICLE_DOFS):
        reported = max(
            float(analytic[index]), 0.0 if jackknife is None else float(jackknife[index])
        )
        control = None
        if holdout:
            axis = np.zeros(3)
            axis[index] = math.radians(opts.control_deg)
            moved = _chi2(holdout, Rotation.from_rotvec(axis).as_matrix() @ rotation, lever, opts)
            control = VehicleFrameControl(
                amount_deg=opts.control_deg,
                holdout_delta_chi2=float(moved - base),
                detected=bool(moved - base >= opts.detection_delta_chi2),
            )
        records.append(
            VehicleFrameDofRecord(
                name=name,  # type: ignore[arg-type]
                value=math.degrees(float(euler[index])),
                std_analytic=math.degrees(float(analytic[index])),
                std_jackknife=None if jackknife is None else math.degrees(float(jackknife[index])),
                std_reported=math.degrees(reported),
                status="estimated"
                if reported <= math.radians(opts.observable_rotation_std_deg)
                else "unobservable",
                known_bad_control=control,
            )
        )
    train_median = _median(train, rotation, lever, opts)
    holdout_median = _median(holdout, rotation, lever, opts)
    status, reasons = _policy(records, train_median, holdout_median, opts)
    return VehicleFrameEvaluation(
        result=result,
        records=tuple(records),
        train_blocks=train_blocks,
        holdout_blocks=holdout_blocks,
        train_motions=len(train),
        holdout_motions=len(holdout),
        jackknife_fits=fits,
        train_median=train_median,
        holdout_median=holdout_median,
        holdout_chi2=base if holdout else None,
        policy_status=status,
        policy_reasons=reasons,
    )


def reference_entry(
    name: str,
    reference: FloatArray,
    evaluation: VehicleFrameEvaluation,
    holdout: Sequence[VehicleMotion],
    options: VehicleFrameRunOptions,
) -> VehicleFrameReference:
    """Difference of the estimate to a reference rotation, and the reference's held-out cost."""

    rotation = evaluation.result.rotation
    assert rotation is not None and evaluation.result.lever_m is not None
    difference = Rotation.from_matrix(rotation @ reference.T).as_rotvec()
    delta = None
    if holdout and evaluation.holdout_chi2 is not None:
        delta = (
            _chi2(holdout, reference, evaluation.result.lever_m, options) - evaluation.holdout_chi2
        )
    return VehicleFrameReference(
        name=name,
        rotation_quat_xyzw=[float(value) for value in Rotation.from_matrix(reference).as_quat()],
        difference_deg=[math.degrees(float(value)) for value in difference],
        holdout_delta_chi2=delta,
    )


def oxts_motions(drive_dir: str | Path, *, block_duration_s: float) -> list[VehicleMotion]:
    """OXTS velocities and angular rates as motions in the OXTS body frame.

    KITTI's ``vf, vl, vu`` and ``wf, wl, wu`` are forward/left/up components in
    a *level* frame that follows the heading, not the body: level =
    ``Ry(pitch) Rx(roll)`` body (checked against ``wx, wy, wz`` and ``ax, ay, az``
    to 3e-4 rad/s and 0.014 m/s^2 on drive 0022).  Both are rotated back into
    the body frame with each packet's roll and pitch.
    """

    root = Path(drive_dir)
    files = sorted((root / "oxts" / "data").glob("*.txt"))
    from calibrex.data.kitti import read_timestamps

    times = [
        stamp.timestamp_ns * 1.0e-9 for stamp in read_timestamps(root / "oxts" / "timestamps.txt")
    ]
    velocities = []
    rates = []
    for path in files:
        velocity, rate = oxts_body_frame_motion(np.loadtxt(path))
        velocities.append(velocity)
        rates.append(rate)
    return body_motions(times[: len(files)], velocities, rates, block_duration_s=block_duration_s)


def oxts_body_frame_motion(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Body-frame velocity and angular rate of one OXTS packet (30 values).

    The level-frame ``vf, vl, vu`` and ``wf, wl, wu`` are rotated back into the
    body frame with the packet's roll and pitch (see :func:`oxts_motions`).
    """

    body_to_level = Rotation.from_euler("YX", [values[4], values[3]]).as_matrix()
    return (
        np.asarray(body_to_level.T @ values[8:11], dtype=np.float64),
        np.asarray(body_to_level.T @ values[20:23], dtype=np.float64),
    )


def body_motions(
    times_s: Sequence[float],
    velocities_mps: Sequence[FloatArray],
    angular_rates_rps: Sequence[FloatArray],
    *,
    block_duration_s: float,
    block_offset: int = 0,
) -> list[VehicleMotion]:
    """Instantaneous body-frame velocity and rate samples as blocked motions."""

    if not len(times_s):
        return []
    start = float(times_s[0])
    return [
        VehicleMotion(
            block=block_offset + int((float(time_s) - start) // block_duration_s),
            start_s=float(time_s),
            end_s=float(time_s),
            velocity_mps=np.asarray(velocity, dtype=np.float64),
            angular_rate_rps=np.asarray(rate, dtype=np.float64),
        )
        for time_s, velocity, rate in zip(times_s, velocities_mps, angular_rates_rps, strict=True)
    ]


def run_kitti_lidar_vehicle(
    drive_dirs: Sequence[str | Path],
    options: VehicleFrameRunOptions | None = None,
    *,
    command: list[str] | None = None,
) -> VehicleFrameRotationArtifact:
    """LiDAR odometry per drive, pooled motions, and the non-holonomic evaluation."""

    opts = options or VehicleFrameRunOptions()
    if not drive_dirs:
        raise ValueError("at least one KITTI drive is required")
    motions: list[VehicleMotion] = []
    oxts: list[VehicleMotion] = []
    vendor: FloatArray | None = None
    digests = []
    for position, path in enumerate(drive_dirs):
        drive = load_kitti_ins_lidar_drive(path)
        if vendor is not None and not np.allclose(drive.vendor_t_imu_lidar, vendor, atol=1e-9):
            raise ValueError("drives from different calibrations cannot be pooled")
        vendor = drive.vendor_t_imu_lidar
        frames = drive.velodyne_frames
        times = [frame.time_s for frame in frames]
        odometry = odometry_from_loader(
            _velodyne_loader(frames),
            len(frames),
            opts.odometry,
            times_s=times,
        )
        offset = position * 100_000
        motions += motions_from_poses(
            times,
            list(odometry.poses),
            step=opts.step_frames,
            block_duration_s=opts.block_duration_s,
            block_offset=offset,
        )
        oxts += [
            VehicleMotion(
                item.block + offset,
                item.start_s,
                item.end_s,
                item.velocity_mps,
                item.angular_rate_rps,
            )
            for item in oxts_motions(path, block_duration_s=opts.block_duration_s)
        ]
        digests.append(drive.input_sha256)
    assert vendor is not None
    evaluation = evaluate_vehicle_frame(motions, opts)
    references = []
    if evaluation.result.rotation is not None:
        holdout = [item for item in motions if item.block in evaluation.holdout_blocks]
        vendor_rotation = vendor[:3, :3]
        references.append(
            reference_entry(
                "KITTI calib_imu_to_velo rotation (OXTS frame, taken as the vehicle frame)",
                vendor_rotation,
                evaluation,
                holdout,
                opts,
            )
        )
        oxts_fit = solve_vehicle_frame(oxts, opts.solver)
        if oxts_fit.rotation is not None:
            references.append(
                reference_entry(
                    "vehicle frame from the OXTS velocities (same solver) composed with "
                    "KITTI calib_imu_to_velo",
                    oxts_fit.rotation @ vendor_rotation,
                    evaluation,
                    holdout,
                    opts,
                )
            )
    return build_vehicle_frame_artifact(
        evaluation,
        modality="lidar",
        motions=len(motions),
        references=references,
        options=opts,
        include_step=True,
        limitations=KITTI_LIMITATIONS,
        provenance=VehicleFrameProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family="kitti_raw",
            sequence_ids=[Path(path).name for path in drive_dirs],
            sensor="velodyne",
            input_sha256=hashlib.sha256("".join(digests).encode("ascii")).hexdigest(),
            dataset_license="CC BY-NC-SA 3.0",
        ),
    )


def build_vehicle_frame_artifact(
    evaluation: VehicleFrameEvaluation,
    *,
    modality: Literal["lidar", "ins"],
    motions: int,
    references: Sequence[VehicleFrameReference],
    options: VehicleFrameRunOptions,
    include_step: bool,
    limitations: Sequence[str],
    provenance: VehicleFrameProvenance,
) -> VehicleFrameRotationArtifact:
    """Assemble the schema-valid artifact from an evaluation."""

    opts = options
    result = evaluation.result
    recorded: dict[str, float | int] = {}
    if include_step:
        recorded["step_frames"] = opts.step_frames
    recorded.update(
        {
            "block_duration_s": opts.block_duration_s,
            "holdout_every": opts.holdout_every,
            "jackknife_groups": opts.jackknife_groups,
            "control_deg": opts.control_deg,
            "detection_delta_chi2": opts.detection_delta_chi2,
            "observable_rotation_std_deg": opts.observable_rotation_std_deg,
            "min_speed_mps": opts.solver.min_speed_mps,
            "velocity_sigma_mps": opts.solver.velocity_sigma_mps,
            "velocity_sigma_fraction": opts.solver.velocity_sigma_fraction,
            "rate_sigma_rps": opts.solver.rate_sigma_rps,
        }
    )
    return VehicleFrameRotationArtifact(
        sensor_modality=modality,
        solver_status=result.status,
        policy_status=evaluation.policy_status,
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_dofs=[item.name for item in evaluation.records if item.status == "estimated"],
        rotation_quat_xyzw=None
        if result.rotation is None
        else [float(value) for value in Rotation.from_matrix(result.rotation).as_quat()],
        lever_m=result.lever_m,
        dofs=list(evaluation.records),
        references=list(references),
        motions=motions,
        train_motions=evaluation.train_motions,
        holdout_motions=evaluation.holdout_motions,
        train_blocks=list(evaluation.train_blocks),
        holdout_blocks=list(evaluation.holdout_blocks),
        jackknife_fits=evaluation.jackknife_fits,
        train_median_normalized_residual=evaluation.train_median,
        holdout_median_normalized_residual=evaluation.holdout_median,
        options=recorded,
        limitations=list(limitations),
        provenance=provenance,
    )


def _chi2(
    motions: Sequence[VehicleMotion],
    rotation: FloatArray,
    lever: float,
    opts: VehicleFrameRunOptions,
) -> float:
    residuals = vehicle_residuals(motions, rotation, lever, opts.solver)
    return float(np.sum(np.minimum(residuals**2, 25.0)))


def _median(
    motions: Sequence[VehicleMotion],
    rotation: FloatArray,
    lever: float,
    opts: VehicleFrameRunOptions,
) -> float | None:
    residuals = vehicle_residuals(motions, rotation, lever, opts.solver)
    return float(np.median(np.abs(residuals))) if len(residuals) else None


def _jackknife(
    train: Sequence[VehicleMotion],
    train_blocks: Sequence[int],
    rotation: FloatArray,
    opts: VehicleFrameRunOptions,
) -> tuple[FloatArray | None, int]:
    groups = min(opts.jackknife_groups, len(train_blocks))
    if groups < 3:
        return None, 0
    deltas = []
    for members in np.array_split(np.asarray(train_blocks), groups):
        left_out = set(members.tolist())
        fit = solve_vehicle_frame(
            [item for item in train if item.block not in left_out], opts.solver, initial=rotation
        )
        if fit.rotation is None:
            return None, 0
        deltas.append(Rotation.from_matrix(fit.rotation @ rotation.T).as_rotvec())
    array = np.asarray(deltas)
    spread = np.sqrt((groups - 1) / groups * np.sum((array - array.mean(axis=0)) ** 2, axis=0))
    return np.asarray(spread, dtype=np.float64), groups


def _policy(
    records: Sequence[VehicleFrameDofRecord],
    train_median: float | None,
    holdout_median: float | None,
    opts: VehicleFrameRunOptions,
) -> tuple[VehiclePolicyStatus, tuple[str, ...]]:
    if holdout_median is None:
        return "inconclusive", ("no held-out moving intervals; holdout evidence is missing",)
    failures: list[str] = []
    warnings: list[str] = []
    if train_median is not None and holdout_median > opts.holdout_residual_ratio * max(
        train_median, 1e-9
    ):
        failures.append(
            f"held-out median normalized residual {holdout_median:.2f} exceeds "
            f"{opts.holdout_residual_ratio:g} times the training residual {train_median:.2f}"
        )
    for record in records:
        control = record.known_bad_control
        if record.status == "estimated" and control is not None and not control.detected:
            warnings.append(
                f"{record.name} is reported as estimated but a known-bad "
                f"{control.amount_deg:g} deg rotation is not detected on held-out blocks "
                f"(delta chi-square {control.holdout_delta_chi2:.1f})"
            )
    unobservable = [record.name for record in records if record.status == "unobservable"]
    notes = (
        ["not constrained by the data, so not calibrated: " + ", ".join(unobservable)]
        if unobservable
        else []
    )
    if failures:
        return "fail", tuple(failures + warnings + notes)
    if warnings:
        return "warn", tuple(warnings + notes)
    if unobservable:
        return "inconclusive", tuple(notes)
    return "pass", ("every rotation axis is estimated and held-out blocks detect every control",)


INS_LIMITATIONS: tuple[str, ...] = (
    "The vehicle frame is defined by the motion: it assumes no side slip and no "
    "vertical velocity, and roll needs turns.",
    "The INS velocity and rates are the unit's own navigation solution, rotated from "
    "KITTI's level forward/left/up frame into the body frame with each packet's roll "
    "and pitch.",
    "The OXTS unit's configured IMU-to-vehicle alignment is not published, so there is "
    "no vendor reference; the LiDAR closure compares this rotation with the "
    "independent LiDAR-to-vehicle estimate composed with calib_imu_to_velo.",
)


def run_kitti_imu_vehicle(
    drive_dirs: Sequence[str | Path],
    options: VehicleFrameRunOptions | None = None,
    *,
    lidar_vehicle_rotation: FloatArray | None = None,
    command: list[str] | None = None,
) -> VehicleFrameRotationArtifact:
    """``R_vehicle_imu`` for the KITTI OXTS unit from its own body-frame velocity and rates.

    ``lidar_vehicle_rotation`` (``R_vehicle_velodyne`` from ``lidar-vehicle``),
    if given, is composed with ``calib_imu_to_velo`` into a closure reference
    for the same rotation from an independent sensor.
    """

    opts = options or VehicleFrameRunOptions()
    if not drive_dirs:
        raise ValueError("at least one KITTI drive is required")
    motions: list[VehicleMotion] = []
    vendor: FloatArray | None = None
    digests = []
    for position, path in enumerate(drive_dirs):
        drive = load_kitti_ins_lidar_drive(path)
        if vendor is not None and not np.allclose(drive.vendor_t_imu_lidar, vendor, atol=1e-9):
            raise ValueError("drives from different calibrations cannot be pooled")
        vendor = drive.vendor_t_imu_lidar
        offset = position * 100_000
        motions += [
            VehicleMotion(
                item.block + offset,
                item.start_s,
                item.end_s,
                item.velocity_mps,
                item.angular_rate_rps,
            )
            for item in oxts_motions(path, block_duration_s=opts.block_duration_s)
        ]
        digests.append(drive.input_sha256)
    assert vendor is not None
    evaluation = evaluate_vehicle_frame(motions, opts)
    references = []
    if evaluation.result.rotation is not None and lidar_vehicle_rotation is not None:
        holdout = [item for item in motions if item.block in evaluation.holdout_blocks]
        # R_vehicle_imu = R_vehicle_velodyne R_velodyne_imu, with R_velodyne_imu = R_imu_velodyne^T.
        references.append(
            reference_entry(
                "closure: LiDAR-to-vehicle estimate composed with KITTI calib_imu_to_velo",
                lidar_vehicle_rotation @ vendor[:3, :3].T,
                evaluation,
                holdout,
                opts,
            )
        )
    return build_vehicle_frame_artifact(
        evaluation,
        modality="ins",
        motions=len(motions),
        references=references,
        options=opts,
        include_step=False,
        limitations=INS_LIMITATIONS,
        provenance=VehicleFrameProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family="kitti_raw",
            sequence_ids=[Path(path).name for path in drive_dirs],
            sensor="oxts",
            input_sha256=hashlib.sha256("".join(digests).encode("ascii")).hexdigest(),
            dataset_license="CC BY-NC-SA 3.0",
        ),
    )
