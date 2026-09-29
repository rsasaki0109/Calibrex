"""Evaluate INS--LiDAR trajectory hand-eye calibration with holdout evidence.

The run estimates ``T_ins_lidar``, the clock offset, and the reference scale
from time blocks held out of evaluation, then:

* inflates each analytic standard deviation to the block-jackknife spread when
  that is larger, because trajectory errors are correlated in time and the
  analytic covariance is overconfident on real drives;
* re-classifies every DoF as estimated, prior-constrained, or unobservable
  with the reported (larger) standard deviation;
* moves each DoF by a known-bad amount and checks whether held-out blocks
  detect it, so an ``estimated`` DoF whose control goes undetected is flagged
  as an overclaim; and
* compares with a dataset reference calibration only after the fit.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.ins_lidar_hand_eye import (
    InsLidarDofRecord,
    InsLidarHandEyeArtifact,
    InsLidarHandEyeProvenance,
    InsLidarKnownBadControl,
    InsLidarOdometrySummary,
    InsLidarPolicyStatus,
    InsLidarResidualSummary,
    InsLidarTransform,
)
from calibrex.core.provenance import git_commit
from calibrex.data.kitti_ins_lidar import (
    KittiVelodyneFrame,
    load_kitti_ins_lidar_drive,
    read_velodyne_xyz,
)
from calibrex.solvers.scan_to_scan_odometry import (
    ScanOdometryOptions,
    ScanRegistration,
    odometry_from_loader,
)
from calibrex.solvers.trajectory_hand_eye_solver import (
    DOF_NAMES,
    DofEstimate,
    DofName,
    MotionResidualStats,
    ReferenceTrajectory,
    SensorMotion,
    TrajectoryHandEyeOptions,
    TrajectoryHandEyeResult,
    TrajectoryHandEyeSolver,
    motion_errors,
    motion_residual_stats,
    motion_sigmas,
    perturb,
)

FloatArray: TypeAlias = NDArray[np.float64]
_EXTRINSIC: frozenset[str] = frozenset({"roll", "pitch", "yaw", "x", "y", "z"})
_UNITS: dict[str, Literal["deg", "m", "s", "ratio"]] = {
    "roll": "deg",
    "pitch": "deg",
    "yaw": "deg",
    "x": "m",
    "y": "m",
    "z": "m",
    "time_offset": "s",
    "reference_scale": "ratio",
}
KITTI_LIMITATIONS: tuple[str, ...] = (
    "Velodyne sweeps are registered as rigid snapshots; motion within a sweep is not compensated.",
    "The KITTI calib_imu_to_velo reference is itself an estimate, not independent "
    "metrology; differences to it are not accuracy errors.",
    "OXTS positions come from a Mercator projection of GNSS/INS output and their "
    "timestamps are logging times; a reference scale and clock offset are "
    "estimated as nuisance parameters for that reason.",
    "Vehicle motion is close to planar, so the translation along the vertical "
    "rotation axis cannot be observed without a prior.",
)


@dataclass(frozen=True)
class InsLidarRunOptions:
    """Motion construction, blocking, controls, and policy."""

    step_frames: tuple[int, ...] = (5, 10, 20)
    block_duration_s: float = 5.0
    holdout_every: int = 3
    rotation_control_deg: float = 1.0
    translation_control_m: float = 0.1
    time_control_s: float = 0.02
    detection_delta_chi2: float = 9.0
    holdout_rotation_limit_deg: float = 0.5
    holdout_translation_limit_m: float = 0.3
    solver: TrajectoryHandEyeOptions = field(default_factory=TrajectoryHandEyeOptions)
    odometry: ScanOdometryOptions = field(default_factory=ScanOdometryOptions)


@dataclass(frozen=True)
class InsLidarEvaluation:
    """Everything needed to build the artifact, independent of the dataset."""

    result: TrajectoryHandEyeResult
    records: tuple[InsLidarDofRecord, ...]
    train_blocks: tuple[int, ...]
    holdout_blocks: tuple[int, ...]
    jackknife_fits: int
    train: MotionResidualStats
    holdout: MotionResidualStats
    policy_status: InsLidarPolicyStatus
    policy_reasons: tuple[str, ...]


def build_sensor_motions(
    frame_times_s: Sequence[float],
    sensor_poses: Sequence[FloatArray],
    options: InsLidarRunOptions,
    *,
    block_offset: int = 0,
    id_prefix: str = "",
) -> list[SensorMotion]:
    """Build block-contained relative motions for every configured frame step."""

    if not frame_times_s:
        return []
    start = float(frame_times_s[0])
    blocks = [
        block_offset + int((float(time) - start) // options.block_duration_s)
        for time in frame_times_s
    ]
    motions: list[SensorMotion] = []
    for step in options.step_frames:
        for index in range(0, len(frame_times_s) - step, step):
            end = index + step
            if blocks[index] != blocks[end]:
                continue
            motions.append(
                SensorMotion(
                    motion_id=f"{id_prefix}step{step}-{index}",
                    block=blocks[index],
                    start_s=float(frame_times_s[index]),
                    end_s=float(frame_times_s[end]),
                    motion=np.linalg.inv(sensor_poses[index]) @ sensor_poses[end],
                )
            )
    return motions


def evaluate_ins_lidar_hand_eye(
    reference: ReferenceTrajectory,
    motions: Sequence[SensorMotion],
    options: InsLidarRunOptions | None = None,
    *,
    reference_transform: FloatArray | None = None,
) -> InsLidarEvaluation:
    """Fit on train blocks and collect jackknife, control, and holdout evidence."""

    opts = options or InsLidarRunOptions()
    blocks = sorted({item.block for item in motions})
    holdout_blocks = tuple(
        block
        for position, block in enumerate(blocks)
        if position % opts.holdout_every == opts.holdout_every - 1
    )
    train_blocks = tuple(block for block in blocks if block not in holdout_blocks)
    train = [item for item in motions if item.block in train_blocks]
    holdout = [item for item in motions if item.block in holdout_blocks]
    solver = TrajectoryHandEyeSolver()
    result = solver.solve(reference, train, opts.solver)
    if result.status != "converged" or result.transform is None:
        return InsLidarEvaluation(
            result=result,
            records=(),
            train_blocks=train_blocks,
            holdout_blocks=holdout_blocks,
            jackknife_fits=0,
            train=result.train,
            holdout=MotionResidualStats(len(holdout), None, None, None, None),
            policy_status="fail",
            policy_reasons=("too few motions inside the reference trajectory to solve",),
        )

    jackknife = _jackknife(solver, reference, train, train_blocks, result, opts)
    holdout_stats = motion_residual_stats(
        reference, holdout, result.transform, result.time_offset_s, result.reference_scale
    )
    records = tuple(
        _record(dof, jackknife.get(dof.name), reference, holdout, result, reference_transform, opts)
        for dof in result.dofs
    )
    status, reasons = _policy(records, result.train, holdout_stats, opts)
    return InsLidarEvaluation(
        result=result,
        records=records,
        train_blocks=train_blocks,
        holdout_blocks=holdout_blocks,
        jackknife_fits=len(train_blocks) if len(train_blocks) >= 3 else 0,
        train=result.train,
        holdout=holdout_stats,
        policy_status=status,
        policy_reasons=reasons,
    )


def run_kitti_ins_lidar_hand_eye(
    drive_dirs: str | Path | Sequence[str | Path],
    options: InsLidarRunOptions | None = None,
    *,
    max_frames: int | None = None,
    command: list[str] | None = None,
) -> InsLidarHandEyeArtifact:
    """Run LiDAR odometry and the joint hand-eye evaluation on KITTI drives.

    Several drives are pooled only when they share one calibration (the same
    rig on the same day); their OXTS tracks are concatenated in time, and the
    minutes between drives are data gaps that no motion may bridge.
    """

    opts = options or InsLidarRunOptions()
    paths = [drive_dirs] if isinstance(drive_dirs, (str, Path)) else list(drive_dirs)
    if not paths:
        raise ValueError("at least one KITTI drive is required")
    drives = sorted(
        (load_kitti_ins_lidar_drive(path) for path in paths),
        key=lambda item: float(item.trajectory.times_s[0]),
    )
    vendor = drives[0].vendor_t_imu_lidar
    for drive in drives[1:]:
        if not np.allclose(drive.vendor_t_imu_lidar, vendor, atol=1e-9):
            raise ValueError(
                f"{drive.drive_dir.name} has a different calib_imu_to_velo; drives from "
                "different calibrations are different rigs and cannot be pooled"
            )
    motions: list[SensorMotion] = []
    registrations = []
    frame_count = 0
    missing: list[str] = []
    for position, drive in enumerate(drives):
        frames = drive.velodyne_frames[:max_frames] if max_frames else drive.velodyne_frames
        times = [frame.time_s for frame in frames]
        odometry = odometry_from_loader(
            _velodyne_loader(frames),
            len(frames),
            opts.odometry,
            times_s=times,
        )
        motions += build_sensor_motions(
            times,
            odometry.poses,
            opts,
            block_offset=position * 100_000,
            id_prefix=f"{drive.drive_dir.name}/",
        )
        registrations += list(odometry.registrations)
        frame_count += len(frames)
        missing += [f"{drive.drive_dir.name}/{index}" for index in drive.missing_velodyne_indices]
    times_s = np.concatenate([drive.trajectory.times_s for drive in drives])
    if np.any(np.diff(times_s) <= 0.0):
        raise ValueError("pooled KITTI drives overlap in time")
    reference = ReferenceTrajectory(
        times_s, np.concatenate([drive.trajectory.poses for drive in drives])
    )
    evaluation = evaluate_ins_lidar_hand_eye(reference, motions, opts, reference_transform=vendor)
    digest = hashlib.sha256("".join(drive.input_sha256 for drive in drives).encode("ascii"))
    return build_ins_lidar_artifact(
        evaluation,
        opts,
        odometry=_odometry_summary(registrations, frame_count, missing),
        provenance=InsLidarHandEyeProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family="kitti_raw",
            drive_ids=[drive.drive_dir.name for drive in drives],
            input_sha256=digest.hexdigest(),
            dataset_license="CC BY-NC-SA 3.0",
        ),
        reference_calibration="KITTI calib_imu_to_velo.txt (inverted to T_imu_velo)",
        limitations=KITTI_LIMITATIONS,
    )


def build_ins_lidar_artifact(
    evaluation: InsLidarEvaluation,
    options: InsLidarRunOptions,
    *,
    odometry: InsLidarOdometrySummary,
    provenance: InsLidarHandEyeProvenance,
    reference_calibration: str | None,
    limitations: Sequence[str],
) -> InsLidarHandEyeArtifact:
    """Assemble the schema-valid artifact from an evaluation."""

    result = evaluation.result
    transform = None
    if result.transform is not None:
        transform = InsLidarTransform(
            parent_frame="ins",
            child_frame="lidar",
            translation_m=[float(value) for value in result.transform[:3, 3]],
            rotation_quat_xyzw=[
                float(value) for value in Rotation.from_matrix(result.transform[:3, :3]).as_quat()
            ],
        )
    solver = options.solver
    return InsLidarHandEyeArtifact(
        solver_status=result.status,
        policy_status=evaluation.policy_status,
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_dofs=[item.name for item in evaluation.records if item.status != "unobservable"],
        transform=transform,
        time_offset_s=result.time_offset_s,
        reference_scale=result.reference_scale,
        dofs=list(evaluation.records),
        options={
            "step_frames": list(options.step_frames),
            "holdout_every": options.holdout_every,
            "rotation_control_deg": options.rotation_control_deg,
            "translation_control_m": options.translation_control_m,
            "time_control_s": options.time_control_s,
            "detection_delta_chi2": options.detection_delta_chi2,
            "estimate_time_offset": solver.estimate_time_offset,
            "estimate_reference_scale": solver.estimate_reference_scale,
            "rotation_sigma_rad": solver.rotation_sigma_rad,
            "translation_sigma_floor_m": solver.translation_sigma_floor_m,
            "translation_sigma_per_m": solver.translation_sigma_per_m,
            "observable_rotation_std_deg": solver.observable_rotation_std_deg,
            "observable_translation_std_m": solver.observable_translation_std_m,
            "observable_time_offset_std_s": solver.observable_time_offset_std_s,
            "priors": [
                {
                    "dof": prior.dof,
                    "value": prior.value,
                    "sigma": prior.sigma,
                    "source": prior.source,
                }
                for prior in solver.priors
            ],
        },
        odometry=odometry,
        block_duration_s=options.block_duration_s,
        train_blocks=list(evaluation.train_blocks),
        holdout_blocks=list(evaluation.holdout_blocks),
        jackknife_fits=evaluation.jackknife_fits,
        train=_summary(evaluation.train),
        holdout=_summary(evaluation.holdout),
        reference_calibration=reference_calibration,
        limitations=list(limitations),
        provenance=provenance,
    )


def _jackknife(
    solver: TrajectoryHandEyeSolver,
    reference: ReferenceTrajectory,
    train: Sequence[SensorMotion],
    train_blocks: Sequence[int],
    result: TrajectoryHandEyeResult,
    opts: InsLidarRunOptions,
) -> dict[str, float]:
    if len(train_blocks) < 3:
        return {}
    single_start = replace(opts.solver, yaw_starts_deg=(result.start_yaw_deg or 0.0,))
    samples: dict[str, list[float]] = {name: [] for name in DOF_NAMES}
    for block in train_blocks:
        subset = [item for item in train if item.block != block]
        fit = solver.solve(reference, subset, single_start)
        if fit.status != "converged":
            return {}
        for dof in fit.dofs:
            samples[dof.name].append(dof.value)
    count = len(train_blocks)
    spread: dict[str, float] = {}
    for name, values in samples.items():
        if len(values) == count:
            array = np.array(values)
            spread[name] = float(
                math.sqrt((count - 1) / count * np.sum((array - array.mean()) ** 2))
            )
    return spread


def _record(
    dof: DofEstimate,
    jackknife_std: float | None,
    reference: ReferenceTrajectory,
    holdout: Sequence[SensorMotion],
    result: TrajectoryHandEyeResult,
    reference_transform: FloatArray | None,
    opts: InsLidarRunOptions,
) -> InsLidarDofRecord:
    to_unit = math.degrees if _UNITS[dof.name] == "deg" else (lambda value: value)
    reported = max(dof.std_data, jackknife_std or 0.0)
    total = max(dof.std_total, jackknife_std or 0.0) if dof.status == "prior" else reported
    threshold = _threshold(dof.name, opts.solver)
    if reported <= threshold:
        status: Literal["estimated", "prior", "unobservable"] = "estimated"
    elif dof.status == "prior" and total <= threshold:
        status = "prior"
    else:
        status = "unobservable"
    reference_value = _reference_value(dof.name, reference_transform)
    return InsLidarDofRecord(
        name=dof.name,
        unit=_UNITS[dof.name],
        value=to_unit(dof.value),
        std_analytic=to_unit(dof.std_data),
        std_jackknife=None if jackknife_std is None else to_unit(jackknife_std),
        std_reported=to_unit(reported),
        status=status,
        reference_value=None if reference_value is None else to_unit(reference_value),
        error_to_reference=None
        if reference_value is None
        else to_unit(_wrapped(dof.name, dof.value - reference_value)),
        known_bad_control=_control(dof.name, reference, holdout, result, opts),
    )


def _control(
    name: DofName,
    reference: ReferenceTrajectory,
    holdout: Sequence[SensorMotion],
    result: TrajectoryHandEyeResult,
    opts: InsLidarRunOptions,
) -> InsLidarKnownBadControl | None:
    if result.transform is None or len(holdout) < 3 or name == "reference_scale":
        return None
    transform = result.transform
    offset = result.time_offset_s
    if name == "time_offset":
        amount, unit = opts.time_control_s, "s"
        offset = offset + amount
        shown = amount
    elif name in {"roll", "pitch", "yaw"}:
        amount = math.radians(opts.rotation_control_deg)
        transform = perturb(transform, name, amount)
        unit, shown = "deg", opts.rotation_control_deg
    else:
        amount = opts.translation_control_m
        transform = perturb(transform, name, amount)
        unit, shown = "m", amount
    covered = [
        item
        for item in holdout
        if all(
            reference.continuous(item.start_s + shift, item.end_s + shift)
            for shift in (result.time_offset_s, offset)
        )
    ]
    baseline = _holdout_chi2(
        reference, covered, result.transform, result, result.time_offset_s, opts
    )
    moved = _holdout_chi2(reference, covered, transform, result, offset, opts)
    delta = moved - baseline
    return InsLidarKnownBadControl(
        amount=shown,
        unit=unit,  # type: ignore[arg-type]
        holdout_delta_chi2=float(delta),
        detected=bool(delta >= opts.detection_delta_chi2),
    )


def _holdout_chi2(
    reference: ReferenceTrajectory,
    motions: Sequence[SensorMotion],
    transform: FloatArray,
    result: TrajectoryHandEyeResult,
    time_offset_s: float,
    opts: InsLidarRunOptions,
) -> float:
    """Return the held-out chi-square under the train-calibrated noise model.

    Squares are truncated at 5 sigma so one broken held-out motion cannot
    decide a control, and the train variance factor rescales the declared
    noise model to the residual spread actually observed.
    """

    if not motions:
        return 0.0
    errors = motion_errors(reference, motions, transform, time_offset_s, result.reference_scale)
    normalized = errors.reshape(-1) / motion_sigmas(motions, opts.solver)
    variance_factor = result.variance_factor or 1.0
    return float(np.sum(np.minimum(normalized**2 / variance_factor, 25.0)))


def _policy(
    records: Sequence[InsLidarDofRecord],
    train: MotionResidualStats,
    holdout: MotionResidualStats,
    opts: InsLidarRunOptions,
) -> tuple[InsLidarPolicyStatus, tuple[str, ...]]:
    failures: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    if holdout.motion_count < 3 or holdout.rotation_median_deg is None:
        return "inconclusive", ("fewer than three held-out motions; holdout evidence is missing",)
    rotation_limit = max(3.0 * (train.rotation_median_deg or 0.0), opts.holdout_rotation_limit_deg)
    translation_limit = max(
        3.0 * (train.translation_median_m or 0.0), opts.holdout_translation_limit_m
    )
    if holdout.rotation_median_deg > rotation_limit:
        failures.append(
            f"held-out rotation median {holdout.rotation_median_deg:.3f} deg exceeds "
            f"{rotation_limit:.3f} deg"
        )
    if (holdout.translation_median_m or 0.0) > translation_limit:
        failures.append(
            f"held-out translation median {holdout.translation_median_m:.3f} m exceeds "
            f"{translation_limit:.3f} m"
        )
    for record in records:
        control = record.known_bad_control
        if control is None:
            continue
        if record.status != "unobservable" and not control.detected:
            warnings.append(
                f"{record.name} is reported as {record.status} but a known-bad "
                f"{control.amount:g} {control.unit} shift is not detected on held-out blocks "
                f"(delta chi-square {control.holdout_delta_chi2:.1f} < "
                f"{opts.detection_delta_chi2:g})"
            )
    unobservable = [
        record.name
        for record in records
        if record.status == "unobservable" and record.name in _EXTRINSIC
    ]
    if unobservable:
        notes.append(
            "not constrained by the data or a prior, so not calibrated: " + ", ".join(unobservable)
        )
    if failures:
        return "fail", tuple(failures + warnings + notes)
    if warnings:
        return "warn", tuple(warnings + notes)
    if unobservable:
        return "inconclusive", tuple(notes)
    return "pass", ("every extrinsic DoF is constrained and held-out blocks detect every control",)


def _threshold(name: str, solver: TrajectoryHandEyeOptions) -> float:
    if name in {"roll", "pitch", "yaw"}:
        return math.radians(solver.observable_rotation_std_deg)
    if name in {"x", "y", "z"}:
        return solver.observable_translation_std_m
    if name == "time_offset":
        return solver.observable_time_offset_std_s
    return solver.observable_reference_scale_std


def _reference_value(name: str, reference_transform: FloatArray | None) -> float | None:
    if reference_transform is None or name not in _EXTRINSIC:
        return None
    euler = Rotation.from_matrix(reference_transform[:3, :3]).as_euler("xyz")
    values = {
        "roll": euler[0],
        "pitch": euler[1],
        "yaw": euler[2],
        "x": reference_transform[0, 3],
        "y": reference_transform[1, 3],
        "z": reference_transform[2, 3],
    }
    return float(values[name])


def _wrapped(name: str, difference: float) -> float:
    if name in {"roll", "pitch", "yaw"}:
        return math.atan2(math.sin(difference), math.cos(difference))
    return difference


def _summary(stats: MotionResidualStats) -> InsLidarResidualSummary:
    return InsLidarResidualSummary(
        motion_count=stats.motion_count,
        rotation_rmse_deg=stats.rotation_rmse_deg,
        translation_rmse_m=stats.translation_rmse_m,
        rotation_median_deg=stats.rotation_median_deg,
        translation_median_m=stats.translation_median_m,
    )


def _velodyne_loader(frames: Sequence[KittiVelodyneFrame]) -> Callable[[int], FloatArray]:
    def load(index: int) -> FloatArray:
        return read_velodyne_xyz(frames[index].path)

    return load


def _odometry_summary(
    registrations: Sequence[ScanRegistration], frame_count: int, missing: Sequence[str]
) -> InsLidarOdometrySummary:
    return InsLidarOdometrySummary(
        frame_count=frame_count,
        missing_frames=list(missing),
        registration_count=len(registrations),
        converged_registrations=sum(1 for item in registrations if item.converged),
        median_correspondences=float(np.median([item.correspondences for item in registrations]))
        if registrations
        else None,
        median_registration_rmse_m=float(np.median([item.rmse_m for item in registrations]))
        if registrations
        else None,
    )
