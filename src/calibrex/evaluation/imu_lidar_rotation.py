"""Evaluate the IMU-LiDAR rotation, clock offset, and gyro bias on held-out windows.

LiDAR odometry windows (see :mod:`calibrex.evaluation.odometry_windows`) give
mean angular rates between consecutive scans; the gyro gives the same rates.
Every third window is held out of the fit; an 8-group window jackknife sets
the reported std when it is larger than the analytic one; each estimated
rotation axis and the clock offset are shifted by a known-bad amount that the
held-out chi-square must detect; and a design reference (for example the
MID360 manual) is compared only after the fit.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.imu_lidar_rotation import (
    ImuLidarKnownBadControl,
    ImuLidarPolicyStatus,
    ImuLidarRotationArtifact,
    ImuLidarRotationDofRecord,
    ImuLidarRotationProvenance,
    ImuLidarWindowSummary,
)
from calibrex.core.progress import emit_stage
from calibrex.core.provenance import git_commit
from calibrex.data.livox_ros2 import (
    MID360_T_LIDAR_IMU,
    ImuSamples,
    LivoxStreamProfile,
    ScanStore,
    bag_input_digest,
    iter_livox_points,
    load_livox_imu,
    resolve_profile,
)
from calibrex.evaluation.odometry_windows import (
    OdometrySegmenter,
    WindowingOptions,
    collect_odometry_windows,
)
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import (
    ROTATION_DOFS,
    GyroSeries,
    ImuLidarRotationSolver,
    RateIntervals,
    RotationOptions,
    RotationResult,
    gyro_rotation_model,
    model_residuals,
    rate_intervals,
)
from calibrex.solvers.scan_to_scan_odometry import RotationModel

FloatArray: TypeAlias = NDArray[np.float64]
_AXES: tuple[str, str, str] = ("roll", "pitch", "yaw")
_UNITS: dict[str, Literal["deg", "s", "rad/s"]] = {
    "roll": "deg",
    "pitch": "deg",
    "yaw": "deg",
    "time_offset": "s",
    "gyro_bias_x": "rad/s",
    "gyro_bias_y": "rad/s",
    "gyro_bias_z": "rad/s",
}
RIGID_SCAN_LIMITATIONS: tuple[str, ...] = (
    "Scans were treated as rigid snapshots (deskew: none): points are not corrected for "
    "motion during the sweep, which biases the LiDAR rotation rates towards the "
    "sweep-average motion and the extrinsic by an amount that grows with the rotation rate.",
    "The reported time offset includes a driver-dependent constant (the scan header stamp "
    "may mark the start, middle or end of the sweep, about half a sweep of ~50 ms).",
)
MID360_LIMITATIONS: tuple[str, ...] = (
    "The reference rotation is the MID360 design value (IMU axes aligned with the "
    "LiDAR); it is not a measurement.",
    "Mean angular rates over one LiDAR period approximate the rotation increment; "
    "the second-order error grows with the rotation per period.",
    "The translation of T_lidar_imu is not estimated by this method.",
)


@dataclass(frozen=True)
class ImuLidarRunOptions:
    """Holdout, jackknife, controls, windowing, and solver settings."""

    holdout_every: int = 3
    jackknife_groups: int = 8
    rotation_control_deg: float = 1.0
    time_control_s: float = 0.01
    detection_delta_chi2: float = 9.0
    coverage_margin_s: float = 0.2
    gyro_deskew_passes: int = 5
    deskew_convergence_sigma: float = 0.25
    feedback_check_deg: float = 2.0
    # Windows whose accumulated reference rotation (sum of per-step angles) is
    # below this many degrees are left out of the fit, the jackknife, and the
    # holdout. 0 keeps every window (the IMU-LiDAR behaviour).
    min_window_rotation_deg: float = 0.0
    # "gyro": per-point-time deskew (LiDAR constant velocity, then gyro passes and
    # a feedback check).  "none": rigid scans, every point is taken as stamped at
    # the scan header stamp; for clouds without a per-point time field.
    deskew: Literal["gyro", "none"] = "gyro"
    windowing: WindowingOptions = field(default_factory=WindowingOptions)
    solver: RotationOptions = field(default_factory=RotationOptions)


@dataclass(frozen=True)
class ImuLidarEvaluation:
    """Dataset-independent evaluation output."""

    result: RotationResult
    records: tuple[ImuLidarRotationDofRecord, ...]
    train_windows: int
    holdout_windows: int
    jackknife_fits: int
    train_median_rps: float | None
    holdout_median_rps: float | None
    interval_count: int
    policy_status: ImuLidarPolicyStatus
    policy_reasons: tuple[str, ...]
    static_windows_excluded: int = 0


def window_rotation_deg(window: OdometryWindow) -> float:
    """Return the accumulated rotation of a window: the sum of per-step angles."""

    rotations = window.poses[:, :3, :3]
    if len(rotations) < 2:
        return 0.0
    steps = np.transpose(rotations[:-1], (0, 2, 1)) @ rotations[1:]
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(steps).as_rotvec(), axis=1).sum()))


def evaluate_imu_lidar_rotation(
    gyro: GyroSeries,
    windows: Sequence[OdometryWindow],
    options: ImuLidarRunOptions | None = None,
    *,
    reference_rotation: FloatArray | None = None,
) -> ImuLidarEvaluation:
    """Fit on train windows; collect jackknife, control, and holdout evidence."""

    opts = options or ImuLidarRunOptions()
    excluded = 0
    if opts.min_window_rotation_deg > 0.0:
        kept = [w for w in windows if window_rotation_deg(w) >= opts.min_window_rotation_deg]
        excluded = len(windows) - len(kept)
        windows = kept
    intervals = rate_intervals(windows, gyro, opts.solver)
    positions = np.arange(len(windows))
    holdout_ids = positions[positions % opts.holdout_every == 1]
    train_ids = positions[positions % opts.holdout_every != 1]
    train = intervals.subset(np.isin(intervals.window_ids, train_ids))
    holdout = intervals.subset(np.isin(intervals.window_ids, holdout_ids))
    solver = ImuLidarRotationSolver()
    result = solver.solve(gyro, train, opts.solver)
    if result.status != "converged" or result.rotation is None:
        return ImuLidarEvaluation(
            result, (), len(train_ids), len(holdout_ids), 0, None, None, len(intervals),
            "fail", ("too few LiDAR rate intervals with gyro coverage to solve",),
            excluded,
        )  # fmt: skip
    jackknife, fits = _jackknife(solver, gyro, train, train_ids, opts, result.rotation)
    reference = _reference_values(reference_rotation)
    records = tuple(
        _record(
            dof.name,
            jackknife.get(dof.name),
            result,
            gyro,
            holdout,
            reference,
            opts,
            reference_rotation,
        )
        for dof in result.dofs
    )
    train_median = _median_rate_residual(gyro, train, result, opts)
    holdout_median = _median_rate_residual(gyro, holdout, result, opts)
    status, reasons = _policy(records, train_median, holdout_median, len(holdout))
    return ImuLidarEvaluation(
        result=result,
        records=records,
        train_windows=len(train_ids),
        holdout_windows=len(holdout_ids),
        jackknife_fits=fits,
        train_median_rps=train_median,
        holdout_median_rps=holdout_median,
        interval_count=len(intervals),
        policy_status=status,
        policy_reasons=reasons,
        static_windows_excluded=excluded,
    )


def run_livox_imu_lidar_rotation(
    bags: Sequence[str | Path],
    profile: str | LivoxStreamProfile,
    options: ImuLidarRunOptions | None = None,
    *,
    dataset_family: str,
    dataset_license: str,
    reference_rotation: FloatArray | None = MID360_T_LIDAR_IMU[:3, :3],
    max_scans: int | None = None,
    max_seconds: float | None = None,
    reference: str | None = None,
    command: list[str] | None = None,
    scan_store: ScanStore | None = None,
) -> ImuLidarRotationArtifact:
    """Run odometry and the rotation evaluation on Livox ROS 2 bags of one rig.

    ``profile`` is a built-in profile name or a custom :class:`LivoxStreamProfile`
    (any PointCloud2 LiDAR with a stated per-point time field).  ``max_seconds``
    analyses only the first scans within that many seconds of each bag's first scan.
    ``reference`` names the source of a non-MID360 ``reference_rotation`` (for
    example a deployed calibration); without it the MID360 design text is recorded.
    ``scan_store`` replays decoded scans and IMU from memory across the odometry
    passes (identical results, no repeated bag reads).
    """

    opts = options or ImuLidarRunOptions()
    if not bags:
        raise ValueError("at least one bag is required")
    stream = resolve_profile(profile)
    rigid = opts.deskew == "none"
    if rigid:
        # Rigid scans: ignore any per-point time field, so no pass deskews.
        stream = replace(stream, point_time_field=None)
    samples = [_load_imu(bag, stream, scan_store) for bag in bags]
    imu = _concatenate(samples)
    gyro = GyroSeries(imu.times_s, imu.gyro_rps)
    margin = opts.coverage_margin_s
    gyro_passes = 0 if rigid else opts.gyro_deskew_passes
    max_passes = 1 + gyro_passes + (1 if gyro_passes and opts.feedback_check_deg > 0.0 else 0)
    passes_started = 0

    def covers(time_s: float) -> bool:
        return gyro.covers(time_s - margin, time_s + margin)

    def collect(model: RotationModel | None) -> OdometrySegmenter:
        nonlocal passes_started
        passes_started += 1
        emit_stage(
            f"odometry + deskew pass {passes_started} of up to {max_passes}"
            + (
                " (rigid scans)"
                if rigid
                else " (LiDAR constant-velocity)"
                if model is None
                else " (gyro)"
            )
        )
        segmenter: OdometrySegmenter | None = None
        for bag in bags:
            segmenter = collect_odometry_windows(
                _iter_scans(bag, stream, max_seconds, scan_store),
                opts.windowing,
                covers=covers,
                prefix=f"{Path(bag).name}/",
                max_scans=max_scans,
                into=segmenter,
                rotation_model=model,
            )
        assert segmenter is not None
        return segmenter

    # Pass 0 deskews with the LiDAR's own constant-velocity motion.  Later
    # passes deskew with gyro rotations mapped by the previous estimate, which
    # fixes fast hand-held rotation but feeds the extrinsic into the odometry.
    segmenter = collect(None)
    emit_stage("rotation estimation")
    evaluation = evaluate_imu_lidar_rotation(
        gyro, segmenter.windows, opts, reference_rotation=reference_rotation
    )
    passes = [_pass_record("rigid_none" if rigid else "lidar_constant_velocity", evaluation)]
    last_change: FloatArray | None = None
    for _ in range(0 if rigid else opts.gyro_deskew_passes):
        previous = evaluation.result
        if previous.rotation is None:
            break
        segmenter = collect(
            gyro_rotation_model(
                gyro, previous.rotation, previous.gyro_bias_rps, previous.time_offset_s
            )
        )
        evaluation = evaluate_imu_lidar_rotation(
            gyro, segmenter.windows, opts, reference_rotation=reference_rotation
        )
        passes.append(_pass_record("gyro", evaluation))
        last_change = _rotation_change_deg(previous, evaluation.result)
        if last_change is not None and np.all(
            last_change <= opts.deskew_convergence_sigma * _rotation_std_deg(evaluation)
        ):
            break
    if last_change is not None:
        evaluation = _gate_convergence(evaluation, last_change)
    feedback = None
    final = evaluation.result
    if (
        not rigid
        and opts.gyro_deskew_passes
        and opts.feedback_check_deg > 0.0
        and final.rotation is not None
    ):
        offset = Rotation.from_rotvec(
            np.full(3, math.radians(opts.feedback_check_deg) / math.sqrt(3.0))
        ).as_matrix()
        perturbed = offset @ final.rotation
        check = evaluate_imu_lidar_rotation(
            gyro,
            collect(
                gyro_rotation_model(gyro, perturbed, final.gyro_bias_rps, final.time_offset_s)
            ).windows,
            opts,
        )
        passes.append(_pass_record("gyro_feedback_check", check))
        if check.result.rotation is not None:
            retained = Rotation.from_matrix(final.rotation.T @ check.result.rotation).magnitude()
            feedback = float(math.degrees(retained) / opts.feedback_check_deg)
    digest, scope = bag_input_digest(bags)
    return rotation_artifact_from_evaluation(
        evaluation,
        opts,
        windows=ImuLidarWindowSummary(
            scans_read=segmenter.scans_read,
            odometry_segments=segmenter.segments,
            unreliable_registrations=segmenter.unreliable,
            windows=len(segmenter.windows),
            rate_intervals=evaluation.interval_count,
            imu_samples=len(imu.times_s),
        ),
        provenance=ImuLidarRotationProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family=dataset_family,
            sequence_ids=[Path(bag).name for bag in bags],
            stream_profile=stream.name,
            input_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
            input_digest_scope=scope,
            dataset_license=dataset_license,
        ),
        reference=None
        if reference_rotation is None
        else reference or "Livox MID360 manual: IMU axes aligned with the LiDAR frame",
        deskew_passes=passes,
        deskew_feedback_ratio=feedback,
        limitations=[
            *(
                MID360_LIMITATIONS
                if reference is None
                else (
                    "The reference rotation is a given calibration (for example the deployed "
                    "one); it is a comparison, not ground truth.",
                    *MID360_LIMITATIONS[1:],
                )
            ),
            *(RIGID_SCAN_LIMITATIONS if rigid else ()),
        ],
        extra_options={"deskew": "none"} if rigid else None,
    )


def rotation_artifact_from_evaluation(
    evaluation: ImuLidarEvaluation,
    options: ImuLidarRunOptions,
    *,
    windows: ImuLidarWindowSummary,
    provenance: ImuLidarRotationProvenance,
    reference: str | None,
    deskew_passes: list[dict[str, Any]] | None = None,
    deskew_feedback_ratio: float | None = None,
    limitations: list[str] | None = None,
    sensor_modality: Literal["lidar", "camera", "trajectory"] = "lidar",
    extra_options: dict[str, Any] | None = None,
) -> ImuLidarRotationArtifact:
    """Assemble a schema-valid rotation artifact from an evaluation."""

    opts = options
    result = evaluation.result
    return ImuLidarRotationArtifact(
        sensor_modality=sensor_modality,
        solver_status=result.status,
        policy_status=evaluation.policy_status,
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_dofs=[item.name for item in evaluation.records if item.status == "estimated"],
        rotation_quat_xyzw=None
        if result.rotation is None
        else [float(value) for value in Rotation.from_matrix(result.rotation).as_quat()],
        time_offset_s=result.time_offset_s,
        gyro_bias_rps=[float(value) for value in result.gyro_bias_rps],
        dofs=list(evaluation.records),
        options={
            "holdout_every": opts.holdout_every,
            "jackknife_groups": opts.jackknife_groups,
            "rotation_control_deg": opts.rotation_control_deg,
            "time_control_s": opts.time_control_s,
            "detection_delta_chi2": opts.detection_delta_chi2,
            "window_duration_s": opts.windowing.window_duration_s,
            "local_map_scans": opts.windowing.odometry.local_map_scans,
            "deskew_iterations": opts.windowing.odometry.deskew_iterations,
            "gyro_deskew_passes": opts.gyro_deskew_passes,
            "deskew_convergence_sigma": opts.deskew_convergence_sigma,
            "feedback_check_deg": opts.feedback_check_deg,
            "model": opts.solver.model,
            "pair_steps": list(opts.solver.pair_steps),
            "rate_sigma_floor_rps": opts.solver.rate_sigma_floor_rps,
            "rate_sigma_fraction": opts.solver.rate_sigma_fraction,
            "observable_rotation_std_deg": opts.solver.observable_rotation_std_deg,
            "observable_time_offset_std_s": opts.solver.observable_time_offset_std_s,
            **(extra_options or {}),
        },
        windows=windows,
        train_windows=evaluation.train_windows,
        holdout_windows=evaluation.holdout_windows,
        jackknife_fits=evaluation.jackknife_fits,
        train_median_rate_residual_rps=evaluation.train_median_rps,
        holdout_median_rate_residual_rps=evaluation.holdout_median_rps,
        reference=reference,
        deskew_passes=deskew_passes or [],
        deskew_feedback_ratio=deskew_feedback_ratio,
        limitations=limitations or [],
        provenance=provenance,
    )


def _rotation_change_deg(previous: RotationResult, current: RotationResult) -> FloatArray | None:
    if previous.rotation is None or current.rotation is None:
        return None
    before = Rotation.from_matrix(previous.rotation).as_euler("xyz", degrees=True)
    after = Rotation.from_matrix(current.rotation).as_euler("xyz", degrees=True)
    return np.asarray(np.abs(after - before), dtype=np.float64)


def _rotation_std_deg(evaluation: ImuLidarEvaluation) -> FloatArray:
    reported = {record.name: record.std_reported for record in evaluation.records}
    return np.array([reported.get(name, math.inf) for name in ("roll", "pitch", "yaw")])


def _gate_convergence(evaluation: ImuLidarEvaluation, change: FloatArray) -> ImuLidarEvaluation:
    """Refuse a pass while the last deskew pass still moved the rotation."""

    std = _rotation_std_deg(evaluation)
    moving = [
        f"{name} moved {delta:.3f} deg in the last gyro-deskew pass (reported std {sigma:.3f} deg)"
        for name, delta, sigma in zip(("roll", "pitch", "yaw"), change, std, strict=True)
        if delta > sigma
    ]
    if not moving or evaluation.policy_status in {"fail", "warn"}:
        return evaluation
    reasons = ("gyro-deskew iteration has not converged: " + "; ".join(moving),)
    return replace(
        evaluation,
        policy_status="warn",
        policy_reasons=reasons + evaluation.policy_reasons,
    )


def _pass_record(source: str, evaluation: ImuLidarEvaluation) -> dict[str, Any]:
    result = evaluation.result
    record: dict[str, Any] = {
        "deskew": source,
        "policy_status": evaluation.policy_status,
        "time_offset_s": result.time_offset_s,
        "holdout_median_rate_residual_rps": evaluation.holdout_median_rps,
    }
    if result.rotation is not None:
        record["rotation_rpy_deg"] = [
            float(value)
            for value in Rotation.from_matrix(result.rotation).as_euler("xyz", degrees=True)
        ]
    return record


def _concatenate(samples: Sequence[ImuSamples]) -> ImuSamples:
    times = np.concatenate([item.times_s for item in samples])
    order = np.argsort(times)
    times = times[order]
    keep = np.r_[True, np.diff(times) > 0.0]
    return ImuSamples(
        times_s=times[keep],
        gyro_rps=np.concatenate([item.gyro_rps for item in samples])[order][keep],
        accel_mps2=np.concatenate([item.accel_mps2 for item in samples])[order][keep],
    )


def _jackknife(
    solver: ImuLidarRotationSolver,
    gyro: GyroSeries,
    train: RateIntervals,
    train_ids: NDArray[np.int64],
    opts: ImuLidarRunOptions,
    full_rotation: FloatArray,
) -> tuple[dict[str, float], int]:
    """Leave-group-out spread.

    Rotation axes are measured as small rotations about the sensor's x, y, z
    axes relative to the full fit, like the analytic std and the controls, so
    the spread stays meaningful for mountings near an Euler singularity.
    """

    groups = min(opts.jackknife_groups, len(train_ids))
    if groups < 3:
        return {}, 0
    samples: dict[str, list[float]] = {name: [] for name in ROTATION_DOFS}
    for members in np.array_split(train_ids, groups):
        fit = solver.solve(gyro, train.subset(~np.isin(train.window_ids, members)), opts.solver)
        if fit.status != "converged" or fit.rotation is None:
            return {}, 0
        local = Rotation.from_matrix(fit.rotation @ full_rotation.T).as_rotvec()
        for dof in fit.dofs:
            if dof.name in _AXES:
                samples[dof.name].append(float(local[_AXES.index(dof.name)]))
            else:
                samples[dof.name].append(dof.value)
    spread: dict[str, float] = {}
    for name, values in samples.items():
        if len(values) == groups:
            array = np.array(values)
            spread[name] = float(
                math.sqrt((groups - 1) / groups * np.sum((array - array.mean()) ** 2))
            )
    return spread, groups


def rebase_rotation_reference(
    artifact: ImuLidarRotationArtifact, reference_rotation: FloatArray | None
) -> ImuLidarRotationArtifact:
    """Recompute the reference comparison of a rotation artifact for another reference.

    Only ``reference_value`` and ``error_to_reference`` of the roll, pitch and
    yaw records depend on the reference; the estimate, its spread and the
    controls do not (the reference is compared after the fit), so a cached
    artifact can be reused for a different candidate.
    """

    estimate = (
        None
        if artifact.rotation_quat_xyzw is None
        else Rotation.from_quat(artifact.rotation_quat_xyzw).as_matrix()
    )
    values = _reference_values(reference_rotation)
    records: list[ImuLidarRotationDofRecord] = []
    for record in artifact.dofs:
        value = values.get(record.name)
        error = None
        if value is not None and reference_rotation is not None and estimate is not None:
            local = Rotation.from_matrix(estimate @ reference_rotation.T).as_rotvec()
            error = math.degrees(float(local[_AXES.index(record.name)]))
        records.append(
            record.model_copy(
                update={
                    "reference_value": None if value is None else math.degrees(value),
                    "error_to_reference": error,
                }
            )
        )
    return artifact.model_copy(update={"dofs": records})


def _reference_values(rotation: FloatArray | None) -> dict[str, float]:
    if rotation is None:
        return {}
    roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
    return {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)}


def _record(
    name: str,
    jackknife: float | None,
    result: RotationResult,
    gyro: GyroSeries,
    holdout: RateIntervals,
    reference: dict[str, float],
    opts: ImuLidarRunOptions,
    reference_rotation: FloatArray | None = None,
) -> ImuLidarRotationDofRecord:
    dof = result.dof(name)  # type: ignore[arg-type]
    convert = math.degrees if _UNITS[name] == "deg" else (lambda value: value)
    reported = max(dof.std, jackknife or 0.0)
    threshold = {
        "deg": math.radians(opts.solver.observable_rotation_std_deg),
        "s": opts.solver.observable_time_offset_std_s,
        "rad/s": opts.solver.observable_gyro_bias_std_rps,
    }[_UNITS[name]]
    reference_value = reference.get(name)
    error = None
    if (
        reference_value is not None
        and reference_rotation is not None
        and result.rotation is not None
    ):
        # Small rotation about the sensor axis that takes the reference to the estimate.
        local = Rotation.from_matrix(result.rotation @ reference_rotation.T).as_rotvec()
        error = convert(float(local[_AXES.index(name)]))
    return ImuLidarRotationDofRecord(
        name=dof.name,
        unit=_UNITS[name],
        value=convert(dof.value),
        std_analytic=convert(dof.std),
        std_jackknife=None if jackknife is None else convert(jackknife),
        std_reported=convert(reported),
        status="estimated" if reported <= threshold else "unobservable",
        reference_value=None if reference_value is None else convert(reference_value),
        error_to_reference=error,
        known_bad_control=_control(name, result, gyro, holdout, opts),
    )


def _control(
    name: str,
    result: RotationResult,
    gyro: GyroSeries,
    holdout: RateIntervals,
    opts: ImuLidarRunOptions,
) -> ImuLidarKnownBadControl | None:
    if result.rotation is None or len(holdout) < 20 or name.startswith("gyro_bias"):
        return None
    rotation = result.rotation
    offset = result.time_offset_s
    if name == "time_offset":
        amount, unit = opts.time_control_s, "s"
        moved_rotation, moved_offset = rotation, offset + amount
    else:
        amount, unit = opts.rotation_control_deg, "deg"
        axis = np.zeros(3)
        axis["xyz".index({"roll": "x", "pitch": "y", "yaw": "z"}[name])] = math.radians(amount)
        moved_rotation = Rotation.from_rotvec(axis).as_matrix() @ rotation
        moved_offset = offset
    baseline = _chi2(gyro, holdout, rotation, offset, result, opts)
    moved = _chi2(gyro, holdout, moved_rotation, moved_offset, result, opts)
    delta = moved - baseline
    return ImuLidarKnownBadControl(
        amount=amount,
        unit=unit,  # type: ignore[arg-type]
        holdout_delta_chi2=float(delta),
        detected=bool(delta >= opts.detection_delta_chi2),
    )


def _chi2(
    gyro: GyroSeries,
    intervals: RateIntervals,
    rotation: FloatArray,
    offset: float,
    result: RotationResult,
    opts: ImuLidarRunOptions,
) -> float:
    residuals = model_residuals(
        gyro, intervals, rotation, result.gyro_bias_rps, offset, opts.solver
    )
    variance = result.variance_factor or 1.0
    return float(np.sum(np.minimum(residuals**2 / variance, 25.0)))


def _median_rate_residual(
    gyro: GyroSeries,
    intervals: RateIntervals,
    result: RotationResult,
    opts: ImuLidarRunOptions,
) -> float | None:
    if len(intervals) == 0 or result.rotation is None:
        return None
    errors = model_residuals(
        gyro,
        intervals,
        result.rotation,
        result.gyro_bias_rps,
        result.time_offset_s,
        opts.solver,
        normalize=False,
    ).reshape(-1, 3)
    return float(np.median(np.linalg.norm(errors, axis=1)))


def _policy(
    records: Sequence[ImuLidarRotationDofRecord],
    train_median: float | None,
    holdout_median: float | None,
    holdout_intervals: int,
) -> tuple[ImuLidarPolicyStatus, tuple[str, ...]]:
    if holdout_intervals < 20 or holdout_median is None:
        return "inconclusive", ("too few held-out rate intervals; holdout evidence is missing",)
    failures: list[str] = []
    warnings: list[str] = []
    if train_median is not None and holdout_median > max(3.0 * train_median, 0.05):
        failures.append(
            f"held-out median rate residual {holdout_median:.4f} rad/s exceeds three times "
            f"the training residual {train_median:.4f} rad/s"
        )
    for record in records:
        control = record.known_bad_control
        if record.status == "estimated" and control is not None and not control.detected:
            warnings.append(
                f"{record.name} is reported as estimated but a known-bad {control.amount:g} "
                f"{control.unit} shift is not detected on held-out windows "
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
    if any(name in {"roll", "pitch", "yaw"} for name in unobservable):
        return "inconclusive", tuple(notes)
    reasons = ["every rotation axis is estimated and held-out windows detect every control"]
    return "pass", tuple(reasons + notes)


@dataclass(frozen=True)
class CandidateScore:
    """Held-out fit of a fixed rotation and clock offset (bias refit on train)."""

    holdout_median_rate_residual_rps: float
    holdout_rms_rate_residual_rps: float
    train_windows: int
    holdout_windows: int
    holdout_intervals: int
    refit_gyro_bias_rps: FloatArray
    holdout_window_medians_rps: dict[float, float]
    holdout_spans_s: tuple[tuple[float, float], ...]


def score_imu_lidar_candidate(
    gyro: GyroSeries,
    windows: Sequence[OdometryWindow],
    rotation: FloatArray,
    time_offset_s: float,
    options: ImuLidarRunOptions | None = None,
    *,
    holdout_spans_s: Sequence[tuple[float, float]] | None = None,
) -> CandidateScore:
    """Score a candidate ``R_lidar_imu`` and ``dt`` on the standard held-out windows.

    Only the gyro bias is refit, on the train windows, because it drifts
    between recordings and is a nuisance for every method alike.  The
    held-out windows and noise model are the ones the native evaluation uses,
    so native and external candidates are scored identically.

    Each candidate recomputes the odometry with its own deskewing, which can
    shift the window segmentation.  Passing ``holdout_spans_s`` (start, end
    times of a reference segmentation's held-out windows) holds out the rate
    intervals whose midpoint falls in a span instead, so every candidate is
    scored on the same stretches of the recording.
    """

    from scipy.optimize import least_squares

    opts = options or ImuLidarRunOptions()
    solver = replace(opts.solver, estimate_time_offset=False)
    intervals = rate_intervals(windows, gyro, solver)
    positions = np.arange(len(windows))
    holdout_ids = positions[positions % opts.holdout_every == 1]
    if holdout_spans_s is None:
        spans = tuple(
            (float(windows[int(i)].times_s[0]), float(windows[int(i)].times_s[-1]))
            for i in holdout_ids
        )
    else:
        spans = tuple((float(start), float(end)) for start, end in holdout_spans_s)
    middle = 0.5 * (intervals.start_s + intervals.end_s)
    span_of = np.full(len(middle), -1, dtype=np.int64)
    for index, (start, end) in enumerate(spans):
        span_of[(middle >= start) & (middle <= end)] = index
    train = intervals.subset(span_of < 0)
    holdout = intervals.subset(span_of >= 0)
    holdout_span = span_of[span_of >= 0]
    if len(train) < 20 or len(holdout) < 20:
        raise ValueError("too few train or held-out rate intervals to score a candidate")

    def residuals(bias: FloatArray) -> FloatArray:
        return model_residuals(gyro, train, rotation, bias, time_offset_s, solver)

    bias = least_squares(residuals, np.zeros(3), loss="huber", f_scale=1.5).x
    errors = model_residuals(
        gyro, holdout, rotation, bias, time_offset_s, solver, normalize=False
    ).reshape(-1, 3)
    norms = np.linalg.norm(errors, axis=1)
    per_window = {
        round(spans[int(index)][0], 1): float(np.median(norms[holdout_span == index]))
        for index in np.unique(holdout_span)
    }
    return CandidateScore(
        holdout_median_rate_residual_rps=float(np.median(norms)),
        holdout_rms_rate_residual_rps=float(np.sqrt(np.mean(norms**2))),
        train_windows=int(len(positions) - len(holdout_ids)),
        holdout_windows=len(per_window),
        holdout_intervals=len(holdout),
        refit_gyro_bias_rps=np.asarray(bias, dtype=np.float64),
        holdout_window_medians_rps=per_window,
        holdout_spans_s=spans,
    )


def _load_imu(bag: str | Path, stream: LivoxStreamProfile, store: ScanStore | None) -> ImuSamples:
    return load_livox_imu(bag, stream) if store is None else store.imu(bag, stream)


def _iter_scans(
    bag: str | Path,
    stream: LivoxStreamProfile,
    max_seconds: float | None,
    store: ScanStore | None,
) -> Iterator[tuple[float, FloatArray, FloatArray | None]]:
    if store is None:
        return iter_livox_points(bag, stream, max_seconds=max_seconds)
    return store.scans(bag, stream, max_seconds=max_seconds)


def collect_livox_windows(
    bags: Sequence[str | Path],
    profile: str | LivoxStreamProfile,
    options: ImuLidarRunOptions | None = None,
    *,
    rotation_model: RotationModel | None = None,
    max_scans: int | None = None,
    max_seconds: float | None = None,
    scan_store: ScanStore | None = None,
) -> tuple[GyroSeries, list[OdometryWindow]]:
    """Load the gyro and the odometry windows of Livox bags (one rig)."""

    opts = options or ImuLidarRunOptions()
    stream = resolve_profile(profile)
    imu = _concatenate([_load_imu(bag, stream, scan_store) for bag in bags])
    gyro = GyroSeries(imu.times_s, imu.gyro_rps)
    margin = opts.coverage_margin_s
    segmenter: OdometrySegmenter | None = None
    for bag in bags:
        segmenter = collect_odometry_windows(
            _iter_scans(bag, stream, max_seconds, scan_store),
            opts.windowing,
            covers=lambda time_s: gyro.covers(time_s - margin, time_s + margin),
            prefix=f"{Path(bag).name}/",
            max_scans=max_scans,
            into=segmenter,
            rotation_model=rotation_model,
        )
    assert segmenter is not None
    return gyro, segmenter.windows
