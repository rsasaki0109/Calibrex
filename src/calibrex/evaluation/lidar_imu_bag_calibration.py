"""Joint LiDAR-IMU calibration of a recorded bag on a cubic B-spline trajectory.

The front end turns a bag into the problem of
:mod:`calibrex.core.continuous_time_bspline_fit`:

1. the angular-rate method (:func:`run_livox_imu_lidar_rotation`) gives an
   independent first estimate of the rotation, clock offset and gyro bias;
2. gyro-deskewed LiDAR odometry over the longest reliable segment seeds the
   spline;
3. each map round builds a local plane map from the scans placed by the
   current spline (voxel centroids with planar kNN normals), associates sampled
   returns, each at its own per-point time, to it and refits the spline jointly
   with the IMU rotation, gyro bias, lever arm, clock offset, accelerometer
   bias and gravity; the gyro and accelerometer noise are re-estimated from the
   train residuals between rounds;
4. every third IMU sample is held out: signed per-axis known-bad controls are
   scored by the whitened held-out chi-square, and the fit's marginal
   covariance gives the block correlations.

The plane map is built from the trajectory being fitted, so trajectory errors
leak into the map and fade only over the rounds, and the formal covariance,
which holds the map fixed, understates the uncertainty.  The artifact reports
the last round's change and the difference from the angular-rate estimate next
to it, and the verdict rests on the controls, the holdout and convergence.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from calibrex import __version__
from calibrex.core.continuous_time_bspline import CubicBSplineTrajectory
from calibrex.core.continuous_time_bspline_fit import (
    CALIBRATION_BLOCKS,
    BSplineImuSamples,
    BSplineLidarImuFitOptions,
    BSplineLidarImuFitResult,
    BSplineLidarImuProblem,
    BSplinePlaneReturns,
    CalibrationBlock,
    LidarImuCalibration,
    fit_bspline_lidar_imu,
    lidar_imu_residuals,
)
from calibrex.core.lidar_imu_bag_calibration import (
    LidarImuBagBlockRecord,
    LidarImuBagCalibrationArtifact,
    LidarImuBagControlRecord,
    LidarImuBagFamily,
    LidarImuBagInitialEstimate,
    LidarImuBagPolicyStatus,
    LidarImuBagProvenance,
    LidarImuBagRound,
    LidarImuBagSeparabilityRecord,
)
from calibrex.core.progress import emit_stage
from calibrex.core.provenance import git_commit
from calibrex.core.se3_manifold import rotation_matrix_to_rotation_vector, rotation_vector_to_matrix
from calibrex.data.livox_ros2 import (
    LivoxStreamProfile,
    ScanStore,
    bag_input_digest,
)
from calibrex.evaluation.imu_lidar_rotation import (
    ImuLidarRunOptions,
    collect_livox_windows,
    run_livox_imu_lidar_rotation,
)
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, gyro_rotation_model
from calibrex.solvers.scan_to_scan_odometry import (
    ScanOdometryOptions,
    estimate_normals,
    preprocess_scan,
)

FloatArray: TypeAlias = NDArray[np.float64]

# Per-block bounds on the formal std for a block to count as estimated, and the
# signed control magnitude per axis (rotation in rad).
_STD_BOUNDS: dict[CalibrationBlock, float] = {
    "imu_rotation": math.radians(0.5),
    "gyro_bias": 0.005,
    "lever_arm": 0.05,
    "imu_clock_offset": 0.005,
    "accel_bias": 0.2,
    "gravity": 0.2,
}
_CONTROLS: tuple[tuple[CalibrationBlock, float, str, LidarImuBagFamily], ...] = (
    ("imu_rotation", math.radians(1.0), "rad", "gyro"),
    ("gyro_bias", 0.005, "rad/s", "gyro"),
    ("imu_clock_offset", 0.005, "s", "gyro"),
    ("lever_arm", 0.05, "m", "accel"),
    ("accel_bias", 0.2, "m/s^2", "accel"),
    ("gravity", 0.2, "m/s^2", "accel"),
)
_UNITS: dict[CalibrationBlock, str] = {
    "imu_rotation": "rad",
    "gyro_bias": "rad/s",
    "lever_arm": "m",
    "imu_clock_offset": "s",
    "accel_bias": "m/s^2",
    "gravity": "m/s^2",
}
LIMITATIONS = (
    "The plane map is built from the trajectory being fitted, so the formal std (map held "
    "fixed) understates the uncertainty; read it with last_round_change and "
    "rotation_difference_from_initial_deg.",
    "One segment (the longest reliable odometry window) is calibrated; other parts of the "
    "bag are not used.",
    "Accelerometer bias and gravity trade off unless the attitude varies widely; see the "
    "separability records.",
)


@dataclass(frozen=True)
class LidarImuBagOptions:
    """Segment, map, association, fit and evidence settings."""

    knot_spacing_s: float = 0.1
    max_segment_s: float = 30.0
    max_rounds: int = 8
    # Rounds that hold the IMU rotation and clock offset at the angular-rate
    # estimate (which does not depend on the map) while the trajectory and the
    # map settle; a map built from the odometry seed would otherwise drag them.
    warmup_rounds: int = 1
    rotation_convergence_deg: float = 0.005
    map_points_per_scan: int = 1500
    returns_per_scan: int = 300
    min_range_m: float = 1.0
    map_voxel_m: float = 0.15
    map_max_planarity: float = 0.1
    map_neighbours: int = 12
    association_gate_m: float = 0.3
    plane_gate_m: float = 0.1
    imu_margin_s: float = 0.05
    holdout_every: int = 3
    fit_iterations: int = 30
    control_delta_chi2: float = 9.0
    entangled_correlation: float = 0.9
    holdout_fail_whitened_rms: float = 2.0
    method_disagreement_deg: float = 1.0
    initial_sigmas: dict[str, float] = field(
        default_factory=lambda: {"plane": 0.02, "gyro": 0.01, "accel": 0.1}
    )
    seed: int = 20261011
    rotation: ImuLidarRunOptions = field(default_factory=ImuLidarRunOptions)


@dataclass(frozen=True)
class _Scan:
    time_s: float
    points: FloatArray
    offsets_s: FloatArray


def calibrate_lidar_imu_bag(
    bag: str | Path,
    profile: LivoxStreamProfile,
    options: LidarImuBagOptions | None = None,
    *,
    dataset_family: str,
    dataset_license: str,
    max_seconds: float | None = None,
    command: list[str] | None = None,
) -> LidarImuBagCalibrationArtifact:
    """Calibrate the IMU against the LiDAR of one bag; see the module docstring."""

    opts = options or LidarImuBagOptions()
    if opts.max_rounds <= opts.warmup_rounds:
        raise ValueError("max_rounds must exceed warmup_rounds so that every block is fitted")
    bag_path = Path(bag)
    rng = np.random.default_rng(opts.seed)
    # Decode the bag once: the angular-rate passes, the odometry segment and
    # the map rounds all replay the same scans.
    store = ScanStore()

    emit_stage("angular-rate initial estimate")
    initial = run_livox_imu_lidar_rotation(
        [bag_path],
        profile,
        opts.rotation,
        dataset_family=dataset_family,
        dataset_license=dataset_license,
        reference_rotation=None,
        max_seconds=max_seconds,
        scan_store=store,
    )
    if initial.rotation_quat_xyzw is None:
        raise ValueError(
            "the angular-rate estimate found no rotation (too few odometry intervals); "
            "record more rotation"
        )
    initial_rotation = Rotation.from_quat(initial.rotation_quat_xyzw).as_matrix()
    initial_bias = np.asarray(initial.gyro_bias_rps, dtype=float)

    emit_stage("gyro-deskewed odometry segment")
    imu = store.imu(bag_path, profile)
    gyro_series = GyroSeries(imu.times_s, imu.gyro_rps)
    segment_options = replace(
        opts.rotation,
        windowing=replace(opts.rotation.windowing, window_duration_s=opts.max_segment_s),
    )
    _gyro, windows = collect_livox_windows(
        [bag_path],
        profile,
        segment_options,
        rotation_model=gyro_rotation_model(
            gyro_series, initial_rotation, initial_bias, initial.time_offset_s
        ),
        max_seconds=max_seconds,
        scan_store=store,
    )
    if not windows:
        raise ValueError("no reliable LiDAR odometry segment in the bag")
    window = max(windows, key=lambda item: item.times_s[-1] - item.times_s[0])
    reference = float(window.times_s[0])
    window_times = np.asarray(window.times_s, dtype=float) - reference
    trajectory = _initial_spline(window_times, np.asarray(window.poses), opts.knot_spacing_s)
    end = trajectory.end_time_sec
    scans = _segment_scans(
        store.scans(bag_path, profile, max_seconds=max_seconds), reference, end, opts, rng
    )
    if len(scans) < 10:
        raise ValueError(f"only {len(scans)} scans fall inside the odometry segment")

    imu_times = imu.times_s - reference
    body = imu_times - initial.time_offset_s
    inside = (body > opts.imu_margin_s) & (body < end - opts.imu_margin_s)
    index = np.flatnonzero(inside)
    held = (np.arange(index.size) % opts.holdout_every) == 1
    train_index, holdout_index = index[~held], index[held]
    if train_index.size < 10 or holdout_index.size < 3:
        raise ValueError("too few IMU samples inside the odometry segment")

    sigmas = dict(opts.initial_sigmas)
    calibration = LidarImuCalibration(
        imu_rotation_vector_rad=np.asarray(
            rotation_matrix_to_rotation_vector(initial_rotation), dtype=float
        ),
        gyro_bias_rad_s=initial_bias,
        lever_arm_body_m=np.zeros(3),
        imu_clock_offset_sec=float(initial.time_offset_s),
        accel_bias_m_s2=np.zeros(3),
        gravity_world_m_s2=_initial_gravity(
            trajectory, imu_times[train_index] - initial.time_offset_s,
            imu.accel_mps2[train_index], initial_rotation,
        ),
    )

    def imu_samples(rows: NDArray[np.intp]) -> tuple[BSplineImuSamples, BSplineImuSamples]:
        return (
            BSplineImuSamples(imu_times[rows], imu.gyro_rps[rows], sigmas["gyro"]),
            BSplineImuSamples(imu_times[rows], imu.accel_mps2[rows], sigmas["accel"]),
        )

    rounds: list[LidarImuBagRound] = []
    previous = calibration
    last_change = {block: np.zeros(_width(block)) for block in CALIBRATION_BLOCKS}
    result: BSplineLidarImuFitResult | None = None
    planes: BSplinePlaneReturns | None = None
    converged = False
    for round_index in range(opts.max_rounds):
        emit_stage(f"map round {round_index + 1} of up to {opts.max_rounds}")
        planes, map_size = _associate(trajectory, scans, opts, sigmas["plane"], rng)
        gyro, accel = imu_samples(train_index)
        estimate = (
            tuple(
                block
                for block in CALIBRATION_BLOCKS
                if block not in ("imu_rotation", "imu_clock_offset")
            )
            if round_index < opts.warmup_rounds
            else CALIBRATION_BLOCKS
        )
        result = fit_bspline_lidar_imu(
            BSplineLidarImuProblem(trajectory, planes, gyro, accel, calibration, estimate),
            BSplineLidarImuFitOptions(max_iterations=opts.fit_iterations),
        )
        trajectory, calibration = result.trajectory, result.calibration
        # Re-estimate each family's noise from its train residuals.
        whitened = lidar_imu_residuals(trajectory, calibration, planes, gyro, accel)
        for family, values in whitened.items():
            sigmas[family] *= float(np.sqrt(np.mean(values * values)))
        change = _rotation_angle_deg(previous, calibration)
        last_change = {
            block: _block_difference(block, previous, calibration) for block in CALIBRATION_BLOCKS
        }
        rounds.append(
            LidarImuBagRound(
                round=round_index,
                map_planes=map_size,
                plane_returns=len(planes.times_sec),
                fit_status=result.status,
                iterations=result.iterations,
                rotation_change_deg=change,
            )
        )
        previous = calibration
        if round_index > opts.warmup_rounds and change < opts.rotation_convergence_deg:
            converged = True
            break
    assert result is not None and planes is not None

    train_gyro, train_accel = imu_samples(train_index)
    holdout_gyro, holdout_accel = imu_samples(holdout_index)
    train_rms = _whitened_rms(trajectory, calibration, planes, train_gyro, train_accel)
    holdout_rms = _whitened_rms(trajectory, calibration, planes, holdout_gyro, holdout_accel)

    def holdout_chi2(family: LidarImuBagFamily, values: LidarImuCalibration) -> float:
        residuals = lidar_imu_residuals(trajectory, values, planes, holdout_gyro, holdout_accel)
        return float(np.sum(residuals[family] ** 2))

    controls = _controls(holdout_chi2, calibration, opts.control_delta_chi2)
    blocks = _block_records(result, calibration, controls, last_change)
    separability = [
        LidarImuBagSeparabilityRecord(
            block_a=_block(left),
            block_b=_block(right),
            max_abs_correlation=value,
            entangled=value >= opts.entangled_correlation,
        )
        for (left, right), value in result.block_correlations().items()
    ]
    difference = float(
        np.degrees(
            np.linalg.norm(
                rotation_matrix_to_rotation_vector(
                    initial_rotation.T
                    @ rotation_vector_to_matrix(calibration.imu_rotation_vector_rad)
                )
            )
        )
    )
    status, reasons = _policy(
        blocks, holdout_rms, converged, difference, result.status, opts
    )
    digest, scope = bag_input_digest([bag_path])
    rotation = rotation_vector_to_matrix(calibration.imu_rotation_vector_rad)
    return LidarImuBagCalibrationArtifact(
        policy_status=status,
        policy_reasons=reasons,
        t_lidar_imu_translation_m=[float(value) for value in calibration.lever_arm_body_m],
        t_lidar_imu_rotation_quat_xyzw=[
            float(value) for value in Rotation.from_matrix(rotation).as_quat()
        ],
        imu_clock_offset_s=float(calibration.imu_clock_offset_sec),
        blocks=blocks,
        initial=LidarImuBagInitialEstimate(
            policy_status=initial.policy_status,
            rotation_vector_rad=[
                float(value) for value in rotation_matrix_to_rotation_vector(initial_rotation)
            ],
            clock_offset_s=float(initial.time_offset_s),
            gyro_bias_rad_s=[float(value) for value in initial_bias],
        ),
        rotation_difference_from_initial_deg=difference,
        segment_start_s=reference,
        segment_duration_s=float(end),
        knot_spacing_s=opts.knot_spacing_s,
        control_points=trajectory.control_point_count,
        scans=len(scans),
        train_imu_samples=int(train_index.size),
        holdout_imu_samples=int(holdout_index.size),
        noise_sigmas={family: float(value) for family, value in sigmas.items()},
        rounds=rounds,
        converged=converged,
        train_whitened_rms=train_rms,
        holdout_whitened_rms=holdout_rms,
        control_delta_chi2_threshold=opts.control_delta_chi2,
        controls=controls,
        entangled_correlation_threshold=opts.entangled_correlation,
        separability=separability,
        options={
            "knot_spacing_s": opts.knot_spacing_s,
            "max_segment_s": opts.max_segment_s,
            "max_rounds": opts.max_rounds,
            "warmup_rounds": opts.warmup_rounds,
            "rotation_convergence_deg": opts.rotation_convergence_deg,
            "map_points_per_scan": opts.map_points_per_scan,
            "returns_per_scan": opts.returns_per_scan,
            "map_voxel_m": opts.map_voxel_m,
            "map_max_planarity": opts.map_max_planarity,
            "association_gate_m": opts.association_gate_m,
            "plane_gate_m": opts.plane_gate_m,
            "holdout_every": opts.holdout_every,
            "holdout_fail_whitened_rms": opts.holdout_fail_whitened_rms,
            "method_disagreement_deg": opts.method_disagreement_deg,
            "seed": opts.seed,
            "max_seconds": max_seconds,
        },
        limitations=list(LIMITATIONS),
        provenance=LidarImuBagProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family=dataset_family,
            dataset_license=dataset_license,
            bag=bag_path.name,
            lidar_topic=profile.point_topic,
            imu_topic=profile.imu_topic,
            point_time_field=profile.point_time_field,
            input_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
            input_digest_scope=scope,
        ),
    )


def _initial_spline(
    times: FloatArray, poses: FloatArray, knot_spacing_s: float
) -> CubicBSplineTrajectory:
    """Seed control point ``k`` with the odometry pose at ``(k - 1) * spacing``."""

    count = math.floor(float(times[-1]) / knot_spacing_s) + 3
    control_times = np.clip((np.arange(count) - 1) * knot_spacing_s, times[0], times[-1])
    rotations = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(control_times).as_matrix()
    positions = np.stack(
        [np.interp(control_times, times, poses[:, axis, 3]) for axis in range(3)], axis=1
    )
    return CubicBSplineTrajectory(0.0, knot_spacing_s, rotations, positions)


def _segment_scans(
    source: Iterator[tuple[float, FloatArray, FloatArray | None]],
    reference: float,
    end: float,
    opts: LidarImuBagOptions,
    rng: np.random.Generator,
) -> list[_Scan]:
    keep = opts.map_points_per_scan + opts.returns_per_scan
    scans: list[_Scan] = []
    for header, points, offsets in source:
        start = header - reference
        times = np.zeros(len(points)) if offsets is None else np.asarray(offsets, dtype=float)
        if start < 0.0 or start + float(np.max(times, initial=0.0)) > end:
            continue
        ranges = np.linalg.norm(points, axis=1)
        valid = np.flatnonzero(np.isfinite(ranges) & (ranges >= opts.min_range_m))
        if valid.size == 0:
            continue
        chosen = rng.choice(valid, size=min(valid.size, keep), replace=False)
        scans.append(_Scan(start, np.asarray(points[chosen], dtype=float), times[chosen]))
    return scans


def _place(trajectory: CubicBSplineTrajectory, times: FloatArray, points: FloatArray) -> FloatArray:
    states = trajectory.evaluate(times)
    return np.asarray(
        np.einsum("tij,tj->ti", states.rotation_world_body, points) + states.position_world_m,
        dtype=float,
    )


def _associate(
    trajectory: CubicBSplineTrajectory,
    scans: list[_Scan],
    opts: LidarImuBagOptions,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[BSplinePlaneReturns, int]:
    """Build the plane map from the placed scans and associate sampled returns to it."""

    map_rows: list[FloatArray] = []
    samples: list[tuple[FloatArray, FloatArray]] = []
    for scan in scans:
        order = rng.permutation(len(scan.points))
        split = min(opts.returns_per_scan, len(order) // 2)
        returns, cloud = order[:split], order[split:]
        map_rows.append(_place(trajectory, scan.time_s + scan.offsets_s[cloud], scan.points[cloud]))
        samples.append((scan.time_s + scan.offsets_s[returns], scan.points[returns]))
    map_options = ScanOdometryOptions(
        voxel_size_m=opts.map_voxel_m,
        min_range_m=0.0,
        max_range_m=math.inf,
        normal_neighbours=opts.map_neighbours,
        max_planarity_ratio=opts.map_max_planarity,
    )
    centroids = preprocess_scan(np.concatenate(map_rows), map_options)
    normals, planar = estimate_normals(centroids, cKDTree(centroids), map_options)
    anchors, normals = centroids[planar], normals[planar]
    if len(anchors) < 10:
        raise ValueError("the plane map has fewer than ten planar cells")
    tree = cKDTree(anchors)
    times: list[FloatArray] = []
    bodies: list[FloatArray] = []
    matched_anchors: list[FloatArray] = []
    matched_normals: list[FloatArray] = []
    for sample_times, points in samples:
        world = _place(trajectory, sample_times, points)
        distance, nearest = tree.query(world)
        offset = np.abs(np.sum(normals[nearest] * (world - anchors[nearest]), axis=1))
        ok = (distance < opts.association_gate_m) & (offset < opts.plane_gate_m)
        times.append(sample_times[ok])
        bodies.append(points[ok])
        matched_anchors.append(anchors[nearest[ok]])
        matched_normals.append(normals[nearest[ok]])
    returns = BSplinePlaneReturns(
        np.concatenate(times),
        np.concatenate(bodies),
        np.concatenate(matched_anchors),
        np.concatenate(matched_normals),
        sigma,
    )
    return returns, len(anchors)


def _initial_gravity(
    trajectory: CubicBSplineTrajectory,
    body_times: FloatArray,
    accel: FloatArray,
    rotation: FloatArray,
) -> FloatArray:
    """Average ``-R_world_lidar R_lidar_imu f``: gravity if the motion averages out."""

    states = trajectory.evaluate(body_times)
    world = np.einsum("tij,tj->ti", states.rotation_world_body, accel @ rotation.T)
    gravity = -np.mean(world, axis=0)
    return np.asarray(gravity * (9.81 / max(float(np.linalg.norm(gravity)), 1.0e-9)), dtype=float)


def _width(block: CalibrationBlock) -> int:
    return 1 if block == "imu_clock_offset" else 3


def _rotation_angle_deg(left: LidarImuCalibration, right: LidarImuCalibration) -> float:
    return float(np.degrees(np.linalg.norm(_block_difference("imu_rotation", left, right))))


def _block_difference(
    block: CalibrationBlock, left: LidarImuCalibration, right: LidarImuCalibration
) -> FloatArray:
    if block == "imu_rotation":
        return np.asarray(
            rotation_matrix_to_rotation_vector(
                rotation_vector_to_matrix(left.imu_rotation_vector_rad).T
                @ rotation_vector_to_matrix(right.imu_rotation_vector_rad)
            ),
            dtype=float,
        )
    return np.asarray(right.vector(block) - left.vector(block), dtype=float)


def _whitened_rms(
    trajectory: CubicBSplineTrajectory,
    calibration: LidarImuCalibration,
    planes: BSplinePlaneReturns,
    gyro: BSplineImuSamples,
    accel: BSplineImuSamples,
) -> dict[str, float]:
    residuals = lidar_imu_residuals(trajectory, calibration, planes, gyro, accel)
    return {
        family: float(np.sqrt(np.mean(values * values))) for family, values in residuals.items()
    }


def _controls(
    chi2: Callable[[LidarImuBagFamily, LidarImuCalibration], float],
    calibration: LidarImuCalibration,
    threshold: float,
) -> list[LidarImuBagControlRecord]:
    baselines = {family: chi2(family, calibration) for family in ("gyro", "accel")}
    records: list[LidarImuBagControlRecord] = []
    for block, magnitude, unit, family in _CONTROLS:
        axes: tuple[int | None, ...] = (None,) if block == "imu_clock_offset" else (0, 1, 2)
        for axis in axes:
            for sign in (-1, 1):
                step = np.zeros(_width(block))
                step[0 if axis is None else axis] = sign * magnitude
                delta = chi2(family, calibration.stepped(block, step)) - baselines[family]
                records.append(
                    LidarImuBagControlRecord(
                        parameter=block,
                        axis=axis,
                        sign=sign,
                        magnitude=magnitude,
                        unit=unit,
                        family=family,
                        delta_chi2=delta,
                        detected=delta >= threshold,
                    )
                )
    return records


def _block_records(
    result: BSplineLidarImuFitResult,
    calibration: LidarImuCalibration,
    controls: list[LidarImuBagControlRecord],
    last_change: dict[CalibrationBlock, FloatArray],
) -> list[LidarImuBagBlockRecord]:
    records: list[LidarImuBagBlockRecord] = []
    for block in CALIBRATION_BLOCKS:
        std = result.block_std(block)
        own = [record for record in controls if record.parameter == block]
        detected = sum(record.detected for record in own)
        reasons = []
        if detected < len(own):
            reasons.append(f"{len(own) - detected} of {len(own)} signed controls not detected")
        if float(np.max(std)) > _STD_BOUNDS[block]:
            reasons.append(f"formal std {float(np.max(std)):.3g} above {_STD_BOUNDS[block]:g}")
        records.append(
            LidarImuBagBlockRecord(
                name=block,
                unit=_UNITS[block],
                estimate=[float(value) for value in calibration.vector(block)],
                formal_std=[float(value) for value in std],
                last_round_change=[float(value) for value in last_change[block]],
                controls_detected=detected,
                controls=len(own),
                verdict="weak" if reasons else "estimated",
                reason="; ".join(reasons) if reasons else "all signed controls detected",
            )
        )
    return records


def _policy(
    blocks: list[LidarImuBagBlockRecord],
    holdout_rms: dict[str, float],
    converged: bool,
    difference_deg: float,
    fit_status: str,
    opts: LidarImuBagOptions,
) -> tuple[LidarImuBagPolicyStatus, list[str]]:
    if fit_status == "singular_system":
        return "fail", ["the joint fit hit a singular system"]
    failing = [
        f"held-out {family} whitened RMS {holdout_rms[family]:.2f} above "
        f"{opts.holdout_fail_whitened_rms:g}: the model does not predict unseen IMU samples"
        for family in ("gyro", "accel")
        if holdout_rms[family] > opts.holdout_fail_whitened_rms
    ]
    if failing:
        return "fail", failing
    rotation = next(record for record in blocks if record.name == "imu_rotation")
    if rotation.verdict == "weak":
        return "inconclusive", [f"imu_rotation is weak: {rotation.reason}"]
    warnings = [
        f"{record.name} is weak: {record.reason}" for record in blocks if record.verdict == "weak"
    ]
    if not converged:
        warnings.append(
            "the map rounds did not converge: the rotation still moved "
            f"more than {opts.rotation_convergence_deg:g} deg in the last round"
        )
    if difference_deg > opts.method_disagreement_deg:
        warnings.append(
            f"the joint rotation differs from the angular-rate estimate by {difference_deg:.2f} deg"
        )
    if warnings:
        return "warn", warnings
    return "pass", [
        "every block is estimated with all signed controls detected on held-out IMU samples, "
        "the map rounds converged and the rotation agrees with the angular-rate estimate"
    ]


def _block(name: str) -> CalibrationBlock:
    for block in CALIBRATION_BLOCKS:
        if block == name:
            return block
    raise ValueError(f"unknown calibration block {name!r}")
