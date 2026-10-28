"""Synthetic joint continuous-time LiDAR-IMU calibration recovery.

One sparse fit estimates the knot path from LiDAR point-to-plane returns
together with the IMU mounting rotation ``R_body_imu``, gyro bias, lever arm,
IMU clock offset, accelerometer bias and gravity.  Unlike the per-parameter
``recover-imu-*`` runners there are no truth pose gauges: the world frame is
fixed only by the plane map.  Every third gyro window and accelerometer
sample is held out (interleaved, so the holdout spans the whole excitation);
each parameter gets signed known-bad controls on that holdout, and the fit's
marginal covariance reports which blocks the data separate.

Two trajectory models are available.  ``screw_linear`` uses the sparse knot
solver: the truth is generated from the knots themselves, the path has
piecewise-constant velocity, so accelerometer samples are kept away from the
knots and the clock offset is observed only by gyro windows that straddle a
knot.  ``cubic_bspline`` uses the C2 spline of
:mod:`calibrex.core.continuous_time_bspline_fit`: the truth is a smooth
continuous trajectory the spline can only approximate, LiDAR returns carry
their own timestamps, gyro and accelerometer samples are direct 100 Hz
factors, and every third IMU sample is held out.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, TypeVar

import numpy as np

from calibrex.core.continuous_time_bspline import CubicBSplineTrajectory, so3_log_batch
from calibrex.core.continuous_time_bspline_fit import (
    BSplineImuSamples,
    BSplineLidarImuFitOptions,
    BSplineLidarImuProblem,
    BSplinePlaneReturns,
    LidarImuCalibration,
    fit_bspline_lidar_imu,
    lidar_imu_residuals,
)
from calibrex.core.continuous_time_lidar_imu_joint import (
    ContinuousTimeLidarImuJointArtifact,
    ContinuousTimeLidarImuJointProvenance,
    LidarImuJointControlRecord,
    LidarImuJointFamily,
    LidarImuJointParameterName,
    LidarImuJointParameterRecord,
    LidarImuJointSeparabilityRecord,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    ContinuousTimeTrajectoryFitResult,
    TrajectoryImuGyroSample,
    TrajectoryImuLeverArmMeasurement,
    TrajectoryImuPreintegrationMeasurement,
    TrajectoryPointToPlaneMeasurement,
    calibration_correlation,
    fit_continuous_trajectory,
    imu_lever_arm_rmse,
    imu_preintegration_rmse,
    interpolate_pose_at,
    predicted_imu_specific_force,
)
from calibrex.core.geometry import SE3, _tuple3
from calibrex.core.provenance import git_commit
from calibrex.core.se3_manifold import (
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
    se3_exp,
    se3_log,
    so3_left_jacobian,
)

TrajectoryModel = Literal["screw_linear", "cubic_bspline"]

_SYNTHETIC_SEED = 20261010
_DURATION_SEC = 6.0
_KNOT_SPACING_SEC = 1.0
_TRUE_IMU_ROTATION_RAD = (0.04, -0.03, 0.05)
_TRUE_GYRO_BIAS_RAD_S = (0.008, -0.006, 0.01)
_TRUE_LEVER_ARM_BODY_M = (0.10, -0.06, 0.15)
_TRUE_CLOCK_OFFSET_SEC = 0.007
_TRUE_ACCEL_BIAS_M_S2 = (0.06, -0.05, 0.04)
_TRUE_GRAVITY_WORLD_M_S2 = (0.0, 0.0, -9.81)
_INITIAL_GRAVITY_WORLD_M_S2 = (0.0, 0.0, -9.7)
_PLANE_NOISE_M = 0.01
_GYRO_NOISE_RAD_S = 0.003
_GYRO_RATE_HZ = 100.0
_ACCEL_NOISE_M_S2 = 0.03
_ACCEL_KNOT_CLEARANCE_SEC = 0.05
_HOLDOUT_EVERY = 3
# A known-bad control is detected when it raises the whitened holdout chi-square
# by at least 9: three sigma along the one known perturbation direction.
_CONTROL_DELTA_CHI2 = 9.0
_ENTANGLED_CORRELATION = 0.9
_MAX_NORMALIZED_ERROR = 4.0
# Holdout RMSE with the estimated calibration against the oracle: the injected
# truth calibration on the same fitted knots.  The train windows are no
# reference because the gyro, far tighter than LiDAR, bends the knots to fit them.
_HOLDOUT_TO_ORACLE_RATIO = 1.5

# Per-block error budgets (rotation in rad), each no looser than the budget of
# the single-parameter ``recover-imu-*`` runner (gyro bias 0.03 rad/s, lever arm
# 0.03 m, clock offset 0.006 s, accelerometer bias 0.06 m/s^2, gravity
# 0.12 m/s^2).  Controls are signed and per axis.
_ERROR_BUDGETS: dict[LidarImuJointParameterName, float] = {
    "imu_rotation": math.radians(0.5),
    "gyro_bias": 0.002,
    "lever_arm": 0.03,
    "imu_clock_offset": 0.002,
    "accel_bias": 0.06,
    "gravity": 0.1,
}
_CONTROLS: tuple[tuple[LidarImuJointParameterName, float, str, LidarImuJointFamily], ...] = (
    ("imu_rotation", math.radians(1.0), "rad", "gyro"),
    ("gyro_bias", 0.005, "rad/s", "gyro"),
    ("imu_clock_offset", 0.005, "s", "gyro"),
    ("lever_arm", 0.05, "m", "accel"),
    ("accel_bias", 0.2, "m/s^2", "accel"),
    ("gravity", 0.2, "m/s^2", "accel"),
)

_PLANES: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, -1.5), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 3.0), (0.0, 0.0, -1.0)),
    ((6.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    ((-6.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ((0.0, 6.0, 0.0), (0.0, -1.0, 0.0)),
    ((0.0, -6.0, 0.0), (0.0, 1.0, 0.0)),
    ((4.0, 4.0, 0.0), (-1.0, -1.0, 0.0)),
    ((-4.0, 4.0, 1.0), (0.6, -0.6, -0.53)),
)


def run_synthetic_lidar_imu_joint_recovery(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    recovery_id: str = "ct-lidar-imu-joint-synthetic",
    trajectory: TrajectoryModel = "screw_linear",
) -> ContinuousTimeLidarImuJointArtifact:
    """Jointly recover LiDAR-IMU calibration with holdout, controls and separability."""

    if trajectory == "cubic_bspline":
        return _run_cubic_bspline(seed=seed, command=command, recovery_id=recovery_id)
    rng = np.random.default_rng(seed)
    timestamps = tuple(
        float(value)
        for value in np.arange(0.0, _DURATION_SEC + 1.0e-9, _KNOT_SPACING_SEC)
    )
    truth = tuple(_truth_pose(timestamp) for timestamp in timestamps)
    lidar = _lidar_measurements(truth, timestamps, rng)
    gyro = _gyro_windows(truth, timestamps, rng)
    accel = _accel_samples(truth, timestamps, rng)
    gyro_train, gyro_holdout = _interleaved_split(gyro)
    accel_train, accel_holdout = _interleaved_split(accel)
    initial_knots = tuple(
        se3_exp(
            knot,
            np.concatenate([rng.normal(0.0, 0.01, 3), rng.normal(0.0, 0.005, 3)]),
        )
        for knot in truth
    )
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps,
        initial_knot_poses=initial_knots,
        point_to_plane_measurements=lidar,
        imu_preintegration_measurements=gyro_train,
        imu_lever_arm_measurements=accel_train,
        initial_gravity_world_m_s2=_INITIAL_GRAVITY_WORLD_M_S2,
        estimate_imu_rotation=True,
        estimate_gyro_bias=True,
        estimate_lever_arm=True,
        estimate_imu_clock_offset=True,
        estimate_accel_bias=True,
        estimate_gravity=True,
    )
    result = fit_continuous_trajectory(
        problem, ContinuousTimeTrajectoryFitOptions(max_iterations=60)
    )
    estimates = _estimates(result)
    train_gyro = _gyro_rmse(gyro_train, timestamps, result, estimates)
    holdout_gyro = _gyro_rmse(gyro_holdout, timestamps, result, estimates)
    train_accel = _accel_rmse(accel_train, timestamps, result, estimates)
    holdout_accel = _accel_rmse(accel_holdout, timestamps, result, estimates)
    oracle = _truth_estimates()
    oracle_gyro = _gyro_rmse(gyro_holdout, timestamps, result, oracle)
    oracle_accel = _accel_rmse(accel_holdout, timestamps, result, oracle)
    if result.calibration_covariance is None:
        raise ValueError("joint LiDAR-IMU recovery produced no covariance")
    parameters = _parameter_records(
        result.calibration_parameter_names,
        np.asarray(result.calibration_covariance, dtype=float),
        estimates,
    )

    def holdout_score(family: LidarImuJointFamily, values: _Estimates) -> float:
        if family == "gyro":
            return _gyro_rmse(gyro_holdout, timestamps, result, values)
        return _accel_rmse(accel_holdout, timestamps, result, values)

    controls = _controls(
        holdout_score,
        estimates,
        baselines={"gyro": holdout_gyro, "accel": holdout_accel},
        chi2_scales={
            "gyro": gyro_holdout[0].weight ** 2 * 3 * len(gyro_holdout),
            "accel": accel_holdout[0].weight ** 2 * 3 * len(accel_holdout),
        },
    )
    separability = _separability(calibration_correlation(result))
    knot_translation = max(
        float(np.linalg.norm(se3_log(estimate, knot)[:3]))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    )
    knot_rotation = max(
        math.degrees(float(np.linalg.norm(se3_log(estimate, knot)[3:])))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    )
    policy_status, policy_reason = _policy(
        fit_status=result.status,
        parameters=parameters,
        holdout_gyro=holdout_gyro,
        oracle_gyro=oracle_gyro,
        holdout_accel=holdout_accel,
        oracle_accel=oracle_accel,
        controls=controls,
        separability=separability,
    )
    return ContinuousTimeLidarImuJointArtifact(
        recovery_id=recovery_id,
        data_source="synthetic",
        interpolation="screw_linear",
        seed=seed,
        knot_count=len(timestamps),
        knot_spacing_sec=_KNOT_SPACING_SEC,
        gyro_residual="preintegrated_window_rad",
        holdout_scheme="interleaved: every third gyro window and accelerometer sample",
        lidar_measurement_count=len(lidar),
        train_gyro_count=len(gyro_train),
        holdout_gyro_count=len(gyro_holdout),
        train_accel_count=len(accel_train),
        holdout_accel_count=len(accel_holdout),
        fit_status=result.status,
        fit_diagnostics=result.diagnostics,
        iterations=result.iterations,
        max_trajectory_translation_error_m=knot_translation,
        max_trajectory_rotation_error_deg=knot_rotation,
        parameters=parameters,
        train_gyro_rmse=train_gyro,
        holdout_gyro_rmse=holdout_gyro,
        train_accel_rmse_m_s2=train_accel,
        holdout_accel_rmse_m_s2=holdout_accel,
        oracle_holdout_gyro_rmse=oracle_gyro,
        oracle_holdout_accel_rmse_m_s2=oracle_accel,
        holdout_to_oracle_ratio_limit=_HOLDOUT_TO_ORACLE_RATIO,
        control_delta_chi2_threshold=_CONTROL_DELTA_CHI2,
        controls=controls,
        entangled_correlation_threshold=_ENTANGLED_CORRELATION,
        separability=separability,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeLidarImuJointProvenance(
            generator=__name__,
            generator_version="0.1",
            git_commit=git_commit(),
            command=command or [],
            seed=seed,
        ),
    )


_Estimates = dict[LidarImuJointParameterName, np.ndarray]
_T = TypeVar("_T")


def _interleaved_split(items: tuple[_T, ...]) -> tuple[tuple[_T, ...], tuple[_T, ...]]:
    train = tuple(item for index, item in enumerate(items) if index % _HOLDOUT_EVERY != 1)
    holdout = tuple(item for index, item in enumerate(items) if index % _HOLDOUT_EVERY == 1)
    return train, holdout


def _truth_pose(timestamp: float) -> SE3:
    translation = np.array(
        [
            0.6 * math.sin(0.5 * timestamp) + 0.1 * timestamp,
            0.5 * math.sin(0.35 * timestamp + 0.2),
            0.15 * math.sin(0.8 * timestamp),
        ]
    )
    rotation = np.array(
        [
            0.9 * math.sin(1.3 * timestamp),
            0.8 * math.sin(1.1 * timestamp + 0.4),
            1.2 * math.sin(0.6 * timestamp),
        ]
    )
    return se3_exp(SE3.identity(), np.concatenate([translation, rotation]))


def _lidar_measurements(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[TrajectoryPointToPlaneMeasurement, ...]:
    planes = [
        (np.asarray(point, dtype=float), np.asarray(normal, dtype=float))
        for point, normal in _PLANES
    ]
    planes = [(point, normal / float(np.linalg.norm(normal))) for point, normal in planes]
    measurements: list[TrajectoryPointToPlaneMeasurement] = []
    for scan_index, time in enumerate(np.arange(0.01, _DURATION_SEC - 0.01, 0.05)):
        pose_inverse = interpolate_pose_at(knots, timestamps, float(time)).inverse()
        for return_index in range(4):
            plane_point, normal = planes[int(rng.integers(len(planes)))]
            tangent = np.cross(normal, np.array([0.3, 0.5, 0.8]))
            tangent /= float(np.linalg.norm(tangent))
            bitangent = np.cross(normal, tangent)
            world = (
                plane_point
                + rng.uniform(-3.0, 3.0) * tangent
                + rng.uniform(-3.0, 3.0) * bitangent
                + rng.normal(0.0, _PLANE_NOISE_M) * normal
            )
            measurements.append(
                TrajectoryPointToPlaneMeasurement(
                    measurement_id=f"lidar-{scan_index:04d}-{return_index}",
                    timestamp_sec=float(time),
                    point_body_m=pose_inverse.transform_point(_tuple3(world)),
                    plane_point_world_m=_tuple3(plane_point),
                    plane_normal_world=_tuple3(normal),
                    weight=1.0 / _PLANE_NOISE_M,
                )
            )
    return tuple(measurements)


def _body_rate(
    knots: tuple[SE3, ...], timestamps: tuple[float, ...], time: float
) -> np.ndarray:
    start = min(max(time, timestamps[0]), timestamps[-1] - 1.0e-4)
    return np.asarray(
        se3_log(
            interpolate_pose_at(knots, timestamps, start),
            interpolate_pose_at(knots, timestamps, start + 1.0e-4),
        )[3:]
        / 1.0e-4,
        dtype=float,
    )


def _gyro_windows(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[TrajectoryImuPreintegrationMeasurement, ...]:
    imu_rotation = rotation_vector_to_matrix(np.asarray(_TRUE_IMU_ROTATION_RAD))
    bias = np.asarray(_TRUE_GYRO_BIAS_RAD_S, dtype=float)
    step = 1.0 / _GYRO_RATE_HZ
    windows: list[TrajectoryImuPreintegrationMeasurement] = []
    # Windows alternate between straddling a knot and lying inside an interval.
    for index, start in enumerate(np.arange(0.3, _DURATION_SEC - 0.5, 0.5)):
        samples = []
        for body_time in np.arange(start, start + 0.4 + 1.0e-9, step):
            omega = (
                imu_rotation.T @ _body_rate(knots, timestamps, float(body_time))
                + bias
                + rng.normal(0.0, _GYRO_NOISE_RAD_S, 3)
            )
            samples.append(
                TrajectoryImuGyroSample(
                    timestamp_sec=float(body_time + _TRUE_CLOCK_OFFSET_SEC),
                    omega_body_rad_s=_tuple3(omega),
                )
            )
        # Integrated white gyro noise over the window, as square-root information.
        sigma = _GYRO_NOISE_RAD_S * step * math.sqrt(len(samples))
        windows.append(
            TrajectoryImuPreintegrationMeasurement(
                measurement_id=f"gyro-{index:03d}",
                gyro_samples=tuple(samples),
                weight=1.0 / sigma,
            )
        )
    return tuple(windows)


def _accel_samples(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[TrajectoryImuLeverArmMeasurement, ...]:
    samples: list[TrajectoryImuLeverArmMeasurement] = []
    for body_time in np.arange(0.2, _DURATION_SEC - 0.2, 0.1):
        time = float(body_time)
        if min(abs(time - knot) for knot in timestamps) <= _ACCEL_KNOT_CLEARANCE_SEC:
            continue
        force = predicted_imu_specific_force(
            knots,
            timestamps,
            time,
            _TRUE_LEVER_ARM_BODY_M,
            accel_bias_body_m_s2=_TRUE_ACCEL_BIAS_M_S2,
            gravity_world_m_s2=_TRUE_GRAVITY_WORLD_M_S2,
            imu_rotation_vector_rad=_TRUE_IMU_ROTATION_RAD,
        ) + rng.normal(0.0, _ACCEL_NOISE_M_S2, 3)
        samples.append(
            TrajectoryImuLeverArmMeasurement(
                measurement_id=f"accel-{len(samples):04d}",
                timestamp_sec=time + _TRUE_CLOCK_OFFSET_SEC,
                accel_body_m_s2=_tuple3(force),
                weight=1.0 / _ACCEL_NOISE_M_S2,
            )
        )
    return tuple(samples)


def _estimates(result: ContinuousTimeTrajectoryFitResult) -> _Estimates:
    values = {
        "imu_rotation": result.imu_rotation_vector_rad,
        "gyro_bias": result.gyro_bias_rad_s,
        "lever_arm": result.lever_arm_body_m,
        "accel_bias": result.accel_bias_body_m_s2,
        "gravity": result.gravity_world_m_s2,
    }
    estimates: _Estimates = {}
    for name, value in values.items():
        if value is None:
            raise ValueError(f"joint LiDAR-IMU recovery produced no {name}")
        estimates[_block_name(name)] = np.asarray(value, dtype=float)
    if result.imu_clock_offset_sec is None:
        raise ValueError("joint LiDAR-IMU recovery produced no clock offset")
    estimates["imu_clock_offset"] = np.asarray([result.imu_clock_offset_sec], dtype=float)
    return estimates


def _truth_estimates() -> _Estimates:
    return {
        "imu_rotation": np.asarray(_TRUE_IMU_ROTATION_RAD, dtype=float),
        "gyro_bias": np.asarray(_TRUE_GYRO_BIAS_RAD_S, dtype=float),
        "lever_arm": np.asarray(_TRUE_LEVER_ARM_BODY_M, dtype=float),
        "imu_clock_offset": np.asarray([_TRUE_CLOCK_OFFSET_SEC], dtype=float),
        "accel_bias": np.asarray(_TRUE_ACCEL_BIAS_M_S2, dtype=float),
        "gravity": np.asarray(_TRUE_GRAVITY_WORLD_M_S2, dtype=float),
    }


def _gyro_rmse(
    measurements: tuple[TrajectoryImuPreintegrationMeasurement, ...],
    timestamps: tuple[float, ...],
    result: ContinuousTimeTrajectoryFitResult,
    estimates: _Estimates,
) -> float:
    value = imu_preintegration_rmse(
        measurements,
        timestamps,
        result.knot_poses,
        estimates["gyro_bias"],
        float(estimates["imu_clock_offset"][0]),
        imu_rotation_vector_rad=estimates["imu_rotation"],
    )
    if value is None:
        raise ValueError("joint LiDAR-IMU recovery has no gyro windows to score")
    return value


def _accel_rmse(
    measurements: tuple[TrajectoryImuLeverArmMeasurement, ...],
    timestamps: tuple[float, ...],
    result: ContinuousTimeTrajectoryFitResult,
    estimates: _Estimates,
) -> float:
    value = imu_lever_arm_rmse(
        measurements,
        timestamps,
        result.knot_poses,
        estimates["lever_arm"],
        float(estimates["imu_clock_offset"][0]),
        estimates["accel_bias"],
        estimates["gravity"],
        imu_rotation_vector_rad=estimates["imu_rotation"],
    )
    if value is None:
        raise ValueError("joint LiDAR-IMU recovery has no accelerometer samples to score")
    return value


def _parameter_records(
    names: tuple[str, ...],
    covariance: np.ndarray,
    estimates: _Estimates,
) -> list[LidarImuJointParameterRecord]:
    stds: dict[str, list[float]] = {}
    for index, label in enumerate(names):
        stds.setdefault(label.split("[", 1)[0], []).append(
            math.sqrt(max(float(covariance[index, index]), 0.0))
        )
    truths: dict[LidarImuJointParameterName, tuple[str, tuple[float, ...], tuple[float, ...]]] = {
        "imu_rotation": ("rad", _TRUE_IMU_ROTATION_RAD, (0.0, 0.0, 0.0)),
        "gyro_bias": ("rad/s", _TRUE_GYRO_BIAS_RAD_S, (0.0, 0.0, 0.0)),
        "lever_arm": ("m", _TRUE_LEVER_ARM_BODY_M, (0.0, 0.0, 0.0)),
        "imu_clock_offset": ("s", (_TRUE_CLOCK_OFFSET_SEC,), (0.0,)),
        "accel_bias": ("m/s^2", _TRUE_ACCEL_BIAS_M_S2, (0.0, 0.0, 0.0)),
        "gravity": ("m/s^2", _TRUE_GRAVITY_WORLD_M_S2, _INITIAL_GRAVITY_WORLD_M_S2),
    }
    records: list[LidarImuJointParameterRecord] = []
    for name, (unit, truth, initial) in truths.items():
        estimate = estimates[name]
        if name == "imu_rotation":
            error = rotation_matrix_to_rotation_vector(
                rotation_vector_to_matrix(np.asarray(truth)).T
                @ rotation_vector_to_matrix(estimate)
            )
        else:
            error = estimate - np.asarray(truth, dtype=float)
        std = stds[name]
        normalized = max(
            abs(float(value)) / max(sigma, 1.0e-300)
            for value, sigma in zip(error, std, strict=True)
        )
        error_norm = float(np.linalg.norm(error))
        records.append(
            LidarImuJointParameterRecord(
                name=name,
                unit=unit,
                truth=[float(value) for value in truth],
                initial=[float(value) for value in initial],
                estimate=[float(value) for value in estimate],
                error=[float(value) for value in error],
                std=std,
                error_norm=error_norm,
                error_budget=_ERROR_BUDGETS[name],
                max_normalized_error=normalized,
                within_budget=error_norm < _ERROR_BUDGETS[name],
            )
        )
    return records


def _controls(
    score: Callable[[LidarImuJointFamily, _Estimates], float],
    estimates: _Estimates,
    *,
    baselines: dict[LidarImuJointFamily, float],
    chi2_scales: dict[LidarImuJointFamily, float],
) -> list[LidarImuJointControlRecord]:
    """Signed per-axis controls scored by holdout RMSE of the affected family.

    Every holdout residual in a family shares one weight ``w``, so the whitened
    chi-square is ``w^2 * n * rmse^2`` and ``chi2_scales`` holds ``w^2 * n``.
    """

    records: list[LidarImuJointControlRecord] = []
    for name, magnitude, unit, family in _CONTROLS:
        axes: tuple[int | None, ...] = (None,) if name == "imu_clock_offset" else (0, 1, 2)
        for axis in axes:
            for sign in (-1, 1):
                perturbed = _perturb(estimates, name, axis, sign * magnitude)
                value = score(family, perturbed)
                baseline = baselines[family]
                delta_chi2 = chi2_scales[family] * (value * value - baseline * baseline)
                records.append(
                    LidarImuJointControlRecord(
                        parameter=name,
                        axis=axis,
                        sign=sign,
                        magnitude=magnitude,
                        unit=unit,
                        family=family,
                        baseline_holdout_rmse=baseline,
                        perturbed_holdout_rmse=value,
                        delta_chi2=delta_chi2,
                        detected=delta_chi2 >= _CONTROL_DELTA_CHI2,
                    )
                )
    return records


def _perturb(
    estimates: _Estimates,
    name: LidarImuJointParameterName,
    axis: int | None,
    step: float,
) -> _Estimates:
    perturbed = {key: np.array(value, copy=True) for key, value in estimates.items()}
    if name == "imu_rotation":
        tangent = np.zeros(3, dtype=float)
        tangent[0 if axis is None else axis] = step
        perturbed[name] = rotation_matrix_to_rotation_vector(
            rotation_vector_to_matrix(estimates[name]) @ rotation_vector_to_matrix(tangent)
        )
    else:
        perturbed[name][0 if axis is None else axis] += step
    return perturbed


def _block_name(name: str) -> LidarImuJointParameterName:
    names: tuple[LidarImuJointParameterName, ...] = (
        "imu_rotation",
        "gyro_bias",
        "lever_arm",
        "imu_clock_offset",
        "accel_bias",
        "gravity",
    )
    for candidate in names:
        if candidate == name:
            return candidate
    raise ValueError(f"unknown calibration block {name!r}")


def _policy(
    *,
    fit_status: str,
    parameters: list[LidarImuJointParameterRecord],
    holdout_gyro: float,
    oracle_gyro: float,
    holdout_accel: float,
    oracle_accel: float,
    controls: list[LidarImuJointControlRecord],
    separability: list[LidarImuJointSeparabilityRecord],
) -> tuple[Literal["pass", "fail"], str]:
    if fit_status == "singular_system":
        return "fail", "the joint fit hit a singular system"
    for record in parameters:
        if not record.within_budget:
            return (
                "fail",
                f"{record.name} error {record.error_norm:.4g} {record.unit} exceeds the "
                f"{record.error_budget:.4g} budget",
            )
        if record.max_normalized_error >= _MAX_NORMALIZED_ERROR:
            return (
                "fail",
                f"{record.name} error is {record.max_normalized_error:.1f} sigma from truth; "
                f"the reported std is overconfident (limit {_MAX_NORMALIZED_ERROR:.0f})",
            )
    if holdout_gyro > _HOLDOUT_TO_ORACLE_RATIO * oracle_gyro:
        return (
            "fail",
            f"holdout gyro RMSE {holdout_gyro:.3g} rad exceeds "
            f"{_HOLDOUT_TO_ORACLE_RATIO:.1f}x the truth-calibration {oracle_gyro:.3g} rad",
        )
    if holdout_accel > _HOLDOUT_TO_ORACLE_RATIO * oracle_accel:
        return (
            "fail",
            f"holdout specific-force RMSE {holdout_accel:.3g} m/s^2 exceeds "
            f"{_HOLDOUT_TO_ORACLE_RATIO:.1f}x the truth-calibration {oracle_accel:.3g} m/s^2",
        )
    missed = [record for record in controls if not record.detected]
    if missed:
        labels = ", ".join(
            f"{record.parameter}"
            + ("" if record.axis is None else f"[{record.axis}]")
            + f" {'+' if record.sign > 0 else '-'}{record.magnitude:g} {record.unit}"
            for record in missed
        )
        return (
            "fail",
            f"{len(missed)} of {len(controls)} signed controls were not detected on "
            f"holdout: {labels}",
        )
    entangled = [
        f"{record.block_a}/{record.block_b} ({record.max_abs_correlation:.2f})"
        for record in separability
        if record.entangled
    ]
    note = (
        "; entangled blocks reported, not separated by this excitation: "
        + ", ".join(entangled)
        if entangled
        else "; every block pair stays below the entanglement threshold"
    )
    return (
        "pass",
        "joint LiDAR-IMU recovery meets every error budget within "
        f"{_MAX_NORMALIZED_ERROR:.0f} sigma, holdout RMSE stays within "
        f"{_HOLDOUT_TO_ORACLE_RATIO:.1f}x the truth calibration, and all "
        f"{len(controls)} signed controls "
        "are detected" + note,
    )


_BSPLINE_KNOT_SPACING_SEC = 0.1
_BSPLINE_IMU_RATE_HZ = 100.0
_BSPLINE_LIDAR_RATE_HZ = 1000.0
_BSPLINE_IMU_MARGIN_SEC = 0.1
_BSPLINE_ERROR_GRID_SEC = 0.05


def _run_cubic_bspline(
    *,
    seed: int,
    command: list[str] | None,
    recovery_id: str,
) -> ContinuousTimeLidarImuJointArtifact:
    rng = np.random.default_rng(seed)
    planes = _bspline_plane_returns(rng)
    body_times = np.arange(
        _BSPLINE_IMU_MARGIN_SEC,
        _DURATION_SEC - _BSPLINE_IMU_MARGIN_SEC,
        1.0 / _BSPLINE_IMU_RATE_HZ,
    )
    gyro_values, accel_values = _bspline_imu_values(body_times, rng)
    imu_times = body_times + _TRUE_CLOCK_OFFSET_SEC
    # Every third IMU sample: a 0.1 s spline cannot follow single samples, so a
    # held-out sample tests the calibration on the spline its neighbours shaped.
    # Holding out whole blocks instead leaves the spline there to LiDAR alone,
    # whose doubly differentiated position swamps the accelerometer controls.
    held = (np.arange(body_times.size) % _HOLDOUT_EVERY) == 1
    gyro_train = BSplineImuSamples(imu_times[~held], gyro_values[~held], _GYRO_NOISE_RAD_S)
    accel_train = BSplineImuSamples(imu_times[~held], accel_values[~held], _ACCEL_NOISE_M_S2)
    gyro_holdout = BSplineImuSamples(imu_times[held], gyro_values[held], _GYRO_NOISE_RAD_S)
    accel_holdout = BSplineImuSamples(imu_times[held], accel_values[held], _ACCEL_NOISE_M_S2)
    count = round(_DURATION_SEC / _BSPLINE_KNOT_SPACING_SEC) + 3
    control_times = (np.arange(count) - 1) * _BSPLINE_KNOT_SPACING_SEC
    initial = CubicBSplineTrajectory(
        start_time_sec=0.0,
        knot_spacing_sec=_BSPLINE_KNOT_SPACING_SEC,
        rotations=np.asarray(
            [
                _smooth_rotation(float(time)) @ rotation_vector_to_matrix(rng.normal(0.0, 0.008, 3))
                for time in control_times
            ]
        ),
        positions=np.asarray(
            [_smooth_position(float(time)) + rng.normal(0.0, 0.01, 3) for time in control_times]
        ),
    )
    zero = np.zeros(3, dtype=float)
    result = fit_bspline_lidar_imu(
        BSplineLidarImuProblem(
            initial_trajectory=initial,
            planes=planes,
            gyro=gyro_train,
            accel=accel_train,
            initial_calibration=LidarImuCalibration(
                imu_rotation_vector_rad=zero,
                gyro_bias_rad_s=zero,
                lever_arm_body_m=zero,
                imu_clock_offset_sec=0.0,
                accel_bias_m_s2=zero,
                gravity_world_m_s2=np.asarray(_INITIAL_GRAVITY_WORLD_M_S2, dtype=float),
            ),
        ),
        BSplineLidarImuFitOptions(max_iterations=60),
    )
    estimates = _estimates_from_calibration(result.calibration)

    def score(samples: tuple[BSplineImuSamples, BSplineImuSamples]) -> Callable[
        [LidarImuJointFamily, _Estimates], float
    ]:
        gyro_samples, accel_samples = samples

        def rmse(family: LidarImuJointFamily, values: _Estimates) -> float:
            residuals = lidar_imu_residuals(
                result.trajectory,
                _calibration_from_estimates(values),
                planes,
                gyro_samples,
                accel_samples,
            )
            sigma = gyro_samples.sigma if family == "gyro" else accel_samples.sigma
            return sigma * float(np.sqrt(np.mean(residuals[family] ** 2)))

        return rmse

    holdout_score = score((gyro_holdout, accel_holdout))
    train_score = score((gyro_train, accel_train))
    oracle = _truth_estimates()
    holdout_gyro = holdout_score("gyro", estimates)
    holdout_accel = holdout_score("accel", estimates)
    oracle_gyro = holdout_score("gyro", oracle)
    oracle_accel = holdout_score("accel", oracle)
    parameters = _parameter_records(
        result.calibration_parameter_names, result.calibration_covariance, estimates
    )
    controls = _controls(
        holdout_score,
        estimates,
        baselines={"gyro": holdout_gyro, "accel": holdout_accel},
        chi2_scales={
            "gyro": 3 * len(gyro_holdout.times_sec) / gyro_holdout.sigma**2,
            "accel": 3 * len(accel_holdout.times_sec) / accel_holdout.sigma**2,
        },
    )
    separability = _separability(result.block_correlations())
    grid = np.arange(0.0, result.trajectory.end_time_sec, _BSPLINE_ERROR_GRID_SEC)
    states = result.trajectory.evaluate(grid)
    truth_rotations = np.asarray([_smooth_rotation(float(time)) for time in grid])
    truth_positions = np.asarray([_smooth_position(float(time)) for time in grid])
    rotation_errors = so3_log_batch(
        np.einsum("tji,tjk->tik", truth_rotations, states.rotation_world_body)
    )
    translation_error = float(
        np.max(np.linalg.norm(states.position_world_m - truth_positions, axis=1))
    )
    rotation_error = math.degrees(float(np.max(np.linalg.norm(rotation_errors, axis=1))))
    policy_status, policy_reason = _policy(
        fit_status=result.status,
        parameters=parameters,
        holdout_gyro=holdout_gyro,
        oracle_gyro=oracle_gyro,
        holdout_accel=holdout_accel,
        oracle_accel=oracle_accel,
        controls=controls,
        separability=separability,
    )
    return ContinuousTimeLidarImuJointArtifact(
        recovery_id=recovery_id,
        data_source="synthetic",
        interpolation="cubic_bspline",
        seed=seed,
        knot_count=result.trajectory.control_point_count,
        knot_spacing_sec=_BSPLINE_KNOT_SPACING_SEC,
        gyro_residual="angular_rate_rad_s",
        holdout_scheme="interleaved: every third gyro and accelerometer sample",
        lidar_measurement_count=len(planes.times_sec),
        train_gyro_count=len(gyro_train.times_sec),
        holdout_gyro_count=len(gyro_holdout.times_sec),
        train_accel_count=len(accel_train.times_sec),
        holdout_accel_count=len(accel_holdout.times_sec),
        fit_status=result.status,
        fit_diagnostics=result.diagnostics,
        iterations=result.iterations,
        max_trajectory_translation_error_m=translation_error,
        max_trajectory_rotation_error_deg=rotation_error,
        parameters=parameters,
        train_gyro_rmse=train_score("gyro", estimates),
        holdout_gyro_rmse=holdout_gyro,
        train_accel_rmse_m_s2=train_score("accel", estimates),
        holdout_accel_rmse_m_s2=holdout_accel,
        oracle_holdout_gyro_rmse=oracle_gyro,
        oracle_holdout_accel_rmse_m_s2=oracle_accel,
        holdout_to_oracle_ratio_limit=_HOLDOUT_TO_ORACLE_RATIO,
        control_delta_chi2_threshold=_CONTROL_DELTA_CHI2,
        controls=controls,
        entangled_correlation_threshold=_ENTANGLED_CORRELATION,
        separability=separability,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeLidarImuJointProvenance(
            generator=__name__,
            generator_version="0.2",
            git_commit=git_commit(),
            command=command or [],
            seed=seed,
        ),
    )


def _smooth_rotation_vector(time: float) -> np.ndarray:
    return np.array(
        [
            0.9 * math.sin(1.3 * time),
            0.8 * math.sin(1.1 * time + 0.4),
            1.2 * math.sin(0.6 * time),
        ]
    )


def _smooth_rotation(time: float) -> np.ndarray:
    return rotation_vector_to_matrix(_smooth_rotation_vector(time))


def _smooth_position(time: float) -> np.ndarray:
    return np.array(
        [
            0.6 * math.sin(0.5 * time) + 0.1 * time,
            0.5 * math.sin(0.35 * time + 0.2),
            0.15 * math.sin(0.8 * time),
        ]
    )


def _smooth_acceleration(time: float) -> np.ndarray:
    return np.array(
        [
            -0.6 * 0.25 * math.sin(0.5 * time),
            -0.5 * 0.35**2 * math.sin(0.35 * time + 0.2),
            -0.15 * 0.64 * math.sin(0.8 * time),
        ]
    )


def _smooth_body_rate(time: float) -> np.ndarray:
    rate = np.array(
        [
            0.9 * 1.3 * math.cos(1.3 * time),
            0.8 * 1.1 * math.cos(1.1 * time + 0.4),
            1.2 * 0.6 * math.cos(0.6 * time),
        ]
    )
    # ``so3_left_jacobian`` satisfies Exp(w + d) ~= Exp(w) Exp(V d), so the body
    # rate of R = Exp(phi(t)) is V(phi) phi'.
    return np.asarray(so3_left_jacobian(_smooth_rotation_vector(time)) @ rate, dtype=float)


def _bspline_imu_values(
    body_times: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    imu_rotation = rotation_vector_to_matrix(np.asarray(_TRUE_IMU_ROTATION_RAD))
    lever = np.asarray(_TRUE_LEVER_ARM_BODY_M, dtype=float)
    gravity = np.asarray(_TRUE_GRAVITY_WORLD_M_S2, dtype=float)
    step = 1.0e-5
    gyro: list[np.ndarray] = []
    accel: list[np.ndarray] = []
    for time in body_times:
        omega = _smooth_body_rate(float(time))
        alpha = (
            _smooth_body_rate(float(time) + step) - _smooth_body_rate(float(time) - step)
        ) / (2.0 * step)
        kinematic = (
            _smooth_rotation(float(time)).T @ (_smooth_acceleration(float(time)) - gravity)
            + np.cross(alpha, lever)
            + np.cross(omega, np.cross(omega, lever))
        )
        gyro.append(
            imu_rotation.T @ omega
            + np.asarray(_TRUE_GYRO_BIAS_RAD_S)
            + rng.normal(0.0, _GYRO_NOISE_RAD_S, 3)
        )
        accel.append(
            imu_rotation.T @ kinematic
            + np.asarray(_TRUE_ACCEL_BIAS_M_S2)
            + rng.normal(0.0, _ACCEL_NOISE_M_S2, 3)
        )
    return np.asarray(gyro), np.asarray(accel)


def _bspline_plane_returns(rng: np.random.Generator) -> BSplinePlaneReturns:
    planes = [
        (np.asarray(point, dtype=float), np.asarray(normal, dtype=float))
        for point, normal in _PLANES
    ]
    planes = [(point, normal / float(np.linalg.norm(normal))) for point, normal in planes]
    times = np.sort(rng.uniform(0.0, _DURATION_SEC, int(_BSPLINE_LIDAR_RATE_HZ * _DURATION_SEC)))
    points: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for time in times:
        plane_point, normal = planes[int(rng.integers(len(planes)))]
        tangent = np.cross(normal, np.array([0.3, 0.5, 0.8]))
        tangent /= float(np.linalg.norm(tangent))
        bitangent = np.cross(normal, tangent)
        world = (
            plane_point
            + rng.uniform(-3.0, 3.0) * tangent
            + rng.uniform(-3.0, 3.0) * bitangent
            + rng.normal(0.0, _PLANE_NOISE_M) * normal
        )
        points.append(
            _smooth_rotation(float(time)).T @ (world - _smooth_position(float(time)))
        )
        anchors.append(plane_point)
        normals.append(normal)
    return BSplinePlaneReturns(
        times_sec=times,
        points_body_m=np.asarray(points),
        plane_points_world_m=np.asarray(anchors),
        plane_normals_world=np.asarray(normals),
        sigma_m=_PLANE_NOISE_M,
    )


def _estimates_from_calibration(calibration: LidarImuCalibration) -> _Estimates:
    return {
        "imu_rotation": np.asarray(calibration.imu_rotation_vector_rad, dtype=float),
        "gyro_bias": np.asarray(calibration.gyro_bias_rad_s, dtype=float),
        "lever_arm": np.asarray(calibration.lever_arm_body_m, dtype=float),
        "imu_clock_offset": np.asarray([calibration.imu_clock_offset_sec], dtype=float),
        "accel_bias": np.asarray(calibration.accel_bias_m_s2, dtype=float),
        "gravity": np.asarray(calibration.gravity_world_m_s2, dtype=float),
    }


def _calibration_from_estimates(estimates: _Estimates) -> LidarImuCalibration:
    return LidarImuCalibration(
        imu_rotation_vector_rad=estimates["imu_rotation"],
        gyro_bias_rad_s=estimates["gyro_bias"],
        lever_arm_body_m=estimates["lever_arm"],
        imu_clock_offset_sec=float(estimates["imu_clock_offset"][0]),
        accel_bias_m_s2=estimates["accel_bias"],
        gravity_world_m_s2=estimates["gravity"],
    )


def _separability(
    correlations: dict[tuple[str, str], float],
) -> list[LidarImuJointSeparabilityRecord]:
    return [
        LidarImuJointSeparabilityRecord(
            block_a=_block_name(left),
            block_b=_block_name(right),
            max_abs_correlation=value,
            entangled=value >= _ENTANGLED_CORRELATION,
        )
        for (left, right), value in correlations.items()
    ]
