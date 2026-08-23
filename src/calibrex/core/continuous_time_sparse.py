"""Sparse manifold trajectory fitting with analytic SE(3) Jacobians.

The fitter estimates the knot poses of a ``screw_linear`` continuous-time
trajectory from body-frame point measurements, point-to-plane measurements,
pose measurements, and IMU gyro pre-integration.  Point measurements are the
native LiDAR factor form: a body-frame point at a timestamp is transformed by
the interpolated pose and compared with a world target.  Point-to-plane
measurements compare the signed distance of that transformed point to a world
plane.  Pose measurements are converted internally into four anchor point
constraints so that every residual uses the analytic point-transform
Jacobian.  IMU factors compare the relative rotation of two interpolated poses
with a bias-corrected gyro integral and optionally estimate a shared gyro
bias.  Lever-arm factors compare gravity-compensated IMU specific force with
the kinematic acceleration of an IMU origin displaced in the body frame.
A shared IMU clock offset maps IMU timestamps to body time as
``t_body = t_imu - tau``.  Accelerometer bias and world-frame gravity enter
the specific-force residual as ``a_kin + b - R^T g``.  Optional diagonal
gyro and accelerometer scale factors apply as ``s_g * (omega - b)`` and
``s_a * a_kin - R^T g + b``.

The normal equations are assembled as a sparse block matrix.  Point, plane,
and pose factors touch the two knots bracketing their timestamp; IMU factors
touch the knots of both endpoints plus the optional bias and clock offset.
Gauss--Newton with Levenberg--Marquardt damping solves the linear system on
the manifold.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from calibrex.core.continuous_time_marginalization import (
    KnotMarginalizationPrior,
    marginalization_prior_blocks,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import (
    interpolate_screw_jacobians,
    point_transform_jacobian,
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
    se3_exp,
    se3_log,
    skew,
    so3_left_jacobian_inverse,
    so3_log,
)

FloatArray: TypeAlias = NDArray[np.float64]

FitStatus = Literal["converged", "max_iterations", "singular_system"]


@dataclass(frozen=True)
class TrajectoryPointMeasurement:
    """A body-frame point observed at a timestamp against a world target."""

    measurement_id: str
    timestamp_sec: float
    point_body_m: tuple[float, float, float]
    target_world_m: tuple[float, float, float]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("point measurement timestamp must be finite")
        values = (
            *self.point_body_m,
            *self.target_world_m,
            self.weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("point measurement values must be finite")
        if self.weight <= 0.0:
            raise ValueError("point measurement weight must be positive")


@dataclass(frozen=True)
class TrajectoryPointToPlaneMeasurement:
    """A body-frame LiDAR return against a world plane at a timestamp."""

    measurement_id: str
    timestamp_sec: float
    point_body_m: tuple[float, float, float]
    plane_point_world_m: tuple[float, float, float]
    plane_normal_world: tuple[float, float, float]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("point-to-plane timestamp must be finite")
        values = (
            *self.point_body_m,
            *self.plane_point_world_m,
            *self.plane_normal_world,
            self.weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("point-to-plane values must be finite")
        if self.weight <= 0.0:
            raise ValueError("point-to-plane weight must be positive")
        normal = np.asarray(self.plane_normal_world, dtype=float)
        if float(np.linalg.norm(normal)) <= 1.0e-12:
            raise ValueError("point-to-plane normal must be non-zero")


@dataclass(frozen=True)
class TrajectoryPoseMeasurement:
    """A measured world pose at a timestamp."""

    measurement_id: str
    timestamp_sec: float
    pose_world_body: SE3
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("pose measurement timestamp must be finite")
        if self.weight <= 0.0:
            raise ValueError("pose measurement weight must be positive")


@dataclass(frozen=True)
class TrajectoryImuGyroSample:
    """A body-frame gyro reading at a timestamp."""

    timestamp_sec: float
    omega_body_rad_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("gyro sample timestamp must be finite")
        if not all(math.isfinite(value) for value in self.omega_body_rad_s):
            raise ValueError("gyro sample rates must be finite")


@dataclass(frozen=True)
class TrajectoryImuPreintegrationMeasurement:
    """Gyro samples spanning two timestamps for a rotation pre-integration."""

    measurement_id: str
    gyro_samples: tuple[TrajectoryImuGyroSample, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if len(self.gyro_samples) < 2:
            raise ValueError("IMU pre-integration requires at least two gyro samples")
        timestamps = [item.timestamp_sec for item in self.gyro_samples]
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("gyro sample timestamps must be strictly increasing")
        if self.weight <= 0.0:
            raise ValueError("IMU pre-integration weight must be positive")

    @property
    def start_time_sec(self) -> float:
        """Return the first gyro timestamp."""

        return self.gyro_samples[0].timestamp_sec

    @property
    def end_time_sec(self) -> float:
        """Return the last gyro timestamp."""

        return self.gyro_samples[-1].timestamp_sec


@dataclass(frozen=True)
class TrajectoryImuLeverArmMeasurement:
    """Gravity-compensated specific force used to observe the IMU lever arm."""

    measurement_id: str
    timestamp_sec: float
    accel_body_m_s2: tuple[float, float, float]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("lever-arm timestamp must be finite")
        if not all(math.isfinite(value) for value in self.accel_body_m_s2):
            raise ValueError("lever-arm acceleration must be finite")
        if self.weight <= 0.0:
            raise ValueError("lever-arm weight must be positive")


@dataclass(frozen=True)
class ContinuousTimeTrajectoryFitProblem:
    """Knot times, initial knots, and measurements for a trajectory fit."""

    knot_timestamps: tuple[float, ...]
    initial_knot_poses: tuple[SE3, ...]
    point_measurements: tuple[TrajectoryPointMeasurement, ...] = ()
    point_to_plane_measurements: tuple[TrajectoryPointToPlaneMeasurement, ...] = ()
    pose_measurements: tuple[TrajectoryPoseMeasurement, ...] = ()
    imu_preintegration_measurements: tuple[
        TrajectoryImuPreintegrationMeasurement, ...
    ] = ()
    imu_lever_arm_measurements: tuple[TrajectoryImuLeverArmMeasurement, ...] = ()
    initial_gyro_bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_lever_arm_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_imu_clock_offset_sec: float = 0.0
    initial_accel_bias_body_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_gravity_world_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_gyro_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    initial_accel_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    estimate_gyro_bias: bool = False
    estimate_lever_arm: bool = False
    estimate_imu_clock_offset: bool = False
    estimate_accel_bias: bool = False
    estimate_gravity: bool = False
    estimate_gyro_scale: bool = False
    estimate_accel_scale: bool = False
    knot_marginalization_priors: tuple[KnotMarginalizationPrior, ...] = ()
    interpolation: Literal["screw_linear"] = "screw_linear"

    def __post_init__(self) -> None:
        if len(self.knot_timestamps) < 2:
            raise ValueError("trajectory fit requires at least two knots")
        if len(self.knot_timestamps) != len(self.initial_knot_poses):
            raise ValueError("knot times and initial poses must have equal length")
        if any(
            right <= left
            for left, right in pairwise(self.knot_timestamps)
        ):
            raise ValueError("knot timestamps must be strictly increasing")
        if not math.isfinite(self.knot_timestamps[0]) or not math.isfinite(
            self.knot_timestamps[-1]
        ):
            raise ValueError("knot timestamps must be finite")
        if (
            not self.point_measurements
            and not self.point_to_plane_measurements
            and not self.pose_measurements
            and not self.imu_preintegration_measurements
            and not self.imu_lever_arm_measurements
        ):
            raise ValueError("trajectory fit requires at least one measurement")
        if not all(math.isfinite(value) for value in self.initial_gyro_bias_rad_s):
            raise ValueError("initial gyro bias must be finite")
        if not all(math.isfinite(value) for value in self.initial_lever_arm_body_m):
            raise ValueError("initial lever arm must be finite")
        if not math.isfinite(self.initial_imu_clock_offset_sec):
            raise ValueError("initial IMU clock offset must be finite")
        if not all(math.isfinite(value) for value in self.initial_accel_bias_body_m_s2):
            raise ValueError("initial accelerometer bias must be finite")
        if not all(math.isfinite(value) for value in self.initial_gravity_world_m_s2):
            raise ValueError("initial gravity must be finite")
        if self.estimate_gyro_bias and not self.imu_preintegration_measurements:
            raise ValueError("gyro bias estimation requires IMU pre-integration measurements")
        if self.estimate_lever_arm and not self.imu_lever_arm_measurements:
            raise ValueError("lever-arm estimation requires IMU lever-arm measurements")
        if self.estimate_imu_clock_offset and not (
            self.imu_preintegration_measurements or self.imu_lever_arm_measurements
        ):
            raise ValueError("IMU clock-offset estimation requires IMU measurements")
        if self.estimate_accel_bias and not self.imu_lever_arm_measurements:
            raise ValueError("accelerometer-bias estimation requires IMU lever-arm measurements")
        if self.estimate_gravity and not self.imu_lever_arm_measurements:
            raise ValueError("gravity estimation requires IMU lever-arm measurements")
        if not all(math.isfinite(value) for value in self.initial_gyro_scale):
            raise ValueError("initial gyro scale must be finite")
        if not all(math.isfinite(value) for value in self.initial_accel_scale):
            raise ValueError("initial accelerometer scale must be finite")
        if any(value <= 0.0 for value in self.initial_gyro_scale):
            raise ValueError("initial gyro scale must be positive")
        if any(value <= 0.0 for value in self.initial_accel_scale):
            raise ValueError("initial accelerometer scale must be positive")
        if self.estimate_gyro_scale and not self.imu_preintegration_measurements:
            raise ValueError("gyro-scale estimation requires IMU pre-integration measurements")
        if self.estimate_accel_scale and not self.imu_lever_arm_measurements:
            raise ValueError("accelerometer-scale estimation requires IMU lever-arm measurements")


@dataclass(frozen=True)
class ContinuousTimeTrajectoryFitOptions:
    """Gauss--Newton options for the sparse manifold fit."""

    max_iterations: int = 30
    initial_damping: float = 1.0e-3
    convergence_gradient_norm: float = 1.0e-8
    convergence_step_norm: float = 1.0e-9
    damping_scale_down: float = 0.5
    damping_scale_up: float = 10.0

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        if self.initial_damping <= 0.0:
            raise ValueError("initial_damping must be positive")
        if self.convergence_gradient_norm <= 0.0:
            raise ValueError("convergence_gradient_norm must be positive")
        if self.convergence_step_norm <= 0.0:
            raise ValueError("convergence_step_norm must be positive")


@dataclass(frozen=True)
class ContinuousTimeTrajectoryFitResult:
    """Optimized knots and convergence evidence for one fit."""

    status: FitStatus
    iterations: int
    knot_poses: tuple[SE3, ...]
    initial_knot_poses: tuple[SE3, ...]
    final_objective: float
    final_point_rmse: float | None
    final_point_to_plane_rmse: float | None
    final_pose_rmse: float | None
    final_imu_rotation_rmse_rad: float | None
    final_lever_arm_rmse_m_s2: float | None
    gyro_bias_rad_s: tuple[float, float, float] | None
    lever_arm_body_m: tuple[float, float, float] | None
    imu_clock_offset_sec: float | None
    accel_bias_body_m_s2: tuple[float, float, float] | None
    gravity_world_m_s2: tuple[float, float, float] | None
    gyro_scale: tuple[float, float, float] | None
    accel_scale: tuple[float, float, float] | None
    max_step_translation_m: float
    max_step_rotation_deg: float
    gradient_norm: float
    nonzero_jacobian_blocks: int


_ANCHOR_POINTS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def fit_continuous_trajectory(
    problem: ContinuousTimeTrajectoryFitProblem,
    options: ContinuousTimeTrajectoryFitOptions | None = None,
) -> ContinuousTimeTrajectoryFitResult:
    """Solve the sparse manifold least-squares trajectory fit."""

    settings = options or ContinuousTimeTrajectoryFitOptions()
    timestamps = problem.knot_timestamps
    knots = list(problem.initial_knot_poses)
    gyro_bias = np.asarray(problem.initial_gyro_bias_rad_s, dtype=float)
    lever_arm = np.asarray(problem.initial_lever_arm_body_m, dtype=float)
    clock_offset = float(problem.initial_imu_clock_offset_sec)
    accel_bias = np.asarray(problem.initial_accel_bias_body_m_s2, dtype=float)
    gravity = np.asarray(problem.initial_gravity_world_m_s2, dtype=float)
    gyro_scale = np.asarray(problem.initial_gyro_scale, dtype=float)
    accel_scale = np.asarray(problem.initial_accel_scale, dtype=float)
    estimate_bias = problem.estimate_gyro_bias
    estimate_lever = problem.estimate_lever_arm
    estimate_clock = problem.estimate_imu_clock_offset
    estimate_accel = problem.estimate_accel_bias
    estimate_grav = problem.estimate_gravity
    estimate_gyro_scale = problem.estimate_gyro_scale
    estimate_accel_scale = problem.estimate_accel_scale
    point_rows: list[tuple[int, tuple[float, float, float]]] = []
    plane_rows: list[tuple[int, TrajectoryPointToPlaneMeasurement]] = []
    pose_rows: list[tuple[int, SE3]] = []
    imu_rows: list[TrajectoryImuPreintegrationMeasurement] = []
    lever_rows: list[TrajectoryImuLeverArmMeasurement] = []
    for point_measurement in problem.point_measurements:
        interval = _interval_index(timestamps, point_measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"point measurement {point_measurement.measurement_id!r} is outside "
                "the knot domain"
            )
        point_rows.append((interval, point_measurement.point_body_m))
    for plane_measurement in problem.point_to_plane_measurements:
        interval = _interval_index(timestamps, plane_measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"point-to-plane measurement {plane_measurement.measurement_id!r} is "
                "outside the knot domain"
            )
        plane_rows.append((interval, plane_measurement))
    for pose_measurement in problem.pose_measurements:
        interval = _interval_index(timestamps, pose_measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"pose measurement {pose_measurement.measurement_id!r} is outside "
                "the knot domain"
            )
        pose_rows.append((interval, pose_measurement.pose_world_body))
    for imu_measurement in problem.imu_preintegration_measurements:
        if (
            _interval_index(
                timestamps,
                _imu_body_time(imu_measurement.start_time_sec, clock_offset),
            )
            is None
            or _interval_index(
                timestamps,
                _imu_body_time(imu_measurement.end_time_sec, clock_offset),
            )
            is None
        ):
            raise ValueError(
                f"IMU pre-integration {imu_measurement.measurement_id!r} is outside "
                "the knot domain"
            )
        imu_rows.append(imu_measurement)
    for lever_measurement in problem.imu_lever_arm_measurements:
        if _interval_index(
            timestamps,
            _imu_body_time(lever_measurement.timestamp_sec, clock_offset),
        ) is None:
            raise ValueError(
                f"lever-arm measurement {lever_measurement.measurement_id!r} is outside "
                "the knot domain"
            )
        lever_rows.append(lever_measurement)

    damping = settings.initial_damping
    gradient_norm = math.inf
    objective = _objective(
        problem,
        point_rows,
        plane_rows,
        pose_rows,
        imu_rows,
        lever_rows,
        knots,
        gyro_bias,
        lever_arm,
        clock_offset,
        accel_bias,
        gravity,
        gyro_scale,
        accel_scale,
    )
    best_objective = objective
    best_knots = list(knots)
    best_bias = np.array(gyro_bias, copy=True)
    best_lever = np.array(lever_arm, copy=True)
    best_clock = clock_offset
    best_accel = np.array(accel_bias, copy=True)
    best_gravity = np.array(gravity, copy=True)
    best_gyro_scale = np.array(gyro_scale, copy=True)
    best_accel_scale = np.array(accel_scale, copy=True)
    max_step_translation = 0.0
    max_step_rotation_deg = 0.0
    status: FitStatus = "max_iterations"
    iterations = 0
    knot_count = len(knots)
    parameter_count = (
        6 * knot_count
        + (3 if estimate_bias else 0)
        + (3 if estimate_lever else 0)
        + (1 if estimate_clock else 0)
        + (3 if estimate_accel else 0)
        + (3 if estimate_grav else 0)
        + (3 if estimate_gyro_scale else 0)
        + (3 if estimate_accel_scale else 0)
    )
    bias_offset = 6 * knot_count
    lever_offset = bias_offset + (3 if estimate_bias else 0)
    clock_offset_index = lever_offset + (3 if estimate_lever else 0)
    accel_offset = clock_offset_index + (1 if estimate_clock else 0)
    gravity_offset = accel_offset + (3 if estimate_accel else 0)
    gyro_scale_offset = gravity_offset + (3 if estimate_grav else 0)
    accel_scale_offset = gyro_scale_offset + (3 if estimate_gyro_scale else 0)
    blocks: list[tuple[int, int, NDArray[np.float64]]] = []

    for iteration in range(settings.max_iterations):
        iterations = iteration + 1
        blocks = []
        residuals: list[float] = []
        weights: list[float] = []
        row_offset = 0
        for index, (interval, point_body) in enumerate(point_rows):
            measurement = problem.point_measurements[index]
            residual, jacobian_left, jacobian_right = _point_residual(
                knots,
                timestamps,
                interval,
                measurement.timestamp_sec,
                point_body,
                measurement.target_world_m,
            )
            residuals.extend(float(value) for value in residual)
            weights.extend([measurement.weight] * 3)
            blocks.append((row_offset, 6 * interval, jacobian_left))
            blocks.append((row_offset, 6 * (interval + 1), jacobian_right))
            row_offset += 3
        for interval, plane_measurement in plane_rows:
            residual, jacobian_left, jacobian_right = _point_to_plane_residual(
                knots,
                timestamps,
                interval,
                plane_measurement,
            )
            residuals.append(float(residual[0]))
            weights.append(plane_measurement.weight)
            blocks.append((row_offset, 6 * interval, jacobian_left))
            blocks.append((row_offset, 6 * (interval + 1), jacobian_right))
            row_offset += 1
        for index, (interval, pose) in enumerate(pose_rows):
            weight = problem.pose_measurements[index].weight
            pose_measurement = problem.pose_measurements[index]
            for anchor in _ANCHOR_POINTS:
                residual, jacobian_left, jacobian_right = _anchor_residual(
                    knots,
                    timestamps,
                    interval,
                    pose_measurement.timestamp_sec,
                    anchor,
                    pose,
                )
                residuals.extend(float(value) for value in residual)
                weights.extend([weight] * 3)
                blocks.append((row_offset, 6 * interval, jacobian_left))
                blocks.append((row_offset, 6 * (interval + 1), jacobian_right))
                row_offset += 3
        for imu_measurement in imu_rows:
            (
                residual,
                knot_blocks,
                jacobian_bias,
                jacobian_clock,
                jacobian_gyro_scale,
            ) = _imu_preintegration_residual(
                knots, timestamps, imu_measurement, gyro_bias, clock_offset, gyro_scale
            )
            residuals.extend(float(value) for value in residual)
            weights.extend([imu_measurement.weight] * 3)
            for knot_index, jacobian in knot_blocks:
                blocks.append((row_offset, 6 * knot_index, jacobian))
            if estimate_bias:
                blocks.append((row_offset, 6 * knot_count, jacobian_bias))
            if estimate_clock:
                blocks.append((row_offset, clock_offset_index, jacobian_clock))
            if estimate_gyro_scale:
                blocks.append((row_offset, gyro_scale_offset, jacobian_gyro_scale))
            row_offset += 3
        for lever_measurement in lever_rows:
            (
                residual,
                jacobian_lever,
                jacobian_clock,
                jacobian_accel,
                jacobian_gravity,
                jacobian_accel_scale,
            ) = _lever_arm_residual(
                knots,
                timestamps,
                lever_measurement,
                lever_arm,
                clock_offset,
                accel_bias,
                gravity,
                accel_scale,
            )
            residuals.extend(float(value) for value in residual)
            weights.extend([lever_measurement.weight] * 3)
            if estimate_lever:
                blocks.append((row_offset, lever_offset, jacobian_lever))
            if estimate_clock:
                blocks.append((row_offset, clock_offset_index, jacobian_clock))
            if estimate_accel:
                blocks.append((row_offset, accel_offset, jacobian_accel))
            if estimate_grav:
                blocks.append((row_offset, gravity_offset, jacobian_gravity))
            if estimate_accel_scale:
                blocks.append((row_offset, accel_scale_offset, jacobian_accel_scale))
            row_offset += 3
        for prior in problem.knot_marginalization_priors:
            if prior.knot_index < 0 or prior.knot_index >= knot_count:
                raise ValueError("marginalization prior knot index is out of range")
            weighted_residual, weighted_jacobian = marginalization_prior_blocks(
                prior, knots[prior.knot_index]
            )
            for axis in range(6):
                residuals.append(float(weighted_residual[axis]))
                weights.append(1.0)
                blocks.append(
                    (
                        row_offset,
                        6 * prior.knot_index,
                        weighted_jacobian[axis : axis + 1],
                    )
                )
                row_offset += 1

        residual_array = np.asarray(residuals, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        jacobian = _assemble_sparse_jacobian(
            blocks,
            residual_count=len(residuals),
            parameter_count=parameter_count,
        )
        weighted_jacobian = jacobian.multiply(weight_array[:, None])
        normal = weighted_jacobian.T @ weighted_jacobian
        rhs = weighted_jacobian.T @ (weight_array * residual_array)
        gradient_norm = float(np.max(np.abs(rhs)))

        if gradient_norm < settings.convergence_gradient_norm:
            status = "converged"
            break

        damping_matrix = normal + damping * _block_diagonal_scale(
            normal, knot_count, parameter_count
        )
        try:
            delta = spsolve(damping_matrix.tocsc(), -rhs)
        except RuntimeError:
            status = "singular_system"
            break
        if not np.all(np.isfinite(delta)):
            status = "singular_system"
            break

        candidate = [
            se3_exp(knot, delta[6 * i : 6 * i + 6]) for i, knot in enumerate(knots)
        ]
        candidate_bias = np.array(gyro_bias, copy=True)
        candidate_lever = np.array(lever_arm, copy=True)
        candidate_clock = clock_offset
        candidate_accel = np.array(accel_bias, copy=True)
        candidate_gravity = np.array(gravity, copy=True)
        candidate_gyro_scale = np.array(gyro_scale, copy=True)
        candidate_accel_scale = np.array(accel_scale, copy=True)
        if estimate_bias:
            candidate_bias = gyro_bias + np.asarray(
                delta[bias_offset : bias_offset + 3], dtype=float
            )
        if estimate_lever:
            candidate_lever = lever_arm + np.asarray(
                delta[lever_offset : lever_offset + 3], dtype=float
            )
        if estimate_clock:
            candidate_clock = clock_offset + float(delta[clock_offset_index])
        if estimate_accel:
            candidate_accel = accel_bias + np.asarray(
                delta[accel_offset : accel_offset + 3], dtype=float
            )
        if estimate_grav:
            candidate_gravity = gravity + np.asarray(
                delta[gravity_offset : gravity_offset + 3], dtype=float
            )
        if estimate_gyro_scale:
            candidate_gyro_scale = gyro_scale + np.asarray(
                delta[gyro_scale_offset : gyro_scale_offset + 3], dtype=float
            )
        if estimate_accel_scale:
            candidate_accel_scale = accel_scale + np.asarray(
                delta[accel_scale_offset : accel_scale_offset + 3], dtype=float
            )
        candidate_objective = _objective(
            problem,
            point_rows,
            plane_rows,
            pose_rows,
            imu_rows,
            lever_rows,
            candidate,
            candidate_bias,
            candidate_lever,
            candidate_clock,
            candidate_accel,
            candidate_gravity,
            candidate_gyro_scale,
            candidate_accel_scale,
        )
        knot_delta = np.asarray(delta[: 6 * knot_count], dtype=float).reshape(-1, 6)
        step_translation = float(np.max(np.abs(knot_delta[:, :3]))) if knot_delta.size else 0.0
        step_rotation_rad = float(np.max(np.abs(knot_delta[:, 3:]))) if knot_delta.size else 0.0
        extra = delta[6 * knot_count :]
        if extra.size:
            step_rotation_rad = max(step_rotation_rad, float(np.max(np.abs(extra))))
        step_rotation = math.degrees(step_rotation_rad)
        if candidate_objective < objective:
            objective = candidate_objective
            knots = candidate
            gyro_bias = candidate_bias
            lever_arm = candidate_lever
            clock_offset = candidate_clock
            accel_bias = candidate_accel
            gravity = candidate_gravity
            gyro_scale = candidate_gyro_scale
            accel_scale = candidate_accel_scale
            if objective < best_objective:
                best_objective = objective
                best_knots = list(knots)
                best_bias = np.array(gyro_bias, copy=True)
                best_lever = np.array(lever_arm, copy=True)
                best_clock = clock_offset
                best_accel = np.array(accel_bias, copy=True)
                best_gravity = np.array(gravity, copy=True)
                best_gyro_scale = np.array(gyro_scale, copy=True)
                best_accel_scale = np.array(accel_scale, copy=True)
            max_step_translation = max(max_step_translation, step_translation)
            max_step_rotation_deg = max(max_step_rotation_deg, step_rotation)
            damping = max(damping * settings.damping_scale_down, 1.0e-12)
            if (
                step_translation < settings.convergence_step_norm
                and step_rotation_rad < settings.convergence_step_norm
            ):
                status = "converged"
                break
        else:
            damping *= settings.damping_scale_up
            if damping > 1.0e12:
                break

    if status != "converged" and best_objective < objective:
        knots = best_knots
        gyro_bias = best_bias
        lever_arm = best_lever
        clock_offset = best_clock
        accel_bias = best_accel
        gravity = best_gravity
        gyro_scale = best_gyro_scale
        accel_scale = best_accel_scale
        objective = best_objective
    final_point_rmse = _point_rmse(problem, point_rows, knots)
    final_point_to_plane_rmse = _point_to_plane_rmse(plane_rows, timestamps, knots)
    final_pose_rmse = _pose_rmse(problem, pose_rows, knots)
    final_imu_rmse = imu_preintegration_rmse(
        tuple(imu_rows), timestamps, knots, gyro_bias, clock_offset, gyro_scale
    )
    final_lever_rmse = imu_lever_arm_rmse(
        tuple(lever_rows),
        timestamps,
        knots,
        lever_arm,
        clock_offset,
        accel_bias,
        gravity,
        accel_scale,
    )
    return ContinuousTimeTrajectoryFitResult(
        status=status,
        iterations=iterations,
        knot_poses=tuple(knots),
        initial_knot_poses=problem.initial_knot_poses,
        final_objective=objective,
        final_point_rmse=final_point_rmse,
        final_point_to_plane_rmse=final_point_to_plane_rmse,
        final_pose_rmse=final_pose_rmse,
        final_imu_rotation_rmse_rad=final_imu_rmse,
        final_lever_arm_rmse_m_s2=final_lever_rmse,
        gyro_bias_rad_s=(
            (float(gyro_bias[0]), float(gyro_bias[1]), float(gyro_bias[2]))
            if imu_rows
            else None
        ),
        lever_arm_body_m=(
            (float(lever_arm[0]), float(lever_arm[1]), float(lever_arm[2]))
            if lever_rows
            else None
        ),
        imu_clock_offset_sec=(
            float(clock_offset) if imu_rows or lever_rows else None
        ),
        accel_bias_body_m_s2=(
            (float(accel_bias[0]), float(accel_bias[1]), float(accel_bias[2]))
            if lever_rows
            else None
        ),
        gravity_world_m_s2=(
            (float(gravity[0]), float(gravity[1]), float(gravity[2]))
            if lever_rows
            else None
        ),
        gyro_scale=(
            (float(gyro_scale[0]), float(gyro_scale[1]), float(gyro_scale[2]))
            if imu_rows
            else None
        ),
        accel_scale=(
            (float(accel_scale[0]), float(accel_scale[1]), float(accel_scale[2]))
            if lever_rows
            else None
        ),
        max_step_translation_m=max_step_translation,
        max_step_rotation_deg=max_step_rotation_deg,
        gradient_norm=gradient_norm,
        nonzero_jacobian_blocks=len(blocks),
    )


def _point_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    timestamp_sec: float,
    point_body: tuple[float, float, float],
    target_world: tuple[float, float, float],
) -> tuple[FloatArray, NDArray[np.float64], NDArray[np.float64]]:
    alpha = _alpha(timestamps, interval, timestamp_sec)
    left = knots[interval]
    right = knots[interval + 1]
    pose = _interpolate(left, right, alpha)
    residual = np.asarray(pose.transform_point(point_body), dtype=float)
    residual -= np.asarray(target_world, dtype=float)
    point_jacobian = point_transform_jacobian(pose, np.asarray(point_body))
    jacobian_left, jacobian_right = interpolate_screw_jacobians(
        left, right, alpha
    )
    return residual, point_jacobian @ jacobian_left, point_jacobian @ jacobian_right


def _point_to_plane_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    measurement: TrajectoryPointToPlaneMeasurement,
) -> tuple[FloatArray, NDArray[np.float64], NDArray[np.float64]]:
    alpha = _alpha(timestamps, interval, measurement.timestamp_sec)
    left = knots[interval]
    right = knots[interval + 1]
    pose = _interpolate(left, right, alpha)
    world = np.asarray(pose.transform_point(measurement.point_body_m), dtype=float)
    plane_point = np.asarray(measurement.plane_point_world_m, dtype=float)
    normal = np.asarray(measurement.plane_normal_world, dtype=float)
    normal = normal / float(np.linalg.norm(normal))
    residual = np.asarray([float(normal @ (world - plane_point))], dtype=float)
    point_jacobian = point_transform_jacobian(
        pose, np.asarray(measurement.point_body_m)
    )
    signed_jacobian = normal.reshape(1, 3) @ point_jacobian
    jacobian_left, jacobian_right = interpolate_screw_jacobians(left, right, alpha)
    return residual, signed_jacobian @ jacobian_left, signed_jacobian @ jacobian_right


def interpolate_pose_at(
    knots: tuple[SE3, ...] | list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
) -> SE3:
    """Interpolate a screw-linear pose at ``timestamp_sec``."""

    interval = _interval_index(timestamps, timestamp_sec)
    if interval is None:
        raise ValueError("timestamp is outside the knot domain")
    alpha = _alpha(timestamps, interval, timestamp_sec)
    return _interpolate(knots[interval], knots[interval + 1], alpha)


def point_to_plane_rmse(
    measurements: tuple[TrajectoryPointToPlaneMeasurement, ...],
    timestamps: tuple[float, ...],
    knots: tuple[SE3, ...] | list[SE3],
) -> float | None:
    """Return RMS signed distance of plane measurements against ``knots``."""

    if not measurements:
        return None
    squared = 0.0
    for measurement in measurements:
        interval = _interval_index(timestamps, measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"point-to-plane measurement {measurement.measurement_id!r} is "
                "outside the knot domain"
            )
        residual, _left, _right = _point_to_plane_residual(
            list(knots), timestamps, interval, measurement
        )
        squared += float(residual[0] * residual[0])
    return math.sqrt(squared / len(measurements))


def imu_preintegration_rmse(
    measurements: tuple[TrajectoryImuPreintegrationMeasurement, ...],
    timestamps: tuple[float, ...],
    knots: tuple[SE3, ...] | list[SE3],
    gyro_bias_rad_s: tuple[float, float, float] | FloatArray,
    clock_offset_sec: float = 0.0,
    gyro_scale: tuple[float, float, float] | FloatArray = (1.0, 1.0, 1.0),
) -> float | None:
    """Return RMS rotation residual of IMU pre-integration factors."""

    if not measurements:
        return None
    bias = np.asarray(gyro_bias_rad_s, dtype=float)
    scale = np.asarray(gyro_scale, dtype=float)
    squared = 0.0
    for measurement in measurements:
        residual, _knots, _bias, _clock, _scale = _imu_preintegration_residual(
            list(knots), timestamps, measurement, bias, clock_offset_sec, scale
        )
        squared += float(np.sum(residual * residual))
    return math.sqrt(squared / (3 * len(measurements)))


def imu_lever_arm_rmse(
    measurements: tuple[TrajectoryImuLeverArmMeasurement, ...],
    timestamps: tuple[float, ...],
    knots: tuple[SE3, ...] | list[SE3],
    lever_arm_body_m: tuple[float, float, float] | FloatArray,
    clock_offset_sec: float = 0.0,
    accel_bias_body_m_s2: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    gravity_world_m_s2: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    accel_scale: tuple[float, float, float] | FloatArray = (1.0, 1.0, 1.0),
) -> float | None:
    """Return RMS specific-force residual of lever-arm measurements."""

    if not measurements:
        return None
    lever = np.asarray(lever_arm_body_m, dtype=float)
    accel_bias = np.asarray(accel_bias_body_m_s2, dtype=float)
    gravity = np.asarray(gravity_world_m_s2, dtype=float)
    scale = np.asarray(accel_scale, dtype=float)
    squared = 0.0
    for measurement in measurements:
        residual, _lever, _clock, _accel, _gravity, _scale = _lever_arm_residual(
            list(knots),
            timestamps,
            measurement,
            lever,
            clock_offset_sec,
            accel_bias,
            gravity,
            scale,
        )
        squared += float(np.sum(residual * residual))
    return math.sqrt(squared / (3 * len(measurements)))


def _lever_arm_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuLeverArmMeasurement,
    lever_arm: FloatArray,
    clock_offset_sec: float = 0.0,
    accel_bias_body_m_s2: FloatArray | None = None,
    gravity_world_m_s2: FloatArray | None = None,
    accel_scale: FloatArray | None = None,
) -> tuple[
    FloatArray,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    accel_bias = np.zeros(3, dtype=float) if accel_bias_body_m_s2 is None else np.asarray(
        accel_bias_body_m_s2, dtype=float
    ).reshape(3)
    gravity = np.zeros(3, dtype=float) if gravity_world_m_s2 is None else np.asarray(
        gravity_world_m_s2, dtype=float
    ).reshape(3)
    scale = np.ones(3, dtype=float) if accel_scale is None else np.asarray(
        accel_scale, dtype=float
    ).reshape(3)
    residual, jacobian_lever, jacobian_accel, jacobian_gravity, jacobian_accel_scale = (
        _lever_arm_kinematics_residual(
            knots,
            timestamps,
            measurement,
            lever_arm,
            clock_offset_sec,
            accel_bias,
            gravity,
            scale,
        )
    )
    epsilon = 1.0e-6
    perturbed, _lever, _accel, _gravity, _scale = _lever_arm_kinematics_residual(
        knots,
        timestamps,
        measurement,
        lever_arm,
        clock_offset_sec + epsilon,
        accel_bias,
        gravity,
        scale,
    )
    jacobian_clock = ((perturbed - residual) / epsilon).reshape(3, 1)
    return (
        residual,
        jacobian_lever,
        jacobian_clock,
        jacobian_accel,
        jacobian_gravity,
        jacobian_accel_scale,
    )


def _lever_arm_kinematics_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuLeverArmMeasurement,
    lever_arm: FloatArray,
    clock_offset_sec: float,
    accel_bias_body_m_s2: FloatArray,
    gravity_world_m_s2: FloatArray,
    accel_scale: FloatArray,
) -> tuple[
    FloatArray,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    body_time = _imu_body_time(measurement.timestamp_sec, clock_offset_sec)
    omega, alpha, accel_origin, rotation = _body_motion(knots, timestamps, body_time)
    lever = np.asarray(lever_arm, dtype=float).reshape(3)
    accel_bias = np.asarray(accel_bias_body_m_s2, dtype=float).reshape(3)
    gravity = np.asarray(gravity_world_m_s2, dtype=float).reshape(3)
    scale = np.asarray(accel_scale, dtype=float).reshape(3)
    kinematic = (
        accel_origin
        + np.cross(alpha, lever)
        + np.cross(omega, np.cross(omega, lever))
        - rotation.T @ gravity
    )
    predicted = scale * kinematic + accel_bias
    residual = predicted - np.asarray(measurement.accel_body_m_s2, dtype=float)
    jacobian_lever = skew(alpha) + skew(omega) @ skew(omega)
    jacobian_accel: NDArray[np.float64] = np.eye(3, dtype=float)
    jacobian_gravity = -np.diag(scale) @ rotation.T
    jacobian_accel_scale = np.diag(kinematic)
    return (
        residual,
        jacobian_lever,
        jacobian_accel,
        jacobian_gravity,
        jacobian_accel_scale,
    )


def _body_motion(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    *,
    epsilon: float = 1.0e-3,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.float64]]:
    span = timestamps[-1] - timestamps[0]
    limited = min(max(timestamp_sec, timestamps[0] + 2.0 * epsilon), timestamps[-1] - 2.0 * epsilon)
    if span <= 4.0 * epsilon:
        raise ValueError("knot domain is too short for lever-arm kinematics")
    pose = interpolate_pose_at(knots, timestamps, limited)
    pose_minus = interpolate_pose_at(knots, timestamps, limited - epsilon)
    pose_plus = interpolate_pose_at(knots, timestamps, limited + epsilon)
    omega = se3_log(pose, pose_plus)[3:] / epsilon
    omega_minus = se3_log(pose_minus, pose)[3:] / epsilon
    alpha = (omega - omega_minus) / epsilon
    origin = np.asarray(pose.translation_m, dtype=float)
    origin_minus = np.asarray(pose_minus.translation_m, dtype=float)
    origin_plus = np.asarray(pose_plus.translation_m, dtype=float)
    accel_world = (origin_plus - 2.0 * origin + origin_minus) / (epsilon * epsilon)
    rotation = _rotation_matrix(pose)
    accel_body = rotation.T @ accel_world
    return omega, alpha, accel_body, rotation


def predicted_imu_specific_force(
    knots: tuple[SE3, ...] | list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    lever_arm_body_m: tuple[float, float, float] | FloatArray,
    *,
    accel_bias_body_m_s2: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    gravity_world_m_s2: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    accel_scale: tuple[float, float, float] | FloatArray = (1.0, 1.0, 1.0),
) -> FloatArray:
    """Return IMU specific force ``s * (a_kin - R^T g) + b`` in the body frame."""

    omega, alpha, accel_origin, rotation = _body_motion(
        list(knots), timestamps, timestamp_sec
    )
    lever = np.asarray(lever_arm_body_m, dtype=float).reshape(3)
    accel_bias = np.asarray(accel_bias_body_m_s2, dtype=float).reshape(3)
    gravity = np.asarray(gravity_world_m_s2, dtype=float).reshape(3)
    scale = np.asarray(accel_scale, dtype=float).reshape(3)
    kinematic = (
        accel_origin
        + np.cross(alpha, lever)
        + np.cross(omega, np.cross(omega, lever))
        - rotation.T @ gravity
    )
    return scale * kinematic + accel_bias


def _imu_preintegration_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuPreintegrationMeasurement,
    gyro_bias: FloatArray,
    clock_offset_sec: float = 0.0,
    gyro_scale: FloatArray | None = None,
) -> tuple[
    FloatArray,
    list[tuple[int, NDArray[np.float64]]],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    scale = np.ones(3, dtype=float) if gyro_scale is None else np.asarray(
        gyro_scale, dtype=float
    ).reshape(3)
    start_time = _imu_body_time(measurement.start_time_sec, clock_offset_sec)
    end_time = _imu_body_time(measurement.end_time_sec, clock_offset_sec)
    start = interpolate_pose_at(knots, timestamps, start_time)
    end = interpolate_pose_at(knots, timestamps, end_time)
    rotation_start = _rotation_matrix(start)
    rotation_end = _rotation_matrix(end)
    delta_rotation, jacobian_right_bias, jacobian_right_scale = _integrate_gyro(
        measurement.gyro_samples, gyro_bias, scale
    )
    rotation_error = delta_rotation.T @ rotation_start.T @ rotation_end
    residual = rotation_matrix_to_rotation_vector(rotation_error)
    log_jacobian = so3_left_jacobian_inverse(residual)
    jacobian_start_rot = -log_jacobian @ delta_rotation.T
    jacobian_end_rot = so3_left_jacobian_inverse(-residual)
    jacobian_bias = -log_jacobian @ jacobian_right_bias
    jacobian_scale = -log_jacobian @ jacobian_right_scale
    omega_start = _angular_rate_at(knots, timestamps, start_time)
    omega_end = _angular_rate_at(knots, timestamps, end_time)
    jacobian_clock = (
        jacobian_start_rot @ (-omega_start) + jacobian_end_rot @ (-omega_end)
    ).reshape(3, 1)
    knot_blocks = _interpolate_rotation_blocks(
        knots,
        timestamps,
        start_time,
        jacobian_start_rot,
    )
    end_blocks = _interpolate_rotation_blocks(
        knots,
        timestamps,
        end_time,
        jacobian_end_rot,
    )
    combined: dict[int, NDArray[np.float64]] = {}
    for knot_index, jacobian in (*knot_blocks, *end_blocks):
        if knot_index in combined:
            combined[knot_index] = combined[knot_index] + jacobian
        else:
            combined[knot_index] = jacobian
    return residual, sorted(combined.items()), jacobian_bias, jacobian_clock, jacobian_scale


def _imu_body_time(imu_time_sec: float, clock_offset_sec: float) -> float:
    return imu_time_sec - clock_offset_sec


def _angular_rate_at(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    *,
    epsilon: float = 1.0e-3,
) -> FloatArray:
    limited = min(
        max(timestamp_sec, timestamps[0]),
        timestamps[-1] - epsilon,
    )
    pose = interpolate_pose_at(knots, timestamps, limited)
    pose_plus = interpolate_pose_at(knots, timestamps, limited + epsilon)
    return se3_log(pose, pose_plus)[3:] / epsilon


def _interpolate_rotation_blocks(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    jacobian_rotation: NDArray[np.float64],
) -> list[tuple[int, NDArray[np.float64]]]:
    interval = _interval_index(timestamps, timestamp_sec)
    if interval is None:
        raise ValueError("timestamp is outside the knot domain")
    alpha = _alpha(timestamps, interval, timestamp_sec)
    jacobian_left, jacobian_right = interpolate_screw_jacobians(
        knots[interval], knots[interval + 1], alpha
    )
    selector = jacobian_rotation @ np.eye(6, dtype=float)[3:, :]
    return [
        (interval, selector @ jacobian_left),
        (interval + 1, selector @ jacobian_right),
    ]


def _integrate_gyro(
    samples: tuple[TrajectoryImuGyroSample, ...],
    gyro_bias: FloatArray,
    gyro_scale: FloatArray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    scale = np.ones(3, dtype=float) if gyro_scale is None else np.asarray(
        gyro_scale, dtype=float
    ).reshape(3)
    bias = np.asarray(gyro_bias, dtype=float).reshape(3)
    rotations: list[NDArray[np.float64]] = []
    phis: list[FloatArray] = []
    dts: list[float] = []
    raws: list[FloatArray] = []
    delta_rotation: NDArray[np.float64] = np.eye(3, dtype=float)
    for left, right in pairwise(samples):
        dt = right.timestamp_sec - left.timestamp_sec
        raw = np.asarray(left.omega_body_rad_s, dtype=float) - bias
        omega = scale * raw
        phi = omega * dt
        rotation = rotation_vector_to_matrix(phi)
        rotations.append(rotation)
        phis.append(phi)
        dts.append(dt)
        raws.append(raw)
        delta_rotation = delta_rotation @ rotation
    jacobian_bias: NDArray[np.float64] = np.zeros((3, 3), dtype=float)
    suffix: NDArray[np.float64] = np.eye(3, dtype=float)
    for rotation, phi, dt in zip(
        reversed(rotations), reversed(phis), reversed(dts), strict=True
    ):
        right_jacobian_inverse = so3_left_jacobian_inverse(-phi)
        jacobian_bias = jacobian_bias + suffix.T @ (right_jacobian_inverse * (-dt))
        suffix = rotation @ suffix
    jacobian_scale: NDArray[np.float64] = np.zeros((3, 3), dtype=float)
    suffix = np.eye(3, dtype=float)
    for rotation, phi, dt, raw in zip(
        reversed(rotations), reversed(phis), reversed(dts), reversed(raws), strict=True
    ):
        right_jacobian_inverse = so3_left_jacobian_inverse(-phi)
        for axis in range(3):
            tangent: NDArray[np.float64] = np.zeros(3, dtype=float)
            tangent[axis] = raw[axis] * dt
            jacobian_scale[:, axis] += suffix.T @ (right_jacobian_inverse @ tangent)
        suffix = rotation @ suffix
    return delta_rotation, jacobian_bias, jacobian_scale


def _rotation_matrix(pose: SE3) -> NDArray[np.float64]:
    return rotation_vector_to_matrix(so3_log(np.asarray(pose.rotation_quat_xyzw)))


def _anchor_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    timestamp_sec: float,
    anchor: tuple[float, float, float],
    pose: SE3,
) -> tuple[FloatArray, NDArray[np.float64], NDArray[np.float64]]:
    alpha = _alpha(timestamps, interval, timestamp_sec)
    left = knots[interval]
    right = knots[interval + 1]
    interpolated = _interpolate(left, right, alpha)
    target = pose.transform_point(anchor)
    residual = np.asarray(interpolated.transform_point(anchor), dtype=float)
    residual -= np.asarray(target, dtype=float)
    point_jacobian = point_transform_jacobian(interpolated, np.asarray(anchor))
    jacobian_left, jacobian_right = interpolate_screw_jacobians(
        left, right, alpha
    )
    return residual, point_jacobian @ jacobian_left, point_jacobian @ jacobian_right


def _interpolate(left: SE3, right: SE3, alpha: float) -> SE3:
    return se3_exp(left, alpha * se3_log(left, right))


def _alpha(
    timestamps: tuple[float, ...], interval: int, timestamp: float
) -> float:
    start = timestamps[interval]
    end = timestamps[interval + 1]
    return (timestamp - start) / (end - start)


def _interval_index(
    timestamps: tuple[float, ...], timestamp: float
) -> int | None:
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        return None
    right = bisect_left(timestamps, timestamp)
    if right == 0:
        return 0
    if right >= len(timestamps) - 1:
        return len(timestamps) - 2
    if timestamps[right] == timestamp:
        return right
    return right - 1


def _assemble_sparse_jacobian(
    blocks: list[tuple[int, int, NDArray[np.float64]]],
    *,
    residual_count: int,
    parameter_count: int,
) -> csc_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_offset, column_offset, jacobian in blocks:
        residual_rows = int(jacobian.shape[0])
        parameter_width = int(jacobian.shape[1])
        for residual_row in range(residual_rows):
            for parameter in range(parameter_width):
                rows.append(row_offset + residual_row)
                columns.append(column_offset + parameter)
                values.append(float(jacobian[residual_row, parameter]))
    return csc_matrix(
        (values, (rows, columns)),
        shape=(residual_count, parameter_count),
        dtype=float,
    )


def _block_diagonal_scale(
    normal: csc_matrix, knot_count: int, parameter_count: int
) -> csc_matrix:
    diagonal = normal.diagonal()
    scaled: NDArray[np.float64] = np.zeros(parameter_count, dtype=float)
    for knot in range(knot_count):
        block = diagonal[6 * knot : 6 * knot + 6]
        block_max = float(np.max(np.abs(block))) if block.size else 0.0
        if block_max > 0.0:
            scaled[6 * knot : 6 * knot + 6] = block_max
    if parameter_count > 6 * knot_count:
        extra = diagonal[6 * knot_count : parameter_count]
        extra_max = float(np.max(np.abs(extra))) if extra.size else 0.0
        if extra_max > 0.0:
            scaled[6 * knot_count : parameter_count] = extra_max
    indices: NDArray[np.int64] = np.arange(parameter_count, dtype=np.int64)
    return csc_matrix(
        (scaled, (indices, indices)),
        shape=(parameter_count, parameter_count),
    )


def _objective(
    problem: ContinuousTimeTrajectoryFitProblem,
    point_rows: list[tuple[int, tuple[float, float, float]]],
    plane_rows: list[tuple[int, TrajectoryPointToPlaneMeasurement]],
    pose_rows: list[tuple[int, SE3]],
    imu_rows: list[TrajectoryImuPreintegrationMeasurement],
    lever_rows: list[TrajectoryImuLeverArmMeasurement],
    knots: list[SE3],
    gyro_bias: FloatArray,
    lever_arm: FloatArray,
    clock_offset_sec: float,
    accel_bias: FloatArray,
    gravity: FloatArray,
    gyro_scale: FloatArray,
    accel_scale: FloatArray,
) -> float:
    objective = 0.0
    for index, (interval, point_body) in enumerate(point_rows):
        point_measurement = problem.point_measurements[index]
        residual, _left, _right = _point_residual(
            knots,
            problem.knot_timestamps,
            interval,
            point_measurement.timestamp_sec,
            point_body,
            point_measurement.target_world_m,
        )
        objective += point_measurement.weight * float(np.sum(residual * residual))
    for interval, plane_measurement in plane_rows:
        residual, _left, _right = _point_to_plane_residual(
            knots,
            problem.knot_timestamps,
            interval,
            plane_measurement,
        )
        objective += plane_measurement.weight * float(residual[0] * residual[0])
    for index, (interval, pose) in enumerate(pose_rows):
        pose_measurement = problem.pose_measurements[index]
        weight = pose_measurement.weight
        for anchor in _ANCHOR_POINTS:
            residual, _left, _right = _anchor_residual(
                knots,
                problem.knot_timestamps,
                interval,
                pose_measurement.timestamp_sec,
                anchor,
                pose,
            )
            objective += weight * float(np.sum(residual * residual))
    for imu_measurement in imu_rows:
        try:
            residual, _knots, _bias, _clock, _scale = _imu_preintegration_residual(
                knots,
                problem.knot_timestamps,
                imu_measurement,
                gyro_bias,
                clock_offset_sec,
                gyro_scale,
            )
        except ValueError:
            return math.inf
        objective += imu_measurement.weight * float(np.sum(residual * residual))
    for lever_measurement in lever_rows:
        try:
            residual, _lever, _clock, _accel, _gravity, _scale = _lever_arm_residual(
                knots,
                problem.knot_timestamps,
                lever_measurement,
                lever_arm,
                clock_offset_sec,
                accel_bias,
                gravity,
                accel_scale,
            )
        except ValueError:
            return math.inf
        objective += lever_measurement.weight * float(np.sum(residual * residual))
    for prior in problem.knot_marginalization_priors:
        delta = se3_log(prior.anchor_pose, knots[prior.knot_index])
        objective += float(delta.T @ prior.information @ delta)
    return objective


def _point_rmse(
    problem: ContinuousTimeTrajectoryFitProblem,
    point_rows: list[tuple[int, tuple[float, float, float]]],
    knots: list[SE3],
) -> float | None:
    if not point_rows:
        return None
    squared = 0.0
    count = 0
    for index, (interval, point_body) in enumerate(point_rows):
        measurement = problem.point_measurements[index]
        residual, _left, _right = _point_residual(
            knots,
            problem.knot_timestamps,
            interval,
            measurement.timestamp_sec,
            point_body,
            measurement.target_world_m,
        )
        squared += float(np.sum(residual * residual))
        count += 3
    return math.sqrt(squared / count) if count else None


def _point_to_plane_rmse(
    plane_rows: list[tuple[int, TrajectoryPointToPlaneMeasurement]],
    timestamps: tuple[float, ...],
    knots: list[SE3],
) -> float | None:
    if not plane_rows:
        return None
    squared = 0.0
    for interval, measurement in plane_rows:
        residual, _left, _right = _point_to_plane_residual(
            knots, timestamps, interval, measurement
        )
        squared += float(residual[0] * residual[0])
    return math.sqrt(squared / len(plane_rows))


def _pose_rmse(
    problem: ContinuousTimeTrajectoryFitProblem,
    pose_rows: list[tuple[int, SE3]],
    knots: list[SE3],
) -> float | None:
    if not pose_rows:
        return None
    squared = 0.0
    count = 0
    for index, (interval, pose) in enumerate(pose_rows):
        measurement = problem.pose_measurements[index]
        for anchor in _ANCHOR_POINTS:
            residual, _left, _right = _anchor_residual(
                knots,
                problem.knot_timestamps,
                interval,
                measurement.timestamp_sec,
                anchor,
                pose,
            )
            squared += float(np.sum(residual * residual))
            count += 3
    return math.sqrt(squared / count) if count else None


def build_dense_knot_normal_equations(
    problem: ContinuousTimeTrajectoryFitProblem,
    knots: list[SE3],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a dense knot-only normal system ``(H, g)`` at the current linearization."""

    if (
        problem.estimate_gyro_bias
        or problem.estimate_lever_arm
        or problem.estimate_imu_clock_offset
        or problem.estimate_accel_bias
        or problem.estimate_gravity
        or problem.estimate_gyro_scale
        or problem.estimate_accel_scale
    ):
        raise ValueError("dense knot normal equations require knot-only problems")
    timestamps = problem.knot_timestamps
    knot_count = len(knots)
    blocks: list[tuple[int, int, NDArray[np.float64]]] = []
    residuals: list[float] = []
    weights: list[float] = []
    row_offset = 0
    for point_measurement in problem.point_measurements:
        interval = _interval_index(timestamps, point_measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"point measurement {point_measurement.measurement_id!r} is outside "
                "the knot domain"
            )
        residual, jacobian_left, jacobian_right = _point_residual(
            knots,
            timestamps,
            interval,
            point_measurement.timestamp_sec,
            point_measurement.point_body_m,
            point_measurement.target_world_m,
        )
        residuals.extend(float(value) for value in residual)
        weights.extend([point_measurement.weight] * 3)
        blocks.append((row_offset, 6 * interval, jacobian_left))
        blocks.append((row_offset, 6 * (interval + 1), jacobian_right))
        row_offset += 3
    for plane_measurement in problem.point_to_plane_measurements:
        interval = _interval_index(timestamps, plane_measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"point-to-plane measurement {plane_measurement.measurement_id!r} is "
                "outside the knot domain"
            )
        residual, jacobian_left, jacobian_right = _point_to_plane_residual(
            knots,
            timestamps,
            interval,
            plane_measurement,
        )
        residuals.append(float(residual[0]))
        weights.append(plane_measurement.weight)
        blocks.append((row_offset, 6 * interval, jacobian_left))
        blocks.append((row_offset, 6 * (interval + 1), jacobian_right))
        row_offset += 1
    for pose_measurement in problem.pose_measurements:
        interval = _interval_index(timestamps, pose_measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"pose measurement {pose_measurement.measurement_id!r} is outside "
                "the knot domain"
            )
        for anchor in _ANCHOR_POINTS:
            residual, jacobian_left, jacobian_right = _anchor_residual(
                knots,
                timestamps,
                interval,
                pose_measurement.timestamp_sec,
                anchor,
                pose_measurement.pose_world_body,
            )
            residuals.extend(float(value) for value in residual)
            weights.extend([pose_measurement.weight] * 3)
            blocks.append((row_offset, 6 * interval, jacobian_left))
            blocks.append((row_offset, 6 * (interval + 1), jacobian_right))
            row_offset += 3
    for prior in problem.knot_marginalization_priors:
        weighted_residual, weighted_jacobian = marginalization_prior_blocks(
            prior, knots[prior.knot_index]
        )
        for axis in range(6):
            residuals.append(float(weighted_residual[axis]))
            weights.append(1.0)
            blocks.append(
                (
                    row_offset,
                    6 * prior.knot_index,
                    weighted_jacobian[axis : axis + 1],
                )
            )
            row_offset += 1
    parameter_count = 6 * knot_count
    jacobian = _assemble_sparse_jacobian(
        blocks,
        residual_count=len(residuals),
        parameter_count=parameter_count,
    )
    residual_array = np.asarray(residuals, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    weighted_jacobian = jacobian.multiply(weight_array[:, None])
    normal = (weighted_jacobian.T @ weighted_jacobian).toarray()
    gradient = weighted_jacobian.T @ (weight_array * residual_array)
    return normal, np.asarray(gradient, dtype=float).reshape(-1)
