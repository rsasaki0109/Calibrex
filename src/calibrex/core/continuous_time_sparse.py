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
``s_a * (a_kin - R^T g) + b``.  An optional IMU mounting rotation ``R_body_imu``
maps IMU-frame vectors into the body frame: gyro readings are
``R_body_imu^T omega_body`` and specific force is
``s_a * (R_body_imu^T (a_kin - R^T g)) + b``, with gyro and accelerometer
biases and scales in the IMU frame.  Without it the IMU axes are the body axes.

The normal equations are assembled as a sparse block matrix.  Point, plane,
and pose factors touch the two knots bracketing their timestamp; gyro factors
touch the knots of both endpoints plus the optional bias and clock offset.
Accelerometer factors touch the union of knots supporting all three samples
of the discrete body-motion stencil, plus their optional calibration states.
Gauss--Newton with Levenberg--Marquardt damping solves the linear system on
the manifold.

Every measurement ``weight`` is scalar square-root information, typically
``1 / sigma``. Both the normal equations and the acceptance objective use
the whitened residual ``weight * residual``. Marginalization priors already
store information and contribute ``delta.T @ information @ delta``.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import pairwise
from struct import pack
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from calibrex.core._se3_batch import _point_jacobians, _sample_segment, _transform_points
from calibrex.core.clock_domain import ClockDomainEvidence, ClockOffsetDomain
from calibrex.core.continuous_time_marginalization import (
    JointKnotMarginalizationPrior,
    KnotMarginalizationPrior,
    joint_marginalization_prior_blocks,
    joint_marginalization_prior_residual,
    marginalization_prior_blocks,
    marginalization_prior_residual,
)
from calibrex.core.fit_diagnostics import (
    FitDiagnostics,
    FitTermination,
    normalized_gradient_norm,
    normalized_step_norm,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import (
    _cross3,
    _rotation_log,
    _ScrewSegment,
    interpolate_screw_jacobians,
    point_transform_jacobian,
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
    se3_exp,
    se3_log,
    skew,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
)

FloatArray: TypeAlias = NDArray[np.float64]

FitStatus = Literal["converged", "max_iterations", "singular_system", "stalled"]


@dataclass(frozen=True)
class _BodyStencil:
    times: tuple[float, float, float]
    poses: tuple[SE3, SE3, SE3]
    log_plus: FloatArray
    log_minus: FloatArray
    acceleration_world: FloatArray
    motion: tuple[FloatArray, FloatArray, FloatArray, FloatArray]


class _ScrewInterpolation:
    """Borrow fixed-state samples during one objective or model evaluation.

    Snapshot the immutable poses; never retain this workspace across updates.
    Objective-only samples prepare no Jacobians. Arrays remain private and
    read-only, so sharing a sample cannot modify another factor's derivative.
    """

    def __init__(self, knots: list[SE3]) -> None:
        self.knots = tuple(knots)
        self._segments: dict[int, _ScrewSegment] = {}
        self._poses: dict[tuple[int, bytes], SE3] = {}
        self._jacobians: dict[tuple[int, bytes], tuple[FloatArray, FloatArray]] = {}
        self.motion: dict[
            bytes, tuple[FloatArray, FloatArray, FloatArray, FloatArray]
        ] = {}
        self.stencils: dict[bytes, _BodyStencil] = {}

    def _segment(self, interval: int) -> _ScrewSegment:
        if interval not in self._segments:
            self._segments[interval] = _ScrewSegment(
                self.knots[interval], self.knots[interval + 1]
            )
        return self._segments[interval]

    def pose(self, interval: int, alpha: float) -> SE3:
        key = (interval, pack("!d", alpha))  # Preserve the sign of zero, too.
        if key not in self._poses:
            self._poses[key] = self._segment(interval).pose(alpha)
        return self._poses[key]

    def jacobians(self, interval: int, alpha: float) -> tuple[FloatArray, FloatArray]:
        key = (interval, pack("!d", alpha))
        if key not in self._jacobians:
            pair = self._segment(interval).jacobians(alpha)
            for array in pair:
                array.setflags(write=False)
            self._jacobians[key] = pair
        return self._jacobians[key]

    def prepare(self, requests: list[tuple[int, float, bool]]) -> None:
        """Batch known queries; scalar fallbacks remain lazy for other times."""
        pending: dict[tuple[int, bytes], tuple[float, bool]] = {}
        for interval, alpha, jacobians in requests:
            key = (interval, pack("!d", alpha))
            if key in pending:
                pending[key] = alpha, jacobians or pending[key][1]
            elif key not in self._poses or (jacobians and key not in self._jacobians):
                pending[key] = alpha, jacobians
        groups: dict[tuple[int, bool], list[tuple[tuple[int, bytes], float]]] = {}
        for key, (alpha, jacobians) in pending.items():
            groups.setdefault((key[0], jacobians), []).append((key, alpha))
        for (interval, jacobians), samples in groups.items():
            if len(samples) < 8:
                for _key, alpha in samples:
                    self.pose(interval, alpha)
                    if jacobians:
                        self.jacobians(interval, alpha)
                continue
            for offset in range(0, len(samples), 256):
                chunk = samples[offset:offset + 256]
                alphas = np.asarray([alpha for _key, alpha in chunk], dtype=float)
                poses, pair = _sample_segment(self._segment(interval), alphas, jacobians=jacobians)
                for i, ((key, _alpha_value), pose) in enumerate(zip(chunk, poses, strict=True)):
                    self._poses.setdefault(key, pose)
                    if pair is not None:
                        blocks = pair[0][i], pair[1][i]
                        for array in blocks:
                            array.setflags(write=False)
                        self._jacobians[key] = blocks


@dataclass(frozen=True)
class TrajectoryPointMeasurement:
    """A body-frame point against a world target; ``weight`` is inverse sigma."""

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
    """A LiDAR return against a world plane; ``weight`` is inverse distance sigma."""

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
    """A measured world pose; ``weight`` whitens each of its four anchor residuals."""

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
    """Gyro pre-integration; ``weight`` is inverse rotation-residual sigma."""

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
    """Specific force observing the IMU lever arm; ``weight`` is inverse acceleration sigma."""

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
    initial_imu_rotation_vector_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    estimate_gyro_bias: bool = False
    estimate_lever_arm: bool = False
    estimate_imu_clock_offset: bool = False
    estimate_accel_bias: bool = False
    estimate_gravity: bool = False
    estimate_gyro_scale: bool = False
    estimate_accel_scale: bool = False
    estimate_imu_rotation: bool = False
    knot_marginalization_priors: tuple[KnotMarginalizationPrior, ...] = ()
    interpolation: Literal["screw_linear"] = "screw_linear"
    joint_knot_marginalization_priors: tuple[JointKnotMarginalizationPrior, ...] = ()

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
            and not self.knot_marginalization_priors
            and not self.joint_knot_marginalization_priors
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
        if not all(math.isfinite(value) for value in self.initial_imu_rotation_vector_rad):
            raise ValueError("initial IMU rotation must be finite")
        if self.estimate_imu_rotation and not (
            self.imu_preintegration_measurements or self.imu_lever_arm_measurements
        ):
            raise ValueError("IMU rotation estimation requires IMU measurements")


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
    """Optimized knots and convergence evidence for one fit.

    ``gradient_norm`` is the infinity norm of the final weighted normal
    right-hand side, half the squared-objective gradient at the returned state.
    Small damped steps alone do not establish stationarity.
    ``frozen_parameter_indices`` records columns with exactly zero information
    at any iteration, not a claim that those coordinates remain unobservable.
    """

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
    imu_rotation_vector_rad: tuple[float, float, float] | None = None
    calibration_parameter_names: tuple[str, ...] = ()
    calibration_covariance: tuple[tuple[float, ...], ...] | None = None
    residual_variance: float | None = None
    diagnostics: FitDiagnostics | None = None
    frozen_parameter_indices: tuple[int, ...] = ()


_ANCHOR_POINTS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class _PlaneBatch:
    """Snapshot constant plane data; evaluate the original factor order in batches."""

    def __init__(
        self, timestamps: tuple[float, ...],
        rows: list[tuple[int, TrajectoryPointToPlaneMeasurement]],
    ) -> None:
        self.intervals = tuple(interval for interval, _measurement in rows)
        self.alphas = np.asarray([
            _alpha(timestamps, interval, measurement.timestamp_sec)
            for interval, measurement in rows
        ], dtype=float)
        self.points = np.asarray([
            m.point_body_m for _interval, m in rows
        ], dtype=float).reshape(-1, 3)
        self.targets = np.asarray([
            m.plane_point_world_m for _interval, m in rows
        ], dtype=float).reshape(-1, 3)
        normals = np.asarray([m.plane_normal_world for _interval, m in rows], dtype=float)
        self.normals = np.asarray([
            normal / float(np.linalg.norm(normal)) for normal in normals
        ], dtype=float).reshape(-1, 3)
        for array in (self.alphas, self.points, self.targets, self.normals):
            array.setflags(write=False)

    def prepare(self, interpolation: _ScrewInterpolation | None, *, jacobians: bool) -> None:
        if interpolation is not None:
            interpolation.prepare([
                (interval, float(alpha), jacobians)
                for interval, alpha in zip(self.intervals, self.alphas, strict=True)
            ])

    def values(
        self, knots: list[SE3], interpolation: _ScrewInterpolation | None,
    ) -> tuple[FloatArray, list[SE3]]:
        poses = [_interpolate_segment(knots, interval, float(alpha), interpolation)
                 for interval, alpha in zip(self.intervals, self.alphas, strict=True)]
        residuals = np.empty(len(poses), dtype=float)
        for offset in range(0, len(poses), 256):
            stop = offset + 256
            world = _transform_points(poses[offset:stop], self.points[offset:stop])
            residuals[offset:stop] = [
                float(normal @ (point - target))
                for normal, point, target in zip(
                    self.normals[offset:stop], world, self.targets[offset:stop], strict=True,
                )
            ]
        return residuals, poses

    def linearize(
        self, knots: list[SE3], interpolation: _ScrewInterpolation | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        residuals, poses = self.values(knots, interpolation)
        if not poses:
            return residuals, np.empty((0, 1, 6)), np.empty((0, 1, 6))
        left = np.empty((len(poses), 1, 6), dtype=float)
        right = np.empty_like(left)
        for offset in range(0, len(poses), 256):
            stop = offset + 256
            signed = self.normals[offset:stop, None, :] @ _point_jacobians(
                poses[offset:stop], self.points[offset:stop],
            )
            pairs = [_interpolation_jacobians(knots, interval, float(alpha), interpolation)
                     for interval, alpha in zip(
                         self.intervals[offset:stop], self.alphas[offset:stop], strict=True,
                     )]
            left[offset:stop] = signed @ np.asarray([pair[0] for pair in pairs])
            right[offset:stop] = signed @ np.asarray([pair[1] for pair in pairs])
        return residuals, left, right


def _plane_batch(
    timestamps: tuple[float, ...], rows: list[tuple[int, TrajectoryPointToPlaneMeasurement]],
) -> _PlaneBatch | None:
    return _PlaneBatch(timestamps, rows) if len(rows) >= 8 else None


def _prepare_evaluation(
    interpolation: _ScrewInterpolation | None,
    problem: ContinuousTimeTrajectoryFitProblem,
    point_rows: list[tuple[int, tuple[float, float, float]]],
    plane_rows: list[tuple[int, TrajectoryPointToPlaneMeasurement]],
    pose_rows: list[tuple[int, SE3]],
    imu_rows: list[TrajectoryImuPreintegrationMeasurement],
    lever_rows: list[TrajectoryImuLeverArmMeasurement],
    clock_offset: float,
    *,
    jacobians: bool,
    clock_step: float = 1.0e-6,
) -> None:
    if interpolation is None:
        return
    timestamps = problem.knot_timestamps
    requests: list[tuple[int, float, bool]] = []

    def request(time: float, derivative: bool) -> None:
        interval = _interval_index(timestamps, time)
        if interval is not None:
            requests.append((interval, _alpha(timestamps, interval, time), derivative))
        # Invalid shifted times still reach the original factor's error path.

    for index, (interval, _point) in enumerate(point_rows):
        time = problem.point_measurements[index].timestamp_sec
        requests.append((interval, _alpha(timestamps, interval, time), jacobians))
    for interval, measurement in plane_rows:
        requests.append((
            interval, _alpha(timestamps, interval, measurement.timestamp_sec), jacobians,
        ))
    for index, (interval, _pose) in enumerate(pose_rows):
        time = problem.pose_measurements[index].timestamp_sec
        requests.append((interval, _alpha(timestamps, interval, time), jacobians))
    for gyro in imu_rows:
        for time in (gyro.start_time_sec, gyro.end_time_sec):
            body_time = _imu_body_time(time, clock_offset)
            request(body_time, jacobians)
    for lever in lever_rows:
        offsets = (clock_offset, clock_offset + clock_step) if jacobians else (clock_offset,)
        for index, offset in enumerate(offsets):
            try:
                times = _body_motion_sample_times(
                    timestamps, _imu_body_time(lever.timestamp_sec, offset), 1.0e-3,
                )
            except ValueError:
                continue
            for time in times:
                request(time, jacobians and index == 0)
    interpolation.prepare(requests)


def fit_continuous_trajectory(
    problem: ContinuousTimeTrajectoryFitProblem,
    options: ContinuousTimeTrajectoryFitOptions | None = None,
) -> ContinuousTimeTrajectoryFitResult:
    """Minimize ``sum(||weight * residual||^2)`` plus information-form priors."""

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
    imu_rotation_vector = np.asarray(problem.initial_imu_rotation_vector_rad, dtype=float)
    estimate_bias = problem.estimate_gyro_bias
    estimate_lever = problem.estimate_lever_arm
    estimate_clock = problem.estimate_imu_clock_offset
    estimate_accel = problem.estimate_accel_bias
    estimate_grav = problem.estimate_gravity
    estimate_gyro_scale = problem.estimate_gyro_scale
    estimate_accel_scale = problem.estimate_accel_scale
    estimate_imu_rot = problem.estimate_imu_rotation
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

    clock_domain = None
    rejected_clock_steps = 0
    if estimate_clock:
        observed_times = [m.start_time_sec for m in imu_rows]
        observed_times.extend(m.end_time_sec for m in imu_rows)
        observed_times.extend(m.timestamp_sec for m in lever_rows)
        clock_domain = ClockOffsetDomain(
            timestamps[0], timestamps[-1], min(observed_times), max(observed_times),
        )
        if not clock_domain.contains(clock_offset):
            raise ValueError("initial IMU clock offset is outside the shared time domain")

    damping = settings.initial_damping
    gradient_norm = math.inf
    plane_batch = _plane_batch(timestamps, plane_rows)
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
        imu_rotation=_imu_rotation_matrix(imu_rotation_vector),
        plane_batch=plane_batch,
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
    best_imu_rotation = np.array(imu_rotation_vector, copy=True)
    max_step_translation = 0.0
    max_step_rotation_deg = 0.0
    status: FitStatus = "max_iterations"
    termination: FitTermination = "iteration_limit"
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
        + (3 if estimate_imu_rot else 0)
    )
    bias_offset = 6 * knot_count
    lever_offset = bias_offset + (3 if estimate_bias else 0)
    clock_offset_index = lever_offset + (3 if estimate_lever else 0)
    accel_offset = clock_offset_index + (1 if estimate_clock else 0)
    gravity_offset = accel_offset + (3 if estimate_accel else 0)
    gyro_scale_offset = gravity_offset + (3 if estimate_grav else 0)
    accel_scale_offset = gyro_scale_offset + (3 if estimate_gyro_scale else 0)
    imu_rotation_offset = accel_scale_offset + (3 if estimate_accel_scale else 0)
    blocks: list[tuple[int, int, NDArray[np.float64]]] = []

    def _linearize(
        knots: list[SE3],
        gyro_bias: FloatArray,
        lever_arm: FloatArray,
        clock_offset: float,
        accel_bias: FloatArray,
        gravity: FloatArray,
        gyro_scale: FloatArray,
        accel_scale: FloatArray,
        imu_rotation_vector: FloatArray,
        clock_step_override: float | None = None,
    ) -> tuple[
        list[tuple[int, int, NDArray[np.float64]]], FloatArray, FloatArray, csc_matrix
    ]:
        interpolation = _ScrewInterpolation(knots)
        clock_step = (clock_domain.difference_step(clock_offset)
                      if clock_domain is not None else 1.0e-6)
        if clock_step_override is not None:
            clock_step = clock_step_override
        clock_side: Literal["left", "right"] | None = (
            ("left" if clock_step > 0 else "right") if clock_domain is not None else None
        )
        _prepare_evaluation(
            interpolation, problem, point_rows, plane_rows, pose_rows, imu_rows, lever_rows,
            clock_offset, jacobians=True, clock_step=clock_step,
        )
        imu_rotation = _imu_rotation_matrix(imu_rotation_vector)
        blocks: list[tuple[int, int, NDArray[np.float64]]] = []
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
                interpolation=interpolation,
            )
            residuals.extend(float(value) for value in residual)
            weights.extend([measurement.weight] * 3)
            blocks.append((row_offset, 6 * interval, jacobian_left))
            blocks.append((row_offset, 6 * (interval + 1), jacobian_right))
            row_offset += 3
        plane_values = plane_batch.linearize(knots, interpolation) if plane_batch else None
        for index, (interval, plane_measurement) in enumerate(plane_rows):
            if plane_values is None:
                residual, jacobian_left, jacobian_right = _point_to_plane_residual(
                    knots, timestamps, interval, plane_measurement, interpolation=interpolation,
                )
                value = float(residual[0])
            else:
                value = float(plane_values[0][index])
                jacobian_left, jacobian_right = plane_values[1][index], plane_values[2][index]
            residuals.append(value)
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
                    interpolation=interpolation,
                )
                residuals.extend(float(value) for value in residual)
                weights.extend([weight] * 3)
                blocks.append((row_offset, 6 * interval, jacobian_left))
                blocks.append((row_offset, 6 * (interval + 1), jacobian_right))
                row_offset += 3
        for imu_measurement in imu_rows:
            factor = _linearize_imu_preintegration(
                knots,
                timestamps,
                imu_measurement,
                gyro_bias,
                clock_offset,
                gyro_scale,
                imu_rotation=imu_rotation,
                interpolation=interpolation,
                rotation_jacobian=estimate_imu_rot,
                clock_side=clock_side,
            )
            residuals.extend(float(value) for value in factor.residual)
            weights.extend([imu_measurement.weight] * 3)
            if estimate_imu_rot:
                assert factor.jacobian_rotation is not None
                blocks.append(
                    (
                        row_offset,
                        imu_rotation_offset,
                        factor.jacobian_rotation,
                    )
                )
            for knot_index, jacobian in factor.knot_blocks:
                blocks.append((row_offset, 6 * knot_index, jacobian))
            if estimate_bias:
                blocks.append((row_offset, 6 * knot_count, factor.jacobian_bias))
            if estimate_clock:
                blocks.append((row_offset, clock_offset_index, factor.jacobian_clock))
            if estimate_gyro_scale:
                blocks.append((row_offset, gyro_scale_offset, factor.jacobian_scale))
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
                imu_rotation=imu_rotation,
                interpolation=interpolation,
                clock_step=clock_step,
            )
            residuals.extend(float(value) for value in residual)
            weights.extend([lever_measurement.weight] * 3)
            if estimate_imu_rot:
                blocks.append(
                    (
                        row_offset,
                        imu_rotation_offset,
                        _lever_arm_rotation_jacobian(
                            knots,
                            timestamps,
                            lever_measurement,
                            lever_arm,
                            clock_offset,
                            gravity,
                            accel_scale,
                            _rotation_or_identity(imu_rotation),
                            interpolation=interpolation,
                        ),
                    )
                )
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
            for knot_index, jacobian in _lever_arm_knot_jacobians(
                knots,
                timestamps,
                lever_measurement,
                lever_arm,
                clock_offset,
                gravity,
                accel_scale,
                imu_rotation=imu_rotation,
                interpolation=interpolation,
            ):
                blocks.append((row_offset, 6 * knot_index, jacobian))
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

        for joint_prior in problem.joint_knot_marginalization_priors:
            prior_residual, prior_blocks = joint_marginalization_prior_blocks(joint_prior, knots)
            residuals.extend(float(value) for value in prior_residual)
            weights.extend([1.0] * len(prior_residual))
            for index, block in prior_blocks:
                blocks.append((row_offset, 6 * index, block))
            row_offset += len(prior_residual)

        residual_array = np.asarray(residuals, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        jacobian = _assemble_sparse_jacobian(
            blocks,
            residual_count=len(residuals),
            parameter_count=parameter_count,
        )
        return blocks, residual_array, weight_array, jacobian

    frozen_parameter_indices: set[int] = set()
    linearization_is_current = False
    for iteration in range(settings.max_iterations):
        iterations = iteration + 1
        blocks, residual_array, weight_array, jacobian = _linearize(
            knots,
            gyro_bias,
            lever_arm,
            clock_offset,
            accel_bias,
            gravity,
            gyro_scale,
            accel_scale,
            imu_rotation_vector,
        )
        weighted_jacobian = jacobian.multiply(weight_array[:, None])
        normal = weighted_jacobian.T @ weighted_jacobian
        rhs = weighted_jacobian.T @ (weight_array * residual_array)
        active_columns = normal.diagonal() > 0.
        frozen_parameter_indices.update(int(i) for i in np.flatnonzero(~active_columns))
        gradient_norm = float(np.max(np.abs(rhs)))
        linearization_is_current = True

        scaled_gradient = normalized_gradient_norm(
            rhs, normal.diagonal(), float(np.linalg.norm(weight_array * residual_array))
        )
        if scaled_gradient < settings.convergence_gradient_norm:
            status = "converged"
            termination = "gradient_tolerance"
            break

        damping_matrix = normal + damping * _block_diagonal_scale(
            normal, knot_count, parameter_count
        )
        try:
            if np.all(active_columns):
                delta = spsolve(damping_matrix.tocsc(), -rhs)
            else:
                # Exactly zero weighted Jacobian columns carry no information.
                # Keep their current values; do not add a prior or diagonal floor.
                reduced = damping_matrix.tocsc()[active_columns, :][:, active_columns]
                delta = np.zeros(parameter_count, dtype=float)
                delta[active_columns] = spsolve(reduced, -rhs[active_columns])
        except RuntimeError:
            status = "singular_system"
            termination = "singular_system"
            break
        if not np.all(np.isfinite(delta)):
            status = "singular_system"
            termination = "singular_system"
            break

        if clock_domain is not None and not clock_domain.contains(
            clock_offset + float(delta[clock_offset_index])
        ):
            # Reject the complete coupled candidate before evaluating any factor.
            # More damping may yield a feasible proposal; never clip timestamps.
            rejected_clock_steps += 1
            damping *= settings.damping_scale_up
            if damping > 1.0e12:
                status = "stalled"
                termination = "clock_domain_limit"
                break
            continue

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
        candidate_imu_rotation = np.array(imu_rotation_vector, copy=True)
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
        if estimate_imu_rot:
            candidate_imu_rotation = rotation_matrix_to_rotation_vector(
                rotation_vector_to_matrix(imu_rotation_vector)
                @ rotation_vector_to_matrix(
                    np.asarray(
                        delta[imu_rotation_offset : imu_rotation_offset + 3], dtype=float
                    )
                )
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
            imu_rotation=_imu_rotation_matrix(candidate_imu_rotation),
            plane_batch=plane_batch,
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
            linearization_is_current = False
            gyro_bias = candidate_bias
            lever_arm = candidate_lever
            clock_offset = candidate_clock
            accel_bias = candidate_accel
            gravity = candidate_gravity
            gyro_scale = candidate_gyro_scale
            accel_scale = candidate_accel_scale
            imu_rotation_vector = candidate_imu_rotation
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
                best_imu_rotation = np.array(imu_rotation_vector, copy=True)
            max_step_translation = max(max_step_translation, step_translation)
            max_step_rotation_deg = max(max_step_rotation_deg, step_rotation)
            damping = max(damping * settings.damping_scale_down, 1.0e-12)
            if normalized_step_norm(
                delta, normal.diagonal(), float(np.linalg.norm(weight_array * residual_array))
            ) < settings.convergence_step_norm:
                status = "stalled"
                termination = "small_step"
                break
        else:
            damping *= settings.damping_scale_up
            if damping > 1.0e12:
                status = "stalled"
                termination = "damping_limit"
                break

    if status != "converged" and best_objective < objective:
        knots = best_knots
        linearization_is_current = False
        gyro_bias = best_bias
        lever_arm = best_lever
        clock_offset = best_clock
        accel_bias = best_accel
        gravity = best_gravity
        gyro_scale = best_gyro_scale
        accel_scale = best_accel_scale
        imu_rotation_vector = best_imu_rotation
        objective = best_objective
    calibration_names: tuple[str, ...] = ()
    calibration_covariance: tuple[tuple[float, ...], ...] | None = None
    residual_variance: float | None = None
    layout = _calibration_layout(problem)
    if linearization_is_current:
        final_residuals, final_weights, final_jacobian = residual_array, weight_array, jacobian
    else:
        _final_blocks, final_residuals, final_weights, final_jacobian = _linearize(
            knots,
            gyro_bias,
            lever_arm,
            clock_offset,
            accel_bias,
            gravity,
            gyro_scale,
            accel_scale,
            imu_rotation_vector,
        )
        final_weighted_jacobian = final_jacobian.multiply(final_weights[:, None])
        final_rhs = final_weighted_jacobian.T @ (final_weights * final_residuals)
        gradient_norm = float(np.max(np.abs(final_rhs)))
    final_weighted = final_jacobian.multiply(final_weights[:, None])
    scaled_gradient = normalized_gradient_norm(
        final_weighted.T @ (final_weights * final_residuals),
        np.asarray(final_weighted.power(2).sum(axis=0)).ravel(),
        float(np.linalg.norm(final_weights * final_residuals)),
    )
    stationary = scaled_gradient < settings.convergence_gradient_norm
    opposite_gradient = None
    if stationary and clock_domain is not None and any(
        _imu_body_time(time, clock_offset) in timestamps[1:-1]
        for measurement in imu_rows
        for time in (measurement.start_time_sec, measurement.end_time_sec)
    ):
        # At an internal screw knot, one vanishing one-sided clock column can
        # hide descent on the other side. Check both sides when both are feasible.
        primary_step = clock_domain.difference_step(clock_offset)
        direction: Literal[-1, 1] = -1 if primary_step > 0 else 1
        if clock_domain.contains(math.nextafter(clock_offset, math.inf * direction)):
            opposite_step = clock_domain.difference_step(clock_offset, direction=direction)
            _other_blocks, other_residual, other_weights, other_jacobian = _linearize(
                knots, gyro_bias, lever_arm, clock_offset, accel_bias, gravity,
                gyro_scale, accel_scale, imu_rotation_vector,
                clock_step_override=opposite_step,
            )
            other_weighted = other_jacobian.multiply(other_weights[:, None])
            opposite_gradient = normalized_gradient_norm(
                other_weighted.T @ (other_weights * other_residual),
                np.asarray(other_weighted.power(2).sum(axis=0)).ravel(),
                float(np.linalg.norm(other_weights * other_residual)),
            )
            stationary = opposite_gradient < settings.convergence_gradient_norm
    if status == "converged" and not stationary:
        status = "stalled"
        termination = "stationarity_not_confirmed"
    if stationary and status != "singular_system":
        status = "converged"
        termination = "gradient_tolerance"
    diagnostics = FitDiagnostics(
        termination_reason=termination, normalized_gradient_norm=scaled_gradient,
        gradient_tolerance=settings.convergence_gradient_norm, stationary=stationary,
        clock_domain=(ClockDomainEvidence(
            lower_sec=clock_domain.lower, upper_sec=clock_domain.upper,
            rejected_steps=rejected_clock_steps,
            derivative_step_sec=clock_domain.difference_step(clock_offset),
            opposite_gradient_norm=opposite_gradient,
        ) if clock_domain is not None else None),
    )
    if layout and status != "singular_system":
        covariance, residual_variance = _marginal_calibration_covariance(
            final_jacobian, final_residuals, final_weights, knot_count
        )
        calibration_names = tuple(
            f"{name}[{axis}]" for name, width in layout for axis in range(width)
        )
        calibration_covariance = tuple(
            tuple(float(value) for value in row) for row in covariance
        )
    final_point_rmse = _point_rmse(problem, point_rows, knots)
    final_point_to_plane_rmse = _point_to_plane_rmse(plane_rows, timestamps, knots)
    final_pose_rmse = _pose_rmse(problem, pose_rows, knots)
    final_imu_rmse = imu_preintegration_rmse(
        tuple(imu_rows),
        timestamps,
        knots,
        gyro_bias,
        clock_offset,
        gyro_scale,
        imu_rotation_vector_rad=imu_rotation_vector,
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
        imu_rotation_vector_rad=imu_rotation_vector,
    )
    return ContinuousTimeTrajectoryFitResult(
        status=status,
        diagnostics=diagnostics,
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
        frozen_parameter_indices=tuple(sorted(frozen_parameter_indices)),
        imu_rotation_vector_rad=(
            (
                float(imu_rotation_vector[0]),
                float(imu_rotation_vector[1]),
                float(imu_rotation_vector[2]),
            )
            if imu_rows or lever_rows
            else None
        ),
        calibration_parameter_names=calibration_names,
        calibration_covariance=calibration_covariance,
        residual_variance=residual_variance,
    )


def _evaluate_point(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    timestamp_sec: float,
    point_body: tuple[float, float, float],
    target_world: tuple[float, float, float],
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[FloatArray, SE3, float]:
    """Evaluate a point without derivatives; shared by scoring and linearization."""

    alpha = _alpha(timestamps, interval, timestamp_sec)
    pose = _interpolate_segment(knots, interval, alpha, interpolation)
    residual = np.asarray(pose.transform_point(point_body), dtype=float)
    residual -= np.asarray(target_world, dtype=float)
    return residual, pose, alpha


def _point_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    timestamp_sec: float,
    point_body: tuple[float, float, float],
    target_world: tuple[float, float, float],
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[FloatArray, NDArray[np.float64], NDArray[np.float64]]:
    residual, pose, alpha = _evaluate_point(
        knots, timestamps, interval, timestamp_sec, point_body, target_world,
        interpolation=interpolation,
    )
    point_jacobian = point_transform_jacobian(pose, np.asarray(point_body))
    jacobian_left, jacobian_right = _interpolation_jacobians(knots, interval, alpha, interpolation)
    return residual, point_jacobian @ jacobian_left, point_jacobian @ jacobian_right


def _evaluate_plane(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    measurement: TrajectoryPointToPlaneMeasurement,
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[FloatArray, SE3, FloatArray, float]:
    """Evaluate a signed distance with the original transform and dot order."""

    alpha = _alpha(timestamps, interval, measurement.timestamp_sec)
    pose = _interpolate_segment(knots, interval, alpha, interpolation)
    world = np.asarray(pose.transform_point(measurement.point_body_m), dtype=float)
    plane_point = np.asarray(measurement.plane_point_world_m, dtype=float)
    normal = np.asarray(measurement.plane_normal_world, dtype=float)
    normal = normal / float(np.linalg.norm(normal))
    residual = np.asarray([float(normal @ (world - plane_point))], dtype=float)
    return residual, pose, normal, alpha


def _point_to_plane_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    measurement: TrajectoryPointToPlaneMeasurement,
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[FloatArray, NDArray[np.float64], NDArray[np.float64]]:
    residual, pose, normal, alpha = _evaluate_plane(
        knots, timestamps, interval, measurement,
        interpolation=interpolation,
    )
    point_jacobian = point_transform_jacobian(
        pose, np.asarray(measurement.point_body_m)
    )
    signed_jacobian = normal.reshape(1, 3) @ point_jacobian
    jacobian_left, jacobian_right = _interpolation_jacobians(knots, interval, alpha, interpolation)
    return residual, signed_jacobian @ jacobian_left, signed_jacobian @ jacobian_right


def interpolate_pose_at(
    knots: tuple[SE3, ...] | list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
) -> SE3:
    """Interpolate a screw-linear pose at ``timestamp_sec``."""

    return _interpolated_pose_at(knots, timestamps, timestamp_sec)


def _interpolated_pose_at(
    knots: tuple[SE3, ...] | list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> SE3:
    interval = _interval_index(timestamps, timestamp_sec)
    if interval is None:
        raise ValueError("timestamp is outside the knot domain")
    alpha = _alpha(timestamps, interval, timestamp_sec)
    return _interpolate_segment(knots, interval, alpha, interpolation)


def point_to_plane_rmse(
    measurements: tuple[TrajectoryPointToPlaneMeasurement, ...],
    timestamps: tuple[float, ...],
    knots: tuple[SE3, ...] | list[SE3],
) -> float | None:
    """Return RMS signed distance of plane measurements against ``knots``."""

    if not measurements:
        return None
    poses = list(knots)
    if len(measurements) >= 8:
        rows = []
        for measurement in measurements:
            interval = _interval_index(timestamps, measurement.timestamp_sec)
            if interval is None:
                raise ValueError(
                    f"point-to-plane measurement {measurement.measurement_id!r} is "
                    "outside the knot domain"
                )
            rows.append((interval, measurement))
        return _point_to_plane_rmse(rows, timestamps, poses)
    interpolation = _ScrewInterpolation(poses)
    squared = 0.0
    for measurement in measurements:
        interval = _interval_index(timestamps, measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"point-to-plane measurement {measurement.measurement_id!r} is "
                "outside the knot domain"
            )
        residual, _pose, _normal, _fraction = _evaluate_plane(
            poses, timestamps, interval, measurement, interpolation=interpolation,
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
    *,
    imu_rotation_vector_rad: tuple[float, float, float] | FloatArray | None = None,
) -> float | None:
    """Return RMS rotation residual of IMU pre-integration factors.

    ``imu_rotation_vector_rad`` is the IMU mounting rotation ``R_body_imu``;
    ``None`` means the IMU axes are the body axes.
    """

    if not measurements:
        return None
    bias = np.asarray(gyro_bias_rad_s, dtype=float)
    scale = np.asarray(gyro_scale, dtype=float)
    imu_rotation = _optional_imu_rotation(imu_rotation_vector_rad)
    squared = 0.0
    poses = list(knots)
    interpolation = _ScrewInterpolation(poses)
    for measurement in measurements:
        residual = _evaluate_imu_preintegration(
            poses,
            timestamps,
            measurement,
            bias,
            clock_offset_sec,
            scale,
            imu_rotation=imu_rotation,
            interpolation=interpolation,
        ).residual
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
    *,
    imu_rotation_vector_rad: tuple[float, float, float] | FloatArray | None = None,
) -> float | None:
    """Return RMS specific-force residual of lever-arm measurements.

    ``imu_rotation_vector_rad`` is the IMU mounting rotation ``R_body_imu``;
    ``None`` means the IMU axes are the body axes.
    """

    if not measurements:
        return None
    lever = np.asarray(lever_arm_body_m, dtype=float)
    accel_bias = np.asarray(accel_bias_body_m_s2, dtype=float)
    gravity = np.asarray(gravity_world_m_s2, dtype=float)
    scale = np.asarray(accel_scale, dtype=float)
    imu_rotation = _optional_imu_rotation(imu_rotation_vector_rad)
    squared = 0.0
    poses = list(knots)
    interpolation = _ScrewInterpolation(poses)
    for measurement in measurements:
        residual, _kinematic, _omega, _alpha, _rotation = _evaluate_lever_arm(
            poses,
            timestamps,
            measurement,
            lever,
            clock_offset_sec,
            accel_bias,
            gravity,
            scale,
            imu_rotation=imu_rotation,
            interpolation=interpolation,
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
    *,
    imu_rotation: NDArray[np.float64] | None = None,
    interpolation: _ScrewInterpolation | None = None,
    clock_step: float = 1.0e-6,
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
            imu_rotation=imu_rotation,
            interpolation=interpolation,
        )
    )
    epsilon = clock_step
    perturbed, _kinematic, _omega, _alpha, _rotation = _evaluate_lever_arm(
        knots,
        timestamps,
        measurement,
        lever_arm,
        clock_offset_sec + epsilon,
        accel_bias,
        gravity,
        scale,
        imu_rotation=imu_rotation,
        interpolation=interpolation,
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
    *,
    imu_rotation: NDArray[np.float64] | None = None,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[
    FloatArray,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    residual, kinematic, omega, alpha, rotation = _evaluate_lever_arm(
        knots, timestamps, measurement, lever_arm, clock_offset_sec,
        accel_bias_body_m_s2, gravity_world_m_s2, accel_scale,
        imu_rotation=imu_rotation, interpolation=interpolation,
    )
    scale = np.asarray(accel_scale, dtype=float).reshape(3)
    jacobian_lever = skew(alpha) + skew(omega) @ skew(omega)
    jacobian_gravity = -np.diag(scale) @ rotation.T
    if imu_rotation is not None:
        jacobian_lever = imu_rotation.T @ jacobian_lever
        jacobian_gravity = -np.diag(scale) @ imu_rotation.T @ rotation.T
    jacobian_lever = scale[:, None] * jacobian_lever
    jacobian_accel: NDArray[np.float64] = np.eye(3, dtype=float)
    jacobian_accel_scale = np.diag(kinematic)
    return (
        residual,
        jacobian_lever,
        jacobian_accel,
        jacobian_gravity,
        jacobian_accel_scale,
    )


def _evaluate_lever_arm(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuLeverArmMeasurement,
    lever_arm: FloatArray,
    clock_offset_sec: float,
    accel_bias_body_m_s2: FloatArray,
    gravity_world_m_s2: FloatArray,
    accel_scale: FloatArray,
    *,
    imu_rotation: NDArray[np.float64] | None = None,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Evaluate the force without derivatives or a perturbed clock sample."""
    body_time = _imu_body_time(measurement.timestamp_sec, clock_offset_sec)
    omega, alpha, accel_origin, rotation = _body_motion(
        knots, timestamps, body_time,
        interpolation=interpolation,
    )
    lever = np.asarray(lever_arm, dtype=float).reshape(3)
    accel_bias = np.asarray(accel_bias_body_m_s2, dtype=float).reshape(3)
    gravity = np.asarray(gravity_world_m_s2, dtype=float).reshape(3)
    scale = np.asarray(accel_scale, dtype=float).reshape(3)
    kinematic = (
        accel_origin
        + _cross3(alpha, lever)
        + _cross3(omega, _cross3(omega, lever))
        - rotation.T @ gravity
    )
    if imu_rotation is not None:
        kinematic = imu_rotation.T @ kinematic
    predicted = scale * kinematic + accel_bias
    residual = predicted - np.asarray(measurement.accel_body_m_s2, dtype=float)
    return residual, kinematic, omega, alpha, rotation


def _body_motion_sample_times(
    timestamps: tuple[float, ...], timestamp_sec: float, epsilon: float,
) -> tuple[float, float, float]:
    """Share the clamped discrete stencil between predictions and derivatives."""

    if timestamps[-1] - timestamps[0] <= 4.0 * epsilon:
        raise ValueError("knot domain is too short for lever-arm kinematics")
    limited = min(max(timestamp_sec, timestamps[0] + 2.0 * epsilon), timestamps[-1] - 2.0 * epsilon)
    return limited, limited - epsilon, limited + epsilon


def _lever_arm_knot_jacobians(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuLeverArmMeasurement,
    lever_arm: FloatArray,
    clock_offset_sec: float,
    gravity_world_m_s2: FloatArray,
    accel_scale: FloatArray,
    *,
    imu_rotation: NDArray[np.float64] | None = None,
    interpolation: _ScrewInterpolation | None = None,
) -> list[tuple[int, NDArray[np.float64]]]:
    """Differentiate the existing discrete specific-force model w.r.t. knots.

    Each sampled pose has a right-tangent derivative with respect to its two
    supporting knots. Chain those through the same rotation logs and second
    position difference as ``_body_motion``. A stencil crossing a segment
    boundary needs contributions from both segments, even at an exact knot.
    """

    epsilon = 1.0e-3
    body_time = _imu_body_time(measurement.timestamp_sec, clock_offset_sec)
    stencil = _body_stencil(
        knots, timestamps, body_time, epsilon=epsilon, interpolation=interpolation,
    )
    center_time, minus_time, plus_time = stencil.times

    def sample(time: float) -> dict[int, NDArray[np.float64]]:
        interval = _interval_index(timestamps, time)
        assert interval is not None  # The shared stencil lies inside the knot domain.
        left, right = _interpolation_jacobians(
            knots, interval, _alpha(timestamps, interval, time), interpolation,
        )
        return {interval: left, interval + 1: right}

    center_blocks, minus_blocks, plus_blocks = (
        sample(time) for time in (center_time, minus_time, plus_time)
    )
    _center, minus, plus = stencil.poses
    omega, _angular_acceleration, acceleration_body, rc = stencil.motion
    rm, rp = _rotation_matrix(minus), _rotation_matrix(plus)
    log_plus, log_minus = stencil.log_plus, stencil.log_minus
    # This historical helper implements the right Exp differential (I - a[w]
    # + b[w]^2). Its inverse at +phi differentiates Log under a right update.
    log_jac_plus = so3_left_jacobian_inverse(log_plus) / epsilon
    log_jac_minus = so3_left_jacobian_inverse(log_minus) / epsilon
    relative_plus = rc.T @ rp
    relative_minus = rm.T @ rc
    gravity_body = rc.T @ gravity_world_m_s2
    lever = np.asarray(lever_arm, dtype=float).reshape(3)
    centrifugal = -skew(_cross3(omega, lever)) - skew(omega) @ skew(lever)
    output_frame = np.diag(accel_scale) @ _rotation_or_identity(imu_rotation).T
    blocks: list[tuple[int, NDArray[np.float64]]] = []
    zero = np.zeros((6, 6), dtype=float)
    for index in sorted(center_blocks.keys() | minus_blocks.keys() | plus_blocks.keys()):
        jc = center_blocks.get(index, zero)
        jm = minus_blocks.get(index, zero)
        jp = plus_blocks.get(index, zero)
        d_omega = log_jac_plus @ (jp[3:] - relative_plus.T @ jc[3:])
        d_omega_minus = log_jac_minus @ (jc[3:] - relative_minus.T @ jm[3:])
        d_alpha = (d_omega - d_omega_minus) / epsilon
        d_acc_world = (rp @ jp[:3] - 2.0 * rc @ jc[:3] + rm @ jm[:3]) / (epsilon * epsilon)
        d_acc_body = skew(acceleration_body) @ jc[3:] + rc.T @ d_acc_world
        d_kinematic = (
            d_acc_body - skew(lever) @ d_alpha + centrifugal @ d_omega
            - skew(gravity_body) @ jc[3:]
        )
        blocks.append((index, output_frame @ d_kinematic))
    return blocks


def _body_motion(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    *,
    epsilon: float = 1.0e-3,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.float64]]:
    key = pack("!dd", timestamp_sec, epsilon)
    if interpolation is not None and key in interpolation.motion:
        return interpolation.motion[key]
    return _body_stencil(
        knots, timestamps, timestamp_sec, epsilon=epsilon, interpolation=interpolation,
    ).motion


def _body_stencil(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    *,
    epsilon: float,
    interpolation: _ScrewInterpolation | None,
) -> _BodyStencil:
    key = pack("!dd", timestamp_sec, epsilon)
    if interpolation is not None and key in interpolation.stencils:
        return interpolation.stencils[key]
    center_time, minus_time, plus_time = _body_motion_sample_times(
        timestamps, timestamp_sec, epsilon,
    )
    pose = _interpolated_pose_at(knots, timestamps, center_time, interpolation=interpolation)
    pose_minus = _interpolated_pose_at(knots, timestamps, minus_time, interpolation=interpolation)
    pose_plus = _interpolated_pose_at(knots, timestamps, plus_time, interpolation=interpolation)
    log_plus = _rotation_log(pose, pose_plus)
    log_minus = _rotation_log(pose_minus, pose)
    omega = log_plus / epsilon
    omega_minus = log_minus / epsilon
    alpha = (omega - omega_minus) / epsilon
    origin = np.asarray(pose.translation_m, dtype=float)
    origin_minus = np.asarray(pose_minus.translation_m, dtype=float)
    origin_plus = np.asarray(pose_plus.translation_m, dtype=float)
    accel_world = (origin_plus - 2.0 * origin + origin_minus) / (epsilon * epsilon)
    rotation = _rotation_matrix(pose)
    accel_body = rotation.T @ accel_world
    result = omega, alpha, accel_body, rotation
    stencil = _BodyStencil(
        (center_time, minus_time, plus_time), (pose, pose_minus, pose_plus),
        log_plus, log_minus, accel_world, result,
    )
    if interpolation is not None:
        for array in (*result, log_plus, log_minus, accel_world):
            array.setflags(write=False)
        interpolation.motion[key] = result
        interpolation.stencils[key] = stencil
    return stencil


def predicted_imu_specific_force(
    knots: tuple[SE3, ...] | list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    lever_arm_body_m: tuple[float, float, float] | FloatArray,
    *,
    accel_bias_body_m_s2: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    gravity_world_m_s2: tuple[float, float, float] | FloatArray = (0.0, 0.0, 0.0),
    accel_scale: tuple[float, float, float] | FloatArray = (1.0, 1.0, 1.0),
    imu_rotation_vector_rad: tuple[float, float, float] | FloatArray | None = None,
) -> FloatArray:
    """Return IMU specific force ``s * (a_kin - R^T g) + b``.

    With ``imu_rotation_vector_rad`` (``R_body_imu``) the kinematic term is
    rotated into the IMU frame before the scale and bias apply; otherwise the
    result is in the body frame.
    """

    omega, alpha, accel_origin, rotation = _body_motion(
        list(knots), timestamps, timestamp_sec
    )
    lever = np.asarray(lever_arm_body_m, dtype=float).reshape(3)
    accel_bias = np.asarray(accel_bias_body_m_s2, dtype=float).reshape(3)
    gravity = np.asarray(gravity_world_m_s2, dtype=float).reshape(3)
    scale = np.asarray(accel_scale, dtype=float).reshape(3)
    kinematic = (
        accel_origin
        + _cross3(alpha, lever)
        + _cross3(omega, _cross3(omega, lever))
        - rotation.T @ gravity
    )
    imu_rotation = _optional_imu_rotation(imu_rotation_vector_rad)
    if imu_rotation is not None:
        kinematic = imu_rotation.T @ kinematic
    return scale * kinematic + accel_bias


@dataclass(frozen=True)
class _ImuPreintegrationValue:
    residual: FloatArray
    delta_imu: FloatArray
    delta_body: FloatArray
    rotation_start: FloatArray
    rotation_end: FloatArray


@dataclass(frozen=True)
class _ImuPreintegrationLinearization:
    residual: FloatArray
    knot_blocks: list[tuple[int, FloatArray]]
    jacobian_bias: FloatArray
    jacobian_clock: FloatArray
    jacobian_scale: FloatArray
    jacobian_rotation: FloatArray | None


def _evaluate_imu_preintegration(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuPreintegrationMeasurement,
    gyro_bias: FloatArray,
    clock_offset_sec: float,
    gyro_scale: FloatArray,
    *,
    imu_rotation: NDArray[np.float64] | None = None,
    interpolation: _ScrewInterpolation | None = None,
    delta_imu: FloatArray | None = None,
) -> _ImuPreintegrationValue:
    """Evaluate the rotation error, optionally borrowing this factor's integral."""
    start_time = _imu_body_time(measurement.start_time_sec, clock_offset_sec)
    end_time = _imu_body_time(measurement.end_time_sec, clock_offset_sec)
    start = _interpolated_pose_at(knots, timestamps, start_time, interpolation=interpolation)
    end = _interpolated_pose_at(knots, timestamps, end_time, interpolation=interpolation)
    rotation_start = _rotation_matrix(start)
    rotation_end = _rotation_matrix(end)
    if delta_imu is None:
        delta_imu = _integrate_gyro_rotation(measurement.gyro_samples, gyro_bias, gyro_scale)
    delta_body = delta_imu
    if imu_rotation is not None:
        delta_body = imu_rotation @ delta_imu @ imu_rotation.T
    rotation_error = delta_body.T @ rotation_start.T @ rotation_end
    residual = rotation_matrix_to_rotation_vector(rotation_error)
    return _ImuPreintegrationValue(
        residual, delta_imu, delta_body, rotation_start, rotation_end,
    )


def _imu_preintegration_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuPreintegrationMeasurement,
    gyro_bias: FloatArray,
    clock_offset_sec: float = 0.0,
    gyro_scale: FloatArray | None = None,
    *,
    imu_rotation: NDArray[np.float64] | None = None,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[
    FloatArray,
    list[tuple[int, NDArray[np.float64]]],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    factor = _linearize_imu_preintegration(
        knots, timestamps, measurement, gyro_bias, clock_offset_sec, gyro_scale,
        imu_rotation=imu_rotation, interpolation=interpolation,
    )
    return (
        factor.residual, factor.knot_blocks, factor.jacobian_bias,
        factor.jacobian_clock, factor.jacobian_scale,
    )


def _linearize_imu_preintegration(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuPreintegrationMeasurement,
    gyro_bias: FloatArray,
    clock_offset_sec: float = 0.0,
    gyro_scale: FloatArray | None = None,
    *,
    imu_rotation: NDArray[np.float64] | None = None,
    interpolation: _ScrewInterpolation | None = None,
    rotation_jacobian: bool = False,
    clock_side: Literal["left", "right"] | None = None,
) -> _ImuPreintegrationLinearization:
    """Prepare every requested block from one factor-local gyro integral."""
    scale = np.ones(3, dtype=float) if gyro_scale is None else np.asarray(
        gyro_scale, dtype=float
    ).reshape(3)
    start_time = _imu_body_time(measurement.start_time_sec, clock_offset_sec)
    end_time = _imu_body_time(measurement.end_time_sec, clock_offset_sec)
    delta_imu, jacobian_right_bias, jacobian_right_scale = _integrate_gyro(
        measurement.gyro_samples, gyro_bias, scale
    )
    value = _evaluate_imu_preintegration(
        knots, timestamps, measurement, gyro_bias, clock_offset_sec, scale,
        imu_rotation=imu_rotation, interpolation=interpolation, delta_imu=delta_imu,
    )
    delta_rotation, residual = value.delta_body, value.residual
    if imu_rotation is not None:
        # Gyro increments are in the IMU frame; conjugate them into the body frame.
        jacobian_right_bias = imu_rotation @ jacobian_right_bias
        jacobian_right_scale = imu_rotation @ jacobian_right_scale
    # The historically named V(phi) implements the right Exp differential.
    # Log(exp(a) exp(r)) uses V(-r)^-1; Log(exp(r) exp(a)) uses V(r)^-1.
    log_jacobian = so3_left_jacobian_inverse(-residual)
    jacobian_start_rot = -log_jacobian @ delta_rotation.T
    jacobian_end_rot = so3_left_jacobian_inverse(residual)
    jacobian_bias = -log_jacobian @ jacobian_right_bias
    jacobian_scale = -log_jacobian @ jacobian_right_scale
    # A positive clock update moves both endpoints to the left in body time.
    # At the first domain endpoint only the opposite clock side is feasible;
    # select that same side for both endpoints, including internal knot ties.
    side: Literal["left", "right"] = (
        clock_side if clock_side is not None
        else ("right" if start_time == timestamps[0] else "left")
    )
    omega_start = _angular_rate_at(
        knots, timestamps, start_time, side=side, interpolation=interpolation,
    )
    omega_end = _angular_rate_at(
        knots, timestamps, end_time, side=side, interpolation=interpolation,
    )
    jacobian_clock = (
        jacobian_start_rot @ (-omega_start) + jacobian_end_rot @ (-omega_end)
    ).reshape(3, 1)
    knot_blocks = _interpolate_rotation_blocks(
        knots,
        timestamps,
        start_time,
        jacobian_start_rot,
        interpolation=interpolation,
    )
    end_blocks = _interpolate_rotation_blocks(
        knots,
        timestamps,
        end_time,
        jacobian_end_rot,
        interpolation=interpolation,
    )
    combined: dict[int, NDArray[np.float64]] = {}
    for knot_index, jacobian in (*knot_blocks, *end_blocks):
        if knot_index in combined:
            combined[knot_index] = combined[knot_index] + jacobian
        else:
            combined[knot_index] = jacobian
    jacobian_rotation = None
    if rotation_jacobian:
        mounting = _rotation_or_identity(imu_rotation)
        mounting_log = log_jacobian
        if imu_rotation is None:
            # The previous separate mounting helper explicitly conjugated by I.
            # Retain its signed-zero behavior while reusing the same integral.
            mounted_delta = mounting @ delta_imu @ mounting.T
            mounted_residual = rotation_matrix_to_rotation_vector(
                mounted_delta.T @ value.rotation_start.T @ value.rotation_end
            )
            mounting_log = so3_left_jacobian_inverse(-mounted_residual)
        jacobian_rotation = np.asarray(
            -mounting_log @ mounting @ (delta_imu.T - np.eye(3, dtype=float)),
            dtype=float,
        )
    return _ImuPreintegrationLinearization(
        residual, sorted(combined.items()), jacobian_bias, jacobian_clock,
        jacobian_scale, jacobian_rotation,
    )


def _imu_body_time(imu_time_sec: float, clock_offset_sec: float) -> float:
    return imu_time_sec - clock_offset_sec


def _imu_rotation_matrix(rotation_vector: FloatArray) -> NDArray[np.float64] | None:
    """Return ``R_body_imu``, or ``None`` when the IMU axes are the body axes."""

    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    if not np.any(vector):
        return None
    return rotation_vector_to_matrix(vector)


def _optional_imu_rotation(
    rotation_vector: tuple[float, float, float] | FloatArray | None,
) -> NDArray[np.float64] | None:
    if rotation_vector is None:
        return None
    return _imu_rotation_matrix(np.asarray(rotation_vector, dtype=float))


def _rotation_or_identity(rotation: NDArray[np.float64] | None) -> NDArray[np.float64]:
    return np.eye(3, dtype=float) if rotation is None else rotation


def _imu_preintegration_rotation_jacobian(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuPreintegrationMeasurement,
    gyro_bias: FloatArray,
    clock_offset_sec: float,
    gyro_scale: FloatArray,
    imu_rotation: NDArray[np.float64],
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> NDArray[np.float64]:
    """Jacobian of the pre-integration residual for ``R_body_imu exp(delta)``.

    With ``dR_body = R dR_imu R^T`` the right perturbation gives
    ``dR_body exp(R (dR_imu^T - I) delta)``, so the residual moves by
    ``-J_l^{-1}(r) R (dR_imu^T - I) delta``.  ``so3_left_jacobian_inverse(-r)``
    is ``J_l^{-1}(r)`` (``log(exp(a) exp(r)) ~= r + J(-r) a``).
    """

    factor = _linearize_imu_preintegration(
        knots,
        timestamps,
        measurement,
        gyro_bias,
        clock_offset_sec,
        gyro_scale,
        imu_rotation=imu_rotation,
        interpolation=interpolation,
        rotation_jacobian=True,
    )
    assert factor.jacobian_rotation is not None
    return factor.jacobian_rotation


def _lever_arm_rotation_jacobian(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    measurement: TrajectoryImuLeverArmMeasurement,
    lever_arm: FloatArray,
    clock_offset_sec: float,
    gravity_world_m_s2: FloatArray,
    accel_scale: FloatArray,
    imu_rotation: NDArray[np.float64],
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> NDArray[np.float64]:
    """Jacobian of the specific-force residual for ``R_body_imu exp(delta)``.

    ``exp(delta)^T R^T k ~= R^T k + [R^T k]_x delta`` for the body-frame
    kinematic term ``k``, scaled per IMU axis.
    """

    body_time = _imu_body_time(measurement.timestamp_sec, clock_offset_sec)
    omega, alpha, accel_origin, rotation = _body_motion(
        knots, timestamps, body_time,
        interpolation=interpolation,
    )
    lever = np.asarray(lever_arm, dtype=float).reshape(3)
    gravity = np.asarray(gravity_world_m_s2, dtype=float).reshape(3)
    scale = np.asarray(accel_scale, dtype=float).reshape(3)
    kinematic = (
        accel_origin
        + _cross3(alpha, lever)
        + _cross3(omega, _cross3(omega, lever))
        - rotation.T @ gravity
    )
    return np.asarray(np.diag(scale) @ skew(imu_rotation.T @ kinematic), dtype=float)


def _angular_rate_at(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    *,
    side: Literal["left", "right"] = "left",
    interpolation: _ScrewInterpolation | None = None,
) -> FloatArray:
    """Return the interval's body angular velocity, with explicit knot limits.

    Screw rotation is ``R_left Exp(alpha * w)``, so its body rate is exactly
    ``w / dt``. At a knot, ``left`` is the positive-clock limit; domain endpoints
    use their single inward interval. No finite time stencil or knot snapping
    is used, including for representable times immediately beside a knot.
    """
    if not timestamps[0] <= timestamp_sec <= timestamps[-1]:
        raise ValueError("timestamp is outside the knot domain")
    search = bisect_left if side == "left" else bisect_right
    interval = max(0, min(search(timestamps, timestamp_sec) - 1, len(timestamps) - 2))
    rotation_vector = (
        _rotation_log(knots[interval], knots[interval + 1])
        if interpolation is None else interpolation._segment(interval).u[3:]
    )
    return rotation_vector / (timestamps[interval + 1] - timestamps[interval])


def _interpolate_rotation_blocks(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    timestamp_sec: float,
    jacobian_rotation: NDArray[np.float64],
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> list[tuple[int, NDArray[np.float64]]]:
    interval = _interval_index(timestamps, timestamp_sec)
    if interval is None:
        raise ValueError("timestamp is outside the knot domain")
    alpha = _alpha(timestamps, interval, timestamp_sec)
    jacobian_left, jacobian_right = _interpolation_jacobians(knots, interval, alpha, interpolation)
    selector = jacobian_rotation @ np.eye(6, dtype=float)[3:, :]
    return [
        (interval, selector @ jacobian_left),
        (interval + 1, selector @ jacobian_right),
    ]


def _integrate_gyro_rotation(
    samples: tuple[TrajectoryImuGyroSample, ...],
    gyro_bias: FloatArray,
    gyro_scale: FloatArray,
) -> FloatArray:
    """Keep the original product order without storing or differentiating increments."""
    scale = np.asarray(gyro_scale, dtype=float).reshape(3)
    bias = np.asarray(gyro_bias, dtype=float).reshape(3)
    delta_rotation: FloatArray = np.eye(3, dtype=float)
    for left, right in pairwise(samples):
        dt = right.timestamp_sec - left.timestamp_sec
        raw = np.asarray(left.omega_body_rad_s, dtype=float) - bias
        omega = scale * raw
        phi = omega * dt
        rotation = rotation_vector_to_matrix(phi)
        delta_rotation = delta_rotation @ rotation
    return delta_rotation


def _integrate_gyro(
    samples: tuple[TrajectoryImuGyroSample, ...],
    gyro_bias: FloatArray,
    gyro_scale: FloatArray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the gyro product and its right-tangent bias/scale derivatives.

    An interval contributes ``Exp(phi)`` with ``phi = scale * (raw - bias) * dt``.
    Chain the right Exp differential through each following rotation. The bias
    perturbation includes the diagonal scale in the IMU frame.
    """

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
    jacobian_scale: NDArray[np.float64] = np.zeros((3, 3), dtype=float)
    suffix: NDArray[np.float64] = np.eye(3, dtype=float)
    for rotation, phi, dt, raw in zip(
        reversed(rotations), reversed(phis), reversed(dts), reversed(raws), strict=True
    ):
        right_exp_derivative = suffix.T @ so3_left_jacobian(phi)
        jacobian_bias += right_exp_derivative @ np.diag(-scale * dt)
        jacobian_scale += right_exp_derivative @ np.diag(raw * dt)
        suffix = rotation @ suffix
    return delta_rotation, jacobian_bias, jacobian_scale


def _rotation_matrix(pose: SE3) -> NDArray[np.float64]:
    return rotation_vector_to_matrix(so3_log(np.asarray(pose.rotation_quat_xyzw)))


def _evaluate_anchor(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    timestamp_sec: float,
    anchor: tuple[float, float, float],
    pose: SE3,
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[FloatArray, SE3, float]:
    """Evaluate a pose anchor without interpolation or point Jacobians."""

    alpha = _alpha(timestamps, interval, timestamp_sec)
    interpolated = _interpolate_segment(knots, interval, alpha, interpolation)
    target = pose.transform_point(anchor)
    residual = np.asarray(interpolated.transform_point(anchor), dtype=float)
    residual -= np.asarray(target, dtype=float)
    return residual, interpolated, alpha


def _anchor_residual(
    knots: list[SE3],
    timestamps: tuple[float, ...],
    interval: int,
    timestamp_sec: float,
    anchor: tuple[float, float, float],
    pose: SE3,
    *,
    interpolation: _ScrewInterpolation | None = None,
) -> tuple[FloatArray, NDArray[np.float64], NDArray[np.float64]]:
    residual, interpolated, alpha = _evaluate_anchor(
        knots, timestamps, interval, timestamp_sec, anchor, pose, interpolation=interpolation
    )
    point_jacobian = point_transform_jacobian(interpolated, np.asarray(anchor))
    jacobian_left, jacobian_right = _interpolation_jacobians(knots, interval, alpha, interpolation)
    return residual, point_jacobian @ jacobian_left, point_jacobian @ jacobian_right


def _interpolate(left: SE3, right: SE3, alpha: float) -> SE3:
    return se3_exp(left, alpha * se3_log(left, right))


def _interpolate_segment(
    knots: tuple[SE3, ...] | list[SE3],
    interval: int,
    alpha: float,
    interpolation: _ScrewInterpolation | None,
) -> SE3:
    if interpolation is not None:
        return interpolation.pose(interval, alpha)
    return _interpolate(knots[interval], knots[interval + 1], alpha)


def _interpolation_jacobians(
    knots: list[SE3],
    interval: int,
    alpha: float,
    interpolation: _ScrewInterpolation | None,
) -> tuple[FloatArray, FloatArray]:
    if interpolation is not None:
        return interpolation.jacobians(interval, alpha)
    return interpolate_screw_jacobians(knots[interval], knots[interval + 1], alpha)


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
    *,
    imu_rotation: NDArray[np.float64] | None = None,
    plane_batch: _PlaneBatch | None = None,
) -> float:
    interpolation = _ScrewInterpolation(knots)
    _prepare_evaluation(
        interpolation, problem, point_rows, plane_rows, pose_rows, imu_rows, lever_rows,
        clock_offset_sec, jacobians=False,
    )
    objective = 0.0
    for index, (interval, point_body) in enumerate(point_rows):
        point_measurement = problem.point_measurements[index]
        residual, _pose, _alpha_value = _evaluate_point(
            knots,
            problem.knot_timestamps,
            interval,
            point_measurement.timestamp_sec,
            point_body,
            point_measurement.target_world_m,
            interpolation=interpolation,
        )
        objective += point_measurement.weight**2 * float(np.sum(residual * residual))
    if plane_batch is None:
        plane_batch = _plane_batch(problem.knot_timestamps, plane_rows)
    plane_values = plane_batch.values(knots, interpolation)[0] if plane_batch else None
    for index, (interval, plane_measurement) in enumerate(plane_rows):
        if plane_values is None:
            residual, _pose, _normal, _alpha_value = _evaluate_plane(
                knots, problem.knot_timestamps, interval, plane_measurement,
                interpolation=interpolation,
            )
            value = residual[0]
        else:
            value = plane_values[index]
        objective += plane_measurement.weight**2 * float(value * value)
    for index, (interval, pose) in enumerate(pose_rows):
        pose_measurement = problem.pose_measurements[index]
        weight = pose_measurement.weight
        for anchor in _ANCHOR_POINTS:
            residual, _pose, _alpha_value = _evaluate_anchor(
                knots,
                problem.knot_timestamps,
                interval,
                pose_measurement.timestamp_sec,
                anchor,
                pose,
                interpolation=interpolation,
            )
            objective += weight**2 * float(np.sum(residual * residual))
    for imu_measurement in imu_rows:
        try:
            residual = _evaluate_imu_preintegration(
                knots,
                problem.knot_timestamps,
                imu_measurement,
                gyro_bias,
                clock_offset_sec,
                gyro_scale,
                imu_rotation=imu_rotation,
                interpolation=interpolation,
            ).residual
        except ValueError:
            return math.inf
        objective += imu_measurement.weight**2 * float(np.sum(residual * residual))
    for lever_measurement in lever_rows:
        try:
            residual, _kinematic, _omega, _alpha, _rotation = _evaluate_lever_arm(
                knots,
                problem.knot_timestamps,
                lever_measurement,
                lever_arm,
                clock_offset_sec,
                accel_bias,
                gravity,
                accel_scale,
                imu_rotation=imu_rotation,
                interpolation=interpolation,
            )
        except ValueError:
            return math.inf
        objective += lever_measurement.weight**2 * float(np.sum(residual * residual))
    for prior in problem.knot_marginalization_priors:
        residual = marginalization_prior_residual(prior, knots[prior.knot_index])
        objective += float(residual @ residual)
    for joint_prior in problem.joint_knot_marginalization_priors:
        residual = joint_marginalization_prior_residual(joint_prior, knots)
        objective += float(residual @ residual)
    return objective


def _point_rmse(
    problem: ContinuousTimeTrajectoryFitProblem,
    point_rows: list[tuple[int, tuple[float, float, float]]],
    knots: list[SE3],
) -> float | None:
    if not point_rows:
        return None
    interpolation = _ScrewInterpolation(knots)
    squared = 0.0
    count = 0
    for index, (interval, point_body) in enumerate(point_rows):
        measurement = problem.point_measurements[index]
        residual, _pose, _fraction = _evaluate_point(
            knots,
            problem.knot_timestamps,
            interval,
            measurement.timestamp_sec,
            point_body,
            measurement.target_world_m,
            interpolation=interpolation,
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
    interpolation = _ScrewInterpolation(knots)
    batch = _plane_batch(timestamps, plane_rows)
    if batch is not None:
        batch.prepare(interpolation, jacobians=False)
        values, _poses = batch.values(knots, interpolation)
        squared = 0.0
        for value in values:
            squared += float(value * value)
        return math.sqrt(squared / len(plane_rows))
    squared = 0.0
    for interval, measurement in plane_rows:
        residual, _pose, _normal, _fraction = _evaluate_plane(
            knots, timestamps, interval, measurement, interpolation=interpolation,
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
    interpolation = _ScrewInterpolation(knots)
    squared = 0.0
    count = 0
    for index, (interval, pose) in enumerate(pose_rows):
        measurement = problem.pose_measurements[index]
        for anchor in _ANCHOR_POINTS:
            residual, _pose, _fraction = _evaluate_anchor(
                knots,
                problem.knot_timestamps,
                interval,
                measurement.timestamp_sec,
                anchor,
                pose,
                interpolation=interpolation,
            )
            squared += float(np.sum(residual * residual))
            count += 3
    return math.sqrt(squared / count) if count else None


def build_dense_knot_normal_equations(
    problem: ContinuousTimeTrajectoryFitProblem,
    knots: list[SE3],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a dense knot-only normal system ``(H, g)`` at the current linearization."""

    normal, gradient, _objective_value, _block_count = _knot_normal_evidence(problem, knots)
    return normal, gradient


def _knot_normal_evidence(
    problem: ContinuousTimeTrajectoryFitProblem, knots: list[SE3],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
    """Return normal equations, squared residual cost and block count at fixed knots."""

    if (
        problem.estimate_gyro_bias
        or problem.estimate_lever_arm
        or problem.estimate_imu_clock_offset
        or problem.estimate_accel_bias
        or problem.estimate_gravity
        or problem.estimate_gyro_scale
        or problem.estimate_accel_scale
        or problem.estimate_imu_rotation
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
    for joint_prior in problem.joint_knot_marginalization_priors:
        prior_residual, prior_blocks = joint_marginalization_prior_blocks(joint_prior, knots)
        residuals.extend(float(value) for value in prior_residual)
        weights.extend([1.0] * len(prior_residual))
        for index, block in prior_blocks:
            blocks.append((row_offset, 6 * index, block))
        row_offset += len(prior_residual)
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
    return (normal, np.asarray(gradient, dtype=float).reshape(-1),
            float(np.sum((weight_array * residual_array) ** 2)), len(blocks))



def _calibration_layout(
    problem: ContinuousTimeTrajectoryFitProblem,
) -> list[tuple[str, int]]:
    """Return estimated calibration blocks and widths in solver column order."""

    layout: list[tuple[str, int]] = []
    if problem.estimate_gyro_bias:
        layout.append(("gyro_bias", 3))
    if problem.estimate_lever_arm:
        layout.append(("lever_arm", 3))
    if problem.estimate_imu_clock_offset:
        layout.append(("imu_clock_offset", 1))
    if problem.estimate_accel_bias:
        layout.append(("accel_bias", 3))
    if problem.estimate_gravity:
        layout.append(("gravity", 3))
    if problem.estimate_gyro_scale:
        layout.append(("gyro_scale", 3))
    if problem.estimate_accel_scale:
        layout.append(("accel_scale", 3))
    if problem.estimate_imu_rotation:
        layout.append(("imu_rotation", 3))
    return layout


def _marginal_calibration_covariance(
    jacobian: csc_matrix,
    residuals: FloatArray,
    weights: FloatArray,
    knot_count: int,
) -> tuple[NDArray[np.float64], float]:
    """Return ``s^2 (H_cc - H_ck H_kk^-1 H_kc)^+`` and ``s^2``.

    The solver treats weights as square-root information (``w * r`` and
    ``w * J``), so ``H = (w J)^T (w J)`` and ``s^2`` is the whitened residual
    variance per degree of freedom.  Knots are marginalized by the Schur
    complement, so the covariance includes trajectory uncertainty.
    """

    weighted = csc_matrix(jacobian.multiply(weights[:, None]))
    whitened = weights * residuals
    parameter_count = int(weighted.shape[1])
    knot_columns = 6 * knot_count
    degrees = max(int(whitened.size) - parameter_count, 1)
    residual_variance = float(whitened @ whitened) / degrees
    normal = (weighted.T @ weighted).toarray()
    h_kk = normal[:knot_columns, :knot_columns]
    h_kc = normal[:knot_columns, knot_columns:]
    h_cc = normal[knot_columns:, knot_columns:]
    schur = h_cc - h_kc.T @ (np.linalg.pinv(h_kk) @ h_kc)
    covariance = np.linalg.pinv(schur) * residual_variance
    return np.asarray(covariance, dtype=float), residual_variance


def calibration_correlation(
    result: ContinuousTimeTrajectoryFitResult,
) -> dict[tuple[str, str], float]:
    """Return the largest absolute correlation between each pair of blocks.

    Blocks are the estimated calibration groups (``imu_rotation``,
    ``gyro_bias``, ...) named in ``result.calibration_parameter_names``.
    """

    if result.calibration_covariance is None:
        return {}
    covariance = np.asarray(result.calibration_covariance, dtype=float)
    scale = np.sqrt(np.clip(np.diag(covariance), 1.0e-300, None))
    correlation = covariance / np.outer(scale, scale)
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(result.calibration_parameter_names):
        groups.setdefault(name.split("[", 1)[0], []).append(index)
    names = list(groups)
    pairs: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairs[(left, right)] = float(
                np.max(np.abs(correlation[np.ix_(groups[left], groups[right])]))
            )
    return pairs
