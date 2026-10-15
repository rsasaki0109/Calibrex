"""Joint LiDAR-IMU calibration on a cumulative cubic B-spline trajectory.

The body frame is the LiDAR frame.  One Levenberg--Marquardt fit estimates the
spline control points from LiDAR point-to-plane returns together with the IMU
calibration blocks:

* gyro: ``omega_imu = R_bi^T omega_body(t_imu - tau) + b_g``;
* accelerometer: ``f_imu = R_bi^T (R^T (a_world - g) + alpha x l +
  omega x (omega x l)) + b_a`` with the IMU origin at ``l`` in the body frame;
* LiDAR: ``n . (R p_body + p - q) = 0`` for a return on a world plane.

``R_bi`` is the IMU mounting rotation (body from IMU, right perturbation),
``tau`` the IMU clock offset (``t_body = t_imu - tau``) and ``g`` gravity in the
world frame.  Residuals are whitened by their noise sigma, so the normal matrix
is ``J^T J`` and the cost is a chi-square.  An optional prior on the gravity
magnitude, ``(|g| - g0) / sigma``, separates gravity from the accelerometer
bias when the attitude barely varies (a level platform that only turns about
the vertical sees ``g_z`` and ``b_a,z`` only through their sum).  Each residual touches the four
control points of its segment; their Jacobian uses the analytic position basis
and SO(3) rotation/rate recursion. Only the clock-offset column retains central
differences, including the segment selection on each side of a knot.
The marginal covariance of the calibration blocks marginalizes the control
points by Schur complement.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_matrix, diags, vstack
from scipy.sparse.linalg import spsolve

from calibrex.core.continuous_time_bspline import (
    BSplineStates,
    CubicBSplineTrajectory,
    _evaluate_local_spline,
    cumulative_basis,
    skew_batch,
    so3_exp_batch,
    so3_log_batch,
)
from calibrex.core.continuous_time_bspline_derivatives import (
    _right_jacobian,
    _rotation_control_derivatives,
    _RotationDerivatives,
)
from calibrex.core.se3_manifold import (
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
)

FloatArray: TypeAlias = NDArray[np.float64]
ResidualFamily = Literal["plane", "gyro", "accel"]
CalibrationBlock = Literal[
    "imu_rotation",
    "gyro_bias",
    "lever_arm",
    "imu_clock_offset",
    "accel_bias",
    "gravity",
]
CALIBRATION_BLOCKS: tuple[CalibrationBlock, ...] = (
    "imu_rotation",
    "gyro_bias",
    "lever_arm",
    "imu_clock_offset",
    "accel_bias",
    "gravity",
)
_BLOCK_WIDTH: dict[CalibrationBlock, int] = {
    "imu_rotation": 3,
    "gyro_bias": 3,
    "lever_arm": 3,
    "imu_clock_offset": 1,
    "accel_bias": 3,
    "gravity": 3,
}
_EPSILON = 1.0e-6


@dataclass(frozen=True)
class BSplinePlaneReturns:
    """LiDAR returns (body frame) on known world planes."""

    times_sec: FloatArray
    points_body_m: FloatArray
    plane_points_world_m: FloatArray
    plane_normals_world: FloatArray
    sigma_m: float

    def __post_init__(self) -> None:
        normals = np.asarray(self.plane_normals_world, dtype=float)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 1.0e-12):
            raise ValueError("plane normals must be non-zero")
        object.__setattr__(self, "plane_normals_world", normals / lengths[:, None])
        _check_rows(self.times_sec, (self.points_body_m, self.plane_points_world_m))
        if self.sigma_m <= 0.0:
            raise ValueError("plane sigma must be positive")


@dataclass(frozen=True)
class BSplineImuSamples:
    """IMU readings (IMU frame) stamped on the IMU clock."""

    times_sec: FloatArray
    values: FloatArray
    sigma: float

    def __post_init__(self) -> None:
        _check_rows(self.times_sec, (self.values,))
        if self.sigma <= 0.0:
            raise ValueError("IMU sigma must be positive")


@dataclass(frozen=True)
class LidarImuCalibration:
    """IMU calibration blocks relative to the LiDAR (body) frame."""

    imu_rotation_vector_rad: FloatArray
    gyro_bias_rad_s: FloatArray
    lever_arm_body_m: FloatArray
    imu_clock_offset_sec: float
    accel_bias_m_s2: FloatArray
    gravity_world_m_s2: FloatArray

    def vector(self, block: CalibrationBlock) -> FloatArray:
        """Return one block as a vector."""

        if block == "imu_clock_offset":
            return np.asarray([self.imu_clock_offset_sec], dtype=float)
        values: dict[CalibrationBlock, FloatArray] = {
            "imu_rotation": self.imu_rotation_vector_rad,
            "gyro_bias": self.gyro_bias_rad_s,
            "lever_arm": self.lever_arm_body_m,
            "accel_bias": self.accel_bias_m_s2,
            "gravity": self.gravity_world_m_s2,
        }
        return np.asarray(values[block], dtype=float).reshape(3)

    def stepped(self, block: CalibrationBlock, step: FloatArray) -> LidarImuCalibration:
        """Apply a tangent step to one block (right perturbation for rotation)."""

        delta = np.asarray(step, dtype=float).reshape(-1)
        if block == "imu_rotation":
            rotated = rotation_matrix_to_rotation_vector(
                rotation_vector_to_matrix(np.asarray(self.imu_rotation_vector_rad))
                @ rotation_vector_to_matrix(delta)
            )
            return replace(self, imu_rotation_vector_rad=np.asarray(rotated, dtype=float))
        if block == "imu_clock_offset":
            return replace(self, imu_clock_offset_sec=self.imu_clock_offset_sec + float(delta[0]))
        if block == "gyro_bias":
            return replace(self, gyro_bias_rad_s=np.asarray(self.gyro_bias_rad_s) + delta)
        if block == "lever_arm":
            return replace(self, lever_arm_body_m=np.asarray(self.lever_arm_body_m) + delta)
        if block == "accel_bias":
            return replace(self, accel_bias_m_s2=np.asarray(self.accel_bias_m_s2) + delta)
        return replace(self, gravity_world_m_s2=np.asarray(self.gravity_world_m_s2) + delta)


@dataclass(frozen=True)
class BSplineLidarImuProblem:
    """Initial spline, measurements and the calibration blocks to estimate."""

    initial_trajectory: CubicBSplineTrajectory
    planes: BSplinePlaneReturns
    gyro: BSplineImuSamples
    accel: BSplineImuSamples
    initial_calibration: LidarImuCalibration
    estimate: tuple[CalibrationBlock, ...] = CALIBRATION_BLOCKS
    # Known gravity magnitude and its sigma (m/s^2); None leaves |g| free.
    gravity_magnitude_m_s2: float | None = None
    gravity_magnitude_sigma_m_s2: float = 0.03

    def __post_init__(self) -> None:
        if self.gravity_magnitude_m_s2 is not None and (
            self.gravity_magnitude_m_s2 <= 0.0 or self.gravity_magnitude_sigma_m_s2 <= 0.0
        ):
            raise ValueError("gravity magnitude and its sigma must be positive")


@dataclass(frozen=True)
class BSplineLidarImuFitOptions:
    """Levenberg--Marquardt options."""

    max_iterations: int = 40
    initial_damping: float = 1.0e-4
    relative_cost_tolerance: float = 1.0e-10
    step_tolerance: float = 1.0e-10
    plane_backend: Literal["numpy", "numba_cpu"] = "numpy"
    plane_workers: int = 1

    def __post_init__(self) -> None:
        if self.plane_backend not in ("numpy", "numba_cpu"):
            raise ValueError("plane backend must be numpy or numba_cpu")
        if self.plane_workers < 1:
            raise ValueError("plane workers must be positive")


@dataclass(frozen=True)
class BSplineLidarImuFitResult:
    """Fitted spline and calibration with convergence and covariance evidence."""

    status: Literal["converged", "max_iterations", "singular_system"]
    iterations: int
    trajectory: CubicBSplineTrajectory
    calibration: LidarImuCalibration
    initial_cost: float
    final_cost: float
    residual_count: int
    parameter_count: int
    calibration_parameter_names: tuple[str, ...]
    calibration_covariance: FloatArray
    residual_variance: float
    execution_backend: str = "numpy"

    def block_std(self, block: CalibrationBlock) -> FloatArray:
        """Return the marginal standard deviation of one estimated block."""

        indices = [
            index
            for index, name in enumerate(self.calibration_parameter_names)
            if name.split("[", 1)[0] == block
        ]
        if not indices:
            raise KeyError(f"{block} was not estimated")
        diagonal = np.diag(self.calibration_covariance)[indices]
        return np.sqrt(np.clip(diagonal, 0.0, None))

    def block_correlations(self) -> dict[tuple[str, str], float]:
        """Return the largest absolute correlation of each pair of blocks."""

        covariance = self.calibration_covariance
        scale = np.sqrt(np.clip(np.diag(covariance), 1.0e-300, None))
        correlation = covariance / np.outer(scale, scale)
        groups: dict[str, list[int]] = {}
        for index, name in enumerate(self.calibration_parameter_names):
            groups.setdefault(name.split("[", 1)[0], []).append(index)
        names = list(groups)
        return {
            (left, right): float(
                np.max(np.abs(correlation[np.ix_(groups[left], groups[right])]))
            )
            for position, left in enumerate(names)
            for right in names[position + 1 :]
        }


def lidar_imu_residuals(
    trajectory: CubicBSplineTrajectory,
    calibration: LidarImuCalibration,
    planes: BSplinePlaneReturns,
    gyro: BSplineImuSamples,
    accel: BSplineImuSamples,
    *,
    families: tuple[ResidualFamily, ...] = ("plane", "gyro", "accel"),
) -> dict[str, FloatArray]:
    """Return whitened residuals for the requested families, in their given order.

    The default evaluates all families. Selecting a subset avoids evaluating
    unrelated spline timestamps, without changing the selected residuals.
    """

    evaluators = _families(planes, gyro, accel)
    if set(families) - evaluators.keys():
        raise ValueError("residual families must be plane, gyro or accel")
    selected = {name: evaluators[name] for name in families}
    return {
        name: family.residual(_gather(trajectory, family.body_times(calibration)), calibration)
        for name, family in selected.items()
    }


def fit_bspline_lidar_imu(
    problem: BSplineLidarImuProblem,
    options: BSplineLidarImuFitOptions | None = None,
) -> BSplineLidarImuFitResult:
    """Jointly fit the spline control points and the selected calibration blocks."""

    settings = options or BSplineLidarImuFitOptions()
    if settings.plane_backend == "numpy":
        return _fit_bspline_lidar_imu(problem, settings, None)
    try:
        from calibrex.solvers.numba_bspline_plane_adapter import _NumbaPlaneEvaluator
    except ImportError as exc:
        raise ValueError("numba_cpu plane fitting requires calibrex[numba]") from exc
    evaluator = _NumbaPlaneEvaluator(
        problem.initial_trajectory, problem.planes, settings.plane_workers,
    )
    try:
        return _fit_bspline_lidar_imu(problem, settings, evaluator)
    finally:
        evaluator.close()


@dataclass
class _Linearization:
    rows: list[NDArray[np.int64]]
    columns: list[NDArray[np.int64]]
    values: list[FloatArray]
    residuals: FloatArray
    parameter_count: int

    def jacobian(self) -> csc_matrix:
        return coo_matrix(
            (np.concatenate(self.values),
             (np.concatenate(self.rows), np.concatenate(self.columns))),
            shape=(len(self.residuals), self.parameter_count),
        ).tocsc()


class _PlaneEvaluator(Protocol):
    segment: NDArray[np.int64]
    identity: str

    def evaluate(
        self, trajectory: CubicBSplineTrajectory, *, derivatives: bool,
    ) -> tuple[FloatArray, FloatArray]: ...

    def normal_system(
        self, trajectory: CubicBSplineTrajectory, imu: _Linearization,
    ) -> tuple[csc_matrix, FloatArray]: ...


def _fit_bspline_lidar_imu(
    problem: BSplineLidarImuProblem, settings: BSplineLidarImuFitOptions,
    plane_evaluator: _PlaneEvaluator | None,
) -> BSplineLidarImuFitResult:
    estimate = tuple(block for block in CALIBRATION_BLOCKS if block in problem.estimate)
    families = _families(problem.planes, problem.gyro, problem.accel)
    trajectory = problem.initial_trajectory
    calibration = problem.initial_calibration
    control_columns = 6 * trajectory.control_point_count
    calibration_columns = sum(_BLOCK_WIDTH[block] for block in estimate)
    parameter_count = control_columns + calibration_columns

    prior_magnitude = problem.gravity_magnitude_m_s2 if "gravity" in estimate else None
    gravity_column = control_columns
    for block in estimate:
        if block == "gravity":
            break
        gravity_column += _BLOCK_WIDTH[block]

    def prior_residual(values: LidarImuCalibration) -> float:
        assert prior_magnitude is not None
        magnitude = float(np.linalg.norm(values.gravity_world_m_s2))
        return (magnitude - prior_magnitude) / problem.gravity_magnitude_sigma_m_s2

    def total_cost(path: CubicBSplineTrajectory, values: LidarImuCalibration) -> float:
        cost = _cost(path, values, families, plane_evaluator)
        if prior_magnitude is None:
            return cost
        return cost + prior_residual(values) ** 2

    def gravity_prior(values: LidarImuCalibration) -> csc_matrix:
        gravity = np.asarray(values.gravity_world_m_s2, dtype=float)
        derivative = gravity / max(float(np.linalg.norm(gravity)), 1.0e-12)
        return coo_matrix(
            (
                derivative / problem.gravity_magnitude_sigma_m_s2,
                (np.zeros(3, dtype=np.int64), gravity_column + np.arange(3)),
            ),
            shape=(1, parameter_count),
        ).tocsc()

    def linearize(
        path: CubicBSplineTrajectory, values: LidarImuCalibration
    ) -> tuple[csc_matrix, FloatArray]:
        jacobian, residuals = _linearize(path, values, families, estimate, plane_evaluator)
        if prior_magnitude is None:
            return jacobian, residuals
        row = gravity_prior(values)
        return (
            vstack([jacobian, row]).tocsc(),
            np.concatenate([residuals, [prior_residual(values)]]),
        )

    cost = total_cost(trajectory, calibration)
    if not math.isfinite(cost):
        raise ValueError("initial residuals are not finite; check the time domain")
    initial_cost = cost
    damping = settings.initial_damping
    status: Literal["converged", "max_iterations", "singular_system"] = "max_iterations"
    iterations = 0
    for iteration in range(settings.max_iterations):
        iterations = iteration + 1
        if plane_evaluator is None:
            jacobian, residuals = linearize(trajectory, calibration)
            normal = (jacobian.T @ jacobian).tocsc()
            gradient = jacobian.T @ residuals
        else:
            imu = _linearize_terms(
                trajectory, calibration, {k: v for k, v in families.items() if k != "plane"},
                estimate, plane_evaluator,
            )
            normal, gradient = plane_evaluator.normal_system(trajectory, imu)
            if prior_magnitude is not None:
                row = gravity_prior(calibration)
                normal = (normal + row.T @ row).tocsc()
                gradient += np.asarray(row.T @ np.array([prior_residual(calibration)]))
        scale = np.maximum(normal.diagonal(), 1.0e-12)
        accepted = False
        while damping < 1.0e12:
            try:
                step = spsolve(
                    (normal + diags(damping * scale)).tocsc(), -gradient
                )
            except RuntimeError:
                status = "singular_system"
                break
            if not np.all(np.isfinite(step)):
                status = "singular_system"
                break
            candidate_trajectory, candidate_calibration = _apply_step(
                trajectory, calibration, step, estimate
            )
            candidate_cost = total_cost(candidate_trajectory, candidate_calibration)
            if candidate_cost < cost:
                accepted = True
                break
            damping *= 10.0
        if status == "singular_system" or not accepted:
            if not accepted and status != "singular_system":
                status = "converged"
            break
        relative = (cost - candidate_cost) / max(cost, 1.0e-300)
        trajectory, calibration, cost = candidate_trajectory, candidate_calibration, candidate_cost
        damping = max(damping * 0.3, 1.0e-12)
        if relative < settings.relative_cost_tolerance or float(
            np.max(np.abs(step))
        ) < settings.step_tolerance:
            status = "converged"
            break

    jacobian, residuals = linearize(trajectory, calibration)
    names = tuple(
        f"{block}[{axis}]" for block in estimate for axis in range(_BLOCK_WIDTH[block])
    )
    covariance, variance = _marginal_covariance(jacobian, residuals, control_columns)
    return BSplineLidarImuFitResult(
        status=status,
        iterations=iterations,
        trajectory=trajectory,
        calibration=calibration,
        initial_cost=initial_cost,
        final_cost=cost,
        residual_count=int(residuals.size),
        parameter_count=parameter_count,
        calibration_parameter_names=names,
        calibration_covariance=covariance,
        residual_variance=variance,
        execution_backend="numpy" if plane_evaluator is None else plane_evaluator.identity,
    )


@dataclass(frozen=True)
class _Gathered:
    rotations: FloatArray
    positions: FloatArray
    fraction: FloatArray
    segment: NDArray[np.int64]
    spacing: float
    relatives: FloatArray
    deltas: FloatArray
    log_jacobians: FloatArray | None
    steps: FloatArray

    def states(self, order: Literal[0, 1, 2] = 2) -> BSplineStates:
        return _evaluate_local_spline(
            self.rotations, self.positions, self.fraction, self.spacing,
            order=order, deltas=self.deltas, steps=self.steps,
        )


_Sensitivity = tuple[FloatArray | None, FloatArray | None]


@dataclass(frozen=True)
class _Family:
    times: FloatArray
    on_imu_clock: bool
    state_residual: Callable[[BSplineStates, LidarImuCalibration], FloatArray]
    # Residual derivatives by world position and world acceleration, each
    # ``(M, width, 3)`` or None; the position control points enter only there.
    position_sensitivity: Callable[[BSplineStates, LidarImuCalibration], _Sensitivity]
    rotation_sensitivity: Callable[
        [BSplineStates, _RotationDerivatives, LidarImuCalibration], FloatArray
    ]
    calibration_sensitivity: Callable[
        [BSplineStates, LidarImuCalibration], dict[CalibrationBlock, FloatArray]
    ]
    depends_on: frozenset[CalibrationBlock]
    derivative_order: Literal[0, 1, 2]

    def residual(self, gathered: _Gathered, calibration: LidarImuCalibration) -> FloatArray:
        return self.state_residual(gathered.states(self.derivative_order), calibration)

    def body_times(self, calibration: LidarImuCalibration) -> FloatArray:
        if self.on_imu_clock:
            return self.times - calibration.imu_clock_offset_sec
        return self.times


def _families(
    planes: BSplinePlaneReturns,
    gyro: BSplineImuSamples,
    accel: BSplineImuSamples,
) -> dict[str, _Family]:
    def plane_residual(states: BSplineStates, _calibration: LidarImuCalibration) -> FloatArray:
        world = (
            np.einsum("tij,tj->ti", states.rotation_world_body, planes.points_body_m)
            + states.position_world_m
        )
        offset = world - planes.plane_points_world_m
        distance = np.sum(planes.plane_normals_world * offset, axis=1)
        return np.asarray(distance[:, None] / planes.sigma_m, dtype=float)

    def gyro_residual(states: BSplineStates, calibration: LidarImuCalibration) -> FloatArray:
        rotation = rotation_vector_to_matrix(np.asarray(calibration.imu_rotation_vector_rad))
        predicted = states.angular_velocity_body_rad_s @ rotation + calibration.gyro_bias_rad_s
        return np.asarray((predicted - gyro.values) / gyro.sigma, dtype=float)

    def accel_residual(states: BSplineStates, calibration: LidarImuCalibration) -> FloatArray:
        rotation = rotation_vector_to_matrix(np.asarray(calibration.imu_rotation_vector_rad))
        lever = np.broadcast_to(
            np.asarray(calibration.lever_arm_body_m), states.position_world_m.shape
        )
        omega = states.angular_velocity_body_rad_s
        kinematic = (
            np.einsum(
                "tji,tj->ti",
                states.rotation_world_body,
                states.linear_acceleration_world_m_s2 - calibration.gravity_world_m_s2,
            )
            + np.cross(states.angular_acceleration_body_rad_s2, lever)
            + np.cross(omega, np.cross(omega, lever))
        )
        predicted = kinematic @ rotation + calibration.accel_bias_m_s2
        return np.asarray((predicted - accel.values) / accel.sigma, dtype=float)

    def plane_sensitivity(
        _states: BSplineStates, _calibration: LidarImuCalibration
    ) -> _Sensitivity:
        return (planes.plane_normals_world[:, None, :] / planes.sigma_m, None)

    def gyro_sensitivity(_states: BSplineStates, _calibration: LidarImuCalibration) -> _Sensitivity:
        return (None, None)

    def accel_sensitivity(states: BSplineStates, calibration: LidarImuCalibration) -> _Sensitivity:
        rotation = rotation_vector_to_matrix(np.asarray(calibration.imu_rotation_vector_rad))
        # predicted = R_bi^T R^T a_world + ..., so d/da = R_bi^T R^T.
        world_to_imu = np.einsum("ji,tkj->tik", rotation, states.rotation_world_body)
        return (None, np.asarray(world_to_imu / accel.sigma, dtype=float))

    def plane_rotation(
        states: BSplineStates, derivatives: _RotationDerivatives, _calibration: LidarImuCalibration
    ) -> FloatArray:
        # d(R p)/d epsilon = -R [p]x, epsilon the output right tangent.
        world = -states.rotation_world_body @ skew_batch(planes.points_body_m)
        return np.einsum(
            "ti,tij->tj", planes.plane_normals_world, world @ derivatives.rotation_tangent
        )[:, None, :] / planes.sigma_m

    def gyro_rotation(
        _states: BSplineStates, derivatives: _RotationDerivatives, calibration: LidarImuCalibration
    ) -> FloatArray:
        rotation = rotation_vector_to_matrix(np.asarray(calibration.imu_rotation_vector_rad))
        return np.asarray(rotation.T @ derivatives.angular_velocity / gyro.sigma, dtype=float)

    def accel_rotation(
        states: BSplineStates, derivatives: _RotationDerivatives, calibration: LidarImuCalibration
    ) -> FloatArray:
        rotation = rotation_vector_to_matrix(np.asarray(calibration.imu_rotation_vector_rad))
        lever = np.broadcast_to(calibration.lever_arm_body_m, states.position_world_m.shape)
        omega = states.angular_velocity_body_rad_s
        body_force = np.einsum(
            "tji,tj->ti", states.rotation_world_body,
            states.linear_acceleration_world_m_s2 - calibration.gravity_world_m_s2,
        )
        derivative = (
            skew_batch(body_force) @ derivatives.rotation_tangent
            - skew_batch(lever) @ derivatives.angular_acceleration
            - (skew_batch(np.cross(omega, lever)) + skew_batch(omega) @ skew_batch(lever))
            @ derivatives.angular_velocity
        )
        return np.asarray(rotation.T @ derivative / accel.sigma, dtype=float)

    def plane_calibration(
        _states: BSplineStates, _calibration: LidarImuCalibration
    ) -> dict[CalibrationBlock, FloatArray]:
        return {}

    def gyro_calibration(
        states: BSplineStates, calibration: LidarImuCalibration
    ) -> dict[CalibrationBlock, FloatArray]:
        rotation = rotation_vector_to_matrix(np.asarray(calibration.imu_rotation_vector_rad))
        prediction = states.angular_velocity_body_rad_s @ rotation
        return {
            "imu_rotation": skew_batch(prediction) / gyro.sigma,
            "gyro_bias": np.broadcast_to(np.eye(3) / gyro.sigma, (prediction.shape[0], 3, 3)),
        }

    def accel_calibration(
        states: BSplineStates, calibration: LidarImuCalibration
    ) -> dict[CalibrationBlock, FloatArray]:
        rotation = rotation_vector_to_matrix(np.asarray(calibration.imu_rotation_vector_rad))
        omega = states.angular_velocity_body_rad_s
        alpha = states.angular_acceleration_body_rad_s2
        lever = np.broadcast_to(calibration.lever_arm_body_m, omega.shape)
        kinematic = (
            np.einsum(
                "tji,tj->ti", states.rotation_world_body,
                states.linear_acceleration_world_m_s2 - calibration.gravity_world_m_s2,
            )
            + np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))
        )
        omega_skew = skew_batch(omega)
        return {
            "imu_rotation": skew_batch(kinematic @ rotation) / accel.sigma,
            "lever_arm": rotation.T @ (skew_batch(alpha) + omega_skew @ omega_skew) / accel.sigma,
            "accel_bias": np.broadcast_to(np.eye(3) / accel.sigma, (omega.shape[0], 3, 3)),
            "gravity": -rotation.T @ np.transpose(states.rotation_world_body, (0, 2, 1))
            / accel.sigma,
        }

    return {
        "plane": _Family(
            np.asarray(planes.times_sec, dtype=float),
            False,
            plane_residual,
            plane_sensitivity,
            plane_rotation,
            plane_calibration,
            frozenset(),
            0,
        ),
        "gyro": _Family(
            np.asarray(gyro.times_sec, dtype=float),
            True,
            gyro_residual,
            gyro_sensitivity,
            gyro_rotation,
            gyro_calibration,
            frozenset({"imu_rotation", "gyro_bias", "imu_clock_offset"}),
            1,
        ),
        "accel": _Family(
            np.asarray(accel.times_sec, dtype=float),
            True,
            accel_residual,
            accel_sensitivity,
            accel_rotation,
            accel_calibration,
            frozenset(
                {"imu_rotation", "lever_arm", "imu_clock_offset", "accel_bias", "gravity"}
            ),
            2,
        ),
    }


def _gather(
    trajectory: CubicBSplineTrajectory, times: FloatArray, *, derivatives: bool = False
) -> _Gathered:
    segment, fraction = trajectory.segments(times)
    slots = segment[:, None] + np.arange(4)[None, :]
    # Evaluate each referenced control edge once, then gather its terms. Restrict
    # this to referenced edges so unused near-pi controls retain their semantics.
    edges, inverse = np.unique(
        segment[:, None] + np.arange(3)[None, :], return_inverse=True
    )
    relative = np.einsum("tji,tjk->tik", trajectory.rotations[edges],
                         trajectory.rotations[edges + 1])
    delta = so3_log_batch(relative)
    log_jacobian = _right_jacobian(delta, inverse=True) if derivatives else None
    inverse = inverse.reshape(len(segment), 3)
    deltas = delta[inverse]
    basis = cumulative_basis(fraction, trajectory.knot_spacing_sec)[0]
    steps = np.stack(
        [so3_exp_batch(basis[:, j + 1, None] * deltas[:, j]) for j in range(3)], axis=1
    )
    return _Gathered(
        rotations=trajectory.rotations[slots],
        positions=trajectory.positions[slots],
        fraction=fraction,
        segment=segment,
        spacing=trajectory.knot_spacing_sec,
        relatives=relative[inverse],
        deltas=deltas,
        log_jacobians=None if log_jacobian is None else log_jacobian[inverse],
        steps=steps,
    )


def _cost(
    trajectory: CubicBSplineTrajectory,
    calibration: LidarImuCalibration,
    families: dict[str, _Family],
    plane_evaluator: _PlaneEvaluator | None = None,
) -> float:
    total = 0.0
    for name, family in families.items():
        if name == "plane" and plane_evaluator is not None:
            residual = plane_evaluator.evaluate(trajectory, derivatives=False)[0]
            total += float(np.sum(residual * residual))
            continue
        try:
            gathered = _gather(trajectory, family.body_times(calibration))
        except ValueError:
            return math.inf
        residual = family.residual(gathered, calibration)
        total += float(np.sum(residual * residual))
    return total


class _SharedImuEvaluation:
    """Share aligned IMU evaluations within one linearization, including clock probes."""

    def __init__(self, trajectory: CubicBSplineTrajectory, times: FloatArray) -> None:
        self.trajectory = trajectory
        self.times = times
        self._states: dict[tuple[float, bool], tuple[_Gathered, BSplineStates]] = {}
        self._derivatives: dict[float, _RotationDerivatives] = {}

    def evaluate(
        self, calibration: LidarImuCalibration, *, derivatives: bool = False,
    ) -> tuple[_Gathered, BSplineStates]:
        key = (calibration.imu_clock_offset_sec, derivatives)
        if key not in self._states:
            gathered = _gather(
                self.trajectory, self.times - calibration.imu_clock_offset_sec,
                derivatives=derivatives,
            )
            self._states[key] = gathered, gathered.states(2)
        return self._states[key]

    def rotation_derivatives(self, calibration: LidarImuCalibration) -> _RotationDerivatives:
        key = calibration.imu_clock_offset_sec
        if key not in self._derivatives:
            gathered, _ = self.evaluate(calibration, derivatives=True)
            self._derivatives[key] = _rotation_control_derivatives(
                gathered.rotations, gathered.fraction, gathered.spacing, order=2,
                relatives=gathered.relatives, deltas=gathered.deltas, steps=gathered.steps,
                log_jacobians=gathered.log_jacobians,
            )
        return self._derivatives[key]


def _linearize(
    trajectory: CubicBSplineTrajectory,
    calibration: LidarImuCalibration,
    families: dict[str, _Family],
    estimate: tuple[CalibrationBlock, ...],
    plane_evaluator: _PlaneEvaluator | None = None,
) -> tuple[csc_matrix, FloatArray]:
    terms = _linearize_terms(trajectory, calibration, families, estimate, plane_evaluator)
    return terms.jacobian(), terms.residuals


def _linearize_terms(
    trajectory: CubicBSplineTrajectory,
    calibration: LidarImuCalibration,
    families: dict[str, _Family],
    estimate: tuple[CalibrationBlock, ...],
    plane_evaluator: _PlaneEvaluator | None = None,
) -> _Linearization:
    control_columns = 6 * trajectory.control_point_count
    parameter_count = control_columns + sum(_BLOCK_WIDTH[block] for block in estimate)
    rows: list[NDArray[np.int64]] = []
    columns: list[NDArray[np.int64]] = []
    values: list[FloatArray] = []
    residual_parts: list[FloatArray] = []
    row_offset = 0
    shared = None
    if (plane_evaluator is not None and "gyro" in families and "accel" in families
        and families["gyro"].on_imu_clock and families["accel"].on_imu_clock
        and np.array_equal(families["gyro"].times, families["accel"].times)):
        shared = _SharedImuEvaluation(trajectory, families["gyro"].times)
    for name, family in families.items():
        if name == "plane" and plane_evaluator is not None:
            base, local = plane_evaluator.evaluate(trajectory, derivatives=True)
            count = len(base)
            residual_parts.append(base)
            row_index = (row_offset + np.arange(count)).reshape(count, 1)
            for slot in range(4):
                control = plane_evaluator.segment + slot
                for axis in range(3):
                    _append(rows, columns, values, row_index, 6 * control + axis,
                            local[:, 6 * slot + axis, None])
                    _append(rows, columns, values, row_index, 6 * control + 3 + axis,
                            local[:, 6 * slot + 3 + axis, None])
            row_offset += count
            continue
        use_shared = shared is not None and name in ("gyro", "accel")
        if use_shared:
            assert shared is not None
            gathered, states = shared.evaluate(calibration, derivatives=True)
        else:
            gathered = _gather(trajectory, family.body_times(calibration), derivatives=True)
            states = gathered.states(family.derivative_order)
        base = family.state_residual(states, calibration)
        count, width = base.shape
        residual_parts.append(base.reshape(-1))
        row_index = row_offset + np.arange(count * width).reshape(count, width)
        # Positions enter linearly: p = sum_k w_k c_k with w_k = l_k - l_{k+1}
        # (l_0 = 1, l_4 = 0), and the acceleration likewise with l''.
        basis, _d_basis, dd_basis = cumulative_basis(gathered.fraction, gathered.spacing)
        position_weights = basis - np.concatenate([basis[:, 1:], np.zeros((count, 1))], axis=1)
        dd_padded = np.concatenate([np.zeros((count, 1)), dd_basis[:, 1:]], axis=1)
        acceleration_weights = dd_padded - np.concatenate(
            [dd_padded[:, 1:], np.zeros((count, 1))], axis=1
        )
        by_position, by_acceleration = family.position_sensitivity(states, calibration)
        rotation_derivative = family.rotation_sensitivity(
            states,
            shared.rotation_derivatives(calibration) if use_shared and shared is not None
            else _rotation_control_derivatives(
                gathered.rotations, gathered.fraction, gathered.spacing,
                order=family.derivative_order, relatives=gathered.relatives,
                deltas=gathered.deltas, steps=gathered.steps,
                log_jacobians=gathered.log_jacobians,
            ),
            calibration,
        )
        for slot in range(4):
            control = gathered.segment + slot
            for axis in range(3):
                if by_position is not None or by_acceleration is not None:
                    derivative = np.zeros((count, width), dtype=float)
                    if by_position is not None:
                        derivative += by_position[:, :, axis] * position_weights[:, slot, None]
                    if by_acceleration is not None:
                        derivative += (
                            by_acceleration[:, :, axis] * acceleration_weights[:, slot, None]
                        )
                    _append(rows, columns, values, row_index, 6 * control + axis, derivative)
                derivative = rotation_derivative[:, :, 3 * slot + axis]
                _append(rows, columns, values, row_index, 6 * control + 3 + axis, derivative)
        column = control_columns
        calibration_derivatives = family.calibration_sensitivity(states, calibration)
        for block in estimate:
            if block not in family.depends_on:
                column += _BLOCK_WIDTH[block]
                continue
            for axis in range(_BLOCK_WIDTH[block]):
                if block == "imu_clock_offset":
                    step = np.array([_EPSILON])
                    plus = calibration.stepped(block, step)
                    minus = calibration.stepped(block, -step)
                    if use_shared:
                        assert shared is not None
                        derivative = (
                            family.state_residual(shared.evaluate(plus)[1], plus)
                            - family.state_residual(shared.evaluate(minus)[1], minus)
                        ) / (2.0 * _EPSILON)
                    else:
                        derivative = (
                            family.residual(_gather(trajectory, family.body_times(plus)), plus)
                            - family.residual(_gather(trajectory, family.body_times(minus)), minus)
                        ) / (2.0 * _EPSILON)
                else:
                    derivative = calibration_derivatives[block][:, :, axis]
                if np.any(derivative):
                    _append(
                        rows,
                        columns,
                        values,
                        row_index,
                        np.full(count, column, dtype=np.int64),
                        derivative,
                    )
                column += 1
        row_offset += count * width
    return _Linearization(rows, columns, values, np.concatenate(residual_parts), parameter_count)


def _append(
    rows: list[NDArray[np.int64]],
    columns: list[NDArray[np.int64]],
    values: list[FloatArray],
    row_index: NDArray[np.int64],
    column: NDArray[np.int64] | int,
    derivative: FloatArray,
) -> None:
    column_array = np.broadcast_to(
        np.asarray(column, dtype=np.int64).reshape(-1, 1), row_index.shape
    )
    rows.append(row_index.reshape(-1))
    columns.append(column_array.reshape(-1))
    values.append(np.asarray(derivative, dtype=float).reshape(-1))


def _apply_step(
    trajectory: CubicBSplineTrajectory,
    calibration: LidarImuCalibration,
    step: FloatArray,
    estimate: tuple[CalibrationBlock, ...],
) -> tuple[CubicBSplineTrajectory, LidarImuCalibration]:
    count = trajectory.control_point_count
    control_step = np.asarray(step[: 6 * count], dtype=float).reshape(count, 6)
    rotations = np.einsum(
        "tij,tjk->tik", trajectory.rotations, so3_exp_batch(control_step[:, 3:])
    )
    updated = replace(
        trajectory,
        rotations=rotations,
        positions=trajectory.positions + control_step[:, :3],
    )
    column = 6 * count
    for block in estimate:
        width = _BLOCK_WIDTH[block]
        calibration = calibration.stepped(block, step[column : column + width])
        column += width
    return updated, calibration


def _marginal_covariance(
    jacobian: csc_matrix,
    residuals: FloatArray,
    control_columns: int,
) -> tuple[FloatArray, float]:
    parameter_count = int(jacobian.shape[1])
    degrees = max(int(residuals.size) - parameter_count, 1)
    variance = float(residuals @ residuals) / degrees
    normal = (jacobian.T @ jacobian).toarray()
    h_cc = normal[:control_columns, :control_columns]
    h_cx = normal[:control_columns, control_columns:]
    h_xx = normal[control_columns:, control_columns:]
    schur = h_xx - h_cx.T @ np.linalg.solve(h_cc + 1.0e-12 * np.eye(control_columns), h_cx)
    return np.asarray(np.linalg.pinv(schur) * variance, dtype=float), variance


def _check_rows(times: FloatArray, arrays: tuple[FloatArray, ...]) -> None:
    count = np.asarray(times).reshape(-1).size
    if count == 0:
        raise ValueError("measurement arrays must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValueError("measurement times must be finite")
    for array in arrays:
        values = np.asarray(array, dtype=float)
        if values.shape != (count, 3):
            raise ValueError("measurement vectors must have shape (N, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("measurement vectors must be finite")
