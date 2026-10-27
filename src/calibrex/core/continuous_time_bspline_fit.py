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
differences where feasible, with inward one-sided differences at domain boundaries.
The marginal covariance of the calibration blocks marginalizes the control
points by Schur complement.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from struct import pack
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_matrix, diags, vstack
from scipy.sparse.linalg import spsolve

from calibrex.core.clock_domain import ClockDomainEvidence, ClockOffsetDomain
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
from calibrex.core.fit_diagnostics import (
    FitDiagnostics,
    FitTermination,
    normalized_gradient_norm,
    normalized_step_norm,
)
from calibrex.core.se3_manifold import (
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
)

FloatArray: TypeAlias = NDArray[np.float64]
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
    plane_backend: Literal["numpy", "cuda"] = "numpy"
    gradient_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if (isinstance(self.max_iterations, (bool, np.bool_))
                or not isinstance(self.max_iterations, (int, np.integer))
                or self.max_iterations < 0):
            raise ValueError("max_iterations must be a nonnegative integer")
        if not math.isfinite(self.initial_damping) or self.initial_damping <= 0.:
            raise ValueError("initial_damping must be finite and positive")
        for name in ("relative_cost_tolerance", "step_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0:
            raise ValueError("gradient_tolerance must be finite and positive")
        if self.plane_backend not in ("numpy", "cuda"):
            raise ValueError("spline plane backend must be numpy or cuda")


@dataclass(frozen=True)
class BSplineLidarImuFitResult:
    """Fitted spline and calibration with convergence and covariance evidence."""

    status: Literal["converged", "max_iterations", "singular_system", "stalled"]
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
    diagnostics: FitDiagnostics | None = None
    execution_backend: str = "numpy"
    execution_statistics: dict[str, int] = field(default_factory=dict)

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
) -> dict[str, FloatArray]:
    """Return whitened residuals per family (``plane``, ``gyro``, ``accel``)."""

    return _FixedTrajectoryResiduals(trajectory, planes, gyro, accel).residuals(calibration)


def fit_bspline_lidar_imu(
    problem: BSplineLidarImuProblem,
    options: BSplineLidarImuFitOptions | None = None,
) -> BSplineLidarImuFitResult:
    """Jointly fit the spline control points and the selected calibration blocks."""

    settings = options or BSplineLidarImuFitOptions()
    evaluator: _PlaneEvaluator | None = None
    if settings.plane_backend == "cuda":
        from calibrex.solvers.cuda_bspline_plane_adapter import _CudaPlaneEvaluator

        evaluator = _CudaPlaneEvaluator(problem.initial_trajectory, problem.planes)
    return _fit_bspline_lidar_imu(problem, settings, evaluator)


class _PlaneEvaluator(Protocol):
    segment: NDArray[np.int64]
    identity: str

    def evaluate(
        self, trajectory: CubicBSplineTrajectory, *, derivatives: bool,
    ) -> tuple[FloatArray, FloatArray]: ...

    def normal_system(
        self, trajectory: CubicBSplineTrajectory, imu_normal: csc_matrix,
        imu_gradient: FloatArray,
    ) -> tuple[csc_matrix, FloatArray]: ...

    def statistics(self) -> dict[str, int]: ...


def _fit_bspline_lidar_imu(
    problem: BSplineLidarImuProblem, settings: BSplineLidarImuFitOptions,
    plane_evaluator: _PlaneEvaluator | None,
) -> BSplineLidarImuFitResult:
    estimate = tuple(block for block in CALIBRATION_BLOCKS if block in problem.estimate)
    families = _families(problem.planes, problem.gyro, problem.accel)
    cache = _EvaluationCache(families)
    trajectory = problem.initial_trajectory
    calibration = problem.initial_calibration
    clock_domain = (_clock_domain(trajectory, families)
                    if "imu_clock_offset" in estimate else None)
    if clock_domain is not None and not clock_domain.contains(calibration.imu_clock_offset_sec):
        raise ValueError("initial IMU clock offset is outside the shared time domain")
    rejected_clock_steps = 0
    control_columns = 6 * trajectory.control_point_count
    clock_column = control_columns + sum(
        _BLOCK_WIDTH[block] for block in estimate[:estimate.index("imu_clock_offset")]
    ) if clock_domain is not None else 0
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
        cost = _cost(path, values, families, cache=cache, plane_evaluator=plane_evaluator)
        if prior_magnitude is None:
            return cost
        return cost + prior_residual(values) ** 2

    def linearize(
        path: CubicBSplineTrajectory, values: LidarImuCalibration, *, imu_only: bool = False,
        clock_direction: Literal[-1, 1] | None = None,
    ) -> tuple[csc_matrix, FloatArray]:
        selected = {k: v for k, v in families.items() if k != "plane"} if imu_only else families
        jacobian, residuals = _linearize(
            path, values, selected, estimate, cache=cache, plane_evaluator=plane_evaluator,
            clock_direction=clock_direction,
        )
        if prior_magnitude is None:
            return jacobian, residuals
        gravity = np.asarray(values.gravity_world_m_s2, dtype=float)
        derivative = gravity / max(float(np.linalg.norm(gravity)), 1.0e-12)
        row = coo_matrix(
            (
                derivative / problem.gravity_magnitude_sigma_m_s2,
                (np.zeros(3, dtype=np.int64), gravity_column + np.arange(3)),
            ),
            shape=(1, parameter_count),
        )
        return (
            vstack([jacobian, row]).tocsc(),
            np.concatenate([residuals, [prior_residual(values)]]),
        )

    cost = total_cost(trajectory, calibration)
    if not math.isfinite(cost):
        raise ValueError("initial residuals are not finite; check the time domain")
    initial_cost = cost
    damping = settings.initial_damping
    status: Literal["converged", "max_iterations", "singular_system", "stalled"] = "max_iterations"
    termination: FitTermination = "iteration_limit"
    iterations = 0
    for iteration in range(settings.max_iterations):
        iterations = iteration + 1
        jacobian, residuals = linearize(
            trajectory, calibration, imu_only=plane_evaluator is not None,
        )
        normal = (jacobian.T @ jacobian).tocsc()
        gradient = jacobian.T @ residuals
        if plane_evaluator is not None:
            normal, gradient = plane_evaluator.normal_system(trajectory, normal, gradient)
        scaled_gradient = normalized_gradient_norm(gradient, normal.diagonal(), math.sqrt(cost))
        if scaled_gradient < settings.gradient_tolerance:
            status = "converged"
            termination = "gradient_tolerance"
            break
        scale = np.maximum(normal.diagonal(), 1.0e-12)
        accepted = False
        domain_rejected = False
        while damping < 1.0e12:
            try:
                step = spsolve(
                    (normal + diags(damping * scale)).tocsc(), -gradient
                )
            except RuntimeError:
                status = "singular_system"
                termination = "singular_system"
                break
            if not np.all(np.isfinite(step)):
                status = "singular_system"
                termination = "singular_system"
                break
            domain_rejected = clock_domain is not None and not clock_domain.contains(
                calibration.imu_clock_offset_sec + float(step[clock_column])
            )
            if domain_rejected:
                rejected_clock_steps += 1
                damping *= 10.0
                continue
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
                status = "stalled"
                termination = "clock_domain_limit" if domain_rejected else "damping_limit"
            break
        scaled_step = normalized_step_norm(step, normal.diagonal(), math.sqrt(cost))
        relative = (cost - candidate_cost) / max(cost, 1.0e-300)
        trajectory, calibration, cost = candidate_trajectory, candidate_calibration, candidate_cost
        damping = max(damping * 0.3, 1.0e-12)
        if (relative < settings.relative_cost_tolerance
                or scaled_step < settings.step_tolerance):
            status = "stalled"
            termination = ("small_cost_change" if relative < settings.relative_cost_tolerance
                           else "small_step")
            break

    # Acceleration is continuous at cubic spline knots, but its clock derivative
    # need not be. A central column can cancel opposing one-sided derivatives.
    internal_knots = trajectory.start_time_sec + trajectory.knot_spacing_sec * np.arange(
        1, trajectory.control_point_count - 3,
    )
    at_clock_knot = clock_domain is not None and any(
        np.any(np.isin(family.body_times(calibration), internal_knots))
        for family in families.values() if family.on_imu_clock
    )
    primary_direction: Literal[-1, 1] | None = None
    if at_clock_knot and clock_domain is not None:
        primary_direction = (1 if clock_domain.difference_step(
            calibration.imu_clock_offset_sec,
        ) > 0 else -1)
    covariance_jacobian, residuals = linearize(trajectory, calibration)
    jacobian = covariance_jacobian
    if primary_direction is not None:
        jacobian, residuals = linearize(
            trajectory, calibration, clock_direction=primary_direction,
        )
    scaled_gradient = normalized_gradient_norm(
        jacobian.T @ residuals, np.asarray(jacobian.power(2).sum(axis=0)).ravel(),
        float(np.linalg.norm(residuals)),
    )
    stationary = scaled_gradient < settings.gradient_tolerance
    opposite_gradient = None
    if primary_direction is not None and clock_domain is not None:
        opposite_direction: Literal[-1, 1] = -1 if primary_direction == 1 else 1
        offset = calibration.imu_clock_offset_sec
        if clock_domain.contains(math.nextafter(offset, math.inf * opposite_direction)):
            other_jacobian, other_residuals = linearize(
                trajectory, calibration, clock_direction=opposite_direction,
            )
            opposite_gradient = normalized_gradient_norm(
                other_jacobian.T @ other_residuals,
                np.asarray(other_jacobian.power(2).sum(axis=0)).ravel(),
                float(np.linalg.norm(other_residuals)),
            )
            stationary = stationary and opposite_gradient < settings.gradient_tolerance
    if status == "converged" and not stationary:
        status = "stalled"
        termination = "stationarity_not_confirmed"
    if stationary and status != "singular_system" and settings.max_iterations > 0:
        status = "converged"
        termination = "gradient_tolerance"
    diagnostics = FitDiagnostics(
        termination_reason=termination, normalized_gradient_norm=scaled_gradient,
        gradient_tolerance=settings.gradient_tolerance, stationary=stationary,
        clock_domain=(ClockDomainEvidence(
            lower_sec=clock_domain.lower, upper_sec=clock_domain.upper,
            rejected_steps=rejected_clock_steps,
            derivative_step_sec=clock_domain.difference_step(calibration.imu_clock_offset_sec),
            opposite_gradient_norm=opposite_gradient,
        ) if clock_domain is not None else None),
    )
    names = tuple(
        f"{block}[{axis}]" for block in estimate for axis in range(_BLOCK_WIDTH[block])
    )
    covariance, variance = _marginal_covariance(covariance_jacobian, residuals, control_columns)
    return BSplineLidarImuFitResult(
        status=status,
        diagnostics=diagnostics,
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
        execution_statistics={} if plane_evaluator is None else plane_evaluator.statistics(),
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
    basis: FloatArray
    d_basis: FloatArray
    dd_basis: FloatArray

    def states(self, order: Literal[0, 1, 2] = 2) -> BSplineStates:
        return _evaluate_local_spline(
            self.rotations, self.positions, self.fraction, self.spacing,
            order=order, deltas=self.deltas, steps=self.steps,
            basis_terms=(self.basis, self.d_basis, self.dd_basis),
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


@dataclass
class _CachedEvaluation:
    gathered: _Gathered
    states: BSplineStates
    order: Literal[0, 1, 2]
    rotation: _RotationDerivatives | None = None
    residuals: dict[int, tuple[LidarImuCalibration, FloatArray]] = field(default_factory=dict)

    def residual(self, family: _Family, calibration: LidarImuCalibration) -> FloatArray:
        previous = self.residuals.get(id(family))
        if previous is None or previous[0] is not calibration:
            previous = (calibration, family.state_residual(self.states, calibration))
            self.residuals[id(family)] = previous
        return previous[1]

    def derivatives(self) -> _RotationDerivatives:
        if self.rotation is None:
            g = self.gathered
            self.rotation = _rotation_control_derivatives(
                g.rotations, g.fraction, g.spacing, order=self.order,
                relatives=g.relatives, deltas=g.deltas, steps=g.steps,
                log_jacobians=g.log_jacobians,
                basis_terms=(g.basis, g.d_basis, g.dd_basis),
            )
        return self.rotation

    def retained_bytes(self) -> int:
        # Count views conservatively; this is a retained-workspace bound, not RSS.
        total = sum(value.nbytes for value in vars(self.gathered).values()
                    if isinstance(value, np.ndarray))
        total += sum(value.nbytes for value in vars(self.states).values())
        if self.rotation is not None:
            total += sum(value.nbytes for value in vars(self.rotation).values())
        return int(total + sum(value.nbytes for _calibration, value in self.residuals.values()))


class _EvaluationCache:
    """One numerical operation's bounded workspace for an immutable trajectory.

    Gyro/accel rows share work only for bit-identical clocks and timestamp arrays.
    Higher-order IMU evaluation has exactly the same rotation/rate prefix. Any
    trajectory change discards the workspace, including rejected LM candidates.
    Nothing is retained between public calls, separate fits or evidence sessions.
    """

    def __init__(
        self, families: dict[str, _Family], *, derivatives: bool = True,
        max_bytes: int = 64 * 1024**2,
    ) -> None:
        representatives: list[_Family] = []
        self.groups: dict[int, int] = {}
        self.orders: dict[int, Literal[0, 1, 2]] = {}
        for family in families.values():
            group = next((i for i, other in enumerate(representatives)
                          if family.on_imu_clock == other.on_imu_clock
                          and (family.derivative_order == 0) == (other.derivative_order == 0)
                          and np.array_equal(family.times.view(np.uint64),
                                             other.times.view(np.uint64))), len(representatives))
            if group == len(representatives):
                representatives.append(family)
            self.groups[id(family)] = group
            self.orders[group] = max(self.orders.get(group, 0), family.derivative_order)
        self.trajectory: CubicBSplineTrajectory | None = None
        self.batches: OrderedDict[tuple[int, bytes], _CachedEvaluation] = OrderedDict()
        self.with_derivatives = derivatives
        self.max_bytes = max_bytes

    def get(
        self, trajectory: CubicBSplineTrajectory, calibration: LidarImuCalibration,
        family: _Family,
    ) -> _CachedEvaluation:
        if self.trajectory is not trajectory:
            self.batches.clear()
            self.trajectory = trajectory
        group = self.groups[id(family)]
        key = (group, pack("d", calibration.imu_clock_offset_sec) if family.on_imu_clock else b"")
        batch = self.batches.get(key)
        if batch is None:
            gathered = _gather(trajectory, family.body_times(calibration),
                               derivatives=self.with_derivatives)
            order = self.orders[group]
            batch = _CachedEvaluation(gathered, gathered.states(order), order)
            self.batches[key] = batch
        else:
            self.batches.move_to_end(key)
        self.trim()
        return batch

    def trim(self) -> None:
        while self.batches and (
            sum(b.retained_bytes() for b in self.batches.values()) > self.max_bytes
        ):
            self.batches.popitem(last=False)

    def residual(
        self, trajectory: CubicBSplineTrajectory, calibration: LidarImuCalibration,
        family: _Family,
    ) -> FloatArray:
        result = self.get(trajectory, calibration, family).residual(family, calibration)
        self.trim()
        return result


class _FixedTrajectoryResiduals:
    """Private evidence session over fixed measurements and one fixed trajectory.

    Calibration controls recompute only their requested residual family, reusing
    spline states at each clock offset within the same bounded workspace as the
    fitter. Inputs must remain immutable for the session's lifetime. Public
    residual calls create a fresh session so edits between calls stay visible.
    """

    def __init__(
        self, trajectory: CubicBSplineTrajectory, planes: BSplinePlaneReturns,
        gyro: BSplineImuSamples, accel: BSplineImuSamples,
    ) -> None:
        self.trajectory = trajectory
        self.families = _families(planes, gyro, accel)
        self.cache = _EvaluationCache(self.families, derivatives=False)

    def residual(self, name: str, calibration: LidarImuCalibration) -> FloatArray:
        return self.cache.residual(self.trajectory, calibration, self.families[name])

    def residuals(self, calibration: LidarImuCalibration) -> dict[str, FloatArray]:
        return {name: self.residual(name, calibration) for name in self.families}


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
    basis, d_basis, dd_basis = cumulative_basis(fraction, trajectory.knot_spacing_sec)
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
        basis=basis, d_basis=d_basis, dd_basis=dd_basis,
    )


def _cost(
    trajectory: CubicBSplineTrajectory,
    calibration: LidarImuCalibration,
    families: dict[str, _Family],
    *, cache: _EvaluationCache | None = None,
    plane_evaluator: _PlaneEvaluator | None = None,
) -> float:
    total = 0.0
    for name, family in families.items():
        if name == "plane" and plane_evaluator is not None:
            residual, _local = plane_evaluator.evaluate(trajectory, derivatives=False)
            total += float(np.sum(residual * residual))
            continue
        try:
            prepared = None if cache is None else cache.get(trajectory, calibration, family)
            gathered = (_gather(trajectory, family.body_times(calibration))
                        if prepared is None else prepared.gathered)
        except ValueError:
            return math.inf
        residual = (family.residual(gathered, calibration) if prepared is None
                    else prepared.residual(family, calibration))
        if cache is not None:
            cache.trim()
        total += float(np.sum(residual * residual))
    return total


def _clock_domain(
    trajectory: CubicBSplineTrajectory, families: dict[str, _Family],
) -> ClockOffsetDomain:
    times = np.concatenate([family.times for family in families.values() if family.on_imu_clock])
    if not times.size:
        raise ValueError("IMU clock-offset estimation requires IMU measurements")
    return ClockOffsetDomain(
        trajectory.start_time_sec, trajectory.end_time_sec,
        float(np.min(times)), float(np.max(times)),
    )


def _linearize(
    trajectory: CubicBSplineTrajectory,
    calibration: LidarImuCalibration,
    families: dict[str, _Family],
    estimate: tuple[CalibrationBlock, ...],
    *, cache: _EvaluationCache | None = None,
    plane_evaluator: _PlaneEvaluator | None = None,
    clock_direction: Literal[-1, 1] | None = None,
) -> tuple[csc_matrix, FloatArray]:
    clock_domain = (_clock_domain(trajectory, families)
                    if "imu_clock_offset" in estimate else None)
    clock_step = (clock_domain.difference_step(
        calibration.imu_clock_offset_sec, direction=clock_direction,
    ) if clock_domain is not None else _EPSILON)
    central_clock = clock_direction is None and clock_domain is not None and all(
        clock_domain.contains(calibration.imu_clock_offset_sec + direction * _EPSILON)
        for direction in (-1, 1)
    )
    control_columns = 6 * trajectory.control_point_count
    parameter_count = control_columns + sum(_BLOCK_WIDTH[block] for block in estimate)
    rows: list[NDArray[np.int64]] = []
    columns: list[NDArray[np.int64]] = []
    values: list[FloatArray] = []
    residual_parts: list[FloatArray] = []
    row_offset = 0
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
        prepared = None if cache is None else cache.get(trajectory, calibration, family)
        gathered = (_gather(trajectory, family.body_times(calibration), derivatives=True)
                    if prepared is None else prepared.gathered)
        states = gathered.states(family.derivative_order) if prepared is None else prepared.states
        base = (family.state_residual(states, calibration) if prepared is None
                else prepared.residual(family, calibration))
        count, width = base.shape
        residual_parts.append(base.reshape(-1))
        row_index = row_offset + np.arange(count * width).reshape(count, width)
        # Positions enter linearly: p = sum_k w_k c_k with w_k = l_k - l_{k+1}
        # (l_0 = 1, l_4 = 0), and the acceleration likewise with l''.
        basis, dd_basis = gathered.basis, gathered.dd_basis
        by_position, by_acceleration = family.position_sensitivity(states, calibration)
        position_weights = (None if by_position is None else basis - np.concatenate(
            [basis[:, 1:], np.zeros((count, 1))], axis=1,
        ))
        acceleration_weights = None
        if by_acceleration is not None:
            dd_padded = np.concatenate([np.zeros((count, 1)), dd_basis[:, 1:]], axis=1)
            acceleration_weights = dd_padded - np.concatenate(
                [dd_padded[:, 1:], np.zeros((count, 1))], axis=1,
            )
        rotation_derivative = family.rotation_sensitivity(
            states,
            _rotation_control_derivatives(
                gathered.rotations, gathered.fraction, gathered.spacing,
                order=family.derivative_order, relatives=gathered.relatives,
                deltas=gathered.deltas, steps=gathered.steps,
                log_jacobians=gathered.log_jacobians,
                basis_terms=(gathered.basis, gathered.d_basis, gathered.dd_basis),
            ) if prepared is None else prepared.derivatives(),
            calibration,
        )
        for slot in range(4):
            control = gathered.segment + slot
            for axis in range(3):
                if by_position is not None or by_acceleration is not None:
                    derivative = np.zeros((count, width), dtype=float)
                    if by_position is not None and position_weights is not None:
                        derivative += by_position[:, :, axis] * position_weights[:, slot, None]
                    if by_acceleration is not None and acceleration_weights is not None:
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
                    step = np.array([_EPSILON if central_clock else clock_step])
                    plus = calibration.stepped(block, step)
                    minus = calibration.stepped(block, -step) if central_clock else calibration
                    if cache is None:
                        above = family.residual(_gather(trajectory, family.body_times(plus)), plus)
                        below = family.residual(
                            _gather(trajectory, family.body_times(minus)), minus,
                        )
                    else:
                        above = cache.residual(trajectory, plus, family)
                        below = cache.residual(trajectory, minus, family)
                    derivative = (above - below) / (
                        plus.imu_clock_offset_sec - minus.imu_clock_offset_sec
                    )
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
        if cache is not None:
            cache.trim()
    jacobian = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(row_offset, parameter_count),
    ).tocsc()
    return jacobian, np.concatenate(residual_parts)


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
