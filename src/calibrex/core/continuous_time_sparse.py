"""Sparse manifold trajectory fitting with analytic SE(3) Jacobians.

The fitter estimates the knot poses of a ``screw_linear`` continuous-time
trajectory from body-frame point measurements, point-to-plane measurements,
and pose measurements.  Point measurements are the native LiDAR/IMU factor
form: a body-frame point at a timestamp is transformed by the interpolated
pose and compared with a world target.  Point-to-plane measurements compare
the signed distance of that transformed point to a world plane.  Pose
measurements are converted internally into four anchor point constraints so
that every residual uses the analytic point-transform Jacobian; no pose-log
Jacobian is required.

The normal equations are assembled as a sparse block matrix in which every
factor touches at most the two knots bracketing its timestamp.  Gauss--Newton
with Levenberg--Marquardt damping solves the linear system on the manifold.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import (
    interpolate_screw_jacobians,
    point_transform_jacobian,
    se3_exp,
    se3_log,
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
class ContinuousTimeTrajectoryFitProblem:
    """Knot times, initial knots, and measurements for a trajectory fit."""

    knot_timestamps: tuple[float, ...]
    initial_knot_poses: tuple[SE3, ...]
    point_measurements: tuple[TrajectoryPointMeasurement, ...] = ()
    point_to_plane_measurements: tuple[TrajectoryPointToPlaneMeasurement, ...] = ()
    pose_measurements: tuple[TrajectoryPoseMeasurement, ...] = ()
    interpolation: Literal["screw_linear"] = "screw_linear"

    def __post_init__(self) -> None:
        if len(self.knot_timestamps) < 2:
            raise ValueError("trajectory fit requires at least two knots")
        if len(self.knot_timestamps) != len(self.initial_knot_poses):
            raise ValueError("knot times and initial poses must have equal length")
        if any(
            right <= left
            for left, right in zip(
                self.knot_timestamps[:-1],
                self.knot_timestamps[1:],
                strict=True,
            )
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
        ):
            raise ValueError("trajectory fit requires at least one measurement")


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
    point_rows: list[tuple[int, tuple[float, float, float]]] = []
    plane_rows: list[tuple[int, TrajectoryPointToPlaneMeasurement]] = []
    pose_rows: list[tuple[int, SE3]] = []
    for measurement in problem.point_measurements:
        interval = _interval_index(timestamps, measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"point measurement {measurement.measurement_id!r} is outside "
                "the knot domain"
            )
        point_rows.append((interval, measurement.point_body_m))
    for measurement in problem.point_to_plane_measurements:
        interval = _interval_index(timestamps, measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"point-to-plane measurement {measurement.measurement_id!r} is "
                "outside the knot domain"
            )
        plane_rows.append((interval, measurement))
    for measurement in problem.pose_measurements:
        interval = _interval_index(timestamps, measurement.timestamp_sec)
        if interval is None:
            raise ValueError(
                f"pose measurement {measurement.measurement_id!r} is outside "
                "the knot domain"
            )
        pose_rows.append((interval, measurement.pose_world_body))

    damping = settings.initial_damping
    gradient_norm = math.inf
    objective = _objective(problem, point_rows, plane_rows, pose_rows, knots)
    best_objective = objective
    best_knots = list(knots)
    max_step_translation = 0.0
    max_step_rotation_deg = 0.0
    status: FitStatus = "max_iterations"
    iterations = 0

    for iteration in range(settings.max_iterations):
        iterations = iteration + 1
        blocks: list[
            tuple[int, int, NDArray[np.float64], FloatArray, float]
        ] = []
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
            blocks.append(
                (row_offset, interval, jacobian_left, residual, measurement.weight)
            )
            blocks.append(
                (row_offset, interval + 1, jacobian_right, residual, measurement.weight)
            )
            row_offset += 3
        for interval, measurement in plane_rows:
            residual, jacobian_left, jacobian_right = _point_to_plane_residual(
                knots,
                timestamps,
                interval,
                measurement,
            )
            residuals.append(float(residual[0]))
            weights.append(measurement.weight)
            blocks.append(
                (row_offset, interval, jacobian_left, residual, measurement.weight)
            )
            blocks.append(
                (
                    row_offset,
                    interval + 1,
                    jacobian_right,
                    residual,
                    measurement.weight,
                )
            )
            row_offset += 1
        for index, (interval, pose) in enumerate(pose_rows):
            weight = problem.pose_measurements[index].weight
            measurement = problem.pose_measurements[index]
            for anchor in _ANCHOR_POINTS:
                residual, jacobian_left, jacobian_right = _anchor_residual(
                    knots,
                    timestamps,
                    interval,
                    measurement.timestamp_sec,
                    anchor,
                    pose,
                )
                residuals.extend(float(value) for value in residual)
                weights.extend([weight] * 3)
                blocks.append((row_offset, interval, jacobian_left, residual, weight))
                blocks.append(
                    (row_offset, interval + 1, jacobian_right, residual, weight)
                )
                row_offset += 3

        residual_array = np.asarray(residuals, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        jacobian = _assemble_sparse_jacobian(
            blocks,
            residual_count=len(residuals),
            knot_count=len(knots),
        )
        weighted_jacobian = jacobian.multiply(weight_array[:, None])
        normal = weighted_jacobian.T @ weighted_jacobian
        rhs = weighted_jacobian.T @ (weight_array * residual_array)
        gradient_norm = float(np.max(np.abs(rhs)))

        if gradient_norm < settings.convergence_gradient_norm:
            status = "converged"
            break

        damping_matrix = (
            normal
            + damping * _block_diagonal_scale(normal, len(knots))
        )
        try:
            delta = spsolve(damping_matrix.tocsc(), -rhs)
        except RuntimeError:
            status = "singular_system"
            break
        if not np.all(np.isfinite(delta)):
            status = "singular_system"
            break

        candidate = [se3_exp(knot, delta[6 * i : 6 * i + 6]) for i, knot in enumerate(knots)]
        candidate_objective = _objective(
            problem, point_rows, plane_rows, pose_rows, candidate
        )
        step_translation = float(np.max(np.abs(delta.reshape(-1, 6)[:, :3])))
        step_rotation = math.degrees(
            float(np.max(np.abs(delta.reshape(-1, 6)[:, 3:])))
        )
        if candidate_objective < objective:
            objective = candidate_objective
            knots = candidate
            if objective < best_objective:
                best_objective = objective
                best_knots = list(knots)
            max_step_translation = max(max_step_translation, step_translation)
            max_step_rotation_deg = max(max_step_rotation_deg, step_rotation)
            damping = max(damping * settings.damping_scale_down, 1.0e-12)
            if step_translation < settings.convergence_step_norm:
                status = "converged"
                break
        else:
            damping *= settings.damping_scale_up
            if damping > 1.0e12:
                break

    if status != "converged" and best_objective < objective:
        knots = best_knots
        objective = best_objective
    final_point_rmse = _point_rmse(problem, point_rows, knots)
    final_point_to_plane_rmse = _point_to_plane_rmse(plane_rows, timestamps, knots)
    final_pose_rmse = _pose_rmse(problem, pose_rows, knots)
    return ContinuousTimeTrajectoryFitResult(
        status=status,
        iterations=iterations,
        knot_poses=tuple(knots),
        initial_knot_poses=problem.initial_knot_poses,
        final_objective=objective,
        final_point_rmse=final_point_rmse,
        final_point_to_plane_rmse=final_point_to_plane_rmse,
        final_pose_rmse=final_pose_rmse,
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
    if right == len(timestamps):
        return len(timestamps) - 2
    if timestamps[right] == timestamp:
        return right
    return right - 1


def _assemble_sparse_jacobian(
    blocks: list[tuple[int, int, NDArray[np.float64], FloatArray, float]],
    *,
    residual_count: int,
    knot_count: int,
) -> csc_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_offset, knot_index, jacobian, _residual, _weight in blocks:
        residual_rows = int(jacobian.shape[0])
        for residual_row in range(residual_rows):
            for parameter in range(6):
                rows.append(row_offset + residual_row)
                columns.append(6 * knot_index + parameter)
                values.append(float(jacobian[residual_row, parameter]))
    return csc_matrix(
        (values, (rows, columns)),
        shape=(residual_count, 6 * knot_count),
        dtype=float,
    )


def _block_diagonal_scale(
    normal: csc_matrix, knot_count: int
) -> csc_matrix:
    diagonal = normal.diagonal()
    scaled = np.zeros(6 * knot_count, dtype=float)
    for knot in range(knot_count):
        block = diagonal[6 * knot : 6 * knot + 6]
        block_max = float(np.max(np.abs(block))) if block.size else 0.0
        if block_max > 0.0:
            scaled[6 * knot : 6 * knot + 6] = block_max
    indices = np.arange(6 * knot_count, dtype=int)
    return csc_matrix(
        (scaled, (indices, indices)),
        shape=(6 * knot_count, 6 * knot_count),
    )


def _objective(
    problem: ContinuousTimeTrajectoryFitProblem,
    point_rows: list[tuple[int, tuple[float, float, float]]],
    plane_rows: list[tuple[int, TrajectoryPointToPlaneMeasurement]],
    pose_rows: list[tuple[int, SE3]],
    knots: list[SE3],
) -> float:
    objective = 0.0
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
        objective += measurement.weight * float(np.sum(residual * residual))
    for interval, measurement in plane_rows:
        residual, _left, _right = _point_to_plane_residual(
            knots,
            problem.knot_timestamps,
            interval,
            measurement,
        )
        objective += measurement.weight * float(residual[0] * residual[0])
    for index, (interval, pose) in enumerate(pose_rows):
        measurement = problem.pose_measurements[index]
        weight = measurement.weight
        for anchor in _ANCHOR_POINTS:
            residual, _left, _right = _anchor_residual(
                knots,
                problem.knot_timestamps,
                interval,
                measurement.timestamp_sec,
                anchor,
                pose,
            )
            objective += weight * float(np.sum(residual * residual))
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