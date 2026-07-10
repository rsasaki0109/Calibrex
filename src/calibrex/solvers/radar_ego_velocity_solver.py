"""Robust scan-wise Radar ego-velocity estimation from Doppler returns."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from calibrex.core.geometry import Vector3

RadarEgoVelocityStatus = Literal[
    "converged", "max_iterations", "insufficient_returns", "degenerate_geometry"
]


@dataclass(frozen=True)
class RadarDopplerObservation:
    """One static-scene Doppler constraint in the Radar frame."""

    line_of_sight_radar: Vector3
    measured_radial_velocity_mps: float
    weight: float = 1.0


@dataclass(frozen=True)
class RadarEgoVelocitySolverOptions:
    """Controls for deterministic Huber IRLS velocity estimation."""

    max_iterations: int = 20
    convergence_tolerance_mps: float = 1.0e-8
    huber_delta_mps: float = 0.2
    min_return_count: int = 5
    rank_tolerance: float = 1.0e-8
    inlier_threshold_mps: float = 1.0


@dataclass(frozen=True)
class RadarEgoVelocityResult:
    """Estimated Radar-frame velocity and observability diagnostics."""

    status: RadarEgoVelocityStatus
    reason: str
    velocity_radar_mps: Vector3 | None
    return_count: int
    inlier_count: int
    rmse_mps: float | None
    rank: int
    condition_number: float | None
    iterations: int

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "velocity_radar_mps": list(self.velocity_radar_mps)
            if self.velocity_radar_mps is not None
            else None,
            "return_count": self.return_count,
            "inlier_count": self.inlier_count,
            "rmse_mps": self.rmse_mps,
            "rank": self.rank,
            "condition_number": self.condition_number,
            "iterations": self.iterations,
            "method": "huber_irls_los_doppler/v0.1",
        }


class RadarEgoVelocitySolver:
    """Estimate ego velocity from ``d_i = -u_i^T v`` constraints."""

    def solve(
        self,
        observations: Sequence[RadarDopplerObservation],
        options: RadarEgoVelocitySolverOptions | None = None,
    ) -> RadarEgoVelocityResult:
        solver_options = options or RadarEgoVelocitySolverOptions()
        rows = [_normalized_row(item) for item in observations if item.weight > 0.0]
        if len(rows) < solver_options.min_return_count:
            return _empty_result(
                "insufficient_returns",
                "not enough positive-weight Doppler returns",
                len(rows),
            )
        base_hessian, _base_gradient = _normal_equations(rows, None)
        eigenvalues = _symmetric_eigenvalues_3x3(base_hessian)
        maximum = max(eigenvalues)
        threshold = solver_options.rank_tolerance * max(1.0, maximum)
        positive = [value for value in eigenvalues if value > threshold]
        rank = len(positive)
        condition = maximum / min(positive) if positive else None
        if rank < 3:
            return RadarEgoVelocityResult(
                status="degenerate_geometry",
                reason="line-of-sight geometry does not constrain 3D velocity",
                velocity_radar_mps=None,
                return_count=len(rows),
                inlier_count=0,
                rmse_mps=None,
                rank=rank,
                condition_number=condition,
                iterations=0,
            )

        velocity: Vector3 = (0.0, 0.0, 0.0)
        status: RadarEgoVelocityStatus = "max_iterations"
        iterations = 0
        for iteration in range(solver_options.max_iterations):
            residuals = [_residual(row, velocity) for row in rows]
            robust_weights = [
                _huber_weight(value, solver_options.huber_delta_mps) for value in residuals
            ]
            hessian, gradient = _normal_equations(rows, robust_weights)
            solved = _solve_3x3(hessian, gradient)
            if solved is None:
                return RadarEgoVelocityResult(
                    status="degenerate_geometry",
                    reason="weighted normal equations became singular",
                    velocity_radar_mps=None,
                    return_count=len(rows),
                    inlier_count=0,
                    rmse_mps=None,
                    rank=rank,
                    condition_number=condition,
                    iterations=iteration,
                )
            iterations = iteration + 1
            step = math.sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(solved, velocity, strict=True)
                )
            )
            velocity = solved
            if step <= solver_options.convergence_tolerance_mps:
                status = "converged"
                break

        residuals = [_residual(row, velocity) for row in rows]
        inliers = [
            value for value in residuals if abs(value) <= solver_options.inlier_threshold_mps
        ]
        rmse = (
            math.sqrt(sum(value * value for value in inliers) / len(inliers))
            if inliers
            else None
        )
        return RadarEgoVelocityResult(
            status=status,
            reason=(
                "velocity update converged"
                if status == "converged"
                else "maximum IRLS iterations reached"
            ),
            velocity_radar_mps=velocity,
            return_count=len(rows),
            inlier_count=len(inliers),
            rmse_mps=rmse,
            rank=rank,
            condition_number=condition,
            iterations=iterations,
        )


_Row = tuple[Vector3, float, float]


def _normalized_row(observation: RadarDopplerObservation) -> _Row:
    x, y, z = observation.line_of_sight_radar
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1.0e-12:
        raise ValueError("Radar Doppler line of sight must be non-zero")
    return (
        (x / norm, y / norm, z / norm),
        -observation.measured_radial_velocity_mps,
        observation.weight,
    )


def _residual(row: _Row, velocity: Vector3) -> float:
    los, target, _weight = row
    return los[0] * velocity[0] + los[1] * velocity[1] + los[2] * velocity[2] - target


def _normal_equations(
    rows: Sequence[_Row],
    robust_weights: Sequence[float] | None,
) -> tuple[list[list[float]], list[float]]:
    hessian = [[0.0] * 3 for _ in range(3)]
    gradient = [0.0] * 3
    for index, (los, target, base_weight) in enumerate(rows):
        weight = base_weight * (robust_weights[index] if robust_weights is not None else 1.0)
        for left in range(3):
            gradient[left] += weight * los[left] * target
            for right in range(3):
                hessian[left][right] += weight * los[left] * los[right]
    return hessian, gradient


def _huber_weight(residual: float, delta: float) -> float:
    absolute = abs(residual)
    return 1.0 if absolute <= delta or absolute == 0.0 else delta / absolute


def _solve_3x3(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> Vector3 | None:
    rows = [[*matrix[index], rhs[index]] for index in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1.0e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(3):
            if row == column:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(rows[row], rows[column], strict=True)
            ]
    return (rows[0][3], rows[1][3], rows[2][3])


def _symmetric_eigenvalues_3x3(matrix: Sequence[Sequence[float]]) -> list[float]:
    values = [[float(item) for item in row] for row in matrix]
    for _ in range(24):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(values[pair[0]][pair[1]]))
        if abs(values[p][q]) < 1.0e-12:
            break
        angle = 0.5 * math.atan2(2.0 * values[p][q], values[q][q] - values[p][p])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for row in range(3):
            left = values[row][p]
            right = values[row][q]
            values[row][p] = cosine * left - sine * right
            values[row][q] = sine * left + cosine * right
        for column in range(3):
            left = values[p][column]
            right = values[q][column]
            values[p][column] = cosine * left - sine * right
            values[q][column] = sine * left + cosine * right
    return sorted(max(0.0, values[index][index]) for index in range(3))


def _empty_result(
    status: RadarEgoVelocityStatus,
    reason: str,
    return_count: int,
) -> RadarEgoVelocityResult:
    return RadarEgoVelocityResult(
        status=status,
        reason=reason,
        velocity_radar_mps=None,
        return_count=return_count,
        inlier_count=0,
        rmse_mps=None,
        rank=0,
        condition_number=None,
        iterations=0,
    )
