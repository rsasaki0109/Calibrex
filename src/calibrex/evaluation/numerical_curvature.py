"""Finite-difference curvature diagnostics for rematched calibration objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ScalarObjective = Callable[[tuple[float, ...]], float]


@dataclass(frozen=True)
class NumericalCurvatureEvaluation:
    """Central-difference Hessian spectrum with explicit non-convexity evidence."""

    parameter_dimension: int
    objective_at_center: float
    finite_difference_steps: tuple[float, ...]
    hessian: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    rank: int
    positive_eigenvalue_count: int
    negative_eigenvalue_count: int
    near_zero_eigenvalue_count: int
    positive_condition_number: float | None
    evaluation_count: int
    rank_tolerance: float

    def as_dict(self) -> dict[str, object]:
        """Return a schema-safe curvature payload."""

        return {
            "parameter_dimension": self.parameter_dimension,
            "objective_at_center": self.objective_at_center,
            "finite_difference_steps": list(self.finite_difference_steps),
            "hessian": [list(row) for row in self.hessian],
            "eigenvalues": list(self.eigenvalues),
            "rank": self.rank,
            "positive_eigenvalue_count": self.positive_eigenvalue_count,
            "negative_eigenvalue_count": self.negative_eigenvalue_count,
            "near_zero_eigenvalue_count": self.near_zero_eigenvalue_count,
            "positive_condition_number": self.positive_condition_number,
            "evaluation_count": self.evaluation_count,
            "rank_tolerance": self.rank_tolerance,
            "diagnostic_kind": "central_difference_objective_hessian_not_covariance",
        }


def evaluate_numerical_curvature(
    objective: ScalarObjective,
    center: Sequence[float],
    steps: Sequence[float],
    *,
    rank_tolerance: float = 1.0e-8,
) -> NumericalCurvatureEvaluation:
    """Evaluate a symmetric central-difference Hessian at ``center``."""

    point = tuple(float(value) for value in center)
    increments = tuple(float(value) for value in steps)
    if not point or len(point) != len(increments):
        raise ValueError("curvature center and steps must have equal non-zero dimension")
    if any(step <= 0.0 for step in increments) or rank_tolerance <= 0.0:
        raise ValueError("curvature steps and rank_tolerance must be positive")
    dimension = len(point)
    hessian = np.zeros((dimension, dimension), dtype=float)
    center_value = _finite_objective(objective, point)
    evaluation_count = 1
    for index, step in enumerate(increments):
        plus = _shift(point, index, step)
        minus = _shift(point, index, -step)
        hessian[index, index] = (
            _finite_objective(objective, plus)
            - 2.0 * center_value
            + _finite_objective(objective, minus)
        ) / (step * step)
        evaluation_count += 2
    for left in range(dimension):
        for right in range(left + 1, dimension):
            left_step = increments[left]
            right_step = increments[right]
            value = (
                _finite_objective(objective, _shift2(point, left, left_step, right, right_step))
                - _finite_objective(objective, _shift2(point, left, left_step, right, -right_step))
                - _finite_objective(objective, _shift2(point, left, -left_step, right, right_step))
                + _finite_objective(objective, _shift2(point, left, -left_step, right, -right_step))
            ) / (4.0 * left_step * right_step)
            hessian[left, right] = value
            hessian[right, left] = value
            evaluation_count += 4
    eigenvalues_array = np.linalg.eigvalsh(hessian)
    scale = max(float(np.max(np.abs(eigenvalues_array))), 1.0)
    threshold = rank_tolerance * scale
    positive = eigenvalues_array[eigenvalues_array > threshold]
    negative_count = int(np.count_nonzero(eigenvalues_array < -threshold))
    rank = int(np.count_nonzero(np.abs(eigenvalues_array) > threshold))
    condition = float(np.max(positive) / np.min(positive)) if positive.size else None
    return NumericalCurvatureEvaluation(
        parameter_dimension=dimension,
        objective_at_center=center_value,
        finite_difference_steps=increments,
        hessian=_matrix_tuple(hessian),
        eigenvalues=tuple(float(value) for value in eigenvalues_array),
        rank=rank,
        positive_eigenvalue_count=int(positive.size),
        negative_eigenvalue_count=negative_count,
        near_zero_eigenvalue_count=dimension - rank,
        positive_condition_number=condition,
        evaluation_count=evaluation_count,
        rank_tolerance=rank_tolerance,
    )


def _finite_objective(objective: ScalarObjective, point: tuple[float, ...]) -> float:
    value = float(objective(point))
    if not np.isfinite(value):
        raise ValueError("curvature objective must remain finite at all probes")
    return value


def _shift(point: tuple[float, ...], index: int, amount: float) -> tuple[float, ...]:
    values = list(point)
    values[index] += amount
    return tuple(values)


def _shift2(
    point: tuple[float, ...],
    left: int,
    left_amount: float,
    right: int,
    right_amount: float,
) -> tuple[float, ...]:
    values = list(point)
    values[left] += left_amount
    values[right] += right_amount
    return tuple(values)


def _matrix_tuple(matrix: NDArray[np.float64]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)
