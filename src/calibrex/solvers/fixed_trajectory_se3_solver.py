"""Fixed-trajectory single-extrinsic SE(3) refinement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isclose, radians, sqrt
from typing import Literal

from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import LidarRigPointToPlaneFactor

SolverStatus = Literal["converged", "max_iterations", "rejected", "insufficient_constraints"]
RobustLoss = Literal["none", "huber"]


@dataclass(frozen=True)
class FixedTrajectorySe3SolverOptions:
    """Controls for a deterministic single-extrinsic optimizer."""

    max_iterations: int = 25
    convergence_tolerance: float = 1.0e-8
    initial_damping: float = 1.0e-3
    max_damping: float = 1.0e9
    min_damping: float = 1.0e-9
    max_translation_step_m: float = 0.05
    max_rotation_step_rad: float = radians(1.0)
    robust_loss: RobustLoss = "huber"
    huber_delta_m: float = 0.05


@dataclass(frozen=True)
class FixedTrajectorySe3Iteration:
    """One optimizer iteration summary."""

    iteration: int
    rmse_m: float | None
    candidate_rmse_m: float | None
    damping: float
    step_translation_norm_m: float
    step_rotation_norm_rad: float
    accepted: bool


@dataclass(frozen=True)
class FixedTrajectorySe3SolverResult:
    """Output of fixed-trajectory single-extrinsic SE(3) refinement."""

    status: SolverStatus
    stop_reason: str
    variable: str
    initial_rmse_m: float | None
    final_rmse_m: float | None
    correction: tuple[float, float, float, float, float, float]
    refined_transform: SE3
    iterations: int
    accepted_steps: int
    rejected_steps: int
    history: list[FixedTrajectorySe3Iteration] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly solver summary."""

        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "variable": self.variable,
            "initial_rmse_m": self.initial_rmse_m,
            "final_rmse_m": self.final_rmse_m,
            "correction": list(self.correction),
            "refined_transform": self.refined_transform.as_dict(),
            "iterations": self.iterations,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "history": [
                {
                    "iteration": item.iteration,
                    "rmse_m": item.rmse_m,
                    "candidate_rmse_m": item.candidate_rmse_m,
                    "damping": item.damping,
                    "step_translation_norm_m": item.step_translation_norm_m,
                    "step_rotation_norm_rad": item.step_rotation_norm_rad,
                    "accepted": item.accepted,
                }
                for item in self.history
            ],
        }


class FixedTrajectorySe3ExtrinsicSolver:
    """Robust LM optimizer for one fixed-trajectory SE(3) extrinsic correction."""

    def solve(
        self,
        factor: LidarRigPointToPlaneFactor,
        options: FixedTrajectorySe3SolverOptions | None = None,
    ) -> FixedTrajectorySe3SolverResult:
        """Refine a single extrinsic correction for a fixed-correspondence factor."""

        solver_options = options or FixedTrajectorySe3SolverOptions()
        correction = [0.0 for _ in range(6)]
        damping = solver_options.initial_damping
        initial_rmse = _rmse(factor.residuals(correction))
        current_rmse = initial_rmse
        accepted_steps = 0
        rejected_steps = 0
        history: list[FixedTrajectorySe3Iteration] = []

        initial_rank = factor.evaluate(correction).rank
        if initial_rank == 0:
            return self._result(
                factor=factor,
                status="insufficient_constraints",
                stop_reason="normal equations have zero rank",
                correction=correction,
                initial_rmse=initial_rmse,
                final_rmse=current_rmse,
                history=history,
                accepted_steps=accepted_steps,
                rejected_steps=rejected_steps,
            )

        status: SolverStatus = "max_iterations"
        stop_reason = "maximum iterations reached"
        for iteration in range(solver_options.max_iterations):
            residuals = factor.residuals(correction)
            linearization = factor.linearize(correction)
            hessian, gradient = _weighted_normal_equations(
                residuals,
                linearization.jacobian,
                solver_options,
            )
            step = _solve_damped_system(hessian, gradient, damping)
            if step is None:
                rejected_steps += 1
                damping *= 10.0
                if damping > solver_options.max_damping:
                    status = "rejected"
                    stop_reason = "normal equations remained singular after damping"
                    break
                continue

            step = _limit_step(step, solver_options)
            candidate = [value + delta for value, delta in zip(correction, step, strict=True)]
            candidate_rmse = _rmse(factor.residuals(candidate))
            accepted = _is_improvement(current_rmse, candidate_rmse)
            history.append(
                FixedTrajectorySe3Iteration(
                    iteration=iteration,
                    rmse_m=current_rmse,
                    candidate_rmse_m=candidate_rmse,
                    damping=damping,
                    step_translation_norm_m=_norm(step[:3]),
                    step_rotation_norm_rad=_norm(step[3:]),
                    accepted=accepted,
                )
            )

            if accepted:
                correction = candidate
                accepted_steps += 1
                damping = max(solver_options.min_damping, damping / 3.0)
                if _has_converged(step, current_rmse, candidate_rmse, solver_options):
                    current_rmse = candidate_rmse
                    status = "converged"
                    stop_reason = "step and residual improvement below tolerance"
                    break
                current_rmse = candidate_rmse
            else:
                if _has_converged(
                    step, current_rmse, candidate_rmse, solver_options
                ):
                    status = "converged"
                    stop_reason = (
                        "candidate step and residual change below tolerance"
                    )
                    break
                rejected_steps += 1
                damping *= 10.0
                if damping > solver_options.max_damping:
                    status = "rejected"
                    stop_reason = "candidate steps stopped improving residuals"
                    break

        return self._result(
            factor=factor,
            status=status,
            stop_reason=stop_reason,
            correction=correction,
            initial_rmse=initial_rmse,
            final_rmse=current_rmse,
            history=history,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
        )

    def _result(
        self,
        *,
        factor: LidarRigPointToPlaneFactor,
        status: SolverStatus,
        stop_reason: str,
        correction: Sequence[float],
        initial_rmse: float | None,
        final_rmse: float | None,
        history: list[FixedTrajectorySe3Iteration],
        accepted_steps: int,
        rejected_steps: int,
    ) -> FixedTrajectorySe3SolverResult:
        typed_correction = _correction_tuple(correction)
        return FixedTrajectorySe3SolverResult(
            status=status,
            stop_reason=stop_reason,
            variable=factor.variable,
            initial_rmse_m=initial_rmse,
            final_rmse_m=final_rmse,
            correction=typed_correction,
            refined_transform=factor.corrected_transform(typed_correction),
            iterations=len(history),
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            history=history,
        )


def _weighted_normal_equations(
    residuals: Sequence[float],
    jacobian: Sequence[Sequence[float]],
    options: FixedTrajectorySe3SolverOptions,
) -> tuple[list[list[float]], list[float]]:
    hessian = [[0.0 for _ in range(6)] for _ in range(6)]
    gradient = [0.0 for _ in range(6)]
    for residual, row in zip(residuals, jacobian, strict=True):
        weight = _robust_weight(residual, options)
        for left in range(6):
            gradient[left] += weight * row[left] * residual
            for right in range(6):
                hessian[left][right] += weight * row[left] * row[right]
    return hessian, gradient


def _robust_weight(residual: float, options: FixedTrajectorySe3SolverOptions) -> float:
    if options.robust_loss == "none":
        return 1.0
    absolute = abs(residual)
    if absolute <= options.huber_delta_m or isclose(absolute, 0.0):
        return 1.0
    return options.huber_delta_m / absolute


def _solve_damped_system(
    hessian: list[list[float]],
    gradient: list[float],
    damping: float,
) -> list[float] | None:
    system = [[float(value) for value in row] for row in hessian]
    rhs = [-float(value) for value in gradient]
    for index in range(6):
        system[index][index] += damping * max(1.0, abs(hessian[index][index]))
    return _solve_linear_system(system, rhs)


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    size = len(rhs)
    rows = [[*list(row), rhs[index]] for index, row in enumerate(matrix)]
    tolerance = 1.0e-12
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) <= tolerance:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        pivot_value = rows[column][column]
        rows[column] = [value / pivot_value for value in rows[column]]
        for row in range(size):
            if row == column:
                continue
            scale = rows[row][column]
            rows[row] = [
                value - scale * pivot_row_value
                for value, pivot_row_value in zip(rows[row], rows[column], strict=True)
            ]
    return [rows[index][-1] for index in range(size)]


def _limit_step(
    step: Sequence[float],
    options: FixedTrajectorySe3SolverOptions,
) -> list[float]:
    limited = [float(value) for value in step]
    translation_norm = _norm(limited[:3])
    if translation_norm > options.max_translation_step_m:
        scale = options.max_translation_step_m / translation_norm
        for index in range(3):
            limited[index] *= scale
    rotation_norm = _norm(limited[3:])
    if rotation_norm > options.max_rotation_step_rad:
        scale = options.max_rotation_step_rad / rotation_norm
        for index in range(3, 6):
            limited[index] *= scale
    return limited


def _is_improvement(current_rmse: float | None, candidate_rmse: float | None) -> bool:
    if candidate_rmse is None:
        return False
    if current_rmse is None:
        return True
    return candidate_rmse <= current_rmse


def _has_converged(
    step: Sequence[float],
    current_rmse: float | None,
    candidate_rmse: float | None,
    options: FixedTrajectorySe3SolverOptions,
) -> bool:
    step_norm = _norm(step)
    if step_norm <= options.convergence_tolerance:
        return True
    if current_rmse is None or candidate_rmse is None:
        return False
    return abs(current_rmse - candidate_rmse) <= options.convergence_tolerance


def _rmse(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sqrt(sum(value * value for value in values) / len(values))


def _norm(values: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in values))


def _correction_tuple(
    correction: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    values = tuple(float(value) for value in correction)
    if len(values) != 6:
        msg = "SE(3) correction must have exactly 6 values"
        raise ValueError(msg)
    return (values[0], values[1], values[2], values[3], values[4], values[5])
