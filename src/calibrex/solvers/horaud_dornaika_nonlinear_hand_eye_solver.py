"""Horaud-Dornaika simultaneous nonlinear hand-eye calibration for ``A X = X B``."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.horaud_dornaika_hand_eye_solver import (
    HoraudDornaikaHandEyeOptions,
    HoraudDornaikaHandEyeSolver,
)
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeEvaluation,
    HandEyeMotionPair,
    evaluate_hand_eye_motions,
)
from calibrex.solvers.tsai_lenz_hand_eye_solver import (
    HandEyeProbeResult,
    evaluate_hand_eye_known_bad_probes,
)

FloatArray: TypeAlias = NDArray[np.float64]
HoraudDornaikaNonlinearHandEyeStatus = Literal[
    "converged",
    "insufficient_motions",
    "initialization_failed",
    "numerical_failure",
    "max_iterations",
    "unobservable",
    "unit_constraint_failed",
]


@dataclass(frozen=True)
class HoraudDornaikaNonlinearHandEyeOptions:
    """Paper objective and numerical evidence policy."""

    min_train_motions: int = 3
    min_rotation_rad: float = math.radians(1.0)
    holdout_ratio: float = 0.2
    split_seed: int = 0
    max_iterations: int = 50
    initial_damping: float = 1.0e-3
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    gradient_tolerance: float = 1.0e-8
    step_tolerance: float = 1.0e-10
    relative_cost_tolerance: float = 1.0e-12
    finite_difference_step: float = 1.0e-6
    rotation_weight: float = 1.0
    translation_weight: float = 1.0
    quaternion_unit_penalty: float = 2.0e6
    rank_tolerance: float = 1.0e-9
    max_data_jacobian_condition_number: float = 1.0e8
    max_quaternion_unit_error: float = 1.0e-5
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_rotation_margin_deg: float = 0.1
    known_bad_translation_margin_m: float = 0.005


@dataclass(frozen=True)
class HoraudDornaikaNonlinearHandEyeIteration:
    """One deterministic Levenberg-Marquardt trial."""

    iteration: int
    accepted: bool
    cost: float
    trial_cost: float
    damping: float
    gradient_inf: float
    step_norm: float


@dataclass(frozen=True)
class HoraudDornaikaNonlinearHandEyeResult:
    """Simultaneous estimate plus optimization and evaluation evidence."""

    status: HoraudDornaikaNonlinearHandEyeStatus
    reason: str
    transform_x: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    iterations: tuple[HoraudDornaikaNonlinearHandEyeIteration, ...]
    accepted_step_count: int
    initial_cost: float | None
    final_cost: float | None
    initial_data_residual_rmse: float | None
    final_data_residual_rmse: float | None
    final_gradient_inf: float | None
    quaternion_norm: float | None
    quaternion_unit_error: float | None
    data_jacobian_singular_values: tuple[float, float, float, float, float, float] | None
    data_jacobian_rank: int
    data_jacobian_condition_number: float | None
    train_evaluation: HandEyeEvaluation
    holdout_evaluation: HandEyeEvaluation
    probes: tuple[HandEyeProbeResult, ...]

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly evidence and primary-paper provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "iterations": [item.__dict__ for item in self.iterations],
            "accepted_step_count": self.accepted_step_count,
            "initial_cost": self.initial_cost,
            "final_cost": self.final_cost,
            "initial_data_residual_rmse": self.initial_data_residual_rmse,
            "final_data_residual_rmse": self.final_data_residual_rmse,
            "final_gradient_inf": self.final_gradient_inf,
            "quaternion_norm": self.quaternion_norm,
            "quaternion_unit_error": self.quaternion_unit_error,
            "data_jacobian_singular_values": self.data_jacobian_singular_values,
            "data_jacobian_rank": self.data_jacobian_rank,
            "data_jacobian_condition_number": self.data_jacobian_condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "horaud_dornaika_simultaneous_nonlinear_hand_eye/v0.1",
            "paper": {
                "title": "Hand-Eye Calibration",
                "authors": "Radu Horaud and Fadi Dornaika",
                "doi": "10.1177/027836499501400301",
                "author_manuscript": "https://arxiv.org/abs/2311.12655",
                "equations": [27, 28, 30, 32],
            },
            "equation": "A X = X B",
            "selected_paper_method": "section_5_2_simultaneous_rotation_translation",
            "initializer": "horaud_dornaika_closed_form_axis_quaternion",
            "parameterization": "raw quaternion xyzw and translation (7 parameters)",
            "optimizer": "deterministic Levenberg-Marquardt with central differences",
            "paper_weights": {
                "lambda1_rotation": 1.0,
                "lambda2_translation": 1.0,
                "lambda_quaternion_unit": 2.0e6,
            },
            "units_limitation": (
                "the paper's lambda1=lambda2=1 combines dimensionless axis residuals "
                "and translations in dataset length units"
            ),
            "evaluation_contract": "held-out AX=XB closure without refitting",
            "implementation": "independent NumPy implementation; no external code copied",
            "external_code_executed": False,
        }


class HoraudDornaikaNonlinearHandEyeSolver:
    """Jointly refine quaternion and translation with paper equations (27)-(32)."""

    def solve(
        self,
        motions: Sequence[HandEyeMotionPair],
        options: HoraudDornaikaNonlinearHandEyeOptions | None = None,
    ) -> HoraudDornaikaNonlinearHandEyeResult:
        solver_options = options or HoraudDornaikaNonlinearHandEyeOptions()
        _validate_options(solver_options)
        usable = _validated_motions(motions, solver_options.min_rotation_rad)
        train_indices, holdout_indices = split_indices(
            len(usable), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        train_ids = tuple(pair.pair_id for pair in train)
        holdout_ids = tuple(pair.pair_id for pair in holdout)
        if len(train) < solver_options.min_train_motions:
            return _empty("insufficient_motions", train_ids, holdout_ids)

        initializer = HoraudDornaikaHandEyeSolver().solve(
            usable,
            HoraudDornaikaHandEyeOptions(
                min_train_motions=solver_options.min_train_motions,
                min_rotation_rad=solver_options.min_rotation_rad,
                holdout_ratio=solver_options.holdout_ratio,
                split_seed=solver_options.split_seed,
                rank_tolerance=solver_options.rank_tolerance,
                max_condition_number=solver_options.max_data_jacobian_condition_number,
            ),
        )
        if initializer.transform_x is None:
            return _empty(
                "initialization_failed",
                train_ids,
                holdout_ids,
                reason=f"closed-form initializer status: {initializer.status}",
            )
        parameters: FloatArray = np.asarray(
            (*initializer.transform_x.rotation_quat_xyzw, *initializer.transform_x.translation_m),
            dtype=np.float64,
        )
        residual, jacobian, data_rows = _linearize(train, parameters, solver_options)
        cost = 0.5 * float(residual @ residual)
        initial_cost = cost
        initial_data_rmse = _rmse(residual[:data_rows])
        damping = solver_options.initial_damping
        history: list[HoraudDornaikaNonlinearHandEyeIteration] = []
        accepted_steps = 0
        converged = False
        numerical_failure = False
        for iteration in range(solver_options.max_iterations):
            gradient = jacobian.T @ residual
            gradient_inf = float(np.max(np.abs(gradient)))
            if gradient_inf <= solver_options.gradient_tolerance:
                converged = True
                break
            normal = jacobian.T @ jacobian
            diagonal = np.maximum(np.diag(normal), 1.0)
            try:
                step = np.linalg.solve(normal + damping * np.diag(diagonal), -gradient)
            except np.linalg.LinAlgError:
                numerical_failure = True
                break
            step_norm = float(np.linalg.norm(step))
            trial_parameters = parameters + step
            trial_residual, trial_jacobian, _trial_rows = _linearize(
                train, trial_parameters, solver_options
            )
            trial_cost = 0.5 * float(trial_residual @ trial_residual)
            accepted = math.isfinite(trial_cost) and trial_cost < cost
            history.append(
                HoraudDornaikaNonlinearHandEyeIteration(
                    iteration=iteration,
                    accepted=accepted,
                    cost=cost,
                    trial_cost=trial_cost,
                    damping=damping,
                    gradient_inf=gradient_inf,
                    step_norm=step_norm,
                )
            )
            if accepted:
                previous_cost = cost
                parameters = trial_parameters
                residual = trial_residual
                jacobian = trial_jacobian
                cost = trial_cost
                accepted_steps += 1
                damping = max(damping * solver_options.damping_decrease, 1.0e-15)
                relative_reduction = (previous_cost - cost) / max(previous_cost, 1.0)
                if (
                    step_norm <= solver_options.step_tolerance
                    or relative_reduction <= solver_options.relative_cost_tolerance
                ):
                    converged = True
                    break
            else:
                damping *= solver_options.damping_increase
                if not math.isfinite(damping) or damping > 1.0e30:
                    numerical_failure = True
                    break

        final_gradient = float(np.max(np.abs(jacobian.T @ residual)))
        status: HoraudDornaikaNonlinearHandEyeStatus
        if numerical_failure:
            status = "numerical_failure"
        elif converged or final_gradient <= solver_options.gradient_tolerance:
            status = "converged"
        else:
            status = "max_iterations"
        spectrum, rank, condition = _physical_diagnostics(train, parameters, solver_options)
        quaternion_norm = float(np.linalg.norm(parameters[:4]))
        unit_error = abs(quaternion_norm - 1.0)
        if status != "converged":
            return _empty(
                status,
                train_ids,
                holdout_ids,
                iterations=tuple(history),
                accepted_steps=accepted_steps,
                initial_cost=initial_cost,
                final_cost=cost,
                initial_data_rmse=initial_data_rmse,
                final_data_rmse=_rmse(residual[:data_rows]),
                final_gradient=final_gradient,
                quaternion_norm=quaternion_norm,
                unit_error=unit_error,
                data_spectrum=spectrum,
                data_rank=rank,
                data_condition=condition,
            )
        if rank != 6 or condition > solver_options.max_data_jacobian_condition_number:
            return _empty(
                "unobservable",
                train_ids,
                holdout_ids,
                iterations=tuple(history),
                accepted_steps=accepted_steps,
                initial_cost=initial_cost,
                final_cost=cost,
                initial_data_rmse=initial_data_rmse,
                final_data_rmse=_rmse(residual[:data_rows]),
                final_gradient=final_gradient,
                quaternion_norm=quaternion_norm,
                unit_error=unit_error,
                data_spectrum=spectrum,
                data_rank=rank,
                data_condition=condition,
            )
        if unit_error > solver_options.max_quaternion_unit_error:
            return _empty(
                "unit_constraint_failed",
                train_ids,
                holdout_ids,
                iterations=tuple(history),
                accepted_steps=accepted_steps,
                initial_cost=initial_cost,
                final_cost=cost,
                initial_data_rmse=initial_data_rmse,
                final_data_rmse=_rmse(residual[:data_rows]),
                final_gradient=final_gradient,
                quaternion_norm=quaternion_norm,
                unit_error=unit_error,
                data_spectrum=spectrum,
                data_rank=rank,
                data_condition=condition,
            )

        quaternion = parameters[:4] / quaternion_norm
        transform = SE3(
            (float(parameters[4]), float(parameters[5]), float(parameters[6])),
            (
                float(quaternion[0]),
                float(quaternion[1]),
                float(quaternion[2]),
                float(quaternion[3]),
            ),
        )
        train_evaluation = evaluate_hand_eye_motions(train, transform)
        holdout_evaluation = evaluate_hand_eye_motions(holdout, transform)
        return HoraudDornaikaNonlinearHandEyeResult(
            status="converged",
            reason="paper Section 5.2 simultaneous nonlinear objective converged",
            transform_x=transform,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            iterations=tuple(history),
            accepted_step_count=accepted_steps,
            initial_cost=initial_cost,
            final_cost=cost,
            initial_data_residual_rmse=initial_data_rmse,
            final_data_residual_rmse=_rmse(residual[:data_rows]),
            final_gradient_inf=final_gradient,
            quaternion_norm=quaternion_norm,
            quaternion_unit_error=unit_error,
            data_jacobian_singular_values=spectrum,
            data_jacobian_rank=rank,
            data_jacobian_condition_number=condition,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=evaluate_hand_eye_known_bad_probes(
                holdout, transform, holdout_evaluation, solver_options
            ),
        )


def _linearize(
    motions: Sequence[HandEyeMotionPair],
    parameters: FloatArray,
    options: HoraudDornaikaNonlinearHandEyeOptions,
) -> tuple[FloatArray, FloatArray, int]:
    residual = _residual(motions, parameters, options, include_penalty=True)
    data_rows = 6 * len(motions)
    jacobian = _central_jacobian(
        lambda values: _residual(motions, values, options, include_penalty=True),
        parameters,
        options.finite_difference_step,
    )
    return residual, jacobian, data_rows


def _residual(
    motions: Sequence[HandEyeMotionPair],
    parameters: FloatArray,
    options: HoraudDornaikaNonlinearHandEyeOptions,
    *,
    include_penalty: bool,
) -> FloatArray:
    quaternion = parameters[:4]
    translation = parameters[4:7]
    rows: list[FloatArray] = []
    for pair in motions:
        root_rotation = math.sqrt(pair.weight * options.rotation_weight)
        root_translation = math.sqrt(pair.weight * options.translation_weight)
        axis_a = _rotation_axis(pair.motion_a)
        axis_b = _rotation_axis(pair.motion_b)
        rows.append(root_rotation * (axis_a - _quaternion_rotate_raw(quaternion, axis_b)))
        rows.append(
            root_translation
            * (
                _quaternion_rotate_raw(
                    quaternion, np.asarray(pair.motion_b.translation_m, dtype=np.float64)
                )
                - (_rotation_matrix(pair.motion_a) - np.eye(3, dtype=np.float64))
                @ translation
                - np.asarray(pair.motion_a.translation_m, dtype=np.float64)
            )
        )
    if include_penalty:
        rows.append(
            np.asarray(
                (
                    math.sqrt(options.quaternion_unit_penalty)
                    * (1.0 - float(quaternion @ quaternion)),
                ),
                dtype=np.float64,
            )
        )
    return np.concatenate(rows)


def _physical_diagnostics(
    motions: Sequence[HandEyeMotionPair],
    parameters: FloatArray,
    options: HoraudDornaikaNonlinearHandEyeOptions,
) -> tuple[tuple[float, float, float, float, float, float], int, float]:
    quaternion = parameters[:4] / np.linalg.norm(parameters[:4])
    physical = np.asarray((0.0, 0.0, 0.0, *parameters[4:7]), dtype=np.float64)

    def physical_residual(values: FloatArray) -> FloatArray:
        delta_quaternion = _rotation_vector_quaternion(values[:3])
        trial_quaternion = _quaternion_multiply(delta_quaternion, quaternion)
        trial = np.asarray((*trial_quaternion, *values[3:6]), dtype=np.float64)
        return _residual(motions, trial, options, include_penalty=False)

    jacobian = _central_jacobian(
        physical_residual,
        physical,
        options.finite_difference_step,
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    spectrum = (
        float(singular[0]),
        float(singular[1]),
        float(singular[2]),
        float(singular[3]),
        float(singular[4]),
        float(singular[5]),
    )
    threshold = options.rank_tolerance * max(spectrum[0], 1.0e-15)
    rank = int(np.count_nonzero(singular > threshold))
    condition = spectrum[0] / spectrum[5] if spectrum[5] > 0.0 else math.inf
    return spectrum, rank, condition


def _central_jacobian(
    function: Callable[[FloatArray], FloatArray], parameters: FloatArray, step: float
) -> FloatArray:
    baseline = function(parameters)
    jacobian: FloatArray = np.empty((len(baseline), len(parameters)), dtype=np.float64)
    for column in range(len(parameters)):
        scale = step * max(1.0, abs(float(parameters[column])))
        positive = parameters.copy()
        negative = parameters.copy()
        positive[column] += scale
        negative[column] -= scale
        jacobian[:, column] = (function(positive) - function(negative)) / (2.0 * scale)
    return jacobian


def _quaternion_rotate_raw(quaternion: FloatArray, vector: FloatArray) -> FloatArray:
    xyz = quaternion[:3]
    scalar = float(quaternion[3])
    return (
        (scalar * scalar - float(xyz @ xyz)) * vector
        + 2.0 * xyz * float(xyz @ vector)
        + 2.0 * scalar * np.cross(xyz, vector)
    )


def _quaternion_multiply(left: FloatArray, right: FloatArray) -> FloatArray:
    left_xyz = left[:3]
    right_xyz = right[:3]
    return np.asarray(
        (
            *(
                float(left[3]) * right_xyz
                + float(right[3]) * left_xyz
                + np.cross(left_xyz, right_xyz)
            ),
            float(left[3] * right[3] - left_xyz @ right_xyz),
        ),
        dtype=np.float64,
    )


def _rotation_vector_quaternion(vector: FloatArray) -> FloatArray:
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-15:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    scale = math.sin(0.5 * angle) / angle
    return np.asarray((*tuple(scale * vector), math.cos(0.5 * angle)), dtype=np.float64)


def _rotation_axis(transform: SE3) -> FloatArray:
    values: FloatArray = np.asarray(transform.rotation_quat_xyzw, dtype=np.float64)
    if values[3] < 0.0:
        values *= -1.0
    norm = float(np.linalg.norm(values[:3]))
    return values[:3] / norm if norm > 0.0 else np.zeros(3, dtype=np.float64)


def _rotation_angle(transform: SE3) -> float:
    x, y, z, scalar = transform.rotation_quat_xyzw
    return 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(scalar))


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, scalar = transform.rotation_quat_xyzw
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * scalar), 2 * (x * z + y * scalar)],
            [2 * (x * y + z * scalar), 1 - 2 * (x * x + z * z), 2 * (y * z - x * scalar)],
            [2 * (x * z - y * scalar), 2 * (y * z + x * scalar), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _validated_motions(
    motions: Sequence[HandEyeMotionPair], min_rotation_rad: float
) -> tuple[HandEyeMotionPair, ...]:
    names: set[str] = set()
    output: list[HandEyeMotionPair] = []
    for pair in sorted(motions, key=lambda item: item.pair_id):
        if not pair.pair_id or pair.pair_id in names:
            raise ValueError("Horaud-Dornaika nonlinear pair IDs must be unique and non-empty")
        if not math.isfinite(pair.weight) or pair.weight <= 0.0:
            raise ValueError("Horaud-Dornaika nonlinear weights must be finite and positive")
        names.add(pair.pair_id)
        if (
            _rotation_angle(pair.motion_a) >= min_rotation_rad
            and _rotation_angle(pair.motion_b) >= min_rotation_rad
        ):
            output.append(pair)
    return tuple(output)


def _validate_options(options: HoraudDornaikaNonlinearHandEyeOptions) -> None:
    if options.min_train_motions < 2:
        raise ValueError("Horaud-Dornaika nonlinear requires at least two train motions")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("Horaud-Dornaika nonlinear holdout ratio must be in [0, 1)")
    if options.max_iterations < 1:
        raise ValueError("Horaud-Dornaika nonlinear max_iterations must be positive")
    if (
        options.rotation_weight != 1.0
        or options.translation_weight != 1.0
        or options.quaternion_unit_penalty != 2.0e6
    ):
        raise ValueError(
            "paper contract requires lambda1=lambda2=1 and quaternion penalty 2e6"
        )
    positive = (
        options.min_rotation_rad,
        options.initial_damping,
        options.damping_increase,
        options.damping_decrease,
        options.gradient_tolerance,
        options.step_tolerance,
        options.relative_cost_tolerance,
        options.finite_difference_step,
        options.rank_tolerance,
        options.max_data_jacobian_condition_number,
        options.max_quaternion_unit_error,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("Horaud-Dornaika nonlinear options must be finite and positive")
    if not 0.0 < options.damping_decrease < 1.0 or options.damping_increase <= 1.0:
        raise ValueError("Horaud-Dornaika nonlinear damping multipliers are invalid")


def _rmse(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _empty(
    status: HoraudDornaikaNonlinearHandEyeStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    reason: str | None = None,
    iterations: tuple[HoraudDornaikaNonlinearHandEyeIteration, ...] = (),
    accepted_steps: int = 0,
    initial_cost: float | None = None,
    final_cost: float | None = None,
    initial_data_rmse: float | None = None,
    final_data_rmse: float | None = None,
    final_gradient: float | None = None,
    quaternion_norm: float | None = None,
    unit_error: float | None = None,
    data_spectrum: tuple[float, float, float, float, float, float] | None = None,
    data_rank: int = 0,
    data_condition: float | None = None,
) -> HoraudDornaikaNonlinearHandEyeResult:
    reasons = {
        "converged": "",
        "insufficient_motions": "too few rotation-excited motion pairs",
        "initialization_failed": "closed-form initializer failed",
        "numerical_failure": "Levenberg-Marquardt linear solve failed",
        "max_iterations": "Levenberg-Marquardt did not meet convergence criteria",
        "unobservable": "physical six-DoF data Jacobian is rank deficient or ill-conditioned",
        "unit_constraint_failed": "paper quaternion penalty did not enforce a unit solution",
    }
    return HoraudDornaikaNonlinearHandEyeResult(
        status=status,
        reason=reason or reasons[status],
        transform_x=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        iterations=iterations,
        accepted_step_count=accepted_steps,
        initial_cost=initial_cost,
        final_cost=final_cost,
        initial_data_residual_rmse=initial_data_rmse,
        final_data_residual_rmse=final_data_rmse,
        final_gradient_inf=final_gradient,
        quaternion_norm=quaternion_norm,
        quaternion_unit_error=unit_error,
        data_jacobian_singular_values=data_spectrum,
        data_jacobian_rank=data_rank,
        data_jacobian_condition_number=data_condition,
        train_evaluation=HandEyeEvaluation(None, None),
        holdout_evaluation=HandEyeEvaluation(None, None),
        probes=(),
    )
