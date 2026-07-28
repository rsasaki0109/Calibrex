"""Dornaika-Horaud simultaneous nonlinear robot-world/hand-eye solver."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.dornaika_horaud_robot_world_hand_eye_solver import (
    DornaikaHoraudRobotWorldHandEyeOptions,
    DornaikaHoraudRobotWorldHandEyeSolver,
)
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyeEvaluation,
    RobotWorldHandEyePosePair,
    RobotWorldHandEyeProbe,
    evaluate_robot_world_hand_eye_known_bad_probes,
    evaluate_robot_world_hand_eye_poses,
)

FloatArray: TypeAlias = NDArray[np.float64]
DornaikaHoraudNonlinearStatus = Literal[
    "converged",
    "insufficient_poses",
    "initialization_failed",
    "numerical_failure",
    "max_iterations",
    "unobservable",
    "unstable_rotation_projection",
]


@dataclass(frozen=True)
class DornaikaHoraudNonlinearOptions:
    min_train_poses: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    max_iterations: int = 50
    initial_damping: float = 1.0e-3
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    gradient_tolerance: float = 1.0e-8
    step_tolerance: float = 1.0e-10
    relative_cost_tolerance: float = 1.0e-12
    rotation_weight: float = 1.0
    translation_weight: float = 1.0
    orthogonality_penalty: float = 1.0e6
    rank_tolerance: float = 1.0e-9
    max_data_jacobian_condition_number: float = 1.0e12
    max_so3_projection_correction_frobenius: float = 1.0e-4
    known_bad_translation_m: float = 0.05
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_margin_m: float = 0.001
    known_bad_rotation_margin_deg: float = 0.1


@dataclass(frozen=True)
class DornaikaHoraudNonlinearIteration:
    iteration: int
    accepted: bool
    cost: float
    trial_cost: float
    damping: float
    gradient_inf: float
    step_norm: float | None


@dataclass(frozen=True)
class DornaikaHoraudNonlinearResult:
    status: DornaikaHoraudNonlinearStatus
    reason: str
    transform_x: SE3 | None
    transform_z: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    iterations: tuple[DornaikaHoraudNonlinearIteration, ...]
    accepted_step_count: int
    initial_cost: float | None
    final_cost: float | None
    initial_data_residual_rmse: float | None
    final_data_residual_rmse: float | None
    final_gradient_inf: float | None
    data_jacobian_singular_values: tuple[float, ...]
    data_jacobian_rank: int
    data_jacobian_condition_number: float | None
    raw_rotation_x_determinant: float | None
    raw_rotation_z_determinant: float | None
    rotation_x_orthogonality_error_frobenius: float | None
    rotation_z_orthogonality_error_frobenius: float | None
    rotation_x_projection_correction_frobenius: float | None
    rotation_z_projection_correction_frobenius: float | None
    train_evaluation: RobotWorldHandEyeEvaluation
    holdout_evaluation: RobotWorldHandEyeEvaluation
    probes: tuple[RobotWorldHandEyeProbe, ...] = ()

    @property
    def transform_y(self) -> SE3 | None:
        """Expose the paper's ``Z`` through the shared robot-world ``Y`` role."""

        return self.transform_z

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly optimization evidence and provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "transform_z": self.transform_z.as_dict() if self.transform_z else None,
            "shared_role_alias": "Y := Z",
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "iterations": [item.__dict__ for item in self.iterations],
            "accepted_step_count": self.accepted_step_count,
            "initial_cost": self.initial_cost,
            "final_cost": self.final_cost,
            "initial_data_residual_rmse": self.initial_data_residual_rmse,
            "final_data_residual_rmse": self.final_data_residual_rmse,
            "final_gradient_inf": self.final_gradient_inf,
            "data_jacobian_singular_values": list(self.data_jacobian_singular_values),
            "data_jacobian_rank": self.data_jacobian_rank,
            "data_jacobian_condition_number": self.data_jacobian_condition_number,
            "raw_rotation_x_determinant": self.raw_rotation_x_determinant,
            "raw_rotation_z_determinant": self.raw_rotation_z_determinant,
            "rotation_x_orthogonality_error_frobenius": (
                self.rotation_x_orthogonality_error_frobenius
            ),
            "rotation_z_orthogonality_error_frobenius": (
                self.rotation_z_orthogonality_error_frobenius
            ),
            "rotation_x_projection_correction_frobenius": (
                self.rotation_x_projection_correction_frobenius
            ),
            "rotation_z_projection_correction_frobenius": (
                self.rotation_z_projection_correction_frobenius
            ),
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "dornaika_horaud_robot_world_hand_eye_nonlinear/v0.1",
            "paper": {
                "title": "Simultaneous Robot-World and Hand-Eye Calibration",
                "authors": "Fadi Dornaika and Radu Horaud",
                "doi": "10.1109/70.704233",
                "author_manuscript": "https://arxiv.org/abs/2311.11818",
            },
            "equation": "A_i X = Z B_i",
            "selected_paper_method": "section_III_B_nonlinear_24_parameter",
            "initializer": "dornaika_horaud_closed_form_quaternion",
            "objective_contract": (
                "paper Section III-B: weighted Frobenius rotation closure plus "
                "translation closure and mu3=mu4 orthogonality penalties"
            ),
            "parameterization": "row-major R_X(9), R_Z(9), t_X(3), t_Z(3)",
            "optimizer": "deterministic Levenberg-Marquardt with analytic Jacobian",
            "paper_weights": {
                "mu1_rotation": 1.0,
                "mu2_translation": 1.0,
                "mu3_rotation_x_orthogonality": 1.0e6,
                "mu4_rotation_z_orthogonality": 1.0e6,
            },
            "units_limitation": (
                "the paper's mu1=mu2=1 mixes dimensionless rotation-matrix entries "
                "and translations in dataset length units"
            ),
            "output_contract": "nearest-SO(3) projection is gated before typed SE3 emission",
            "external_code_executed": False,
            "evaluation_contract": "closure on held-out absolute pose pairs without refitting",
        }


class DornaikaHoraudNonlinearSolver:
    """Refine both transforms with the paper's 24-parameter nonlinear objective."""

    def solve(
        self,
        poses: Sequence[RobotWorldHandEyePosePair],
        options: DornaikaHoraudNonlinearOptions | None = None,
    ) -> DornaikaHoraudNonlinearResult:
        solver_options = options or DornaikaHoraudNonlinearOptions()
        _validate_options(solver_options)
        usable = _validated_poses(poses)
        train_indices, holdout_indices = split_indices(
            len(usable), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        train_ids = tuple(pair.pair_id for pair in train)
        holdout_ids = tuple(pair.pair_id for pair in holdout)
        if len(train) < solver_options.min_train_poses:
            return _empty("insufficient_poses", train_ids, holdout_ids)

        initializer = DornaikaHoraudRobotWorldHandEyeSolver().solve(
            usable,
            DornaikaHoraudRobotWorldHandEyeOptions(
                min_train_poses=solver_options.min_train_poses,
                holdout_ratio=solver_options.holdout_ratio,
                split_seed=solver_options.split_seed,
                rank_tolerance=solver_options.rank_tolerance,
                minimum_rotation_normalized_gap=1.0e-6,
                max_translation_condition_number=(
                    solver_options.max_data_jacobian_condition_number
                ),
            ),
        )
        if initializer.transform_x is None or initializer.transform_z is None:
            return _empty(
                "initialization_failed",
                train_ids,
                holdout_ids,
                reason=f"closed-form initializer status: {initializer.status}",
            )
        parameters = _parameters(initializer.transform_x, initializer.transform_z)
        residual, jacobian, data_row_count = _linearize(train, parameters, solver_options)
        cost = 0.5 * float(residual @ residual)
        initial_cost = cost
        initial_data_rmse = _rmse(residual[:data_row_count])
        damping = solver_options.initial_damping
        history: list[DornaikaHoraudNonlinearIteration] = []
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
                step = np.linalg.solve(
                    normal + damping * np.diag(diagonal),
                    -gradient,
                )
            except np.linalg.LinAlgError:
                numerical_failure = True
                break
            step_norm = float(np.linalg.norm(step))
            trial_parameters = parameters + step
            trial_residual, trial_jacobian, _trial_data_rows = _linearize(
                train, trial_parameters, solver_options
            )
            trial_cost = 0.5 * float(trial_residual @ trial_residual)
            accepted = math.isfinite(trial_cost) and trial_cost < cost
            history.append(
                DornaikaHoraudNonlinearIteration(
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
        status: DornaikaHoraudNonlinearStatus
        if numerical_failure:
            status = "numerical_failure"
        elif converged or final_gradient <= solver_options.gradient_tolerance:
            status = "converged"
        else:
            status = "max_iterations"
        diagnostics = _diagnostics(train, parameters, solver_options, residual, data_row_count)
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
                final_gradient=final_gradient,
                **diagnostics,
            )
        if (
            diagnostics["data_rank"] < 24
            or diagnostics["data_condition"] is None
            or diagnostics["data_condition"] > solver_options.max_data_jacobian_condition_number
        ):
            return _empty(
                "unobservable",
                train_ids,
                holdout_ids,
                iterations=tuple(history),
                accepted_steps=accepted_steps,
                initial_cost=initial_cost,
                final_cost=cost,
                initial_data_rmse=initial_data_rmse,
                final_gradient=final_gradient,
                **diagnostics,
            )
        correction_max = max(diagnostics["projection_x"], diagnostics["projection_z"])
        if correction_max > solver_options.max_so3_projection_correction_frobenius:
            return _empty(
                "unstable_rotation_projection",
                train_ids,
                holdout_ids,
                iterations=tuple(history),
                accepted_steps=accepted_steps,
                initial_cost=initial_cost,
                final_cost=cost,
                initial_data_rmse=initial_data_rmse,
                final_gradient=final_gradient,
                **diagnostics,
            )
        rotation_x = _nearest_rotation(parameters[:9].reshape((3, 3)))
        rotation_z = _nearest_rotation(parameters[9:18].reshape((3, 3)))
        transform_x = _se3(rotation_x, parameters[18:21])
        transform_z = _se3(rotation_z, parameters[21:24])
        train_evaluation = evaluate_robot_world_hand_eye_poses(train, transform_x, transform_z)
        holdout_evaluation = evaluate_robot_world_hand_eye_poses(holdout, transform_x, transform_z)
        probes = evaluate_robot_world_hand_eye_known_bad_probes(
            holdout,
            transform_x,
            transform_z,
            holdout_evaluation,
            solver_options,
        )
        return DornaikaHoraudNonlinearResult(
            status="converged",
            reason="paper Section III-B simultaneous nonlinear objective converged",
            transform_x=transform_x,
            transform_z=transform_z,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            iterations=tuple(history),
            accepted_step_count=accepted_steps,
            initial_cost=initial_cost,
            final_cost=cost,
            initial_data_residual_rmse=initial_data_rmse,
            final_data_residual_rmse=diagnostics["final_data_rmse"],
            final_gradient_inf=final_gradient,
            data_jacobian_singular_values=diagnostics["data_spectrum"],
            data_jacobian_rank=diagnostics["data_rank"],
            data_jacobian_condition_number=diagnostics["data_condition"],
            raw_rotation_x_determinant=diagnostics["determinant_x"],
            raw_rotation_z_determinant=diagnostics["determinant_z"],
            rotation_x_orthogonality_error_frobenius=diagnostics["orthogonality_x"],
            rotation_z_orthogonality_error_frobenius=diagnostics["orthogonality_z"],
            rotation_x_projection_correction_frobenius=diagnostics["projection_x"],
            rotation_z_projection_correction_frobenius=diagnostics["projection_z"],
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=probes,
        )


class _Diagnostics(TypedDict):
    data_spectrum: tuple[float, ...]
    data_rank: int
    data_condition: float | None
    final_data_rmse: float
    determinant_x: float
    determinant_z: float
    orthogonality_x: float
    orthogonality_z: float
    projection_x: float
    projection_z: float


def _linearize(
    poses: Sequence[RobotWorldHandEyePosePair],
    parameters: FloatArray,
    options: DornaikaHoraudNonlinearOptions,
) -> tuple[FloatArray, FloatArray, int]:
    rotation_x = parameters[:9].reshape((3, 3))
    rotation_z = parameters[9:18].reshape((3, 3))
    translation_x = parameters[18:21]
    translation_z = parameters[21:24]
    identity: FloatArray = np.eye(3, dtype=np.float64)
    residual_rows: list[FloatArray] = []
    jacobian_rows: list[FloatArray] = []
    for pair in poses:
        rotation_a = _rotation_matrix(pair.pose_a)
        rotation_b = _rotation_matrix(pair.pose_b)
        translation_a = np.asarray(pair.pose_a.translation_m)
        translation_b = np.asarray(pair.pose_b.translation_m)
        root_rotation = math.sqrt(pair.weight * options.rotation_weight)
        rotation_residual = root_rotation * (rotation_a @ rotation_x - rotation_z @ rotation_b)
        rotation_jacobian: FloatArray = np.zeros((9, 24), dtype=np.float64)
        rotation_jacobian[:, :9] = root_rotation * np.kron(rotation_a, identity)
        rotation_jacobian[:, 9:18] = -root_rotation * np.kron(identity, rotation_b.T)
        residual_rows.append(rotation_residual.reshape(-1))
        jacobian_rows.append(rotation_jacobian)

        root_translation = math.sqrt(pair.weight * options.translation_weight)
        translation_residual = root_translation * (
            rotation_a @ translation_x + translation_a - rotation_z @ translation_b - translation_z
        )
        translation_jacobian: FloatArray = np.zeros((3, 24), dtype=np.float64)
        translation_jacobian[:, 9:18] = -root_translation * np.kron(
            identity, translation_b.reshape((1, 3))
        )
        translation_jacobian[:, 18:21] = root_translation * rotation_a
        translation_jacobian[:, 21:24] = -root_translation * identity
        residual_rows.append(translation_residual)
        jacobian_rows.append(translation_jacobian)
    data_row_count = len(residual_rows) * 6
    root_penalty = math.sqrt(options.orthogonality_penalty)
    for offset, rotation in ((0, rotation_x), (9, rotation_z)):
        penalty_residual = root_penalty * (rotation @ rotation.T - identity)
        penalty_jacobian: FloatArray = np.zeros((9, 24), dtype=np.float64)
        penalty_jacobian[:, offset : offset + 9] = root_penalty * _orthogonality_jacobian(rotation)
        residual_rows.append(penalty_residual.reshape(-1))
        jacobian_rows.append(penalty_jacobian)
    return np.concatenate(residual_rows), np.vstack(jacobian_rows), data_row_count


def _diagnostics(
    poses: Sequence[RobotWorldHandEyePosePair],
    parameters: FloatArray,
    options: DornaikaHoraudNonlinearOptions,
    residual: FloatArray,
    data_row_count: int,
) -> _Diagnostics:
    _full_residual, full_jacobian, _rows = _linearize(poses, parameters, options)
    data_jacobian = full_jacobian[:data_row_count]
    singular = np.linalg.svd(data_jacobian, compute_uv=False)
    spectrum = tuple(float(value) for value in singular)
    threshold = options.rank_tolerance * spectrum[0]
    rank = int(np.count_nonzero(singular > threshold))
    condition = spectrum[0] / spectrum[-1] if spectrum[-1] > 0.0 else math.inf
    rotation_x = parameters[:9].reshape((3, 3))
    rotation_z = parameters[9:18].reshape((3, 3))
    projected_x = _nearest_rotation(rotation_x)
    projected_z = _nearest_rotation(rotation_z)
    return _Diagnostics(
        data_spectrum=spectrum,
        data_rank=rank,
        data_condition=condition,
        final_data_rmse=_rmse(residual[:data_row_count]),
        determinant_x=float(np.linalg.det(rotation_x)),
        determinant_z=float(np.linalg.det(rotation_z)),
        orthogonality_x=float(np.linalg.norm(rotation_x @ rotation_x.T - np.eye(3), ord="fro")),
        orthogonality_z=float(np.linalg.norm(rotation_z @ rotation_z.T - np.eye(3), ord="fro")),
        projection_x=float(np.linalg.norm(rotation_x - projected_x, ord="fro")),
        projection_z=float(np.linalg.norm(rotation_z - projected_z, ord="fro")),
    )


def _orthogonality_jacobian(rotation: FloatArray) -> FloatArray:
    jacobian: FloatArray = np.zeros((9, 9), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            residual_index = 3 * row + column
            for variable_row in range(3):
                for variable_column in range(3):
                    variable_index = 3 * variable_row + variable_column
                    value = 0.0
                    if row == variable_row:
                        value += rotation[column, variable_column]
                    if column == variable_row:
                        value += rotation[row, variable_column]
                    jacobian[residual_index, variable_index] = value
    return jacobian


def _parameters(transform_x: SE3, transform_z: SE3) -> FloatArray:
    return np.concatenate(
        (
            _rotation_matrix(transform_x).reshape(-1),
            _rotation_matrix(transform_z).reshape(-1),
            np.asarray(transform_x.translation_m),
            np.asarray(transform_z.translation_m),
        )
    )


def _nearest_rotation(matrix: FloatArray) -> FloatArray:
    left, _singular, right_t = np.linalg.svd(matrix)
    correction: FloatArray = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(left @ right_t)
    return left @ correction @ right_t


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


def _se3(rotation: FloatArray, translation: FloatArray) -> SE3:
    return SE3(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
    )


def _rmse(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _validated_poses(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[RobotWorldHandEyePosePair, ...]:
    names: set[str] = set()
    output: list[RobotWorldHandEyePosePair] = []
    for pair in sorted(poses, key=lambda item: item.pair_id):
        if not pair.pair_id or pair.pair_id in names:
            raise ValueError("Dornaika-Horaud nonlinear pair IDs must be unique and non-empty")
        if not math.isfinite(pair.weight) or pair.weight <= 0.0:
            raise ValueError("Dornaika-Horaud nonlinear weights must be finite and positive")
        names.add(pair.pair_id)
        output.append(pair)
    return tuple(output)


def _validate_options(options: DornaikaHoraudNonlinearOptions) -> None:
    if options.min_train_poses < 3:
        raise ValueError("Dornaika-Horaud nonlinear requires at least three train poses")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("Dornaika-Horaud nonlinear holdout ratio must be in [0, 1)")
    if options.max_iterations < 1:
        raise ValueError("Dornaika-Horaud nonlinear max_iterations must be positive")
    if (
        options.rotation_weight != 1.0
        or options.translation_weight != 1.0
        or options.orthogonality_penalty != 1.0e6
    ):
        raise ValueError(
            "Dornaika-Horaud nonlinear paper contract requires mu1=mu2=1 and mu3=mu4=1e6"
        )
    positive = (
        options.initial_damping,
        options.damping_increase,
        options.damping_decrease,
        options.gradient_tolerance,
        options.step_tolerance,
        options.relative_cost_tolerance,
        options.rotation_weight,
        options.translation_weight,
        options.orthogonality_penalty,
        options.rank_tolerance,
        options.max_data_jacobian_condition_number,
        options.max_so3_projection_correction_frobenius,
        options.known_bad_translation_m,
        options.known_bad_rotation_deg,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("Dornaika-Horaud nonlinear options must be finite and positive")
    if not 0.0 < options.damping_decrease < 1.0 or options.damping_increase <= 1.0:
        raise ValueError("Dornaika-Horaud nonlinear damping multipliers are invalid")
    margins = (
        options.known_bad_translation_margin_m,
        options.known_bad_rotation_margin_deg,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in margins):
        raise ValueError("Dornaika-Horaud nonlinear probe margins must be non-negative")


def _empty(
    status: DornaikaHoraudNonlinearStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    reason: str | None = None,
    iterations: tuple[DornaikaHoraudNonlinearIteration, ...] = (),
    accepted_steps: int = 0,
    initial_cost: float | None = None,
    final_cost: float | None = None,
    initial_data_rmse: float | None = None,
    final_gradient: float | None = None,
    data_spectrum: tuple[float, ...] = (),
    data_rank: int = 0,
    data_condition: float | None = None,
    final_data_rmse: float | None = None,
    determinant_x: float | None = None,
    determinant_z: float | None = None,
    orthogonality_x: float | None = None,
    orthogonality_z: float | None = None,
    projection_x: float | None = None,
    projection_z: float | None = None,
) -> DornaikaHoraudNonlinearResult:
    reasons = {
        "insufficient_poses": "fewer than three train absolute-pose pairs",
        "initialization_failed": "closed-form initialization failed",
        "numerical_failure": "Levenberg-Marquardt linear solve failed",
        "max_iterations": "Levenberg-Marquardt did not satisfy a stopping criterion",
        "unobservable": "final data Jacobian is rank deficient or unstable",
        "unstable_rotation_projection": "optimized matrices require excessive SO(3) correction",
    }
    return DornaikaHoraudNonlinearResult(
        status=status,
        reason=reason or reasons.get(status, "nonlinear solve did not produce transforms"),
        transform_x=None,
        transform_z=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        iterations=iterations,
        accepted_step_count=accepted_steps,
        initial_cost=initial_cost,
        final_cost=final_cost,
        initial_data_residual_rmse=initial_data_rmse,
        final_data_residual_rmse=final_data_rmse,
        final_gradient_inf=final_gradient,
        data_jacobian_singular_values=data_spectrum,
        data_jacobian_rank=data_rank,
        data_jacobian_condition_number=data_condition,
        raw_rotation_x_determinant=determinant_x,
        raw_rotation_z_determinant=determinant_z,
        rotation_x_orthogonality_error_frobenius=orthogonality_x,
        rotation_z_orthogonality_error_frobenius=orthogonality_z,
        rotation_x_projection_correction_frobenius=projection_x,
        rotation_z_projection_correction_frobenius=projection_z,
        train_evaluation=RobotWorldHandEyeEvaluation(None, None),
        holdout_evaluation=RobotWorldHandEyeEvaluation(None, None),
    )
