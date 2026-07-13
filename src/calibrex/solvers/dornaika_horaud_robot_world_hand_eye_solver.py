"""Dornaika-Horaud closed-form quaternion robot-world/hand-eye solver."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyeEvaluation,
    RobotWorldHandEyePosePair,
    RobotWorldHandEyeProbe,
    evaluate_robot_world_hand_eye_known_bad_probes,
    evaluate_robot_world_hand_eye_poses,
)

FloatArray: TypeAlias = NDArray[np.float64]
DornaikaHoraudRobotWorldHandEyeStatus = Literal[
    "converged",
    "insufficient_poses",
    "degenerate_rotation",
    "degenerate_translation",
]


@dataclass(frozen=True)
class DornaikaHoraudRobotWorldHandEyeOptions:
    min_train_poses: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-9
    minimum_rotation_normalized_gap: float = 1.0e-3
    max_translation_condition_number: float = 1.0e8
    known_bad_translation_m: float = 0.05
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_margin_m: float = 0.001
    known_bad_rotation_margin_deg: float = 0.1


@dataclass(frozen=True)
class DornaikaHoraudRobotWorldHandEyeResult:
    status: DornaikaHoraudRobotWorldHandEyeStatus
    reason: str
    transform_x: SE3 | None
    transform_z: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    rotation_singular_values: tuple[float, ...]
    rotation_normalized_gap: float | None
    rotation_dominant_multiplicity: int
    rotation_objective: float | None
    quaternion_x_unit_error: float | None
    quaternion_z_unit_error: float | None
    quaternion_sign_flip_count: int
    quaternion_sign_synchronization_fraction: float | None
    translation_singular_values: tuple[float, ...]
    translation_rank: int
    translation_condition_number: float | None
    train_evaluation: RobotWorldHandEyeEvaluation
    holdout_evaluation: RobotWorldHandEyeEvaluation
    probes: tuple[RobotWorldHandEyeProbe, ...] = ()

    @property
    def transform_y(self) -> SE3 | None:
        """Expose the paper's ``Z`` through the shared robot-world ``Y`` role."""

        return self.transform_z

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly estimates, diagnostics, and paper provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "transform_z": self.transform_z.as_dict() if self.transform_z else None,
            "shared_role_alias": "Y := Z",
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "rotation_singular_values": list(self.rotation_singular_values),
            "rotation_normalized_gap": self.rotation_normalized_gap,
            "rotation_dominant_multiplicity": self.rotation_dominant_multiplicity,
            "rotation_objective": self.rotation_objective,
            "quaternion_x_unit_error": self.quaternion_x_unit_error,
            "quaternion_z_unit_error": self.quaternion_z_unit_error,
            "quaternion_sign_flip_count": self.quaternion_sign_flip_count,
            "quaternion_sign_synchronization_fraction": (
                self.quaternion_sign_synchronization_fraction
            ),
            "translation_singular_values": list(self.translation_singular_values),
            "translation_rank": self.translation_rank,
            "translation_condition_number": self.translation_condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "dornaika_horaud_robot_world_hand_eye_closed_form/v0.1",
            "paper": {
                "title": "Simultaneous Robot-World and Hand-Eye Calibration",
                "authors": "Fadi Dornaika and Radu Horaud",
                "doi": "10.1109/70.704233",
                "author_manuscript": "https://arxiv.org/abs/2311.11818",
            },
            "equation": "A_i X = Z B_i",
            "rotation_contract": (
                "paper equations (9)-(15): minimize the positive quaternion quadratic "
                "under separate unit constraints using the dominant singular pair of "
                "C = -sum_i weight_i Q(q_Ai)^T W(q_Bi)"
            ),
            "translation_contract": (
                "weighted least squares [R_Ai, -I] [t_X; t_Z] = R_Z t_Bi - t_Ai"
            ),
            "selected_paper_method": "closed_form_quaternion_then_translation",
            "quaternion_sign_contract": (
                "deterministic maximum-spanning-tree synchronization using the invariant "
                "sign of dot(q_Ai,q_Aj)*dot(q_Bi,q_Bj) before applying the paper equations"
            ),
            "nonlinear_method_executed": False,
            "external_code_executed": False,
            "evaluation_contract": "closure on held-out absolute pose pairs without refitting",
        }


class DornaikaHoraudRobotWorldHandEyeSolver:
    """Solve ``A_i X = Z B_i`` with the paper's closed-form quaternion method."""

    def solve(
        self,
        poses: Sequence[RobotWorldHandEyePosePair],
        options: DornaikaHoraudRobotWorldHandEyeOptions | None = None,
    ) -> DornaikaHoraudRobotWorldHandEyeResult:
        solver_options = options or DornaikaHoraudRobotWorldHandEyeOptions()
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

        signs, sign_flips, sign_fraction = _quaternion_signs(train)
        quaternion_matrix = _quaternion_cross_matrix(train, signs)
        left, singular, right_t = np.linalg.svd(quaternion_matrix)
        spectrum = tuple(float(value) for value in singular)
        if not spectrum or spectrum[0] <= 0.0:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                rotation_spectrum=spectrum,
                sign_flips=sign_flips,
                sign_fraction=sign_fraction,
            )
        threshold = solver_options.rank_tolerance * spectrum[0]
        multiplicity = int(np.count_nonzero(singular >= spectrum[0] - threshold))
        gap = (spectrum[0] - spectrum[1]) / spectrum[0]
        if multiplicity != 1 or gap < solver_options.minimum_rotation_normalized_gap:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                rotation_spectrum=spectrum,
                rotation_gap=gap,
                dominant_multiplicity=multiplicity,
                sign_flips=sign_flips,
                sign_fraction=sign_fraction,
            )

        quaternion_x = left[:, 0]
        quaternion_z = -right_t[0]
        quaternion_x /= np.linalg.norm(quaternion_x)
        quaternion_z /= np.linalg.norm(quaternion_z)
        objective = _rotation_objective(train, signs, quaternion_x, quaternion_z)
        unit_error_x = abs(float(quaternion_x @ quaternion_x) - 1.0)
        unit_error_z = abs(float(quaternion_z @ quaternion_z) - 1.0)
        rotation_z = _rotation_matrix_from_scalar_first_quaternion(quaternion_z)
        translation = _solve_translations(train, rotation_z, solver_options)
        if translation.solution is None:
            return _empty(
                "degenerate_translation",
                train_ids,
                holdout_ids,
                rotation_spectrum=spectrum,
                rotation_gap=gap,
                dominant_multiplicity=multiplicity,
                rotation_objective=objective,
                unit_error_x=unit_error_x,
                unit_error_z=unit_error_z,
                sign_flips=sign_flips,
                sign_fraction=sign_fraction,
                translation_spectrum=translation.singular_values,
                translation_rank=translation.rank,
                translation_condition=translation.condition_number,
            )
        transform_x = _se3_from_scalar_first_quaternion(translation.solution[:3], quaternion_x)
        transform_z = _se3_from_scalar_first_quaternion(translation.solution[3:], quaternion_z)
        train_evaluation = evaluate_robot_world_hand_eye_poses(train, transform_x, transform_z)
        holdout_evaluation = evaluate_robot_world_hand_eye_poses(holdout, transform_x, transform_z)
        probes = evaluate_robot_world_hand_eye_known_bad_probes(
            holdout,
            transform_x,
            transform_z,
            holdout_evaluation,
            solver_options,
        )
        return DornaikaHoraudRobotWorldHandEyeResult(
            status="converged",
            reason="closed-form unit-quaternion rotations and conditional translations solved",
            transform_x=transform_x,
            transform_z=transform_z,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            rotation_singular_values=spectrum,
            rotation_normalized_gap=gap,
            rotation_dominant_multiplicity=multiplicity,
            rotation_objective=objective,
            quaternion_x_unit_error=unit_error_x,
            quaternion_z_unit_error=unit_error_z,
            quaternion_sign_flip_count=sign_flips,
            quaternion_sign_synchronization_fraction=sign_fraction,
            translation_singular_values=translation.singular_values,
            translation_rank=translation.rank,
            translation_condition_number=translation.condition_number,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=probes,
        )


@dataclass(frozen=True)
class _TranslationSolution:
    solution: FloatArray | None
    singular_values: tuple[float, ...]
    rank: int
    condition_number: float | None


def _quaternion_cross_matrix(
    poses: Sequence[RobotWorldHandEyePosePair],
    signs: FloatArray,
) -> FloatArray:
    matrix = np.zeros((4, 4), dtype=np.float64)
    for index, pair in enumerate(poses):
        quaternion_a = _scalar_first_quaternion(pair.pose_a)
        quaternion_b = signs[index] * _scalar_first_quaternion(pair.pose_b)
        matrix -= pair.weight * _left_matrix(quaternion_a).T @ _right_matrix(quaternion_b)
    return matrix


def _rotation_objective(
    poses: Sequence[RobotWorldHandEyePosePair],
    signs: FloatArray,
    quaternion_x: FloatArray,
    quaternion_z: FloatArray,
) -> float:
    weighted_squared_error = 0.0
    total_weight = 0.0
    for index, pair in enumerate(poses):
        residual = (
            _left_matrix(_scalar_first_quaternion(pair.pose_a)) @ quaternion_x
            - _right_matrix(signs[index] * _scalar_first_quaternion(pair.pose_b)) @ quaternion_z
        )
        weighted_squared_error += pair.weight * float(residual @ residual)
        total_weight += pair.weight
    return weighted_squared_error / total_weight


def _quaternion_signs(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[FloatArray, int, float | None]:
    """Synchronize the quaternion double-cover signs without a calibration estimate."""

    quaternions_a = np.vstack([_scalar_first_quaternion(pair.pose_a) for pair in poses])
    quaternions_b = np.vstack([_scalar_first_quaternion(pair.pose_b) for pair in poses])
    scores = (quaternions_a @ quaternions_a.T) * (quaternions_b @ quaternions_b.T)
    count = len(poses)
    signs = np.ones(count, dtype=np.float64)
    selected = np.zeros(count, dtype=np.bool_)
    selected[0] = True
    best_strength = np.abs(scores[0]).copy()
    best_parent = np.zeros(count, dtype=np.int64)
    best_strength[0] = -1.0
    for _index in range(1, count):
        child = int(np.argmax(best_strength))
        if best_strength[child] < 0.0:
            break
        parent = int(best_parent[child])
        edge = scores[parent, child]
        signs[child] = signs[parent] * (1.0 if edge >= 0.0 else -1.0)
        selected[child] = True
        best_strength[child] = -1.0
        candidates = np.flatnonzero(~selected)
        strengths = np.abs(scores[child, candidates])
        improved = strengths > best_strength[candidates]
        updated = candidates[improved]
        best_strength[updated] = strengths[improved]
        best_parent[updated] = child
    upper = np.triu_indices(count, k=1)
    pair_scores = scores[upper]
    predicted = signs[upper[0]] * signs[upper[1]]
    total_strength = float(np.sum(np.abs(pair_scores)))
    consistent_strength = float(np.sum(np.abs(pair_scores[predicted * pair_scores >= 0.0])))
    fraction = consistent_strength / total_strength if total_strength > 0.0 else None
    return signs, int(np.count_nonzero(signs < 0.0)), fraction


def _solve_translations(
    poses: Sequence[RobotWorldHandEyePosePair],
    rotation_z: FloatArray,
    options: DornaikaHoraudRobotWorldHandEyeOptions,
) -> _TranslationSolution:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    identity = np.eye(3, dtype=np.float64)
    for pair in poses:
        root_weight = math.sqrt(pair.weight)
        rotation_a = _rotation_matrix(pair.pose_a)
        rows.append(root_weight * np.hstack((rotation_a, -identity)))
        rhs.append(
            root_weight
            * (
                rotation_z @ np.asarray(pair.pose_b.translation_m)
                - np.asarray(pair.pose_a.translation_m)
            )
        )
    matrix = np.vstack(rows)
    vector = np.concatenate(rhs)
    singular = np.linalg.svd(matrix, compute_uv=False)
    spectrum = tuple(float(value) for value in singular)
    threshold = options.rank_tolerance * spectrum[0]
    rank = int(np.count_nonzero(singular > threshold))
    condition = spectrum[0] / spectrum[-1] if spectrum[-1] > 0.0 else math.inf
    if rank < 6 or condition > options.max_translation_condition_number:
        return _TranslationSolution(None, spectrum, rank, condition)
    solution, _residuals, _rank, _values = np.linalg.lstsq(matrix, vector, rcond=None)
    return _TranslationSolution(solution, spectrum, rank, condition)


def _left_matrix(quaternion: FloatArray) -> FloatArray:
    scalar, x, y, z = quaternion
    return np.asarray(
        [
            [scalar, -x, -y, -z],
            [x, scalar, -z, y],
            [y, z, scalar, -x],
            [z, -y, x, scalar],
        ],
        dtype=np.float64,
    )


def _right_matrix(quaternion: FloatArray) -> FloatArray:
    scalar, x, y, z = quaternion
    return np.asarray(
        [
            [scalar, -x, -y, -z],
            [x, scalar, z, -y],
            [y, -z, scalar, x],
            [z, y, -x, scalar],
        ],
        dtype=np.float64,
    )


def _scalar_first_quaternion(transform: SE3) -> FloatArray:
    x, y, z, scalar = transform.rotation_quat_xyzw
    quaternion = np.asarray((scalar, x, y, z), dtype=np.float64)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def _se3_from_scalar_first_quaternion(translation: FloatArray, quaternion: FloatArray) -> SE3:
    return SE3(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
            float(quaternion[0]),
        ),
    )


def _rotation_matrix_from_scalar_first_quaternion(quaternion: FloatArray) -> FloatArray:
    scalar, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * scalar), 2 * (x * z + y * scalar)],
            [2 * (x * y + z * scalar), 1 - 2 * (x * x + z * z), 2 * (y * z - x * scalar)],
            [2 * (x * z - y * scalar), 2 * (y * z + x * scalar), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(transform: SE3) -> FloatArray:
    return _rotation_matrix_from_scalar_first_quaternion(_scalar_first_quaternion(transform))


def _validated_poses(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[RobotWorldHandEyePosePair, ...]:
    names: set[str] = set()
    output: list[RobotWorldHandEyePosePair] = []
    for pair in sorted(poses, key=lambda item: item.pair_id):
        if not pair.pair_id or pair.pair_id in names:
            raise ValueError("Dornaika-Horaud pair IDs must be unique and non-empty")
        if not math.isfinite(pair.weight) or pair.weight <= 0.0:
            raise ValueError("Dornaika-Horaud weights must be finite and positive")
        names.add(pair.pair_id)
        output.append(pair)
    return tuple(output)


def _validate_options(options: DornaikaHoraudRobotWorldHandEyeOptions) -> None:
    if options.min_train_poses < 3:
        raise ValueError("Dornaika-Horaud requires at least three train poses")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("Dornaika-Horaud holdout ratio must be in [0, 1)")
    positive = (
        options.rank_tolerance,
        options.minimum_rotation_normalized_gap,
        options.max_translation_condition_number,
        options.known_bad_translation_m,
        options.known_bad_rotation_deg,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("Dornaika-Horaud numerical options must be finite and positive")
    margins = (
        options.known_bad_translation_margin_m,
        options.known_bad_rotation_margin_deg,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in margins):
        raise ValueError("Dornaika-Horaud probe margins must be finite and non-negative")


def _empty(
    status: DornaikaHoraudRobotWorldHandEyeStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    rotation_spectrum: tuple[float, ...] = (),
    rotation_gap: float | None = None,
    dominant_multiplicity: int = 0,
    rotation_objective: float | None = None,
    unit_error_x: float | None = None,
    unit_error_z: float | None = None,
    sign_flips: int = 0,
    sign_fraction: float | None = None,
    translation_spectrum: tuple[float, ...] = (),
    translation_rank: int = 0,
    translation_condition: float | None = None,
) -> DornaikaHoraudRobotWorldHandEyeResult:
    reasons = {
        "insufficient_poses": "fewer than three train absolute-pose pairs",
        "degenerate_rotation": "closed-form quaternion minimum is not unique",
        "degenerate_translation": "conditional translation system is rank deficient or unstable",
    }
    return DornaikaHoraudRobotWorldHandEyeResult(
        status=status,
        reason=reasons.get(status, "Dornaika-Horaud solve did not produce transforms"),
        transform_x=None,
        transform_z=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        rotation_singular_values=rotation_spectrum,
        rotation_normalized_gap=rotation_gap,
        rotation_dominant_multiplicity=dominant_multiplicity,
        rotation_objective=rotation_objective,
        quaternion_x_unit_error=unit_error_x,
        quaternion_z_unit_error=unit_error_z,
        quaternion_sign_flip_count=sign_flips,
        quaternion_sign_synchronization_fraction=sign_fraction,
        translation_singular_values=translation_spectrum,
        translation_rank=translation_rank,
        translation_condition_number=translation_condition,
        train_evaluation=RobotWorldHandEyeEvaluation(None, None),
        holdout_evaluation=RobotWorldHandEyeEvaluation(None, None),
    )
