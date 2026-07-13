"""Horaud-Dornaika closed-form hand-eye calibration for ``A X = X B``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.evaluation.holdout import split_indices
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
HoraudDornaikaStatus = Literal[
    "converged", "insufficient_motions", "degenerate_rotation", "degenerate_translation"
]


@dataclass(frozen=True)
class HoraudDornaikaHandEyeOptions:
    """Numerical and evidence policy for the closed-form baseline."""

    min_train_motions: int = 3
    min_rotation_rad: float = math.radians(1.0)
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-7
    max_condition_number: float = 1.0e6
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_rotation_margin_deg: float = 0.1
    known_bad_translation_margin_m: float = 0.005


@dataclass(frozen=True)
class HoraudDornaikaHandEyeResult:
    """Closed-form estimate plus split, observability, and falsification evidence."""

    status: HoraudDornaikaStatus
    reason: str
    transform_x: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    rotation_axis_singular_values: tuple[float, float, float] | None
    rotation_axis_rank: int
    rotation_axis_condition_number: float | None
    quaternion_objective_eigenvalues: tuple[float, float, float, float] | None
    quaternion_minimum_eigengap: float | None
    quaternion_normalized_eigengap: float | None
    translation_singular_values: tuple[float, float, float] | None
    translation_rank: int
    translation_condition_number: float | None
    train_evaluation: HandEyeEvaluation
    holdout_evaluation: HandEyeEvaluation
    probes: tuple[HandEyeProbeResult, ...]

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly output with primary-paper provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "rotation_axis_singular_values": self.rotation_axis_singular_values,
            "rotation_axis_rank": self.rotation_axis_rank,
            "rotation_axis_condition_number": self.rotation_axis_condition_number,
            "quaternion_objective_eigenvalues": self.quaternion_objective_eigenvalues,
            "quaternion_minimum_eigengap": self.quaternion_minimum_eigengap,
            "quaternion_normalized_eigengap": self.quaternion_normalized_eigengap,
            "translation_singular_values": self.translation_singular_values,
            "translation_rank": self.translation_rank,
            "translation_condition_number": self.translation_condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "horaud_dornaika_axis_quaternion_linear_translation/v0.1",
            "paper_doi": "10.1177/027836499501400301",
            "paper_equations": [16, 18, 27, 29],
            "equation": "A X = X B",
            "motion_convention": "A and B are corresponding relative SE(3) motions",
            "implementation": "independent NumPy implementation; no external code copied",
        }


class HoraudDornaikaHandEyeSolver:
    """Solve rotation-axis alignment by a quaternion eigenproblem, then translation."""

    def solve(
        self,
        motions: Sequence[HandEyeMotionPair],
        options: HoraudDornaikaHandEyeOptions | None = None,
    ) -> HoraudDornaikaHandEyeResult:
        solver_options = options or HoraudDornaikaHandEyeOptions()
        usable = sorted(
            (
                pair
                for pair in motions
                if pair.weight > 0.0
                and _rotation_angle(pair.motion_a) >= solver_options.min_rotation_rad
                and _rotation_angle(pair.motion_b) >= solver_options.min_rotation_rad
            ),
            key=lambda pair: pair.pair_id,
        )
        train_indices, holdout_indices = split_indices(
            len(usable), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        train_ids = tuple(pair.pair_id for pair in train)
        holdout_ids = tuple(pair.pair_id for pair in holdout)
        if len(train) < solver_options.min_train_motions:
            return _empty("insufficient_motions", train_ids, holdout_ids)

        (
            rotation_quaternion,
            axis_spectrum,
            axis_rank,
            axis_condition,
            objective_eigenvalues,
            eigengap,
            normalized_eigengap,
        ) = _solve_rotation(train, solver_options)
        if rotation_quaternion is None:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                axis_spectrum=axis_spectrum,
                axis_rank=axis_rank,
                axis_condition=axis_condition,
                objective_eigenvalues=objective_eigenvalues,
                eigengap=eigengap,
                normalized_eigengap=normalized_eigengap,
            )
        transform_rotation = SE3((0.0, 0.0, 0.0), rotation_quaternion)
        rotation_matrix = _rotation_matrix(transform_rotation)
        translation, translation_spectrum, translation_rank, translation_condition = (
            _solve_translation(train, rotation_matrix, solver_options)
        )
        if translation is None:
            return _empty(
                "degenerate_translation",
                train_ids,
                holdout_ids,
                axis_spectrum,
                axis_rank,
                axis_condition,
                objective_eigenvalues,
                eigengap,
                normalized_eigengap,
                translation_spectrum,
                translation_rank,
                translation_condition,
            )
        transform = SE3(
            (float(translation[0]), float(translation[1]), float(translation[2])),
            rotation_quaternion,
        )
        train_evaluation = evaluate_hand_eye_motions(train, transform)
        holdout_evaluation = evaluate_hand_eye_motions(holdout, transform)
        return HoraudDornaikaHandEyeResult(
            status="converged",
            reason="axis-quaternion eigenproblem and linear translation system solved",
            transform_x=transform,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            rotation_axis_singular_values=axis_spectrum,
            rotation_axis_rank=axis_rank,
            rotation_axis_condition_number=axis_condition,
            quaternion_objective_eigenvalues=objective_eigenvalues,
            quaternion_minimum_eigengap=eigengap,
            quaternion_normalized_eigengap=normalized_eigengap,
            translation_singular_values=translation_spectrum,
            translation_rank=translation_rank,
            translation_condition_number=translation_condition,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=evaluate_hand_eye_known_bad_probes(
                holdout, transform, holdout_evaluation, solver_options
            ),
        )


def _solve_rotation(
    motions: Sequence[HandEyeMotionPair], options: HoraudDornaikaHandEyeOptions
) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[float, float, float],
    int,
    float,
    tuple[float, float, float, float],
    float,
    float,
]:
    objective: FloatArray = np.zeros((4, 4), dtype=np.float64)
    axis_rows: list[FloatArray] = []
    for pair in motions:
        axis_a = _rotation_axis(pair.motion_a)
        axis_b = _rotation_axis(pair.motion_b)
        root_weight = math.sqrt(pair.weight)
        system = _left_matrix((*axis_a, 0.0)) - _right_matrix((*axis_b, 0.0))
        objective += pair.weight * system.T @ system
        axis_rows.append(root_weight * axis_b)

    axis_values = np.linalg.svd(np.asarray(axis_rows), compute_uv=False)
    padded = np.pad(axis_values, (0, max(0, 3 - len(axis_values))))[:3]
    axis_spectrum = (float(padded[0]), float(padded[1]), float(padded[2]))
    axis_rank = int(np.count_nonzero(padded > options.rank_tolerance * max(padded[0], 1.0e-15)))
    axis_condition = float(padded[0] / padded[axis_rank - 1]) if axis_rank > 0 else math.inf
    eigenvalues, eigenvectors = np.linalg.eigh(objective)
    objective_values = (
        float(eigenvalues[0]),
        float(eigenvalues[1]),
        float(eigenvalues[2]),
        float(eigenvalues[3]),
    )
    eigengap = float(eigenvalues[1] - eigenvalues[0])
    normalized_gap = eigengap / max(float(eigenvalues[-1]), 1.0e-15)
    if axis_rank < 2 or normalized_gap <= options.rank_tolerance:
        return (
            None,
            axis_spectrum,
            axis_rank,
            axis_condition,
            objective_values,
            eigengap,
            normalized_gap,
        )
    quaternion = eigenvectors[:, 0]
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return (
        (
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ),
        axis_spectrum,
        axis_rank,
        axis_condition,
        objective_values,
        eigengap,
        normalized_gap,
    )


def _solve_translation(
    motions: Sequence[HandEyeMotionPair],
    rotation_x: FloatArray,
    options: HoraudDornaikaHandEyeOptions,
) -> tuple[FloatArray | None, tuple[float, float, float], int, float]:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    identity: FloatArray = np.eye(3, dtype=np.float64)
    for pair in motions:
        root_weight = math.sqrt(pair.weight)
        rows.extend(root_weight * (_rotation_matrix(pair.motion_a) - identity))
        value = rotation_x @ np.asarray(pair.motion_b.translation_m) - np.asarray(
            pair.motion_a.translation_m
        )
        rhs.extend(root_weight * value)
    matrix = np.asarray(rows, dtype=np.float64)
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    values = (float(spectrum[0]), float(spectrum[1]), float(spectrum[2]))
    rank = int(np.count_nonzero(spectrum > options.rank_tolerance * spectrum[0]))
    condition = values[0] / values[2] if values[2] > 0.0 else math.inf
    if rank < 3 or condition > options.max_condition_number:
        return None, values, rank, condition
    translation, _residuals, _rank, _singular = np.linalg.lstsq(matrix, np.asarray(rhs), rcond=None)
    return translation, values, rank, condition


def _rotation_axis(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    if w < 0.0:
        x, y, z = -x, -y, -z
    vector: FloatArray = np.asarray((x, y, z), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else np.zeros(3, dtype=np.float64)


def _rotation_angle(transform: SE3) -> float:
    x, y, z, w = transform.rotation_quat_xyzw
    vector_norm = math.sqrt(x * x + y * y + z * z)
    return 2.0 * math.atan2(vector_norm, abs(w))


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _left_matrix(quaternion: Sequence[float]) -> FloatArray:
    x, y, z, w = quaternion
    return np.asarray(
        ((w, -z, y, x), (z, w, -x, y), (-y, x, w, z), (-x, -y, -z, w)),
        dtype=np.float64,
    )


def _right_matrix(quaternion: Sequence[float]) -> FloatArray:
    x, y, z, w = quaternion
    return np.asarray(
        ((w, z, -y, x), (-z, w, x, y), (y, -x, w, z), (-x, -y, -z, w)),
        dtype=np.float64,
    )


def _empty(
    status: HoraudDornaikaStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    axis_spectrum: tuple[float, float, float] | None = None,
    axis_rank: int = 0,
    axis_condition: float | None = None,
    objective_eigenvalues: tuple[float, float, float, float] | None = None,
    eigengap: float | None = None,
    normalized_eigengap: float | None = None,
    translation_spectrum: tuple[float, float, float] | None = None,
    translation_rank: int = 0,
    translation_condition: float | None = None,
) -> HoraudDornaikaHandEyeResult:
    reasons = {
        "insufficient_motions": "too few rotation-excited motion pairs",
        "degenerate_rotation": "rotation axes do not constrain a unique quaternion minimum",
        "degenerate_translation": "motion rotations do not constrain translation",
        "converged": "",
    }
    return HoraudDornaikaHandEyeResult(
        status=status,
        reason=reasons[status],
        transform_x=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        rotation_axis_singular_values=axis_spectrum,
        rotation_axis_rank=axis_rank,
        rotation_axis_condition_number=axis_condition,
        quaternion_objective_eigenvalues=objective_eigenvalues,
        quaternion_minimum_eigengap=eigengap,
        quaternion_normalized_eigengap=normalized_eigengap,
        translation_singular_values=translation_spectrum,
        translation_rank=translation_rank,
        translation_condition_number=translation_condition,
        train_evaluation=HandEyeEvaluation(None, None),
        holdout_evaluation=HandEyeEvaluation(None, None),
        probes=(),
    )
