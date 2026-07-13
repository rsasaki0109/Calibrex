"""Chou-Kamel quaternion hand-eye calibration for ``A X = X B``."""

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
ChouKamelStatus = Literal[
    "converged", "insufficient_motions", "degenerate_rotation", "degenerate_translation"
]


@dataclass(frozen=True)
class ChouKamelHandEyeOptions:
    """Numerical and evidence policy for the Chou-Kamel closed form."""

    min_train_motions: int = 2
    min_rotation_rad: float = math.radians(1.0)
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-7
    minimum_rotation_nullspace_gap: float = 1.0e-3
    max_translation_condition_number: float = 1.0e6
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_rotation_margin_deg: float = 0.1
    known_bad_translation_margin_m: float = 0.005


@dataclass(frozen=True)
class ChouKamelHandEyeResult:
    """Estimate with split, rank, nullspace, and falsification evidence."""

    status: ChouKamelStatus
    reason: str
    transform_x: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    rotation_singular_values: tuple[float, float, float, float] | None
    rotation_rank: int
    rotation_observable_condition_number: float | None
    rotation_nullspace_gap: float | None
    rotation_nullspace_residual: float | None
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
            "rotation_singular_values": self.rotation_singular_values,
            "rotation_rank": self.rotation_rank,
            "rotation_observable_condition_number": self.rotation_observable_condition_number,
            "rotation_nullspace_gap": self.rotation_nullspace_gap,
            "rotation_nullspace_residual": self.rotation_nullspace_residual,
            "translation_singular_values": self.translation_singular_values,
            "translation_rank": self.translation_rank,
            "translation_condition_number": self.translation_condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "chou_kamel_normalized_quaternion_generalized_inverse/v0.1",
            "paper_doi": "10.1177/027836499101000305",
            "paper_title": (
                "Finding the Position and Orientation of a Sensor on a Robot "
                "Manipulator Using Quaternions"
            ),
            "equation": "A X = X B",
            "motion_convention": "A and B are corresponding relative SE(3) motions",
            "implementation": "independent NumPy implementation; no external code copied",
        }


class ChouKamelHandEyeSolver:
    """Solve the normalized-quaternion nullspace, then conditional translation."""

    def solve(
        self,
        motions: Sequence[HandEyeMotionPair],
        options: ChouKamelHandEyeOptions | None = None,
    ) -> ChouKamelHandEyeResult:
        solver_options = options or ChouKamelHandEyeOptions()
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

        quaternion, spectrum, rank, condition, gap, residual = _solve_rotation(
            train, solver_options
        )
        if quaternion is None:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                spectrum,
                rank,
                condition,
                gap,
                residual,
            )
        rotation_x = _rotation_matrix(SE3((0.0, 0.0, 0.0), quaternion))
        translation, translation_spectrum, translation_rank, translation_condition = (
            _solve_translation(train, rotation_x, solver_options)
        )
        if translation is None:
            return _empty(
                "degenerate_translation",
                train_ids,
                holdout_ids,
                spectrum,
                rank,
                condition,
                gap,
                residual,
                translation_spectrum,
                translation_rank,
                translation_condition,
            )
        transform = SE3(
            (float(translation[0]), float(translation[1]), float(translation[2])),
            quaternion,
        )
        train_evaluation = evaluate_hand_eye_motions(train, transform)
        holdout_evaluation = evaluate_hand_eye_motions(holdout, transform)
        return ChouKamelHandEyeResult(
            status="converged",
            reason="normalized-quaternion and conditional translation systems solved",
            transform_x=transform,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            rotation_singular_values=spectrum,
            rotation_rank=rank,
            rotation_observable_condition_number=condition,
            rotation_nullspace_gap=gap,
            rotation_nullspace_residual=residual,
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
    motions: Sequence[HandEyeMotionPair], options: ChouKamelHandEyeOptions
) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float],
    int,
    float,
    float,
    float,
]:
    rows: list[FloatArray] = []
    for pair in motions:
        quaternion_a = _canonical_quaternion(pair.motion_a)
        quaternion_b = _canonical_quaternion(pair.motion_b)
        rows.extend(
            math.sqrt(pair.weight)
            * (_left_matrix(quaternion_a) - _right_matrix(quaternion_b))
        )
    matrix = np.asarray(rows, dtype=np.float64)
    _left, singular_values, right_transpose = np.linalg.svd(matrix, full_matrices=False)
    values: tuple[float, float, float, float] = (
        float(singular_values[0]),
        float(singular_values[1]),
        float(singular_values[2]),
        float(singular_values[3]),
    )
    threshold = options.rank_tolerance * max(float(singular_values[0]), 1.0e-15)
    rank = int(np.count_nonzero(singular_values[:3] > threshold))
    condition = (
        float(singular_values[0] / singular_values[2])
        if singular_values[2] > 0.0
        else math.inf
    )
    gap = (
        float(
            max(singular_values[2] - singular_values[3], 0.0)
            / max(singular_values[2], 1.0e-15)
        )
        if singular_values[2] > threshold
        else 0.0
    )
    quaternion = right_transpose[-1]
    quaternion /= np.linalg.norm(quaternion)
    residual = float(np.linalg.norm(matrix @ quaternion) / math.sqrt(len(matrix)))
    if rank != 3 or gap < options.minimum_rotation_nullspace_gap:
        return None, values, rank, condition, gap, residual
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    result = tuple(float(value) for value in quaternion)
    return (result[0], result[1], result[2], result[3]), values, rank, condition, gap, residual


def _solve_translation(
    motions: Sequence[HandEyeMotionPair],
    rotation_x: FloatArray,
    options: ChouKamelHandEyeOptions,
) -> tuple[FloatArray | None, tuple[float, float, float], int, float]:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    identity: FloatArray = np.eye(3, dtype=np.float64)
    for pair in motions:
        root_weight = math.sqrt(pair.weight)
        rows.extend(root_weight * (_rotation_matrix(pair.motion_a) - identity))
        rhs.extend(
            root_weight
            * (
                rotation_x @ np.asarray(pair.motion_b.translation_m)
                - np.asarray(pair.motion_a.translation_m)
            )
        )
    matrix = np.asarray(rows, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    values: tuple[float, float, float] = (
        float(singular_values[0]),
        float(singular_values[1]),
        float(singular_values[2]),
    )
    threshold = options.rank_tolerance * max(float(singular_values[0]), 1.0e-15)
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = values[0] / values[2] if values[2] > 0.0 else math.inf
    if rank != 3 or condition > options.max_translation_condition_number:
        return None, (values[0], values[1], values[2]), rank, condition
    translation, _residuals, _rank, _singular = np.linalg.lstsq(
        matrix, np.asarray(rhs), rcond=None
    )
    return translation, (values[0], values[1], values[2]), rank, condition


def _canonical_quaternion(transform: SE3) -> tuple[float, float, float, float]:
    values: FloatArray = np.asarray(transform.rotation_quat_xyzw, dtype=np.float64)
    for value in reversed(values):
        if abs(float(value)) > 1.0e-15:
            if value < 0.0:
                values *= -1.0
            break
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _rotation_angle(transform: SE3) -> float:
    x, y, z, w = transform.rotation_quat_xyzw
    return 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(w))


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
    status: ChouKamelStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    rotation_spectrum: tuple[float, float, float, float] | None = None,
    rotation_rank: int = 0,
    rotation_condition: float | None = None,
    rotation_gap: float | None = None,
    rotation_residual: float | None = None,
    translation_spectrum: tuple[float, float, float] | None = None,
    translation_rank: int = 0,
    translation_condition: float | None = None,
) -> ChouKamelHandEyeResult:
    reasons = {
        "insufficient_motions": "too few rotation-excited motion pairs",
        "degenerate_rotation": "quaternion system does not have one observable nullspace",
        "degenerate_translation": "motion rotations do not constrain translation",
        "converged": "",
    }
    return ChouKamelHandEyeResult(
        status=status,
        reason=reasons[status],
        transform_x=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        rotation_singular_values=rotation_spectrum,
        rotation_rank=rotation_rank,
        rotation_observable_condition_number=rotation_condition,
        rotation_nullspace_gap=rotation_gap,
        rotation_nullspace_residual=rotation_residual,
        translation_singular_values=translation_spectrum,
        translation_rank=translation_rank,
        translation_condition_number=translation_condition,
        train_evaluation=HandEyeEvaluation(None, None),
        holdout_evaluation=HandEyeEvaluation(None, None),
        probes=(),
    )
