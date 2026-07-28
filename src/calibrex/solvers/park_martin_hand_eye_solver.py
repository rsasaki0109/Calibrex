"""Park-Martin style motion hand-eye calibration for ``A X = X B``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.evaluation.holdout import split_indices

FloatArray: TypeAlias = NDArray[np.float64]
HandEyeStatus = Literal[
    "converged", "insufficient_motions", "degenerate_rotation", "degenerate_translation"
]


@dataclass(frozen=True)
class HandEyeMotionPair:
    """Synchronized relative motions satisfying ``motion_a X = X motion_b``."""

    pair_id: str
    motion_a: SE3
    motion_b: SE3
    weight: float = 1.0


@dataclass(frozen=True)
class ParkMartinHandEyeOptions:
    min_train_motions: int = 3
    min_rotation_rad: float = math.radians(1.0)
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-7
    max_condition_number: float = 1.0e6


@dataclass(frozen=True)
class HandEyeEvaluation:
    rotation_closure_rmse_deg: float | None
    translation_closure_rmse_m: float | None


@dataclass(frozen=True)
class ParkMartinHandEyeResult:
    status: HandEyeStatus
    reason: str
    transform_x: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    rotation_axis_singular_values: tuple[float, float, float] | None
    translation_singular_values: tuple[float, float, float] | None
    rotation_axis_rank: int
    translation_rank: int
    translation_condition_number: float | None
    train_evaluation: HandEyeEvaluation
    holdout_evaluation: HandEyeEvaluation

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly calibration output with provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "rotation_axis_singular_values": self.rotation_axis_singular_values,
            "translation_singular_values": self.translation_singular_values,
            "rotation_axis_rank": self.rotation_axis_rank,
            "translation_rank": self.translation_rank,
            "translation_condition_number": self.translation_condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "method": "park_martin_log_rotation_linear_translation/v0.1",
            "paper_doi": "10.1109/70.326576",
            "equation": "A X = X B",
            "motion_convention": "A and B are corresponding relative SE(3) motions",
        }


class ParkMartinHandEyeSolver:
    """Solve hand-eye rotation in Lie algebra, then translation linearly."""

    def solve(
        self,
        motions: Sequence[HandEyeMotionPair],
        options: ParkMartinHandEyeOptions | None = None,
    ) -> ParkMartinHandEyeResult:
        solver_options = options or ParkMartinHandEyeOptions()
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

        rotation, rotation_spectrum, rotation_rank = _solve_rotation(train, solver_options)
        if rotation is None or rotation_rank < 2:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                rotation_spectrum=rotation_spectrum,
                rotation_rank=rotation_rank,
            )
        translation, translation_spectrum, translation_rank, condition = _solve_translation(
            train, rotation, solver_options
        )
        if translation is None or translation_rank < 3:
            return _empty(
                "degenerate_translation",
                train_ids,
                holdout_ids,
                rotation_spectrum,
                translation_spectrum,
                rotation_rank,
                translation_rank,
                condition,
            )
        transform = SE3(
            (float(translation[0]), float(translation[1]), float(translation[2])),
            quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
        )
        return ParkMartinHandEyeResult(
            status="converged",
            reason="rotation and translation hand-eye systems solved",
            transform_x=transform,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            rotation_axis_singular_values=rotation_spectrum,
            translation_singular_values=translation_spectrum,
            rotation_axis_rank=rotation_rank,
            translation_rank=translation_rank,
            translation_condition_number=condition,
            train_evaluation=evaluate_hand_eye_motions(train, transform),
            holdout_evaluation=evaluate_hand_eye_motions(holdout, transform),
        )


def evaluate_hand_eye_motions(
    motions: Sequence[HandEyeMotionPair], transform_x: SE3
) -> HandEyeEvaluation:
    """Evaluate ``A X`` versus ``X B`` closure without refitting."""

    if not motions:
        return HandEyeEvaluation(None, None)
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for pair in motions:
        left = pair.motion_a.compose(transform_x)
        right = transform_x.compose(pair.motion_b)
        delta = left.inverse().compose(right)
        rotation_errors.append(_rotation_angle(delta))
        translation_errors.append(float(np.linalg.norm(delta.translation_m)))
    return HandEyeEvaluation(
        rotation_closure_rmse_deg=math.degrees(_rmse(rotation_errors)),
        translation_closure_rmse_m=_rmse(translation_errors),
    )


def _solve_rotation(
    motions: Sequence[HandEyeMotionPair], options: ParkMartinHandEyeOptions
) -> tuple[FloatArray | None, tuple[float, float, float], int]:
    covariance: FloatArray = np.zeros((3, 3), dtype=np.float64)
    beta_rows: list[FloatArray] = []
    for pair in motions:
        alpha = _rotation_vector(pair.motion_a)
        beta = _rotation_vector(pair.motion_b)
        covariance += pair.weight * np.outer(alpha, beta)
        beta_rows.append(math.sqrt(pair.weight) * beta)
    spectrum = np.linalg.svd(np.asarray(beta_rows), compute_uv=False)
    padded = np.pad(spectrum, (0, max(0, 3 - len(spectrum))))[:3]
    values = (float(padded[0]), float(padded[1]), float(padded[2]))
    rank = int(np.count_nonzero(padded > options.rank_tolerance * padded[0]))
    if rank < 2:
        return None, values, rank
    left, _values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    return left @ correction @ right_t, values, rank


def _solve_translation(
    motions: Sequence[HandEyeMotionPair],
    rotation_x: FloatArray,
    options: ParkMartinHandEyeOptions,
) -> tuple[FloatArray | None, tuple[float, float, float], int, float]:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    identity = np.eye(3)
    for pair in motions:
        root_weight = math.sqrt(pair.weight)
        rotation_a = _rotation_matrix(pair.motion_a)
        rows.extend(root_weight * (rotation_a - identity))
        value = rotation_x @ np.asarray(pair.motion_b.translation_m) - np.asarray(
            pair.motion_a.translation_m
        )
        rhs.extend(root_weight * value)
    matrix = np.asarray(rows)
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    values = (float(spectrum[0]), float(spectrum[1]), float(spectrum[2]))
    rank = int(np.count_nonzero(spectrum > options.rank_tolerance * spectrum[0]))
    condition = values[0] / values[2] if values[2] > 0.0 else math.inf
    if rank < 3 or condition > options.max_condition_number:
        return None, values, rank, condition
    translation, _residuals, _rank, _singular = np.linalg.lstsq(matrix, np.asarray(rhs), rcond=None)
    return translation, values, rank, condition


def _rotation_vector(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector = np.asarray((x, y, z), dtype=np.float64)
    sine_half = float(np.linalg.norm(vector))
    if sine_half <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sine_half, w)
    return angle * vector / sine_half


def _rotation_angle(transform: SE3) -> float:
    return float(np.linalg.norm(_rotation_vector(transform)))


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


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _empty(
    status: HandEyeStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    rotation_spectrum: tuple[float, float, float] | None = None,
    translation_spectrum: tuple[float, float, float] | None = None,
    rotation_rank: int = 0,
    translation_rank: int = 0,
    condition: float | None = None,
) -> ParkMartinHandEyeResult:
    reasons = {
        "insufficient_motions": "too few rotation-excited motion pairs",
        "degenerate_rotation": "motion rotation axes do not span two directions",
        "degenerate_translation": "motion rotations do not constrain translation",
        "converged": "",
    }
    return ParkMartinHandEyeResult(
        status,
        reasons[status],
        None,
        train_ids,
        holdout_ids,
        rotation_spectrum,
        translation_spectrum,
        rotation_rank,
        translation_rank,
        condition,
        HandEyeEvaluation(None, None),
        HandEyeEvaluation(None, None),
    )
