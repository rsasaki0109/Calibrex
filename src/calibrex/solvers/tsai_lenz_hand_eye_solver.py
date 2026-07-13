"""Independent Tsai-Lenz hand-eye calibration for ``A X = X B``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeEvaluation,
    HandEyeMotionPair,
    evaluate_hand_eye_motions,
)

FloatArray: TypeAlias = NDArray[np.float64]
TsaiLenzStatus = Literal[
    "converged", "insufficient_motions", "degenerate_rotation", "degenerate_translation"
]
HandEyeDof = Literal["x", "y", "z", "roll", "pitch", "yaw"]


@dataclass(frozen=True)
class TsaiLenzHandEyeOptions:
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
class HandEyeProbeResult:
    dof: HandEyeDof
    amount: float
    unit: Literal["m", "deg"]
    rotation_closure_rmse_deg: float | None
    translation_closure_rmse_m: float | None
    rotation_delta_deg: float | None
    translation_delta_m: float | None
    detectable: bool | None


class HandEyeProbeOptions(Protocol):
    """Structural options required by the shared falsification protocol."""

    @property
    def known_bad_rotation_deg(self) -> float: ...

    @property
    def known_bad_translation_m(self) -> float: ...

    @property
    def known_bad_rotation_margin_deg(self) -> float: ...

    @property
    def known_bad_translation_margin_m(self) -> float: ...


@dataclass(frozen=True)
class TsaiLenzHandEyeResult:
    status: TsaiLenzStatus
    reason: str
    transform_x: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    rotation_singular_values: tuple[float, float, float] | None
    translation_singular_values: tuple[float, float, float] | None
    rotation_rank: int
    translation_rank: int
    rotation_condition_number: float | None
    translation_condition_number: float | None
    train_evaluation: HandEyeEvaluation
    holdout_evaluation: HandEyeEvaluation
    probes: tuple[HandEyeProbeResult, ...]

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly solver output and paper provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "rotation_singular_values": self.rotation_singular_values,
            "translation_singular_values": self.translation_singular_values,
            "rotation_rank": self.rotation_rank,
            "translation_rank": self.translation_rank,
            "rotation_condition_number": self.rotation_condition_number,
            "translation_condition_number": self.translation_condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "tsai_lenz_modified_rodrigues_linear_translation/v0.1",
            "paper_doi": "10.1109/70.34770",
            "equation": "A X = X B",
            "motion_convention": "A and B are corresponding relative SE(3) motions",
            "implementation": "independent NumPy implementation; no external code copied",
        }


class TsaiLenzHandEyeSolver:
    """Solve Tsai-Lenz rotation and translation with explicit rank diagnostics."""

    def solve(
        self,
        motions: Sequence[HandEyeMotionPair],
        options: TsaiLenzHandEyeOptions | None = None,
    ) -> TsaiLenzHandEyeResult:
        solver_options = options or TsaiLenzHandEyeOptions()
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

        rotation, rotation_spectrum, rotation_rank, rotation_condition = _solve_rotation(
            train, solver_options
        )
        if rotation is None:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                rotation_spectrum=rotation_spectrum,
                rotation_rank=rotation_rank,
                rotation_condition=rotation_condition,
            )
        translation, translation_spectrum, translation_rank, translation_condition = (
            _solve_translation(train, rotation, solver_options)
        )
        if translation is None:
            return _empty(
                "degenerate_translation",
                train_ids,
                holdout_ids,
                rotation_spectrum,
                translation_spectrum,
                rotation_rank,
                translation_rank,
                rotation_condition,
                translation_condition,
            )
        transform = SE3(
            (float(translation[0]), float(translation[1]), float(translation[2])),
            _quaternion_from_matrix(rotation),
        )
        train_evaluation = evaluate_hand_eye_motions(train, transform)
        holdout_evaluation = evaluate_hand_eye_motions(holdout, transform)
        return TsaiLenzHandEyeResult(
            status="converged",
            reason="Tsai-Lenz rotation and translation systems solved",
            transform_x=transform,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            rotation_singular_values=rotation_spectrum,
            translation_singular_values=translation_spectrum,
            rotation_rank=rotation_rank,
            translation_rank=translation_rank,
            rotation_condition_number=rotation_condition,
            translation_condition_number=translation_condition,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=evaluate_hand_eye_known_bad_probes(
                holdout, transform, holdout_evaluation, solver_options
            ),
        )


def _solve_rotation(
    motions: Sequence[HandEyeMotionPair], options: TsaiLenzHandEyeOptions
) -> tuple[FloatArray | None, tuple[float, float, float], int, float]:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    for pair in motions:
        root_weight = math.sqrt(pair.weight)
        vector_a = np.asarray(_canonical_quaternion(pair.motion_a)[:3])
        vector_b = np.asarray(_canonical_quaternion(pair.motion_b)[:3])
        rows.extend(root_weight * _skew(vector_a + vector_b))
        rhs.extend(root_weight * (vector_b - vector_a))
    matrix = np.asarray(rows)
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    values, rank, condition = _spectrum_diagnostics(spectrum, options.rank_tolerance)
    if rank < 3 or condition > options.max_condition_number:
        return None, values, rank, condition
    parameter, _residuals, _rank, _singular = np.linalg.lstsq(
        matrix, np.asarray(rhs), rcond=None
    )
    scale = math.sqrt(1.0 + float(parameter @ parameter))
    quaternion = (
        float(parameter[0] / scale),
        float(parameter[1] / scale),
        float(parameter[2] / scale),
        1.0 / scale,
    )
    return _matrix_from_quaternion(quaternion), values, rank, condition


def _solve_translation(
    motions: Sequence[HandEyeMotionPair],
    rotation_x: FloatArray,
    options: TsaiLenzHandEyeOptions,
) -> tuple[FloatArray | None, tuple[float, float, float], int, float]:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    identity = np.eye(3)
    for pair in motions:
        root_weight = math.sqrt(pair.weight)
        rows.extend(root_weight * (_rotation_matrix(pair.motion_a) - identity))
        value = rotation_x @ np.asarray(pair.motion_b.translation_m) - np.asarray(
            pair.motion_a.translation_m
        )
        rhs.extend(root_weight * value)
    matrix = np.asarray(rows)
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    values, rank, condition = _spectrum_diagnostics(spectrum, options.rank_tolerance)
    if rank < 3 or condition > options.max_condition_number:
        return None, values, rank, condition
    translation, _residuals, _rank, _singular = np.linalg.lstsq(
        matrix, np.asarray(rhs), rcond=None
    )
    return translation, values, rank, condition


def evaluate_hand_eye_known_bad_probes(
    holdout: Sequence[HandEyeMotionPair],
    transform: SE3,
    baseline: HandEyeEvaluation,
    options: HandEyeProbeOptions,
) -> tuple[HandEyeProbeResult, ...]:
    axes: tuple[Vector3, Vector3, Vector3] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    probes: list[HandEyeProbeResult] = []
    translation_dofs: tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")
    for index, translation_dof in enumerate(translation_dofs):
        axis = axes[index]
        for sign in (-1.0, 1.0):
            amount = sign * options.known_bad_translation_m
            delta = SE3(
                (amount * axis[0], amount * axis[1], amount * axis[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
            probes.append(
                _probe(
                    translation_dof,
                    amount,
                    "m",
                    holdout,
                    delta.compose(transform),
                    baseline,
                    options,
                )
            )
    rotation_dofs: tuple[Literal["roll", "pitch", "yaw"], ...] = (
        "roll",
        "pitch",
        "yaw",
    )
    for index, rotation_dof in enumerate(rotation_dofs):
        axis = axes[index]
        for sign in (-1.0, 1.0):
            amount = sign * options.known_bad_rotation_deg
            half = math.radians(amount) / 2.0
            delta = SE3(
                (0.0, 0.0, 0.0),
                (
                    axis[0] * math.sin(half),
                    axis[1] * math.sin(half),
                    axis[2] * math.sin(half),
                    math.cos(half),
                ),
            )
            probes.append(
                _probe(
                    rotation_dof,
                    amount,
                    "deg",
                    holdout,
                    delta.compose(transform),
                    baseline,
                    options,
                )
            )
    return tuple(probes)


def _probe(
    dof: HandEyeDof,
    amount: float,
    unit: Literal["m", "deg"],
    holdout: Sequence[HandEyeMotionPair],
    candidate: SE3,
    baseline: HandEyeEvaluation,
    options: HandEyeProbeOptions,
) -> HandEyeProbeResult:
    evaluated = evaluate_hand_eye_motions(holdout, candidate)
    rotation_delta = _difference(
        evaluated.rotation_closure_rmse_deg, baseline.rotation_closure_rmse_deg
    )
    translation_delta = _difference(
        evaluated.translation_closure_rmse_m, baseline.translation_closure_rmse_m
    )
    detectable = (
        rotation_delta > options.known_bad_rotation_margin_deg
        or translation_delta > options.known_bad_translation_margin_m
        if rotation_delta is not None and translation_delta is not None
        else None
    )
    return HandEyeProbeResult(
        dof=dof,
        amount=amount,
        unit=unit,
        rotation_closure_rmse_deg=evaluated.rotation_closure_rmse_deg,
        translation_closure_rmse_m=evaluated.translation_closure_rmse_m,
        rotation_delta_deg=rotation_delta,
        translation_delta_m=translation_delta,
        detectable=detectable,
    )


def _difference(value: float | None, baseline: float | None) -> float | None:
    return value - baseline if value is not None and baseline is not None else None


def _canonical_quaternion(transform: SE3) -> tuple[float, float, float, float]:
    x, y, z, w = transform.rotation_quat_xyzw
    return (-x, -y, -z, -w) if w < 0.0 else (x, y, z, w)


def _rotation_angle(transform: SE3) -> float:
    _x, _y, _z, w = _canonical_quaternion(transform)
    return 2.0 * math.acos(min(1.0, max(-1.0, w)))


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _rotation_matrix(transform: SE3) -> FloatArray:
    return _matrix_from_quaternion(transform.rotation_quat_xyzw)


def _matrix_from_quaternion(quaternion: tuple[float, float, float, float]) -> FloatArray:
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_from_matrix(rotation: FloatArray) -> tuple[float, float, float, float]:
    from calibrex.core.geometry import quaternion_xyzw_from_rotation_matrix

    return quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1))


def _spectrum_diagnostics(
    spectrum: FloatArray, rank_tolerance: float
) -> tuple[tuple[float, float, float], int, float]:
    values = (float(spectrum[0]), float(spectrum[1]), float(spectrum[2]))
    rank = int(np.count_nonzero(spectrum > rank_tolerance * spectrum[0]))
    condition = values[0] / values[2] if values[2] > 0.0 else math.inf
    return values, rank, condition


def _empty(
    status: TsaiLenzStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    rotation_spectrum: tuple[float, float, float] | None = None,
    translation_spectrum: tuple[float, float, float] | None = None,
    rotation_rank: int = 0,
    translation_rank: int = 0,
    rotation_condition: float | None = None,
    translation_condition: float | None = None,
) -> TsaiLenzHandEyeResult:
    reasons = {
        "insufficient_motions": "too few rotation-excited motion pairs",
        "degenerate_rotation": "Tsai-Lenz rotation system is rank deficient",
        "degenerate_translation": "motion rotations do not constrain translation",
        "converged": "",
    }
    return TsaiLenzHandEyeResult(
        status=status,
        reason=reasons[status],
        transform_x=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        rotation_singular_values=rotation_spectrum,
        translation_singular_values=translation_spectrum,
        rotation_rank=rotation_rank,
        translation_rank=translation_rank,
        rotation_condition_number=rotation_condition,
        translation_condition_number=translation_condition,
        train_evaluation=HandEyeEvaluation(None, None),
        holdout_evaluation=HandEyeEvaluation(None, None),
        probes=(),
    )
