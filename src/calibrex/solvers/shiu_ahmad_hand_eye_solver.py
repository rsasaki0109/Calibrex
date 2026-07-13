"""Shiu-Ahmad two-motion hand-eye calibration for ``A X = X B``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
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
ShiuAhmadStatus = Literal[
    "converged",
    "insufficient_motions",
    "no_unique_motion_pair",
    "degenerate_rotation",
    "degenerate_translation",
]


@dataclass(frozen=True)
class ShiuAhmadHandEyeOptions:
    """Numerical policy around the paper's exact two-motion construction."""

    min_train_motions: int = 3
    min_rotation_rad: float = math.radians(1.0)
    max_rotation_rad: float = math.pi - math.radians(1.0)
    holdout_ratio: float = 0.2
    split_seed: int = 0
    minimum_axis_separation_sine: float = 0.1
    rank_tolerance: float = 1.0e-7
    max_rotation_condition_number: float = 1.0e8
    max_translation_condition_number: float = 1.0e8
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_rotation_margin_deg: float = 0.1
    known_bad_translation_margin_m: float = 0.005


@dataclass(frozen=True)
class ShiuAhmadHandEyeResult:
    status: ShiuAhmadStatus
    reason: str
    transform_x: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    fit_pair_ids: tuple[str, ...]
    selected_axis_separation_sine: float | None
    selected_rotation_angle_mismatch_deg_max: float | None
    rotation_singular_values: tuple[float, float, float, float] | None
    rotation_rank: int
    rotation_condition_number: float | None
    rotation_linear_residual_rmse: float | None
    beta_unit_circle_error_max: float | None
    two_rotation_solution_disagreement_deg: float | None
    translation_singular_values: tuple[float, float, float] | None
    translation_rank: int
    translation_condition_number: float | None
    train_evaluation: HandEyeEvaluation
    holdout_evaluation: HandEyeEvaluation
    probes: tuple[HandEyeProbeResult, ...]
    options: ShiuAhmadHandEyeOptions

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly result data and complete method provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "fit_pair_ids": list(self.fit_pair_ids),
            "selected_axis_separation_sine": self.selected_axis_separation_sine,
            "selected_rotation_angle_mismatch_deg_max": (
                self.selected_rotation_angle_mismatch_deg_max
            ),
            "rotation_singular_values": self.rotation_singular_values,
            "rotation_rank": self.rotation_rank,
            "rotation_condition_number": self.rotation_condition_number,
            "rotation_linear_residual_rmse": self.rotation_linear_residual_rmse,
            "beta_unit_circle_error_max": self.beta_unit_circle_error_max,
            "two_rotation_solution_disagreement_deg": (
                self.two_rotation_solution_disagreement_deg
            ),
            "translation_singular_values": self.translation_singular_values,
            "translation_rank": self.translation_rank,
            "translation_condition_number": self.translation_condition_number,
            "train_evaluation": asdict(self.train_evaluation),
            "holdout_evaluation": asdict(self.holdout_evaluation),
            "known_bad_probes": [asdict(probe) for probe in self.probes],
            "resolved_options": asdict(self.options),
            "method": "shiu_ahmad_two_motion_geometric_hand_eye/v0.1",
            "primary_paper": {
                "title": (
                    "Calibration of Wrist-Mounted Robotic Sensors by Solving "
                    "Homogeneous Transform Equations of the Form AX=XB"
                ),
                "authors": ["Y. C. Shiu", "S. Ahmad"],
                "year": 1989,
                "doi": "10.1109/70.88014",
                "equations": [37, 38, 39, 42, 44, 45, 46, 47],
                "source_pdf": "https://people.csail.mit.edu/tieu/stuff/Shiu1989.pdf",
            },
            "equation": "A X = X B",
            "motion_convention": "A and B are corresponding relative SE(3) motions",
            "fit_policy": (
                "select the maximum-axis-separation pair using training data only; "
                "fit exactly two equations as required by paper Section IV"
            ),
            "evaluation_policy": (
                "evaluate all common train motions and untouched common holdout motions"
            ),
            "fidelity_boundary": (
                "the paper's two-motion estimator is preserved; deterministic pair "
                "selection is a Calibrex wrapper for datasets containing more than two motions"
            ),
            "implementation": "independent NumPy implementation; no external code copied",
            "external_code_executed": False,
        }


class ShiuAhmadHandEyeSolver:
    """Solve the paper's two simultaneous equations with explicit uniqueness gates."""

    def solve(
        self,
        motions: Sequence[HandEyeMotionPair],
        options: ShiuAhmadHandEyeOptions | None = None,
    ) -> ShiuAhmadHandEyeResult:
        opts = options or ShiuAhmadHandEyeOptions()
        _validate_options(opts)
        usable = sorted(
            (
                pair
                for pair in motions
                if pair.weight > 0.0
                and opts.min_rotation_rad <= _rotation_angle(pair.motion_a)
                <= opts.max_rotation_rad
                and opts.min_rotation_rad <= _rotation_angle(pair.motion_b)
                <= opts.max_rotation_rad
            ),
            key=lambda pair: pair.pair_id,
        )
        train_indices, holdout_indices = split_indices(
            len(usable), opts.holdout_ratio, seed=opts.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        train_ids = tuple(pair.pair_id for pair in train)
        holdout_ids = tuple(pair.pair_id for pair in holdout)
        if len(train) < opts.min_train_motions:
            return _empty("insufficient_motions", train_ids, holdout_ids, opts)

        selected, separation = _select_motion_pair(train)
        if selected is None or separation < opts.minimum_axis_separation_sine:
            return _empty(
                "no_unique_motion_pair",
                train_ids,
                holdout_ids,
                opts,
                axis_separation=separation,
            )
        first, second = selected
        fit_ids = (first.pair_id, second.pair_id)
        angle_mismatch = math.degrees(
            max(
                abs(_rotation_angle(pair.motion_a) - _rotation_angle(pair.motion_b))
                for pair in selected
            )
        )

        rotation_solution = _solve_rotation(first, second, opts)
        if rotation_solution.rotation is None:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                opts,
                fit_ids=fit_ids,
                axis_separation=separation,
                angle_mismatch_deg=angle_mismatch,
                rotation=rotation_solution,
            )
        translation_solution = _solve_translation(
            selected, rotation_solution.rotation, opts
        )
        if translation_solution.translation is None:
            return _empty(
                "degenerate_translation",
                train_ids,
                holdout_ids,
                opts,
                fit_ids=fit_ids,
                axis_separation=separation,
                angle_mismatch_deg=angle_mismatch,
                rotation=rotation_solution,
                translation=translation_solution,
            )
        transform = SE3(
            (
                float(translation_solution.translation[0]),
                float(translation_solution.translation[1]),
                float(translation_solution.translation[2]),
            ),
            quaternion_xyzw_from_rotation_matrix(rotation_solution.rotation.reshape(-1)),
        )
        train_evaluation = evaluate_hand_eye_motions(train, transform)
        holdout_evaluation = evaluate_hand_eye_motions(holdout, transform)
        return ShiuAhmadHandEyeResult(
            status="converged",
            reason="Shiu-Ahmad Section IV two-motion systems solved",
            transform_x=transform,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            fit_pair_ids=fit_ids,
            selected_axis_separation_sine=separation,
            selected_rotation_angle_mismatch_deg_max=angle_mismatch,
            rotation_singular_values=rotation_solution.spectrum,
            rotation_rank=rotation_solution.rank,
            rotation_condition_number=rotation_solution.condition,
            rotation_linear_residual_rmse=rotation_solution.residual_rmse,
            beta_unit_circle_error_max=rotation_solution.unit_circle_error,
            two_rotation_solution_disagreement_deg=rotation_solution.disagreement_deg,
            translation_singular_values=translation_solution.spectrum,
            translation_rank=translation_solution.rank,
            translation_condition_number=translation_solution.condition,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=evaluate_hand_eye_known_bad_probes(
                holdout, transform, holdout_evaluation, opts
            ),
            options=opts,
        )


@dataclass(frozen=True)
class _RotationSolution:
    rotation: FloatArray | None
    spectrum: tuple[float, float, float, float] | None
    rank: int = 0
    condition: float | None = None
    residual_rmse: float | None = None
    unit_circle_error: float | None = None
    disagreement_deg: float | None = None


@dataclass(frozen=True)
class _TranslationSolution:
    translation: FloatArray | None
    spectrum: tuple[float, float, float] | None
    rank: int = 0
    condition: float | None = None


def _select_motion_pair(
    motions: Sequence[HandEyeMotionPair],
) -> tuple[tuple[HandEyeMotionPair, HandEyeMotionPair] | None, float]:
    best: tuple[HandEyeMotionPair, HandEyeMotionPair] | None = None
    best_separation = -1.0
    for first, second in combinations(motions, 2):
        separation = float(
            np.linalg.norm(
                np.cross(_rotation_axis(first.motion_a), _rotation_axis(second.motion_a))
            )
        )
        if separation > best_separation:
            best = (first, second)
            best_separation = separation
    return best, max(best_separation, 0.0)


def _solve_rotation(
    first: HandEyeMotionPair,
    second: HandEyeMotionPair,
    options: ShiuAhmadHandEyeOptions,
) -> _RotationSolution:
    axis_a1 = _rotation_axis(first.motion_a)
    axis_a2 = _rotation_axis(second.motion_a)
    particular_1 = _rotation_between(_rotation_axis(first.motion_b), axis_a1)
    particular_2 = _rotation_between(_rotation_axis(second.motion_b), axis_a2)
    identity: FloatArray = np.eye(3, dtype=np.float64)
    skew_1 = _skew(axis_a1)
    skew_2 = _skew(axis_a2)
    outer_1 = np.outer(axis_a1, axis_a1)
    outer_2 = np.outer(axis_a2, axis_a2)
    columns = (
        (identity - outer_1) @ particular_1,
        skew_1 @ particular_1,
        -(identity - outer_2) @ particular_2,
        -skew_2 @ particular_2,
    )
    matrix = np.column_stack([column.reshape(-1) for column in columns])
    rhs = (outer_2 @ particular_2 - outer_1 @ particular_1).reshape(-1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    spectrum = (
        float(singular[0]),
        float(singular[1]),
        float(singular[2]),
        float(singular[3]),
    )
    threshold = options.rank_tolerance * max(spectrum[0], 1.0e-15)
    rank = int(np.count_nonzero(singular > threshold))
    condition = spectrum[0] / spectrum[3] if spectrum[3] > 0.0 else math.inf
    parameters, _residuals, _rank, _values = np.linalg.lstsq(matrix, rhs, rcond=None)
    residual = float(np.linalg.norm(matrix @ parameters - rhs) / math.sqrt(len(rhs)))
    unit_error = max(
        abs(float(np.hypot(parameters[0], parameters[1])) - 1.0),
        abs(float(np.hypot(parameters[2], parameters[3])) - 1.0),
    )
    beta_1 = math.atan2(float(parameters[1]), float(parameters[0]))
    beta_2 = math.atan2(float(parameters[3]), float(parameters[2]))
    rotation_1 = _axis_angle_rotation(axis_a1, beta_1) @ particular_1
    rotation_2 = _axis_angle_rotation(axis_a2, beta_2) @ particular_2
    disagreement = _rotation_delta_deg(rotation_1, rotation_2)
    if rank < 4 or condition > options.max_rotation_condition_number:
        return _RotationSolution(
            None, spectrum, rank, condition, residual, unit_error, disagreement
        )
    return _RotationSolution(
        rotation_1, spectrum, rank, condition, residual, unit_error, disagreement
    )


def _solve_translation(
    selected: tuple[HandEyeMotionPair, HandEyeMotionPair],
    rotation_x: FloatArray,
    options: ShiuAhmadHandEyeOptions,
) -> _TranslationSolution:
    identity: FloatArray = np.eye(3, dtype=np.float64)
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    for pair in selected:
        root_weight = math.sqrt(pair.weight)
        rows.extend(root_weight * (_rotation_matrix(pair.motion_a) - identity))
        rhs.extend(
            root_weight
            * (
                rotation_x @ np.asarray(pair.motion_b.translation_m, dtype=np.float64)
                - np.asarray(pair.motion_a.translation_m, dtype=np.float64)
            )
        )
    matrix = np.asarray(rows, dtype=np.float64)
    values = np.linalg.svd(matrix, compute_uv=False)
    spectrum = (float(values[0]), float(values[1]), float(values[2]))
    threshold = options.rank_tolerance * max(spectrum[0], 1.0e-15)
    rank = int(np.count_nonzero(values > threshold))
    condition = spectrum[0] / spectrum[2] if spectrum[2] > 0.0 else math.inf
    if rank < 3 or condition > options.max_translation_condition_number:
        return _TranslationSolution(None, spectrum, rank, condition)
    translation, _residuals, _rank, _values = np.linalg.lstsq(
        matrix, np.asarray(rhs), rcond=None
    )
    return _TranslationSolution(translation, spectrum, rank, condition)


def _rotation_axis(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    if w < 0.0:
        x, y, z = -x, -y, -z
    vector: FloatArray = np.asarray((x, y, z), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("rotation axis is undefined for identity motion")
    return vector / norm


def _rotation_angle(transform: SE3) -> float:
    x, y, z, w = transform.rotation_quat_xyzw
    return 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(w))


def _rotation_between(source: FloatArray, target: FloatArray) -> FloatArray:
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine > 1.0e-12:
        return _axis_angle_rotation(cross / sine, math.atan2(sine, cosine))
    if cosine > 0.0:
        return np.eye(3, dtype=np.float64)
    basis: FloatArray = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(source)))]
    axis = np.cross(source, basis)
    axis /= np.linalg.norm(axis)
    return _axis_angle_rotation(axis, math.pi)


def _axis_angle_rotation(axis: FloatArray, angle: float) -> FloatArray:
    skew = _skew(axis)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rotation_delta_deg(first: FloatArray, second: FloatArray) -> float:
    cosine = float(np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _validate_options(options: ShiuAhmadHandEyeOptions) -> None:
    if options.min_train_motions < 2:
        raise ValueError("min_train_motions must be at least two")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in [0, 1)")
    if not 0.0 < options.min_rotation_rad < options.max_rotation_rad < math.pi:
        raise ValueError("rotation limits must satisfy 0 < min < max < pi")
    if not 0.0 < options.minimum_axis_separation_sine <= 1.0:
        raise ValueError("minimum_axis_separation_sine must be in (0, 1]")
    if options.rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")


def _empty(
    status: ShiuAhmadStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    options: ShiuAhmadHandEyeOptions,
    *,
    fit_ids: tuple[str, ...] = (),
    axis_separation: float | None = None,
    angle_mismatch_deg: float | None = None,
    rotation: _RotationSolution | None = None,
    translation: _TranslationSolution | None = None,
) -> ShiuAhmadHandEyeResult:
    reasons = {
        "insufficient_motions": "too few paper-eligible rotation-excited train motions",
        "no_unique_motion_pair": "no train pair has the paper-required nonparallel axes",
        "degenerate_rotation": "Shiu-Ahmad Eq. (44) rotation system is not full rank",
        "degenerate_translation": "Shiu-Ahmad Eq. (46) translation system is not full rank",
        "converged": "",
    }
    rotation = rotation or _RotationSolution(None, None)
    translation = translation or _TranslationSolution(None, None)
    return ShiuAhmadHandEyeResult(
        status=status,
        reason=reasons[status],
        transform_x=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        fit_pair_ids=fit_ids,
        selected_axis_separation_sine=axis_separation,
        selected_rotation_angle_mismatch_deg_max=angle_mismatch_deg,
        rotation_singular_values=rotation.spectrum,
        rotation_rank=rotation.rank,
        rotation_condition_number=rotation.condition,
        rotation_linear_residual_rmse=rotation.residual_rmse,
        beta_unit_circle_error_max=rotation.unit_circle_error,
        two_rotation_solution_disagreement_deg=rotation.disagreement_deg,
        translation_singular_values=translation.spectrum,
        translation_rank=translation.rank,
        translation_condition_number=translation.condition,
        train_evaluation=HandEyeEvaluation(None, None),
        holdout_evaluation=HandEyeEvaluation(None, None),
        probes=(),
        options=options,
    )
