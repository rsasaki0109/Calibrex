"""Daniilidis simultaneous dual-quaternion hand-eye calibration."""

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
DaniilidisStatus = Literal[
    "converged",
    "insufficient_motions",
    "degenerate_dual_quaternion_system",
    "constraint_failure",
]


@dataclass(frozen=True)
class DaniilidisHandEyeOptions:
    min_train_motions: int = 3
    min_rotation_rad: float = math.radians(1.0)
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-7
    max_observable_condition_number: float = 1.0e8
    constraint_tolerance: float = 1.0e-8
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_rotation_margin_deg: float = 0.1
    known_bad_translation_margin_m: float = 0.005


@dataclass(frozen=True)
class DaniilidisHandEyeResult:
    status: DaniilidisStatus
    reason: str
    transform_x: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    linear_singular_values: tuple[float, ...] | None
    linear_rank: int
    expected_nullity: int
    observable_condition_number: float | None
    nullspace_gap: float | None
    candidate_count: int
    unit_constraint_error: float | None
    study_constraint_error: float | None
    train_evaluation: HandEyeEvaluation
    holdout_evaluation: HandEyeEvaluation
    probes: tuple[HandEyeProbeResult, ...]

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly simultaneous-solver output and provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "linear_singular_values": self.linear_singular_values,
            "linear_rank": self.linear_rank,
            "expected_nullity": self.expected_nullity,
            "observable_condition_number": self.observable_condition_number,
            "nullspace_gap": self.nullspace_gap,
            "candidate_count": self.candidate_count,
            "unit_constraint_error": self.unit_constraint_error,
            "study_constraint_error": self.study_constraint_error,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "daniilidis_dual_quaternion_nullspace/v0.1",
            "paper_doi": "10.1177/02783649922066213",
            "equation": "A X = X B",
            "motion_convention": "A and B are corresponding relative SE(3) motions",
            "constraints": ["real quaternion unit norm", "Study real-dual orthogonality"],
            "implementation": "independent NumPy implementation; no external code copied",
        }


class DaniilidisHandEyeSolver:
    """Solve rotation and translation simultaneously in dual-quaternion form."""

    def solve(
        self,
        motions: Sequence[HandEyeMotionPair],
        options: DaniilidisHandEyeOptions | None = None,
    ) -> DaniilidisHandEyeResult:
        solver_options = options or DaniilidisHandEyeOptions()
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

        matrix = _dual_quaternion_system(train)
        _left, spectrum, right_t = np.linalg.svd(matrix, full_matrices=False)
        singular_values = tuple(float(value) for value in spectrum)
        rank = int(
            np.count_nonzero(spectrum > solver_options.rank_tolerance * spectrum[0])
        )
        condition = float(spectrum[0] / spectrum[5]) if spectrum[5] > 0.0 else math.inf
        gap = float(spectrum[5] / max(spectrum[6], 1.0e-15))
        if rank < 6 or condition > solver_options.max_observable_condition_number:
            return _empty(
                "degenerate_dual_quaternion_system",
                train_ids,
                holdout_ids,
                singular_values,
                rank,
                condition,
                gap,
            )

        candidates = _study_constrained_candidates(
            right_t[-2], right_t[-1], solver_options.constraint_tolerance
        )
        scored = [(_closure_score(train, candidate), candidate) for candidate in candidates]
        if not scored:
            return _empty(
                "constraint_failure",
                train_ids,
                holdout_ids,
                singular_values,
                rank,
                condition,
                gap,
            )
        _score, transform = min(scored, key=lambda item: item[0])
        real, dual = _dual_quaternion(transform)
        unit_error = abs(float(real @ real) - 1.0)
        study_error = abs(float(real @ dual))
        train_evaluation = evaluate_hand_eye_motions(train, transform)
        holdout_evaluation = evaluate_hand_eye_motions(holdout, transform)
        return DaniilidisHandEyeResult(
            status="converged",
            reason="dual-quaternion nullspace and Study constraints solved",
            transform_x=transform,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            linear_singular_values=singular_values,
            linear_rank=rank,
            expected_nullity=2,
            observable_condition_number=condition,
            nullspace_gap=gap,
            candidate_count=len(candidates),
            unit_constraint_error=unit_error,
            study_constraint_error=study_error,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=evaluate_hand_eye_known_bad_probes(
                holdout, transform, holdout_evaluation, solver_options
            ),
        )


def _dual_quaternion_system(motions: Sequence[HandEyeMotionPair]) -> FloatArray:
    blocks: list[FloatArray] = []
    zero = np.zeros((4, 4), dtype=np.float64)
    for pair in motions:
        real_a, dual_a = _dual_quaternion(pair.motion_a)
        real_b, dual_b = _dual_quaternion(pair.motion_b)
        real_system = _left_matrix(real_a) - _right_matrix(real_b)
        dual_system = _left_matrix(dual_a) - _right_matrix(dual_b)
        root_weight = math.sqrt(pair.weight)
        blocks.append(
            root_weight
            * np.block([[real_system, zero], [dual_system, real_system]])
        )
    return np.vstack(blocks)


def _study_constrained_candidates(
    first: FloatArray, second: FloatArray, tolerance: float
) -> list[SE3]:
    real_first, dual_first = first[:4], first[4:]
    real_second, dual_second = second[:4], second[4:]
    coefficient_a = float(real_first @ dual_first)
    coefficient_b = float(real_first @ dual_second + real_second @ dual_first)
    coefficient_c = float(real_second @ dual_second)
    ratios = _real_quadratic_roots(
        coefficient_a, coefficient_b, coefficient_c, tolerance
    )
    raw_candidates = [ratio * first + second for ratio in ratios]
    if abs(coefficient_a) <= tolerance:
        raw_candidates.append(first)

    candidates: list[SE3] = []
    for raw in raw_candidates:
        real = raw[:4].copy()
        dual = raw[4:].copy()
        norm = float(np.linalg.norm(real))
        if norm <= tolerance:
            continue
        real /= norm
        dual /= norm
        dual -= real * float(real @ dual)
        if real[3] < 0.0:
            real *= -1.0
            dual *= -1.0
        translation_quaternion = 2.0 * _quaternion_multiply(
            dual, _quaternion_conjugate(real)
        )
        candidates.append(
            SE3(
                (
                    float(translation_quaternion[0]),
                    float(translation_quaternion[1]),
                    float(translation_quaternion[2]),
                ),
                (float(real[0]), float(real[1]), float(real[2]), float(real[3])),
            )
        )
    return candidates


def _real_quadratic_roots(a: float, b: float, c: float, tolerance: float) -> list[float]:
    if abs(a) <= tolerance:
        return [-c / b] if abs(b) > tolerance else []
    discriminant = b * b - 4.0 * a * c
    if discriminant < -tolerance:
        return []
    root = math.sqrt(max(0.0, discriminant))
    return [(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)]


def _dual_quaternion(transform: SE3) -> tuple[FloatArray, FloatArray]:
    real = np.asarray(_canonical_quaternion(transform), dtype=np.float64)
    translation = np.asarray((*transform.translation_m, 0.0), dtype=np.float64)
    dual = 0.5 * _quaternion_multiply(translation, real)
    return real, dual


def _canonical_quaternion(transform: SE3) -> tuple[float, float, float, float]:
    x, y, z, w = transform.rotation_quat_xyzw
    return (-x, -y, -z, -w) if w < 0.0 else (x, y, z, w)


def _rotation_angle(transform: SE3) -> float:
    _x, _y, _z, w = _canonical_quaternion(transform)
    return 2.0 * math.acos(min(1.0, max(-1.0, w)))


def _left_matrix(quaternion: FloatArray) -> FloatArray:
    x, y, z, w = quaternion
    return np.asarray(
        ((w, -z, y, x), (z, w, -x, y), (-y, x, w, z), (-x, -y, -z, w))
    )


def _right_matrix(quaternion: FloatArray) -> FloatArray:
    x, y, z, w = quaternion
    return np.asarray(
        ((w, z, -y, x), (-z, w, x, y), (y, -x, w, z), (-x, -y, -z, w))
    )


def _quaternion_multiply(left: FloatArray, right: FloatArray) -> FloatArray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def _quaternion_conjugate(quaternion: FloatArray) -> FloatArray:
    return np.asarray((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]))


def _closure_score(motions: Sequence[HandEyeMotionPair], transform: SE3) -> float:
    evaluation = evaluate_hand_eye_motions(motions, transform)
    rotation = evaluation.rotation_closure_rmse_deg or 0.0
    translation = evaluation.translation_closure_rmse_m or 0.0
    return math.radians(rotation) ** 2 + translation**2


def _empty(
    status: DaniilidisStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    spectrum: tuple[float, ...] | None = None,
    rank: int = 0,
    condition: float | None = None,
    gap: float | None = None,
) -> DaniilidisHandEyeResult:
    reasons = {
        "insufficient_motions": "too few rotation-excited motion pairs",
        "degenerate_dual_quaternion_system": (
            "dual-quaternion motion system lacks six observable directions"
        ),
        "constraint_failure": "nullspace has no real unit Study-quadric solution",
        "converged": "",
    }
    return DaniilidisHandEyeResult(
        status=status,
        reason=reasons[status],
        transform_x=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        linear_singular_values=spectrum,
        linear_rank=rank,
        expected_nullity=2,
        observable_condition_number=condition,
        nullspace_gap=gap,
        candidate_count=0,
        unit_constraint_error=None,
        study_constraint_error=None,
        train_evaluation=HandEyeEvaluation(None, None),
        holdout_evaluation=HandEyeEvaluation(None, None),
        probes=(),
    )
