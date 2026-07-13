"""Incremental Andreff-Horaud-Espiau hand-eye calibration for ``A X = X B``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
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
AndreffStatus = Literal[
    "converged", "insufficient_motions", "degenerate_rotation", "degenerate_translation"
]


@dataclass(frozen=True)
class AndreffHandEyeOptions:
    """Numerical, split, and falsification policy for the incremental solver."""

    min_train_motions: int = 2
    min_rotation_rad: float = 0.0
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-7
    absolute_singular_tolerance: float = 1.0e-12
    minimum_rotation_width: float = 1.0e-4
    max_rotation_nullspace_ratio: float = 0.25
    max_rotation_condition_number: float = 1.0e8
    max_translation_condition_number: float = 1.0e8
    determinant_tolerance: float = 1.0e-12
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_rotation_margin_deg: float = 0.1
    known_bad_translation_margin_m: float = 0.005


@dataclass(frozen=True)
class AndreffIncrementalUpdate:
    """Observability state after one motion update."""

    pair_id: str
    motion_count: int
    rotation_axis_rank: int
    rotation_observable_rank: int
    rotation_minimum_width: float | None
    translation_rank: int
    translation_rotation_rank: int

    def as_dict(self) -> dict[str, object]:
        return self.__dict__


@dataclass(frozen=True)
class AndreffHandEyeResult:
    """Two-stage estimate and complete incremental algebraic evidence."""

    status: AndreffStatus
    reason: str
    transform_x: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    incremental_updates: tuple[AndreffIncrementalUpdate, ...]
    rotation_singular_values: tuple[float, ...] | None
    rotation_axis_singular_values: tuple[float, float, float] | None
    rotation_axis_rank: int
    rotation_observable_rank: int
    rotation_condition_number: float | None
    rotation_minimum_width: float | None
    rotation_nullspace_ratio: float | None
    translation_singular_values: tuple[float, float, float] | None
    translation_rank: int
    translation_condition_number: float | None
    translation_rotation_singular_values: tuple[float, ...] | None
    translation_rotation_rank: int
    raw_rotation_determinant: float | None
    determinant_normalized_orthogonality_error: float | None
    so3_projection_correction_frobenius: float | None
    train_evaluation: HandEyeEvaluation
    holdout_evaluation: HandEyeEvaluation
    probes: tuple[HandEyeProbeResult, ...]

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly evidence with primary-paper provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "incremental_updates": [update.as_dict() for update in self.incremental_updates],
            "rotation_singular_values": self.rotation_singular_values,
            "rotation_axis_singular_values": self.rotation_axis_singular_values,
            "rotation_axis_rank": self.rotation_axis_rank,
            "rotation_observable_rank": self.rotation_observable_rank,
            "rotation_expected_rank": 8,
            "rotation_condition_number": self.rotation_condition_number,
            "rotation_minimum_width": self.rotation_minimum_width,
            "rotation_nullspace_ratio": self.rotation_nullspace_ratio,
            "translation_singular_values": self.translation_singular_values,
            "translation_rank": self.translation_rank,
            "translation_condition_number": self.translation_condition_number,
            "translation_rotation_singular_values": self.translation_rotation_singular_values,
            "translation_rotation_rank": self.translation_rotation_rank,
            "raw_rotation_determinant": self.raw_rotation_determinant,
            "determinant_normalized_orthogonality_error": (
                self.determinant_normalized_orthogonality_error
            ),
            "so3_projection_correction_frobenius": (self.so3_projection_correction_frobenius),
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "andreff_horaud_espiau_incremental_kronecker_two_stage/v0.1",
            "paper_title": "On-line Hand-Eye Calibration",
            "paper_doi": "10.1109/IM.1999.805374",
            "paper_url": ("https://perception.inrialpes.fr/Publications/1999/AHE99/3dim99.pdf"),
            "paper_authors": ["Nicolas Andreff", "Radu Horaud", "Bernard Espiau"],
            "paper_equations": [10, 13, 14, 15],
            "equation": "A X = X B",
            "vectorization": "row-major vec; rotation block I9 - kron(R_A, R_B)",
            "online_state": (
                "9x9 rotation, 3x3 translation, and 3x9 coupling sufficient statistics"
            ),
            "full_12_variable_solution_executed": False,
            "implementation": "independent NumPy implementation; no external code copied",
        }


@dataclass(frozen=True)
class _AccumulatorDiagnostics:
    rotation_spectrum: tuple[float, ...]
    rotation_axis_spectrum: tuple[float, float, float]
    rotation_axis_rank: int
    rotation_rank: int
    rotation_condition: float | None
    rotation_width: float | None
    rotation_nullspace_ratio: float | None
    translation_spectrum: tuple[float, float, float]
    translation_rank: int
    translation_condition: float | None
    translation_rotation_spectrum: tuple[float, ...]
    translation_rotation_rank: int


class AndreffKroneckerAccumulator:
    """Incrementally accumulate equation (10) without retaining raw motions."""

    def __init__(self, options: AndreffHandEyeOptions | None = None) -> None:
        self.options = options or AndreffHandEyeOptions()
        self._rotation_information: FloatArray = np.zeros((9, 9), dtype=np.float64)
        self._rotation_axis_information: FloatArray = np.zeros((3, 3), dtype=np.float64)
        self._translation_information: FloatArray = np.zeros((3, 3), dtype=np.float64)
        self._translation_camera_rhs: FloatArray = np.zeros(3, dtype=np.float64)
        self._translation_rotation_coupling: FloatArray = np.zeros((3, 9), dtype=np.float64)
        self._translation_rotation_information: FloatArray = np.zeros((9, 9), dtype=np.float64)
        self._pair_ids: list[str] = []
        self._updates: list[AndreffIncrementalUpdate] = []

    @property
    def pair_ids(self) -> tuple[str, ...]:
        """Return update lineage in deterministic arrival order."""

        return tuple(self._pair_ids)

    @property
    def updates(self) -> tuple[AndreffIncrementalUpdate, ...]:
        """Return rank evolution after every accepted update."""

        return tuple(self._updates)

    def update(self, motion: HandEyeMotionPair) -> AndreffIncrementalUpdate:
        """Add one weighted motion to the online sufficient statistics."""

        if motion.weight <= 0.0 or not math.isfinite(motion.weight):
            raise ValueError("Andreff motion weight must be finite and positive")
        if motion.pair_id in self._pair_ids:
            raise ValueError(f"duplicate Andreff motion pair_id: {motion.pair_id}")
        rotation_a = _rotation_matrix(motion.motion_a)
        rotation_b = _rotation_matrix(motion.motion_b)
        translation_a: FloatArray = np.asarray(motion.motion_a.translation_m, dtype=np.float64)
        translation_b: FloatArray = np.asarray(motion.motion_b.translation_m, dtype=np.float64)
        weight = motion.weight
        rotation_block = np.eye(9, dtype=np.float64) - np.kron(rotation_a, rotation_b)
        translation_block = np.eye(3, dtype=np.float64) - rotation_a
        rotation_translation_block = np.kron(
            np.eye(3, dtype=np.float64), translation_b.reshape(1, 3)
        )
        self._rotation_information += weight * rotation_block.T @ rotation_block
        rotation_axis = _rotation_axis(motion.motion_b)
        if rotation_axis is not None:
            self._rotation_axis_information += weight * np.outer(rotation_axis, rotation_axis)
        self._translation_information += weight * translation_block.T @ translation_block
        self._translation_camera_rhs += weight * translation_block.T @ translation_a
        self._translation_rotation_coupling += (
            weight * translation_block.T @ rotation_translation_block
        )
        self._translation_rotation_information += (
            weight * rotation_translation_block.T @ rotation_translation_block
        )
        self._pair_ids.append(motion.pair_id)
        diagnostics = self.diagnostics()
        update = AndreffIncrementalUpdate(
            pair_id=motion.pair_id,
            motion_count=len(self._pair_ids),
            rotation_axis_rank=diagnostics.rotation_axis_rank,
            rotation_observable_rank=diagnostics.rotation_rank,
            rotation_minimum_width=diagnostics.rotation_width,
            translation_rank=diagnostics.translation_rank,
            translation_rotation_rank=diagnostics.translation_rotation_rank,
        )
        self._updates.append(update)
        return update

    def diagnostics(self) -> _AccumulatorDiagnostics:
        """Compute current rank and spectral diagnostics without estimating ``X``."""

        rotation_spectrum = _information_singular_values(self._rotation_information)
        rotation_axis_spectrum = _three_information_singular_values(self._rotation_axis_information)
        rotation_axis_rank = _rank(
            rotation_axis_spectrum,
            self.options.rank_tolerance,
            self.options.absolute_singular_tolerance,
        )
        rotation_scale = rotation_spectrum[0]
        numerical_rotation_rank = (
            sum(
                value > self.options.rank_tolerance * rotation_scale
                for value in rotation_spectrum[:8]
            )
            if rotation_scale > self.options.absolute_singular_tolerance
            else 0
        )
        rotation_rank = (
            numerical_rotation_rank if rotation_axis_rank >= 2 else min(numerical_rotation_rank, 6)
        )
        rotation_condition = (
            rotation_spectrum[0] / rotation_spectrum[7]
            if rotation_rank == 8 and rotation_spectrum[7] > 0.0
            else None
        )
        rotation_width = (
            rotation_spectrum[7] / rotation_spectrum[0] if rotation_spectrum[0] > 0.0 else None
        )
        rotation_nullspace_ratio = (
            rotation_spectrum[8] / rotation_spectrum[7] if rotation_spectrum[7] > 0.0 else None
        )
        translation_spectrum = _three_information_singular_values(self._translation_information)
        translation_rank, translation_condition = _rank_condition(
            translation_spectrum,
            self.options.rank_tolerance,
            self.options.absolute_singular_tolerance,
        )
        translation_rotation_spectrum = _information_singular_values(
            self._translation_rotation_information
        )
        translation_rotation_rank = _rank(
            translation_rotation_spectrum,
            self.options.rank_tolerance,
            self.options.absolute_singular_tolerance,
        )
        return _AccumulatorDiagnostics(
            rotation_spectrum=rotation_spectrum,
            rotation_axis_spectrum=rotation_axis_spectrum,
            rotation_axis_rank=rotation_axis_rank,
            rotation_rank=rotation_rank,
            rotation_condition=rotation_condition,
            rotation_width=rotation_width,
            rotation_nullspace_ratio=rotation_nullspace_ratio,
            translation_spectrum=translation_spectrum,
            translation_rank=translation_rank,
            translation_condition=translation_condition,
            translation_rotation_spectrum=translation_rotation_spectrum,
            translation_rotation_rank=translation_rotation_rank,
        )

    def estimate(
        self,
    ) -> tuple[
        SE3 | None,
        AndreffStatus,
        str,
        _AccumulatorDiagnostics,
        float | None,
        float | None,
        float | None,
    ]:
        """Solve the paper's rotation-kernel then translation two-stage system."""

        diagnostics = self.diagnostics()
        if (
            diagnostics.rotation_rank < 8
            or diagnostics.rotation_width is None
            or diagnostics.rotation_width < self.options.minimum_rotation_width
            or diagnostics.rotation_nullspace_ratio is None
            or diagnostics.rotation_nullspace_ratio > self.options.max_rotation_nullspace_ratio
            or diagnostics.rotation_condition is None
            or diagnostics.rotation_condition > self.options.max_rotation_condition_number
        ):
            return (
                None,
                "degenerate_rotation",
                "Kronecker rotation kernel lacks eight observable directions",
                diagnostics,
                None,
                None,
                None,
            )
        eigenvalues, eigenvectors = np.linalg.eigh(self._rotation_information)
        del eigenvalues
        raw_rotation = eigenvectors[:, 0].reshape(3, 3)
        raw_determinant = float(np.linalg.det(raw_rotation))
        if abs(raw_determinant) <= self.options.determinant_tolerance:
            return (
                None,
                "degenerate_rotation",
                "Kronecker kernel rotation has near-zero determinant",
                diagnostics,
                raw_determinant,
                None,
                None,
            )
        determinant_normalized = (
            math.copysign(1.0, raw_determinant) * raw_rotation / abs(raw_determinant) ** (1.0 / 3.0)
        )
        orthogonality_error = float(
            np.linalg.norm(determinant_normalized.T @ determinant_normalized - np.eye(3), ord="fro")
        )
        left, _singular, right_t = np.linalg.svd(determinant_normalized)
        correction: FloatArray = np.eye(3, dtype=np.float64)
        correction[2, 2] = np.linalg.det(left @ right_t)
        rotation = left @ correction @ right_t
        projection_correction = float(np.linalg.norm(rotation - determinant_normalized, ord="fro"))
        if (
            diagnostics.translation_rank < 3
            or diagnostics.translation_condition is None
            or diagnostics.translation_condition > self.options.max_translation_condition_number
        ):
            return (
                None,
                "degenerate_translation",
                "motion rotations do not constrain all translation directions",
                diagnostics,
                raw_determinant,
                orthogonality_error,
                projection_correction,
            )
        translation_rhs = (
            self._translation_camera_rhs
            - self._translation_rotation_coupling @ rotation.reshape(-1)
        )
        translation = np.linalg.solve(self._translation_information, translation_rhs)
        transform = SE3(
            (float(translation[0]), float(translation[1]), float(translation[2])),
            quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
        )
        return (
            transform,
            "converged",
            "incremental Kronecker rotation and conditional translation systems solved",
            diagnostics,
            raw_determinant,
            orthogonality_error,
            projection_correction,
        )


class AndreffHandEyeSolver:
    """Fit the incremental two-stage solver on train motions and evaluate holdout."""

    def solve(
        self,
        motions: Sequence[HandEyeMotionPair],
        options: AndreffHandEyeOptions | None = None,
    ) -> AndreffHandEyeResult:
        solver_options = options or AndreffHandEyeOptions()
        usable = sorted(
            (
                motion
                for motion in motions
                if motion.weight > 0.0
                and _rotation_angle(motion.motion_a) >= solver_options.min_rotation_rad
                and _rotation_angle(motion.motion_b) >= solver_options.min_rotation_rad
            ),
            key=lambda motion: motion.pair_id,
        )
        train_indices, holdout_indices = split_indices(
            len(usable), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        train_ids = tuple(motion.pair_id for motion in train)
        holdout_ids = tuple(motion.pair_id for motion in holdout)
        if len(train) < solver_options.min_train_motions:
            return _empty("insufficient_motions", train_ids, holdout_ids)
        accumulator = AndreffKroneckerAccumulator(solver_options)
        for motion in train:
            accumulator.update(motion)
        (
            transform,
            status,
            reason,
            diagnostics,
            raw_determinant,
            orthogonality_error,
            projection_correction,
        ) = accumulator.estimate()
        if transform is None:
            return _empty(
                status,
                train_ids,
                holdout_ids,
                accumulator.updates,
                diagnostics,
                raw_determinant,
                orthogonality_error,
                projection_correction,
                reason,
            )
        train_evaluation = evaluate_hand_eye_motions(train, transform)
        holdout_evaluation = evaluate_hand_eye_motions(holdout, transform)
        return AndreffHandEyeResult(
            status=status,
            reason=reason,
            transform_x=transform,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            incremental_updates=accumulator.updates,
            rotation_singular_values=diagnostics.rotation_spectrum,
            rotation_axis_singular_values=diagnostics.rotation_axis_spectrum,
            rotation_axis_rank=diagnostics.rotation_axis_rank,
            rotation_observable_rank=diagnostics.rotation_rank,
            rotation_condition_number=diagnostics.rotation_condition,
            rotation_minimum_width=diagnostics.rotation_width,
            rotation_nullspace_ratio=diagnostics.rotation_nullspace_ratio,
            translation_singular_values=diagnostics.translation_spectrum,
            translation_rank=diagnostics.translation_rank,
            translation_condition_number=diagnostics.translation_condition,
            translation_rotation_singular_values=(diagnostics.translation_rotation_spectrum),
            translation_rotation_rank=diagnostics.translation_rotation_rank,
            raw_rotation_determinant=raw_determinant,
            determinant_normalized_orthogonality_error=orthogonality_error,
            so3_projection_correction_frobenius=projection_correction,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=evaluate_hand_eye_known_bad_probes(
                holdout, transform, holdout_evaluation, solver_options
            ),
        )


def _information_singular_values(information: FloatArray) -> tuple[float, ...]:
    eigenvalues = np.linalg.eigvalsh(information)
    return tuple(float(math.sqrt(max(0.0, value))) for value in eigenvalues[::-1])


def _three_information_singular_values(
    information: FloatArray,
) -> tuple[float, float, float]:
    values = _information_singular_values(information)
    return (values[0], values[1], values[2])


def _rank(values: Sequence[float], tolerance: float, absolute_tolerance: float) -> int:
    if not values or values[0] <= absolute_tolerance:
        return 0
    threshold = max(absolute_tolerance, tolerance * values[0])
    return sum(value > threshold for value in values)


def _rank_condition(
    values: tuple[float, float, float], tolerance: float, absolute_tolerance: float
) -> tuple[int, float | None]:
    rank = _rank(values, tolerance, absolute_tolerance)
    condition = values[0] / values[2] if rank == 3 and values[2] > 0.0 else None
    return rank, condition


def _rotation_angle(transform: SE3) -> float:
    x, y, z, w = transform.rotation_quat_xyzw
    return 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(w))


def _rotation_axis(transform: SE3) -> FloatArray | None:
    x, y, z, w = transform.rotation_quat_xyzw
    if w < 0.0:
        x, y, z = -x, -y, -z
    axis: FloatArray = np.asarray((x, y, z), dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1.0e-15 else None


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


def _empty(
    status: AndreffStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    updates: tuple[AndreffIncrementalUpdate, ...] = (),
    diagnostics: _AccumulatorDiagnostics | None = None,
    raw_determinant: float | None = None,
    orthogonality_error: float | None = None,
    projection_correction: float | None = None,
    reason_override: str | None = None,
) -> AndreffHandEyeResult:
    reasons = {
        "insufficient_motions": "too few usable motion pairs",
        "degenerate_rotation": "Kronecker rotation kernel is not observable",
        "degenerate_translation": "motion rotations do not constrain translation",
        "converged": "",
    }
    return AndreffHandEyeResult(
        status=status,
        reason=reason_override or reasons[status],
        transform_x=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        incremental_updates=updates,
        rotation_singular_values=(diagnostics.rotation_spectrum if diagnostics else None),
        rotation_axis_singular_values=(diagnostics.rotation_axis_spectrum if diagnostics else None),
        rotation_axis_rank=diagnostics.rotation_axis_rank if diagnostics else 0,
        rotation_observable_rank=diagnostics.rotation_rank if diagnostics else 0,
        rotation_condition_number=(diagnostics.rotation_condition if diagnostics else None),
        rotation_minimum_width=diagnostics.rotation_width if diagnostics else None,
        rotation_nullspace_ratio=(diagnostics.rotation_nullspace_ratio if diagnostics else None),
        translation_singular_values=(diagnostics.translation_spectrum if diagnostics else None),
        translation_rank=diagnostics.translation_rank if diagnostics else 0,
        translation_condition_number=(diagnostics.translation_condition if diagnostics else None),
        translation_rotation_singular_values=(
            diagnostics.translation_rotation_spectrum if diagnostics else None
        ),
        translation_rotation_rank=(diagnostics.translation_rotation_rank if diagnostics else 0),
        raw_rotation_determinant=raw_determinant,
        determinant_normalized_orthogonality_error=orthogonality_error,
        so3_projection_correction_frobenius=projection_correction,
        train_evaluation=HandEyeEvaluation(None, None),
        holdout_evaluation=HandEyeEvaluation(None, None),
        probes=(),
    )
