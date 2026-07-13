"""Zhuang-Roth-Sudhakar linear quaternion robot-world/hand-eye solver."""

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
ZhuangRothSudhakarStatus = Literal[
    "converged",
    "insufficient_poses",
    "special_a_scalar_zero",
    "degenerate_rotation",
    "special_z_scalar_zero",
    "inconsistent_quaternion_normalization",
    "degenerate_translation",
]


@dataclass(frozen=True)
class ZhuangRothSudhakarOptions:
    """Numerical and evaluation policy for the 1994 linear method."""

    min_train_poses: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-9
    minimum_abs_a_scalar: float = 1.0e-8
    minimum_abs_z_scalar: float = 1.0e-8
    max_rotation_condition_number: float = 1.0e8
    max_quaternion_normalization_disagreement: float = 0.05
    max_translation_condition_number: float = 1.0e8
    known_bad_translation_m: float = 0.05
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_margin_m: float = 0.001
    known_bad_rotation_margin_deg: float = 0.1


@dataclass(frozen=True)
class ZhuangRothSudhakarResult:
    """Transforms and falsifiable diagnostics for the Zhuang linear solve."""

    status: ZhuangRothSudhakarStatus
    reason: str
    transform_x: SE3 | None
    transform_z: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    rotation_singular_values: tuple[float, ...]
    rotation_rank: int
    rotation_condition_number: float | None
    minimum_abs_a_scalar: float | None
    recovered_abs_z_scalar: float | None
    quaternion_x_raw_norm: float | None
    quaternion_z_raw_norm: float | None
    quaternion_normalization_disagreement: float | None
    scalar_reconstruction_rmse: float | None
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
        """Return schema-friendly estimates, diagnostics, and provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "transform_z": self.transform_z.as_dict() if self.transform_z else None,
            "shared_role_alias": "Y := Z",
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "rotation_singular_values": list(self.rotation_singular_values),
            "rotation_rank": self.rotation_rank,
            "rotation_condition_number": self.rotation_condition_number,
            "minimum_abs_a_scalar": self.minimum_abs_a_scalar,
            "recovered_abs_z_scalar": self.recovered_abs_z_scalar,
            "quaternion_x_raw_norm": self.quaternion_x_raw_norm,
            "quaternion_z_raw_norm": self.quaternion_z_raw_norm,
            "quaternion_normalization_disagreement": (
                self.quaternion_normalization_disagreement
            ),
            "scalar_reconstruction_rmse": self.scalar_reconstruction_rmse,
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
            "method": "zhuang_roth_sudhakar_robot_world_hand_eye_linear/v0.1",
            "paper": {
                "title": (
                    "Simultaneous robot/world and tool/flange calibration by solving "
                    "homogeneous transformation equations of the form AX=YB"
                ),
                "authors": "Hanqi Zhuang, Zvi S. Roth, and Raghavan Sudhakar",
                "doi": "10.1109/70.313105",
                "venue": "IEEE Transactions on Robotics and Automation 10(4), 1994",
            },
            "equation": "A_i X = Z B_i",
            "rotation_contract": (
                "paper linear quaternion elimination: equations (4)-(8), solve the "
                "stacked 3n-by-6 system for [x/z0; z/z0], reconstruct x0/z0, then "
                "normalize q_X and q_Z separately"
            ),
            "overdetermined_extension": (
                "positive pose weights, weighted least squares, and weighted-mean scalar "
                "reconstruction extend the paper's exact three-pose construction"
            ),
            "special_configuration_contract": (
                "reject a0 approximately zero or recovered z0 approximately zero, the "
                "two failure configurations stated by the paper"
            ),
            "translation_contract": (
                "weighted least squares [R_Ai, -I] [t_X; t_Z] = R_Z t_Bi - t_Ai"
            ),
            "quaternion_sign_contract": (
                "deterministic maximum-spanning-tree synchronization using the invariant "
                "sign of dot(q_Ai,q_Aj)*dot(q_Bi,q_Bj) before paper elimination"
            ),
            "external_code_executed": False,
            "evaluation_contract": "closure on held-out absolute pose pairs without refitting",
        }


class ZhuangRothSudhakarSolver:
    """Solve ``A_i X = Z B_i`` with the 1994 linear quaternion method."""

    def solve(
        self,
        poses: Sequence[RobotWorldHandEyePosePair],
        options: ZhuangRothSudhakarOptions | None = None,
    ) -> ZhuangRothSudhakarResult:
        solver_options = options or ZhuangRothSudhakarOptions()
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
        minimum_abs_a = min(abs(_scalar_first_quaternion(pair.pose_a)[0]) for pair in train)
        if minimum_abs_a < solver_options.minimum_abs_a_scalar:
            return _empty(
                "special_a_scalar_zero",
                train_ids,
                holdout_ids,
                minimum_abs_a=minimum_abs_a,
                sign_flips=sign_flips,
                sign_fraction=sign_fraction,
            )

        matrix, vector = _rotation_system(train, signs)
        singular = np.linalg.svd(matrix, compute_uv=False)
        spectrum = tuple(float(value) for value in singular)
        threshold = solver_options.rank_tolerance * spectrum[0]
        rank = int(np.count_nonzero(singular > threshold))
        condition = spectrum[0] / spectrum[-1] if spectrum[-1] > 0.0 else math.inf
        if rank < 6 or condition > solver_options.max_rotation_condition_number:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                rotation_spectrum=spectrum,
                rotation_rank=rank,
                rotation_condition=condition,
                minimum_abs_a=minimum_abs_a,
                sign_flips=sign_flips,
                sign_fraction=sign_fraction,
            )

        scaled_vectors, _residuals, _rank, _values = np.linalg.lstsq(
            matrix, vector, rcond=None
        )
        scaled_x = scaled_vectors[:3]
        scaled_z = scaled_vectors[3:]
        scaled_x_scalars = _reconstructed_x_scalars(train, signs, scaled_x, scaled_z)
        weights = np.asarray([pair.weight for pair in train], dtype=np.float64)
        scaled_x_scalar = float(np.average(scaled_x_scalars, weights=weights))
        scalar_rmse = float(
            np.sqrt(np.average(np.square(scaled_x_scalars - scaled_x_scalar), weights=weights))
        )
        raw_quaternion_x = np.concatenate(([scaled_x_scalar], scaled_x))
        raw_quaternion_z = np.concatenate(([1.0], scaled_z))
        norm_x = float(np.linalg.norm(raw_quaternion_x))
        norm_z = float(np.linalg.norm(raw_quaternion_z))
        recovered_abs_z_scalar = 1.0 / norm_z if norm_z > 0.0 else 0.0
        if recovered_abs_z_scalar < solver_options.minimum_abs_z_scalar:
            return _empty(
                "special_z_scalar_zero",
                train_ids,
                holdout_ids,
                rotation_spectrum=spectrum,
                rotation_rank=rank,
                rotation_condition=condition,
                minimum_abs_a=minimum_abs_a,
                recovered_abs_z_scalar=recovered_abs_z_scalar,
                raw_norm_x=norm_x,
                raw_norm_z=norm_z,
                scalar_rmse=scalar_rmse,
                sign_flips=sign_flips,
                sign_fraction=sign_fraction,
            )
        normalization_disagreement = abs(norm_x - norm_z) / max(norm_x, norm_z)
        if (
            not math.isfinite(normalization_disagreement)
            or normalization_disagreement
            > solver_options.max_quaternion_normalization_disagreement
        ):
            return _empty(
                "inconsistent_quaternion_normalization",
                train_ids,
                holdout_ids,
                rotation_spectrum=spectrum,
                rotation_rank=rank,
                rotation_condition=condition,
                minimum_abs_a=minimum_abs_a,
                recovered_abs_z_scalar=recovered_abs_z_scalar,
                raw_norm_x=norm_x,
                raw_norm_z=norm_z,
                normalization_disagreement=normalization_disagreement,
                scalar_rmse=scalar_rmse,
                sign_flips=sign_flips,
                sign_fraction=sign_fraction,
            )

        quaternion_x = raw_quaternion_x / norm_x
        quaternion_z = raw_quaternion_z / norm_z
        rotation_z = _rotation_matrix_from_scalar_first_quaternion(quaternion_z)
        translation = _solve_translations(train, rotation_z, solver_options)
        if translation.solution is None:
            return _empty(
                "degenerate_translation",
                train_ids,
                holdout_ids,
                rotation_spectrum=spectrum,
                rotation_rank=rank,
                rotation_condition=condition,
                minimum_abs_a=minimum_abs_a,
                recovered_abs_z_scalar=recovered_abs_z_scalar,
                raw_norm_x=norm_x,
                raw_norm_z=norm_z,
                normalization_disagreement=normalization_disagreement,
                scalar_rmse=scalar_rmse,
                sign_flips=sign_flips,
                sign_fraction=sign_fraction,
                translation_spectrum=translation.singular_values,
                translation_rank=translation.rank,
                translation_condition=translation.condition_number,
            )

        transform_x = _se3(translation.solution[:3], quaternion_x)
        transform_z = _se3(translation.solution[3:], quaternion_z)
        train_evaluation = evaluate_robot_world_hand_eye_poses(train, transform_x, transform_z)
        holdout_evaluation = evaluate_robot_world_hand_eye_poses(holdout, transform_x, transform_z)
        probes = evaluate_robot_world_hand_eye_known_bad_probes(
            holdout,
            transform_x,
            transform_z,
            holdout_evaluation,
            solver_options,
        )
        return ZhuangRothSudhakarResult(
            status="converged",
            reason="linear quaternion rotations and conditional translations solved",
            transform_x=transform_x,
            transform_z=transform_z,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            rotation_singular_values=spectrum,
            rotation_rank=rank,
            rotation_condition_number=condition,
            minimum_abs_a_scalar=minimum_abs_a,
            recovered_abs_z_scalar=recovered_abs_z_scalar,
            quaternion_x_raw_norm=norm_x,
            quaternion_z_raw_norm=norm_z,
            quaternion_normalization_disagreement=normalization_disagreement,
            scalar_reconstruction_rmse=scalar_rmse,
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


def _rotation_system(
    poses: Sequence[RobotWorldHandEyePosePair], signs: FloatArray
) -> tuple[FloatArray, FloatArray]:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    identity = np.eye(3, dtype=np.float64)
    for index, pair in enumerate(poses):
        quaternion_a = _scalar_first_quaternion(pair.pose_a)
        quaternion_b = signs[index] * _scalar_first_quaternion(pair.pose_b)
        scalar_a, vector_a = quaternion_a[0], quaternion_a[1:]
        scalar_b, vector_b = quaternion_b[0], quaternion_b[1:]
        root_weight = math.sqrt(pair.weight)
        block_x = (
            scalar_a * identity
            + np.outer(vector_a, vector_a) / scalar_a
            + _skew(vector_a)
        )
        block_z = (
            -scalar_b * identity
            - np.outer(vector_a, vector_b) / scalar_a
            + _skew(vector_b)
        )
        rows.append(root_weight * np.hstack((block_x, block_z)))
        rhs.append(root_weight * (vector_b - (scalar_b / scalar_a) * vector_a))
    return np.vstack(rows), np.concatenate(rhs)


def _reconstructed_x_scalars(
    poses: Sequence[RobotWorldHandEyePosePair],
    signs: FloatArray,
    scaled_x: FloatArray,
    scaled_z: FloatArray,
) -> FloatArray:
    values = np.empty(len(poses), dtype=np.float64)
    for index, pair in enumerate(poses):
        quaternion_a = _scalar_first_quaternion(pair.pose_a)
        quaternion_b = signs[index] * _scalar_first_quaternion(pair.pose_b)
        values[index] = (
            quaternion_a[1:] @ scaled_x
            + quaternion_b[0]
            - quaternion_b[1:] @ scaled_z
        ) / quaternion_a[0]
    return values


def _solve_translations(
    poses: Sequence[RobotWorldHandEyePosePair],
    rotation_z: FloatArray,
    options: ZhuangRothSudhakarOptions,
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


def _quaternion_signs(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[FloatArray, int, float | None]:
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
        signs[child] = signs[parent] * (1.0 if scores[parent, child] >= 0.0 else -1.0)
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


def _scalar_first_quaternion(transform: SE3) -> FloatArray:
    x, y, z, scalar = transform.rotation_quat_xyzw
    quaternion = np.asarray((scalar, x, y, z), dtype=np.float64)
    return -quaternion if quaternion[0] < 0.0 else quaternion


def _rotation_matrix(transform: SE3) -> FloatArray:
    return _rotation_matrix_from_scalar_first_quaternion(_scalar_first_quaternion(transform))


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


def _se3(translation: FloatArray, quaternion: FloatArray) -> SE3:
    return SE3(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
            float(quaternion[0]),
        ),
    )


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def _validated_poses(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[RobotWorldHandEyePosePair, ...]:
    names: set[str] = set()
    output: list[RobotWorldHandEyePosePair] = []
    for pair in sorted(poses, key=lambda item: item.pair_id):
        if not pair.pair_id or pair.pair_id in names:
            raise ValueError("Zhuang-Roth-Sudhakar pair IDs must be unique and non-empty")
        if not math.isfinite(pair.weight) or pair.weight <= 0.0:
            raise ValueError("Zhuang-Roth-Sudhakar weights must be finite and positive")
        names.add(pair.pair_id)
        output.append(pair)
    return tuple(output)


def _validate_options(options: ZhuangRothSudhakarOptions) -> None:
    if options.min_train_poses < 3:
        raise ValueError("Zhuang-Roth-Sudhakar requires at least three train poses")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("Zhuang-Roth-Sudhakar holdout ratio must be in [0, 1)")
    positive = (
        options.rank_tolerance,
        options.minimum_abs_a_scalar,
        options.minimum_abs_z_scalar,
        options.max_rotation_condition_number,
        options.max_quaternion_normalization_disagreement,
        options.max_translation_condition_number,
        options.known_bad_translation_m,
        options.known_bad_rotation_deg,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("Zhuang-Roth-Sudhakar numerical options must be finite and positive")
    margins = (
        options.known_bad_translation_margin_m,
        options.known_bad_rotation_margin_deg,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in margins):
        raise ValueError("Zhuang-Roth-Sudhakar probe margins must be finite and non-negative")


def _empty(
    status: ZhuangRothSudhakarStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    rotation_spectrum: tuple[float, ...] = (),
    rotation_rank: int = 0,
    rotation_condition: float | None = None,
    minimum_abs_a: float | None = None,
    recovered_abs_z_scalar: float | None = None,
    raw_norm_x: float | None = None,
    raw_norm_z: float | None = None,
    normalization_disagreement: float | None = None,
    scalar_rmse: float | None = None,
    sign_flips: int = 0,
    sign_fraction: float | None = None,
    translation_spectrum: tuple[float, ...] = (),
    translation_rank: int = 0,
    translation_condition: float | None = None,
) -> ZhuangRothSudhakarResult:
    reasons = {
        "insufficient_poses": "fewer than three train absolute-pose pairs",
        "special_a_scalar_zero": "paper elimination is undefined for an input a0 near zero",
        "degenerate_rotation": "stacked six-variable rotation system is deficient or unstable",
        "special_z_scalar_zero": "paper parameterization is undefined for recovered z0 near zero",
        "inconsistent_quaternion_normalization": (
            "independent quaternion normalizations disagree beyond the declared gate"
        ),
        "degenerate_translation": "conditional translation system is deficient or unstable",
    }
    return ZhuangRothSudhakarResult(
        status=status,
        reason=reasons[status],
        transform_x=None,
        transform_z=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        rotation_singular_values=rotation_spectrum,
        rotation_rank=rotation_rank,
        rotation_condition_number=rotation_condition,
        minimum_abs_a_scalar=minimum_abs_a,
        recovered_abs_z_scalar=recovered_abs_z_scalar,
        quaternion_x_raw_norm=raw_norm_x,
        quaternion_z_raw_norm=raw_norm_z,
        quaternion_normalization_disagreement=normalization_disagreement,
        scalar_reconstruction_rmse=scalar_rmse,
        quaternion_sign_flip_count=sign_flips,
        quaternion_sign_synchronization_fraction=sign_fraction,
        translation_singular_values=translation_spectrum,
        translation_rank=translation_rank,
        translation_condition_number=translation_condition,
        train_evaluation=RobotWorldHandEyeEvaluation(None, None),
        holdout_evaluation=RobotWorldHandEyeEvaluation(None, None),
    )
