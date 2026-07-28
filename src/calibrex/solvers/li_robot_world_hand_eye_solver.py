"""Li-Wang-Wu simultaneous Kronecker robot-world/hand-eye calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyeEvaluation,
    RobotWorldHandEyePosePair,
    RobotWorldHandEyeProbe,
    evaluate_robot_world_hand_eye_known_bad_probes,
    evaluate_robot_world_hand_eye_poses,
)

FloatArray: TypeAlias = NDArray[np.float64]
LiRobotWorldHandEyeStatus = Literal[
    "converged",
    "insufficient_poses",
    "degenerate_linear_system",
    "unstable_rotation_projection",
]


@dataclass(frozen=True)
class LiRobotWorldHandEyeOptions:
    min_train_poses: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-9
    max_condition_number: float = 1.0e8
    max_so3_projection_correction_frobenius: float = 0.05
    known_bad_translation_m: float = 0.05
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_margin_m: float = 0.001
    known_bad_rotation_margin_deg: float = 0.1


@dataclass(frozen=True)
class LiRobotWorldHandEyeResult:
    status: LiRobotWorldHandEyeStatus
    reason: str
    transform_x: SE3 | None
    transform_z: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    linear_singular_values: tuple[float, ...]
    linear_rank: int
    linear_condition_number: float | None
    raw_linear_residual_rmse: float | None
    raw_rotation_x_determinant: float | None
    raw_rotation_z_determinant: float | None
    rotation_x_projection_correction_frobenius: float | None
    rotation_z_projection_correction_frobenius: float | None
    train_evaluation: RobotWorldHandEyeEvaluation
    holdout_evaluation: RobotWorldHandEyeEvaluation
    probes: tuple[RobotWorldHandEyeProbe, ...] = ()

    @property
    def transform_y(self) -> SE3 | None:
        """Expose Li's paper notation ``Z`` through the shared ``Y`` role."""

        return self.transform_z

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly simultaneous-solver evidence and provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "transform_z": self.transform_z.as_dict() if self.transform_z else None,
            "shared_role_alias": "Y := Z",
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "linear_singular_values": list(self.linear_singular_values),
            "linear_rank": self.linear_rank,
            "linear_condition_number": self.linear_condition_number,
            "raw_linear_residual_rmse": self.raw_linear_residual_rmse,
            "raw_rotation_x_determinant": self.raw_rotation_x_determinant,
            "raw_rotation_z_determinant": self.raw_rotation_z_determinant,
            "rotation_x_projection_correction_frobenius": (
                self.rotation_x_projection_correction_frobenius
            ),
            "rotation_z_projection_correction_frobenius": (
                self.rotation_z_projection_correction_frobenius
            ),
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "li_wang_wu_simultaneous_robot_world_hand_eye_kronecker/v0.1",
            "paper": {
                "title": (
                    "Simultaneous robot-world and hand-eye calibration using "
                    "dual-quaternions and Kronecker product"
                ),
                "authors": "Aiguo Li, Lin Wang, and Defeng Wu",
                "doi": "10.5897/IJPS.9000501",
            },
            "equation": "A_i X = Z B_i",
            "linear_contract": (
                "paper equations (17)-(19), row-major stack: one weighted least-squares "
                "system for vec(R_X), vec(R_Z), t_X, and t_Z"
            ),
            "orthogonalization_contract": (
                "nearest-SO(3) projection after the simultaneous linear solve"
            ),
            "translation_recomputed_after_rotation_projection": False,
            "limitation": (
                "translations remain the simultaneous raw-linear estimates after rotation "
                "projection, matching the paper method and preserving projection mismatch"
            ),
            "evaluation_contract": "closure on held-out absolute pose pairs without refitting",
        }


class LiRobotWorldHandEyeSolver:
    """Solve Li et al.'s simultaneous 24-variable Kronecker system."""

    def solve(
        self,
        poses: Sequence[RobotWorldHandEyePosePair],
        options: LiRobotWorldHandEyeOptions | None = None,
    ) -> LiRobotWorldHandEyeResult:
        solver_options = options or LiRobotWorldHandEyeOptions()
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

        matrix, rhs = _linear_system(train)
        singular = np.linalg.svd(matrix, compute_uv=False)
        spectrum = tuple(float(value) for value in singular)
        threshold = solver_options.rank_tolerance * spectrum[0]
        rank = int(np.count_nonzero(singular > threshold))
        condition = spectrum[0] / spectrum[-1] if spectrum[-1] > 0.0 else math.inf
        if rank < 24 or condition > solver_options.max_condition_number:
            return _empty(
                "degenerate_linear_system",
                train_ids,
                holdout_ids,
                spectrum=spectrum,
                rank=rank,
                condition=condition,
            )
        solution, _residuals, _rank, _values = np.linalg.lstsq(matrix, rhs, rcond=None)
        raw_residual = float(np.sqrt(np.mean(np.square(matrix @ solution - rhs))))
        raw_rotation_x = solution[:9].reshape((3, 3))
        raw_rotation_z = solution[9:18].reshape((3, 3))
        rotation_x = _nearest_rotation(raw_rotation_x)
        rotation_z = _nearest_rotation(raw_rotation_z)
        determinant_x = float(np.linalg.det(raw_rotation_x))
        determinant_z = float(np.linalg.det(raw_rotation_z))
        correction_x = float(np.linalg.norm(raw_rotation_x - rotation_x, ord="fro"))
        correction_z = float(np.linalg.norm(raw_rotation_z - rotation_z, ord="fro"))
        if (
            not math.isfinite(determinant_x)
            or not math.isfinite(determinant_z)
            or max(correction_x, correction_z)
            > solver_options.max_so3_projection_correction_frobenius
        ):
            return _empty(
                "unstable_rotation_projection",
                train_ids,
                holdout_ids,
                spectrum=spectrum,
                rank=rank,
                condition=condition,
                raw_residual=raw_residual,
                determinant_x=determinant_x,
                determinant_z=determinant_z,
                correction_x=correction_x,
                correction_z=correction_z,
            )
        transform_x = _se3(rotation_x, solution[18:21])
        transform_z = _se3(rotation_z, solution[21:24])
        train_evaluation = evaluate_robot_world_hand_eye_poses(train, transform_x, transform_z)
        holdout_evaluation = evaluate_robot_world_hand_eye_poses(holdout, transform_x, transform_z)
        probes = evaluate_robot_world_hand_eye_known_bad_probes(
            holdout,
            transform_x,
            transform_z,
            holdout_evaluation,
            solver_options,
        )
        return LiRobotWorldHandEyeResult(
            status="converged",
            reason="Li-Wang-Wu simultaneous Kronecker system solved and rotations projected",
            transform_x=transform_x,
            transform_z=transform_z,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            linear_singular_values=spectrum,
            linear_rank=rank,
            linear_condition_number=condition,
            raw_linear_residual_rmse=raw_residual,
            raw_rotation_x_determinant=determinant_x,
            raw_rotation_z_determinant=determinant_z,
            rotation_x_projection_correction_frobenius=correction_x,
            rotation_z_projection_correction_frobenius=correction_z,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=probes,
        )


def _linear_system(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[FloatArray, FloatArray]:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    identity_3: FloatArray = np.eye(3, dtype=np.float64)
    zeros_rotation_translation: FloatArray = np.zeros((9, 6), dtype=np.float64)
    zeros_translation_rotation: FloatArray = np.zeros((3, 9), dtype=np.float64)
    for pair in poses:
        root_weight = math.sqrt(pair.weight)
        rotation_a = _rotation_matrix(pair.pose_a)
        rotation_b = _rotation_matrix(pair.pose_b)
        translation_a = np.asarray(pair.pose_a.translation_m)
        translation_b = np.asarray(pair.pose_b.translation_m)
        rotation_rows = np.hstack(
            (
                np.kron(rotation_a, identity_3),
                -np.kron(identity_3, rotation_b.T),
                zeros_rotation_translation,
            )
        )
        translation_rows = np.hstack(
            (
                zeros_translation_rotation,
                np.kron(identity_3, translation_b.reshape((1, 3))),
                -rotation_a,
                identity_3,
            )
        )
        rows.extend((root_weight * rotation_rows, root_weight * translation_rows))
        rhs.extend((np.zeros(9, dtype=np.float64), root_weight * translation_a))
    return np.vstack(rows), np.concatenate(rhs)


def _validated_poses(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[RobotWorldHandEyePosePair, ...]:
    names: set[str] = set()
    output: list[RobotWorldHandEyePosePair] = []
    for pair in sorted(poses, key=lambda item: item.pair_id):
        if not pair.pair_id or pair.pair_id in names:
            raise ValueError("Li robot-world/hand-eye pair IDs must be unique and non-empty")
        if not math.isfinite(pair.weight) or pair.weight <= 0.0:
            raise ValueError("Li robot-world/hand-eye weights must be finite and positive")
        names.add(pair.pair_id)
        output.append(pair)
    return tuple(output)


def _validate_options(options: LiRobotWorldHandEyeOptions) -> None:
    if options.min_train_poses < 3:
        raise ValueError("Li robot-world/hand-eye requires at least three train poses")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("Li robot-world/hand-eye holdout ratio must be in [0, 1)")
    positive = (
        options.rank_tolerance,
        options.max_condition_number,
        options.max_so3_projection_correction_frobenius,
        options.known_bad_translation_m,
        options.known_bad_rotation_deg,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("Li robot-world/hand-eye numerical options must be finite and positive")
    margins = (
        options.known_bad_translation_margin_m,
        options.known_bad_rotation_margin_deg,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in margins):
        raise ValueError("Li robot-world/hand-eye probe margins must be finite and non-negative")


def _nearest_rotation(matrix: FloatArray) -> FloatArray:
    left, _singular, right_t = np.linalg.svd(matrix)
    correction: FloatArray = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(left @ right_t)
    return left @ correction @ right_t


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


def _se3(rotation: FloatArray, translation: FloatArray) -> SE3:
    return SE3(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
    )


def _empty(
    status: LiRobotWorldHandEyeStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    spectrum: tuple[float, ...] = (),
    rank: int = 0,
    condition: float | None = None,
    raw_residual: float | None = None,
    determinant_x: float | None = None,
    determinant_z: float | None = None,
    correction_x: float | None = None,
    correction_z: float | None = None,
) -> LiRobotWorldHandEyeResult:
    reasons = {
        "insufficient_poses": "fewer than three train absolute-pose pairs",
        "degenerate_linear_system": "simultaneous 24-variable system is rank deficient",
        "unstable_rotation_projection": "raw linear rotations require excessive SO(3) correction",
    }
    return LiRobotWorldHandEyeResult(
        status=status,
        reason=reasons.get(status, "Li robot-world/hand-eye solve did not produce transforms"),
        transform_x=None,
        transform_z=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        linear_singular_values=spectrum,
        linear_rank=rank,
        linear_condition_number=condition,
        raw_linear_residual_rmse=raw_residual,
        raw_rotation_x_determinant=determinant_x,
        raw_rotation_z_determinant=determinant_z,
        rotation_x_projection_correction_frobenius=correction_x,
        rotation_z_projection_correction_frobenius=correction_z,
        train_evaluation=RobotWorldHandEyeEvaluation(None, None),
        holdout_evaluation=RobotWorldHandEyeEvaluation(None, None),
    )
