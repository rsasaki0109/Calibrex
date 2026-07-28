"""Shah's separable Kronecker solver for robot-world/hand-eye calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.evaluation.holdout import split_indices

FloatArray: TypeAlias = NDArray[np.float64]
RobotWorldHandEyeStatus = Literal[
    "converged",
    "insufficient_poses",
    "degenerate_rotation",
    "degenerate_translation",
]
RobotWorldTransformName = Literal["X", "Y"]
RobotWorldProbeUnit = Literal["m", "deg"]


class RobotWorldHandEyeProbeOptions(Protocol):
    """Structural options required by the shared AX=YB falsification protocol."""

    @property
    def known_bad_translation_m(self) -> float: ...

    @property
    def known_bad_rotation_deg(self) -> float: ...

    @property
    def known_bad_translation_margin_m(self) -> float: ...

    @property
    def known_bad_rotation_margin_deg(self) -> float: ...


@dataclass(frozen=True)
class RobotWorldHandEyePosePair:
    """Synchronized absolute poses satisfying ``pose_a X = Y pose_b``."""

    pair_id: str
    pose_a: SE3
    pose_b: SE3
    weight: float = 1.0


@dataclass(frozen=True)
class ShahRobotWorldHandEyeOptions:
    min_train_poses: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-9
    minimum_rotation_normalized_gap: float = 1.0e-6
    max_translation_condition_number: float = 1.0e8
    max_so3_projection_correction_frobenius: float = 0.25
    known_bad_translation_m: float = 0.05
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_margin_m: float = 0.001
    known_bad_rotation_margin_deg: float = 0.1


@dataclass(frozen=True)
class RobotWorldHandEyeEvaluation:
    rotation_closure_rmse_deg: float | None
    translation_closure_rmse_m: float | None


@dataclass(frozen=True)
class RobotWorldHandEyeProbe:
    transform_name: RobotWorldTransformName
    dof: str
    amount: float
    unit: RobotWorldProbeUnit
    rotation_closure_rmse_deg: float | None
    translation_closure_rmse_m: float | None
    rotation_delta_deg: float | None
    translation_delta_m: float | None
    detectable: bool | None


@dataclass(frozen=True)
class ShahRobotWorldHandEyeResult:
    status: RobotWorldHandEyeStatus
    reason: str
    transform_x: SE3 | None
    transform_y: SE3 | None
    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    rotation_singular_values: tuple[float, ...]
    rotation_normalized_gap: float | None
    rotation_dominant_multiplicity: int
    rotation_x_projection_correction_frobenius: float | None
    rotation_y_projection_correction_frobenius: float | None
    translation_singular_values: tuple[float, ...]
    translation_rank: int
    translation_condition_number: float | None
    train_evaluation: RobotWorldHandEyeEvaluation
    holdout_evaluation: RobotWorldHandEyeEvaluation
    probes: tuple[RobotWorldHandEyeProbe, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return the complete schema-friendly solution and evidence contract."""

        return {
            "status": self.status,
            "reason": self.reason,
            "transform_x": self.transform_x.as_dict() if self.transform_x else None,
            "transform_y": self.transform_y.as_dict() if self.transform_y else None,
            "train_pair_ids": list(self.train_pair_ids),
            "holdout_pair_ids": list(self.holdout_pair_ids),
            "rotation_singular_values": list(self.rotation_singular_values),
            "rotation_normalized_gap": self.rotation_normalized_gap,
            "rotation_dominant_multiplicity": self.rotation_dominant_multiplicity,
            "rotation_x_projection_correction_frobenius": (
                self.rotation_x_projection_correction_frobenius
            ),
            "rotation_y_projection_correction_frobenius": (
                self.rotation_y_projection_correction_frobenius
            ),
            "translation_singular_values": list(self.translation_singular_values),
            "translation_rank": self.translation_rank,
            "translation_condition_number": self.translation_condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "shah_separable_robot_world_hand_eye_kronecker/v0.1",
            "paper": {
                "title": (
                    "Solving the Robot-World/Hand-Eye Calibration Problem Using "
                    "the Kronecker Product"
                ),
                "author": "Mili I. Shah",
                "doi": "10.1115/1.4024473",
            },
            "equation": "A_j X = Y B_j",
            "rotation_contract": (
                "dominant singular vectors of sum_j weight_j (R_Bj kron R_Aj); "
                "determinant normalization followed by nearest-SO(3) projection"
            ),
            "translation_contract": (
                "weighted least squares [I, -R_Aj] [t_Y; t_X] = t_Aj - R_Y t_Bj"
            ),
            "evaluation_contract": "closure on held-out absolute pose pairs without refitting",
        }


@dataclass(frozen=True)
class _ShahRotationSolution:
    rotation_x: FloatArray | None
    rotation_y: FloatArray | None
    singular_values: tuple[float, ...]
    normalized_gap: float | None
    dominant_multiplicity: int
    projection_x: float | None
    projection_y: float | None


class ShahRobotWorldHandEyeSolver:
    """Solve ``A_j X = Y B_j`` using Shah's separable Kronecker construction."""

    def solve(
        self,
        poses: Sequence[RobotWorldHandEyePosePair],
        options: ShahRobotWorldHandEyeOptions | None = None,
    ) -> ShahRobotWorldHandEyeResult:
        solver_options = options or ShahRobotWorldHandEyeOptions()
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

        rotation_solution = _solve_rotations(train, solver_options)
        if rotation_solution.rotation_x is None or rotation_solution.rotation_y is None:
            return _empty(
                "degenerate_rotation",
                train_ids,
                holdout_ids,
                rotation_spectrum=rotation_solution.singular_values,
                rotation_gap=rotation_solution.normalized_gap,
                dominant_multiplicity=rotation_solution.dominant_multiplicity,
                projection_x=rotation_solution.projection_x,
                projection_y=rotation_solution.projection_y,
            )
        rotation_x = rotation_solution.rotation_x
        rotation_y = rotation_solution.rotation_y
        translation_solution = _solve_translations(train, rotation_y, solver_options)
        if translation_solution[0] is None:
            return _empty(
                "degenerate_translation",
                train_ids,
                holdout_ids,
                rotation_spectrum=rotation_solution.singular_values,
                rotation_gap=rotation_solution.normalized_gap,
                dominant_multiplicity=rotation_solution.dominant_multiplicity,
                projection_x=rotation_solution.projection_x,
                projection_y=rotation_solution.projection_y,
                translation_spectrum=translation_solution[2],
                translation_rank=translation_solution[3],
                translation_condition=translation_solution[4],
            )
        translation_yx, _matrix, translation_spectrum, translation_rank, condition = (
            translation_solution
        )
        assert translation_yx is not None
        transform_y = _se3(rotation_y, translation_yx[:3])
        transform_x = _se3(rotation_x, translation_yx[3:])
        train_evaluation = evaluate_robot_world_hand_eye_poses(train, transform_x, transform_y)
        holdout_evaluation = evaluate_robot_world_hand_eye_poses(holdout, transform_x, transform_y)
        probes = evaluate_robot_world_hand_eye_known_bad_probes(
            holdout,
            transform_x,
            transform_y,
            holdout_evaluation,
            solver_options,
        )
        return ShahRobotWorldHandEyeResult(
            status="converged",
            reason="Shah rotation and conditional translation systems solved",
            transform_x=transform_x,
            transform_y=transform_y,
            train_pair_ids=train_ids,
            holdout_pair_ids=holdout_ids,
            rotation_singular_values=rotation_solution.singular_values,
            rotation_normalized_gap=rotation_solution.normalized_gap,
            rotation_dominant_multiplicity=rotation_solution.dominant_multiplicity,
            rotation_x_projection_correction_frobenius=rotation_solution.projection_x,
            rotation_y_projection_correction_frobenius=rotation_solution.projection_y,
            translation_singular_values=translation_spectrum,
            translation_rank=translation_rank,
            translation_condition_number=condition,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=probes,
        )


def evaluate_robot_world_hand_eye_poses(
    poses: Sequence[RobotWorldHandEyePosePair],
    transform_x: SE3,
    transform_y: SE3,
) -> RobotWorldHandEyeEvaluation:
    """Evaluate held-out ``A X`` versus ``Y B`` closure without refitting."""

    if not poses:
        return RobotWorldHandEyeEvaluation(None, None)
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for pair in poses:
        left = pair.pose_a.compose(transform_x)
        right = transform_y.compose(pair.pose_b)
        delta = left.inverse().compose(right)
        rotation_errors.append(_rotation_angle_deg(delta))
        translation_errors.append(float(np.linalg.norm(delta.translation_m)))
    return RobotWorldHandEyeEvaluation(
        rotation_closure_rmse_deg=_rmse(rotation_errors),
        translation_closure_rmse_m=_rmse(translation_errors),
    )


def evaluate_robot_world_hand_eye_known_bad_probes(
    holdout: Sequence[RobotWorldHandEyePosePair],
    transform_x: SE3,
    transform_y: SE3,
    baseline: RobotWorldHandEyeEvaluation,
    options: RobotWorldHandEyeProbeOptions,
) -> tuple[RobotWorldHandEyeProbe, ...]:
    """Apply signed six-DoF falsification controls independently to X and Y."""

    probes: list[RobotWorldHandEyeProbe] = []
    for transform_name in ("X", "Y"):
        for dof_index, dof in enumerate(("x", "y", "z")):
            for sign in (-1.0, 1.0):
                amount = sign * options.known_bad_translation_m
                translation = [0.0, 0.0, 0.0]
                translation[dof_index] = amount
                probes.append(
                    _probe(
                        transform_name,
                        dof,
                        amount,
                        "m",
                        SE3(
                            (translation[0], translation[1], translation[2]),
                            (0.0, 0.0, 0.0, 1.0),
                        ),
                        holdout,
                        transform_x,
                        transform_y,
                        baseline,
                        options,
                    )
                )
        for dof_index, dof in enumerate(("roll", "pitch", "yaw")):
            for sign in (-1.0, 1.0):
                amount = sign * options.known_bad_rotation_deg
                half = math.radians(amount) / 2.0
                quaternion = [0.0, 0.0, 0.0, math.cos(half)]
                quaternion[dof_index] = math.sin(half)
                probes.append(
                    _probe(
                        transform_name,
                        dof,
                        amount,
                        "deg",
                        SE3(
                            (0.0, 0.0, 0.0),
                            (quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
                        ),
                        holdout,
                        transform_x,
                        transform_y,
                        baseline,
                        options,
                    )
                )
    return tuple(probes)


def _solve_rotations(
    poses: Sequence[RobotWorldHandEyePosePair],
    options: ShahRobotWorldHandEyeOptions,
) -> _ShahRotationSolution:
    cross_covariance: FloatArray = np.zeros((9, 9), dtype=np.float64)
    for pair in poses:
        cross_covariance += pair.weight * np.kron(
            _rotation_matrix(pair.pose_b), _rotation_matrix(pair.pose_a)
        )
    left, singular, right_t = np.linalg.svd(cross_covariance)
    spectrum = tuple(float(value) for value in singular)
    if not spectrum or spectrum[0] <= 0.0:
        return _ShahRotationSolution(None, None, spectrum, None, 0, None, None)
    threshold = options.rank_tolerance * spectrum[0]
    dominant_multiplicity = int(np.count_nonzero(singular >= spectrum[0] - threshold))
    gap = (spectrum[0] - spectrum[1]) / spectrum[0]
    if dominant_multiplicity != 1 or gap < options.minimum_rotation_normalized_gap:
        return _ShahRotationSolution(None, None, spectrum, gap, dominant_multiplicity, None, None)
    normalized_y = _determinant_normalized(left[:, 0].reshape((3, 3), order="F"))
    normalized_x = _determinant_normalized(right_t[0].reshape((3, 3), order="F"))
    if normalized_x is None or normalized_y is None:
        return _ShahRotationSolution(None, None, spectrum, gap, dominant_multiplicity, None, None)
    rotation_x = _nearest_rotation(normalized_x)
    rotation_y = _nearest_rotation(normalized_y)
    correction_x = float(np.linalg.norm(normalized_x - rotation_x, ord="fro"))
    correction_y = float(np.linalg.norm(normalized_y - rotation_y, ord="fro"))
    if max(correction_x, correction_y) > options.max_so3_projection_correction_frobenius:
        return _ShahRotationSolution(
            None,
            None,
            spectrum,
            gap,
            dominant_multiplicity,
            correction_x,
            correction_y,
        )
    return _ShahRotationSolution(
        rotation_x,
        rotation_y,
        spectrum,
        gap,
        dominant_multiplicity,
        correction_x,
        correction_y,
    )


def _solve_translations(
    poses: Sequence[RobotWorldHandEyePosePair],
    rotation_y: FloatArray,
    options: ShahRobotWorldHandEyeOptions,
) -> tuple[FloatArray | None, FloatArray, tuple[float, ...], int, float]:
    rows: list[FloatArray] = []
    rhs: list[FloatArray] = []
    identity: FloatArray = np.eye(3, dtype=np.float64)
    for pair in poses:
        root_weight = math.sqrt(pair.weight)
        rotation_a = _rotation_matrix(pair.pose_a)
        rows.append(root_weight * np.hstack((identity, -rotation_a)))
        rhs.append(
            root_weight
            * (
                np.asarray(pair.pose_a.translation_m)
                - rotation_y @ np.asarray(pair.pose_b.translation_m)
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
        return None, matrix, spectrum, rank, condition
    solution, _residuals, _rank, _values = np.linalg.lstsq(matrix, vector, rcond=None)
    return solution, matrix, spectrum, rank, condition


def _probe(
    transform_name: RobotWorldTransformName,
    dof: str,
    amount: float,
    unit: RobotWorldProbeUnit,
    delta: SE3,
    holdout: Sequence[RobotWorldHandEyePosePair],
    transform_x: SE3,
    transform_y: SE3,
    baseline: RobotWorldHandEyeEvaluation,
    options: RobotWorldHandEyeProbeOptions,
) -> RobotWorldHandEyeProbe:
    candidate_x = delta.compose(transform_x) if transform_name == "X" else transform_x
    candidate_y = delta.compose(transform_y) if transform_name == "Y" else transform_y
    evaluation = evaluate_robot_world_hand_eye_poses(holdout, candidate_x, candidate_y)
    rotation_delta = _difference(
        evaluation.rotation_closure_rmse_deg, baseline.rotation_closure_rmse_deg
    )
    translation_delta = _difference(
        evaluation.translation_closure_rmse_m, baseline.translation_closure_rmse_m
    )
    detectable = (
        rotation_delta > options.known_bad_rotation_margin_deg
        or translation_delta > options.known_bad_translation_margin_m
        if rotation_delta is not None and translation_delta is not None
        else None
    )
    return RobotWorldHandEyeProbe(
        transform_name=transform_name,
        dof=dof,
        amount=amount,
        unit=unit,
        rotation_closure_rmse_deg=evaluation.rotation_closure_rmse_deg,
        translation_closure_rmse_m=evaluation.translation_closure_rmse_m,
        rotation_delta_deg=rotation_delta,
        translation_delta_m=translation_delta,
        detectable=detectable,
    )


def _validated_poses(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[RobotWorldHandEyePosePair, ...]:
    names: set[str] = set()
    output: list[RobotWorldHandEyePosePair] = []
    for pair in sorted(poses, key=lambda item: item.pair_id):
        if not pair.pair_id or pair.pair_id in names:
            raise ValueError("robot-world/hand-eye pair IDs must be unique and non-empty")
        if not math.isfinite(pair.weight) or pair.weight <= 0.0:
            raise ValueError("robot-world/hand-eye weights must be finite and positive")
        names.add(pair.pair_id)
        output.append(pair)
    return tuple(output)


def _validate_options(options: ShahRobotWorldHandEyeOptions) -> None:
    if options.min_train_poses < 3:
        raise ValueError("Shah robot-world/hand-eye requires at least three train poses")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("robot-world/hand-eye holdout ratio must be in [0, 1)")
    positive = (
        options.rank_tolerance,
        options.minimum_rotation_normalized_gap,
        options.max_translation_condition_number,
        options.max_so3_projection_correction_frobenius,
        options.known_bad_translation_m,
        options.known_bad_rotation_deg,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("robot-world/hand-eye numerical options must be finite and positive")
    margins = (
        options.known_bad_translation_margin_m,
        options.known_bad_rotation_margin_deg,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in margins):
        raise ValueError("robot-world/hand-eye probe margins must be finite and non-negative")


def _determinant_normalized(matrix: FloatArray) -> FloatArray | None:
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-15:
        return None
    return matrix / float(np.cbrt(determinant))


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


def _rotation_angle_deg(transform: SE3) -> float:
    _x, _y, _z, w = transform.rotation_quat_xyzw
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, abs(w)))))


def _rmse(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _difference(value: float | None, baseline: float | None) -> float | None:
    return value - baseline if value is not None and baseline is not None else None


def _empty(
    status: RobotWorldHandEyeStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    rotation_spectrum: tuple[float, ...] = (),
    rotation_gap: float | None = None,
    dominant_multiplicity: int = 0,
    projection_x: float | None = None,
    projection_y: float | None = None,
    translation_spectrum: tuple[float, ...] = (),
    translation_rank: int = 0,
    translation_condition: float | None = None,
) -> ShahRobotWorldHandEyeResult:
    reasons = {
        "insufficient_poses": "fewer than three train absolute-pose pairs",
        "degenerate_rotation": "dominant Kronecker rotation direction is not unique or stable",
        "degenerate_translation": "conditional six-dimensional translation system is degenerate",
    }
    return ShahRobotWorldHandEyeResult(
        status=status,
        reason=reasons.get(status, "robot-world/hand-eye solve did not produce a transform"),
        transform_x=None,
        transform_y=None,
        train_pair_ids=train_ids,
        holdout_pair_ids=holdout_ids,
        rotation_singular_values=rotation_spectrum,
        rotation_normalized_gap=rotation_gap,
        rotation_dominant_multiplicity=dominant_multiplicity,
        rotation_x_projection_correction_frobenius=projection_x,
        rotation_y_projection_correction_frobenius=projection_y,
        translation_singular_values=translation_spectrum,
        translation_rank=translation_rank,
        translation_condition_number=translation_condition,
        train_evaluation=RobotWorldHandEyeEvaluation(None, None),
        holdout_evaluation=RobotWorldHandEyeEvaluation(None, None),
    )
