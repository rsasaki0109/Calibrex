"""Camera-LiDAR calibration from corresponding board centres and normals.

This is a ROS-independent implementation of the measurement features proposed
by Verma et al. (ITSC 2019). It uses a deterministic robust Procrustes solver,
not code or the genetic optimizer from the paper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import (
    SE3,
    Vector3,
    quaternion_xyzw_from_rotation_matrix,
    rotate_vector_xyzw,
)
from calibrex.evaluation.holdout import split_indices

FloatArray: TypeAlias = NDArray[np.float64]
PointPlaneStatus = Literal[
    "converged",
    "insufficient_observations",
    "degenerate_geometry",
    "max_iterations",
]


@dataclass(frozen=True)
class PointPlaneObservation:
    """Corresponding checkerboard centre and oriented normal in both frames."""

    frame_id: str
    camera_center_m: Vector3
    camera_normal: Vector3
    lidar_center_m: Vector3
    lidar_normal: Vector3
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        object.__setattr__(
            self, "camera_center_m", _finite_vector3(self.camera_center_m, "camera centre")
        )
        object.__setattr__(self, "camera_normal", _unit(self.camera_normal))
        object.__setattr__(
            self, "lidar_center_m", _finite_vector3(self.lidar_center_m, "LiDAR centre")
        )
        object.__setattr__(self, "lidar_normal", _unit(self.lidar_normal))
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("observation weight must be finite and non-negative")


@dataclass(frozen=True)
class PointPlaneSolverOptions:
    min_train_observations: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    normal_scale_m: float = 1.0
    huber_delta_m: float = 0.05
    max_iterations: int = 20
    convergence_tolerance: float = 1.0e-10
    rank_tolerance: float = 1.0e-7
    max_condition_number: float = 1.0e6
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_normal_margin_deg: float = 0.1
    known_bad_center_margin_m: float = 0.005


@dataclass(frozen=True)
class PointPlaneEvaluation:
    center_rmse_m: float | None
    normal_rmse_deg: float | None
    plane_offset_rmse_m: float | None


@dataclass(frozen=True)
class PointPlaneProbe:
    dof: Literal["x", "y", "z", "roll", "pitch", "yaw"]
    amount: float
    unit: Literal["m", "deg"]
    center_rmse_m: float | None
    normal_rmse_deg: float | None
    center_delta_m: float | None
    normal_delta_deg: float | None
    detectable: bool | None


@dataclass(frozen=True)
class PointPlaneLidarCameraResult:
    status: PointPlaneStatus
    reason: str
    transform_camera_lidar: SE3 | None
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    joint_singular_values: tuple[float, float, float, float, float, float] | None
    joint_rank: int
    joint_condition_number: float | None
    weak_joint_direction: tuple[float, float, float, float, float, float] | None
    train_evaluation: PointPlaneEvaluation
    holdout_evaluation: PointPlaneEvaluation
    probes: tuple[PointPlaneProbe, ...]
    iterations: int
    solver_options: PointPlaneSolverOptions

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "transform_camera_lidar": (
                self.transform_camera_lidar.as_dict()
                if self.transform_camera_lidar is not None
                else None
            ),
            "train_frame_ids": list(self.train_frame_ids),
            "holdout_frame_ids": list(self.holdout_frame_ids),
            "joint_singular_values": self.joint_singular_values,
            "joint_rank": self.joint_rank,
            "joint_condition_number": self.joint_condition_number,
            "weak_joint_direction": self.weak_joint_direction,
            "joint_parameter_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
            "train_evaluation": asdict(self.train_evaluation),
            "holdout_evaluation": asdict(self.holdout_evaluation),
            "known_bad_probes": [asdict(probe) for probe in self.probes],
            "iterations": self.iterations,
            "solver_options": asdict(self.solver_options),
            "method": "verma_center_normal_procrustes_irls/v0.1",
            "paper_doi": "10.1109/ITSC.2019.8917108",
            "paper_arxiv": "1904.12433",
            "frame_convention": "p_camera = R_camera_lidar p_lidar + t_camera_lidar",
            "normal_sign_policy": "preserve_oriented_correspondences",
            "implementation_note": (
                "independent weighted Procrustes/IRLS baseline; the paper's genetic "
                "optimizer and source code are not copied"
            ),
        }


class PointPlaneLidarCameraSolver:
    """Fit one SE(3) to board-centre and board-normal correspondences."""

    def solve(
        self,
        observations: Sequence[PointPlaneObservation],
        options: PointPlaneSolverOptions | None = None,
    ) -> PointPlaneLidarCameraResult:
        solver_options = options or PointPlaneSolverOptions()
        _validate_options(solver_options)
        usable = sorted(
            (item for item in observations if item.weight > 0.0),
            key=lambda item: item.frame_id,
        )
        train_indices, holdout_indices = split_indices(
            len(usable), solver_options.holdout_ratio, solver_options.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        train_ids = tuple(item.frame_id for item in train)
        holdout_ids = tuple(item.frame_id for item in holdout)
        if len(train) < solver_options.min_train_observations:
            return _empty("insufficient_observations", train_ids, holdout_ids, solver_options)

        robust_weights = np.ones(len(train), dtype=np.float64)
        transform: SE3 | None = None
        iterations = 0
        status: PointPlaneStatus = "max_iterations"
        for iteration in range(1, solver_options.max_iterations + 1):
            iterations = iteration
            candidate = _weighted_fit(
                train,
                robust_weights,
                solver_options.normal_scale_m,
            )
            if candidate is None:
                return _empty("degenerate_geometry", train_ids, holdout_ids, solver_options)
            if transform is not None and _transform_delta(candidate, transform) <= (
                solver_options.convergence_tolerance
            ):
                transform = candidate
                status = "converged"
                break
            transform = candidate
            residuals = np.asarray(
                [
                    _capture_residual_m(item, transform, solver_options.normal_scale_m)
                    for item in train
                ],
                dtype=np.float64,
            )
            robust_weights = np.minimum(
                1.0,
                solver_options.huber_delta_m / np.maximum(residuals, 1.0e-15),
            )
        if transform is None:
            return _empty("degenerate_geometry", train_ids, holdout_ids, solver_options)

        joint = _joint_observability(train, transform, robust_weights, solver_options)
        if (
            joint.rank < 6
            or joint.condition_number is None
            or joint.condition_number > solver_options.max_condition_number
        ):
            return PointPlaneLidarCameraResult(
                status="degenerate_geometry",
                reason="centre/normal Jacobian is rank deficient or ill-conditioned",
                transform_camera_lidar=transform,
                train_frame_ids=train_ids,
                holdout_frame_ids=holdout_ids,
                joint_singular_values=joint.singular_values,
                joint_rank=joint.rank,
                joint_condition_number=joint.condition_number,
                weak_joint_direction=joint.weak_direction,
                train_evaluation=evaluate_point_plane_observations(train, transform),
                holdout_evaluation=evaluate_point_plane_observations(holdout, transform),
                probes=(),
                iterations=iterations,
                solver_options=solver_options,
            )
        train_evaluation = evaluate_point_plane_observations(train, transform)
        holdout_evaluation = evaluate_point_plane_observations(holdout, transform)
        probes = _known_bad_probes(holdout, transform, holdout_evaluation, solver_options)
        return PointPlaneLidarCameraResult(
            status=status,
            reason=(
                "robust centre/normal alignment converged"
                if status == "converged"
                else "robust centre/normal alignment reached the iteration limit"
            ),
            transform_camera_lidar=transform,
            train_frame_ids=train_ids,
            holdout_frame_ids=holdout_ids,
            joint_singular_values=joint.singular_values,
            joint_rank=joint.rank,
            joint_condition_number=joint.condition_number,
            weak_joint_direction=joint.weak_direction,
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=probes,
            iterations=iterations,
            solver_options=solver_options,
        )


def evaluate_point_plane_observations(
    observations: Sequence[PointPlaneObservation], transform: SE3
) -> PointPlaneEvaluation:
    """Evaluate unchanged centre, normal, and induced plane-offset closure."""

    center_squared: list[float] = []
    normal_squared: list[float] = []
    offset_squared: list[float] = []
    for item in observations:
        predicted_center = transform.transform_point(item.lidar_center_m)
        center_squared.append(_squared_distance(predicted_center, item.camera_center_m))
        predicted_normal = rotate_vector_xyzw(transform.rotation_quat_xyzw, item.lidar_normal)
        angle = _oriented_angle(predicted_normal, item.camera_normal)
        normal_squared.append(angle * angle)
        lidar_offset = -_dot(item.lidar_normal, item.lidar_center_m)
        camera_offset = -_dot(item.camera_normal, item.camera_center_m)
        predicted_offset = lidar_offset - _dot(predicted_normal, transform.translation_m)
        offset_squared.append((predicted_offset - camera_offset) ** 2)
    normal_root = _root_mean(normal_squared)
    return PointPlaneEvaluation(
        center_rmse_m=_root_mean(center_squared),
        normal_rmse_deg=math.degrees(normal_root) if normal_root is not None else None,
        plane_offset_rmse_m=_root_mean(offset_squared),
    )


@dataclass(frozen=True)
class _JointObservability:
    singular_values: tuple[float, float, float, float, float, float]
    rank: int
    condition_number: float | None
    weak_direction: tuple[float, float, float, float, float, float]


def _weighted_fit(
    observations: Sequence[PointPlaneObservation],
    robust_weights: FloatArray,
    normal_scale_m: float,
) -> SE3 | None:
    weights = np.asarray(
        [item.weight * robust for item, robust in zip(observations, robust_weights, strict=True)]
    )
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    camera_centers = np.asarray([item.camera_center_m for item in observations])
    lidar_centers = np.asarray([item.lidar_center_m for item in observations])
    camera_mean = np.sum(camera_centers * weights[:, None], axis=0) / total
    lidar_mean = np.sum(lidar_centers * weights[:, None], axis=0) / total
    covariance = np.zeros((3, 3), dtype=np.float64)
    for index, item in enumerate(observations):
        covariance += weights[index] * np.outer(
            camera_centers[index] - camera_mean,
            lidar_centers[index] - lidar_mean,
        )
        covariance += (
            weights[index] * normal_scale_m**2 * np.outer(item.camera_normal, item.lidar_normal)
        )
    left, _spectrum, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    rotation = left @ correction @ right_t
    translation = camera_mean - rotation @ lidar_mean
    return SE3(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
    )


def _joint_observability(
    observations: Sequence[PointPlaneObservation],
    transform: SE3,
    robust_weights: FloatArray,
    options: PointPlaneSolverOptions,
) -> _JointObservability:
    rotation = _rotation_matrix(transform)
    rows: list[FloatArray] = []
    for item, robust in zip(observations, robust_weights, strict=True):
        root_weight = math.sqrt(item.weight * float(robust))
        point = rotation @ np.asarray(item.lidar_center_m)
        point_jacobian = np.hstack((-_skew(point), np.eye(3)))
        normal = rotation @ np.asarray(item.lidar_normal)
        normal_jacobian = np.hstack((-options.normal_scale_m * _skew(normal), np.zeros((3, 3))))
        rows.extend(root_weight * row for row in point_jacobian)
        rows.extend(root_weight * row for row in normal_jacobian)
    matrix = np.asarray(rows)
    _u, spectrum, vh = np.linalg.svd(matrix, full_matrices=False)
    maximum = float(spectrum[0]) if len(spectrum) else 0.0
    threshold = options.rank_tolerance * maximum
    rank = int(np.count_nonzero(spectrum > threshold))
    condition = float(spectrum[0] / spectrum[-1]) if spectrum[-1] > threshold else None
    values = (
        float(spectrum[0]),
        float(spectrum[1]),
        float(spectrum[2]),
        float(spectrum[3]),
        float(spectrum[4]),
        float(spectrum[5]),
    )
    weak = (
        float(vh[-1, 0]),
        float(vh[-1, 1]),
        float(vh[-1, 2]),
        float(vh[-1, 3]),
        float(vh[-1, 4]),
        float(vh[-1, 5]),
    )
    return _JointObservability(values, rank, condition, weak)


def _known_bad_probes(
    holdout: Sequence[PointPlaneObservation],
    transform: SE3,
    baseline: PointPlaneEvaluation,
    options: PointPlaneSolverOptions,
) -> tuple[PointPlaneProbe, ...]:
    probes: list[PointPlaneProbe] = []
    for dof in ("x", "y", "z", "roll", "pitch", "yaw"):
        amount = (
            options.known_bad_translation_m
            if dof in {"x", "y", "z"}
            else options.known_bad_rotation_deg
        )
        unit: Literal["m", "deg"] = "m" if dof in {"x", "y", "z"} else "deg"
        for signed_amount in (-amount, amount):
            candidate = _perturb(transform, dof, signed_amount)
            evaluation = evaluate_point_plane_observations(holdout, candidate)
            center_delta = _difference(evaluation.center_rmse_m, baseline.center_rmse_m)
            normal_delta = _difference(evaluation.normal_rmse_deg, baseline.normal_rmse_deg)
            detectable = (
                center_delta > options.known_bad_center_margin_m
                or normal_delta > options.known_bad_normal_margin_deg
                if center_delta is not None and normal_delta is not None
                else None
            )
            probes.append(
                PointPlaneProbe(
                    dof,
                    signed_amount,
                    unit,
                    evaluation.center_rmse_m,
                    evaluation.normal_rmse_deg,
                    center_delta,
                    normal_delta,
                    detectable,
                )
            )
    return tuple(probes)


def _perturb(transform: SE3, dof: str, amount: float) -> SE3:
    if dof in {"x", "y", "z"}:
        translation = [0.0, 0.0, 0.0]
        translation[("x", "y", "z").index(dof)] = amount
        delta = SE3(
            (translation[0], translation[1], translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    else:
        axis = {
            "roll": (1.0, 0.0, 0.0),
            "pitch": (0.0, 1.0, 0.0),
            "yaw": (0.0, 0.0, 1.0),
        }[dof]
        angle = math.radians(amount)
        sine = math.sin(angle / 2.0)
        delta = SE3(
            (0.0, 0.0, 0.0),
            (axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(angle / 2.0)),
        )
    return delta.compose(transform)


def _capture_residual_m(
    item: PointPlaneObservation, transform: SE3, normal_scale_m: float
) -> float:
    center = math.sqrt(
        _squared_distance(transform.transform_point(item.lidar_center_m), item.camera_center_m)
    )
    normal = _oriented_angle(
        rotate_vector_xyzw(transform.rotation_quat_xyzw, item.lidar_normal),
        item.camera_normal,
    )
    return math.hypot(center, normal_scale_m * normal)


def _transform_delta(left: SE3, right: SE3) -> float:
    relative = right.inverse().compose(left)
    translation = math.sqrt(sum(value * value for value in relative.translation_m))
    scalar = min(1.0, abs(relative.rotation_quat_xyzw[3]))
    return math.hypot(translation, 2.0 * math.acos(scalar))


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


def _skew(value: FloatArray) -> FloatArray:
    x, y, z = value
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _finite_vector3(value: Vector3, label: str) -> Vector3:
    if len(value) != 3 or not all(math.isfinite(item) for item in value):
        raise ValueError(f"{label} must contain three finite values")
    return (float(value[0]), float(value[1]), float(value[2]))


def _unit(value: Vector3) -> Vector3:
    value = _finite_vector3(value, "board normal")
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1.0e-12:
        raise ValueError("board normal must be non-zero")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def _oriented_angle(left: Vector3, right: Vector3) -> float:
    cosine = _dot(left, right)
    return math.acos(min(1.0, max(-1.0, cosine)))


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _squared_distance(left: Vector3, right: Vector3) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _root_mean(squared: Sequence[float]) -> float | None:
    return math.sqrt(sum(squared) / len(squared)) if squared else None


def _difference(value: float | None, baseline: float | None) -> float | None:
    return value - baseline if value is not None and baseline is not None else None


def _validate_options(options: PointPlaneSolverOptions) -> None:
    if options.min_train_observations < 3:
        raise ValueError("at least three training observations are required")
    if options.normal_scale_m <= 0.0 or options.huber_delta_m <= 0.0:
        raise ValueError("normal scale and Huber delta must be positive")
    if options.max_iterations <= 0 or options.convergence_tolerance <= 0.0:
        raise ValueError("iteration count and convergence tolerance must be positive")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("holdout ratio must be in [0, 1)")
    if options.rank_tolerance <= 0.0 or options.max_condition_number <= 1.0:
        raise ValueError("rank tolerance and maximum condition number must be valid")
    positive = (
        options.normal_scale_m,
        options.huber_delta_m,
        options.convergence_tolerance,
        options.rank_tolerance,
        options.max_condition_number,
        options.known_bad_rotation_deg,
        options.known_bad_translation_m,
    )
    margins = (options.known_bad_normal_margin_deg, options.known_bad_center_margin_m)
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
        raise ValueError(
            "solver scales, tolerances, limits, and probes must be finite and positive"
        )
    if not all(math.isfinite(value) and value >= 0.0 for value in margins):
        raise ValueError("known-bad detection margins must be finite and non-negative")


def _empty(
    status: Literal["insufficient_observations", "degenerate_geometry"],
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    options: PointPlaneSolverOptions,
) -> PointPlaneLidarCameraResult:
    return PointPlaneLidarCameraResult(
        status=status,
        reason=(
            "at least three positive-weight training captures are required"
            if status == "insufficient_observations"
            else "centre/normal geometry does not constrain six DoF"
        ),
        transform_camera_lidar=None,
        train_frame_ids=train_ids,
        holdout_frame_ids=holdout_ids,
        joint_singular_values=None,
        joint_rank=0,
        joint_condition_number=None,
        weak_joint_direction=None,
        train_evaluation=PointPlaneEvaluation(None, None, None),
        holdout_evaluation=PointPlaneEvaluation(None, None, None),
        probes=(),
        iterations=0,
        solver_options=options,
    )
