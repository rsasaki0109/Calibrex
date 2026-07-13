"""Plane-only planar-board LiDAR-camera extrinsic calibration.

This is an independent implementation of the plane-correspondence geometry
used by Zhang and Pless (IROS 2004). Feature extraction is intentionally left
to adapters: the ROS-independent core consumes oriented planes only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3, quaternion_xyzw_from_rotation_matrix
from calibrex.evaluation.holdout import split_indices

PlanarBoardStatus = Literal[
    "converged", "insufficient_observations", "degenerate_normals", "numerical_failure"
]
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class OrientedPlane:
    """Unit-normal plane using ``normal dot point + offset_m = 0``."""

    normal: Vector3
    offset_m: float

    def __post_init__(self) -> None:
        norm = math.sqrt(sum(value * value for value in self.normal))
        if norm <= 1.0e-12:
            raise ValueError("plane normal must be non-zero")
        object.__setattr__(self, "normal", tuple(value / norm for value in self.normal))
        object.__setattr__(self, "offset_m", float(self.offset_m) / norm)


@dataclass(frozen=True)
class PlanarBoardObservation:
    """Corresponding board planes in camera and LiDAR frames."""

    frame_id: str
    camera_plane: OrientedPlane
    lidar_plane: OrientedPlane
    weight: float = 1.0


@dataclass(frozen=True)
class PlanarBoardSolverOptions:
    min_train_observations: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-6
    max_condition_number: float = 1.0e4
    max_iterations: int = 20
    convergence_tolerance: float = 1.0e-10
    huber_delta: float = 1.5
    angular_scale_m_per_rad: float = 1.0
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_normal_margin_deg: float = 0.1
    known_bad_offset_margin_m: float = 0.005
    normal_sign_policy: Literal["initial_alignment", "preserve"] = "initial_alignment"


@dataclass(frozen=True)
class PlanarBoardEvaluation:
    normal_rmse_deg: float | None
    offset_rmse_m: float | None


@dataclass(frozen=True)
class PlanarBoardProbeResult:
    """Held-out response to one deliberately wrong transform component."""

    dof: Literal["x", "y", "z", "roll", "pitch", "yaw"]
    amount: float
    unit: Literal["m", "deg"]
    normal_rmse_deg: float | None
    offset_rmse_m: float | None
    normal_delta_deg: float | None
    offset_delta_m: float | None
    detectable: bool | None


@dataclass(frozen=True)
class PlanarBoardLidarCameraResult:
    status: PlanarBoardStatus
    reason: str
    transform_camera_lidar: SE3 | None
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    flipped_lidar_normal_frame_ids: tuple[str, ...]
    normal_singular_values: tuple[float, float, float] | None
    normal_rank: int
    normal_condition_number: float | None
    train_evaluation: PlanarBoardEvaluation
    holdout_evaluation: PlanarBoardEvaluation
    probes: tuple[PlanarBoardProbeResult, ...]
    normal_sign_policy: Literal["initial_alignment", "preserve"]
    iterations: int

    def as_dict(self) -> dict[str, object]:
        """Return schema-friendly results with method provenance."""

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
            "flipped_lidar_normal_frame_ids": list(self.flipped_lidar_normal_frame_ids),
            "normal_singular_values": self.normal_singular_values,
            "normal_rank": self.normal_rank,
            "normal_condition_number": self.normal_condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "iterations": self.iterations,
            "method": "zhang_pless_plane_correspondence_irls/v0.1",
            "paper_doi": "10.1109/IROS.2004.1389752",
            "frame_convention": "p_camera = R_camera_lidar p_lidar + t_camera_lidar",
            "normal_sign_policy": self.normal_sign_policy,
            "extractor": "external adapter; not part of native solver",
        }


class PlanarBoardLidarCameraSolver:
    """Estimate a LiDAR-to-camera transform from diverse corresponding planes."""

    def solve(
        self,
        observations: Sequence[PlanarBoardObservation],
        initial_transform: SE3 | None = None,
        options: PlanarBoardSolverOptions | None = None,
    ) -> PlanarBoardLidarCameraResult:
        solver_options = options or PlanarBoardSolverOptions()
        initial = initial_transform or SE3.identity()
        usable = sorted(
            (item for item in observations if item.weight > 0.0),
            key=lambda item: item.frame_id,
        )
        train_indices, holdout_indices = split_indices(
            len(usable), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        oriented, flipped = (
            _orient_observations(usable, initial)
            if solver_options.normal_sign_policy == "initial_alignment"
            else (list(usable), ())
        )
        train = [oriented[index] for index in train_indices]
        holdout = [oriented[index] for index in holdout_indices]
        train_ids = tuple(item.frame_id for item in train)
        holdout_ids = tuple(item.frame_id for item in holdout)
        if len(train) < solver_options.min_train_observations:
            return _empty_result(
                "insufficient_observations",
                train_ids,
                holdout_ids,
                flipped,
                normal_sign_policy=solver_options.normal_sign_policy,
            )

        spectrum = np.linalg.svd(
            np.asarray([item.camera_plane.normal for item in train]), compute_uv=False
        )
        padded = np.pad(spectrum, (0, max(0, 3 - len(spectrum))))[:3]
        singular_values = (float(padded[0]), float(padded[1]), float(padded[2]))
        rank = int(np.count_nonzero(padded > solver_options.rank_tolerance * padded[0]))
        condition = float(padded[0] / padded[2]) if padded[2] > 0.0 else math.inf
        if rank < 3 or condition > solver_options.max_condition_number:
            return _empty_result(
                "degenerate_normals",
                train_ids,
                holdout_ids,
                flipped,
                singular_values,
                rank,
                condition,
                normal_sign_policy=solver_options.normal_sign_policy,
            )

        robust = np.ones(len(train), dtype=float)
        transform: SE3 | None = None
        iterations = 0
        for iteration in range(solver_options.max_iterations):
            candidate = _fit_transform(train, robust)
            if candidate is None:
                return _empty_result(
                    "numerical_failure",
                    train_ids,
                    holdout_ids,
                    flipped,
                    singular_values,
                    rank,
                    condition,
                    normal_sign_policy=solver_options.normal_sign_policy,
                )
            residuals = np.asarray(
                [_combined_residual(item, candidate, solver_options) for item in train]
            )
            scale = max(1.0e-12, 1.4826 * float(np.median(np.abs(residuals))))
            threshold = solver_options.huber_delta * scale
            updated = np.minimum(1.0, threshold / np.maximum(residuals, 1.0e-12))
            iterations = iteration + 1
            transform = candidate
            if float(np.max(np.abs(updated - robust))) <= solver_options.convergence_tolerance:
                break
            robust = updated

        assert transform is not None
        return PlanarBoardLidarCameraResult(
            status="converged",
            reason="weighted plane alignment converged",
            transform_camera_lidar=transform,
            train_frame_ids=train_ids,
            holdout_frame_ids=holdout_ids,
            flipped_lidar_normal_frame_ids=flipped,
            normal_singular_values=singular_values,
            normal_rank=rank,
            normal_condition_number=condition,
            train_evaluation=evaluate_planar_board_observations(train, transform),
            holdout_evaluation=evaluate_planar_board_observations(holdout, transform),
            probes=_known_bad_probes(holdout, transform, solver_options),
            normal_sign_policy=solver_options.normal_sign_policy,
            iterations=iterations,
        )


def evaluate_planar_board_observations(
    observations: Sequence[PlanarBoardObservation], transform_camera_lidar: SE3
) -> PlanarBoardEvaluation:
    """Evaluate held-out plane-normal and plane-offset closure."""

    if not observations:
        return PlanarBoardEvaluation(None, None)
    rotation = _rotation_matrix(transform_camera_lidar)
    normal_errors: list[float] = []
    offset_errors: list[float] = []
    translation = np.asarray(transform_camera_lidar.translation_m)
    for item in observations:
        predicted = rotation @ np.asarray(item.lidar_plane.normal)
        camera = np.asarray(item.camera_plane.normal)
        normal_errors.append(math.acos(float(np.clip(predicted @ camera, -1.0, 1.0))))
        predicted_offset = item.lidar_plane.offset_m - float(camera @ translation)
        offset_errors.append(predicted_offset - item.camera_plane.offset_m)
    return PlanarBoardEvaluation(
        normal_rmse_deg=math.degrees(
            math.sqrt(sum(x * x for x in normal_errors) / len(normal_errors))
        ),
        offset_rmse_m=math.sqrt(sum(x * x for x in offset_errors) / len(offset_errors)),
    )


def _orient_observations(
    observations: Sequence[PlanarBoardObservation], initial: SE3
) -> tuple[list[PlanarBoardObservation], tuple[str, ...]]:
    rotation = _rotation_matrix(initial)
    result: list[PlanarBoardObservation] = []
    flipped: list[str] = []
    for item in observations:
        lidar = item.lidar_plane
        alignment = float(
            (rotation @ np.asarray(lidar.normal)) @ np.asarray(item.camera_plane.normal)
        )
        if alignment < 0.0:
            nx, ny, nz = lidar.normal
            lidar = OrientedPlane((-nx, -ny, -nz), -lidar.offset_m)
            flipped.append(item.frame_id)
        result.append(PlanarBoardObservation(item.frame_id, item.camera_plane, lidar, item.weight))
    return result, tuple(flipped)


def _fit_transform(
    observations: Sequence[PlanarBoardObservation], robust: FloatArray
) -> SE3 | None:
    weights = np.asarray([item.weight for item in observations]) * robust
    covariance = np.zeros((3, 3), dtype=float)
    for item, weight in zip(observations, weights, strict=True):
        covariance += weight * np.outer(item.camera_plane.normal, item.lidar_plane.normal)
    left, _singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    rotation = left @ correction @ right_t
    normals = np.asarray([item.camera_plane.normal for item in observations])
    rhs = np.asarray(
        [item.lidar_plane.offset_m - item.camera_plane.offset_m for item in observations]
    )
    root_weights = np.sqrt(weights)
    translation, _residuals, rank, _spectrum = np.linalg.lstsq(
        normals * root_weights[:, None], rhs * root_weights, rcond=None
    )
    if rank < 3:
        return None
    quaternion = quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1))
    translation_tuple = (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )
    return SE3(translation_tuple, quaternion)


def _combined_residual(
    item: PlanarBoardObservation, transform: SE3, options: PlanarBoardSolverOptions
) -> float:
    evaluation = evaluate_planar_board_observations([item], transform)
    angle = math.radians(evaluation.normal_rmse_deg or 0.0)
    offset = evaluation.offset_rmse_m or 0.0
    return math.hypot(options.angular_scale_m_per_rad * angle, offset)


def _known_bad_probes(
    holdout: Sequence[PlanarBoardObservation],
    transform: SE3,
    options: PlanarBoardSolverOptions,
) -> tuple[PlanarBoardProbeResult, ...]:
    baseline = evaluate_planar_board_observations(holdout, transform)
    probes: list[PlanarBoardProbeResult] = []
    axes: tuple[Vector3, Vector3, Vector3] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation_dofs: tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")
    for index, translation_dof in enumerate(translation_dofs):
        for sign in (-1.0, 1.0):
            amount = sign * options.known_bad_translation_m
            axis = axes[index]
            translation: Vector3 = (
                amount * axis[0],
                amount * axis[1],
                amount * axis[2],
            )
            candidate = SE3(translation, (0.0, 0.0, 0.0, 1.0)).compose(transform)
            probes.append(
                _probe_result(
                    translation_dof, amount, "m", holdout, candidate, baseline, options
                )
            )
    rotation_dofs: tuple[Literal["roll", "pitch", "yaw"], ...] = (
        "roll",
        "pitch",
        "yaw",
    )
    for index, rotation_dof in enumerate(rotation_dofs):
        for sign in (-1.0, 1.0):
            amount = sign * options.known_bad_rotation_deg
            half = math.radians(amount) / 2.0
            axis = axes[index]
            quaternion = (
                axis[0] * math.sin(half),
                axis[1] * math.sin(half),
                axis[2] * math.sin(half),
                math.cos(half),
            )
            candidate = SE3((0.0, 0.0, 0.0), quaternion).compose(transform)
            probes.append(
                _probe_result(
                    rotation_dof, amount, "deg", holdout, candidate, baseline, options
                )
            )
    return tuple(probes)


def _probe_result(
    dof: Literal["x", "y", "z", "roll", "pitch", "yaw"],
    amount: float,
    unit: Literal["m", "deg"],
    holdout: Sequence[PlanarBoardObservation],
    candidate: SE3,
    baseline: PlanarBoardEvaluation,
    options: PlanarBoardSolverOptions,
) -> PlanarBoardProbeResult:
    evaluated = evaluate_planar_board_observations(holdout, candidate)
    normal_delta = (
        evaluated.normal_rmse_deg - baseline.normal_rmse_deg
        if evaluated.normal_rmse_deg is not None and baseline.normal_rmse_deg is not None
        else None
    )
    offset_delta = (
        evaluated.offset_rmse_m - baseline.offset_rmse_m
        if evaluated.offset_rmse_m is not None and baseline.offset_rmse_m is not None
        else None
    )
    detectable = (
        normal_delta > options.known_bad_normal_margin_deg
        or offset_delta > options.known_bad_offset_margin_m
        if normal_delta is not None and offset_delta is not None
        else None
    )
    return PlanarBoardProbeResult(
        dof=dof,
        amount=amount,
        unit=unit,
        normal_rmse_deg=evaluated.normal_rmse_deg,
        offset_rmse_m=evaluated.offset_rmse_m,
        normal_delta_deg=normal_delta,
        offset_delta_m=offset_delta,
        detectable=detectable,
    )


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    matrix: FloatArray = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return matrix


def _empty_result(
    status: PlanarBoardStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    flipped: tuple[str, ...],
    spectrum: tuple[float, float, float] | None = None,
    rank: int = 0,
    condition: float | None = None,
    normal_sign_policy: Literal["initial_alignment", "preserve"] = "initial_alignment",
) -> PlanarBoardLidarCameraResult:
    return PlanarBoardLidarCameraResult(
        status=status,
        reason={
            "insufficient_observations": "too few training board poses",
            "degenerate_normals": "board normals do not stably span three dimensions",
            "numerical_failure": "plane alignment failed numerically",
            "converged": "",
        }[status],
        transform_camera_lidar=None,
        train_frame_ids=train_ids,
        holdout_frame_ids=holdout_ids,
        flipped_lidar_normal_frame_ids=flipped,
        normal_singular_values=spectrum,
        normal_rank=rank,
        normal_condition_number=condition,
        train_evaluation=PlanarBoardEvaluation(None, None),
        holdout_evaluation=PlanarBoardEvaluation(None, None),
        probes=(),
        normal_sign_policy=normal_sign_policy,
        iterations=0,
    )
