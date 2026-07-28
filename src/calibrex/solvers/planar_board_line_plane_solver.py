"""Planar-board LiDAR-camera calibration using line and plane correspondences."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3, quaternion_xyzw_from_rotation_matrix
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.planar_board_lidar_camera_solver import OrientedPlane

FloatArray: TypeAlias = NDArray[np.float64]
LinePlaneStatus = Literal[
    "converged", "insufficient_constraints", "degenerate_geometry", "numerical_failure"
]


@dataclass(frozen=True)
class OrientedLine3D:
    """A 3D line represented by one point and an oriented unit direction."""

    point: Vector3
    direction: Vector3

    def __post_init__(self) -> None:
        norm = math.sqrt(sum(value * value for value in self.direction))
        if norm <= 1.0e-12:
            raise ValueError("line direction must be non-zero")
        x, y, z = self.direction
        object.__setattr__(self, "direction", (x / norm, y / norm, z / norm))


@dataclass(frozen=True)
class BoardBoundaryCorrespondence:
    """Corresponding finite-board boundary lines in camera and LiDAR frames."""

    boundary_id: str
    camera_line: OrientedLine3D
    lidar_line: OrientedLine3D
    weight: float = 1.0


@dataclass(frozen=True)
class LinePlaneBoardObservation:
    """One board capture containing a plane and its corresponding boundaries."""

    frame_id: str
    camera_plane: OrientedPlane
    lidar_plane: OrientedPlane
    boundaries: tuple[BoardBoundaryCorrespondence, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class LinePlaneSolverOptions:
    holdout_ratio: float = 0.2
    split_seed: int = 0
    rank_tolerance: float = 1.0e-7
    max_condition_number: float = 1.0e6
    minimum_edge_angle_deg: float = 10.0


@dataclass(frozen=True)
class LinePlaneEvaluation:
    plane_normal_rmse_deg: float | None
    plane_offset_rmse_m: float | None
    boundary_direction_rmse_deg: float | None
    boundary_distance_rmse_m: float | None


@dataclass(frozen=True)
class PlanarBoardLinePlaneResult:
    status: LinePlaneStatus
    reason: str
    transform_camera_lidar: SE3 | None
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    rotation_singular_values: tuple[float, float, float] | None
    translation_singular_values: tuple[float, float, float] | None
    rotation_rank: int
    translation_rank: int
    condition_number: float | None
    train_evaluation: LinePlaneEvaluation
    holdout_evaluation: LinePlaneEvaluation

    def as_dict(self) -> dict[str, object]:
        """Return a schema-friendly result including method provenance."""

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
            "rotation_singular_values": self.rotation_singular_values,
            "translation_singular_values": self.translation_singular_values,
            "rotation_rank": self.rotation_rank,
            "translation_rank": self.translation_rank,
            "condition_number": self.condition_number,
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "method": "zhou_line_plane_closed_form/v0.1",
            "paper_doi": "10.1109/IROS.2018.8593660",
            "frame_convention": "p_camera = R_camera_lidar p_lidar + t_camera_lidar",
            "extractor": "external adapter; not part of native solver",
            "correspondence_policy": (
                "edge IDs and oriented directions must be matched by the extractor"
            ),
            "discrete_symmetry_policy": "unlabeled board permutations are not resolved",
        }


class PlanarBoardLinePlaneSolver:
    """Estimate LiDAR-to-camera SE(3) from board plane and boundary lines."""

    def solve(
        self,
        observations: Sequence[LinePlaneBoardObservation],
        options: LinePlaneSolverOptions | None = None,
    ) -> PlanarBoardLinePlaneResult:
        solver_options = options or LinePlaneSolverOptions()
        usable = sorted(
            (item for item in observations if item.weight > 0.0),
            key=lambda item: item.frame_id,
        )
        train_indices, holdout_indices = split_indices(
            len(usable), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        train_ids = tuple(item.frame_id for item in train)
        holdout_ids = tuple(item.frame_id for item in holdout)
        if not train or sum(len(item.boundaries) for item in train) < 2:
            return _empty("insufficient_constraints", train_ids, holdout_ids)
        if not _has_nonparallel_edges(train, solver_options.minimum_edge_angle_deg):
            return _empty("degenerate_geometry", train_ids, holdout_ids)

        rotation, rotation_spectrum, rotation_rank = _solve_rotation(train, solver_options)
        if rotation is None or rotation_rank < 3:
            return _empty(
                "degenerate_geometry",
                train_ids,
                holdout_ids,
                rotation_spectrum=rotation_spectrum,
                rotation_rank=rotation_rank,
            )
        translation, translation_spectrum, translation_rank = _solve_translation(
            train, rotation, solver_options
        )
        if translation is None or translation_rank < 3:
            return _empty(
                "degenerate_geometry",
                train_ids,
                holdout_ids,
                rotation_spectrum=rotation_spectrum,
                translation_spectrum=translation_spectrum,
                rotation_rank=rotation_rank,
                translation_rank=translation_rank,
            )
        smallest = min(rotation_spectrum[2], translation_spectrum[2])
        largest = max(rotation_spectrum[0], translation_spectrum[0])
        condition = largest / smallest if smallest > 0.0 else math.inf
        if condition > solver_options.max_condition_number:
            return _empty(
                "degenerate_geometry",
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
        return PlanarBoardLinePlaneResult(
            status="converged",
            reason="line and plane alignment solved",
            transform_camera_lidar=transform,
            train_frame_ids=train_ids,
            holdout_frame_ids=holdout_ids,
            rotation_singular_values=rotation_spectrum,
            translation_singular_values=translation_spectrum,
            rotation_rank=rotation_rank,
            translation_rank=translation_rank,
            condition_number=condition,
            train_evaluation=evaluate_line_plane_observations(train, transform),
            holdout_evaluation=evaluate_line_plane_observations(holdout, transform),
        )


def evaluate_line_plane_observations(
    observations: Sequence[LinePlaneBoardObservation], transform: SE3
) -> LinePlaneEvaluation:
    """Evaluate independent plane and boundary closure metrics."""

    if not observations:
        return LinePlaneEvaluation(None, None, None, None)
    rotation = _rotation_matrix(transform)
    translation = np.asarray(transform.translation_m)
    plane_angles: list[float] = []
    plane_offsets: list[float] = []
    line_angles: list[float] = []
    line_distances: list[float] = []
    for item in observations:
        camera_normal = np.asarray(item.camera_plane.normal)
        predicted_normal = rotation @ np.asarray(item.lidar_plane.normal)
        plane_angles.append(_unsigned_angle(predicted_normal, camera_normal))
        predicted_offset = item.lidar_plane.offset_m - float(camera_normal @ translation)
        plane_offsets.append(predicted_offset - item.camera_plane.offset_m)
        for boundary in item.boundaries:
            camera_direction = np.asarray(boundary.camera_line.direction)
            predicted_direction = rotation @ np.asarray(boundary.lidar_line.direction)
            line_angles.append(_unsigned_angle(predicted_direction, camera_direction))
            predicted_point = rotation @ np.asarray(boundary.lidar_line.point) + translation
            delta = predicted_point - np.asarray(boundary.camera_line.point)
            projector = np.eye(3) - np.outer(camera_direction, camera_direction)
            line_distances.append(float(np.linalg.norm(projector @ delta)))
    return LinePlaneEvaluation(
        plane_normal_rmse_deg=_angle_rmse_deg(plane_angles),
        plane_offset_rmse_m=_rmse(plane_offsets),
        boundary_direction_rmse_deg=_angle_rmse_deg(line_angles),
        boundary_distance_rmse_m=_rmse(line_distances),
    )


def _solve_rotation(
    observations: Sequence[LinePlaneBoardObservation], options: LinePlaneSolverOptions
) -> tuple[FloatArray | None, tuple[float, float, float], int]:
    covariance: FloatArray = np.zeros((3, 3), dtype=np.float64)
    for item in observations:
        covariance += item.weight * np.outer(item.camera_plane.normal, item.lidar_plane.normal)
        for boundary in item.boundaries:
            weight = item.weight * boundary.weight
            covariance += weight * np.outer(
                boundary.camera_line.direction, boundary.lidar_line.direction
            )
    left, spectrum, right_t = np.linalg.svd(covariance)
    values = (float(spectrum[0]), float(spectrum[1]), float(spectrum[2]))
    rank = int(np.count_nonzero(spectrum > options.rank_tolerance * spectrum[0]))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    return left @ correction @ right_t, values, rank


def _solve_translation(
    observations: Sequence[LinePlaneBoardObservation],
    rotation: FloatArray,
    options: LinePlaneSolverOptions,
) -> tuple[FloatArray | None, tuple[float, float, float], int]:
    rows: list[FloatArray] = []
    rhs: list[float] = []
    for item in observations:
        root_weight = math.sqrt(item.weight)
        normal = np.asarray(item.camera_plane.normal)
        rows.append(root_weight * normal)
        rhs.append(root_weight * (item.lidar_plane.offset_m - item.camera_plane.offset_m))
        for boundary in item.boundaries:
            direction = np.asarray(boundary.camera_line.direction)
            projector = np.eye(3) - np.outer(direction, direction)
            target = np.asarray(boundary.camera_line.point) - rotation @ np.asarray(
                boundary.lidar_line.point
            )
            edge_weight = math.sqrt(item.weight * boundary.weight)
            for row, value in zip(projector, projector @ target, strict=True):
                rows.append(edge_weight * row)
                rhs.append(edge_weight * float(value))
    matrix = np.asarray(rows)
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    values = (float(spectrum[0]), float(spectrum[1]), float(spectrum[2]))
    rank = int(np.count_nonzero(spectrum > options.rank_tolerance * spectrum[0]))
    if rank < 3:
        return None, values, rank
    solution, _residuals, _rank, _singular = np.linalg.lstsq(matrix, np.asarray(rhs), rcond=None)
    return solution, values, rank


def _has_nonparallel_edges(
    observations: Sequence[LinePlaneBoardObservation], minimum_angle_deg: float
) -> bool:
    directions = [
        np.asarray(boundary.camera_line.direction)
        for item in observations
        for boundary in item.boundaries
        if boundary.weight > 0.0
    ]
    return any(
        math.degrees(_unsigned_angle(left, right)) >= minimum_angle_deg
        for index, left in enumerate(directions)
        for right in directions[index + 1 :]
    )


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


def _unsigned_angle(left: FloatArray, right: FloatArray) -> float:
    cosine = abs(float(left @ right) / float(np.linalg.norm(left) * np.linalg.norm(right)))
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def _angle_rmse_deg(values: Sequence[float]) -> float | None:
    value = _rmse(values)
    return math.degrees(value) if value is not None else None


def _rmse(values: Sequence[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def _empty(
    status: LinePlaneStatus,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    rotation_spectrum: tuple[float, float, float] | None = None,
    translation_spectrum: tuple[float, float, float] | None = None,
    rotation_rank: int = 0,
    translation_rank: int = 0,
    condition: float | None = None,
) -> PlanarBoardLinePlaneResult:
    reasons = {
        "insufficient_constraints": "at least one plane and two boundaries are required",
        "degenerate_geometry": "plane and boundary geometry does not constrain six DoF",
        "numerical_failure": "line and plane solve failed numerically",
        "converged": "",
    }
    return PlanarBoardLinePlaneResult(
        status=status,
        reason=reasons[status],
        transform_camera_lidar=None,
        train_frame_ids=train_ids,
        holdout_frame_ids=holdout_ids,
        rotation_singular_values=rotation_spectrum,
        translation_singular_values=translation_spectrum,
        rotation_rank=rotation_rank,
        translation_rank=translation_rank,
        condition_number=condition,
        train_evaluation=LinePlaneEvaluation(None, None, None, None),
        holdout_evaluation=LinePlaneEvaluation(None, None, None, None),
    )
