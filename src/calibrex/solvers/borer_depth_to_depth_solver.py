"""Deterministic depth-to-depth mutual-information objective.

This independently implements the fixed-pose evaluator described by Borer
et al. (arXiv:2311.01905). Learned depth generation remains outside the core.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3

FloatArray: TypeAlias = NDArray[np.float64]
DepthProjectionKind = Literal["pinhole", "double_sphere", "mei"]


@dataclass(frozen=True)
class DepthToDepthCameraModel:
    """Camera model for a depth-map grid."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    projection: DepthProjectionKind = "pinhole"
    xi: float = 0.0
    alpha: float = 0.5
    distortion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.width <= 1 or self.height <= 1:
            raise ValueError("camera image dimensions must exceed one pixel")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if self.projection == "double_sphere" and not 0.0 < self.alpha < 1.0:
            raise ValueError("double-sphere alpha must be in (0, 1)")
        if self.projection == "mei" and not math.isfinite(self.xi):
            raise ValueError("MEI xi must be finite")
        if not all(math.isfinite(value) for value in self.distortion):
            raise ValueError("camera distortion coefficients must be finite")


@dataclass(frozen=True)
class DepthToDepthObservation:
    """One synchronized camera depth map and LiDAR point cloud."""

    frame_id: str
    depth_map: FloatArray
    lidar_points: FloatArray
    camera: DepthToDepthCameraModel
    lidar_range_m: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_map, dtype=float)
        points = np.asarray(self.lidar_points, dtype=float)
        if depth.shape != (self.camera.height, self.camera.width):
            raise ValueError("depth_map shape must match the camera model")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("lidar_points must have shape Nx3")
        if not np.all(np.isfinite(points)):
            raise ValueError("lidar_points must be finite")
        object.__setattr__(self, "depth_map", depth)
        object.__setattr__(self, "lidar_points", points)
        object.__setattr__(self, "lidar_range_m", np.linalg.norm(points, axis=1))


@dataclass(frozen=True)
class DepthToDepthOptions:
    """Frozen histogram and visibility settings for D2D evaluation."""

    histogram_bins: int = 32
    min_visible_points: int = 64
    use_z_buffer: bool = True
    camera_depth_range: tuple[float, float] | None = None
    lidar_range_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.histogram_bins < 8:
            raise ValueError("histogram_bins must be at least 8")
        if self.min_visible_points < 4:
            raise ValueError("min_visible_points must be at least 4")
        for limits in (self.camera_depth_range, self.lidar_range_m):
            if limits is not None and (
                not math.isfinite(limits[0])
                or not math.isfinite(limits[1])
                or limits[0] >= limits[1]
            ):
                raise ValueError("histogram ranges must be finite and increasing")


@dataclass(frozen=True)
class DepthPairProjection:
    """Visible, paired depth features for one observation."""

    camera_depth: FloatArray
    lidar_range_m: FloatArray
    pixel_u: NDArray[np.int64]
    pixel_v: NDArray[np.int64]
    projected_count_before_visibility: int


@dataclass(frozen=True)
class LidarImageCorrespondenceProjection:
    """One LiDAR point projected into the image with sub-pixel coordinates."""

    point_lidar_m: tuple[float, float, float]
    image_u_px: float
    image_v_px: float


@dataclass(frozen=True)
class DepthToDepthFrameEvaluation:
    """Per-frame D2D mutual-information evidence."""

    frame_id: str
    mutual_information: float
    normalized_mutual_information: float
    visible_point_count: int
    projected_count_before_visibility: int


@dataclass(frozen=True)
class DepthToDepthEvaluation:
    """Average per-frame D2D objective for a fixed transform."""

    mutual_information: float
    normalized_mutual_information: float
    visible_point_count: int
    evaluated_frame_count: int
    skipped_frame_ids: tuple[str, ...]
    frames: tuple[DepthToDepthFrameEvaluation, ...]
    method: str = "borer_depth_to_depth_mutual_information/v0.1"
    primary_source: str = "https://arxiv.org/abs/2311.01905"

    def as_dict(self) -> dict[str, object]:
        """Return a schema-safe evaluator record."""

        return {
            "method": self.method,
            "primary_source": self.primary_source,
            "mutual_information": self.mutual_information,
            "normalized_mutual_information": self.normalized_mutual_information,
            "visible_point_count": self.visible_point_count,
            "evaluated_frame_count": self.evaluated_frame_count,
            "skipped_frame_ids": list(self.skipped_frame_ids),
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "mutual_information": frame.mutual_information,
                    "normalized_mutual_information": frame.normalized_mutual_information,
                    "visible_point_count": frame.visible_point_count,
                    "projected_count_before_visibility": (
                        frame.projected_count_before_visibility
                    ),
                }
                for frame in self.frames
            ],
        }


def project_depth_pairs(
    observation: DepthToDepthObservation,
    transform_camera_lidar: SE3,
    *,
    use_z_buffer: bool = True,
) -> DepthPairProjection:
    """Project LiDAR points and sample paired camera/LiDAR depth features."""

    points_lidar = observation.lidar_points
    rotation = _rotation_matrix(transform_camera_lidar.rotation_quat_xyzw)
    translation: FloatArray = np.asarray(
        transform_camera_lidar.translation_m, dtype=float
    )
    points_camera = points_lidar @ rotation.T + translation
    projected = _project(points_camera, observation.camera)
    valid_indices = np.flatnonzero(projected[2])
    if valid_indices.size == 0:
        return _empty_projection()
    u_float = projected[0][valid_indices]
    v_float = projected[1][valid_indices]
    pixel_u = np.rint(u_float).astype(np.int64)
    pixel_v = np.rint(v_float).astype(np.int64)
    inside = (
        (pixel_u >= 0)
        & (pixel_u < observation.camera.width)
        & (pixel_v >= 0)
        & (pixel_v < observation.camera.height)
    )
    valid_indices = valid_indices[inside]
    pixel_u = pixel_u[inside]
    pixel_v = pixel_v[inside]
    projected_count = int(valid_indices.size)
    if use_z_buffer and valid_indices.size:
        camera_range = np.linalg.norm(points_camera[valid_indices], axis=1)
        pixel_linear = pixel_v * observation.camera.width + pixel_u
        by_depth = np.argsort(camera_range, kind="stable")
        _pixels, first = np.unique(pixel_linear[by_depth], return_index=True)
        selected = by_depth[first]
        valid_indices = valid_indices[selected]
        pixel_u = pixel_u[selected]
        pixel_v = pixel_v[selected]
    camera_depth = observation.depth_map[pixel_v, pixel_u]
    lidar_range = observation.lidar_range_m[valid_indices]
    finite = (
        np.isfinite(camera_depth)
        & (camera_depth > 0.0)
        & np.isfinite(lidar_range)
        & (lidar_range > 0.0)
    )
    return DepthPairProjection(
        camera_depth=np.asarray(camera_depth[finite], dtype=float),
        lidar_range_m=np.asarray(lidar_range[finite], dtype=float),
        pixel_u=pixel_u[finite],
        pixel_v=pixel_v[finite],
        projected_count_before_visibility=projected_count,
    )


def project_lidar_image_correspondences(
    observation: DepthToDepthObservation,
    transform_camera_lidar: SE3,
    *,
    max_points: int | None = 200,
    depth_relative_gate: float = 0.25,
    use_z_buffer: bool = True,
    seed: int = 0,
) -> tuple[LidarImageCorrespondenceProjection, ...]:
    """Project visible LiDAR points into image coordinates for correspondence export."""

    points_lidar = observation.lidar_points
    rotation = _rotation_matrix(transform_camera_lidar.rotation_quat_xyzw)
    translation: FloatArray = np.asarray(
        transform_camera_lidar.translation_m, dtype=float
    )
    points_camera = points_lidar @ rotation.T + translation
    projected = _project(points_camera, observation.camera)
    valid_indices = np.flatnonzero(projected[2])
    if valid_indices.size == 0:
        return ()
    u_float = projected[0][valid_indices]
    v_float = projected[1][valid_indices]
    pixel_u = np.rint(u_float).astype(np.int64)
    pixel_v = np.rint(v_float).astype(np.int64)
    inside = (
        (pixel_u >= 0)
        & (pixel_u < observation.camera.width)
        & (pixel_v >= 0)
        & (pixel_v < observation.camera.height)
    )
    valid_indices = valid_indices[inside]
    u_float = u_float[inside]
    v_float = v_float[inside]
    pixel_u = pixel_u[inside]
    pixel_v = pixel_v[inside]
    if use_z_buffer and valid_indices.size:
        camera_range = np.linalg.norm(points_camera[valid_indices], axis=1)
        pixel_linear = pixel_v * observation.camera.width + pixel_u
        by_depth = np.argsort(camera_range, kind="stable")
        _pixels, first = np.unique(pixel_linear[by_depth], return_index=True)
        selected = by_depth[first]
        valid_indices = valid_indices[selected]
        u_float = u_float[selected]
        v_float = v_float[selected]
        pixel_u = pixel_u[selected]
        pixel_v = pixel_v[selected]
    camera_depth = observation.depth_map[pixel_v, pixel_u]
    lidar_range = observation.lidar_range_m[valid_indices]
    finite = (
        np.isfinite(camera_depth)
        & (camera_depth > 0.0)
        & np.isfinite(lidar_range)
        & (lidar_range > 0.0)
    )
    valid_indices = valid_indices[finite]
    u_float = u_float[finite]
    v_float = v_float[finite]
    camera_depth = camera_depth[finite]
    lidar_range = lidar_range[finite]
    if depth_relative_gate > 0.0:
        relative_error = np.abs(camera_depth - lidar_range) / np.maximum(lidar_range, 1.0e-9)
        consistent = relative_error <= depth_relative_gate
        valid_indices = valid_indices[consistent]
        u_float = u_float[consistent]
        v_float = v_float[consistent]
    projections = [
        LidarImageCorrespondenceProjection(
            point_lidar_m=(
                float(points_lidar[index, 0]),
                float(points_lidar[index, 1]),
                float(points_lidar[index, 2]),
            ),
            image_u_px=float(u_float[item]),
            image_v_px=float(v_float[item]),
        )
        for item, index in enumerate(valid_indices)
    ]
    if max_points is not None and len(projections) > max_points:
        rng = random.Random(seed)
        selected = sorted(rng.sample(range(len(projections)), max_points))
        projections = [projections[index] for index in selected]
    return tuple(projections)


def evaluate_depth_to_depth_mi(
    observations: tuple[DepthToDepthObservation, ...] | list[DepthToDepthObservation],
    transform_camera_lidar: SE3,
    options: DepthToDepthOptions | None = None,
) -> DepthToDepthEvaluation:
    """Average per-frame histogram MI exactly once for a fixed pose."""

    settings = options or DepthToDepthOptions()
    if not observations or not any(
        np.any(np.isfinite(item.depth_map) & (item.depth_map > 0.0))
        and item.lidar_points.size
        for item in observations
    ):
        return DepthToDepthEvaluation(
            mutual_information=0.0,
            normalized_mutual_information=0.0,
            visible_point_count=0,
            evaluated_frame_count=0,
            skipped_frame_ids=tuple(item.frame_id for item in observations),
            frames=(),
        )
    camera_range = settings.camera_depth_range or _camera_depth_range(observations)
    lidar_range = settings.lidar_range_m or _lidar_range(observations)
    frames = []
    skipped = []
    for observation in observations:
        pairs = project_depth_pairs(
            observation,
            transform_camera_lidar,
            use_z_buffer=settings.use_z_buffer,
        )
        if pairs.camera_depth.size < settings.min_visible_points:
            skipped.append(observation.frame_id)
            continue
        mi, normalized = _mutual_information(
            pairs.camera_depth,
            pairs.lidar_range_m,
            bins=settings.histogram_bins,
            camera_range=camera_range,
            lidar_range=lidar_range,
        )
        frames.append(
            DepthToDepthFrameEvaluation(
                frame_id=observation.frame_id,
                mutual_information=mi,
                normalized_mutual_information=normalized,
                visible_point_count=int(pairs.camera_depth.size),
                projected_count_before_visibility=(
                    pairs.projected_count_before_visibility
                ),
            )
        )
    return DepthToDepthEvaluation(
        mutual_information=(
            float(np.mean([frame.mutual_information for frame in frames]))
            if frames
            else 0.0
        ),
        normalized_mutual_information=(
            float(
                np.mean([frame.normalized_mutual_information for frame in frames])
            )
            if frames
            else 0.0
        ),
        visible_point_count=sum(frame.visible_point_count for frame in frames),
        evaluated_frame_count=len(frames),
        skipped_frame_ids=tuple(skipped),
        frames=tuple(frames),
    )


def resolve_depth_to_depth_options(
    observations: tuple[DepthToDepthObservation, ...]
    | list[DepthToDepthObservation],
    options: DepthToDepthOptions | None = None,
) -> DepthToDepthOptions:
    """Resolve invariant histogram ranges once for repeated pose evaluations."""

    settings = options or DepthToDepthOptions()
    return replace(
        settings,
        camera_depth_range=(
            settings.camera_depth_range or _camera_depth_range(observations)
        ),
        lidar_range_m=settings.lidar_range_m or _lidar_range(observations),
    )


def _project(
    points_camera: FloatArray,
    camera: DepthToDepthCameraModel,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]
    if camera.projection == "pinhole":
        valid = z > 1.0e-9
        denominator = np.where(valid, z, 1.0)
    elif camera.projection == "double_sphere":
        d1 = np.linalg.norm(points_camera, axis=1)
        z1 = camera.xi * d1 + z
        d2 = np.sqrt(x * x + y * y + z1 * z1)
        denominator = camera.alpha * d2 + (1.0 - camera.alpha) * z1
        w1 = (
            camera.alpha / (1.0 - camera.alpha)
            if camera.alpha <= 0.5
            else (1.0 - camera.alpha) / camera.alpha
        )
        w2 = (w1 + camera.xi) / math.sqrt(
            2.0 * w1 * camera.xi + camera.xi * camera.xi + 1.0
        )
        valid = (
            (d1 > 1.0e-9)
            & (denominator > 1.0e-9)
            & (z > -w2 * d1)
        )
        denominator = np.where(valid, denominator, 1.0)
        u = camera.fx * x / denominator + camera.cx
        v = camera.fy * y / denominator + camera.cy
    else:
        distance = np.linalg.norm(points_camera, axis=1)
        # KITTI-360's official fisheye projection retains only positive
        # signed camera range (the forward optical hemisphere).
        valid = (distance > 1.0e-9) & (z > 1.0e-9)
        safe_distance = np.where(valid, distance, 1.0)
        normalized_x = x / safe_distance
        normalized_y = y / safe_distance
        normalized_z = z / safe_distance
        denominator = normalized_z + camera.xi
        valid &= denominator > 1.0e-9
        safe_denominator = np.where(valid, denominator, 1.0)
        normalized_x /= safe_denominator
        normalized_y /= safe_denominator
        radius_squared = normalized_x**2 + normalized_y**2
        k1, k2, p1, p2 = camera.distortion
        radial = 1.0 + k1 * radius_squared + k2 * radius_squared**2
        distorted_x = (
            normalized_x * radial
            + 2.0 * p1 * normalized_x * normalized_y
            + p2 * (radius_squared + 2.0 * normalized_x**2)
        )
        distorted_y = (
            normalized_y * radial
            + p1 * (radius_squared + 2.0 * normalized_y**2)
            + 2.0 * p2 * normalized_x * normalized_y
        )
        u = camera.fx * distorted_x + camera.cx
        v = camera.fy * distorted_y + camera.cy
    if camera.projection == "pinhole":
        u = camera.fx * x / denominator + camera.cx
        v = camera.fy * y / denominator + camera.cy
    valid &= np.isfinite(u) & np.isfinite(v)
    return u, v, valid


def _mutual_information(
    camera_depth: FloatArray,
    lidar_depth: FloatArray,
    *,
    bins: int,
    camera_range: tuple[float, float],
    lidar_range: tuple[float, float],
) -> tuple[float, float]:
    histogram, _camera_edges, _lidar_edges = np.histogram2d(
        camera_depth,
        lidar_depth,
        bins=bins,
        range=(camera_range, lidar_range),
    )
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.0, 0.0
    joint = histogram / total
    camera_marginal = joint.sum(axis=1)
    lidar_marginal = joint.sum(axis=0)
    camera_entropy = _entropy(camera_marginal)
    lidar_entropy = _entropy(lidar_marginal)
    joint_entropy = _entropy(joint.ravel())
    mi = max(0.0, camera_entropy + lidar_entropy - joint_entropy)
    normalizer = math.sqrt(camera_entropy * lidar_entropy)
    return mi, mi / normalizer if normalizer > 0.0 else 0.0


def _entropy(probability: FloatArray) -> float:
    positive = probability[probability > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _camera_depth_range(
    observations: tuple[DepthToDepthObservation, ...] | list[DepthToDepthObservation],
) -> tuple[float, float]:
    parts = [
        item.depth_map[np.isfinite(item.depth_map) & (item.depth_map > 0.0)]
        for item in observations
        if np.any(np.isfinite(item.depth_map) & (item.depth_map > 0.0))
    ]
    values = np.concatenate(parts) if parts else np.empty(0, dtype=float)
    return _stable_range(values, label="camera depth")


def _lidar_range(
    observations: tuple[DepthToDepthObservation, ...] | list[DepthToDepthObservation],
) -> tuple[float, float]:
    parts = [item.lidar_range_m for item in observations if item.lidar_points.size]
    values = np.concatenate(parts) if parts else np.empty(0, dtype=float)
    return _stable_range(values, label="LiDAR range")


def _stable_range(values: FloatArray, *, label: str) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError(f"cannot derive {label} histogram range from empty data")
    lower = float(np.min(values))
    upper = float(np.max(values))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{label} histogram range is not finite")
    if lower == upper:
        padding = max(abs(lower) * 1.0e-6, 1.0e-6)
        return lower - padding, upper + padding
    return lower, upper


def _rotation_matrix(quaternion_xyzw: tuple[float, float, float, float]) -> FloatArray:
    x, y, z, w = quaternion_xyzw
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def _empty_projection() -> DepthPairProjection:
    return DepthPairProjection(
        camera_depth=np.empty(0, dtype=float),
        lidar_range_m=np.empty(0, dtype=float),
        pixel_u=np.empty(0, dtype=np.int64),
        pixel_v=np.empty(0, dtype=np.int64),
        projected_count_before_visibility=0,
    )
