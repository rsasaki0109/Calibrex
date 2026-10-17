"""Exact native TUM transforms and captured initial plane geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core import geometry
from calibrex.core.geometry import SE3, Vector3
from calibrex.data.livox import (
    LivoxPointRecord,
    VoxelPlaneMap,
    VoxelPlaneMatch,
    VoxelPlaneQueryContext,
    _PlaneGeometry,
    _VoxelPlane,
)
from calibrex.data.tum_rgbd import TUMDepthPoseEntry, TUMTrajectoryEntry
from calibrex.graph._point_plane_batches import _transform_points

_MIN_POINTS = 64
_CHUNK_ROWS = 4096
_FRAME_PROPERTY = TUMDepthPoseEntry.transform_world_camera
_POINT_METHOD = SE3.transform_point
_SE3_INIT = SE3.__post_init__
_SE3_CONSTRUCTOR = SE3.__init__
_SE3_IDENTITY = getattr(SE3.identity, "__func__", None)
_TUPLE3 = geometry._tuple3
_TUPLE4 = geometry._tuple4
_ROTATE = geometry.rotate_vector_xyzw
_NORMALIZE = geometry.normalize_quaternion_xyzw
_CENTROID_PROPERTY = _VoxelPlane.centroid
_NORMAL_PROPERTY = _VoxelPlane.normal
_MATCHES_METHOD = VoxelPlaneQueryContext.nearest_matches
_InitialPlane: TypeAlias = _VoxelPlane | _PlaneGeometry | VoxelPlaneMatch


def _native_transform_methods() -> bool:
    return (
        SE3.transform_point is _POINT_METHOD
        and SE3.__post_init__ is _SE3_INIT
        and SE3.__init__ is _SE3_CONSTRUCTOR
        and geometry._tuple3 is _TUPLE3
        and geometry._tuple4 is _TUPLE4
        and geometry.rotate_vector_xyzw is _ROTATE
        and geometry.normalize_quaternion_xyzw is _NORMALIZE
    )


def _frame_initial_transforms(frame: TUMDepthPoseEntry) -> tuple[SE3, SE3] | None:
    """Share the native frozen frame values across one frame's measurements."""
    if (
        type(frame) is not TUMDepthPoseEntry
        or type(frame.trajectory) is not TUMTrajectoryEntry
        or TUMDepthPoseEntry.transform_world_camera is not _FRAME_PROPERTY
        or not _native_transform_methods()
        or getattr(SE3.identity, "__func__", None) is not _SE3_IDENTITY
        or not _ordinary_vector(frame.trajectory.translation_m, 3)
        or not _ordinary_vector(frame.trajectory.rotation_quat_xyzw, 4)
    ):
        return None
    return frame.transform_world_camera, SE3.identity()


def _ordinary_vector(value: object, dimension: int) -> bool:
    return (
        type(value) is tuple
        and len(value) == dimension
        and all(
            type(item) is float and math.isfinite(item) and abs(item) <= 1e100 for item in value
        )
    )


def _coordinates(points: Sequence[Vector3], transform: SE3) -> NDArray[np.float64] | None:
    if (
        type(points) not in (list, tuple)
        or len(points) < _MIN_POINTS
        or type(transform) is not SE3
        or not _native_transform_methods()
        or not _ordinary_vector(transform.translation_m, 3)
        or not _ordinary_vector(transform.rotation_quat_xyzw, 4)
        or not all(_ordinary_vector(point, 3) for point in points)
    ):
        return None
    coordinates = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    world = np.empty_like(coordinates)
    for start in range(0, len(points), _CHUNK_ROWS):
        world[start : start + _CHUNK_ROWS] = _transform_points(
            coordinates[start : start + _CHUNK_ROWS], transform
        )
    return world


def _frame_coordinates(
    frame: TUMDepthPoseEntry, points: Sequence[Vector3]
) -> NDArray[np.float64] | None:
    if (
        type(points) not in (list, tuple)
        or not _native_transform_methods()
        or type(frame) is not TUMDepthPoseEntry
        or type(frame.trajectory) is not TUMTrajectoryEntry
        or TUMDepthPoseEntry.transform_world_camera is not _FRAME_PROPERTY
        or not _ordinary_vector(frame.trajectory.translation_m, 3)
        or not _ordinary_vector(frame.trajectory.rotation_quat_xyzw, 4)
        or len(points) < _MIN_POINTS
    ):
        return None
    return _coordinates(points, frame.transform_world_camera)


def _frame_records(frame: TUMDepthPoseEntry, points: list[Vector3]) -> list[LivoxPointRecord]:
    world = _frame_coordinates(frame, points)
    if world is None:
        return [
            LivoxPointRecord((*frame.transform_world_camera.transform_point(point), 0.0), None)
            for point in points
        ]
    return [LivoxPointRecord((*point, 0.0), None) for point in world.tolist()]


def _query_snapshot(planes: VoxelPlaneMap) -> VoxelPlaneQueryContext | None:
    if (
        type(planes) is not dict
        or _VoxelPlane.centroid is not _CENTROID_PROPERTY
        or _VoxelPlane.normal is not _NORMAL_PROPERTY
        or VoxelPlaneQueryContext.nearest_matches is not _MATCHES_METHOD
    ):
        return None
    for key, plane in planes.items():
        if (
            type(key) is not tuple
            or len(key) != 3
            or any(type(item) is not int for item in key)
            or type(plane) is not _VoxelPlane
            or type(plane.count) is not int
            or type(plane.normal_count) is not int
            or not _ordinary_vector((plane.x_sum, plane.y_sum, plane.z_sum), 3)
            or not _ordinary_vector((plane.normal_x_sum, plane.normal_y_sum, plane.normal_z_sum), 3)
            or (
                plane.fallback_points is not None
                and (
                    type(plane.fallback_points) is not list
                    or len(plane.fallback_points) > 12
                    or not all(_ordinary_vector(point, 3) for point in plane.fallback_points)
                )
            )
        ):
            return None
    return VoxelPlaneQueryContext(planes)


def _frame_matches(
    frame: TUMDepthPoseEntry,
    points: list[Vector3],
    query: VoxelPlaneQueryContext | None,
    *,
    voxel_size_m: float,
    correspondence_gate_m: float,
) -> Sequence[_InitialPlane | None] | None:
    if query is None:
        return None
    world = _frame_coordinates(frame, points)
    if world is None:
        return None
    captured = query._nearest_geometries(
        world, voxel_size_m=voxel_size_m, correspondence_gate_m=correspondence_gate_m
    )
    return (
        captured
        if captured is not None
        else query.nearest_matches(
            world, voxel_size_m=voxel_size_m, correspondence_gate_m=correspondence_gate_m
        )
    )


def _pair_matches(
    points: tuple[Vector3, ...],
    transform: SE3,
    query: VoxelPlaneQueryContext | None,
    *,
    voxel_size_m: float,
    correspondence_gate_m: float,
) -> list[VoxelPlaneMatch | None] | None:
    if query is None:
        return None
    coordinates = _coordinates(points, transform)
    return (
        None
        if coordinates is None
        else query.nearest_matches(
            coordinates, voxel_size_m=voxel_size_m, correspondence_gate_m=correspondence_gate_m
        )
    )
