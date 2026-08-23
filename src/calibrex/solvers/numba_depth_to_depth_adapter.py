"""Optional Numba adapter for exact fused D2D projection.

The adapter fuses camera projection, rounded-pixel validation, and stable
nearest-range visibility into one compiled CPU pass.  It preserves the array
ordering and tie semantics of :func:`project_depth_pairs`; Numba remains an
optional dependency and is never imported by the core NumPy path.
"""

from __future__ import annotations

import math
from importlib.metadata import version
from typing import Final, TypeAlias

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthPairProjection,
    DepthToDepthObservation,
    _rotation_matrix,
    project_depth_pairs,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

NUMBA_DEPTH_PAIR_PROJECTOR_VERSION: Final = (
    "calibrex.numba_depth_pair_projector/v0.1"
)
_PINHOLE: Final = 0
_DOUBLE_SPHERE: Final = 1
_MEI: Final = 2


def numba_depth_pair_projector_identity() -> str:
    """Return a provenance identity including the loaded Numba version."""

    return f"{NUMBA_DEPTH_PAIR_PROJECTOR_VERSION};numba={version('numba')}"


def project_depth_pairs_numba(
    observation: DepthToDepthObservation,
    transform_camera_lidar: SE3,
    *,
    use_z_buffer: bool = True,
) -> DepthPairProjection:
    """Project depth pairs with the exact fused Numba CPU adapter."""

    if not use_z_buffer:
        return project_depth_pairs(
            observation,
            transform_camera_lidar,
            use_z_buffer=False,
        )
    projection_kind = {
        "pinhole": _PINHOLE,
        "double_sphere": _DOUBLE_SPHERE,
        "mei": _MEI,
    }[observation.camera.projection]
    camera = observation.camera
    distortion = (*camera.distortion, 0.0, 0.0, 0.0, 0.0)
    rotation = _rotation_matrix(transform_camera_lidar.rotation_quat_xyzw)
    translation: FloatArray = np.asarray(
        transform_camera_lidar.translation_m,
        dtype=np.float64,
    )
    (
        camera_depth,
        lidar_range,
        pixel_u,
        pixel_v,
        projected_count,
    ) = _project_depth_pairs_fused(
        np.ascontiguousarray(observation.lidar_points, dtype=np.float64),
        np.ascontiguousarray(observation.depth_map, dtype=np.float64),
        np.ascontiguousarray(observation.lidar_range_m, dtype=np.float64),
        rotation,
        translation,
        camera.width,
        camera.height,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        projection_kind,
        camera.xi,
        camera.alpha,
        distortion[0],
        distortion[1],
        distortion[2],
        distortion[3],
    )
    return DepthPairProjection(
        camera_depth=camera_depth,
        lidar_range_m=lidar_range,
        pixel_u=pixel_u,
        pixel_v=pixel_v,
        projected_count_before_visibility=int(projected_count),
    )


@njit(cache=True)  # type: ignore[untyped-decorator]
def _project_depth_pairs_fused(
    points: FloatArray,
    depth: FloatArray,
    lidar_ranges: FloatArray,
    rotation: FloatArray,
    translation: FloatArray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    projection_kind: int,
    xi: float,
    alpha: float,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
) -> tuple[FloatArray, FloatArray, IntArray, IntArray, int]:
    pixel_count = width * height
    sentinel = points.shape[0]
    best_range: FloatArray = np.full(
        pixel_count,
        np.inf,
        dtype=np.float64,
    )
    best_index: IntArray = np.full(
        pixel_count,
        sentinel,
        dtype=np.int64,
    )
    projected_count = 0

    for index in range(points.shape[0]):
        point_x = points[index, 0]
        point_y = points[index, 1]
        point_z = points[index, 2]
        x = (
            point_x * rotation[0, 0]
            + point_y * rotation[0, 1]
            + point_z * rotation[0, 2]
            + translation[0]
        )
        y = (
            point_x * rotation[1, 0]
            + point_y * rotation[1, 1]
            + point_z * rotation[1, 2]
            + translation[1]
        )
        z = (
            point_x * rotation[2, 0]
            + point_y * rotation[2, 1]
            + point_z * rotation[2, 2]
            + translation[2]
        )

        camera_range = 0.0
        u = 0.0
        v = 0.0
        if projection_kind == _PINHOLE:
            if not z > 1.0e-9:
                continue
            u = fx * x / z + cx
            v = fy * y / z + cy
        elif projection_kind == _DOUBLE_SPHERE:
            camera_range = math.sqrt(x * x + y * y + z * z)
            z1 = xi * camera_range + z
            d2 = math.sqrt(x * x + y * y + z1 * z1)
            denominator = alpha * d2 + (1.0 - alpha) * z1
            w1 = (
                alpha / (1.0 - alpha)
                if alpha <= 0.5
                else (1.0 - alpha) / alpha
            )
            w2 = (w1 + xi) / math.sqrt(
                2.0 * w1 * xi + xi * xi + 1.0
            )
            if not (
                camera_range > 1.0e-9
                and denominator > 1.0e-9
                and z > -w2 * camera_range
            ):
                continue
            u = fx * x / denominator + cx
            v = fy * y / denominator + cy
        else:
            camera_range = math.sqrt(x * x + y * y + z * z)
            if not (camera_range > 1.0e-9 and z > 1.0e-9):
                continue
            denominator = z / camera_range + xi
            if not denominator > 1.0e-9:
                continue
            normalized_x = (x / camera_range) / denominator
            normalized_y = (y / camera_range) / denominator
            radius_squared = (
                normalized_x * normalized_x + normalized_y * normalized_y
            )
            radial = (
                1.0
                + k1 * radius_squared
                + k2 * radius_squared * radius_squared
            )
            distorted_x = (
                normalized_x * radial
                + 2.0 * p1 * normalized_x * normalized_y
                + p2 * (radius_squared + 2.0 * normalized_x * normalized_x)
            )
            distorted_y = (
                normalized_y * radial
                + p1 * (radius_squared + 2.0 * normalized_y * normalized_y)
                + 2.0 * p2 * normalized_x * normalized_y
            )
            u = fx * distorted_x + cx
            v = fy * distorted_y + cy

        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        pixel_u = round(u)
        pixel_v = round(v)
        if pixel_u < 0 or pixel_u >= width or pixel_v < 0 or pixel_v >= height:
            continue
        projected_count += 1
        if projection_kind == _PINHOLE:
            camera_range = math.sqrt(x * x + y * y + z * z)
        pixel = pixel_v * width + pixel_u
        if camera_range < best_range[pixel]:
            best_range[pixel] = camera_range
            best_index[pixel] = index

    visible_count = 0
    for pixel in range(pixel_count):
        selected_index = int(best_index[pixel])
        if selected_index >= sentinel:
            continue
        pixel_v = pixel // width
        pixel_u = pixel - pixel_v * width
        camera_depth = depth[pixel_v, pixel_u]
        lidar_range = lidar_ranges[selected_index]
        if (
            math.isfinite(camera_depth)
            and camera_depth > 0.0
            and math.isfinite(lidar_range)
            and lidar_range > 0.0
        ):
            visible_count += 1

    camera_depth_output: FloatArray = np.empty(
        visible_count,
        dtype=np.float64,
    )
    lidar_range_output: FloatArray = np.empty(
        visible_count,
        dtype=np.float64,
    )
    pixel_u_output: IntArray = np.empty(
        visible_count,
        dtype=np.int64,
    )
    pixel_v_output: IntArray = np.empty(
        visible_count,
        dtype=np.int64,
    )
    cursor = 0
    for pixel in range(pixel_count):
        selected_index = int(best_index[pixel])
        if selected_index >= sentinel:
            continue
        pixel_v = pixel // width
        pixel_u = pixel - pixel_v * width
        camera_depth = depth[pixel_v, pixel_u]
        lidar_range = lidar_ranges[selected_index]
        if not (
            math.isfinite(camera_depth)
            and camera_depth > 0.0
            and math.isfinite(lidar_range)
            and lidar_range > 0.0
        ):
            continue
        camera_depth_output[cursor] = camera_depth
        lidar_range_output[cursor] = lidar_range
        pixel_u_output[cursor] = pixel_u
        pixel_v_output[cursor] = pixel_v
        cursor += 1

    return (
        camera_depth_output,
        lidar_range_output,
        pixel_u_output,
        pixel_v_output,
        projected_count,
    )
