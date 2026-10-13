"""Optional Numba adapter for exact fused D2D projection.

The adapter fuses camera projection, rounded-pixel validation, stable nearest
visibility, and histogram accumulation. Prepared frames freeze features and
bin assignments; thread-local scratch visits and clears only touched pixels.
Pair-array access remains available. Numba is optional and lazily imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import version
from threading import Lock, local
from typing import Final, TypeAlias

import numpy as np
from numba import njit
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthPairHistogram,
    DepthPairProjection,
    DepthToDepthObservation,
    DepthToDepthOptions,
    _rotation_matrix,
    project_depth_pairs,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

NUMBA_DEPTH_PAIR_PROJECTOR_VERSION: Final = (
    "calibrex.numba_depth_pair_projector/v0.2"
)
_PINHOLE: Final = 0
_DOUBLE_SPHERE: Final = 1
_MEI: Final = 2


def numba_depth_pair_projector_identity() -> str:
    """Return a provenance identity including the loaded Numba version."""

    return f"{NUMBA_DEPTH_PAIR_PROJECTOR_VERSION};numba={version('numba')};nogil=true"


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


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
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
        pixel, camera_range = _project_pixel_and_range(
            points[index, 0], points[index, 1], points[index, 2],
            rotation, translation, width, height, fx, fy, cx, cy,
            projection_kind, xi, alpha, k1, k2, p1, p2,
        )
        if pixel < 0:
            continue
        projected_count += 1
        if best_index[pixel] == sentinel or camera_range < best_range[pixel]:
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


@njit(cache=True, inline="always")  # type: ignore[untyped-decorator]
def _project_pixel_and_range(
    point_x: float, point_y: float, point_z: float,
    rotation: FloatArray, translation: FloatArray,
    width: int, height: int, fx: float, fy: float, cx: float, cy: float,
    projection_kind: int, xi: float, alpha: float,
    k1: float, k2: float, p1: float, p2: float,
) -> tuple[int, float]:
    x = (point_x * rotation[0, 0] + point_y * rotation[0, 1]
         + point_z * rotation[0, 2] + translation[0])
    y = (point_x * rotation[1, 0] + point_y * rotation[1, 1]
         + point_z * rotation[1, 2] + translation[1])
    z = (point_x * rotation[2, 0] + point_y * rotation[2, 1]
         + point_z * rotation[2, 2] + translation[2])
    camera_range = 0.0
    if projection_kind == _PINHOLE:
        if not z > 1.0e-9:
            return -1, 0.0
        u, v = fx * x / z + cx, fy * y / z + cy
    elif projection_kind == _DOUBLE_SPHERE:
        camera_range = math.sqrt(x * x + y * y + z * z)
        z1 = xi * camera_range + z
        d2 = math.sqrt(x * x + y * y + z1 * z1)
        denominator = alpha * d2 + (1.0 - alpha) * z1
        w1 = alpha / (1.0 - alpha) if alpha <= 0.5 else (1.0 - alpha) / alpha
        w2 = (w1 + xi) / math.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
        if not (camera_range > 1.0e-9 and denominator > 1.0e-9 and z > -w2 * camera_range):
            return -1, 0.0
        u, v = fx * x / denominator + cx, fy * y / denominator + cy
    else:
        camera_range = math.sqrt(x * x + y * y + z * z)
        if not (camera_range > 1.0e-9 and z > 1.0e-9):
            return -1, 0.0
        denominator = z / camera_range + xi
        if not denominator > 1.0e-9:
            return -1, 0.0
        normalized_x = (x / camera_range) / denominator
        normalized_y = (y / camera_range) / denominator
        radius_squared = normalized_x * normalized_x + normalized_y * normalized_y
        radial = 1.0 + k1 * radius_squared + k2 * radius_squared * radius_squared
        distorted_x = (normalized_x * radial + 2.0 * p1 * normalized_x * normalized_y
                       + p2 * (radius_squared + 2.0 * normalized_x * normalized_x))
        distorted_y = (
            normalized_y * radial + p1 * (radius_squared + 2.0 * normalized_y * normalized_y)
            + 2.0 * p2 * normalized_x * normalized_y
        )
        u, v = fx * distorted_x + cx, fy * distorted_y + cy
    if not (math.isfinite(u) and math.isfinite(v)):
        return -1, 0.0
    pixel_u, pixel_v = round(u), round(v)
    if pixel_u < 0 or pixel_u >= width or pixel_v < 0 or pixel_v >= height:
        return -1, 0.0
    if projection_kind == _PINHOLE:
        camera_range = math.sqrt(x * x + y * y + z * z)
    return pixel_v * width + pixel_u, camera_range


@dataclass(frozen=True)
class _HistogramInputs:
    observation: DepthToDepthObservation
    camera_bins: IntArray
    lidar_bins: IntArray


@dataclass(frozen=True)
class _HistogramWorkspace:
    best_index: IntArray
    best_range: FloatArray
    touched_pixels: IntArray


class NumbaDepthPairProjector:
    """Exact fused CPU histograms with explicit snapshots and thread-local scratch.

    Prepared observations are immutable snapshots. One histogram configuration
    per prepared frame is retained; other observations are evaluated without
    retaining them. Scratch grows to the largest encountered image per thread.
    """

    def __init__(
        self, observations: tuple[DepthToDepthObservation, ...] = (),
        *, histogram_options: DepthToDepthOptions | None = None,
    ) -> None:
        self._prepared: dict[int, tuple[DepthToDepthObservation, DepthToDepthObservation]] = {}
        self._histograms: dict[
            int, tuple[tuple[int, tuple[float, float], tuple[float, float]], _HistogramInputs]
        ] = {}
        self._lock = Lock()
        self._local = local()
        for observation in observations:
            if id(observation) in self._prepared:
                continue
            snapshot = DepthToDepthObservation(
                frame_id=observation.frame_id, camera=observation.camera,
                depth_scale_convention=observation.depth_scale_convention,
                depth_map=observation.depth_map.copy(order="C"),
                lidar_points=observation.lidar_points.copy(order="C"),
            )
            # Preserve the already-frozen source ranges, including callers that
            # prepared a point array independently of its cached range feature.
            object.__setattr__(snapshot, "lidar_range_m", observation.lidar_range_m.copy())
            for array in (snapshot.depth_map, snapshot.lidar_points, snapshot.lidar_range_m):
                array.setflags(write=False)
            self._prepared[id(observation)] = (observation, snapshot)
            if (histogram_options is not None and histogram_options.camera_depth_range is not None
                    and histogram_options.lidar_range_m is not None):
                self._histogram_inputs(
                    observation, histogram_options.histogram_bins,
                    histogram_options.camera_depth_range, histogram_options.lidar_range_m,
                )

    def identity(self) -> str:
        """Identify fused histograms, scratch policy, and the dependency version."""

        return numba_depth_pair_projector_identity() + ";histogram=sparse_prebinned;cache=snapshot"

    def __call__(
        self, observation: DepthToDepthObservation, transform_camera_lidar: SE3,
        *, use_z_buffer: bool = True,
    ) -> DepthPairProjection:
        """Retain pair-array access for consumers that need individual samples."""

        return project_depth_pairs_numba(
            self._snapshot(observation), transform_camera_lidar, use_z_buffer=use_z_buffer,
        )

    def histogram_depth_pairs(
        self, observation: DepthToDepthObservation, transform_camera_lidar: SE3,
        *, use_z_buffer: bool, bins: int,
        camera_range: tuple[float, float], lidar_range: tuple[float, float],
    ) -> DepthPairHistogram:
        """Project directly to counts without materializing paired feature arrays."""

        inputs = self._histogram_inputs(observation, bins, camera_range, lidar_range)
        item = inputs.observation
        camera = item.camera
        pixel_count = camera.width * camera.height
        workspace: _HistogramWorkspace | None = getattr(self._local, "workspace", None)
        if workspace is None or workspace.best_index.size < pixel_count:
            workspace = _HistogramWorkspace(
                np.full(pixel_count, -1, dtype=np.int64),
                np.empty(pixel_count, dtype=np.float64),
                np.empty(pixel_count, dtype=np.int64),
            )
            self._local.workspace = workspace
        distortion = (*camera.distortion, 0.0, 0.0, 0.0, 0.0)
        try:
            histogram, visible, projected = _sparse_histogram_fused(
                item.lidar_points, inputs.camera_bins, inputs.lidar_bins,
                _rotation_matrix(transform_camera_lidar.rotation_quat_xyzw),
                np.asarray(transform_camera_lidar.translation_m, dtype=np.float64),
                camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy,
                {"pinhole": _PINHOLE, "double_sphere": _DOUBLE_SPHERE,
                 "mei": _MEI}[camera.projection],
                camera.xi, camera.alpha, *distortion[:4], bins, use_z_buffer,
                workspace.best_index, workspace.best_range, workspace.touched_pixels,
            )
        except BaseException:
            # A failed native call must not leave stale selected pixels behind.
            self._local.workspace = None
            raise
        return DepthPairHistogram(histogram.astype(np.float64), int(visible), int(projected))

    def _snapshot(self, observation: DepthToDepthObservation) -> DepthToDepthObservation:
        prepared = self._prepared.get(id(observation))
        return prepared[1] if prepared is not None else observation

    def _histogram_inputs(
        self, observation: DepthToDepthObservation, bins: int,
        camera_range: tuple[float, float], lidar_range: tuple[float, float],
    ) -> _HistogramInputs:
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError("histogram bins must be a positive integer")
        if not all(
            len(limits) == 2 and all(math.isfinite(value) for value in limits)
            and limits[0] < limits[1] for limits in (camera_range, lidar_range)
        ):
            raise ValueError("histogram ranges must be finite and increasing")
        key = (bins, camera_range, lidar_range)
        prepared = id(observation) in self._prepared
        with self._lock:
            cached = self._histograms.get(id(observation)) if prepared else None
            if cached is not None and cached[0] == key:
                return cached[1]
            snapshot = self._snapshot(observation)
            inputs = _HistogramInputs(
                snapshot, _feature_bins(snapshot.depth_map.ravel(), bins, camera_range),
                _feature_bins(snapshot.lidar_range_m, bins, lidar_range),
            )
            if prepared:
                self._histograms[id(observation)] = (key, inputs)
            return inputs


def _feature_bins(values: FloatArray, bins: int, limits: tuple[float, float]) -> IntArray:
    edges = np.linspace(*limits, bins + 1)
    indices: IntArray = np.asarray(np.searchsorted(edges, values, side="right") - 1, dtype=np.int64)
    indices[values == edges[-1]] -= 1
    indices[(values < edges[0]) | (values > edges[-1])] = -1
    indices[~np.isfinite(values) | (values <= 0.0)] = -2
    indices.setflags(write=False)
    return indices


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _sparse_histogram_fused(
    points: FloatArray, camera_bins: IntArray, lidar_bins: IntArray,
    rotation: FloatArray, translation: FloatArray,
    width: int, height: int, fx: float, fy: float, cx: float, cy: float,
    projection_kind: int, xi: float, alpha: float,
    k1: float, k2: float, p1: float, p2: float,
    bins: int, use_z_buffer: bool,
    best_index: IntArray, best_range: FloatArray, touched_pixels: IntArray,
) -> tuple[IntArray, int, int]:
    histogram: IntArray = np.zeros((bins, bins), dtype=np.int64)
    touched_count = visible_count = projected_count = 0
    for index in range(points.shape[0]):
        pixel, camera_range = _project_pixel_and_range(
            points[index, 0], points[index, 1], points[index, 2],
            rotation, translation, width, height, fx, fy, cx, cy,
            projection_kind, xi, alpha, k1, k2, p1, p2,
        )
        if pixel < 0:
            continue
        projected_count += 1
        if not use_z_buffer:
            camera_bin, lidar_bin = camera_bins[pixel], lidar_bins[index]
            if camera_bin >= -1 and lidar_bin >= -1:
                visible_count += 1
                if camera_bin >= 0 and lidar_bin >= 0:
                    histogram[camera_bin, lidar_bin] += 1
        elif best_index[pixel] < 0:
            touched_pixels[touched_count] = pixel
            touched_count += 1
            best_index[pixel], best_range[pixel] = index, camera_range
        elif camera_range < best_range[pixel]:
            best_index[pixel], best_range[pixel] = index, camera_range
    # Only pixels touched by this scan need inspection and clearing. Each
    # thread owns its scratch; ties retain the earliest input point.
    for cursor in range(touched_count):
        pixel = touched_pixels[cursor]
        index = best_index[pixel]
        best_index[pixel] = -1
        camera_bin, lidar_bin = camera_bins[pixel], lidar_bins[index]
        if camera_bin >= -1 and lidar_bin >= -1:
            visible_count += 1
            if camera_bin >= 0 and lidar_bin >= 0:
                histogram[camera_bin, lidar_bin] += 1
    return histogram, visible_count, projected_count
