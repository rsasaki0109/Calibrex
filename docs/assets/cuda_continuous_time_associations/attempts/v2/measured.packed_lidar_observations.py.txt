"""Private native correspondence batches for the optional compiled LiDAR solver."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import Vector3
from calibrex.data.adaptive_voxel import AdaptivePlaneQueryContext
from calibrex.data.livox import VoxelPlaneQueryContext
from calibrex.graph.lidar_point_to_plane import (
    _native_packing_supported,
    _normalize3,
    _ObservationArrays,
    _PackedObservations,
)

FloatArray: TypeAlias = NDArray[np.float64]
_NATIVE_ADAPTIVE_QUERY = AdaptivePlaneQueryContext.nearest_planes
_NATIVE_UNIFORM_QUERY = VoxelPlaneQueryContext.nearest_matches


@dataclass(frozen=True)
class _PlaneColumns:
    centroids: FloatArray
    raw_normals: FloatArray
    normals: FloatArray
    valid: NDArray[np.bool_]


def _columns(vectors: Sequence[tuple[Vector3, Vector3 | None]]) -> _PlaneColumns | None:
    if any(type(vector) is not tuple or len(vector) != 3
           or any(type(value) is not float or not math.isfinite(value) or abs(value) >= 1e100
                  for value in vector)
           for pair in vectors for vector in pair if vector is not None):
        return None
    try:
        centroids = np.asarray([pair[0] for pair in vectors], dtype=np.float64).reshape((-1, 3))
        raw = np.asarray([pair[1] or (0.0,)*3 for pair in vectors],
                         dtype=np.float64).reshape((-1, 3))
        normals = np.asarray([_normalize3(pair[1]) if pair[1] is not None else (0.0,)*3
                              for pair in vectors], dtype=np.float64).reshape((-1, 3))
    except ValueError:
        return None
    valid = np.asarray([pair[1] is not None for pair in vectors], dtype=np.bool_)
    for array in (centroids, raw, normals, valid):
        array.flags.writeable = False
    return _PlaneColumns(centroids, raw, normals, valid)


class _PackedObservationBuilder:
    """Capture point/plane columns once and retain ordered native association rules."""

    def __init__(
        self, points: FloatArray, uniform: VoxelPlaneQueryContext | None,
        adaptive: AdaptivePlaneQueryContext | None, uniform_columns: _PlaneColumns | None,
        adaptive_columns: _PlaneColumns | None,
    ) -> None:
        self.points = points
        self.uniform = uniform
        self.adaptive = adaptive
        self.uniform_columns = uniform_columns
        self.adaptive_columns = adaptive_columns

    @classmethod
    def capture(
        cls, points: Sequence[Vector3], uniform: VoxelPlaneQueryContext | None,
        adaptive: AdaptivePlaneQueryContext | None,
    ) -> _PackedObservationBuilder | None:
        if (not _native_packing_supported() or type(points) not in (tuple, list)
            or any(type(point) is not tuple or len(point) != 3
                   or any(type(value) is not float or not math.isfinite(value)
                          or abs(value) >= 1e100 for value in point) for point in points)
            or (uniform is not None and
                (type(uniform) is not VoxelPlaneQueryContext or not uniform._native_geometry))
            or (adaptive is not None and
                (type(adaptive) is not AdaptivePlaneQueryContext
                 or not adaptive._native_geometry))):
            return None
        left = _columns([(uniform._planes[key].centroid, uniform._planes[key].normal)
                         for key in uniform._keys]) if uniform is not None else None
        right = _columns([(plane.centroid, plane.normal) for plane in adaptive._planes]) \
            if adaptive is not None else None
        if ((uniform is not None and left is None) or (adaptive is not None and right is None)):
            return None
        xyz = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        xyz.flags.writeable = False
        return cls(xyz, uniform, adaptive, left, right)

    def build(
        self, indices: NDArray[np.int64], poses: FloatArray, world: FloatArray,
        *, voxel_size_m: float, correspondence_gate_m: float,
        _position_query: Callable[..., tuple[NDArray[np.int64] | None,
                                             NDArray[np.int64] | None] | None] | None = None,
    ) -> _PackedObservations | None:
        if (len(world) < 64 or not _native_packing_supported()
            or np.any(np.abs(world) >= 1e100) or not np.isfinite(world).all()
            or (self.uniform is not None and
                ("nearest_matches" in self.uniform.__dict__
                 or type(self.uniform).nearest_matches is not _NATIVE_UNIFORM_QUERY))
            or (self.adaptive is not None and
                ("nearest_planes" in self.adaptive.__dict__
                 or type(self.adaptive).nearest_planes is not _NATIVE_ADAPTIVE_QUERY))):
            return None
        matches = []
        columns = []
        queried = (_position_query(world, voxel_size_m=voxel_size_m,
                                   correspondence_gate_m=correspondence_gate_m)
                   if _position_query is not None else None)
        if self.uniform is not None:
            positions = queried[0] if queried is not None else self.uniform._nearest_positions(
                world, voxel_size_m=voxel_size_m, correspondence_gate_m=correspondence_gate_m,
            )
            if positions is None:
                return None
            assert self.uniform_columns is not None
            matches.append(positions)
            columns.append(self.uniform_columns)
        if self.adaptive is not None:
            positions = queried[1] if queried is not None else self.adaptive._nearest_positions(
                world, correspondence_gate_m=correspondence_gate_m,
            )
            if positions is None:
                return None
            assert self.adaptive_columns is not None
            matches.append(positions)
            columns.append(self.adaptive_columns)
        if not matches or any(len(column.centroids) == 0 for column in columns):
            return None
        centroids = [column.centroids[np.maximum(position, 0)]
                     for position, column in zip(matches, columns, strict=True)]
        normals = [column.normals[np.maximum(position, 0)]
                   for position, column in zip(matches, columns, strict=True)]
        present = [(position >= 0) & column.valid[np.maximum(position, 0)]
                   for position, column in zip(matches, columns, strict=True)]
        chosen_centroids, chosen_normals = centroids[0], normals[0]
        valid = present[0]
        if len(matches) == 2:
            scores = []
            for position, column, centroid in zip(matches, columns, centroids, strict=True):
                raw = column.raw_normals[np.maximum(position, 0)]
                delta = world-centroid
                scores.append(np.abs(raw[:, 0]*delta[:, 0] + raw[:, 1]*delta[:, 1]
                                     + raw[:, 2]*delta[:, 2]))
            if not all(np.isfinite(score).all() for score in scores):
                return None
            right = present[1] & (~present[0] | (scores[1] < scores[0]))
            for row in np.flatnonzero(present[0] & present[1] & (scores[0] == scores[1])):
                point = tuple(world[row].tolist())
                right[row] = (math.dist(point, centroids[1][row].tolist())
                              < math.dist(point, centroids[0][row].tolist()))
            chosen_centroids = np.where(right[:, None], centroids[1], centroids[0])
            chosen_normals = np.where(right[:, None], normals[1], normals[0])
            valid = present[0] | present[1]
        arrays = [self.points[indices[valid]], chosen_centroids[valid], chosen_normals[valid],
                  poses[valid], np.ones(int(np.count_nonzero(valid)), dtype=np.float64)]
        for array in arrays:
            array.flags.writeable = False
        return _PackedObservations(_ObservationArrays(
            arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], None,
        ))
