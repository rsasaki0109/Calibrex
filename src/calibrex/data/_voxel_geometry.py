"""Bounded immutable voxel geometry with scalar accumulation and tie order."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from calibrex.data.livox import _PlaneGeometry


def _cross(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.stack((
        left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
        left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
        left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
    ), axis=-1)


def _norm(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(
        values[..., 0] * values[..., 0] + values[..., 1] * values[..., 1]
        + values[..., 2] * values[..., 2]
    )


def _point_plane_geometry(
    points: NDArray[np.float64], voxel_size_m: float,
) -> dict[tuple[int, int, int], _PlaneGeometry] | None:
    if (
        type(points) is not np.ndarray or points.dtype != np.float64
        or points.ndim != 2 or points.shape[1] != 3 or not 64 <= len(points) <= 65536
        or type(voxel_size_m) is not float or not math.isfinite(voxel_size_m)
        or voxel_size_m <= 0.0 or not np.isfinite(points).all()
        or np.any(np.abs(points) > 1e30)
    ):
        return None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cells = np.floor(points / voxel_size_m)
    if (not np.isfinite(cells).all() or np.any(cells < -(2.0**63) + 2048)
        or np.any(cells >= 2.0**63 - 2048)):
        return None
    keys, first, inverse, counts = np.unique(
        cells.astype(np.int64), axis=0, return_index=True,
        return_inverse=True, return_counts=True,
    )
    # Preserve the scalar accumulator's initial +0 and input addition order.
    sums = np.zeros((len(keys), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    centroids = sums / counts[:, None]
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    result: dict[tuple[int, int, int], _PlaneGeometry] = {}
    # The scalar plane retains only the first twelve points. Keep that prefix,
    # sqrt before argmax, and first-tie selection even when sqrt rounds a tie.
    for begin in range(0, len(keys), 4096):
        rows = np.arange(begin, min(begin + 4096, len(keys)))
        rows = rows[counts[rows] >= 3]
        if not len(rows):
            continue
        positions = starts[rows, None] + np.minimum(
            np.arange(12), counts[rows, None] - 1,
        )
        samples = points[order[positions]]
        relative = samples - samples[:, :1]
        candidates = relative[:, 1:]
        selected = np.arange(len(rows))
        first_vector = candidates[selected, np.argmax(_norm(candidates), axis=1)]
        crosses = _cross(first_vector[:, None], candidates)
        second_vector = candidates[selected, np.argmax(_norm(crosses), axis=1)]
        normals = _cross(first_vector, second_vector)
        lengths = _norm(normals)
        valid = lengths > 1e-12
        for row, normal in zip(
            rows[valid].tolist(), (normals[valid] / lengths[valid, None]).tolist(), strict=True,
        ):
            x, y, z = keys[row].tolist()
            cx, cy, cz = centroids[row].tolist()
            nx, ny, nz = normal
            result[(x, y, z)] = _PlaneGeometry((cx, cy, cz), (nx, ny, nz))
    # Match the original map's insertion order, independently of voxel sorting.
    return {tuple(keys[row].tolist()): result[tuple(keys[row].tolist())]
            for row in np.argsort(first).tolist() if tuple(keys[row].tolist()) in result}
