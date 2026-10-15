"""Optional fused CPU covariance preparation; PCA still uses native NumPy eigh.

Only covariance assembly is compiled. Neighbour membership and ordering stay
with SciPy, and FP64 additions follow the native neighbour order without FMA
contraction or reassociation. No process-wide thread settings are changed.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray

from calibrex.solvers.scan_to_scan_odometry import (
    FloatArray,
    ScanOdometryOptions,
    _normals_from_neighbours,
)


def execution_identity() -> str:
    """Identify the covariance adapter, compiler and exact executed source bytes."""
    return (
        f"calibrex.numba_scan_covariance/v0.1;numba={version('numba')};"
        "fastmath=false;parallel=false;eigh=numpy;"
        f"source_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"
    )


class NumbaNormalPreparer:
    """Prepare one target's normal rows and report actual compiled/fallback work."""

    def __init__(self) -> None:
        self.execution_backend = execution_identity()
        self.calls = self.rows = self.fallback_rows = 0

    def prepare(
        self, points: FloatArray, indices: NDArray[np.intp], k: int,
        options: ScanOdometryOptions,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Fuse the FP64 gather, mean and six distinct covariance accumulations."""
        # Float32/custom-array arithmetic has its own native rounding semantics.
        if (type(points) is not np.ndarray or points.dtype != np.float64
                or indices.ndim != 2 or indices.shape[1] != k or k < 2):
            self.fallback_rows += len(indices)
            return _normals_from_neighbours(points, indices, k, options)
        covariance = _covariance(points, indices)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        self.calls += 1
        self.rows += len(indices)
        planar = eigenvalues[:, 0] <= options.max_planarity_ratio * np.maximum(
            eigenvalues[:, 1], 1e-12,
        )
        return eigenvectors[:, :, 0], planar

    def statistics(self) -> dict[str, int]:
        """Return completed work; the adapter uses one calling CPU thread."""
        return {
            "numba_normal_covariance_calls": self.calls,
            "numba_normal_covariance_rows": self.rows,
            "native_normal_covariance_fallback_rows": self.fallback_rows,
        }


@njit(cache=True, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _covariance(points: FloatArray, indices: NDArray[np.intp]) -> FloatArray:
    n, k = indices.shape
    result = np.empty((n, 3, 3), dtype=np.float64)
    for row in range(n):
        x = y = z = 0.0
        for column in range(k):
            point = indices[row, column]
            x += points[point, 0]
            y += points[point, 1]
            z += points[point, 2]
        x /= k
        y /= k
        z /= k
        aa = bb = cc = ab = ac = bc = 0.0
        for column in range(k):
            point = indices[row, column]
            a, b, c = points[point, 0] - x, points[point, 1] - y, points[point, 2] - z
            aa += a * a
            bb += b * b
            cc += c * c
            ab += a * b
            ac += a * c
            bc += b * c
        result[row, 0, 0] = aa / k
        result[row, 1, 1] = bb / k
        result[row, 2, 2] = cc / k
        result[row, 0, 1] = result[row, 1, 0] = ab / k
        result[row, 0, 2] = result[row, 2, 0] = ac / k
        result[row, 1, 2] = result[row, 2, 1] = bc / k
    return result
