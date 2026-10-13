"""Scoped spatial searches with the native NumPy distance and tie ordering."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class _ExactNeighbors:
    """Own one immutable reference cloud; never retain it beyond an invocation.

    The tree only proposes candidates. Distances are recomputed with the
    original three-coordinate NumPy reduction. A guarded boundary search
    resolves duplicates and near ties by the original reference index.
    Non-finite or extreme coordinates use bounded exhaustive row batches.
    """

    def __init__(self, reference: FloatArray) -> None:
        self._reference = np.array(reference, dtype=np.float64, order="C", copy=True)
        self._reference.flags.writeable = False
        self._tree: cKDTree | None = None
        if len(self._reference) and _tree_safe(self._reference):
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self._reference)

    def nearest(self, points: FloatArray) -> tuple[IntArray, FloatArray]:
        indices, squared = self.query(points, 1)
        return indices[:, 0], squared[:, 0]

    def query(self, points: FloatArray, count: int) -> tuple[IntArray, FloatArray]:
        if count < 1 or not len(self._reference):
            raise ValueError("nearest search requires a positive count and reference cloud")
        count = min(count, len(self._reference))
        if not len(points):
            return np.empty((0, count), dtype=np.int64), np.empty((0, count), dtype=np.float64)
        tree = self._tree
        if tree is None or not _tree_safe(points):
            return self._exhaustive(points, count)
        width = min(count + 1, len(self._reference))
        _, candidates = tree.query(points, k=list(range(1, width + 1)), eps=0.0)
        candidates = np.asarray(candidates, dtype=np.int64)
        squared = np.sum((points[:, None, :] - self._reference[candidates]) ** 2, axis=2)
        order = np.lexsort((candidates, squared), axis=1)
        candidates = np.take_along_axis(candidates, order, axis=1)
        squared = np.take_along_axis(squared, order, axis=1)
        indices = candidates[:, :count].copy()
        distances = squared[:, :count].copy()
        if width == count:
            return indices, distances

        cutoff = squared[:, count - 1]
        guard = np.maximum(64.0 * np.finfo(np.float64).eps * cutoff, np.finfo(np.float64).tiny)
        ambiguous = np.flatnonzero(squared[:, count] <= cutoff + guard)
        if not len(ambiguous):
            return indices, distances
        radii = np.nextafter(np.sqrt(cutoff[ambiguous] + guard[ambiguous]), np.inf)
        # Resolve rows independently, so duplicate-heavy clouds do not retain
        # an N-by-M list of candidates. Work remains exhaustive in that case.
        for row, radius in zip(ambiguous, radii, strict=True):
            nearby = np.asarray(tree.query_ball_point(points[row], radius, eps=0.0), dtype=np.int64)
            nearby = np.union1d(nearby, candidates[row])
            exact = np.sum((points[row] - self._reference[nearby]) ** 2, axis=1)
            selected = np.lexsort((nearby, exact))[:count]
            indices[row] = nearby[selected]
            distances[row] = exact[selected]
        return indices, distances

    def _exhaustive(self, points: FloatArray, count: int) -> tuple[IntArray, FloatArray]:
        indices = np.empty((len(points), count), dtype=np.int64)
        distances = np.empty((len(points), count), dtype=np.float64)
        rows = max(1, 65536 // len(self._reference))
        for start in range(0, len(points), rows):
            stop = start + rows
            squared = np.sum(
                (points[start:stop, None, :] - self._reference[None, :, :]) ** 2, axis=2,
            )
            # argmin's NaN behavior differs from argsort; preserve both callers.
            selected = (
                np.argmin(squared, axis=1)[:, None] if count == 1 else
                np.argsort(squared, axis=1, kind="stable")[:, :count]
            )
            indices[start:stop] = selected
            distances[start:stop] = np.take_along_axis(squared, selected, axis=1)
        return indices, distances


def _tree_safe(points: FloatArray) -> bool:
    # Keep squared distances and the boundary guard far below float64 overflow.
    return bool(np.isfinite(points).all() and np.max(np.abs(points), initial=0.0) <= 1.0e150)
