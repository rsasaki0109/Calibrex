"""Research-only local depth-boundary tangents; uncertain neighborhoods stay neutral."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from tools import kitti_camera_lidar_orientation as oriented

from calibrex.evaluation import camera_lidar_edge as edge

SETTINGS = {
    "neighbors": 8,
    "radius_m": 0.75,
    "minimum_neighbors": 5,
    "maximum_second_eigenvalue_ratio": 0.2,
}


def boundary_tangents(points: NDArray[np.floating]) -> NDArray[np.float32]:
    """Fit local lines to prepared depth edges, without poses or image scores."""
    result = np.zeros_like(points, dtype=np.float32)
    if len(points) < 5:
        return result
    distances, indices = cKDTree(points).query(points, k=8, distance_upper_bound=0.75)
    for i, (distance, index) in enumerate(zip(distances, indices, strict=True)):
        neighbors = points[index[np.isfinite(distance)]].astype(np.float64)
        if len(neighbors) < 5:
            continue
        centered = neighbors - neighbors.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered)
        if values[-1] > 1e-12 and values[-2] / values[-1] <= 0.2:
            result[i] = vectors[:, -1]
    return result


class BoundaryObjective(oriented.OrientationObjective):
    """Compare projected boundary tangent to the perpendicular image gradient."""

    def __init__(
        self,
        frames: Sequence[edge.PreparedFrame],
        camera: edge.EdgeCamera,
        start: NDArray[np.float64],
        features: Sequence[oriented.OrientationFeatures],
        tangents: Sequence[NDArray[np.float32]],
    ) -> None:
        if len(tangents) != len(frames) or any(
            tangent.shape != frame.points.shape or not np.isfinite(tangent).all()
            for frame, tangent in zip(frames, tangents, strict=True)
        ):
            raise ValueError("one finite tangent per prepared point is required")
        rotated = []
        for feature in features:
            tensor = feature.tensor[..., [2, 1, 0]].copy()
            tensor[..., 1] *= -1
            rotated.append(oriented.OrientationFeatures(feature.raw_response, tensor))
        super().__init__(frames, camera, start, rotated)
        self._azimuth = np.concatenate(tangents).astype(np.float32) @ start[:3, :3].T
