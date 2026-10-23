"""Fixed candidate-image tile weights for spent-data edge-objective experiments."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from pydantic import Field

from calibrex.core.result import StrictModel
from calibrex.evaluation import camera_lidar_edge as edge


class TileSupport(StrictModel):
    time_s: float
    point_counts: list[int] = Field(min_length=4, max_length=4)
    original_weight: list[float] = Field(min_length=4, max_length=4)
    balanced_weight: list[float] = Field(min_length=4, max_length=4)


def balance_frame(
    frame: edge.PreparedFrame, camera: edge.EdgeCamera, candidate: np.ndarray,
) -> tuple[edge.PreparedFrame, TileSupport]:
    """Give occupied 2x2 tiles equal mass, preserving each frame's original total.

    Tiles are assigned once at the candidate, before any optimization. Weights
    stay fixed when trial poses move points; no frame or tile is selected by a
    fitted score, held-out outcome, injection label or reference calibration.
    """
    if candidate.shape != (4, 4) or not np.all(np.isfinite(candidate)):
        raise ValueError("candidate must be a finite 4x4 matrix")
    if (len(frame.points) == 0 or len(frame.weights) != len(frame.points)
            or not np.all(np.isfinite(frame.weights)) or np.any(frame.weights <= 0)):
        raise ValueError("need nonempty points with positive finite matching weights")
    if frame.scale <= 0 or frame.edge.ndim != 2 or min(frame.edge.shape) < 2:
        raise ValueError("invalid prepared image geometry")
    points = frame.points.astype(np.float64) @ candidate[:3, :3].T + candidate[:3, 3]
    u, v, valid = camera.project(points)
    u = (u - 0.5 * (frame.scale - 1)) / frame.scale
    v = (v - 0.5 * (frame.scale - 1)) / frame.scale
    height, width = frame.edge.shape
    inside = (valid & np.isfinite(u) & np.isfinite(v)
              & (u >= 0) & (v >= 0) & (u < width - 1) & (v < height - 1))
    if not np.all(inside):
        raise ValueError("prepared points must project inside the candidate image")
    tile = (u >= width / 2).astype(np.int64) + 2 * (v >= height / 2).astype(np.int64)
    counts = np.bincount(tile, minlength=4)
    mass = np.bincount(tile, weights=frame.weights.astype(np.float64), minlength=4)
    occupied = int(np.count_nonzero(counts))
    balanced = (frame.weights.astype(np.float64) / mass[tile]
                * (mass.sum() / occupied)).astype(np.float32)
    balanced.setflags(write=False)
    return replace(frame, weights=balanced), TileSupport(
        time_s=frame.time_s, point_counts=counts.tolist(), original_weight=mass.tolist(),
        balanced_weight=np.bincount(tile, weights=balanced.astype(np.float64),
                                    minlength=4).tolist(),
    )
