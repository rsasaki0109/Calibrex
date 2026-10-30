"""Camera-only temporal disparity checks with explicit visibility categories."""

from __future__ import annotations

import numpy as np

STATUS = {
    0: "invalid_projection",
    1: "out_of_view",
    2: "missing_depth",
    3: "consistent",
    4: "nearer_observed_surface",
    5: "farther_observed_surface",
    6: "unsupported_motion",
}


def classify(
    points: np.ndarray,
    target_depth: np.ndarray,
    transform: np.ndarray,
    intrinsics: list[float],
    fb: float,
) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    u = fx * xyz[:, 0] / np.maximum(xyz[:, 2], 1e-9) + cx
    v = fy * xyz[:, 1] / np.maximum(xyz[:, 2], 1e-9) + cy
    h, w = target_depth.shape
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.1)
    inside = valid & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    result = np.zeros(len(points), dtype=np.uint8)
    result[valid] = 1
    result[inside] = 2
    ids = np.flatnonzero(inside)
    x = np.floor(u[ids]).astype(int)
    y = np.floor(v[ids]).astype(int)
    values = np.stack(
        [
            target_depth[y, x],
            target_depth[y, x + 1],
            target_depth[y + 1, x],
            target_depth[y + 1, x + 1],
        ],
        axis=1,
    )
    supported = np.isfinite(values).all(axis=1) & (values > 0.1).all(axis=1)
    a = u[ids] - x
    b = v[ids] - y
    weights = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], axis=1)
    observed = np.sum(np.where(np.isfinite(values), values, 0) * weights, axis=1)
    delta = fb / np.maximum(observed, 1e-9) - fb / xyz[ids, 2]
    labels = np.where(np.abs(delta) <= 1.0, 3, np.where(delta > 1, 4, 5)).astype(np.uint8)
    result[ids[supported]] = labels[supported]
    return result
