"""Camera-only metric PnP from image matches and stereo depth; no LiDAR input."""

from __future__ import annotations

import cv2
import numpy as np


def metric_pnp(points: np.ndarray, pixels: np.ndarray, k: np.ndarray) -> dict:
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    if len(points) < 6:
        return {"status": "insufficient", "count": len(points), "inliers": 0, "transform": None}
    success, r, t, inliers = cv2.solvePnPRansac(
        np.ascontiguousarray(points, float),
        np.ascontiguousarray(pixels, float),
        k,
        None,
        iterationsCount=1000,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 6:
        return {"status": "failed", "count": len(points), "inliers": 0, "transform": None}
    ids = inliers.ravel()
    r, t = cv2.solvePnPRefineLM(points[ids], pixels[ids], k, None, r, t)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(r)[0]
    transform[:3, 3] = t.ravel()
    return {
        "status": "estimated",
        "count": len(points),
        "inliers": len(ids),
        "inlier_ids": ids.tolist(),
        "transform": transform.tolist(),
    }


def sample_points(
    depth: np.ndarray, pixels: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    u = np.rint(pixels[:, 0]).astype(int)
    v = np.rint(pixels[:, 1]).astype(int)
    valid = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    z = depth[np.clip(v, 0, depth.shape[0] - 1), np.clip(u, 0, depth.shape[1] - 1)]
    valid &= np.isfinite(z) & (z > 0.1) & (z <= 80)
    selected = pixels[valid]
    z = z[valid]
    points = np.column_stack(
        [(selected[:, 0] - k[0, 2]) * z / k[0, 0], (selected[:, 1] - k[1, 2]) * z / k[1, 1], z]
    )
    return points, valid
