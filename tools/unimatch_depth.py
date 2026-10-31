"""Convert external rectified disparity to depth on original left-camera rays."""

from __future__ import annotations

import numpy as np
from tools.stereo_rerectification import rectification, restore_left_depth


def depth_from_prediction(disparity: np.ndarray, pl: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """Apply common stereo support and 0.1-80 m range; no confidence tuning."""
    import cv2

    h, w = disparity.shape
    r1, r2, p1, p2 = rectification(pl, pr, (w, h))
    masks = []
    for p, r, new in [(pl, r1, p1), (pr, r2, p2)]:
        x, y = cv2.initUndistortRectifyMap(
            p[:, :3], np.zeros(5), r, new[:, :3], (w, h), cv2.CV_32FC1
        )
        masks.append((x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1))
    finite = np.isfinite(disparity)
    v, u = np.indices(disparity.shape)
    ur = np.rint(u - np.where(finite, disparity, 0)).astype(int)
    valid = finite & (disparity > 0) & masks[0] & (ur >= 0) & (ur < w)
    valid &= masks[1][v, np.clip(ur, 0, w - 1)]
    depth = np.full(disparity.shape, np.nan)
    depth[valid] = (p1[0, 3] - p2[0, 3]) / disparity[valid]
    depth[(depth < 0.1) | (depth > 80)] = np.nan
    return restore_left_depth(depth, r1, pl[:, :3], p1[:, :3])
