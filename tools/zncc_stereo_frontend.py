"""Image-only dense ZNCC stereo control; no LiDAR-guided search or learned weights."""

from __future__ import annotations

import numpy as np
from tools.stereo_rerectification import rectification, restore_left_depth

SETTINGS = {
    "patch_size": 9,
    "max_disparity": 128,
    "min_correlation": 0.9,
    "min_texture": 6.0,
    "uniqueness_margin": 0.05,
}


def disparity_zncc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Full 0..127 px search, nonadjacent-peak uniqueness and parabolic subpixel fit."""
    import cv2

    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("matching grayscale images required")
    left_values, r = left.astype(float), right.astype(float)
    h, w = left_values.shape
    size = 9

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            x, cv2.CV_64F, (size, size), normalize=True, borderType=cv2.BORDER_CONSTANT
        )

    lm, rm = box(left_values), box(r)
    lv, rv = (
        np.maximum(box(left_values * left_values) - lm * lm, 0),
        np.maximum(box(r * r) - rm * rm, 0),
    )
    scores = np.full((128, h, w), -2, dtype=np.float32)
    for d in range(min(128, w - 8)):
        shifted = np.zeros_like(r)
        shifted[:, d:] = r[:, : w - d]
        mean = np.zeros_like(rm)
        var = np.zeros_like(rv)
        mean[:, d:] = rm[:, : w - d]
        var[:, d:] = rv[:, : w - d]
        score = (box(left_values * shifted) - lm * mean) / np.sqrt(np.maximum(lv * var, 1e-12))
        good = (lv >= 36) & (var >= 36)
        good[:4, :] = False
        good[-4:, :] = False
        good[:, : d + 4] = False
        good[:, -4:] = False
        scores[d] = np.where(good, score, -2)
    best = np.argmax(scores, axis=0)
    v, u = np.indices(left_values.shape)
    peak = scores[best, v, u]
    second = np.full(left_values.shape, -2, dtype=float)
    for d in range(128):
        second = np.maximum(second, np.where(abs(best - d) > 1, scores[d], -2))
    before = scores[np.maximum(best - 1, 0), v, u]
    after = scores[np.minimum(best + 1, 127), v, u]
    curvature = before - 2 * peak + after
    delta = np.divide(
        0.5 * (before - after), curvature, out=np.zeros_like(peak), where=abs(curvature) > 1e-9
    )
    valid = (
        (best > 0)
        & (best < 127)
        & (peak >= 0.9)
        & (peak - second >= 0.05)
        & (before > -2)
        & (after > -2)
    )
    result = best.astype(float) + np.clip(delta, -0.5, 0.5)
    result[~valid] = np.nan
    return result


def compute_depth(
    left: np.ndarray, right: np.ndarray, pl: np.ndarray, pr: np.ndarray
) -> np.ndarray:
    import cv2

    h, w = left.shape
    r1, r2, p1, p2 = rectification(pl, pr, (w, h))
    images = []
    masks = []
    for image, p, r, new in [(left, pl, r1, p1), (right, pr, r2, p2)]:
        x, y = cv2.initUndistortRectifyMap(
            p[:, :3], np.zeros(5), r, new[:, :3], (w, h), cv2.CV_32FC1
        )
        images.append(cv2.remap(image, x, y, cv2.INTER_LINEAR))
        support = ((x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)).astype(np.uint8)
        masks.append(
            cv2.erode(
                support, np.ones((9, 9), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
            )
            > 0
        )
    disparity = disparity_zncc(*images)
    v, u = np.indices(disparity.shape)
    valid = np.isfinite(disparity) & (disparity > 0) & masks[0]
    ur = np.rint(u - np.where(valid, disparity, 0)).astype(int)
    valid &= (ur >= 0) & (ur < w) & masks[1][v, np.clip(ur, 0, w - 1)]
    depth = np.full(disparity.shape, np.nan)
    depth[valid] = (p1[0, 3] - p2[0, 3]) / disparity[valid]
    depth[(depth < 0.1) | (depth > 80)] = np.nan
    return restore_left_depth(depth, r1, pl[:, :3], p1[:, :3])
