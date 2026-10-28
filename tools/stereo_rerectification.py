"""Experimental re-rectification of pinhole stereo with non-horizontal translation."""

from __future__ import annotations

from typing import Any

import numpy as np


def rectification(
    left_projection: np.ndarray, right_projection: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    kl, kr = left_projection[:, :3], right_projection[:, :3]
    delta = np.linalg.solve(kr, right_projection[:, 3]) - np.linalg.solve(kl, left_projection[:, 3])
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        kl,
        np.zeros(5),
        kr,
        np.zeros(5),
        size,
        np.eye(3),
        delta,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=size,
    )
    return r1, r2, p1, p2


def restore_left_depth(
    rectified_depth: np.ndarray,
    rotation: np.ndarray,
    original_k: np.ndarray,
    rectified_k: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor depth on original rays; preserve invalid pixels and optical z."""
    import cv2

    v, u = np.indices(rectified_depth.shape)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    rays = pixels @ np.linalg.inv(original_k).T @ rotation.T
    projection = rays @ rectified_k.T
    map_u = (projection[..., 0] / projection[..., 2]).astype(np.float32)
    map_v = (projection[..., 1] / projection[..., 2]).astype(np.float32)
    sampled = cv2.remap(
        rectified_depth,
        map_u,
        map_v,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    depth = sampled / rays[..., 2]
    depth[(rays[..., 2] <= 0) | (depth < 0.1) | (depth > 80)] = np.nan
    return depth


def compute_depth(
    left: np.ndarray,
    right: np.ndarray,
    pl: np.ndarray,
    pr: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("matching grayscale images required")
    h, w = left.shape
    r1, r2, p1, p2 = rectification(pl, pr, (w, h))
    images = []
    masks = []
    for im, p, r, new in [(left, pl, r1, p1), (right, pr, r2, p2)]:
        x, y = cv2.initUndistortRectifyMap(
            p[:, :3], np.zeros(5), r, new[:, :3], (w, h), cv2.CV_32FC1
        )
        images.append(cv2.remap(im, x, y, cv2.INTER_LINEAR))
        masks.append((x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1))
    disparity = cv2.StereoSGBM_create(**settings).compute(*images).astype(float) / 16
    v, u = np.indices(disparity.shape)
    ur = np.rint(u - disparity).astype(int)
    valid = (disparity > 0) & masks[0] & (ur >= 0) & (ur < w)
    valid &= masks[1][v, np.clip(ur, 0, w - 1)]
    depth = np.full(disparity.shape, np.nan)
    depth[valid] = (p1[0, 3] - p2[0, 3]) / disparity[valid]
    depth[(depth < 0.1) | (depth > 80)] = np.nan
    restored = restore_left_depth(depth, r1, pl[:, :3], p1[:, :3])
    return restored, {
        "r1": r1.tolist(),
        "r2": r2.tolist(),
        "p1": p1.tolist(),
        "p2": p2.tolist(),
        "valid_pixels": int(np.isfinite(restored).sum()),
        "opencv_version": cv2.__version__,
        "interpolation": "linear image; nearest depth",
    }
