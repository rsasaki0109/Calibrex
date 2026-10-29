"""Continuous support weighting for experimental inverse-depth pose alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

from calibrex.solvers.stereo_translation_refiner import StereoSurfaceObservation


@dataclass(frozen=True)
class StereoSupportField:
    disparity: np.ndarray
    confidence: np.ndarray


def build_support_field(
    observation: StereoSurfaceObservation, max_range_px: float, taper_pixels: float = 2.0
) -> StereoSupportField:
    """Taper near missing data, 3x3 depth jumps and the image boundary.

    Nearest-filled disparity is solely an interpolation extension, never support.
    Confidence is zero on unsafe pixels and reaches one two pixels inward.
    """
    if not np.isfinite(taper_pixels) or taper_pixels <= 0:
        raise ValueError("positive taper width required")
    depth = observation.depth_m
    valid = np.isfinite(depth) & (depth > 0)
    disparity = observation.intrinsics[0] * observation.baseline_m / np.where(valid, depth, 1.0)
    spread = maximum_filter(np.where(valid, disparity, -np.inf), size=3) - minimum_filter(
        np.where(valid, disparity, np.inf), size=3
    )
    safe = valid & (spread <= max_range_px)
    safe[[0, -1], :] = False
    safe[:, [0, -1]] = False
    if not safe.any():
        return StereoSupportField(
            np.zeros_like(depth, dtype=float), np.zeros_like(depth, dtype=float)
        )
    indices = distance_transform_edt(~safe, return_distances=False, return_indices=True)
    extended = disparity[indices[0], indices[1]]
    distance = distance_transform_edt(safe)
    s = np.clip((distance - 1.0) / taper_pixels, 0.0, 1.0)
    return StereoSupportField(extended, s * s * (3 - 2 * s))


def support_residuals(
    camera: np.ndarray,
    observation: StereoSurfaceObservation,
    field: StereoSupportField,
    sigma_px: float,
    missing_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous measurement and missing-support residual channels.

    The sin/cos blend avoids sqrt(weight) derivative singularities. Bilinear
    sampling is continuous but not globally differentiable at pixel boundaries.
    """
    h, w = field.disparity.shape
    fx, fy, cx, cy = observation.intrinsics
    z = np.maximum(camera[:, 2], 0.1)
    u = np.clip(fx * camera[:, 0] / z + cx, 0, w - 1)
    v = np.clip(fy * camera[:, 1] / z + cy, 0, h - 1)
    x = np.minimum(np.floor(u).astype(int), w - 2)
    y = np.minimum(np.floor(v).astype(int), h - 2)
    a, b = u - x, v - y

    def sample(array: np.ndarray) -> np.ndarray:
        return (
            (1 - a) * (1 - b) * array[y, x]
            + a * (1 - b) * array[y, x + 1]
            + (1 - a) * b * array[y + 1, x]
            + a * b * array[y + 1, x + 1]
        )

    confidence = sample(field.confidence) * np.clip((camera[:, 2] - 0.1) / 0.1, 0, 1)
    residual = (sample(field.disparity) - fx * observation.baseline_m / z) / sigma_px
    angle = confidence * np.pi / 2
    return np.sin(angle) * residual, np.cos(angle) * missing_penalty
