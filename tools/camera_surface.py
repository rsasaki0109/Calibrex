"""Fixed-scale affine disparity surfaces with conservative angular confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d, uniform_filter, uniform_filter1d
from scipy.stats import chi2


@dataclass(frozen=True)
class StereoGeometry:
    fx: float
    fy: float
    cx: float
    cy: float
    fb: float
    sigma_px: float = 0.5

    def __post_init__(self):
        values = np.array([self.fx, self.fy, self.cx, self.cy, self.fb, self.sigma_px])
        if not np.isfinite(values).all() or min(self.fx, self.fy, self.fb, self.sigma_px) <= 0:
            raise ValueError("finite positive focal lengths, stereo scale and noise required")


def estimate(depth: np.ndarray, window: int, geometry: StereoGeometry) -> dict[str, np.ndarray]:
    """Return all scales on the same ten-pixel inset domain; no truth inputs."""
    if window not in (5, 11, 21):
        raise ValueError("window must be5,11 or21")
    valid = np.isfinite(depth) & (depth > 0)
    d = np.divide(geometry.fb, depth, out=np.zeros_like(depth), where=valid)
    h = window // 2
    offset = np.arange(-h, h + 1, dtype=float)
    c = uniform_filter(d, size=window)
    ex2 = float(np.mean(offset**2))
    a = correlate1d(uniform_filter1d(d, size=window, axis=0), offset, axis=1) / (window * ex2)
    b = correlate1d(uniform_filter1d(d, size=window, axis=1), offset, axis=0) / (window * ex2)
    rms = np.sqrt(np.maximum(0, uniform_filter(d * d, size=window) - c * c - ex2 * (a * a + b * b)))
    full = uniform_filter(valid.astype(float), size=window) > 1 - 1e-10
    crop = (slice(10, -10), slice(10, -10))
    a, b, c, rms, full = [x[crop] for x in [a, b, c, rms, full]]
    v, u = np.indices(c.shape)
    v = v + 10
    u = u + 10
    mapping = np.zeros((*c.shape, 3, 3))
    mapping[..., 0, 0] = geometry.fx
    mapping[..., 1, 1] = geometry.fy
    mapping[..., 2, 0] = -(u - geometry.cx)
    mapping[..., 2, 1] = -(v - geometry.cy)
    mapping[..., 2, 2] = 1
    coefficients = np.stack([a, b, c], axis=-1)
    normal = np.einsum("...ij,...j->...i", mapping, coefficients)
    length = np.maximum(np.linalg.norm(normal, axis=-1), 1e-15)
    normal = normal / length[..., None]
    variance = np.array(
        [
            geometry.sigma_px**2 / (window**2 * ex2),
            geometry.sigma_px**2 / (window**2 * ex2),
            geometry.sigma_px**2 / window**2,
        ]
    )
    covariance = (mapping * variance) @ np.swapaxes(mapping, -1, -2)
    radius = np.degrees(
        np.arcsin(
            np.clip(
                np.sqrt(chi2.ppf(0.95, 3) * np.linalg.eigvalsh(covariance)[..., -1]) / length, 0, 1
            )
        )
    )
    sigma = (geometry.sigma_px / window) * (geometry.fb / np.maximum(c, 1e-12)) ** 2 / geometry.fb
    accepted = full & (c > 0) & (rms <= 1) & (radius <= 15) & (sigma <= 0.2)
    return {
        "accepted": accepted,
        "normal": normal,
        "radius": radius,
        "rms": rms,
        "depth": geometry.fb / np.maximum(c, 1e-12),
    }
