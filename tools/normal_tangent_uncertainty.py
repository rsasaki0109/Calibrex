"""First-order tangent-plane angular uncertainty; approximate coverage."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import chi2
from tools.known_geometry_joint import CAMERA


def cone_radius(coefficients: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Approximate 95% tangent ellipse enclosed by an angular circle."""
    shape = coefficients.shape[:-1]
    mapping = np.zeros((*shape, 3, 3))
    mapping[..., 0, 0] = CAMERA.fx
    mapping[..., 1, 1] = CAMERA.fy
    mapping[..., 2, 0] = -(u - CAMERA.cx)
    mapping[..., 2, 1] = -(v - CAMERA.cy)
    mapping[..., 2, 2] = 1
    variance = np.array([0.25 / 50, 0.25 / 50, 0.25 / 25])
    covariance = (mapping * variance) @ np.swapaxes(mapping, -1, -2)
    normal = np.einsum("...ij,...j->...i", mapping, coefficients)
    length = np.maximum(np.linalg.norm(normal, axis=-1), 1e-15)
    direction = normal / length[..., None]
    tangent = np.eye(3) - direction[..., :, None] * direction[..., None, :]
    angular = tangent @ covariance @ tangent / length[..., None, None] ** 2
    radius = np.degrees(
        np.sqrt(chi2.ppf(0.95, 2) * np.maximum(0.0, np.linalg.eigvalsh(angular)[..., -1]))
    )
    return np.minimum(90.0, radius)


def normal_cone(depth: np.ndarray) -> np.ndarray:
    d = np.divide(
        120.0, depth, out=np.full_like(depth, np.nan), where=np.isfinite(depth) & (depth > 0)
    )
    patches = sliding_window_view(d, (5, 5))
    dy, dx = np.mgrid[-2:3, -2:3]
    coefficients = np.stack(
        [
            np.sum(patches * dx, axis=(-2, -1)) / 50,
            np.sum(patches * dy, axis=(-2, -1)) / 50,
            np.mean(patches, axis=(-2, -1)),
        ],
        axis=-1,
    )
    v, u = np.indices(coefficients.shape[:-1])
    valid = np.isfinite(coefficients).all(axis=-1)
    radius = cone_radius(np.where(valid[..., None], coefficients, 0), u + 2, v + 2)
    return np.where(valid, radius, 90.0)
