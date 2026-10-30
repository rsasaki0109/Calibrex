"""Disjoint checkerboard predictive test for the fixed21-pixel surface fit."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import chi2
from tools.coherent_surface import estimate


def predictive_statistic(depth: np.ndarray) -> np.ndarray:
    """Exact Gaussian predictive chi-square, including shared fit uncertainty."""
    dy, dx = np.mgrid[-10:11, -10:11]
    design = np.stack([dx, dy, np.ones_like(dx)], axis=-1).astype(float)
    train = (dx + dy) % 2 == 0
    holdout = ~train
    xt = design[train]
    xh = design[holdout]
    mt = xt.T @ xt
    mh = xh.T @ xh
    inverse = np.linalg.inv(mt)
    full_inverse = np.linalg.inv(mt + mh)
    valid = np.isfinite(depth) & (depth > 0)
    d = np.divide(120.0, depth, out=np.zeros_like(depth), where=valid)

    def correlate(kernel):
        return fftconvolve(d, kernel[::-1, ::-1], mode="same")[10:-10, 10:-10]

    bt = np.stack([correlate(design[..., k] * train) for k in range(3)], axis=-1)
    bh = np.stack([correlate(design[..., k] * holdout) for k in range(3)], axis=-1)
    theta = bt @ inverse
    squares = fftconvolve(d * d, holdout[::-1, ::-1], mode="same")[10:-10, 10:-10]
    rss = (
        squares - 2 * np.sum(theta * bh, axis=-1) + np.einsum("...i,ij,...j->...", theta, mh, theta)
    )
    e = bh - theta @ mh
    statistic = (rss - np.einsum("...i,ij,...j->...", e, full_inverse, e)) / 0.25
    full = (
        fftconvolve(valid.astype(float), np.ones((21, 21)), mode="same")[10:-10, 10:-10]
        > 441 - 1e-7
    )
    return np.where(full, np.maximum(0.0, statistic), np.inf)


def gated_estimate(depth: np.ndarray) -> dict[str, np.ndarray]:
    result = estimate(depth, 21)
    statistic = predictive_statistic(depth)
    result["accepted"] &= statistic <= chi2.ppf(0.99, 220)
    return result


def plane_validation() -> dict[str, float | int]:
    """Independent exact-plane predictive coverage sanity check."""
    dy, dx = np.mgrid[-10:11, -10:11]
    design = np.stack([dx, dy, np.ones_like(dx)], axis=-1).reshape(-1, 3)
    train = ((dx + dy) % 2 == 0).ravel()
    xt, xh = design[train], design[~train]
    rng = np.random.default_rng(20261031)
    errors = rng.normal(0, 0.5, size=(10000, 441))
    coefficients = errors[:, train] @ xt @ np.linalg.inv(xt.T @ xt)
    residual = errors[:, ~train] - coefficients @ xh.T
    e = residual @ xh
    statistic = (
        np.sum(residual**2, axis=1)
        - np.einsum("ni,ij,nj->n", e, np.linalg.inv(design.T @ design), e)
    ) / 0.25
    return {
        "seed": 20261031,
        "samples": 10000,
        "degrees_of_freedom": 220,
        "acceptance_fraction": float(np.mean(statistic <= chi2.ppf(0.99, 220))),
    }
