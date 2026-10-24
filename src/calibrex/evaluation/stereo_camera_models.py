"""Unified/MEI and double-sphere cameras for model-neutral stereo rectification.

Projection domains follow the unified camera and double-sphere models. Invalid
rays/pixels are masked or returned as NaN, never folded into a plausible ray.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _validate(fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> None:
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("camera intrinsics must be finite with positive focal lengths")
    if width <= 1 or height <= 1:
        raise ValueError("camera dimensions must exceed one pixel")


def _radtan(x: FloatArray, y: FloatArray, d: tuple[float, ...]) -> tuple[FloatArray, FloatArray]:
    k1, k2, p1, p2 = d
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    return (
        x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
        y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
    )


@dataclass(frozen=True)
class MeiCamera:
    """Unified central camera with xi and optional four-coefficient radtan distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    distortion: tuple[float, float, float, float]
    width: int
    height: int

    def __post_init__(self) -> None:
        _validate(self.fx, self.fy, self.cx, self.cy, self.width, self.height)
        if not math.isfinite(self.xi) or self.xi < 0:
            raise ValueError("MEI xi must be finite and nonnegative")
        if len(self.distortion) != 4 or not all(math.isfinite(v) for v in self.distortion):
            raise ValueError("MEI needs finite distortion [k1, k2, p1, p2]")

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        """Project camera-frame points on the injective unified-camera domain."""

        xyz = np.asarray(points, dtype=np.float64)
        norm = np.linalg.norm(xyz, axis=1)
        z = xyz[:, 2]
        denominator = z + self.xi * norm
        limit = self.xi if self.xi <= 1 else 1 / self.xi
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(norm) & (norm > 1e-12)
        valid &= (z > -limit * norm) & (denominator > 1e-12)
        safe = np.where(valid, denominator, 1.0)
        x, y = _radtan(xyz[:, 0] / safe, xyz[:, 1] / safe, self.distortion)
        u, v = self.fx * x + self.cx, self.fy * y + self.cy
        return u, v, valid & np.isfinite(u) & np.isfinite(v)

    def unproject(self, u: FloatArray, v: FloatArray) -> FloatArray:
        """Return unit rays, with NaN for pixels outside the inverse model's domain."""

        xd = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        yd = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        k1, k2, p1, p2 = self.distortion
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            for _ in range(20):
                r2 = x * x + y * y
                radial = 1 + k1 * r2 + k2 * r2 * r2
                dx, dy = 2 * x * (k1 + 2 * k2 * r2), 2 * y * (k1 + 2 * k2 * r2)
                a, b = radial + x * dx + 2 * p1 * y + 6 * p2 * x, x * dy + 2 * p1 * x + 2 * p2 * y
                c, d = y * dx + 2 * p1 * x + 2 * p2 * y, radial + y * dy + 6 * p1 * y + 2 * p2 * x
                determinant = a * d - b * c
                px, py = _radtan(x, y, self.distortion)
                ex, ey = px - xd, py - yd
                safe = np.where(np.abs(determinant) > 1e-12, determinant, np.nan)
                x, y = x - (d * ex - b * ey) / safe, y - (-c * ex + a * ey) / safe
            r2 = x * x + y * y
            radicand = 1 + (1 - self.xi * self.xi) * r2
            scale = (self.xi + np.sqrt(np.maximum(radicand, 0))) / (1 + r2)
            rays = np.stack([scale * x, scale * y, scale - self.xi], axis=-1)
            px, py = _radtan(x, y, self.distortion)
            valid = (radicand >= 0) & (np.hypot(px - xd, py - yd) < 1e-8)
            valid &= np.isfinite(rays).all(axis=-1)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return np.where(valid[..., None], rays, np.nan)


@dataclass(frozen=True)
class DoubleSphereCamera:
    """Double-sphere camera (xi, alpha), without an additional distortion layer."""

    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    alpha: float
    width: int
    height: int

    def __post_init__(self) -> None:
        _validate(self.fx, self.fy, self.cx, self.cy, self.width, self.height)
        if not math.isfinite(self.xi) or not 0 < self.alpha < 1:
            raise ValueError("double-sphere needs finite xi and alpha in (0, 1)")
        w1 = self.alpha / (1 - self.alpha) if self.alpha <= 0.5 else (1 - self.alpha) / self.alpha
        if 2 * w1 * self.xi + self.xi * self.xi + 1 <= 1e-12:
            raise ValueError("degenerate double-sphere domain")

    def project(self, points: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        """Project points in the double-sphere visibility domain, including rear rays."""

        xyz = np.asarray(points, dtype=np.float64)
        x, y, z = xyz.T
        norm = np.linalg.norm(xyz, axis=1)
        z1 = self.xi * norm + z
        denominator = self.alpha * np.sqrt(x * x + y * y + z1 * z1) + (1 - self.alpha) * z1
        w1 = self.alpha / (1 - self.alpha) if self.alpha <= 0.5 else (1 - self.alpha) / self.alpha
        w2 = (w1 + self.xi) / math.sqrt(2 * w1 * self.xi + self.xi * self.xi + 1)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(norm) & (norm > 1e-12)
        valid &= (denominator > 1e-12) & (z > -w2 * norm)
        safe = np.where(valid, denominator, 1.0)
        u, v = self.fx * x / safe + self.cx, self.fy * y / safe + self.cy
        return u, v, valid & np.isfinite(u) & np.isfinite(v)

    def unproject(self, u: FloatArray, v: FloatArray) -> FloatArray:
        """Return unit rays, with NaN for pixels outside the inverse domain."""

        x = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        y = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        r2 = x * x + y * y
        domain = 1 - (2 * self.alpha - 1) * r2
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (1 - self.alpha * self.alpha * r2) / (
                self.alpha * np.sqrt(np.maximum(domain, 0)) + 1 - self.alpha
            )
            radical = z * z + (1 - self.xi * self.xi) * r2
            scale = (z * self.xi + np.sqrt(np.maximum(radical, 0))) / (z * z + r2)
            rays = np.stack([scale * x, scale * y, scale * z - self.xi], axis=-1)
            valid = (domain >= 0) & (radical >= 0) & np.isfinite(rays).all(axis=-1)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return np.where(valid[..., None], rays, np.nan)
