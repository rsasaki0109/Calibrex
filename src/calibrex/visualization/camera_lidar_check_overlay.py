"""Overlay images for the ``camera-lidar`` check: one LiDAR scan drawn on its image.

The check judges the rotation of ``T_camera_lidar`` with an edge-alignment objective,
which can reach a false optimum that looks confident (``docs/benchmarks/
camera_lidar_v4_candidates.md``). A picture lets a person tell the two apart: under a
right calibration the points sit on the objects they hit. :func:`overlay_rgb` draws one
synchronized scan on its image under a given ``T_camera_lidar``, coloured by depth (red
near, blue far) on a dimmed gray image, keeping the nearest point where several land on
one pixel. :func:`encode_overlay` stores it as JPEG with Pillow, or as PNG with the
standard library when Pillow is missing.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from calibrex.evaluation.camera_lidar_edge import EdgeCamera

MAX_SIDE_PX = 960
"""The longer side of an overlay image after downscaling."""
MAX_POINTS = 80_000
NEAR_M, FAR_M = 1.0, 80.0
_RAMP = np.array(
    [
        [255, 40, 40],  # near: red
        [255, 210, 0],
        [60, 220, 60],
        [0, 200, 255],
        [70, 90, 255],  # far: blue
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class OverlayImage:
    """One encoded overlay image, written to the evidence directory by the check."""

    role: str
    media_type: str
    data: bytes

    @property
    def suffix(self) -> str:
        return ".jpg" if self.media_type == "image/jpeg" else ".png"


def depth_colors(depth_m: NDArray[np.float64]) -> NDArray[np.uint8]:
    """RGB colours of depths on a log scale from ``NEAR_M`` (red) to ``FAR_M`` (blue)."""

    ratio = np.log(np.clip(depth_m, NEAR_M, FAR_M) / NEAR_M) / math.log(FAR_M / NEAR_M)
    position = ratio * (len(_RAMP) - 1)
    lower = np.minimum(np.floor(position).astype(np.intp), len(_RAMP) - 2)
    weight = (position - lower)[:, None]
    colors = _RAMP[lower] * (1.0 - weight) + _RAMP[lower + 1] * weight
    return np.asarray(np.rint(colors), dtype=np.uint8)


def _downscale(gray: NDArray[np.uint8], factor: int) -> NDArray[np.float64]:
    if factor == 1:
        return gray.astype(np.float64)
    height, width = (gray.shape[0] // factor) * factor, (gray.shape[1] // factor) * factor
    blocks = gray[:height, :width].astype(np.float64)
    return blocks.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def overlay_rgb(
    gray: NDArray[np.uint8],
    points_lidar: NDArray[np.floating],
    camera: EdgeCamera,
    t_camera_lidar: NDArray[np.float64],
    *,
    max_side: int = MAX_SIDE_PX,
    max_points: int = MAX_POINTS,
) -> NDArray[np.uint8]:
    """``(h, w, 3)`` image: the scan projected under ``t_camera_lidar`` on the gray image."""

    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("overlay needs a non-empty 2-D gray image")
    factor = max(1, math.ceil(max(gray.shape) / max_side))
    base = _downscale(gray, factor) * 0.7
    height, width = base.shape
    canvas = np.repeat(np.asarray(np.clip(base, 0, 255), dtype=np.uint8)[:, :, None], 3, axis=2)
    points = np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).astype(np.intp)]
    moved = points @ t_camera_lidar[:3, :3].T + t_camera_lidar[:3, 3]
    u, v, valid = camera.project(moved)
    columns = np.floor(np.asarray(u) / factor).astype(np.int64)
    rows = np.floor(np.asarray(v) / factor).astype(np.int64)
    keep = (
        np.asarray(valid)
        & (moved[:, 2] > 0.1)
        & (columns >= 0)
        & (columns < width - 1)
        & (rows >= 0)
        & (rows < height - 1)
    )
    if not np.any(keep):
        return canvas
    depth = moved[keep, 2]
    columns, rows = columns[keep], rows[keep]
    colors = depth_colors(depth)
    # 2 x 2 dots; where dots overlap the nearest point wins, deterministically.
    dot_rows = np.concatenate([rows, rows, rows + 1, rows + 1])
    dot_columns = np.concatenate([columns, columns + 1, columns, columns + 1])
    dot_depth = np.tile(depth, 4)
    dot_colors = np.tile(colors, (4, 1))
    order = np.argsort(dot_depth, kind="stable")
    linear = dot_rows[order] * width + dot_columns[order]
    _unique, first = np.unique(linear, return_index=True)
    chosen = order[first]
    canvas[dot_rows[chosen], dot_columns[chosen]] = dot_colors[chosen]
    return canvas


def encode_png(rgb: NDArray[np.uint8]) -> bytes:
    """A standard-library PNG (8-bit RGB, no filter) of an ``(h, w, 3)`` image."""

    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    rows = np.zeros((height, 1 + width * 3), dtype=np.uint8)
    rows[:, 1:] = image.reshape(height, width * 3)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def encode_overlay(role: str, rgb: NDArray[np.uint8]) -> OverlayImage:
    """JPEG (quality 85) with Pillow, else a PNG from the standard library."""

    try:
        from PIL import Image
    except ImportError:
        return OverlayImage(role, "image/png", encode_png(rgb))
    buffer = BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=85
    )
    return OverlayImage(role, "image/jpeg", buffer.getvalue())
