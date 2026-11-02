"""Overlay images for the ``camera-lidar`` check: one LiDAR scan drawn on its image.

The check judges the rotation of ``T_camera_lidar`` with an edge-alignment objective,
which can reach a false optimum that looks confident (``docs/benchmarks/
camera_lidar_v4_candidates.md``). A picture lets a person tell the two apart: under a
right calibration the points sit on the objects they hit. :func:`overlay_rgb` draws one
synchronized scan on its image under a given ``T_camera_lidar``, coloured by depth (red
near, blue far) on a dimmed gray image, keeping the nearest point where several land on
one pixel. :func:`encode_overlay` stores it as JPEG with Pillow, or as PNG with the
standard library when Pillow is missing.

One scan of a non-repetitive LiDAR (a Livox rosette) covers its field of view sparsely.
:func:`neighbour_scans` and :func:`register_neighbours` add the scans recorded within
``NEIGHBOUR_HALF_WINDOW_S`` of it, each registered onto it with the check's point-to-plane
scan registration, so the drawing does not depend on a motion model. A neighbour whose
registration did not converge, left a residual above ``MAX_NEIGHBOUR_RMSE_M`` or matched
too few points is left out.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from calibrex.evaluation.camera_lidar_edge import EdgeCamera

MAX_SIDE_PX = 960
"""The longer side of an overlay image after downscaling."""
MAX_POINTS = 80_000
NEIGHBOUR_HALF_WINDOW_S = 0.25
"""Scans whose header stamps lie this close to the drawn scan's are added to it."""
MAX_NEIGHBOUR_SCANS = 4
MAX_NEIGHBOUR_RMSE_M = 0.1
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


def neighbour_scans(
    bag: str | Path,
    lidar_topic: str,
    reference_time_s: float,
    *,
    half_window_s: float = NEIGHBOUR_HALF_WINDOW_S,
    max_scans: int = MAX_NEIGHBOUR_SCANS,
) -> list[NDArray[np.float64]]:
    """The scans (LiDAR frame) recorded within ``half_window_s`` of the drawn scan.

    The drawn scan is the one whose header stamp is the latest at or before
    ``reference_time_s`` (the frame's reference time can lie inside its sweep); it is
    not returned. The nearest ``max_scans`` others are.
    """

    from calibrex.data.livox_ros2 import decode_lidar_scan
    from calibrex.data.rosbag2 import iter_messages, require_cdr_connection

    found: list[tuple[float, NDArray[np.float64]]] = []
    for connection, timestamp_ns, payload in iter_messages(bag, topics={lidar_topic}):
        require_cdr_connection(connection)
        scan = decode_lidar_scan(connection.message_type, lidar_topic, timestamp_ns, payload)
        header_s = scan.timestamp_ns * 1.0e-9
        if header_s > reference_time_s + 2.0 * half_window_s + 1.0:
            break
        if abs(header_s - reference_time_s) <= 2.0 * half_window_s + 1.0:
            found.append((header_s, scan.xyz))
    earlier = [item for item in found if item[0] <= reference_time_s + 1.0e-6]
    if not earlier:
        return []
    drawn = max(earlier, key=lambda item: item[0])[0]
    others = [
        item
        for item in found
        if item[0] != drawn and abs(item[0] - drawn) <= half_window_s + 1.0e-9
    ]
    others.sort(key=lambda item: abs(item[0] - drawn))
    return [xyz for _time, xyz in others[:max_scans]]


def register_neighbours(
    reference: NDArray[np.floating],
    neighbours: Sequence[NDArray[np.floating]],
    *,
    max_rmse_m: float = MAX_NEIGHBOUR_RMSE_M,
) -> tuple[NDArray[np.float64], int]:
    """``(reference plus registered neighbours, number of neighbours used)``.

    Each neighbour is registered onto the reference with point-to-plane ICP from the
    identity (consecutive scans of one sweep rate barely move) and kept only when the
    registration converged with an RMS residual of at most ``max_rmse_m`` and enough
    correspondences.
    """

    from scipy.spatial import cKDTree

    from calibrex.solvers.scan_to_scan_odometry import (
        ScanOdometryOptions,
        TargetNormals,
        preprocess_scan,
        register_point_to_plane,
    )

    base = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if not neighbours:
        return base, 0
    options = ScanOdometryOptions(min_range_m=1.0)
    target = preprocess_scan(base, options)
    if len(target) < options.min_correspondences:
        return base, 0
    tree = cKDTree(target)
    normals = TargetNormals(target, tree, options)
    parts = [base]
    for neighbour in neighbours:
        points = np.asarray(neighbour, dtype=np.float64).reshape(-1, 3)
        source = preprocess_scan(points, options)
        if len(source) < options.min_correspondences:
            continue
        registration = register_point_to_plane(
            source, target, np.eye(4), options, target_tree=tree, target_normals=normals
        )
        if (
            not registration.converged
            or registration.rmse_m > max_rmse_m
            or registration.correspondences < options.min_correspondences
        ):
            continue
        transform = registration.transform
        parts.append(points @ transform[:3, :3].T + transform[:3, 3])
    return np.concatenate(parts), len(parts) - 1


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
