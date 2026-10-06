"""A synthetic camera-LiDAR scene shared by the camera-lidar tests.

Fronto-parallel rectangles at several depths in front of a camera, seen by a pinhole
camera (rendered image) and by a spinning multi-beam LiDAR (ray-cast ranges).  Both
see the same rectangle borders, so the edge-alignment objective peaks at the true
extrinsic.  The camera frame is the optical frame (``z`` forward); the LiDAR frame
has ``x`` forward, ``y`` left, ``z`` up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

FloatArray = NDArray[np.float64]

WIDTH, HEIGHT = 320, 240
FOCAL = 240.0
CX, CY = 160.0, 120.0
BEAMS = 32
# camera <- lidar rotation of a LiDAR that looks where the camera looks
R_CAMERA_LIDAR_NOMINAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class Rect:
    """A rectangle in the plane ``z = depth`` of the camera frame."""

    depth: float
    cx: float
    cy: float
    width: float
    height: float
    gray: int


def make_scene(seed: int, count: int = 14) -> list[Rect]:
    """A seeded layout of rectangles, plus a far background wall."""

    rng = np.random.default_rng(seed)
    rects = [Rect(30.0, 0.0, 0.0, 200.0, 200.0, 90)]
    for _ in range(count):
        rects.append(
            Rect(
                depth=float(rng.uniform(5.0, 16.0)),
                cx=float(rng.uniform(-7.0, 7.0)),
                cy=float(rng.uniform(-2.0, 2.0)),
                width=float(rng.uniform(1.0, 3.0)),
                height=float(rng.uniform(1.0, 3.0)),
                gray=int(rng.integers(30, 230)),
            )
        )
    return rects


def render_image(rects: list[Rect]) -> NDArray[np.uint8]:
    """Painter's-algorithm render of the rectangles with a pinhole camera."""

    u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    xn = (u - CX) / FOCAL
    yn = (v - CY) / FOCAL
    image = np.full((HEIGHT, WIDTH), 90.0)
    for rect in sorted(rects, key=lambda item: -item.depth):
        mask = (np.abs(xn * rect.depth - rect.cx) < rect.width / 2) & (
            np.abs(yn * rect.depth - rect.cy) < rect.height / 2
        )
        image[mask] = rect.gray
    return np.asarray(image, dtype=np.uint8)


def lidar_directions() -> FloatArray:
    """Unit rays of a 32-beam spinning LiDAR over +-45 deg of azimuth (``x`` forward)."""

    azimuth = np.radians(np.arange(-45.0, 45.0, 0.15))
    elevation = np.radians(np.linspace(-12.0, 12.0, BEAMS))
    az, el = np.meshgrid(azimuth, elevation)
    return np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
    ).reshape(-1, 3)


def cast_points(rects: list[Rect], t_camera_lidar: FloatArray) -> NDArray[np.float32]:
    """LiDAR points (in the LiDAR frame) of the nearest rectangle along every ray."""

    rays_lidar = lidar_directions()
    rotation = t_camera_lidar[:3, :3]
    origin = t_camera_lidar[:3, 3]
    rays_camera = rays_lidar @ rotation.T
    best = np.full(rays_lidar.shape[0], np.inf)
    for rect in rects:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (rect.depth - origin[2]) / rays_camera[:, 2]
        hit = origin[None, :] + s[:, None] * rays_camera
        inside = (
            (s > 0.5)
            & (np.abs(hit[:, 0] - rect.cx) < rect.width / 2)
            & (np.abs(hit[:, 1] - rect.cy) < rect.height / 2)
        )
        best = np.where(inside & (s < best), s, best)
    keep = np.isfinite(best)
    return np.asarray(rays_lidar[keep] * best[keep, None], dtype=np.float32)


def true_transform(
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation_m: tuple[float, float, float] = (0.05, -0.08, -0.10),
) -> FloatArray:
    """``T_camera_lidar`` of the scene: the nominal rotation turned by ``rotation_deg``."""

    matrix = np.eye(4)
    matrix[:3, :3] = (
        Rotation.from_rotvec(np.radians(rotation_deg)).as_matrix() @ R_CAMERA_LIDAR_NOMINAL
    )
    matrix[:3, 3] = translation_m
    return matrix


def perturbed(truth: FloatArray, rotation_deg: tuple[float, float, float]) -> FloatArray:
    """``truth`` with a small rotation about the camera axes applied on the left."""

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(np.radians(rotation_deg)).as_matrix()
    return np.asarray(matrix @ truth, dtype=np.float64)
