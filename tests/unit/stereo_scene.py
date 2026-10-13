"""A rendered fisheye stereo pair inside a textured box-shaped room (tests only)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from calibrex.evaluation.fisheye_stereo import EquidistantCamera

LEFT = EquidistantCamera(160.0, 160.5, 161.0, 118.0, (-0.03, -0.006, 0.004, -0.002), 320, 240)
RIGHT = EquidistantCamera(161.0, 161.2, 158.0, 121.5, (-0.035, -0.004, 0.003, -0.001), 320, 240)
# the right camera 0.1 m along the left camera's x, slightly turned
TRUTH = np.eye(4)
TRUTH[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.4, 0.2], degrees=True).as_matrix()
TRUTH[:3, 3] = [0.1, 0.001, -0.002]
ROOM_M = np.array([6.0, 4.0, 2.5])  # half-sizes of the room (x, y, z)


@lru_cache(maxsize=1)
def _texture() -> np.ndarray:
    rng = np.random.default_rng(11)
    blurred = ndimage.gaussian_filter(rng.uniform(0.0, 1.0, (1024, 2048)), 2.0)
    return 128.0 + 40.0 * (blurred - blurred.mean()) / blurred.std()


def render(camera: EquidistantCamera, rotation: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """The camera's image of a textured box-shaped room around the world origin."""

    rows, cols = np.mgrid[0 : camera.height, 0 : camera.width].astype(np.float64)
    rays = camera.unproject(cols.ravel(), rows.ravel()) @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = np.where(rays > 0.0, (ROOM_M - origin) / rays, (-ROOM_M - origin) / rays)
    reach = np.where(np.isfinite(reach) & (reach > 0.0), reach, np.inf)
    wall = np.argmin(reach, axis=1)
    points = origin + rays * reach[np.arange(len(rays)), wall][:, None]
    # each wall shows the texture over its two in-plane coordinates, offset per wall
    first = np.where(wall == 0, points[:, 1], points[:, 0])
    second = np.where(wall == 2, points[:, 1], points[:, 2])
    texture = _texture()
    scale = 120.0  # texels per metre
    u = first * scale + 300.0 * wall + 0.5 * texture.shape[1]
    v = second * scale + 150.0 * wall + 0.5 * texture.shape[0]
    image = ndimage.map_coordinates(texture, [v, u], order=1, mode="wrap")
    return np.clip(image, 0.0, 255.0).reshape(camera.height, camera.width)


def render_pairs(
    truth: np.ndarray, count: int = 8, seed: int = 4
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``count`` pairs of the room, the rig turned at random, the right camera at ``truth``."""

    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(count):
        world_left = Rotation.from_euler(
            "zyx",
            [rng.uniform(-180, 180), rng.uniform(-30, 30), rng.uniform(-20, 20)],
            degrees=True,
        ).as_matrix()
        left = render(LEFT, world_left, np.zeros(3))
        right = render(RIGHT, world_left @ truth[:3, :3], world_left @ truth[:3, 3])
        pairs.append((left, right))
    return pairs
