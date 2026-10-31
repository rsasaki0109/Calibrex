"""Frozen alternative analytic sphere arrangements for occlusion controls."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from tools.known_geometry_joint import CAMERA, SPHERES, reference

from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation

SCENES = {
    "original": SPHERES,
    "near_foreground": [
        ((-1.5, -0.7, 5.0), 1.1),
        ((2.0, 1.0, 9.0), 1.8),
        ((-3.0, 2.0, 13.0), 2.0),
        ((3.0, -2.0, 15.0), 2.5),
    ],
    "staggered": [
        ((-2.5, -1.5, 8.0), 1.4),
        ((0.5, 0.5, 6.0), 0.9),
        ((3.0, 2.0, 12.0), 2.0),
        ((-4.0, 2.0, 16.0), 2.5),
        ((3.0, -2.0, 10.0), 1.2),
    ],
}


def raycast(origin: np.ndarray, directions: np.ndarray, spheres: list) -> np.ndarray:
    """Return first intersection parameter for arbitrary unnormalized rays."""
    normal = np.array([0.12, -0.08, 1.0])
    denom = directions @ normal
    distance = np.divide(
        22.0 - origin @ normal, denom, out=np.full(denom.shape, np.inf), where=denom > 0
    )
    a = np.sum(directions * directions, axis=-1)
    for center, radius in spheres:
        offset = origin - np.array(center)
        b = directions @ offset
        c = offset @ offset - radius * radius
        discriminant = b * b - a * c
        root = np.sqrt(np.maximum(discriminant, 0))
        near, far = (-b - root) / a, (-b + root) / a
        hit = np.where(near > 0, near, far)
        distance = np.minimum(distance, np.where((discriminant >= 0) & (hit > 0), hit, np.inf))
    return distance


def observations(scene: str, condition: str) -> list[DepthToDepthObservation]:
    if condition not in ("exact_z_visible", "exact_z_all"):
        raise ValueError("unknown condition")
    spheres = SCENES[scene]
    ref = reference()
    rotation = Rotation.from_quat(ref.rotation_quat_xyzw).as_matrix()
    t = np.array(ref.translation_m)
    result = []
    v, u = np.indices((CAMERA.height, CAMERA.width))
    rays = np.stack([(u - CAMERA.cx) / CAMERA.fx, (v - CAMERA.cy) / CAMERA.fy, np.ones_like(u)], -1)
    lv, lu = np.meshgrid(np.linspace(6.3, 185.3, 48), np.linspace(8.7, 310.7, 80), indexing="ij")
    lidar_rays_camera = np.stack(
        [(lu - CAMERA.cx) / CAMERA.fx, (lv - CAMERA.cy) / CAMERA.fy, np.ones_like(lu)], -1
    ).reshape(-1, 3)
    for i, origin in enumerate(
        [np.zeros(3), np.array([1.0, 0.2, 0.4]), np.array([-0.8, -0.3, 0.7])]
    ):
        z = raycast(origin, rays, spheres)
        depth = z.copy()
        hit = raycast(origin + t, lidar_rays_camera, spheres)
        points_camera = t + hit[:, None] * lidar_rays_camera
        # Continuous camera-ray visibility, independent of solver pixel sampling.
        camera_hit = raycast(origin, points_camera, spheres)
        visible = np.isclose(camera_hit, 1.0, atol=1e-7, rtol=0)
        if condition != "exact_z_all":
            points_camera = points_camera[visible]
        points_lidar = (points_camera - t) @ rotation
        result.append(
            DepthToDepthObservation(
                f"analytic-{i}",
                depth,
                points_lidar,
                CAMERA,
                depth_scale_convention="metric_z",
            )
        )
    return result
