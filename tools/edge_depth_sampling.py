"""Image-only nearest-surface gated bilinear optical-depth sampling."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation


def depth_residual(
    observation: DepthToDepthObservation, pose: SE3
) -> tuple[np.ndarray, np.ndarray]:
    """Gate neighbors by depth proximity to nearest pixel; preserve support policy."""
    camera = observation.camera
    rotation = Rotation.from_quat(pose.rotation_quat_xyzw).as_matrix()
    xyz = observation.lidar_points @ rotation.T + pose.translation_m
    u = camera.fx * xyz[:, 0] / np.maximum(xyz[:, 2], 1e-9) + camera.cx
    v = camera.fy * xyz[:, 1] / np.maximum(xyz[:, 2], 1e-9) + camera.cy
    valid = (
        (xyz[:, 2] > 0.1) & (u >= 0) & (u < camera.width - 1) & (v >= 0) & (v < camera.height - 1)
    )
    x = np.floor(np.clip(u, 0, camera.width - 2)).astype(int)
    y = np.floor(np.clip(v, 0, camera.height - 2)).astype(int)
    depth = observation.depth_map
    if observation.depth_scale_convention == "metric_range":
        vv, uu = np.indices(depth.shape)
        depth = depth / np.sqrt(
            1 + ((uu - camera.cx) / camera.fx) ** 2 + ((vv - camera.cy) / camera.fy) ** 2
        )
    samples = np.stack([depth[y, x], depth[y, x + 1], depth[y + 1, x], depth[y + 1, x + 1]], axis=1)
    valid &= np.isfinite(samples).all(axis=1) & (samples > 0).all(axis=1)
    a, b = u - x, v - y
    weights = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], axis=1)
    anchor = (a >= 0.5).astype(int) + 2 * (b >= 0.5).astype(int)
    anchor_depth = samples[np.arange(len(samples)), anchor]
    same_surface = np.abs(samples - anchor_depth[:, None]) <= 0.2
    weights = np.where(same_surface, weights, 0.0)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
    measured = np.sum(np.where(np.isfinite(samples), samples, 0) * weights, axis=1)
    residual = np.where(valid, xyz[:, 2] - measured, 1.0)
    return residual, valid
