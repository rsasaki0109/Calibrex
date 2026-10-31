"""Oracle first-surface depth sampler for synthetic measurement diagnostics only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from tools.occlusion_geometry import SCENES, raycast

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation


@dataclass(frozen=True)
class AnalyticObservation(DepthToDepthObservation):
    scene: str = "original"
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    measurement_offset_m: float = 0.0


def depth_residual(observation: AnalyticObservation, pose: SE3) -> tuple[np.ndarray, np.ndarray]:
    """Continuous first-surface optical z at candidate projections, same image domain."""
    camera = observation.camera
    rotation = Rotation.from_quat(pose.rotation_quat_xyzw).as_matrix()
    xyz = observation.lidar_points @ rotation.T + pose.translation_m
    z = np.maximum(xyz[:, 2], 1e-9)
    u = camera.fx * xyz[:, 0] / z + camera.cx
    v = camera.fy * xyz[:, 1] / z + camera.cy
    valid = (
        (xyz[:, 2] > 0.1) & (u >= 0) & (u < camera.width - 1) & (v >= 0) & (v < camera.height - 1)
    )
    residual = np.ones(len(xyz))
    ids = np.flatnonzero(valid)
    rays = xyz[ids] / xyz[ids, 2, None]
    measured = (
        raycast(np.array(observation.camera_origin), rays, SCENES[observation.scene])
        + observation.measurement_offset_m
    )
    finite = np.isfinite(measured) & (measured > 0)
    residual[ids[finite]] = xyz[ids[finite], 2] - measured[finite]
    valid[ids[~finite]] = False
    return residual, valid
