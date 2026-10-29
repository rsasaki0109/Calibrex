"""Experimental single-frame stereo/LiDAR registration using existing tangent ICP."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.core.geometry import SE3
from calibrex.solvers.chen_medioni_point_to_plane_icp_solver import (
    ChenMedioniPointToPlaneIcpSolver,
    ChenMedioniPointToPlaneOptions,
)
from calibrex.solvers.robust_point_to_point_icp_solver import IcpPoint, RobustPointToPointIcpOptions
from calibrex.solvers.stereo_translation_refiner import StereoSurfaceObservation

COMMON = RobustPointToPointIcpOptions(
    correspondence_distance_m=0.5,
    trim_fraction=0.8,
    mutual_correspondences=True,
    min_correspondences=64,
    holdout_ratio=0.2,
    holdout_voxel_size_m=0.5,
    effective_diagnostics=False,
    multi_start_diagnostics=False,
)
PLANES = ChenMedioniPointToPlaneOptions()


def point_records(points: np.ndarray, prefix: str) -> list[IcpPoint]:
    return [IcpPoint(f"{prefix}-{i:06d}", tuple(p)) for i, p in enumerate(points)]


def subsample(points: np.ndarray) -> np.ndarray:
    """Deterministic 0.2 m voxel-first sampling, capped at 6,000 points."""
    _, first = np.unique(np.floor(points / 0.2).astype(np.int64), axis=0, return_index=True)
    ids = np.sort(first)
    if len(ids) > 6000:
        ids = ids[np.linspace(0, len(ids) - 1, 6000, dtype=int)]
    return points[ids]


def register(source: np.ndarray, target: np.ndarray, initial: np.ndarray) -> dict[str, Any]:
    transform = SE3(tuple(initial[:3, 3]), tuple(Rotation.from_matrix(initial[:3, :3]).as_quat()))
    fit = ChenMedioniPointToPlaneIcpSolver().solve(
        point_records(source, "source"), point_records(target, "target"), transform, COMMON, PLANES
    )
    candidate = None
    if fit.transform_target_source is not None:
        candidate = np.eye(4)
        candidate[:3, :3] = Rotation.from_quat(
            fit.transform_target_source.rotation_quat_xyzw
        ).as_matrix()
        candidate[:3, 3] = fit.transform_target_source.translation_m
    accepted = fit.status == "converged" and candidate is not None
    return {
        "accepted": accepted,
        "result": fit.as_dict(),
        "candidate_T_camera_lidar": candidate.tolist() if candidate is not None else None,
        "output_T_camera_lidar": (candidate if accepted else initial).tolist(),
        "source_points": len(source),
        "target_points": len(target),
        "common_options": asdict(COMMON),
        "plane_options": asdict(PLANES),
    }


def run(observation: StereoSurfaceObservation, initial: np.ndarray) -> dict[str, Any]:
    """Build independent target geometry; preserve unmatched points as missing evidence."""
    depth = observation.depth_m
    v, u = np.indices(depth.shape)
    fx, fy, cx, cy = observation.intrinsics
    valid = np.isfinite(depth) & (depth >= 0.1) & (depth <= 80)
    target = np.column_stack(
        ((u[valid] - cx) * depth[valid] / fx, (v[valid] - cy) * depth[valid] / fy, depth[valid])
    )
    points = observation.lidar_points
    camera = points @ initial[:3, :3].T + initial[:3, 3]
    z = camera[:, 2]
    safe = np.maximum(z, 0.1)
    px, py = fx * camera[:, 0] / safe + cx, fy * camera[:, 1] / safe + cy
    visible = (
        (z >= 0.1)
        & (z <= 80)
        & (px >= 0)
        & (px < depth.shape[1])
        & (py >= 0)
        & (py < depth.shape[0])
    )
    return register(subsample(points[visible]), subsample(target), initial)
