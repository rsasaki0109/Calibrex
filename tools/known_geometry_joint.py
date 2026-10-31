"""Independent analytic ray casting control for joint D2D calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    evaluate_depth_to_depth_mi,
    resolve_depth_to_depth_options,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofSolver, apply_local_se3_delta

CAMERA = DepthToDepthCameraModel(width=320, height=192, fx=240, fy=240, cx=159.5, cy=95.5)
SPHERES = [
    ((-3.0, -1.0, 9.0), 1.8),
    ((2.0, 1.0, 11.0), 2.3),
    ((0.0, -2.0, 7.0), 1.0),
    ((4.0, -2.0, 15.0), 2.0),
    ((-4.0, 3.0, 14.0), 2.5),
]
CONDITIONS = (
    "exact_z_visible",
    "exact_z_all",
    "exact_range_visible",
    "biased_z_visible",
    "holes_z_visible",
)


def raycast(origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return first intersection parameter for arbitrary unnormalized rays."""
    normal = np.array([0.12, -0.08, 1.0])
    denom = directions @ normal
    distance = np.divide(
        22.0 - origin @ normal, denom, out=np.full(denom.shape, np.inf), where=denom > 0
    )
    a = np.sum(directions * directions, axis=-1)
    for center, radius in SPHERES:
        offset = origin - np.array(center)
        b = directions @ offset
        c = offset @ offset - radius * radius
        discriminant = b * b - a * c
        root = np.sqrt(np.maximum(discriminant, 0))
        near, far = (-b - root) / a, (-b + root) / a
        hit = np.where(near > 0, near, far)
        distance = np.minimum(distance, np.where((discriminant >= 0) & (hit > 0), hit, np.inf))
    return distance


def reference() -> SE3:
    return SE3(
        (0.2, -0.1, 0.3),
        tuple(Rotation.from_euler("xyz", [2.0, -3.0, 1.0], degrees=True).as_quat()),
    )


def observations(condition: str) -> list[DepthToDepthObservation]:
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
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
        z = raycast(origin, rays)
        depth = z.copy()
        if condition == "exact_range_visible":
            depth *= np.linalg.norm(rays, axis=-1)
        elif condition == "biased_z_visible":
            depth += 0.25
        elif condition == "holes_z_visible":
            depth[50:140, 110:210] = np.nan
        hit = raycast(origin + t, lidar_rays_camera)
        points_camera = t + hit[:, None] * lidar_rays_camera
        # Continuous camera-ray visibility, independent of solver pixel sampling.
        camera_hit = raycast(origin, points_camera)
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
                depth_scale_convention="metric_range"
                if condition == "exact_range_visible"
                else "metric_z",
            )
        )
    return result


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = "synthetic diagnostic only; not held-out real calibration evidence"
    source_commit: str
    plan_sha256: str
    cases: list[dict[str, Any]] = []


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    spec = json.loads(plan.read_text())
    for p, sha in spec["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed frozen source")
    report = Report(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    ref = reference()
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(ref.rotation_quat_xyzw).as_matrix()
    matrix[:3, 3] = ref.translation_m
    for condition in CONDITIONS:
        batch = observations(condition)
        options = resolve_depth_to_depth_options(batch)
        truth = evaluate_depth_to_depth_mi(batch, ref, options)
        for sign in [1, -1, 0]:
            initial = apply_local_se3_delta(
                ref,
                tuple(sign * np.array([1.0, -0.5, 0.75])),
                tuple(sign * np.array([0.15, -0.1, 0.1])),
            )
            fit = BorerSixDofSolver().solve(batch, initial)
            estimated = np.eye(4)
            estimated[:3, :3] = Rotation.from_quat(
                fit.transform_camera_lidar.rotation_quat_xyzw
            ).as_matrix()
            estimated[:3, 3] = fit.transform_camera_lidar.translation_m
            metrics = pose_metrics(estimated, matrix, fit.status == "converged")
            report.cases.append(
                {
                    "condition": condition,
                    "sign": sign,
                    "point_counts": [len(o.lidar_points) for o in batch],
                    "valid_depth_counts": [int(np.isfinite(o.depth_map).sum()) for o in batch],
                    "truth_evaluation": truth.as_dict(),
                    "result": fit.as_dict(),
                    "metrics": metrics,
                }
            )
            output.write_text(report.model_dump_json(indent=2) + "\n")
            print(condition, sign, metrics, flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.plan, args.output)
