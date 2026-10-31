"""Matched single-frame surface ICP / D2D comparison on twelve spent-data seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation
from tools.evaluate_disparity_pose import PosePlan
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_camera_lidar_development_matrix import pose_metrics
from tools.run_stereo_translation_refinement import digest
from tools.stereo_surface_registration import run as register

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofSolver
from calibrex.solvers.stereo_translation_refiner import StereoSurfaceObservation


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = "spent-data central-frame surface ICP versus matched-frame D2D; no SOTA claim"
    source_commit: str
    plan_sha256: str
    cases: list[dict[str, Any]]


def finite_json(value: Any) -> Any:
    """Represent nonfinite solver evidence explicitly as strings, never JSON NaN."""
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def evaluate(plan_path: Path, output: Path) -> Evaluation:
    plan = PosePlan.model_validate_json(plan_path.read_text())
    for p, sha in plan.source_sha256.items():
        if digest(Path(p)) != sha:
            raise ValueError("changed source " + p)
    if len(plan.cases) != 12:
        raise ValueError("twelve cases required")
    if output.exists():
        raise ValueError("do not overwrite results")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = Evaluation(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=digest(plan_path),
        cases=[],
    )
    for case in plan.cases:
        bundle = Path(case.bundle)
        seed = bundle / "seed.json"
        d2d = bundle / "d2d.json"
        if digest(seed) != case.seed_sha256 or digest(d2d) != case.d2d_sha256:
            raise ValueError("changed initializer/baseline")
        initial = np.array(json.loads(d2d.read_text())["initial_T_camera_lidar"])
        config_path = bundle / "calib-frontend.json"
        if not config_path.exists():
            config_path = bundle / "calib.json"
        config = json.loads(config_path.read_text())
        names = config["meta"]["bag_names"]
        if len(names) != 3:
            raise ValueError("three ordered archived frames required")
        name = names[1]
        mapping_path = bundle / "frontend.json"
        mapping = json.loads(mapping_path.read_text())
        cloud = bundle / (name + ".ply")
        depth_path = bundle / (name + ".depth.npy")
        for p in [cloud, depth_path]:
            if digest(p) != mapping[p.name]["sha256"]:
                raise ValueError("changed sensor data")
        meta_path = Path(mapping[cloud.name]["source"]).parent / "frontend.json"
        meta = json.loads(meta_path.read_text())
        depth = np.load(depth_path, allow_pickle=False)
        points = read_koide_ply(cloud)[:, :3]
        intrinsics = tuple(config["camera"]["intrinsics"])
        obs = StereoSurfaceObservation(
            name, depth, points, intrinsics, meta["stereo"]["baseline_m"]
        )
        icp = register(obs, initial)
        transform = SE3(
            tuple(initial[:3, 3]), tuple(Rotation.from_matrix(initial[:3, :3]).as_quat())
        )
        model = DepthToDepthCameraModel(
            width=depth.shape[1],
            height=depth.shape[0],
            **dict(zip(("fx", "fy", "cx", "cy"), intrinsics, strict=True)),
        )
        baseline = BorerSixDofSolver().solve(
            [DepthToDepthObservation(name, depth, points, model)], transform
        )
        estimate = np.eye(4)
        estimate[:3, :3] = Rotation.from_quat(
            baseline.transform_camera_lidar.rotation_quat_xyzw
        ).as_matrix()
        estimate[:3, 3] = baseline.transform_camera_lidar.translation_m
        reference = np.array(json.loads(seed.read_text())["reference_T_camera_lidar"])
        candidate = (
            np.array(icp["candidate_T_camera_lidar"])
            if icp["candidate_T_camera_lidar"] is not None
            else None
        )
        if not icp["accepted"]:
            np.testing.assert_array_equal(icp["output_T_camera_lidar"], initial)
        record = {
            "case_id": case.case_id,
            "frame_id": name,
            "input_sha256": {
                str(p): digest(p)
                for p in [seed, d2d, config_path, mapping_path, cloud, depth_path, meta_path]
            },
            "initial_T_camera_lidar": initial.tolist(),
            "initial": pose_metrics(initial, reference, True),
            "icp": finite_json(icp),
            "icp_metrics": pose_metrics(candidate, reference, icp["accepted"]),
            "icp_raw_metrics": pose_metrics(candidate, reference, candidate is not None),
            "d2d": finite_json(baseline.as_dict()),
            "d2d_output_T_camera_lidar": estimate.tolist(),
            "d2d_metrics": pose_metrics(estimate, reference, baseline.status == "converged"),
        }
        report.cases.append(record)
        output.write_text(report.model_dump_json(indent=2) + "\n")
        print(
            case.case_id,
            name,
            icp["result"]["status"],
            record["icp_metrics"],
            "d2d",
            record["d2d_metrics"],
            flush=True,
        )
    output.with_suffix(".schema.json").write_text(
        json.dumps(Evaluation.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    evaluate(a.plan, a.output)
