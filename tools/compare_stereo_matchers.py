"""Image-only ZNCC versus SGBM with matched single-frame D2D calibration."""

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
from tools.stereo_rerectification import compute_depth as sgbm_depth
from tools.zncc_stereo_frontend import SETTINGS
from tools.zncc_stereo_frontend import compute_depth as zncc_depth

from calibrex.core.geometry import SE3
from calibrex.data.kitti import read_calibration_file
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofSolver


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = "spent-data central-frame image-only ZNCC versus SGBM; fixed single-frame D2D"
    source_commit: str
    plan_sha256: str
    input_sha256: dict[str, str]
    frames: list[dict[str, Any]]
    cases: list[dict[str, Any]]


def evaluate(plan_path: Path, workspace: Path, output: Path) -> Comparison:
    import cv2

    cv2.setNumThreads(1)
    plan = PosePlan.model_validate_json(plan_path.read_text())
    for p, sha in plan.source_sha256.items():
        if digest(Path(p)) != sha:
            raise ValueError("changed source " + p)
    if len(plan.cases) != 12:
        raise ValueError("twelve cases required")
    if output.exists():
        raise ValueError("do not overwrite results")
    workspace.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = Comparison(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=digest(plan_path),
        input_sha256={},
        frames=[],
        cases=[],
    )
    cache: dict[str, dict[str, np.ndarray]] = {}
    for case in plan.cases:
        bundle = Path(case.bundle)
        seed = bundle / "seed.json"
        d2d = bundle / "d2d.json"
        if digest(seed) != case.seed_sha256 or digest(d2d) != case.d2d_sha256:
            raise ValueError("changed seed")
        initial = np.array(json.loads(d2d.read_text())["initial_T_camera_lidar"])
        config_path = bundle / "calib-frontend.json"
        if not config_path.exists():
            config_path = bundle / "calib.json"
        config = json.loads(config_path.read_text())
        name = config["meta"]["bag_names"][1]
        mapping_path = bundle / "frontend.json"
        mapping = json.loads(mapping_path.read_text())
        cloud = bundle / (name + ".ply")
        if digest(cloud) != mapping[cloud.name]["sha256"]:
            raise ValueError("changed cloud")
        points = read_koide_ply(cloud)[:, :3]
        meta_path = Path(mapping[cloud.name]["source"]).parent / "frontend.json"
        meta = json.loads(meta_path.read_text())
        key = str(meta_path)
        for p in [seed, d2d, config_path, mapping_path, cloud, meta_path]:
            report.input_sha256[str(p)] = digest(p)
        if key not in cache:
            images = [Path(p) for p in meta["source_sha256"] if p.endswith(".png")]
            cam_path = next(
                Path(p) for p in meta["source_sha256"] if p.endswith("calib_cam_to_cam.txt")
            )
            for p in [*images, cam_path]:
                if digest(p) != meta["source_sha256"][str(p)]:
                    raise ValueError("changed stereo input")
                report.input_sha256[str(p)] = digest(p)
            if cv2.__version__ != meta["stereo"]["opencv_version"]:
                raise ValueError("changed OpenCV")
            cam = read_calibration_file(cam_path)
            pl, pr = [np.array(cam["P_rect_" + c]).reshape(3, 4) for c in ["02", "03"]]
            left, right = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in images]
            old, geometry = sgbm_depth(left, right, pl, pr, meta["stereo"]["settings"])
            new = zncc_depth(left, right, pl, pr)
            cache[key] = {"sgbm": old, "zncc": new}
            files = {}
            for method, depth in cache[key].items():
                path = workspace / (case.case_id + "-" + method + ".npy")
                np.save(path, depth)
                files[str(path)] = digest(path)
            report.frames.append(
                {
                    "case_id": case.case_id,
                    "frame_id": name,
                    "depth_sha256": files,
                    "valid_pixels": {m: int(np.isfinite(d).sum()) for m, d in cache[key].items()},
                    "rectification": geometry,
                    "zncc_settings": SETTINGS,
                    "sgbm_settings": meta["stereo"]["settings"],
                }
            )
        intrinsics = config["camera"]["intrinsics"]
        transform = SE3(
            tuple(initial[:3, 3]), tuple(Rotation.from_matrix(initial[:3, :3]).as_quat())
        )
        fits = {}
        for method, depth in cache[key].items():
            model = DepthToDepthCameraModel(
                width=depth.shape[1],
                height=depth.shape[0],
                **dict(zip(("fx", "fy", "cx", "cy"), intrinsics, strict=True)),
            )
            fit = BorerSixDofSolver().solve(
                [DepthToDepthObservation(name, depth, points, model)], transform
            )
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(
                fit.transform_camera_lidar.rotation_quat_xyzw
            ).as_matrix()
            matrix[:3, 3] = fit.transform_camera_lidar.translation_m
            fits[method] = {"result": fit.as_dict(), "output_T_camera_lidar": matrix.tolist()}
        reference = np.array(json.loads(seed.read_text())["reference_T_camera_lidar"])
        for f in fits.values():
            f["metrics"] = pose_metrics(
                np.array(f["output_T_camera_lidar"]),
                reference,
                f["result"]["status"] == "converged",
            )
        # Paired point residuals on the exact common valid stereo pixel set, no depth gate.
        camera = points @ reference[:3, :3].T + reference[:3, 3]
        camera = camera[camera[:, 2] > 0.1]
        fx, fy, cx, cy = intrinsics
        u = np.rint(fx * camera[:, 0] / camera[:, 2] + cx).astype(int)
        v = np.rint(fy * camera[:, 1] / camera[:, 2] + cy).astype(int)
        h, w = next(iter(cache[key].values())).shape
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        camera, u, v = camera[inside], u[inside], v[inside]
        order = np.argsort(camera[:, 2], kind="stable")
        _, first = np.unique((v * w + u)[order], return_index=True)
        ids = order[first]
        camera, u, v = camera[ids], u[ids], v[ids]
        common = np.isfinite(cache[key]["sgbm"][v, u]) & np.isfinite(cache[key]["zncc"][v, u])
        residuals = {}
        for m, d in cache[key].items():
            z = camera[common, 2] - d[v[common], u[common]]
            residuals[m] = {
                "count": len(z),
                "median_z_m": float(np.median(z)) if len(z) else None,
                "p90_absolute_z_m": float(np.percentile(abs(z), 90)) if len(z) else None,
            }
        report.cases.append(
            {
                "case_id": case.case_id,
                "frame_id": name,
                "initial_T_camera_lidar": initial.tolist(),
                "methods": fits,
                "paired_reference_residuals": residuals,
            }
        )
        output.write_text(report.model_dump_json(indent=2) + "\n")
        print(case.case_id, {m: f["metrics"] for m, f in fits.items()}, flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(Comparison.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["plan", "workspace", "output"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    evaluate(a.plan, a.workspace, a.output)
