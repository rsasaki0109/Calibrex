"""Fixed symmetric versus asymmetric metric fitting across three analytic scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from tools.joint_metric_pose import depth_residual
from tools.joint_metric_pose import solve as symmetric
from tools.joint_multistart_control import matrix
from tools.joint_occlusion_pose import solve as asymmetric
from tools.known_geometry_joint import Report, reference
from tools.occlusion_geometry import SCENES, observations
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    for p, sha in json.loads(plan.read_text())["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed source")
    report = Report(
        scope="one-sided occlusion residual; three synthetic scene layouts",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    )
    for scene in SCENES:
        for condition in ["exact_z_visible", "exact_z_all"]:
            batch = observations(scene, condition)
            for sign in [1, -1, 0]:
                initial = apply_local_se3_delta(
                    reference(),
                    tuple(sign * np.array([1.0, -0.5, 0.75])),
                    tuple(sign * np.array([0.15, -0.1, 0.1])),
                )
                methods = {}
                for name, solver in [("symmetric", symmetric), ("asymmetric", asymmetric)]:
                    began = time.perf_counter()
                    fit = solver(batch, initial)
                    elapsed = time.perf_counter() - began
                    pose = SE3(
                        tuple(fit["transform"]["translation_m"]),
                        tuple(fit["transform"]["rotation_quat_xyzw"]),
                    )
                    diagnostics = []
                    for obs in batch:
                        residual, valid = depth_residual(obs, pose)
                        diagnostics.append(
                            {
                                "points": len(residual),
                                "supported": int(valid.sum()),
                                "behind_over_20cm": int(((residual > 0.2) & valid).sum()),
                                "in_front_over_20cm": int(((residual < -0.2) & valid).sum()),
                            }
                        )
                    methods[name] = {
                        "result": fit,
                        "seconds": elapsed,
                        "population": diagnostics,
                        "metrics": pose_metrics(
                            matrix(pose), matrix(reference()), fit["converged"]
                        ),
                    }
                report.cases.append(
                    {"scene": scene, "condition": condition, "sign": sign, "methods": methods}
                )
                output.write_text(report.model_dump_json(indent=2) + "\n")
                print(
                    scene,
                    condition,
                    sign,
                    {k: v["metrics"] for k, v in methods.items()},
                    flush=True,
                )
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
