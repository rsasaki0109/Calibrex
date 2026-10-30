"""Frozen frame-wise native ICP comparison on prepared stereo/LiDAR geometry."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tools.evaluate_real_metric_transfer import as_pose
from tools.joint_multistart_control import matrix
from tools.known_geometry_joint import Report
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.solvers.chen_medioni_point_to_plane_icp_solver import ChenMedioniPointToPlaneIcpSolver
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpPoint,
    RobustPointToPointIcpOptions,
    RobustPointToPointIcpSolver,
)


def run(plan_path: Path, workspace: Path, output: Path) -> None:
    if output.exists() or workspace.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed frozen file " + p)
    workspace.mkdir(parents=True)
    report = Report(
        scope="108 frame-wise stereo/LiDAR native ICP fits",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    options = RobustPointToPointIcpOptions(multi_start_diagnostics=False)
    for frame in plan["frames"]:
        arrays = np.load(frame["path"], allow_pickle=False)
        source = [IcpPoint(f"s-{i:06d}", tuple(x)) for i, x in enumerate(arrays["source_lidar"])]
        target = [IcpPoint(f"t-{i:06d}", tuple(x)) for i, x in enumerate(arrays["target_camera"])]
        for start in plan["starts"][frame["drive"]]:
            for method in ["point", "plane"]:
                began = time.perf_counter()
                if method == "point":
                    result = RobustPointToPointIcpSolver().solve(
                        source, target, as_pose(start["initial"]), options
                    )
                else:
                    result = ChenMedioniPointToPlaneIcpSolver().solve(
                        source, target, as_pose(start["initial"]), options
                    )
                elapsed = time.perf_counter() - began
                full = result.as_dict()
                artifact = workspace / (
                    frame["drive"]
                    + "-"
                    + frame["frame"]
                    + "-"
                    + start["start"]
                    + "-"
                    + method
                    + ".json.gz"
                )
                with gzip.open(artifact, "wt") as f:
                    json.dump(full, f, allow_nan=False)
                summary = {
                    k: v
                    for k, v in full.items()
                    if k not in ["train_source_ids", "holdout_source_ids"]
                }
                summary["train_source_count"] = len(result.train_source_ids)
                summary["holdout_source_count"] = len(result.holdout_source_ids)
                pose = result.transform_target_source
                metrics = pose_metrics(
                    matrix(pose) if pose else None,
                    np.array(start["reference"]),
                    result.status == "converged",
                )
                report.cases.append(
                    {
                        "drive": frame["drive"],
                        "frame": frame["frame"],
                        "start": start["start"],
                        "method": method,
                        "source_points": len(source),
                        "target_points": len(target),
                        "resolved_common_options": asdict(options),
                        "result": summary,
                        "pose_metrics": metrics,
                        "seconds_including_diagnostics": elapsed,
                        "full_result_path": str(artifact),
                        "full_result_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                )
                output.write_text(report.model_dump_json(indent=2) + "\n")
                print(
                    frame["drive"],
                    frame["frame"],
                    start["start"],
                    method,
                    result.status,
                    metrics,
                    flush=True,
                )
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["plan", "workspace", "output"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.plan, a.workspace, a.output)
