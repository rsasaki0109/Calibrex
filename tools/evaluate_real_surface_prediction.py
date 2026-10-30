"""Camera-parametric parity and offline real stereo surface quality transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tools.camera_surface import StereoGeometry, estimate
from tools.camera_surface_prediction import gated_estimate
from tools.evaluate_direct_disparity import noisy_depth
from tools.known_geometry_joint import Report
from tools.occlusion_geometry import SCENES, observations
from tools.surface_prediction_gate import gated_estimate as old_estimate

from calibrex.solvers.chen_medioni_point_to_plane_icp_solver import (
    ChenMedioniPointToPlaneOptions,
    _estimate_target_normals,
)


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    parity = 0
    analytic = StereoGeometry(240, 240, 159.5, 95.5, 120)
    for scene in SCENES:
        for i, o in enumerate(observations(scene, "exact_z_all")):
            for seed in [0, 730, 731]:
                d = (
                    o.depth_map.copy()
                    if seed == 0
                    else noisy_depth(o.depth_map, seed + 100 * i, 0.5)
                )
                a, b = old_estimate(d), gated_estimate(d, analytic)
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                parity += 1
    report = Report(
        scope="18 offline real surface frames and27 exact analytic parity checks",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for frame in plan["frames"]:
        geometry = StereoGeometry(**frame["geometry"])
        depth = np.load(frame["depth"], allow_pickle=False)
        arrays = np.load(frame["path"], allow_pickle=False)
        transform = np.array(frame["reference"])
        source = arrays["source_lidar"] @ transform[:3, :3].T + transform[:3, 3]
        normals, valid, _ = _estimate_target_normals(source, ChenMedioniPointToPlaneOptions())
        tree = cKDTree(source)
        estimates = [
            ("baseline21", estimate(depth, 21, geometry)),
            ("predictive21", gated_estimate(depth, geometry)),
        ]
        for method, e in estimates:
            mask = e["accepted"]
            v, u = np.nonzero(mask)
            v = v + 10
            u = u + 10
            z = e["depth"][mask]
            target = np.column_stack(
                [(u - geometry.cx) * z / geometry.fx, (v - geometry.cy) * z / geometry.fy, z]
            )
            distance, index = tree.query(target)
            supported = (distance <= 1) & valid[index]
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.abs(
                            np.sum(e["normal"][mask][supported] * normals[index[supported]], axis=1)
                        ),
                        0,
                        1,
                    )
                )
            )
            cells = len(
                set(
                    zip(
                        ((v - 10) * 4 // mask.shape[0]).tolist(),
                        ((u - 10) * 4 // mask.shape[1]).tolist(),
                        strict=True,
                    )
                )
            )
            report.cases.append(
                {
                    "drive": frame["drive"],
                    "frame": frame["frame"],
                    "method": method,
                    "analytic_parity_checks": parity,
                    "geometry": frame["geometry"],
                    "selected": int(mask.sum()),
                    "domain": int(mask.size),
                    "retention": float(mask.mean()),
                    "cells": cells,
                    "nearest_lidar_quantiles_m": np.quantile(distance, [0.1, 0.5, 0.9]).tolist()
                    if len(distance)
                    else None,
                    "within_1m_fraction": float(np.mean(distance <= 1)) if len(distance) else None,
                    "normal_pairs": int(supported.sum()),
                    "within_15deg_fraction": float(np.mean(angle <= 15)) if len(angle) else None,
                    "normal_angle_quantiles_deg": np.quantile(angle, [0.1, 0.5, 0.9]).tolist()
                    if len(angle)
                    else None,
                }
            )
            output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.plan, a.output)
