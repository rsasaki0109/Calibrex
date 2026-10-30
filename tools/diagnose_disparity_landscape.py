"""Offline fixed-bias pose paths for every archived direct-disparity fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tools.diagnose_analytic_stopping import soft_cost
from tools.evaluate_direct_disparity import noisy_depth
from tools.evaluate_pose_bias import planes
from tools.joint_disparity_bias import residuals
from tools.known_geometry_joint import Report, reference
from tools.occlusion_geometry import observations

from calibrex.core.geometry import SE3


def interpolate(start: SE3, end: SE3, fraction: float) -> SE3:
    rotations = Rotation.from_quat([start.rotation_quat_xyzw, end.rotation_quat_xyzw])
    quat = Slerp([0, 1], rotations)(fraction).as_quat()
    translation = (1 - fraction) * np.array(start.translation_m) + fraction * np.array(
        end.translation_m
    )
    return SE3(tuple(translation), tuple(quat))


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for p, sha in plan["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed frozen file " + p)
    archive = json.loads(Path(plan["archive"]).read_text())["cases"]
    report = Report(
        scope="171 offline pose-path diagnostics; no new calibration fits",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    cache = {}
    for case in archive:
        key = (case["scene"], case["condition"], case["noise_seed"], case["injected_bias_m"])
        if key not in cache:
            clean = (
                planes()
                if case["scene"] == "plane"
                else observations(case["scene"], case["condition"])
            )
            depths = [
                o.depth_map.copy()
                if case["sigma_px"] == 0
                else noisy_depth(o.depth_map, case["noise_seed"] + 100 * i, case["sigma_px"])
                for i, o in enumerate(clean)
            ]
            cache[key] = [
                replace(o, depth_map=d + case["injected_bias_m"])
                for o, d in zip(clean, depths, strict=True)
            ]
        batch = cache[key]
        assert [hashlib.sha256(o.depth_map.tobytes()).hexdigest() for o in batch] == case[
            "depth_array_sha256"
        ]
        p = case["result"]["transform"]
        fitted = SE3(tuple(p["translation_m"]), tuple(p["rotation_quat_xyzw"]))
        tracks = []
        for label, bias in [
            ("injected", case["injected_bias_m"]),
            ("fitted", case["result"]["bias_m"]),
        ]:
            samples = []
            common = None
            for alpha in np.linspace(0, 1, 11):
                _, s, valid = residuals(batch, interpolate(reference(), fitted, float(alpha)), bias)
                common = valid.copy() if common is None else common & valid
                samples.append((float(alpha), s, valid))
            tracks.append(
                {
                    "bias_source": label,
                    "bias_m": bias,
                    "common_supported": int(common.sum()),
                    "path": [
                        {
                            "fraction_to_fit": a,
                            "cost": soft_cost(s),
                            "supported": int(v.sum()),
                            "common_support_cost": soft_cost(s[common]),
                        }
                        for a, s, v in samples
                    ],
                }
            )
        np.testing.assert_allclose(
            tracks[1]["path"][-1]["cost"], case["result"]["cost"], rtol=1e-9, atol=1e-8
        )
        report.cases.append(
            {
                k: case[k]
                for k in ["scene", "condition", "noise_seed", "sigma_px", "injected_bias_m", "sign"]
            }
            | {"archived_joint_recovered": case["joint_recovered"], "tracks": tracks}
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
