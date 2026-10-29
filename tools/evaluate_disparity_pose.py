"""Compare pixel-scaled inverse-depth pose fitting with archived D2D on spent data."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_camera_lidar_development_matrix import pose_metrics
from tools.run_stereo_translation_refinement import digest

from calibrex.solvers.stereo_disparity_pose import (
    DisparityPoseOptions,
    DisparityPoseResult,
    solve_disparity_pose,
)
from calibrex.solvers.stereo_translation_refiner import StereoSurfaceObservation


class PoseCase(BaseModel):
    case_id: str
    bundle: str
    d2d_sha256: str
    seed_sha256: str


class PosePlan(BaseModel):
    support_mode: Literal["hard", "tapered"] = "hard"
    scope: str
    cases: list[PoseCase]
    source_sha256: dict[str, str]


class PoseCaseResult(BaseModel):
    case_id: str
    input_sha256: dict[str, str]
    initial_T_camera_lidar: list[list[float]]
    result: DisparityPoseResult
    initial: dict[str, Any]
    d2d: dict[str, Any]
    candidate: dict[str, Any]
    returned: dict[str, Any]
    stage_recovered: bool


class PoseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str
    source_commit: str
    plan_sha256: str
    cases: list[PoseCaseResult]


def evaluate(plan_path: Path, output: Path) -> PoseEvaluation:
    plan = PosePlan.model_validate_json(plan_path.read_text())
    if len(plan.cases) != 12 or len({c.case_id for c in plan.cases}) != 12:
        raise ValueError("exact twelve-case inventory required")
    for filename, expected in plan.source_sha256.items():
        if digest(Path(filename)) != expected:
            raise ValueError("frozen source changed: " + filename)
    if output.exists():
        raise ValueError("do not overwrite evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = PoseEvaluation(
        scope=plan.scope,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=digest(plan_path),
        cases=[],
    )
    for case in plan.cases:
        bundle = Path(case.bundle)
        seed_path = bundle / "seed.json"
        d2d_path = bundle / "d2d.json"
        if digest(seed_path) != case.seed_sha256 or digest(d2d_path) != case.d2d_sha256:
            raise ValueError("frozen seed/baseline changed")
        # Input pose is read from the baseline solver report; it contains no reference pose.
        baseline = json.loads(d2d_path.read_text())
        initial = np.array(baseline["initial_T_camera_lidar"])
        config_path = bundle / "calib-frontend.json"
        if not config_path.exists():
            config_path = bundle / "calib.json"
        config = json.loads(config_path.read_text())
        mapping_path = bundle / "frontend.json"
        mapping = json.loads(mapping_path.read_text())
        inputs = {str(p): digest(p) for p in [seed_path, d2d_path, config_path, mapping_path]}
        obs = []
        for name in config["meta"]["bag_names"]:
            cloud = bundle / (name + ".ply")
            depth = bundle / (name + ".depth.npy")
            for p in [cloud, depth]:
                if digest(p) != mapping[p.name]["sha256"]:
                    raise ValueError("changed sensor input")
                inputs[str(p)] = digest(p)
            meta_path = Path(mapping[cloud.name]["source"]).parent / "frontend.json"
            inputs[str(meta_path)] = digest(meta_path)
            meta = json.loads(meta_path.read_text())
            obs.append(
                StereoSurfaceObservation(
                    name,
                    np.load(depth, allow_pickle=False),
                    read_koide_ply(cloud)[:, :3],
                    tuple(config["camera"]["intrinsics"]),
                    meta["stereo"]["baseline_m"],
                )
            )
        fit = solve_disparity_pose(
            obs, initial, DisparityPoseOptions(support_mode=plan.support_mode)
        )
        # Reference is used only by the scorer after fitting.
        seed = json.loads(seed_path.read_text())
        np.testing.assert_array_equal(initial, seed["initial_T_camera_lidar"])
        reference = np.array(seed["reference_T_camera_lidar"])
        returned = pose_metrics(np.array(fit.output_T_camera_lidar), reference, True)
        record = PoseCaseResult(
            case_id=case.case_id,
            input_sha256=inputs,
            initial_T_camera_lidar=initial.tolist(),
            result=fit,
            initial=pose_metrics(initial, reference, True),
            d2d=pose_metrics(
                np.array(baseline["output_T_camera_lidar"]),
                reference,
                baseline["result"]["status"] == "converged",
            ),
            candidate=pose_metrics(np.array(fit.candidate_T_camera_lidar), reference, True),
            returned=returned,
            stage_recovered=fit.accepted and returned["recovered"],
        )
        report.cases.append(record)
        output.write_text(report.model_dump_json(indent=2) + "\n")
        print(
            case.case_id,
            fit.status,
            record.candidate["rotation_deg"],
            record.candidate["translation_m"],
            record.stage_recovered,
            flush=True,
        )
    output.with_suffix(".schema.json").write_text(
        json.dumps(PoseEvaluation.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    evaluate(a.plan, a.output)
