"""Evaluation-only stereo-motion attribution against an OXTS/calibration proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tools.joint_multistart_control import matrix
from tools.known_geometry_joint import Report

from calibrex.data.kitti import (
    kitti_oxts_pose_world_imu,
    read_calibration_file,
    read_oxts_packet,
    read_timestamps,
)


def interpolate(stamps: np.ndarray, poses: np.ndarray, stamp: int) -> np.ndarray:
    j = int(np.searchsorted(stamps, stamp))
    if j < len(stamps) and stamps[j] == stamp:
        return poses[j].copy()
    if j == 0 or j == len(stamps):
        raise ValueError("no extrapolation")
    alpha = (int(stamp) - int(stamps[j - 1])) / (int(stamps[j]) - int(stamps[j - 1]))
    result = np.eye(4)
    result[:3, :3] = Slerp([0, 1], Rotation.from_matrix(poses[j - 1 : j + 1, :3, :3]))(
        alpha
    ).as_matrix()
    result[:3, 3] = (1 - alpha) * poses[j - 1, :3, 3] + alpha * poses[j, :3, 3]
    return result


def camera_motion(
    world_imu_first: np.ndarray, world_imu_second: np.ndarray, camera_imu: np.ndarray
) -> np.ndarray:
    return (
        camera_imu @ np.linalg.inv(world_imu_second) @ world_imu_first @ np.linalg.inv(camera_imu)
    )


def errors(estimate: np.ndarray, proxy: np.ndarray) -> dict:
    return {
        "rotation_deg": float(
            np.degrees(Rotation.from_matrix(estimate[:3, :3] @ proxy[:3, :3].T).magnitude())
        ),
        "translation_m": float(np.linalg.norm(estimate[:3, 3] - proxy[:3, 3])),
        "estimate_translation_m": float(np.linalg.norm(estimate[:3, 3])),
        "proxy_translation_m": float(np.linalg.norm(proxy[:3, 3])),
    }


def run(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    report = Report(
        scope="36 evaluation-only motion proxy comparisons; no selection changes",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    archive = json.loads(Path(plan["motion_report"]).read_text())["cases"]
    for d in plan["drives"]:
        cam = read_calibration_file(Path(d["camera_calibration"]))
        imu = read_calibration_file(Path(d["imu_calibration"]))
        lidar = read_calibration_file(Path(d["lidar_calibration"]))
        projection = np.array(cam["P_rect_02"]).reshape(3, 4)
        shift = np.eye(4)
        shift[:3, 3] = np.linalg.solve(projection[:, :3], projection[:, 3])
        rect = np.eye(4)
        rect[:3, :3] = np.array(cam["R_rect_00"]).reshape(3, 3)
        velo = np.eye(4)
        velo[:3, :3] = np.array(lidar["R"]).reshape(3, 3)
        velo[:3, 3] = lidar["T"]
        vendor = np.eye(4)
        vendor[:3, :3] = np.array(imu["R"]).reshape(3, 3)
        vendor[:3, 3] = imu["T"]
        camera_lidar = shift @ rect @ velo
        np.testing.assert_allclose(
            camera_lidar, np.array(d["camera_lidar_reference"]), rtol=0, atol=1e-8
        )
        ci = camera_lidar @ vendor
        camera_times = read_timestamps(Path(d["camera_timestamps"]))
        imu_times = read_timestamps(Path(d["imu_timestamps"]))
        packets = [read_oxts_packet(x["path"]) for x in d["packets"]]
        poses = np.stack([matrix(kitti_oxts_pose_world_imu(p, packets[0])) for p in packets])
        stamps = np.array(
            [imu_times[x["index"]].timestamp_ns for x in d["packets"]], dtype=np.int64
        )
        by_index = {x["index"]: p for x, p in zip(d["packets"], poses, strict=True)}
        for case in [x for x in archive if x["drive"] == d["drive"]]:
            first, second = case["indices"]
            estimate = np.array(case["directions"][0]["transform"])
            nominal = camera_motion(
                interpolate(stamps, poses, camera_times[first].timestamp_ns),
                interpolate(stamps, poses, camera_times[second].timestamp_ns),
                ci,
            )
            variants = {}
            for offset in [-10000000, 0, 10000000]:
                proxy = camera_motion(
                    interpolate(stamps, poses, camera_times[first].timestamp_ns + offset),
                    interpolate(stamps, poses, camera_times[second].timestamp_ns + offset),
                    ci,
                )
                variants[str(offset // 1000000)] = {
                    "estimate_error": errors(estimate, proxy),
                    "proxy_change_from_nominal": errors(proxy, nominal),
                }
            indexed = camera_motion(by_index[first], by_index[second], ci)
            rotation_only = ci.copy()
            rotation_only[:3, 3] = 0
            no_lever = camera_motion(
                interpolate(stamps, poses, camera_times[first].timestamp_ns),
                interpolate(stamps, poses, camera_times[second].timestamp_ns),
                rotation_only,
            )
            report.cases.append(
                {
                    "drive": d["drive"],
                    "indices": case["indices"],
                    "motion_internal_pass": case["internal_consistency_passed"],
                    "nominal_error": errors(estimate, nominal),
                    "nominal_proxy": nominal.tolist(),
                    "time_shift_variants_ms": variants,
                    "same_index_proxy_change": errors(indexed, nominal),
                    "omit_lever_arm_proxy_change": errors(no_lever, nominal),
                    "camera_imu_translation_m": ci[:3, 3].tolist(),
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
