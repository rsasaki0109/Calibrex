"""Prepare portable, provenance-locked actual-frame ICP evaluation fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import jsonschema
import numpy as np

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.data.kitti import kitti_oxts_pose_world_imu, read_oxts_packet
from calibrex.data.kitti_raw_lidar_window_integration import _read_imu_to_velodyne
from calibrex.solvers.robust_point_to_point_icp_solver import (
    RobustPointToPointIcpOptions,
    _rotation_matrix,
)

SCHEMA = Path("tools/schemas/icp_pair_fixture.schema.json")
SOURCES = ("tools/prepare_kitti_icp_pairs.py", str(SCHEMA),
           "src/calibrex/data/kitti.py", "src/calibrex/core/geometry.py",
           "src/calibrex/solvers/robust_point_to_point_icp_solver.py",
           "src/calibrex/data/kitti_raw_lidar_window_integration.py")


def prepare_pair(root: Path, calibration: Path, source_id: int, target_id: int,
                 max_iterations: int) -> dict[str, object]:
    names = [f"{value:010d}" for value in (source_id, target_id)]
    pose_paths = [root / "oxts/data" / (name + ".txt") for name in names]
    packets = [read_oxts_packet(path) for path in pose_paths]
    poses = [kitti_oxts_pose_world_imu(packet, packets[0]) for packet in packets]
    imu_from_velo = _read_imu_to_velodyne(calibration)
    matrices = []
    for pose in poses:
        matrix = np.eye(4)
        matrix[:3, :3] = _rotation_matrix(pose)
        matrix[:3, 3] = pose.translation_m
        matrices.append(matrix @ imu_from_velo)
    relative = np.linalg.inv(matrices[1]) @ matrices[0]
    reference = SE3(tuple(relative[:3, 3]),
                    quaternion_xyzw_from_rotation_matrix(relative[:3, :3].reshape(-1)))
    perturbation = SE3((.03, -.02, .015), (.001, -.002, .0015, 1.))
    cloud_names = ["velodyne_points/data/" + name + ".bin" for name in names]
    files = {name: root / name for name in cloud_names}
    files.update({"oxts/data/" + name + ".txt": path
                  for name, path in zip(names, pose_paths, strict=True)})
    files["calib_imu_to_velo.txt"] = calibration
    fixture = {
        "schema_version": "calibrex.icp_pair_fixture/v0.1",
        "source": cloud_names[0], "target": cloud_names[1],
        "initial_transform": reference.compose(perturbation).as_dict(),
        "reference_transform": reference.as_dict(),
        "options": asdict(RobustPointToPointIcpOptions(max_iterations=max_iterations)),
        "provenance": {
            "generator": "calibrex.prepare_kitti_icp_pairs/v0.1",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                      text=True).strip(),
            "source_sha256": {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in SOURCES},
            "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in files.items()},
            "sequence": root.name, "source_frame": names[0], "target_frame": names[1],
            "initial_perturbation": perturbation.as_dict(),
            "reference_role": "OXTS navigation seed and consistency reference; no deskew",
            "reference_formula": ("inverse(T_world_imu_target*T_imu_velo)*"
                                  "(T_world_imu_source*T_imu_velo)"),
        },
    }
    jsonschema.validate(fixture, json.loads(SCHEMA.read_bytes()))
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--source-frames", default="115,230,345")
    parser.add_argument("--frame-gap", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.max_iterations < 1 or args.frame_gap < 1:
        parser.error("iteration budget and frame gap must be positive")
    args.output.mkdir(parents=True)
    for frame in map(int, args.source_frames.split(",")):
        fixture = prepare_pair(args.data_root, args.calibration, frame,
                               frame + args.frame_gap, args.max_iterations)
        path = args.output / f"{frame:010d}-{frame+args.frame_gap:010d}.json"
        path.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(path, flush=True)


if __name__ == "__main__":
    main()
