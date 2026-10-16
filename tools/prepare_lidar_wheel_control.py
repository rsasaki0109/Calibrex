"""Prepare a full public OXTS wheel-proxy/generated-sensor execution control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.data.kitti import kitti_oxts_pose_world_imu, read_oxts_packet, read_timestamps
from calibrex.data.kitti_ins_lidar import se3_matrix
from calibrex.solvers.vehicle_frame_solver import motions_from_poses

VERSION = "slac.lidar_wheel_preparation_benchmark/v0.1"
SCOPE = (
    "Full public KITTI OXTS vf/wu wheel proxy and generated sensor motions; "
    "KITTI has no wheel odometry. No raw LiDAR registration or independent accuracy evidence."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_control(
    times: NDArray[np.float64], body: NDArray[np.float64], values: NDArray[np.float64]
) -> dict:
    """Use every original pose and proxy sample with unchanged motion defaults."""

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_euler("xyz", [1.0, -2.0, 3.0], degrees=True).as_matrix()
    extrinsic[:3, 3] = [0.8, -0.3, 0.9]
    motions = motions_from_poses(times, list(body @ extrinsic))
    return {
        "schema_version": VERSION,
        "record_type": "problem",
        "motions": [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(m).items()}
            for m in motions
        ],
        "times_s": times.tolist(),
        "speed_mps": values[:, 8].tolist(),
        "yaw_rate_rps": values[:, 22].tolist(),
        "options": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oxts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.oxts_dir
    files = [source / "timestamps.txt", *[source / f"data/{i:010d}.txt" for i in range(2762)]]
    original = json.loads((source / "provenance.json").read_bytes())
    members = original.get("members", original.get("files"))
    if any(sha(p) != members[p.relative_to(source).as_posix()]["sha256"] for p in files):
        raise ValueError("original download input changed")
    times = np.asarray(
        [t.timestamp_ns * 1.0e-9 for t in read_timestamps(source / "timestamps.txt")][:2762]
    )
    packets = [read_oxts_packet(source / f"data/{i:010d}.txt") for i in range(2762)]
    body = np.stack([se3_matrix(kitti_oxts_pose_world_imu(p, packets[0])) for p in packets])
    values = np.array([[float(v) for v in p.read_text().split()] for p in files[1:]])
    problem = derive_control(times, body, values)
    problem["provenance"] = {
        "generator_sha256": sha(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset": "KITTI 2011_09_30_drive_0018_sync OXTS",
        "license": "CC BY-NC-SA 3.0",
        "scope": SCOPE,
        "source_provenance_sha256": sha(source / "provenance.json"),
        "source_files": {p.relative_to(source).as_posix(): sha(p) for p in files},
        "reference_pose_indices": [0, 2761],
        "sensor_rotation_xyz_deg": [1.0, -2.0, 3.0],
        "sensor_translation_m": [0.8, -0.3, 0.9],
        "wheel_proxy_columns": {"speed_mps": 8, "yaw_rate_rps": 22},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(problem, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"motions": len(problem["motions"]), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
