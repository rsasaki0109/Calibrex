#!/usr/bin/env python3
"""Generate official-model motion-compensated KITTI-360 Velodyne scans."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from calibrex import __version__
from calibrex.core.io import write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.depth import load_depth_provider
from calibrex.data.kitti360_camera_lidar_problem import (
    motion_compensate_kitti360_scan,
    read_kitti360_poses,
    read_kitti360_velodyne_to_pose_matrix,
)
from calibrex.data.manifest import DatasetManifest, StreamManifest

KITTI360_SCRIPTS_REPOSITORY = (
    "https://github.com/autonomousvision/kitti360Scripts"
)
KITTI360_SCRIPTS_COMMIT = "32f6d64eef27c32b52c542b4e95be8af9c9c6444"


def main(argv: list[str] | None = None) -> int:
    """Compensate provider-selected scans and emit a dataset manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_path", type=Path)
    parser.add_argument("depth_provider", type=Path)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args(argv)

    sequence = args.sequence_path.resolve()
    provider_path = args.depth_provider.resolve()
    calibration = args.calibration_root.resolve()
    poses_path = args.poses.resolve()
    output_directory = args.output_directory.resolve()
    manifest_output = args.manifest_output.resolve()
    provider = load_depth_provider(provider_path)
    poses = read_kitti360_poses(poses_path)
    velodyne_to_pose = read_kitti360_velodyne_to_pose_matrix(calibration)
    output_directory.mkdir(parents=True, exist_ok=True)

    input_paths = [
        provider_path,
        poses_path,
        calibration / "calib_cam_to_velo.txt",
        calibration / "calib_cam_to_pose.txt",
    ]
    output_paths = []
    compensated_frame_count = 0
    for position, observation in enumerate(provider.observations, start=1):
        frame_id = int(observation.frame_id)
        source = (
            sequence
            / "velodyne_points"
            / "data"
            / f"{observation.frame_id}.bin"
        )
        points = np.fromfile(source, dtype=np.float32)
        if points.size % 4:
            raise ValueError(f"invalid KITTI-360 scan shape: {source}")
        points = points.reshape(-1, 4)
        compensated = motion_compensate_kitti360_scan(
            points,
            frame_id=frame_id,
            poses=poses,
            velodyne_to_pose=velodyne_to_pose,
        )
        target = output_directory / source.name
        compensated.tofile(target)
        input_paths.append(source)
        output_paths.append(target)
        adjacent = frame_id + 1 if frame_id in (0, 1) else frame_id - 1
        compensated_frame_count += int(
            frame_id in poses and adjacent in poses
        )
        print(
            f"[{position}/{len(provider.observations)}] {observation.frame_id}",
            flush=True,
        )

    manifest = DatasetManifest(
        name=f"{sequence.name}-motion-compensated-velodyne",
        description=(
            "Provider-selected KITTI-360 Velodyne scans uncurled with the "
            "official azimuth-weighted ego-motion model"
        ),
        time_base="KITTI-360 Velodyne stream timestamps",
        streams={
            "velodyne_points": StreamManifest(
                kind="pointcloud",
                count=len(output_paths),
                path=str(output_directory),
                sensor="Velodyne HDL-64E",
                frame_id="lidar",
                fields=["x", "y", "z", "intensity"],
            )
        },
        provenance={
            "generator": "tools.motion_compensate_kitti360",
            "generator_version": __version__,
            "git_commit": git_commit() or "unknown",
            "command": " ".join(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
            ),
            "reference_repository": KITTI360_SCRIPTS_REPOSITORY,
            "reference_commit": KITTI360_SCRIPTS_COMMIT,
            "algorithm": (
                "official curl: s=0.5*atan2(y,x)/pi; rotation-vector and "
                "translation scaled by s"
            ),
            "input_sha256": _paths_digest(input_paths),
            "output_sha256": _paths_digest(output_paths),
            "selected_frame_count": str(len(output_paths)),
            "ego_motion_applied_frame_count": str(compensated_frame_count),
            "identity_fallback_frame_count": str(
                len(output_paths) - compensated_frame_count
            ),
        },
    )
    write_mapping(manifest_output, manifest.model_dump(mode="json"))
    print(f"manifest={manifest_output}", flush=True)
    return 0


def _paths_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        file_digest = sha256_path(path)
        if file_digest is None:
            raise ValueError(f"required file is unreadable: {path}")
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(file_digest.encode())
        digest.update(b"\n")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
