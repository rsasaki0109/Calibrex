#!/usr/bin/env python3
"""Integrate provenance-locked KITTI-360 LiDAR windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calibrex.data.kitti360_lidar_window_integration import (
    KITTI360_LIDAR_WINDOW_INTEGRATOR_VERSION,
    integrate_kitti360_lidar_windows,
    verify_kitti360_lidar_window_integration_files,
)

TOOL_VERSION = "calibrex.tools.integrate_kitti360_lidar_windows/v0.1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Motion-compensate and integrate centered KITTI-360 Velodyne "
            "windows into each center sensor frame"
        )
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--center-frame-ids", required=True)
    parser.add_argument("--neighbor-radius", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--artifact-id")
    parser.add_argument("--minimum-range-m", type=float, default=1.0)
    parser.add_argument("--voxel-resolution-m", type=float, default=0.002)
    parser.add_argument(
        "--maximum-pose-extrapolation-frames",
        type=int,
        default=0,
        help="bounded endpoint pose extrapolation; zero rejects it",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    centers = _parse_frame_ids(args.center_frame_ids)
    artifact = integrate_kitti360_lidar_windows(
        args.selection_manifest,
        args.poses,
        args.calibration_root,
        centers,
        neighbor_radius_frames=args.neighbor_radius,
        output_directory=args.output_directory,
        artifact_id=args.artifact_id,
        minimum_range_m=args.minimum_range_m,
        voxel_resolution_m=args.voxel_resolution_m,
        maximum_pose_extrapolation_frames=args.maximum_pose_extrapolation_frames,
        generator="tools.integrate_kitti360_lidar_windows",
        generator_version=(f"{TOOL_VERSION}+{KITTI360_LIDAR_WINDOW_INTEGRATOR_VERSION}"),
        command=sys.argv,
    )
    issues = verify_kitti360_lidar_window_integration_files(artifact)
    if issues:
        raise OSError("generated integration failed verification: " + ", ".join(issues))
    manifest_path = args.manifest_output.resolve()
    partial_path = manifest_path.with_name(
        f".{manifest_path.name}.calibrex-partial{manifest_path.suffix}"
    )
    partial_path.unlink(missing_ok=True)
    artifact.save(partial_path)
    partial_path.replace(manifest_path)
    print(
        f"saved {len(artifact.windows)} integrated windows to "
        f"{args.output_directory.resolve()} and {manifest_path}",
        flush=True,
    )
    return 0


def _parse_frame_ids(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("center-frame-ids must contain at least one numeric frame")
    normalized: list[str] = []
    for value in values:
        if not value.isdigit() or int(value) < 0:
            raise ValueError(f"invalid center frame ID: {value!r}")
        normalized.append(f"{int(value):010d}")
    if len(normalized) != len(set(normalized)):
        raise ValueError("center frame IDs must be unique")
    return sorted(normalized, key=int)


if __name__ == "__main__":
    raise SystemExit(main())
