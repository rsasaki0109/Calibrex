#!/usr/bin/env python3
"""Integrate provenance-locked KITTI raw LiDAR/OXTS windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calibrex.data.kitti_raw_lidar_window_integration import (
    KITTI_RAW_LIDAR_WINDOW_INTEGRATOR_VERSION,
    integrate_kitti_raw_lidar_windows,
    verify_kitti_raw_lidar_window_integration_files,
)

TOOL_VERSION = "calibrex.tools.integrate_kitti_raw_lidar_windows/v0.1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Motion-compensate and integrate centered KITTI raw Velodyne/OXTS "
            "windows into each center Velodyne frame"
        )
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--center-frame-ids", required=True)
    parser.add_argument("--neighbor-radius", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--artifact-id")
    parser.add_argument("--minimum-range-m", type=float, default=1.0)
    parser.add_argument("--voxel-resolution-m", type=float, default=0.002)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    centers = _parse_frame_ids(args.center_frame_ids)
    artifact = integrate_kitti_raw_lidar_windows(
        args.selection_manifest,
        args.calibration,
        centers,
        neighbor_radius_frames=args.neighbor_radius,
        output_directory=args.output_directory,
        artifact_id=args.artifact_id,
        minimum_range_m=args.minimum_range_m,
        voxel_resolution_m=args.voxel_resolution_m,
        generator="tools.integrate_kitti_raw_lidar_windows",
        generator_version=(
            f"{TOOL_VERSION}+{KITTI_RAW_LIDAR_WINDOW_INTEGRATOR_VERSION}"
        ),
        command=sys.argv,
    )
    issues = verify_kitti_raw_lidar_window_integration_files(artifact)
    if issues:
        raise OSError("generated raw integration failed verification: " + ", ".join(issues))
    manifest_path = args.manifest_output.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = manifest_path.with_name(
        f".{manifest_path.name}.calibrex-partial{manifest_path.suffix}"
    )
    partial_path.unlink(missing_ok=True)
    artifact.save(partial_path)
    partial_path.replace(manifest_path)
    print(
        f"saved {len(artifact.windows)} raw integrated windows to "
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
    if normalized != sorted(set(normalized), key=int):
        raise ValueError("center frame IDs must be unique and ordered")
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
