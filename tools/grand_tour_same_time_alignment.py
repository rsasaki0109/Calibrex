#!/usr/bin/env python3
"""Odometry-free check of a LiDAR-LiDAR extrinsic on GrandTour: same-time scan overlap.

An evaluation harness, not part of the Calibrex package (it needs ``zarr<3``).

For each selected target scan, the reference scan nearest in time is taken alone (no
odometry, no map). The target scan is moved into the reference frame by each candidate
``T_reference_target``. The tool reports two numbers per candidate:

- the fraction of target points within 0.5 m of a planar reference point;
- the median absolute point-to-plane distance.

The candidates are a Calibrex artifact's estimate and the dataset's static tf (composed
through ``--tf-chain``). Motion between the two scans' times (about 10 ms here) is not
corrected, so even a perfect extrinsic has a residual of a few millimetres.

    uv run --with "zarr<3" python tools/grand_tour_same_time_alignment.py \\
        --mission-data WORK/data --reference hesai_points --target livox_points \\
        --artifact WORK/runs/rigid_full.yaml --tf-reference hesai_lidar \\
        --tf-target livox_lidar --frames 0 500 1000 1500 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.grand_tour_zarr_to_rosbag2 import transform_matrix


def tf_to_root(tf: dict[str, Any], frame: str) -> np.ndarray:
    """``T_root_frame`` by chaining the static tf's parents up to the root."""

    matrix = np.eye(4)
    while frame in tf:
        entry = tf[frame]
        matrix = transform_matrix(entry) @ matrix
        frame = entry["base_frame_id"]
    return matrix


def artifact_transform(path: Path) -> np.ndarray:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    transform = data["transform"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(transform["rotation_quat_xyzw"]).as_matrix()
    matrix[:3, 3] = transform["translation_m"]
    return matrix


def overlap(
    reference: np.ndarray, target: np.ndarray, transform: np.ndarray
) -> tuple[float, float]:
    """Matched fraction and median |point-to-plane| (m) of ``target`` moved by ``transform``."""

    tree = cKDTree(reference)
    _, neighbours = tree.query(reference, k=10)
    values = reference[neighbours]
    centered = values - values.mean(axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
    normals = eigenvectors[:, :, 0]
    planar = eigenvalues[:, 0] <= 0.2 * eigenvalues[:, 1]
    moved = target @ transform[:3, :3].T + transform[:3, 3]
    distances, indices = tree.query(moved, distance_upper_bound=0.5)
    valid = np.isfinite(distances)
    valid[valid] &= planar[indices[valid]]
    residual = np.abs(
        np.einsum(
            "ij,ij->i", moved[valid] - reference[indices[valid]], normals[indices[valid]]
        )
    )
    return float(valid.mean()), float(np.median(residual)) if len(residual) else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mission-data", type=Path, required=True)
    parser.add_argument("--reference", required=True, help="reference Zarr group")
    parser.add_argument("--target", required=True, help="target Zarr group")
    parser.add_argument("--artifact", type=Path, required=True, help="calibrex lidar-lidar YAML")
    parser.add_argument("--tf-reference", required=True, help="reference frame in the tf group")
    parser.add_argument("--tf-target", required=True, help="target frame in the tf group")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 500, 1000, 1500, 2000])
    parser.add_argument("--min-range", type=float, default=1.0)
    parser.add_argument("--max-range", type=float, default=40.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import zarr

    root = zarr.open_group(str(args.mission_data), mode="r")
    tf = dict(root["tf"].attrs["tf"])
    candidates = {
        "estimate": artifact_transform(args.artifact),
        "dataset_tf": np.linalg.inv(tf_to_root(tf, args.tf_reference))
        @ tf_to_root(tf, args.tf_target),
    }

    def scan(group: str, index: int) -> tuple[np.ndarray, float]:
        count = int(root[group]["valid"][index, 0])
        points = root[group]["points"][index, :count].astype(np.float64)
        ranges = np.linalg.norm(points, axis=1)
        keep = (ranges > args.min_range) & (ranges < args.max_range)
        return points[keep], float(root[group]["timestamp"][index])

    reference_stamps = root[args.reference]["timestamp"][:]
    rows = []
    for index in args.frames:
        target, target_time = scan(args.target, index)
        nearest = int(np.argmin(np.abs(reference_stamps - target_time)))
        reference, reference_time = scan(args.reference, nearest)
        row: dict[str, Any] = {
            "target_index": index,
            "reference_index": nearest,
            "time_gap_s": target_time - reference_time,
        }
        for name, transform in candidates.items():
            matched, median = overlap(reference, target, transform)
            row[name] = {"matched_fraction": matched, "median_point_to_plane_m": median}
        rows.append(row)
        print(
            f"{args.target} {index}: "
            + " | ".join(
                f"{name} matched {row[name]['matched_fraction']:.2f} "
                f"median {1000 * row[name]['median_point_to_plane_m']:.1f} mm"
                for name in candidates
            )
        )
    result = {
        "mission_data": str(args.mission_data),
        "reference": args.reference,
        "target": args.target,
        "artifact": str(args.artifact),
        "tf_reference": args.tf_reference,
        "tf_target": args.tf_target,
        "candidates": {name: matrix.tolist() for name, matrix in candidates.items()},
        "rows": rows,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
