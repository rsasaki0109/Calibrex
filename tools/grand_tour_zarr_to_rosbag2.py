#!/usr/bin/env python3
"""Convert GrandTour LiDAR Zarr groups to a ROS-free rosbag2 for ``calibrex lidar-lidar``.

An evaluation harness, not part of the Calibrex package (it needs ``zarr<3``, for example
``uv run --with "zarr<3" python tools/grand_tour_zarr_to_rosbag2.py ...``).

The GrandTour dataset (ETH Zurich RSL, Hugging Face ``leggedrobotics/grand_tour_dataset``,
MIT license) stores every topic of a mission as a Zarr group. A LiDAR group holds
``timestamp`` (scan header, s), ``valid`` (points per scan), ``points`` (float32 xyz) and
``time`` (per-point capture time; seconds for the Hesai XT-32, nanoseconds as float64 for
the Livox MID360). Each scan becomes one ``sensor_msgs/msg/PointCloud2`` with fields
``x y z`` (float32) and ``time`` (float64, kept in the dataset's absolute encoding), and
is written at its header time.

The groups' ``transform`` attribute (``T_box_base_lidar``, the Boxi calibration) is also
written as ``T_box_<group>.yaml`` for ``--body-transforms``. It is the dataset's
calibration, a reference, not ground truth.

    uv run --with "zarr<3" python tools/grand_tour_zarr_to_rosbag2.py \\
        --mission-data WORK/data --groups hesai_points livox_points --output WORK/bag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibrex.data.ros_cdr_writer import encode_pointcloud2
from calibrex.data.rosbag2_writer import Rosbag2Writer

FLOAT32 = 7
FLOAT64 = 8
FIELDS = (
    ("x", 0, FLOAT32, 1),
    ("y", 4, FLOAT32, 1),
    ("z", 8, FLOAT32, 1),
    ("time", 16, FLOAT64, 1),
)
POINT_DTYPE = np.dtype(
    {"names": ["x", "y", "z", "time"], "formats": ["<f4", "<f4", "<f4", "<f8"],
     "offsets": [0, 4, 8, 16], "itemsize": 24}
)  # fmt: skip


def transform_matrix(transform: dict[str, Any]) -> np.ndarray:
    """4x4 ``T_base_child`` from a GrandTour ``transform`` attribute."""

    rotation = transform["rotation"]
    translation = transform["translation"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        [rotation["x"], rotation["y"], rotation["z"], rotation["w"]]
    ).as_matrix()
    matrix[:3, 3] = [translation["x"], translation["y"], translation["z"]]
    return matrix


def pack_scan(points: np.ndarray, times: np.ndarray) -> bytes:
    """Pack finite, non-origin points with their per-point times."""

    keep = np.isfinite(points).all(axis=1) & np.any(points != 0.0, axis=1) & np.isfinite(times)
    packed = np.zeros(int(keep.sum()), dtype=POINT_DTYPE)
    packed["x"], packed["y"], packed["z"] = points[keep, 0], points[keep, 1], points[keep, 2]
    packed["time"] = times[keep]
    return packed.tobytes()


def thin_scan(
    points: np.ndarray,
    times: np.ndarray,
    max_points: int,
    max_range_m: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Emulate a sparser LiDAR: crop by range, then keep a random ``max_points`` subset.

    The kept points stay in capture order. A Unitree L1 measures about 21,600 points per
    second (about 2,000 per 10 Hz scan) and reaches 10-15 m on dark surfaces.
    """

    keep = np.isfinite(points).all(axis=1)
    if max_range_m is not None:
        keep &= np.linalg.norm(np.where(np.isfinite(points), points, 0.0), axis=1) <= max_range_m
    indices = np.flatnonzero(keep)
    if len(indices) > max_points:
        indices = np.sort(rng.choice(indices, max_points, replace=False))
    return points[indices], times[indices]


def convert(
    mission_data: Path,
    groups: list[str],
    output: Path,
    max_seconds: float | None,
    *,
    sparse: dict[str, tuple[int, float | None]] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    import zarr

    root = zarr.open_group(str(mission_data), mode="r")
    summary: dict[str, Any] = {"mission_data": str(mission_data), "groups": {}}
    sparse = sparse or {}
    rng = np.random.default_rng(seed)
    with Rosbag2Writer(output) as writer:
        for name in groups:
            group = root[name]
            topic = f"/{name}"
            writer.add_topic(topic, "sensor_msgs/msg/PointCloud2")
            frame_id = str(group.attrs["frame_id"])
            transform = transform_matrix(group.attrs["transform"])
            (output.parent / f"T_box_{name}.yaml").write_text(
                "T: " + json.dumps(transform.tolist()) + "\n", encoding="utf-8"
            )
            stamps = group["timestamp"][:]
            valid = group["valid"][:, 0]
            limit = len(stamps)
            if max_seconds is not None:
                limit = int(np.searchsorted(stamps, stamps[0] + max_seconds, side="right"))
            chunk = group["points"].chunks[0]
            written = 0
            for start in range(0, limit, chunk):
                stop = min(start + chunk, limit)
                points = group["points"][start:stop]
                times = group["time"][start:stop]
                for row in range(stop - start):
                    index = start + row
                    count = int(valid[index])
                    stamp_ns = round(float(stamps[index]) * 1.0e9)
                    scan_points = points[row, :count].astype(np.float64)
                    scan_times = times[row, :count]
                    if name in sparse:
                        scan_points, scan_times = thin_scan(
                            scan_points, scan_times, sparse[name][0], sparse[name][1], rng
                        )
                    payload = encode_pointcloud2(
                        frame_id=frame_id, timestamp_ns=stamp_ns, fields=FIELDS,
                        point_step=POINT_DTYPE.itemsize, data=pack_scan(scan_points, scan_times),
                    )  # fmt: skip
                    writer.write(topic, stamp_ns, payload)
                    written += 1
            summary["groups"][name] = {
                "topic": topic, "frame_id": frame_id, "scans": written,
                "first_stamp_s": float(stamps[0]), "last_stamp_s": float(stamps[limit - 1]),
                "description": str(group.attrs.get("description", "")),
                "T_box_lidar": transform.tolist(),
                **({"emulated_sparse": {"max_points": sparse[name][0],
                                        "max_range_m": sparse[name][1], "seed": seed}}
                   if name in sparse else {}),
            }  # fmt: skip
    (output.parent / "conversion.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mission-data", type=Path, required=True, help="a mission's data/ Zarr root"
    )
    parser.add_argument("--groups", nargs="+", default=["hesai_points", "livox_points"])
    parser.add_argument("--output", type=Path, required=True, help="rosbag2 directory to create")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--sparse",
        nargs=3,
        action="append",
        default=[],
        metavar=("GROUP", "MAX_POINTS", "MAX_RANGE_M"),
        help="emulate a sparser LiDAR on GROUP (MAX_RANGE_M 'none' keeps every range)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed of --sparse")
    args = parser.parse_args(argv)
    sparse = {
        group: (int(points), None if limit == "none" else float(limit))
        for group, points, limit in args.sparse
    }
    summary = convert(
        args.mission_data,
        args.groups,
        args.output,
        args.max_seconds,
        sparse=sparse,
        seed=args.seed,
    )
    print(json.dumps({name: item["scans"] for name, item in summary["groups"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
