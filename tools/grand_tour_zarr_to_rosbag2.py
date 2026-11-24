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


def convert(
    mission_data: Path, groups: list[str], output: Path, max_seconds: float | None
) -> dict[str, Any]:
    import zarr

    root = zarr.open_group(str(mission_data), mode="r")
    summary: dict[str, Any] = {"mission_data": str(mission_data), "groups": {}}
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
                    payload = encode_pointcloud2(
                        frame_id=frame_id, timestamp_ns=stamp_ns, fields=FIELDS,
                        point_step=POINT_DTYPE.itemsize,
                        data=pack_scan(points[row, :count].astype(np.float64), times[row, :count]),
                    )  # fmt: skip
                    writer.write(topic, stamp_ns, payload)
                    written += 1
            summary["groups"][name] = {
                "topic": topic, "frame_id": frame_id, "scans": written,
                "first_stamp_s": float(stamps[0]), "last_stamp_s": float(stamps[limit - 1]),
                "description": str(group.attrs.get("description", "")),
                "T_box_lidar": transform.tolist(),
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
    args = parser.parse_args(argv)
    summary = convert(args.mission_data, args.groups, args.output, args.max_seconds)
    print(json.dumps({name: item["scans"] for name, item in summary["groups"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
