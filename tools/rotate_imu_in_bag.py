#!/usr/bin/env python3
"""Copy a rosbag2 with its IMU "remounted" by a known rotation (drift known-bad control).

    python tools/rotate_imu_in_bag.py SRC DST --axis 0 0 1 --angle-deg 1.0 [--max-duration-s 130]

Every angular_velocity and linear_acceleration vector of every Imu message is
rotated by R (Rodrigues about --axis); all other messages are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrex.data.rosbag2_imu_rotate import rotate_imu_in_bag, rotation_from_axis_angle_deg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--axis", type=float, nargs=3, default=[0.0, 0.0, 1.0], metavar="X")
    parser.add_argument("--angle-deg", type=float, required=True)
    parser.add_argument("--imu-topic", action="append", default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = rotate_imu_in_bag(
        args.source,
        args.destination,
        rotation_from_axis_angle_deg(args.axis, args.angle_deg),
        imu_topics=args.imu_topic,
        max_duration_s=args.max_duration_s,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
