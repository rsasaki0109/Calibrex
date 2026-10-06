"""Convert raw ``velodyne_msgs/msg/VelodyneScan`` topics of a rosbag2 into PointCloud2 topics.

Autoware sample bags carry per-sensor Velodyne packets (sensor frames, per-firing time)
next to a concatenated cloud already transformed into ``base_link``. ``calibrex check``
cannot test the physical mounting from the transformed cloud; this writes a derived bag
whose clouds are in the sensor frames (Autoware axes: x forward, y left; VLP-16 and VLP-32C
verified against an Autoware concatenated cloud, dual return and other models are
experimental/unverified)::

    python tools/velodyne_scan_to_pointcloud2.py BAG OUT_BAG \
        --drop /sensing/lidar/concatenated/pointcloud
    calibrex check OUT_BAG --vehicle-frame base_link
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calibrex.data.rosbag2_velodyne import convert_velodyne_bag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Axes are Autoware's (x forward, y left). VLP-16 and VLP-32C single return are "
            "verified against an Autoware concatenated cloud; dual return and other models "
            "are experimental and unverified."
        ),
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--drop", action="append", default=[], metavar="TOPIC")
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--max-duration-s", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = convert_velodyne_bag(
        args.bag,
        args.output,
        drop_topics=args.drop,
        keep_images=args.keep_images,
        max_duration_s=args.max_duration_s,
        overwrite=args.overwrite,
    )
    for packets, points in summary.cloud_topics.items():
        print(f"{packets} -> {points}")
    print(
        f"wrote {summary.destination}: "
        f"{summary.clouds_written} clouds, {summary.messages_copied} messages copied, "
        f"{summary.messages_dropped} dropped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
