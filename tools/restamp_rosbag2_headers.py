#!/usr/bin/env python3
"""Copy topics of a rosbag2 with their header stamps replaced by the bag receive time.

A data-prep tool, not part of the Calibrex package. Some recordings stamp each sensor
on its own clock (for example a camera and a Livox LiDAR each counting from their own
boot), so ``calibrex check`` cannot pair images with scans by header stamp. When the
rig is static during the recording, any pairing sees the same geometry, and the bag
receive time is a usable common clock; this tool writes that time into the
``std_msgs/Header`` stamp of the chosen topics. Do not use it on a moving rig: receive
latency differs between sensors and would read as a clock offset.

Only messages whose first field is a ``std_msgs/Header`` (Image, CameraInfo,
PointCloud2, Imu, ...) in little-endian CDR are supported. Payloads are otherwise
copied byte for byte::

    python tools/restamp_rosbag2_headers.py SRC_BAG DST_BAG \\
        --topic /image --topic /camera_info --topic /livox/points
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from calibrex.data.rosbag2 import iter_messages
from calibrex.data.rosbag2_writer import Rosbag2Writer

_LITTLE_ENDIAN_CDR = (b"\x00\x01", b"\x00\x03")


def restamp(payload: bytes, timestamp_ns: int) -> bytes:
    """Return ``payload`` with its leading header stamp set to ``timestamp_ns``."""

    if len(payload) < 12 or payload[:2] not in _LITTLE_ENDIAN_CDR:
        raise ValueError("expected a little-endian CDR message that starts with a header")
    data = bytearray(payload)
    struct.pack_into("<iI", data, 4, timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000)
    return bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="rosbag2 directory, .db3 or .mcap")
    parser.add_argument("output", type=Path, help="rosbag2 directory to write")
    parser.add_argument(
        "--topic", action="append", required=True, help="topic to copy and restamp (repeatable)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    topics = set(args.topic)
    counts: dict[str, int] = {}
    with Rosbag2Writer(
        args.output,
        custom_data={"generator": "tools/restamp_rosbag2_headers.py", "source": str(args.source)},
        overwrite=args.overwrite,
    ) as writer:
        for connection, timestamp_ns, payload in iter_messages(args.source, topics=topics):
            if connection.topic not in counts:
                writer.add_topic(connection.topic, connection.message_type)
                counts[connection.topic] = 0
            writer.write(connection.topic, timestamp_ns, restamp(payload, timestamp_ns))
            counts[connection.topic] += 1
    missing = sorted(topics - set(counts))
    for topic, count in sorted(counts.items()):
        print(f"{topic}: {count} message(s) restamped to the bag receive time")
    if missing:
        print(f"not in the bag: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
