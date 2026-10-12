#!/usr/bin/env python3
"""Cut a partially downloaded ROS 1 bag at its last complete top-level record.

A data-prep helper, not part of the Calibrex package. Public ROS 1 bags are often
larger than a check needs (``calibrex check --max-duration-s`` reads the first minutes
only), so the head of a bag can be fetched with an HTTP range request::

    curl -L -r 0-6442450943 -o exp21_head.bag URL
    python tools/trim_partial_rosbag1.py exp21_head.bag

A range rarely ends on a record boundary. Calibrex's ROS 1 reader scans a bag
sequentially, so a bag cut right after a complete chunk reads as a shorter recording;
a bag cut inside a record raises "unexpected end of ROS bag file". This tool walks the
records and truncates the file after the last complete one, in place.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

MAGIC = b"#ROSBAG V2.0\n"
MAX_HEADER_BYTES = 1 << 20


def last_complete_record_end(path: Path) -> tuple[int, int]:
    """``(byte offset after the last complete record, number of complete records)``."""

    size = path.stat().st_size
    records = 0
    with path.open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} is not a ROS 1 bag (v2.0)")
        end = source.tell()
        while True:
            start = source.tell()
            prefix = source.read(4)
            if len(prefix) < 4:
                break
            (header_length,) = struct.unpack("<I", prefix)
            if header_length == 0 or header_length > MAX_HEADER_BYTES:
                source.seek(start + 1)  # padding between records
                continue
            source.seek(header_length, os.SEEK_CUR)
            data_prefix = source.read(4)
            if len(data_prefix) < 4:
                break
            (data_length,) = struct.unpack("<I", data_prefix)
            record_end = source.tell() + data_length
            if record_end > size:
                break
            source.seek(record_end)
            end = record_end
            records += 1
    return end, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    args = parser.parse_args(argv)
    end, records = last_complete_record_end(args.bag)
    size = args.bag.stat().st_size
    os.truncate(args.bag, end)
    print(f"{records} complete records; kept {end} of {size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
