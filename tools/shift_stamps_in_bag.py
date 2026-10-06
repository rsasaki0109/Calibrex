#!/usr/bin/env python3
"""Copy a rosbag2 with the header stamps of some topics shifted (clock-offset known-bad control).

    python tools/shift_stamps_in_bag.py SRC DST --topic /imu --shift-ms 5 [--max-duration-s 130]

Adds --shift-ms to the std_msgs/Header stamp of every message on each --topic; all other bytes of
every message are copied unchanged. The bag's log time is left alone unless --shift-log-time is
given (a clock-offset change moves the sensor's stamps, not the recorder's receive time).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrex.data.rosbag2_time_shift import shift_stamps_in_bag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--topic", action="append", required=True)
    parser.add_argument("--shift-ms", type=float, required=True)
    parser.add_argument("--shift-log-time", action="store_true")
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = shift_stamps_in_bag(
        args.source,
        args.destination,
        args.shift_ms * 1e-3,
        topics=args.topic,
        shift_log_time=args.shift_log_time,
        max_duration_s=args.max_duration_s,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
