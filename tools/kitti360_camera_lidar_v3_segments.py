#!/usr/bin/env python3
"""Metadata-only segment rule for the KITTI-360 camera-LiDAR v3 check.

Each sequence is cut into disjoint 60 s segments by a fixed rule on archive member counts
alone (``docs/assets/kitti360_sota_readiness/paired-frame-counts.json``). Starting at
the first frame ID common to the rectified image_00 and Velodyne archives, a segment is
600 consecutive IDs (60 s at 10 Hz; ``calibrex check`` reads at most 48 scans 1 s apart),
and the next segment starts 1200 IDs later, leaving a 60 s gap. A sequence contributes
every segment that fits inside its common range. Spent sequences give development
segments by the same rule. No image, point cloud, OXTS record or timestamp is read.

    python tools/kitti360_camera_lidar_v3_segments.py COUNTS.json OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calibrex.kitti360_camera_lidar_v3_segments/v0.1"
SEGMENT_FRAMES = 600
STRIDE_FRAMES = 1200
SPENT = ("0000", "0002", "0003")
CANDIDATES = ("0004", "0005", "0006", "0007", "0009", "0010")


def segments(record: dict[str, Any]) -> list[dict[str, int]]:
    """Every rule segment of one sequence; the common range must be gap-free by count."""

    first = record["common_first"]
    count = record["common_count"]
    for stream in ("image", "lidar"):
        span = record[f"{stream}_last"] - record[f"{stream}_first"] + 1
        if span != record[f"{stream}_count"]:
            raise ValueError(f"{record['sequence']}: {stream} IDs are not contiguous")
    last_common = min(record["image_last"], record["lidar_last"])
    if last_common - first + 1 != count:
        raise ValueError(f"{record['sequence']}: common range and count disagree")
    if count < SEGMENT_FRAMES:
        return []
    return [
        {"start_frame": first + k * STRIDE_FRAMES, "frames": SEGMENT_FRAMES}
        for k in range((count - SEGMENT_FRAMES) // STRIDE_FRAMES + 1)
    ]


def build(counts: dict[str, Any]) -> dict[str, Any]:
    by_sequence = {r["sequence"]: r for r in counts["sequences"]}
    if counts["provenance"].get("sensor_payload_bytes") != 0:
        raise ValueError("frame counts must come from metadata only")
    result: dict[str, Any] = {"evaluation": [], "development": [], "not_counted": []}
    for sequence, record in sorted(by_sequence.items()):
        role = (
            "evaluation" if sequence in CANDIDATES else "development" if sequence in SPENT else None
        )
        if role is None:
            result["not_counted"].append(sequence)
            continue
        for index, segment in enumerate(segments(record)):
            result[role].append({"id": f"{sequence}-s{index:02d}", "sequence": sequence, **segment})
    missing = sorted(set(CANDIDATES) - set(by_sequence))
    if missing:
        raise ValueError(f"no frame counts for candidate sequences {missing}")
    result["spent_without_counts"] = sorted(set(SPENT) - set(by_sequence))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("counts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse overwrite: the segment list is fixed once written")
    counts = json.loads(args.counts.read_text())
    result = {
        "schema_version": SCHEMA,
        "rule": {
            "segment_frames": SEGMENT_FRAMES,
            "stride_frames": STRIDE_FRAMES,
            "start": "first frame ID common to the image_00 and Velodyne archives",
            "candidates": list(CANDIDATES),
            "spent": list(SPENT),
        },
        **build(counts),
        "sensor_payload_bytes_read": 0,
        "provenance": {
            "generator": "tools/kitti360_camera_lidar_v3_segments.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "counts_sha256": hashlib.sha256(args.counts.read_bytes()).hexdigest(),
            "git_commit": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(len(result["evaluation"]), "evaluation and", len(result["development"]), "development")
    return 0


if __name__ == "__main__":
    sys.exit(main())
