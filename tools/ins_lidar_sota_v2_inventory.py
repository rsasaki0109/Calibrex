#!/usr/bin/env python3
"""Metadata-only inventories for the second INS-LiDAR rotation audit (v2).

KITTI raw: every recording of the eighteen-archive pool that is eligible (600 Velodyne
and OXTS frames listed) and that no INS-LiDAR method has read. Boreas: the first three
recordings after the first audit's three that list every required file. Only HEAD
requests, ZIP central directories and bucket listings are read; no sensor or calibration
object is fetched.

::

    python tools/ins_lidar_sota_v2_inventory.py docs/assets/ins_lidar_sota_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import boreas_ins_lidar_metadata as boreas  # noqa: E402
from tools import ins_lidar_sota_inventory as kitti  # noqa: E402

V1_EVALUATION = ("2011_09_29_0071", "2011_09_30_0020", "2011_09_30_0027")
"""The first audit's KITTI evaluation recordings: spent, and now development data."""
BOREAS_V1_PREFIXES = 3
"""The first audit inventoried the first three Boreas recording prefixes."""
BOREAS_RECORDINGS = 3
KITTI_SELECTION_RULE = (
    "From the eighteen other-day KITTI raw archives inventoried for camera-LiDAR v2, "
    "select every recording ID with at least 600 archived Velodyne frames and 600 "
    "Velodyne/OXTS members in its first-600-index download metadata that no INS-LiDAR "
    "method has read: the eligible recordings minus the first audit's three. Inspect only "
    "live HEAD and ZIP central-directory metadata; retain all of them even if a pin or "
    "budget fails."
)
BOREAS_SELECTION_RULE = (
    "Original Boreas (2020/2021) recording prefixes in lexicographic order, after the "
    "first three (the first audit's): inventory them in order until three list "
    "applanix/gps_post_process.csv, calib/T_applanix_lidar.txt and 600 LiDAR scans; those "
    "three are the evaluation recordings, and incomplete ones met on the way are retained "
    "as ineligible. First 600 lexicographically ordered numeric-stem LiDAR .bin keys per "
    "recording. Names, sizes and ETags only; no sensor or calibration object bodies read."
)
KITTI_POOL = "docs/assets/kitti_camera_lidar_v2/evaluation/runs"


def kitti_inventory(source_root: Path = ROOT) -> kitti.KittiInsLidarInventory:
    return kitti.collect(
        source_root / KITTI_POOL,
        source_root=source_root,
        choose=lambda names: [name for name in names if name not in V1_EVALUATION],
        selection_rule=KITTI_SELECTION_RULE,
        generator="tools.ins_lidar_sota_v2_inventory",
    )


def boreas_inventory(directory: Path) -> boreas.BoreasInsLidarMetadata:
    """Inventory Boreas, keeping every listing response beside the result."""

    responses = directory / "boreas_metadata.responses"
    responses.mkdir()
    count = 0

    def capture(url: str) -> bytes:
        nonlocal count
        body = boreas.fetch_listing(url)
        (responses / f"{count:03d}.xml").write_bytes(body)
        count += 1
        return body

    provenance = boreas.MetadataProvenance(
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_sha256=hashlib.sha256(Path(boreas.__file__).read_bytes()).hexdigest(),
        command=[sys.executable, *sys.argv],
    )
    return boreas.inventory(
        provenance,
        fetch=capture,
        skip=BOREAS_V1_PREFIXES,
        complete=BOREAS_RECORDINGS,
        selection_rule=BOREAS_SELECTION_RULE,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    out: Path = args.output_dir
    if (out / "kitti_metadata.json").exists() or (out / "boreas_metadata.json").exists():
        parser.error("preserve the earlier inventory and choose a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    kitti_result = kitti_inventory()
    (out / "kitti_metadata.json").write_text(
        kitti_result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    boreas_result = boreas_inventory(out)
    (out / "boreas_metadata.json").write_text(
        boreas_result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "kitti": [
                    {"id": r.case_id, "range_bytes": r.selected_range_bytes}
                    for r in kitti_result.selected
                ],
                "boreas": [
                    {
                        "id": r.recording_id,
                        "scans": len(r.lidar),
                        "lidar_bytes": r.selected_lidar_bytes,
                        "missing": r.missing_required,
                    }
                    for r in boreas_result.recordings
                ],
                "sensor_object_download_bytes": 0,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
