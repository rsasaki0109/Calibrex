#!/usr/bin/env python3
"""Fix the Argoverse 2 development and evaluation logs of a camera-LiDAR v4 check (metadata only).

The rule uses anonymous S3 listings of ``s3://argoverse/datasets/av2/sensor/`` alone:

* development: the first ``DEVELOPMENT_LOGS`` validation logs by log ID;
* evaluation: every test log;
* reserve, never read by this work: the other validation logs and every train log.

For each chosen log the tool lists the members a ``calibrex convert av2`` run reads
(the two calibration tables, the poses, every LiDAR sweep and every
``ring_front_center`` image) and pins them by the SHA-256 of their sorted
``key, size, ETag`` lines. A log missing any of them is recorded and kept, never
replaced. No feather table, sweep or image is downloaded.

    python tools/av2_camera_lidar_v4_split.py OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calibrex.av2_camera_lidar_v4_split/v0.1"
BUCKET = "https://argoverse.s3.amazonaws.com/"
PREFIX = "datasets/av2/sensor/"
DEVELOPMENT_LOGS = 30
CAMERA = "ring_front_center"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
Lister = Callable[[str, str], list[Any]]


def s3_list(prefix: str, delimiter: str = "/") -> list[Any]:
    """Every common prefix (str) and object ``(key, size, etag)`` under ``prefix``."""

    out: list[Any] = []
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "delimiter": delimiter, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = BUCKET + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        out += [p.findtext("s:Prefix", namespaces=NS) for p in root.findall("s:CommonPrefixes", NS)]
        out += [
            (
                c.findtext("s:Key", namespaces=NS),
                int(c.findtext("s:Size", namespaces=NS) or -1),
                c.findtext("s:ETag", namespaces=NS),
            )
            for c in root.findall("s:Contents", NS)
        ]
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            return out
        token = root.findtext("s:NextContinuationToken", namespaces=NS)


def logs(split: str, lister: Lister = s3_list) -> list[str]:
    """Log IDs of one split, sorted."""

    prefixes = [p for p in lister(f"{PREFIX}{split}/", "/") if isinstance(p, str)]
    return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes)


def rule(val: Sequence[str], test: Sequence[str]) -> dict[str, list[str]]:
    """The fixed split: first validation logs by ID for development, every test log."""

    if len(set(val)) != len(val) or len(set(test)) != len(test) or set(val) & set(test):
        raise ValueError("log IDs must be distinct within and across splits")
    ordered = sorted(val)
    return {
        "development": ordered[:DEVELOPMENT_LOGS],
        "evaluation": sorted(test),
        "reserve_validation": ordered[DEVELOPMENT_LOGS:],
    }


def pin(split: str, log: str, lister: Lister = s3_list) -> dict[str, Any]:
    """The members one conversion reads, pinned by a digest of key, size and ETag."""

    base = f"{PREFIX}{split}/{log}/"
    objects = [o for o in lister(base, "/") if not isinstance(o, str)]
    objects += [o for o in lister(base + "calibration/", "/") if not isinstance(o, str)]
    sweeps = [o for o in lister(base + "sensors/lidar/", "") if not isinstance(o, str)]
    images = [o for o in lister(base + f"sensors/cameras/{CAMERA}/", "") if not isinstance(o, str)]
    wanted = {
        base + "city_SE3_egovehicle.feather",
        base + "calibration/egovehicle_SE3_sensor.feather",
        base + "calibration/intrinsics.feather",
    }
    tables = [o for o in objects if o[0] in wanted]
    sweeps = [o for o in sweeps if o[0].endswith(".feather")]
    images = [o for o in images if o[0].endswith(".jpg")]
    members = sorted(tables + sweeps + images)
    digest = hashlib.sha256()
    for key, size, etag in members:
        digest.update(f"{key}\0{size}\0{etag}\n".encode())
    missing = sorted(wanted - {o[0] for o in tables})
    if not sweeps:
        missing.append(base + "sensors/lidar/*.feather")
    if not images:
        missing.append(base + f"sensors/cameras/{CAMERA}/*.jpg")
    return {
        "log": log,
        "split": split,
        "complete": not missing,
        "missing": missing,
        "sweeps": len(sweeps),
        "images": len(images),
        "bytes": sum(size for _, size, _ in members),
        "members_sha256": digest.hexdigest(),
    }


def build(lister: Lister = s3_list) -> dict[str, Any]:
    split = rule(logs("val", lister), logs("test", lister))
    return {
        "schema_version": SCHEMA,
        "rule": {
            "development": f"first {DEVELOPMENT_LOGS} validation logs by log ID",
            "evaluation": "every test log",
            "reserve": "the other validation logs and every train log; never read",
            "camera": CAMERA,
            "pin": "SHA-256 of sorted 'key NUL size NUL etag' lines of the read members",
        },
        "development": [pin("val", log, lister) for log in split["development"]],
        "evaluation": [pin("test", log, lister) for log in split["evaluation"]],
        "reserve_validation": split["reserve_validation"],
        "train_logs": len(logs("train", lister)),
        "sensor_payload_bytes_read": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse overwrite: the split is fixed once written")
    result = build()
    result["provenance"] = {
        "generator": "tools/av2_camera_lidar_v4_split.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": BUCKET + PREFIX,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    incomplete = [
        p["log"] for p in result["development"] + result["evaluation"] if not p["complete"]
    ]
    print(
        len(result["development"]),
        "development,",
        len(result["evaluation"]),
        "evaluation;",
        "incomplete:",
        incomplete,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
