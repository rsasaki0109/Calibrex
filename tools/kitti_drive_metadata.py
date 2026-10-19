#!/usr/bin/env python3
"""Metadata of remote KITTI raw drive archives: the zip's file list only, no data.

An evaluation harness, not part of the Calibrex package. For every drive it reads the
archive's size and S3 ETag with an HTTP HEAD request, fetches only the zip's end of
central directory and central directory with HTTP range requests, and presents only
those bytes to ``zipfile`` through a seekable view. No image, point cloud or OXTS
record is downloaded or decoded. It reports the number of frames per stream, which is
how the evaluation drives of the KITTI camera-lidar pre-registration are chosen before
any estimator reads them::

    python tools/kitti_drive_metadata.py OUT.json 0002 0011 ...
    python tools/kitti_drive_metadata.py OUT.json --date 2011_09_30 0016 0020 ...
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.kitti_zip_metadata import read_central_directory

S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
DATE = "2011_09_26"
STREAMS = ("image_00", "image_01", "image_02", "image_03", "velodyne_points", "oxts")


def fetch(url: str, start: int, end: int) -> bytes:
    return subprocess.run(
        ["curl", "-sSf", "-m", "300", "-r", f"{start}-{end}", url],
        check=True,
        capture_output=True,
    ).stdout


def head(url: str) -> tuple[int, str]:
    lines = subprocess.run(
        ["curl", "-sSI", "-m", "60", url], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    fields = {}
    for line in lines:
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()
    return int(fields["content-length"]), fields["etag"].strip('"')


def metadata(drive: str, date: str = DATE) -> dict[str, object]:
    name = f"{date}_drive_{drive}"
    url = f"{S3}/{name}/{name}_sync.zip"
    size, etag = head(url)
    infos = read_central_directory(size, lambda lo, hi: fetch(url, lo, hi))
    names = [info.filename for info in infos]
    counts = {
        stream: sum(1 for item in names if f"/{stream}/data/" in item and not item.endswith("/"))
        for stream in STREAMS
    }
    return {"size_bytes": size, "etag": etag, "frames": counts, "entries": len(names)}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    date = DATE
    if "--date" in args:
        index = args.index("--date")
        date = args[index + 1]
        args = args[:index] + args[index + 2 :]
    output, drives = Path(args[0]), args[1:]
    result = {}
    for drive in drives:
        result[drive] = metadata(drive, date)
        print(drive, json.dumps(result[drive]), flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
