#!/usr/bin/env python3
"""Metadata of remote KITTI raw drive archives: the zip's file list only, no data.

An evaluation harness, not part of the Calibrex package. For every drive it reads the
archive's size and S3 ETag with an HTTP HEAD request, fetches only the zip's end of
central directory and central directory with HTTP range requests, writes them into a
sparse local skeleton, and lists it with ``zipfile``. No image, point cloud or OXTS
record is downloaded or decoded. It reports the number of frames per stream, which is
how the evaluation drives of the KITTI camera-lidar pre-registration are chosen before
any estimator reads them::

    python tools/kitti_drive_metadata.py OUT.json 0002 0011 ...
    python tools/kitti_drive_metadata.py OUT.json --date 2011_09_30 0016 0020 ...
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

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
    tail_length = min(size, 1 << 16)
    tail = fetch(url, size - tail_length, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise SystemExit(f"{drive}: no end of central directory in the last {tail_length} bytes")
    directory_size, directory_offset = struct.unpack_from("<II", tail, eocd + 12)
    zip64 = directory_offset == 0xFFFFFFFF or directory_size == 0xFFFFFFFF
    if zip64:  # the zip64 end of central directory record holds the real values
        locator = tail.rfind(b"PK\x06\x07", 0, eocd)
        record_offset = struct.unpack_from("<Q", tail, locator + 8)[0]
        record = fetch(url, record_offset, record_offset + 55)
        directory_size, directory_offset = struct.unpack_from("<QQ", record, 40)
    directory = fetch(url, directory_offset, directory_offset + directory_size - 1)
    with tempfile.TemporaryDirectory() as scratch:
        skeleton = Path(scratch) / "skeleton.zip"
        with skeleton.open("wb") as stream:  # sparse: only the directory and the tail
            stream.seek(directory_offset)
            stream.write(directory)
            stream.seek(size - tail_length)
            stream.write(tail)
        names = zipfile.ZipFile(skeleton).namelist()
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
