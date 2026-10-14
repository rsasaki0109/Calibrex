#!/usr/bin/env python3
"""Fetch the head of a KITTI raw drive: the members camera-lidar reads, by HTTP range.

An evaluation harness, not part of the Calibrex package. ``calibrex check`` takes at most 48
scans 1 s apart from the start of a recording, so it reads a KITTI drive's first 48 s or
so; whole archives run to 21 GB. This tool reads the archive's central directory by HTTP
range requests (as ``tools/kitti_drive_metadata.py`` does), selects the members of the
first ``--max-frames`` frames of ``image_02``, ``velodyne_points`` and ``oxts`` with their
timestamp and format files, fetches them in as few contiguous ranges as possible, checks
every member's CRC-32 against the central directory, and writes them where unzipping the
archive would. Every range request carries ``If-Match`` with the ETag of the first HEAD
request, so all bytes come from that one object (S3 answers 412 to any other), and with
``--size``/``--etag`` (a pre-registration's pin) a different archive is not fetched at
all. The timestamp files are cut to ``--max-frames`` lines, so the result reads as a drive
of that length. A drive shorter than ``--max-frames`` is fetched whole (those members).
``fetch.json`` records the archive's size and ETag, every range, the number of members
and the SHA-256 of their list (name, size, CRC-32); ``--members`` writes the list::

    python tools/kitti_drive_fetch.py --date 2011_09_30 --drive 0028 --max-frames 600 \\
        --output-dir WORK
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
FRAME_STREAMS = ("image_02", "velodyne_points", "oxts")
STREAM_FILES = (
    "image_02/timestamps.txt",
    "velodyne_points/timestamps.txt",
    "velodyne_points/timestamps_start.txt",
    "velodyne_points/timestamps_end.txt",
    "oxts/timestamps.txt",
    "oxts/dataformat.txt",
)
TIMESTAMP_FILES = (
    "image_02/timestamps.txt",
    "velodyne_points/timestamps.txt",
    "velodyne_points/timestamps_start.txt",
    "velodyne_points/timestamps_end.txt",
    "oxts/timestamps.txt",
)
MAX_GAP_BYTES = 1 << 20  # members closer than this are fetched in one range
WORKERS = 8  # ranges fetched at once


def _curl(args: list[str]) -> bytes:
    result = subprocess.run(
        ["curl", "-sSf", "--retry", "3", "-m", "900", *args], capture_output=True, check=False
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise OSError(f"curl exit {result.returncode}: {message}")
    return result.stdout


def head(url: str) -> tuple[int, str]:
    lines = _curl(["-I", url]).decode("utf-8", "replace").splitlines()
    fields = {}
    for line in lines:
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()
    return int(fields["content-length"]), fields["etag"].strip('"')


def fetch_range(url: str, start: int, end: int, etag: str) -> bytes:
    """Bytes ``start`` to ``end`` of the object whose ETag is ``etag`` (412 from any other)."""

    data = _curl(["-H", f'If-Match: "{etag}"', "-r", f"{start}-{end}", url])
    if len(data) != end - start + 1:
        raise OSError(f"range {start}-{end}: got {len(data)} bytes")
    return data


def central_directory(url: str, size: int, etag: str) -> list[zipfile.ZipInfo]:
    """The archive's members, read from its central directory only."""

    tail_length = min(size, 1 << 16)
    tail = fetch_range(url, size - tail_length, size - 1, etag)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("no end of central directory")
    directory_size, directory_offset = struct.unpack_from("<II", tail, eocd + 12)
    if directory_offset == 0xFFFFFFFF or directory_size == 0xFFFFFFFF:
        locator = tail.rfind(b"PK\x06\x07", 0, eocd)
        if locator < 0:
            raise ValueError("no zip64 end of central directory locator")
        record_offset = struct.unpack_from("<Q", tail, locator + 8)[0]
        record = fetch_range(url, record_offset, record_offset + 55, etag)
        directory_size, directory_offset = struct.unpack_from("<QQ", record, 40)
    directory = fetch_range(url, directory_offset, directory_offset + directory_size - 1, etag)
    with tempfile.TemporaryDirectory() as scratch:
        skeleton = Path(scratch) / "skeleton.zip"
        with skeleton.open("wb") as stream:  # sparse: only the directory and the tail
            stream.seek(directory_offset)
            stream.write(directory)
            stream.seek(size - tail_length)
            stream.write(tail)
        infos = zipfile.ZipFile(skeleton).infolist()
    return sorted(infos, key=lambda info: info.header_offset)


def frame_index(name: str) -> int | None:
    stem = Path(name).stem
    return int(stem) if stem.isdigit() else None


def select(infos: list[zipfile.ZipInfo], prefix: str, max_frames: int) -> list[zipfile.ZipInfo]:
    """The members of the first ``max_frames`` frames and the streams' text files."""

    chosen = []
    for info in infos:
        if not info.filename.startswith(prefix) or info.is_dir():
            continue
        relative = info.filename[len(prefix) :]
        if relative in STREAM_FILES:
            chosen.append(info)
            continue
        stream, _, rest = relative.partition("/")
        if stream in FRAME_STREAMS and rest.startswith("data/"):
            index = frame_index(rest)
            if index is not None and index < max_frames:
                chosen.append(info)
    return chosen


def spans(
    infos: list[zipfile.ZipInfo], chosen: list[zipfile.ZipInfo], size: int
) -> list[tuple[int, int]]:
    """Byte spans of the chosen members, each up to the next member's header, merged."""

    next_offset = {
        info.header_offset: following.header_offset for info, following in itertools.pairwise(infos)
    }
    ranges: list[tuple[int, int]] = []
    for info in chosen:
        start = info.header_offset
        end = next_offset.get(start, size) - 1
        if ranges and start - ranges[-1][1] <= MAX_GAP_BYTES:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def extract(blob: bytes, base: int, info: zipfile.ZipInfo) -> bytes:
    """A member's bytes from a fetched range starting at archive offset ``base``."""

    offset = info.header_offset - base
    if blob[offset : offset + 4] != b"PK\x03\x04":
        raise ValueError(f"{info.filename}: no local header at {info.header_offset}")
    name_length, extra_length = struct.unpack_from("<HH", blob, offset + 26)
    start = offset + 30 + name_length + extra_length
    data = blob[start : start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        content = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(data, -15)
    else:
        raise ValueError(f"{info.filename}: compression {info.compress_type} not supported")
    if zlib.crc32(content) & 0xFFFFFFFF != info.CRC or len(content) != info.file_size:
        raise ValueError(f"{info.filename}: CRC or size does not match the central directory")
    return content


def drive_url(date: str, drive: str) -> str:
    name = f"{date}_drive_{drive}"
    return f"{S3}/{name}/{name}_sync.zip"


def members_sha256(members: list[dict[str, Any]]) -> str:
    """SHA-256 of the members' list: ``[name, bytes, crc32]`` per member, as compact JSON."""

    rows = [[m["name"], m["bytes"], m["crc32"]] for m in sorted(members, key=lambda m: m["name"])]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def fetch_drive(
    date: str,
    drive: str,
    max_frames: int,
    output_dir: Path,
    expected: tuple[int, str] | None = None,
) -> dict[str, Any]:
    """Fetch the drive's head; ``expected`` is a pinned (size, ETag), checked before any range."""

    name = f"{date}_drive_{drive}"
    url = drive_url(date, drive)
    started = time.monotonic()
    size, etag = head(url)
    if expected is not None and (size, etag) != tuple(expected):
        raise ValueError(
            f"size {size} and ETag {etag} differ from the pin {expected[0]} {expected[1]}"
        )
    infos = central_directory(url, size, etag)
    prefix = f"{date}/{name}_sync/"
    chosen = select(infos, prefix, max_frames)
    ranges = spans(infos, chosen, size)
    by_range: dict[int, list[zipfile.ZipInfo]] = {}
    for info in chosen:
        index = next(i for i, (lo, hi) in enumerate(ranges) if lo <= info.header_offset <= hi)
        by_range.setdefault(index, []).append(info)
    members = []
    fetched = 0

    def get(item: tuple[int, tuple[int, int]]) -> tuple[int, bytes]:
        index, (lo, hi) = item
        return index, fetch_range(url, lo, hi, etag)

    with ThreadPoolExecutor(WORKERS) as pool:
        blobs = pool.map(get, enumerate(ranges))
    for index, blob in blobs:
        lo = ranges[index][0]
        fetched += len(blob)
        for info in by_range.get(index, []):
            content = extract(blob, lo, info)
            relative = info.filename[len(prefix) :]
            if relative in TIMESTAMP_FILES:
                lines = content.decode("utf-8").splitlines(keepends=True)
                content = "".join(lines[:max_frames]).encode("utf-8")
            target = output_dir / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            members.append({"name": info.filename, "bytes": info.file_size, "crc32": info.CRC})
    frames = {
        stream: sum(1 for m in members if f"/{stream}/data/" in m["name"])
        for stream in FRAME_STREAMS
    }
    return {
        "url": url,
        "size_bytes": size,
        "etag": etag,
        "max_frames": max_frames,
        "frames": frames,
        "archive_frames": {
            stream: sum(
                1
                for info in infos
                if info.filename.startswith(f"{prefix}{stream}/data/") and not info.is_dir()
            )
            for stream in FRAME_STREAMS
        },
        "ranges": [[lo, hi] for lo, hi in ranges],
        "fetched_bytes": fetched,
        "member_count": len(members),
        "members_sha256": members_sha256(members),
        "members": members,
        "crc_verified": True,
        "wall_s": round(time.monotonic() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--drive", required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=None, help="the pinned archive size")
    parser.add_argument("--etag", default=None, help="the pinned archive ETag")
    parser.add_argument("--record", type=Path, default=None, help="default OUTPUT_DIR/fetch.json")
    parser.add_argument("--members", type=Path, default=None, help="write the members' list here")
    args = parser.parse_args(argv)
    if (args.size is None) != (args.etag is None):
        parser.error("--size and --etag go together")
    expected = None if args.size is None else (args.size, args.etag)
    record = fetch_drive(args.date, args.drive, args.max_frames, args.output_dir, expected)
    members = record.pop("members")
    if args.members is not None:
        args.members.write_text(json.dumps(members, indent=1) + "\n", encoding="utf-8")
    path = args.record or args.output_dir / "fetch.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.date} {args.drive}: {record['frames']} of {record['archive_frames']}, "
        f"{record['fetched_bytes'] / 1e6:.0f} MB in {len(record['ranges'])} range(s), "
        f"{record['wall_s']} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
