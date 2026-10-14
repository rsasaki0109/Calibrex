#!/usr/bin/env python3
"""Download a Hilti 2022 bag, or its first bytes, and hash it in the same pass.

An evaluation harness, not part of the Calibrex package. Range requests run in
parallel (the network here is capped per connection); the chunks are written and
hashed strictly in file order, so the SHA-256 is of exactly the bytes on disk and no
second read of the file is needed. A head (``--bytes``) is cut after the last complete
top-level record, so the bag reader reads it to its end; ``bytes`` records the kept
range and ``requested_bytes`` the requested one. ``download.json`` records the URL, the byte range,
the SHA-256 of the downloaded bytes and, for a full download, whether it equals the
SHA-256 that the dataset's LFS pointer publishes.

::

    python tools/hilti_download.py exp05_construction_upper_level_2.bag \\
        --bytes 5000000000 --output exp05_head.bag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPOSITORY = "https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022"
BAG_MAGIC = b"#ROSBAG V2.0\n"


def lfs_entry(name: str) -> dict[str, object]:
    """``size`` and ``sha256`` the dataset's LFS pointer publishes for ``rosbags/name``."""

    listing = subprocess.run(
        [
            "curl",
            "-sSL",
            "-m",
            "60",
            f"{REPOSITORY.replace('/datasets/', '/api/datasets/')}/tree/main/rosbags",
        ],
        capture_output=True,
        check=True,
    ).stdout
    for item in json.loads(listing):
        if item["path"] == f"rosbags/{name}":
            return {"size": int(item["size"]), "sha256": item["lfs"]["oid"]}
    raise SystemExit(f"{name} is not in the dataset")


def fetch(url: str, start: int, end: int) -> bytes:
    """Bytes ``start..end`` inclusive, retried until complete."""

    for attempt in range(6):
        data = subprocess.run(
            ["curl", "-sSL", "-m", "900", "-r", f"{start}-{end}", url],
            capture_output=True,
            check=False,
        ).stdout
        if len(data) == end - start + 1:
            return data
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"bytes {start}-{end} incomplete after retries")


def last_record_end(path: Path, length: int) -> int:
    """End offset of the last top-level ROS bag record that lies wholly in the first
    ``length`` bytes (each record: header length, header, data length, data)."""

    position = len(BAG_MAGIC)
    with path.open("rb") as stream:
        while True:
            stream.seek(position)
            prefix = stream.read(4)
            if len(prefix) < 4:
                return position
            header_length = struct.unpack("<I", prefix)[0]
            stream.seek(position + 4 + header_length)
            data_prefix = stream.read(4)
            if len(data_prefix) < 4:
                return position
            end = position + 8 + header_length + struct.unpack("<I", data_prefix)[0]
            if end > length:
                return position
            position = end


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help="file name under rosbags/")
    parser.add_argument("--bytes", type=int, default=None, help="first N bytes (default: all)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--connections", type=int, default=12)
    parser.add_argument("--chunk-mb", type=int, default=32)
    args = parser.parse_args(argv)

    entry = lfs_entry(args.name)
    size = int(entry["size"])  # type: ignore[call-overload]
    length = size if args.bytes is None else min(args.bytes, size)
    url = f"{REPOSITORY}/resolve/main/rosbags/{args.name}"
    chunk = args.chunk_mb << 20
    starts = list(range(0, length, chunk))
    digest = hashlib.sha256()
    snapshots: dict[int, Any] = {}  # the hash state at every chunk start, to end it anywhere
    started = time.monotonic()
    with ThreadPoolExecutor(args.connections) as pool, args.output.open("wb") as out:
        pending: deque[Future[bytes]] = deque()
        queued = 0
        done = 0
        while done < len(starts):
            while queued < len(starts) and len(pending) < 2 * args.connections:
                first = starts[queued]
                pending.append(pool.submit(fetch, url, first, min(first + chunk, length) - 1))
                queued += 1
            data = pending.popleft().result()
            snapshots[starts[done]] = digest.copy()
            out.write(data)
            digest.update(data)
            done += 1
    kept = length
    if length < size:
        # a head ends inside a record; the bag reader needs whole records, so the file is cut
        # after the last complete top-level record (only record lengths are read to find it)
        kept = last_record_end(args.output, length)
        base = max(offset for offset in snapshots if offset <= kept)
        tail_digest = snapshots[base].copy()
        with args.output.open("rb") as stream:
            stream.seek(base)
            tail_digest.update(stream.read(kept - base))
        with args.output.open("r+b") as stream:
            stream.truncate(kept)
        digest = tail_digest
    sha256 = digest.hexdigest()
    record = {
        "name": args.name,
        "url": url,
        "bytes": [0, kept],
        "requested_bytes": length,
        "full_size": size,
        "full": length == size,
        "sha256": sha256,
        "lfs_sha256": entry["sha256"],
        "matches_lfs": (sha256 == entry["sha256"]) if length == size else None,
        "wall_s": round(time.monotonic() - started, 1),
    }
    args.output.with_name(args.output.name + ".download.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record))
    return 0 if record["matches_lfs"] in (True, None) else 1


if __name__ == "__main__":
    sys.exit(main())
