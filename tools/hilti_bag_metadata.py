#!/usr/bin/env python3
"""Metadata of remote Hilti 2022 ROS 1 bags: the bag header and the index only.

An evaluation harness, not part of the Calibrex package. No message data is
downloaded: bytes 0-8191 (the bag header record) and ``index_pos`` to the end (the
connection and chunk-info records) are fetched with HTTP range requests and written
into a sparse local skeleton file, which the Calibrex index reader parses. It reports
each bag's size, duration (first chunk start to last chunk end) and per-topic message
counts. It is how the evaluation recordings of
``docs/benchmarks/hilti_camera_sota_preregistration.yaml`` were chosen before any
estimator read them::

    python tools/hilti_bag_metadata.py OUT_DIR exp05_construction_upper_level_2.bag ...
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

from calibrex.data import rosbag1
from calibrex.data.bag_formats import _read_index

BASE = (
    "https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2022/resolve/main/rosbags/"
)


def fetch(url: str, start: int, end: int | None) -> bytes:
    span = f"{start}-" if end is None else f"{start}-{end}"
    return subprocess.run(
        ["curl", "-sSL", "-m", "300", "-r", span, url], check=True, capture_output=True
    ).stdout


def content_length(url: str) -> int:
    headers = subprocess.run(["curl", "-sSLI", url], capture_output=True, text=True).stdout
    return int(headers.lower().split("content-length:")[-1].split()[0])


def metadata(name: str, workdir: Path) -> dict[str, object]:
    url = BASE + name
    head = fetch(url, 0, 8191)
    if not head.startswith(rosbag1.BAG_MAGIC):
        raise SystemExit(f"{name} is not a ROS 1 bag")
    offset = len(rosbag1.BAG_MAGIC)
    length = struct.unpack_from("<I", head, offset)[0]
    fields = rosbag1._read_header_fields(head[offset + 4 : offset + 4 + length])
    index_pos = rosbag1._uint64(fields.get("index_pos"))
    if not index_pos:
        raise SystemExit(f"{name} has no index")
    tail = fetch(url, index_pos, None)
    skeleton = workdir / f"{name}.skeleton"
    with skeleton.open("wb") as stream:  # sparse: only the header and the index are written
        stream.write(head)
        stream.seek(index_pos)
        stream.write(tail)
    index = _read_index(skeleton)
    skeleton.unlink()
    if index is None:
        raise SystemExit(f"{name}: the index could not be read")
    starts, ends, position = [], [], 0
    while position + 4 <= len(tail):
        header_length = struct.unpack_from("<I", tail, position)[0]
        position += 4
        record = rosbag1._read_header_fields(tail[position : position + header_length])
        position += header_length
        position += 4 + struct.unpack_from("<I", tail, position)[0]
        if rosbag1._record_op(record) == rosbag1.OP_CHUNK_INFO:
            starts.append(rosbag1._time_field_to_ns(record.get("start_time")) or 0)
            ends.append(rosbag1._time_field_to_ns(record.get("end_time")) or 0)
    counts: dict[str, int] = {}
    for conn_id, connection in index.connections.items():
        counts[connection.topic] = counts.get(connection.topic, 0) + index.counts.get(conn_id, 0)
    return {
        "size_bytes": content_length(url),
        "index_bytes": len(tail),
        "duration_s": round((max(ends) - min(starts)) * 1e-9, 1),
        "chunks": len(starts),
        "counts": {
            topic: count
            for topic, count in sorted(counts.items())
            if topic.startswith("/alphasense") or "hesai" in topic
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    workdir, names = Path(args[0]), args[1:]
    result = {}
    for name in names:
        result[name] = metadata(name, workdir)
        print(name, json.dumps(result[name]), flush=True)
    (workdir / "candidate_metadata.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
