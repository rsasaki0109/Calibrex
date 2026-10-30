"""Bounded archive-metadata planning and exact member acquisition for spent drives."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.kitti_drive_fetch import central_directory, extract, fetch_range, head


def index(plan_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    drives = []
    total = 0
    for d in plan["drives"]:
        size, etag = head(d["url"])
        if (size, etag) != (d["size"], d["etag"]):
            raise ValueError("archive changed")
        infos = central_directory(d["url"], size, etag)
        members = []
        for name in d["members"]:
            info = next(x for x in infos if x.filename == name)
            end = min(
                size - 1,
                info.header_offset
                + 30
                + len(info.filename.encode())
                + len(info.extra)
                + info.compress_size
                + 4096,
            )
            members.append(
                {
                    "name": name,
                    "offset": info.header_offset,
                    "end": end,
                    "compressed": info.compress_size,
                    "bytes": info.file_size,
                    "crc32": info.CRC,
                    "compression": info.compress_type,
                }
            )
        count = sum(m["end"] - m["offset"] + 1 for m in members)
        if count > plan["per_drive_cap_bytes"]:
            raise ValueError("drive cap exceeded")
        total += count
        drives.append(dict(d, members=members, requested_bytes=count))
        print(d["drive"], count, flush=True)
    if total > plan["total_cap_bytes"]:
        raise ValueError("total cap exceeded")
    output.write_text(
        json.dumps(
            {
                "base_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "total_requested_bytes": total,
                "drives": drives,
            },
            indent=2,
        )
        + "\n"
    )


def acquire(plan_path: Path, workspace: Path, output: Path) -> None:
    if workspace.exists() or output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for p, sha in plan["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError("changed source " + p)
    metadata = Path(plan["index"])
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != plan["index_sha256"]:
        raise ValueError("changed index")
    spec = json.loads(metadata.read_text())
    workspace.mkdir(parents=True)
    records = []
    for d in spec["drives"]:
        if head(d["url"]) != (d["size"], d["etag"]):
            raise ValueError("archive changed")

        def fetch(m, d=d):
            info = zipfile.ZipInfo(m["name"])
            info.header_offset = m["offset"]
            info.compress_size = m["compressed"]
            info.file_size = m["bytes"]
            info.CRC = m["crc32"]
            info.compress_type = m["compression"]
            payload = extract(
                fetch_range(d["url"], m["offset"], m["end"], d["etag"]), m["offset"], info
            )
            path = workspace / m["name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return {
                "drive": d["drive"],
                "member": m["name"],
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        # Fixed independent member reads; archive metadata and total byte limit already frozen.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(fetch, d["members"]):
                records.append(r)
                output.write_text(
                    json.dumps(
                        {
                            "scope": "72 adjacent images on6 spent drives",
                            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                            "members": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        print(d["drive"], "acquired", len(d["members"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["index", "acquire"])
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workspace", type=Path)
    a = p.parse_args()
    if a.stage == "index":
        index(a.plan, a.output)
    else:
        acquire(a.plan, a.workspace, a.output)
