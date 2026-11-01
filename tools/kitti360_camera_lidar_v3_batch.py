#!/usr/bin/env python3
"""Sequential KITTI-360 development batch for the v3 camera-LiDAR check (spent data only).

Runs the fixed development segments of spent sequences 0000 and 0002
(``docs/assets/kitti_camera_lidar_v3/kitti360_prep/segments.json``). It never names an
evaluation candidate: archives and segments of candidate sequences are refused before
any request. For each segment the batch:

1. re-reads each archive's central directory and requires every metadata range to match
   the committed SHA-256 bindings (``inventory``);
2. range-reads exactly the segment's members (600 rectified ``image_00`` PNGs, 600 scans,
   their OXTS packets, the three timestamp files and the calibration) under the pinned
   ETag, checking each member's local header, size and CRC-32;
3. converts with ``motion_source=oxts``;
4. runs the seven real checks through ``kitti_camera_lidar_v3_replay.capture_drive`` and
   replays the default rule against the opt-in cross-start agreement rule (0.5 deg).

Raw members and the bag are then deleted. Captured NPZ inputs stay only while free disk
remains above a reserve; deletions are recorded with digests. Every failure is retained
as its own attempt. Development only: no gate, no claim.

    python tools/kitti360_camera_lidar_v3_batch.py inventory WORK_ROOT
    python tools/kitti360_camera_lidar_v3_batch.py run WORK_ROOT [--segment ID] [--retry-failed]
    python tools/kitti360_camera_lidar_v3_batch.py publish WORK_ROOT DEST
    python tools/kitti360_camera_lidar_v3_batch.py aggregate DEST SUMMARY.json
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import json
import platform
import shutil
import struct
import subprocess
import sys
import time
import traceback
import urllib.error
import zlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.request import Request

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calibrex.kitti360_camera_lidar_v3_batch/v0.1"
BINDINGS = ROOT / "docs/assets/kitti360_sota_readiness/metadata-range-bindings.json"
SEGMENTS = ROOT / "docs/assets/kitti_camera_lidar_v3/kitti360_prep/segments.json"
SEQUENCES = ("0000", "0002")
SHARED_ARCHIVES = ("calibration", "oxts", "perspective-timestamps", "velodyne-timestamps")
AGREEMENT = {"start_agreement_deg": 0.5}
REQUEST_BYTES = 64 * 1024**2
FETCH_CAP_BYTES = 3 * 1024**3
RETRIES = 3
START_FREE_BYTES = 8 * 1024**3
KEEP_ARRAYS_FREE_BYTES = 12 * 1024**3
FROZEN = (
    "tools/kitti360_camera_lidar_v3_batch.py",
    "tools/kitti360_archive_metadata.py",
    "tools/kitti360_camera_lidar_v3_segments.py",
    "tools/kitti_camera_lidar_v3_replay.py",
    "tools/kitti_camera_lidar_v3_development.py",
    "tools/kitti_camera_lidar_sota_run.py",
    "tools/score_kitti_camera_lidar_v2.py",
    "src/calibrex/evaluation/camera_lidar_edge.py",
    "src/calibrex/data/kitti360_to_rosbag2.py",
)
PUBLISHED_SUFFIXES = {".json", ".log", ".yaml", ".txt"}
RETRYABLE = {"FAIL", "RUNNING"}
TRANSIENT = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)


class SegmentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA
    segment: str
    attempt: int
    status: str
    stage: str
    error: str | None = None
    details: dict[str, Any] = {}
    provenance: dict[str, Any] = {}


class HttpRejected(OSError):
    """A non-transient HTTP answer; never retried."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool(name: str) -> ModuleType:
    key = f"_kitti360_v3_batch_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def segments() -> dict[str, dict[str, Any]]:
    """The fixed development segments of the spent sequences this batch may read."""

    fixed = json.loads(SEGMENTS.read_text())
    candidates = set(fixed["rule"]["candidates"])
    result = {item["id"]: item for item in fixed["development"]}
    sequences = {item["sequence"] for item in result.values()}
    if sequences & candidates or not sequences <= set(SEQUENCES):
        raise ValueError("development segments must come from spent sequences 0000/0002")
    return result


def archive_names(sequence: str) -> tuple[str, ...]:
    if sequence not in SEQUENCES:
        raise ValueError(f"not a spent development sequence: {sequence}")
    return (*SHARED_ARCHIVES, f"{sequence}-camera00", f"{sequence}-velodyne")


def _bindings() -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in json.loads(BINDINGS.read_text())["archives"]}


def inventory(work: Path, *, read: Callable[[str], dict[str, Any]] | None = None) -> list[str]:
    """Re-list the spent archives' central directories; every range must match its binding."""

    read = read or _tool("kitti360_archive_metadata").inventory
    bindings = _bindings()
    names = sorted({name for sequence in SEQUENCES for name in archive_names(sequence)})
    directory = work / "inventory"
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        target = directory / f"{name}.json"
        if target.is_file():
            continue
        pin = bindings[name]
        listed = read(pin["url"])
        if (listed["size_bytes"], listed["etag"]) != (pin["size_bytes"], pin["etag"]):
            raise ValueError(f"{name}: archive size or ETag differs from its binding")
        if len(listed["members"]) != pin["member_count"]:
            raise ValueError(f"{name}: member count differs from its binding")
        ranges = [
            {key: item[key] for key in ("role", "start", "end", "sha256")}
            for item in listed["provenance"]["ranges"]
        ]
        expected = [
            {key: item[key] for key in ("role", "start", "end", "sha256")}
            for item in pin["provenance"]["ranges"]
        ]
        if ranges != expected:
            raise ValueError(f"{name}: metadata ranges differ from the committed SHA-256 bindings")
        target.write_text(json.dumps(listed, indent=2) + "\n", encoding="utf-8")
        written.append(name)
    return written


def _load_inventory(work: Path, name: str) -> dict[str, Any]:
    listed = json.loads((work / "inventory" / f"{name}.json").read_text())
    pin = _bindings()[name]
    if (listed["url"], listed["size_bytes"], listed["etag"]) != (
        pin["url"],
        pin["size_bytes"],
        pin["etag"],
    ):
        raise ValueError(f"{name}: local inventory does not match its binding")
    return listed


def _get(url: str, etag: str, size: int, start: int, end: int, retries: int = RETRIES) -> bytes:
    """One exact ranged GET under the pinned ETag; transport failures retry with a fresh request."""

    metadata = _tool("kitti360_archive_metadata")
    length = end - start + 1
    for attempt in range(retries + 1):
        request = Request(
            url,
            headers={
                "Range": f"bytes={start}-{end}",
                "If-Match": etag,
                "Accept-Encoding": "identity",
            },
        )
        try:
            with metadata.urlopen(request, timeout=120) as response:
                headers = {key.lower(): value for key, value in response.headers.items()}
                if response.status != 206 or response.geturl() != url:
                    raise HttpRejected("range request must return 206 without a redirect")
                if headers.get("content-range") != f"bytes {start}-{end}/{size}":
                    raise HttpRejected("Content-Range differs from the exact request")
                if headers.get("etag") != etag:
                    raise HttpRejected("range ETag differs from the pin")
                payload = response.read(length + 1)
        except urllib.error.HTTPError as exc:
            raise HttpRejected(f"HTTP {exc.code} for bytes {start}-{end}") from None
        except TRANSIENT:
            if attempt == retries:
                raise
            time.sleep(2.0 * 2**attempt)
            continue
        if len(payload) == length:
            return payload
        if attempt == retries:
            raise ValueError(f"range body {len(payload)} bytes, expected {length}")
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def _spans(listed: dict[str, Any], wanted: set[str]) -> Iterator[list[tuple[dict[str, Any], int]]]:
    """Requests of adjacent wanted members, each with its byte end (exclusive).

    A member's bytes run from its local header to the next member's local header (or the
    central directory), so a run of adjacent wanted members covers no other member.
    """

    directory = next(r for r in listed["provenance"]["ranges"] if r["role"] == "central_directory")
    ordered = sorted(listed["members"], key=lambda m: m["local_header_offset"])
    ends = [m["local_header_offset"] for m in ordered[1:]] + [directory["start"]]
    current: list[tuple[dict[str, Any], int]] = []
    for member, end in zip(ordered, ends, strict=True):
        if member["name"] not in wanted:
            if current:
                yield current
            current = []
            continue
        if current and end - current[0][0]["local_header_offset"] > REQUEST_BYTES:
            yield current
            current = []
        current.append((member, end))
    if current:
        yield current


def _member_content(member: dict[str, Any], record: bytes) -> bytes:
    """Decode one member from its exact byte span; header, size and CRC must match."""

    fields = struct.unpack("<4s5H3I2H", record[:30])
    signature, flags, method = fields[0], fields[2], fields[3]
    name_len, extra_len = fields[-2:]
    if signature != b"PK\x03\x04" or flags & 1 or method != member["compression"]:
        raise ValueError(f"{member['name']}: unexpected local header")
    name = record[30 : 30 + name_len].decode("utf-8" if flags & 2048 else "cp437")
    if name != member["name"]:
        raise ValueError(f"{member['name']}: local header names {name}")
    begin = 30 + name_len + extra_len
    compressed = record[begin : begin + member["compressed_bytes"]]
    trailer = len(record) - begin - member["compressed_bytes"]
    descriptor = (12, 16, 20, 24) if flags & 8 else (0,)
    if len(compressed) != member["compressed_bytes"] or trailer not in descriptor:
        raise ValueError(f"{member['name']}: member span does not match the central directory")
    if method == 0:
        content = compressed
    elif method == 8:
        decoder = zlib.decompressobj(-15)
        try:
            content = decoder.decompress(compressed, member["size_bytes"] + 1)
        except zlib.error as exc:
            raise ValueError(f"{member['name']}: corrupt deflate stream ({exc})") from None
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError(f"{member['name']}: deflate stream does not end at its size")
    else:
        raise ValueError(f"{member['name']}: unsupported compression {method}")
    if len(content) != member["size_bytes"] or f"{zlib.crc32(content):08x}" != member["crc32"]:
        raise ValueError(f"{member['name']}: size or CRC-32 mismatch")
    return content


def acquire(
    work: Path,
    segment: dict[str, Any],
    data: Path,
    *,
    cap_bytes: int = FETCH_CAP_BYTES,
    get: Callable[..., bytes] = _get,
) -> dict[str, Any]:
    """Range-read exactly one spent segment's members into the official split layout."""

    sequence = segment["sequence"]
    names = archive_names(sequence)
    seq = f"2013_05_28_drive_{sequence}_sync"
    frames = range(segment["start_frame"], segment["start_frame"] + segment["frames"])
    # Archive -> (prefix of the extracted path, member names).
    members: dict[str, tuple[str, list[str]]] = {
        "calibration": (
            "",
            [
                f"calibration/{f}"
                for f in ("perspective.txt", "calib_cam_to_pose.txt", "calib_cam_to_velo.txt")
            ],
        ),
        "perspective-timestamps": ("data_2d_raw/", [f"{seq}/image_00/timestamps.txt"]),
        "velodyne-timestamps": ("data_3d_raw/", [f"{seq}/velodyne_points/timestamps.txt"]),
        "oxts": (
            "",
            [
                f"data_poses/{seq}/oxts/timestamps.txt",
                *(f"data_poses/{seq}/oxts/data/{i:010d}.txt" for i in frames),
            ],
        ),
        names[4]: ("data_2d_raw/", [f"{seq}/image_00/data_rect/{i:010d}.png" for i in frames]),
        names[5]: ("data_3d_raw/", [f"{seq}/velodyne_points/data/{i:010d}.bin" for i in frames]),
    }
    plan = {
        name: {member: prefix + member for member in listed}
        for name, (prefix, listed) in members.items()
    }
    receipts: dict[str, Any] = {"archives": {}, "fetched_bytes": 0}
    digest = hashlib.sha256()
    for name, wanted in plan.items():
        listed = _load_inventory(work, name)
        by_name = {m["name"]: m for m in listed["members"]}
        missing = sorted(set(wanted) - set(by_name))
        if missing:
            raise ValueError(f"{name}: {len(missing)} segment members absent, e.g. {missing[0]}")
        fetched = 0
        for run in _spans(listed, set(wanted)):
            start = run[0][0]["local_header_offset"]
            end = run[-1][1]
            if receipts["fetched_bytes"] + end - start > cap_bytes:
                raise ValueError("segment exceeds the fetch cap")
            payload = get(listed["url"], listed["etag"], listed["size_bytes"], start, end - 1)
            receipts["fetched_bytes"] += len(payload)
            fetched += len(payload)
            for member, member_end in run:
                offset = member["local_header_offset"] - start
                content = _member_content(member, payload[offset : member_end - start])
                target = data / wanted[member["name"]]
                if not target.resolve().is_relative_to(data.resolve()) or target.exists():
                    raise ValueError(f"unsafe or repeated target {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                digest.update(
                    f"{member['name']}\0{member['size_bytes']}\0{member['crc32']}\n".encode()
                )
        receipts["archives"][name] = {
            "url": listed["url"],
            "etag": listed["etag"],
            "members": len(wanted),
            "fetched_bytes": fetched,
        }
    receipts["members_sha256"] = digest.hexdigest()
    return receipts


def _latest(work: Path, segment: str) -> tuple[int, dict[str, Any] | None]:
    attempts = sorted(
        int(p.name.removeprefix("attempt_"))
        for p in (work / segment).glob("attempt_*")
        if p.is_dir()
    )
    if not attempts:
        return 0, None
    report = work / segment / f"attempt_{attempts[-1]}" / "report.json"
    return attempts[-1], json.loads(report.read_text()) if report.is_file() else None


def run_segment(
    segment_id: str,
    work: Path,
    *,
    keep_arrays_free: int = KEEP_ARRAYS_FREE_BYTES,
    free: Callable[[], int] | None = None,
    get: Callable[..., bytes] = _get,
) -> SegmentReport:
    """One new attempt for one spent segment; the attempt directory is never reused."""

    segment = segments()[segment_id]
    free = free or (lambda: shutil.disk_usage(work).free)
    attempt = _latest(work, segment_id)[0] + 1
    out = work / segment_id / f"attempt_{attempt}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = SegmentReport(
        segment=segment_id,
        attempt=attempt,
        status="RUNNING",
        stage="preflight",
        details={"segment": segment},
        provenance={
            "source_commit": _commit(),
            "frozen_modules_sha256": {name: _sha(ROOT / name) for name in FROZEN},
            "segments_sha256": _sha(SEGMENTS),
            "bindings_sha256": _sha(BINDINGS),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "development_only": True,
            "evaluation_candidate_bytes_read": 0,
            "new_accuracy_claim": False,
        },
    )

    def save() -> None:
        report.details["wall_s"] = round(time.monotonic() - started, 1)
        (out / "report.json").write_text(report.model_dump_json(indent=2) + "\n", "utf-8")

    replay = _tool("kitti_camera_lidar_v3_replay")
    try:
        report.details["free_bytes_at_start"] = free()
        if report.details["free_bytes_at_start"] < START_FREE_BYTES:
            raise RuntimeError("insufficient free disk before acquisition")
        report.stage = "fetch"
        save()
        acquired = acquire(work, segment, out / "raw", get=get)
        (out / "fetch.json").write_text(json.dumps(acquired, indent=2) + "\n", "utf-8")
        report.details |= {
            "fetched_bytes": acquired["fetched_bytes"],
            "members_sha256": acquired["members_sha256"],
        }
        report.stage = "convert"
        save()
        from calibrex.data.kitti360_to_rosbag2 import convert_kitti360_to_rosbag2

        converted = convert_kitti360_to_rosbag2(
            out / "raw",
            out / "bag",
            sequence=f"2013_05_28_drive_{segment['sequence']}_sync",
            camera="image_00",
            velodyne_time="azimuth",
            motion_source="oxts",
            start_frame=segment["start_frame"],
            max_frames=segment["frames"],
            command=[sys.executable, str(Path(__file__)), "run", "SPENT_DEVELOPMENT_ONLY"],
        )
        shutil.copyfile(converted.provenance_path, out / "conversion.json")
        report.details["conversion_sha256"] = _sha(out / "conversion.json")
        shutil.rmtree(out / "raw")
        report.details["raw_deleted"] = True
        report.stage = "capture"
        save()
        code = replay.capture_drive(out / "bag", segment_id, out / "snapshots" / segment_id)
        report.details["capture_exit_code"] = code
        shutil.rmtree(out / "bag")
        report.details["bag_deleted"] = True
        if code != 0:
            raise RuntimeError("native capture failed; see retained per-run logs")
        report.stage = "compare"
        save()
        result = replay.compare_drives(
            out / "snapshots", out / "comparison", drives=[segment_id], proposed=AGREEMENT
        )
        report.details["proposed_options"] = result.get("proposed_options")
        report.details["comparison_errors"] = result["errors"]
        if result["errors"]:
            raise RuntimeError(f"comparison errors: {result['errors']}")
        report.stage = "complete"
        report.status = "PASS_DEVELOPMENT"
    except Exception as exc:
        report.status = "FAIL"
        report.error = f"{type(exc).__name__}: {exc}"
        (out / "failure.txt").write_text(traceback.format_exc(), "utf-8")
    finally:
        for leftover in ("raw", "bag"):
            if (out / leftover).exists():
                shutil.rmtree(out / leftover)
                report.details[f"{leftover}_deleted"] = True
        arrays = sorted((out / "snapshots").rglob("arrays.npz"))
        if arrays and free() < keep_arrays_free:
            report.details["arrays_deleted"] = {
                p.relative_to(out).as_posix(): _sha(p) for p in arrays
            }
            for path in arrays:
                path.unlink()
        report.details["arrays_kept"] = [
            p.relative_to(out).as_posix() for p in sorted((out / "snapshots").rglob("arrays.npz"))
        ]
        save()
    return report


def run(work: Path, chosen: Sequence[str] | None = None, retry_failed: bool = False) -> int:
    """Process segments in order under a single-runner lock; earlier attempts stay untouched."""

    import fcntl

    known = segments()
    order = list(known) if not chosen else list(chosen)
    unknown = [item for item in order if item not in known]
    if unknown or len(set(order)) != len(order):
        raise ValueError(f"choose distinct spent development segments: {unknown}")
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".batch.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("another batch runner holds this work root") from None
        failures = 0
        for segment_id in order:
            attempt, previous = _latest(work, segment_id)
            retry = retry_failed and previous is not None and previous["status"] in RETRYABLE
            if previous is not None and not retry:
                print(segment_id, "skip", previous["status"], f"attempt_{attempt}", flush=True)
                continue
            report = run_segment(segment_id, work)
            failures += report.status != "PASS_DEVELOPMENT"
            print(segment_id, report.status, report.stage, report.details.get("wall_s"), flush=True)
    return 1 if failures else 0


def publish(work: Path, dest: Path) -> int:
    """Copy reports and non-sensor receipts; never images, clouds, bags or NPZ arrays."""

    count = 0
    for report_path in sorted(work.glob("*/attempt_*/report.json")):
        source = report_path.parent
        target = dest / source.parent.name / source.name
        if target.exists():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in PUBLISHED_SUFFIXES:
                relative = path.relative_to(source)
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                (target / relative).write_bytes(path.read_bytes())
                count += 1
    files = [
        {"path": p.relative_to(dest).as_posix(), "bytes": p.stat().st_size, "sha256": _sha(p)}
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    manifest = {
        "schema_version": SCHEMA,
        "policy": "unmodified reports and non-sensor receipts; no images, clouds, bags, "
        "archive payloads or NPZ inputs",
        "files": files,
        "provenance": _provenance(),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    return count


def _provenance() -> dict[str, Any]:
    return {
        "generator": "tools/kitti360_camera_lidar_v3_batch.py",
        "generator_sha256": _sha(Path(__file__)),
        "git_commit": _commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def aggregate(dest: Path, output: Path) -> dict[str, Any]:
    """Pool each segment's latest published scores under the default and agreement rules."""

    summarize = _tool("kitti_camera_lidar_v3_development").summarize_scores
    scorer = _tool("score_kitti_camera_lidar_v2")
    scores: dict[str, dict[str, Any]] = {"current": {}, "proposed": {}}
    statuses: dict[str, str] = {}
    for segment_id in segments():
        attempt, report = _latest(dest, segment_id)
        statuses[segment_id] = "missing" if report is None else report["status"]
        comparison = dest / segment_id / f"attempt_{attempt}" / "comparison" / "comparison.json"
        if report is not None and report["status"] == "PASS_DEVELOPMENT" and comparison.is_file():
            per_segment = json.loads(comparison.read_text())["scores"]
            for rule in scores:
                scores[rule][segment_id] = per_segment[rule][segment_id]
        else:
            for rule in scores:
                scores[rule][segment_id] = asdict(scorer.DriveScore("failed", statuses[segment_id]))
    complete = all(status == "PASS_DEVELOPMENT" for status in statuses.values())
    result = {
        "schema_version": SCHEMA,
        "analysis": "development; spent KITTI-360 segments only",
        "status": "complete" if complete else "incomplete",
        "audit_thresholds": None,
        "segment_status": statuses,
        "metrics": {rule: summarize(values) for rule, values in scores.items()},
        "rules": {"current": "default", "proposed": AGREEMENT},
        "attempts": {segment_id: _latest(dest, segment_id)[0] for segment_id in segments()},
        "provenance": _provenance(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("work", type=Path)
    r = sub.add_parser("run")
    r.add_argument("work", type=Path)
    r.add_argument("--segment", action="append")
    r.add_argument("--retry-failed", action="store_true")
    u = sub.add_parser("publish")
    u.add_argument("work", type=Path)
    u.add_argument("dest", type=Path)
    a = sub.add_parser("aggregate")
    a.add_argument("dest", type=Path)
    a.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        print("inventoried", inventory(args.work))
        return 0
    if args.command == "run":
        return run(args.work, args.segment, args.retry_failed)
    if args.command == "publish":
        print(publish(args.work, args.dest), "files published")
        return 0
    print(aggregate(args.dest, args.output)["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
