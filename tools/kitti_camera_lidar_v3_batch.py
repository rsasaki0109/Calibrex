#!/usr/bin/env python3
"""Sequential spent-drive capture batch for the v3 camera-LiDAR development comparison.

Development only: all 55 drives are spent. For each pinned drive the batch verifies the
archive pin and calibration hash, fetches the first 600 frames, converts them, runs the
seven real checks through ``kitti_camera_lidar_v3_replay.capture_drive`` and replays both
far-start rules on that drive. Raw frames and the bag are then deleted. Captured NPZ
inputs stay only while free disk remains above a reserve; their deletion is recorded with
their digests. Every failure is retained. Nothing is overwritten or retried automatically.

    python tools/kitti_camera_lidar_v3_batch.py pin META_DIR PINS.json
    python tools/kitti_camera_lidar_v3_batch.py run PINS.json WORK_ROOT [--drive D] [--retry-failed]
    python tools/kitti_camera_lidar_v3_batch.py publish WORK_ROOT DEST
    python tools/kitti_camera_lidar_v3_batch.py aggregate DEST SUMMARY.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calibrex.camera_lidar_v3_spent_batch/v0.1"
MAX_FRAMES = 600
FETCH_CAP_BYTES = 2 * 1024**3
START_FREE_BYTES = 8 * 1024**3
KEEP_ARRAYS_FREE_BYTES = 12 * 1024**3
MARGIN = 3.5
CALIBRATION_SOURCES = (
    "docs/assets/kitti_camera_lidar_v2/development/calibration_download.json",
    "docs/assets/kitti_camera_lidar_v2/evaluation/runs/calibration_download.json",
)
FROZEN = (
    "tools/kitti_camera_lidar_v3_batch.py",
    "tools/kitti_drive_fetch.py",
    "tools/kitti_camera_lidar_v3_replay.py",
    "tools/kitti_camera_lidar_v3_development.py",
    "tools/score_kitti_camera_lidar_v2.py",
    "src/calibrex/evaluation/camera_lidar_edge.py",
    "src/calibrex/data/kitti_raw_to_rosbag2.py",
)
PUBLISHED_SUFFIXES = {".json", ".log", ".yaml"}


class DriveReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA
    drive: str
    attempt: int
    status: str
    stage: str
    error: str | None = None
    details: dict[str, Any] = {}
    provenance: dict[str, Any] = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool(name: str) -> ModuleType:
    key = f"_v3_batch_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def pin(metadata_dir: Path, output: Path) -> dict[str, Any]:
    """Pin archive size/ETag/frame counts and per-day calibration hashes before payloads."""

    spent = _tool("kitti_camera_lidar_v3_replay").spent_drives()
    calibration: dict[str, str] = {}
    for source in CALIBRATION_SOURCES:
        for date, record in json.loads((ROOT / source).read_text()).items():
            if calibration.setdefault(date, record["sha256"]) != record["sha256"]:
                raise ValueError(f"conflicting calibration hashes for {date}")
    drives: dict[str, Any] = {}
    for path in sorted(metadata_dir.glob("*.json")):
        date = path.stem
        for number, record in json.loads(path.read_text()).items():
            drives[f"{date}_{number}"] = {
                "date": date,
                "number": number,
                "size_bytes": record["size_bytes"],
                "etag": record["etag"],
                "frames": record["frames"],
            }
    if set(drives) != set(spent):
        raise ValueError("metadata must cover exactly the 55 spent drives")
    missing = {d["date"] for d in drives.values()} - set(calibration)
    if missing:
        raise ValueError(f"no calibration hash for {sorted(missing)}")
    pins = {
        "schema_version": SCHEMA,
        "purpose": "spent development capture only; no fresh evaluation data",
        "max_frames": MAX_FRAMES,
        "fetch_cap_bytes": FETCH_CAP_BYTES,
        "calibration_sha256": {date: calibration[date] for date in sorted(calibration)},
        "drives": {drive: drives[drive] for drive in spent},
    }
    output.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    return pins


def _latest(work: Path, drive: str) -> tuple[int, dict[str, Any] | None]:
    attempts = sorted(
        int(p.name.removeprefix("attempt_")) for p in (work / drive).glob("attempt_*") if p.is_dir()
    )
    if not attempts:
        return 0, None
    report = work / drive / f"attempt_{attempts[-1]}" / "report.json"
    return attempts[-1], json.loads(report.read_text()) if report.is_file() else None


def _calibration(work: Path, date: str, sha256: str) -> Path:
    """Download one day's calibration once, unpack it only if it matches its pin."""

    directory = work / "calibration" / date
    archive = work / "calibration" / f"{date}_calib.zip"
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        fetch = _tool("kitti_drive_fetch")
        with urllib.request.urlopen(f"{fetch.S3}/{date}_calib.zip", timeout=60) as response:
            data = response.read(1048577)
        if len(data) > 1048576:
            raise ValueError("calibration archive exceeds 1 MiB")
        archive.write_bytes(data)
    if _sha(archive) != sha256:
        raise ValueError(f"calibration {date} differs from its pin")
    if not directory.is_dir():
        with zipfile.ZipFile(archive) as content:
            if content.testzip() is not None:
                raise ValueError("calibration archive CRC failure")
            directory.mkdir(parents=True)
            for name in ("calib_cam_to_cam.txt", "calib_imu_to_velo.txt", "calib_velo_to_cam.txt"):
                (directory / name).write_bytes(content.read(f"{date}/{name}"))
    return directory


def run_drive(
    pins: dict[str, Any],
    drive: str,
    work: Path,
    *,
    keep_arrays_free: int = KEEP_ARRAYS_FREE_BYTES,
    free: Callable[[], int] | None = None,
) -> DriveReport:
    """One new attempt for one drive; the attempt directory is never reused."""

    free = free or (lambda: shutil.disk_usage(work).free)
    pin_record = pins["drives"][drive]
    attempt = _latest(work, drive)[0] + 1
    out = work / drive / f"attempt_{attempt}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    report = DriveReport(
        drive=drive,
        attempt=attempt,
        status="RUNNING",
        stage="preflight",
        provenance={
            "source_commit": commit,
            "frozen_modules_sha256": {name: _sha(ROOT / name) for name in FROZEN},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "development_only": True,
            "new_evaluation_started": False,
            "new_accuracy_claim": False,
        },
    )

    def save() -> None:
        report.details["wall_s"] = round(time.monotonic() - started, 1)
        (out / "report.json").write_text(report.model_dump_json(indent=2) + "\n", "utf-8")

    replay = _tool("kitti_camera_lidar_v3_replay")
    fetch = _tool("kitti_drive_fetch")
    try:
        report.details["free_bytes_at_start"] = free()
        if report.details["free_bytes_at_start"] < START_FREE_BYTES:
            raise RuntimeError("insufficient free disk before acquisition")
        save()
        report.stage = "calibration"
        calibration = _calibration(
            work, pin_record["date"], pins["calibration_sha256"][pin_record["date"]]
        )
        report.stage = "fetch"
        save()
        acquired = fetch.fetch_drive(
            pin_record["date"],
            pin_record["number"],
            pins["max_frames"],
            out / "raw",
            expected=(pin_record["size_bytes"], pin_record["etag"]),
            max_fetch_bytes=pins["fetch_cap_bytes"],
        )
        (out / "fetch.json").write_text(json.dumps(acquired, indent=2) + "\n", "utf-8")
        report.details |= {
            "fetched_bytes": acquired["fetched_bytes"],
            "frames": acquired["frames"],
            "members_sha256": acquired["members_sha256"],
        }
        report.stage = "convert"
        save()
        from calibrex.data.kitti_raw_to_rosbag2 import convert_kitti_raw_to_rosbag2

        name = f"{pin_record['date']}_drive_{pin_record['number']}_sync"
        converted = convert_kitti_raw_to_rosbag2(
            out / "raw" / pin_record["date"] / name,
            out / "bag",
            calibration_dir=calibration,
            camera="image_02",
            velodyne_time="azimuth",
            command=[sys.executable, str(Path(__file__)), "run", "SPENT_DEVELOPMENT_ONLY"],
        )
        shutil.copyfile(converted.provenance_path, out / "conversion.json")
        report.details["conversion_sha256"] = _sha(out / "conversion.json")
        shutil.rmtree(out / "raw")
        report.details["raw_deleted"] = True
        report.stage = "capture"
        save()
        code = replay.capture_drive(out / "bag", drive, out / "snapshots" / drive)
        report.details["capture_exit_code"] = code
        shutil.rmtree(out / "bag")
        report.details["bag_deleted"] = True
        if code != 0:
            raise RuntimeError("native capture failed; see retained per-run logs")
        report.stage = "compare"
        save()
        result = replay.compare_drives(
            out / "snapshots", out / "comparison", drives=[drive], margin=MARGIN
        )
        report.details["comparison_status"] = result["status"]
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


def run(
    pins_path: Path, work: Path, drives: Sequence[str] | None = None, retry_failed: bool = False
) -> int:
    """Process drives in pin order. A drive with an attempt is skipped unless its latest
    attempt failed and ``retry_failed`` asks for a new, separate attempt."""

    pins = json.loads(pins_path.read_text())
    if pins.get("schema_version") != SCHEMA:
        raise ValueError("unexpected pin schema")
    order = list(pins["drives"]) if not drives else list(drives)
    failures = 0
    for drive in order:
        if drive not in pins["drives"]:
            raise ValueError(f"{drive} is not pinned")
        attempt, previous = _latest(work, drive)
        if previous is not None and not (retry_failed and previous["status"] == "FAIL"):
            print(drive, "skip", previous["status"], f"attempt_{attempt}", flush=True)
            continue
        report = run_drive(pins, drive, work)
        failures += report.status != "PASS_DEVELOPMENT"
        print(drive, report.status, report.stage, report.details.get("wall_s"), flush=True)
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
            if not path.is_file() or path.suffix not in PUBLISHED_SUFFIXES | {".txt"}:
                continue
            relative = path.relative_to(source)
            if relative.parts[0] == "calibration":
                continue
            data = path.read_bytes()
            if relative.as_posix() == "fetch.json":
                record = json.loads(data)
                record.pop("members")  # members_sha256 still pins every name/size/CRC
                data = (json.dumps(record, indent=2) + "\n").encode()
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            (target / relative).write_bytes(data)
            count += 1
    return count


def aggregate(dest: Path, output: Path) -> dict[str, Any]:
    """Pool each drive's latest published single-drive scores under both rules."""

    replay = _tool("kitti_camera_lidar_v3_replay")
    summarize = _tool("kitti_camera_lidar_v3_development").summarize_scores
    scorer = _tool("score_kitti_camera_lidar_v2")
    scores: dict[str, dict[str, Any]] = {"current": {}, "proposed": {}}
    statuses: dict[str, str] = {}
    for drive in replay.spent_drives():
        attempt, report = _latest(dest, drive)
        statuses[drive] = "missing" if report is None else report["status"]
        comparison = dest / drive / f"attempt_{attempt}" / "comparison" / "comparison.json"
        if report is not None and report["status"] == "PASS_DEVELOPMENT" and comparison.is_file():
            per_drive = json.loads(comparison.read_text())["scores"]
            for rule in scores:
                scores[rule][drive] = per_drive[rule][drive]
        else:
            for rule in scores:
                scores[rule][drive] = asdict(scorer.DriveScore("failed", statuses[drive]))
    complete = all(status == "PASS_DEVELOPMENT" for status in statuses.values())
    result = {
        "schema_version": SCHEMA,
        "analysis": "post hoc development; spent drives only",
        "status": "complete" if complete else "incomplete",
        "audit_thresholds": None,
        "drive_status": statuses,
        "metrics": {rule: summarize(values) for rule, values in scores.items()},
        "margin": MARGIN,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pin")
    p.add_argument("metadata_dir", type=Path)
    p.add_argument("output", type=Path)
    r = sub.add_parser("run")
    r.add_argument("pins", type=Path)
    r.add_argument("work", type=Path)
    r.add_argument("--drive", action="append")
    r.add_argument("--retry-failed", action="store_true")
    u = sub.add_parser("publish")
    u.add_argument("work", type=Path)
    u.add_argument("dest", type=Path)
    a = sub.add_parser("aggregate")
    a.add_argument("dest", type=Path)
    a.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "pin":
        pin(args.metadata_dir, args.output)
        return 0
    if args.command == "run":
        return run(args.pins, args.work, args.drive, args.retry_failed)
    if args.command == "publish":
        print(publish(args.work, args.dest), "files published")
        return 0
    result = aggregate(args.dest, args.output)
    print(result["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
