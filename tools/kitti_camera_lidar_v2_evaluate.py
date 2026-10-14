#!/usr/bin/env python3
"""Drive the KITTI camera-lidar v2 runs: fetch each drive's head, run, delete.

An evaluation harness, not part of the Calibrex package. Drives are named ``DATE_DRIVE``
(``2011_09_30_0028``) and may come from any KITTI raw day. For each day it downloads
``DATE_calib.zip`` once (its SHA-256 must equal the pinned one when a pre-registration is
given). For each drive it fetches the first ``max_frames`` frames of ``image_02``,
``velodyne_points`` and ``oxts`` with ``tools/kitti_drive_fetch.py`` (HTTP ranges of the
archive, every member's CRC-32 checked; ``calibrex check`` reads only a recording's first
48 s or so, and the fetched head of a long drive gives the same check as the whole drive),
runs ``tools/kitti_camera_lidar_sota_run.py`` with the injections, and deletes the drive and
the bag. With a pre-registration, every day's calibration and every drive's archive must
have a pin: a calibration that differs from its pin (or has none) stops everything, and a
drive whose archive size or ETag differs from its pin (or has none) is not fetched and
fails. The next fetch overlaps the current runs. A drive whose ``manifest.json`` exists is
not run again; every command, exit code, wall time and failure reason goes to
``evaluate_record.json``::

    python tools/kitti_camera_lidar_v2_evaluate.py --drive 2011_09_26_0005 --inject x:3 \\
        --max-frames 600 --velodyne-time azimuth --work-dir WORK --output-dir RUNS
    python tools/kitti_camera_lidar_v2_evaluate.py \\
        --preregistration docs/benchmarks/kitti_camera_lidar_v2_preregistration.yaml \\
        --work-dir WORK --output-dir RUNS
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


def _fetch_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "kitti_drive_fetch", ROOT / "tools" / "kitti_drive_fetch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["kitti_drive_fetch"] = module
    spec.loader.exec_module(module)
    return module


class Record:
    """Append-only log of every command."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        )
        self.lock = threading.Lock()

    def run(self, label: str, command: list[str], log: Path) -> int:
        started = time.monotonic()
        with log.open("a", encoding="utf-8") as stream:
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
        self.add(label, command, code, time.monotonic() - started)
        return code

    def add(
        self, label: str, command: list[str], code: int, wall_s: float, detail: str | None = None
    ) -> None:
        entry: dict[str, Any] = {
            "label": label,
            "command": command,
            "exit_code": code,
            "wall_s": round(wall_s, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if detail is not None:
            entry["detail"] = detail
        with self.lock:
            self.entries.append(entry)
            self.path.write_text(json.dumps(self.entries, indent=2) + "\n", encoding="utf-8")
        suffix = f" ({detail})" if detail is not None else ""
        print(f"[{time.strftime('%H:%M:%S')}] {label}: exit {code}{suffix}", flush=True)


def split_drive(name: str) -> tuple[str, str]:
    """``2011_09_30_0028`` -> ``("2011_09_30", "0028")``."""

    date, _, drive = name.rpartition("_")
    if len(date) != 10 or len(drive) != 4:
        raise ValueError(f"a drive is named DATE_DRIVE, e.g. 2011_09_30_0028: {name}")
    return date, drive


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration(
    record: Record, date: str, work_dir: Path, pinned: str | None, require_pin: bool
) -> dict[str, Any]:
    """Download ``DATE_calib.zip`` and unpack it when it matches its pin; return its record."""

    target = work_dir / f"{date}_calib.zip"
    url = f"{S3}/{date}_calib.zip"
    code = record.run(
        f"download {target.name}",
        ["curl", "-sS", "-f", "--retry", "3", "-o", str(target), url],
        work_dir / f"{date}_calib.log",
    )
    entry: dict[str, Any] = {"url": url, "ok": False, "pinned_sha256": pinned}
    if code != 0:
        entry["reason"] = f"download failed: curl exit {code}"
        return entry
    entry["sha256"] = sha256_file(target)
    if pinned is None and require_pin:
        entry["reason"] = "the pre-registration pins no SHA-256 for this day"
    elif pinned is not None and entry["sha256"] != pinned:
        entry["reason"] = "SHA-256 differs from the pre-registration"
    else:
        entry["ok"] = True
        with zipfile.ZipFile(target) as archive:
            archive.extractall(work_dir)
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--drive", action="append", default=[], help="DATE_DRIVE")
    parser.add_argument("--inject", action="append", default=[], help="AXIS:DEG")
    parser.add_argument("--velodyne-time", choices=("none", "azimuth"), default="azimuth")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan: dict[str, Any] = {}
    if args.preregistration is not None:
        plan = yaml.safe_load(args.preregistration.read_text(encoding="utf-8"))
        drives = [str(name) for name in plan["recordings"]["evaluation"]]
        injections = [str(spec) for spec in plan["run_recipe"]["injections"]]
        velodyne_time = str(plan["run_recipe"]["velodyne_time"])
        max_frames = int(plan["run_recipe"]["max_frames"])
    else:
        drives, injections = args.drive, args.inject
        velodyne_time, max_frames = args.velodyne_time, args.max_frames
    if not drives:
        parser.error("no drives: pass --drive or --preregistration")
    fetcher = _fetch_module()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = Record(args.work_dir / "evaluate_record.json")
    require_pin = bool(plan)
    pinned_calibration = plan.get("recordings", {}).get("calibration_archive_sha256", {})
    pinned_archives = plan.get("recordings", {}).get("archives", {})

    calibrations: dict[str, dict[str, Any]] = {}
    for date in sorted({split_drive(name)[0] for name in drives}):
        calibrations[date] = calibration(
            record, date, args.work_dir, pinned_calibration.get(date), require_pin
        )
    (args.output_dir / "calibration_download.json").write_text(
        json.dumps(calibrations, indent=2) + "\n", encoding="utf-8"
    )
    if not all(item["ok"] for item in calibrations.values()):
        print("a calibration archive is missing or differs from the pre-registration")
        return 1

    pending = [d for d in drives if not (args.output_dir / d / "manifest.json").exists()]
    fetched: dict[str, dict[str, Any]] = {}
    threads: dict[str, threading.Thread] = {}

    def fetch(name: str) -> None:
        date, drive = split_drive(name)
        started = time.monotonic()
        pin = pinned_archives.get(name)
        expected = None if pin is None else (int(pin["size_bytes"]), str(pin["etag"]))
        try:
            if expected is None and require_pin:
                raise ValueError("the pre-registration pins no size and ETag for this drive")
            entry = fetcher.fetch_drive(date, drive, max_frames, args.work_dir / name, expected)
            fetched[name] = {"ok": True, **entry}
        except Exception as exc:  # any failure (network, CRC, a pin) fails the drive
            fetched[name] = {"ok": False, "url": fetcher.drive_url(date, drive), "reason": str(exc)}
        record.add(
            f"fetch {name}",
            ["kitti_drive_fetch", date, drive, str(max_frames), *map(str, expected or ())],
            0 if fetched[name]["ok"] else 1,
            time.monotonic() - started,
            None if fetched[name]["ok"] else str(fetched[name]["reason"]),
        )

    def start(name: str) -> None:
        threads[name] = threading.Thread(target=fetch, args=(name,))
        threads[name].start()

    if pending:
        start(pending[0])
    for index, name in enumerate(pending):
        threads[name].join()
        if index + 1 < len(pending):
            start(pending[index + 1])
        date, drive = split_drive(name)
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        entry = fetched[name]
        members = entry.pop("members", None)
        (out / "download.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        if members is not None:  # its SHA-256 is in download.json
            (args.work_dir / f"{name}_members.json").write_text(
                json.dumps(members, indent=1) + "\n", encoding="utf-8"
            )
        drive_root = args.work_dir / name
        if not entry["ok"]:
            print(f"{name}: {entry['reason']}; its units fail", flush=True)
            shutil.rmtree(drive_root, ignore_errors=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "tools" / "kitti_camera_lidar_sota_run.py"),
            str(drive_root / date / f"{date}_drive_{drive}_sync"),
            "--recording",
            name,
            "--calib-dir",
            str(args.work_dir / date),
            *[item for spec in injections for item in ("--inject", spec)],
            "--velodyne-time",
            velodyne_time,
            "--bag-dir",
            str(args.work_dir / f"bag_{name}"),
            "--download-record",
            str(out / "download.json"),
            "--output-dir",
            str(out),
        ]
        record.run(f"camera-lidar {name}", command, args.work_dir / f"{name}_run.log")
        shutil.rmtree(drive_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
