#!/usr/bin/env python3
"""Drive the KITTI camera-lidar SOTA runs: download, run, delete, one drive at a time.

An evaluation harness, not part of the Calibrex package. For every 2011_09_26 drive it
downloads ``2011_09_26_drive_XXXX_sync.zip`` from the KITTI raw S3 bucket, records its
size, S3 ETag and SHA-256, unzips it, runs ``tools/kitti_camera_lidar_sota_run.py`` with
the injections, and deletes the archive, the drive and the converted bag. The next
download overlaps the current runs. A drive whose ``manifest.json`` exists is not run
again, so an interrupted driver can be restarted; every command, exit code and wall
time goes to ``evaluate_record.json``.

Development runs list their drives and injections on the command line; the evaluation
reads them, with each archive's pinned size and ETag and the calibration archive's
SHA-256, from the pre-registration (``--preregistration``), and a drive whose archive
does not match its pinned size or ETag is not run::

    python tools/kitti_camera_lidar_sota_evaluate.py --drive 0005 --drive 0009 \\
        --inject x:3 --inject y:3 --inject z:3 --work-dir WORK --output-dir RUNS
    python tools/kitti_camera_lidar_sota_evaluate.py \\
        --preregistration docs/benchmarks/kitti_camera_lidar_preregistration.yaml \\
        --work-dir WORK --output-dir RUNS
"""

from __future__ import annotations

import argparse
import hashlib
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
DATE = "2011_09_26"


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

    def add(self, label: str, command: list[str], code: int, wall_s: float) -> None:
        with self.lock:
            self.entries.append(
                {
                    "label": label,
                    "command": command,
                    "exit_code": code,
                    "wall_s": round(wall_s, 1),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            self.path.write_text(json.dumps(self.entries, indent=2) + "\n", encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {label}: exit {code}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def head(url: str) -> tuple[int | None, str | None]:
    """The archive's Content-Length and ETag, from an HTTP HEAD request."""

    lines = subprocess.run(
        ["curl", "-sSI", "-m", "60", url], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    size = etag = None
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            size = int(value.strip())
        elif key.strip().lower() == "etag":
            etag = value.strip().strip('"')
    return size, etag


def drive_url(drive: str) -> str:
    name = f"{DATE}_drive_{drive}"
    return f"{S3}/{name}/{name}_sync.zip"


def fetch(record: Record, url: str, target: Path, pinned: dict[str, Any] | None) -> dict[str, Any]:
    """Download ``url`` to ``target``; return its download record (``ok`` false on failure)."""

    size, etag = head(url)
    entry: dict[str, Any] = {"url": url, "size_bytes": size, "etag": etag}
    if pinned is not None and (size, etag) != (pinned["size_bytes"], pinned["etag"]):
        entry.update(ok=False, reason="size or ETag differs from the pre-registration")
        record.add(f"verify {target.name}", ["curl", "-I", url], 1, 0.0)
        return entry
    started = time.monotonic()
    for attempt in range(3):  # a failed download reads no message and may be repeated
        command = ["curl", "-sS", "-f", "--retry", "3", "-o", str(target), url]
        code = record.run(
            f"download {target.name} (attempt {attempt + 1})", command, target.with_suffix(".log")
        )
        if code == 0 and target.stat().st_size == size:
            break
        target.unlink(missing_ok=True)
    else:
        entry.update(ok=False, reason="download failed")
        return entry
    entry.update(ok=True, sha256=sha256_file(target), wall_s=round(time.monotonic() - started, 1))
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--drive", action="append", default=[], help="development: e.g. 0005")
    parser.add_argument("--inject", action="append", default=[], help="development: AXIS:DEG")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan: dict[str, Any] = {}
    if args.preregistration is not None:
        plan = yaml.safe_load(args.preregistration.read_text(encoding="utf-8"))
        drives = list(plan["recordings"]["evaluation"])
        injections = list(plan["run_recipe"]["injections"])
    else:
        drives, injections = args.drive, args.inject
    if not drives:
        parser.error("no drives: pass --drive or --preregistration")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = Record(args.work_dir / "evaluate_record.json")

    calib_zip = args.work_dir / f"{DATE}_calib.zip"
    calib = fetch(record, f"{S3}/{DATE}_calib.zip", calib_zip, None)
    pinned_calib = plan.get("recordings", {}).get("calibration_archive_sha256")
    if not calib["ok"] or (pinned_calib is not None and calib["sha256"] != pinned_calib):
        print("the calibration archive is missing or differs from the pre-registration")
        return 1
    with zipfile.ZipFile(calib_zip) as archive:
        archive.extractall(args.work_dir)
    (args.output_dir / "calibration_download.json").write_text(
        json.dumps(calib, indent=2) + "\n", encoding="utf-8"
    )
    calib_dir = args.work_dir / DATE
    pinned = plan.get("recordings", {}).get("archives", {})

    pending = [d for d in drives if not (args.output_dir / d / "manifest.json").exists()]
    downloads: dict[str, dict[str, Any]] = {}
    threads: dict[str, threading.Thread] = {}

    def download(drive: str) -> None:
        target = args.work_dir / f"{DATE}_drive_{drive}_sync.zip"
        downloads[drive] = fetch(record, drive_url(drive), target, pinned.get(drive))

    def start(drive: str) -> None:
        threads[drive] = threading.Thread(target=download, args=(drive,))
        threads[drive].start()

    if pending:
        start(pending[0])
    for index, drive in enumerate(pending):
        threads[drive].join()
        if index + 1 < len(pending):
            start(pending[index + 1])
        archive = args.work_dir / f"{DATE}_drive_{drive}_sync.zip"
        out = args.output_dir / drive
        out.mkdir(parents=True, exist_ok=True)
        (out / "download.json").write_text(
            json.dumps(downloads[drive], indent=2) + "\n", encoding="utf-8"
        )
        if not downloads[drive]["ok"]:
            print(f"{drive}: {downloads[drive]['reason']}; its units fail", flush=True)
            archive.unlink(missing_ok=True)
            continue
        started = time.monotonic()
        with zipfile.ZipFile(archive) as opened:
            opened.extractall(args.work_dir)
        record.add(f"unzip {archive.name}", ["unzip", archive.name], 0, time.monotonic() - started)
        archive.unlink()
        drive_dir = calib_dir / f"{DATE}_drive_{drive}_sync"
        command = [
            sys.executable,
            str(ROOT / "tools" / "kitti_camera_lidar_sota_run.py"),
            str(drive_dir),
            "--recording",
            drive,
            "--calib-dir",
            str(calib_dir),
            *[item for spec in injections for item in ("--inject", spec)],
            "--bag-dir",
            str(args.work_dir / f"bag_{drive}"),
            "--download-record",
            str(out / "download.json"),
            "--output-dir",
            str(out),
        ]
        record.run(f"camera-lidar {drive}", command, args.work_dir / f"{drive}_run.log")
        shutil.rmtree(drive_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
