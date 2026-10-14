#!/usr/bin/env python3
"""Drive the pre-registered Hilti camera audits: download, run, delete, one recording at a time.

An evaluation harness, not part of the Calibrex package. It reads the recordings, files
and pinned hashes from the claim's pre-registration
(``docs/benchmarks/hilti_camera_sota_preregistration.yaml``, or
``docs/benchmarks/hilti_camera_imu_v3_preregistration.yaml`` for ``camera-imu-v3``) and
runs its ``run_recipe``:

* ``camera-camera``: for every evaluation recording, download the first 5 GB
  (``tools/hilti_download.py``), run ``tools/hilti_camera_sota_run.py`` with the two
  pre-registered injections, delete the download. The next download overlaps the
  current runs (network and CPU are separate bottlenecks here).
* ``camera-imu-v2`` and ``camera-imu-v3``: for every evaluation recording, download the
  full bag, check its SHA-256 against the pinned one, run v1's ``calibrex camera-imu
  rotation`` command for cam0 and cam1, delete the bag. v3 keeps each run's log beside its
  artifact, where its scorer reads an estimator refusal.

A recording whose outputs already exist is not run again, so an interrupted driver can
be restarted; every command, exit code and wall time goes to ``evaluate_record.json``.
The calibration files' SHA-256 must equal the pinned ones, or nothing runs.

::

    python tools/hilti_camera_sota_evaluate.py camera-camera --calibration-dir CAL \\
        --work-dir WORK --output-dir RUNS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATIONS = {
    "camera-camera": ROOT / "docs" / "benchmarks" / "hilti_camera_sota_preregistration.yaml",
    "camera-imu-v2": ROOT / "docs" / "benchmarks" / "hilti_camera_sota_preregistration.yaml",
    "camera-imu-v3": ROOT / "docs" / "benchmarks" / "hilti_camera_imu_v3_preregistration.yaml",
}
HEAD_BYTES = 5_000_000_000
CAMERA_IMU_COMMAND = [
    "--imu-topic",
    "/alphasense/imu",
    "--frame-stride",
    "4",
    "--min-window-rotation-deg",
    "1.0",
    "--observable-rotation-std-deg",
    "0.3",
    "--dataset-family",
    "hilti2022",
]
LICENSE = "Hilti SLAM Challenge 2022 terms"


class Record:
    """Append-only log of every command of the evaluation."""

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
        with self.lock:
            self.entries.append(
                {
                    "label": label,
                    "command": command,
                    "exit_code": code,
                    "wall_s": round(time.monotonic() - started, 1),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            self.path.write_text(json.dumps(self.entries, indent=2) + "\n", encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {label}: exit {code}", flush=True)
        return code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def download(record: Record, file: str, target: Path, head: bool, log: Path) -> int:
    command = [sys.executable, str(ROOT / "tools" / "hilti_download.py"), file]
    command += ["--bytes", str(HEAD_BYTES)] if head else []
    command += ["--output", str(target)]
    for attempt in range(3):  # a failed download reads no message and may be repeated
        code = record.run(f"download {file} (attempt {attempt + 1})", command, log)
        if code == 0:
            return 0
        target.unlink(missing_ok=True)
    return code


def camera_camera(args: argparse.Namespace, plan: dict[str, Any], record: Record) -> None:
    recordings = plan["recordings"]["evaluation"]["camera-camera"]
    bags = plan["recordings"]["bags"]
    pending = [r for r in recordings if not (args.output_dir / r / "manifest.json").exists()]
    downloads: dict[str, threading.Thread] = {}
    results: dict[str, int] = {}

    def fetch(recording: str) -> None:
        target = args.work_dir / f"{recording}_head.bag"
        log = args.work_dir / f"{recording}_download.log"
        results[recording] = download(record, bags[recording]["file"], target, True, log)

    def start(recording: str) -> None:
        downloads[recording] = threading.Thread(target=fetch, args=(recording,))
        downloads[recording].start()

    if pending:
        start(pending[0])
    for index, recording in enumerate(pending):
        downloads[recording].join()
        if index + 1 < len(pending):
            start(pending[index + 1])
        bag = args.work_dir / f"{recording}_head.bag"
        if results[recording] != 0:
            print(f"{recording}: download failed; its units fail", flush=True)
            continue
        record.run(
            f"camera-camera {recording}",
            [
                sys.executable,
                str(ROOT / "tools" / "hilti_camera_sota_run.py"),
                str(bag),
                "--recording",
                recording,
                "--calibration-dir",
                str(args.calibration_dir),
                "--pairs",
                "camera-camera",
                "--inject",
                "cam3:y:1.0",
                "--inject",
                "cam1:y:1.0",
                "--download-record",
                str(bag.with_name(bag.name + ".download.json")),
                "--output-dir",
                str(args.output_dir / recording),
            ],
            args.work_dir / f"{recording}_run.log",
        )
        (args.output_dir / recording).mkdir(parents=True, exist_ok=True)
        download_record = bag.with_name(bag.name + ".download.json")
        if download_record.exists():
            (args.output_dir / recording / "download.json").write_bytes(
                download_record.read_bytes()
            )
        bag.unlink(missing_ok=True)


def camera_imu(args: argparse.Namespace, plan: dict[str, Any], record: Record) -> None:
    recordings = plan["recordings"]["evaluation"][args.claim]
    log_dir = args.output_dir if args.claim == "camera-imu-v3" else args.work_dir
    bags = plan["recordings"]["bags"]
    chain = args.calibration_dir / "calib_3_cam0-1-camchain-imucam.yaml"
    for recording in recordings:
        outputs = [args.output_dir / f"{camera}_{recording}.yaml" for camera in ("cam0", "cam1")]
        done_marker = args.output_dir / f"{recording}.done"
        if done_marker.exists():
            continue
        bag = args.work_dir / bags[recording]["file"]
        log = args.work_dir / f"{recording}_download.log"
        if download(record, bags[recording]["file"], bag, False, log) != 0:
            print(f"{recording}: download failed or hash mismatch; its units fail", flush=True)
            bag.unlink(missing_ok=True)
            done_marker.write_text("download failed\n", encoding="utf-8")
            continue
        downloaded = json.loads(bag.with_name(bag.name + ".download.json").read_text())
        assert downloaded["sha256"] == bags[recording]["lfs_sha256"], recording
        (args.output_dir / f"{recording}.download.json").write_text(
            json.dumps(downloaded, indent=2) + "\n", encoding="utf-8"
        )
        threads = []
        for camera, output in zip(("cam0", "cam1"), outputs, strict=True):
            command = [
                str(Path(sys.executable).with_name("calibrex")),
                "camera-imu",
                "rotation",
                str(bag),
                "--camchain",
                str(chain),
                "--camera",
                camera,
                "--image-topic",
                f"/alphasense/{camera}/image_raw",
                *CAMERA_IMU_COMMAND,
                "--dataset-license",
                LICENSE,
                "--output",
                str(output),
            ]
            log_path = log_dir / f"{camera}_{recording}.log"
            label = f"camera-imu {camera} {recording}"
            thread = threading.Thread(target=record.run, args=(label, command, log_path))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        bag.unlink(missing_ok=True)
        done_marker.write_text("runs finished\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("claim", choices=tuple(PREREGISTRATIONS))
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = yaml.safe_load(PREREGISTRATIONS[args.claim].read_text(encoding="utf-8"))
    for name, digest in plan["recordings"]["calibration_sha256"].items():
        if sha256_file(args.calibration_dir / name) != digest:
            parser.error(f"{name} does not match the pre-registered SHA-256")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = Record(args.work_dir / "evaluate_record.json")
    if args.claim == "camera-camera":
        camera_camera(args, plan, record)
    else:
        camera_imu(args, plan, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
