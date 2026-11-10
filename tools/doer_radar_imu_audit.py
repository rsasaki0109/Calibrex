#!/usr/bin/env python3
"""The pre-registered radar-imu check audit on Doer & Trommer's radar-inertial recordings.

An evaluation harness, not part of the Calibrex package. Each recording carries a TI
IWR6843AOP radar (``sensor_msgs/PointCloud2`` with ``v_doppler_mps``) and an ADIS16448
IMU; the authors' radar-to-IMU calibration (a translation and a Hamilton quaternion,
``q_b_r``) is the reference. For every bag the harness runs ``calibrex check --pairs
radar-imu --fail-on partial`` with four candidates written from that reference (as
calibrated, and with ``T_radar_imu`` turned 3 deg about the radar's x, y and z axes),
sharing one estimator cache, and records the radar-imu pair record and the exit status.

Two sources:

* ``--bags DIR``: every ``*.bag`` below a local folder (the development run on the
  ICINS 2021 recordings);
* ``--archive``: the IROS 2021 archive, pinned by its Google Drive id and size and read
  over HTTP range requests (it does not fit on the disk); each bag member is unpacked
  alone, checked, and deleted.

::

    python tools/doer_radar_imu_audit.py --calib CALIB --bags DIR --work-dir WORK --output-dir RUNS
    python tools/doer_radar_imu_audit.py --calib CALIB --archive --work-dir WORK --output-dir RUNS

See docs/benchmarks/doer_radar_imu_preregistration.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrex.check.verdict import check_exit_status  # noqa: E402

ARCHIVE_ID = "11d3Lt0wBUGz1Psp7rbS27YBGGveK2OCD"
ARCHIVE_NAME = "irs_rtvi_datasets_2021.zip"
ARCHIVE_SIZE = 9772383980
ARCHIVE_PAGE = "https://drive.usercontent.google.com/download?id={id}&export=download"
EXPECTED_ARCHIVE_BAGS = 9
RADAR_TOPIC = "/sensor_platform/radar/scan"
IMU_TOPIC = "/sensor_platform/imu"
PAIRS = "radar-imu"
FAIL_ON = "partial"
TURN_DEG = 3.0
VARIANTS: dict[str, str | None] = {"reference": None, "rx3": "roll", "ry3": "pitch", "rz3": "yaw"}
AXIS_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}
CHUNK_BYTES = 32 << 20

MAX_REFERENCE_FAILS = 0
MIN_REFERENCE_PASS_FRACTION = 0.25
MAX_TURNED_UNFLAGGED = 0
MAX_JUDGED_TURNS_MISSED = 0
MIN_DETECTING_BAG_FRACTION = 0.25


def reference_rotation(calib: dict[str, Any]) -> Rotation:
    """``R_b_r``: radar vectors into the IMU (body) frame."""

    return Rotation.from_quat(
        [calib["q_b_r_x"], calib["q_b_r_y"], calib["q_b_r_z"], calib["q_b_r_w"]]
    )


def write_candidates(calib_path: Path, directory: Path) -> dict[str, Path]:
    """The reference and three turned copies as ``slac.check_frames`` files (T_imu_radar).

    A turn of ``d`` about the radar axis ``k`` is ``T_radar_imu' = R_delta T_radar_imu``.
    """

    calib = yaml.safe_load(calib_path.read_text(encoding="utf-8"))
    radar_to_imu = reference_rotation(calib)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, axis in VARIANTS.items():
        rotation = radar_to_imu
        if axis is not None:
            turn = np.zeros(3)
            turn[AXIS_INDEX[axis]] = np.radians(TURN_DEG)
            rotation = (Rotation.from_rotvec(turn) * radar_to_imu.inv()).inv()
        frames = {
            "schema_version": "slac.check_frames/v0.1",
            "frames": [
                {
                    "name": "radar",
                    "parent": "imu",
                    "translation_m": [float(calib[f"l_b_r_{k}"]) for k in "xyz"],
                    "rotation_quat_xyzw": [float(v) for v in rotation.as_quat()],
                }
            ],
        }
        path = directory / f"{name}.yaml"
        path.write_text(yaml.safe_dump(frames, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


def check_command(bag: Path, candidate: Path, output: Path, cache: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import sys; from calibrex.cli.main import main; sys.exit(main())",
        "check", str(bag),
        "--tf", str(candidate),
        "--pairs", PAIRS,
        "--frame-map", f"{RADAR_TOPIC}=radar",
        "--frame-map", f"{IMU_TOPIC}=imu",
        "--cache-dir", str(cache),
        "--fail-on", FAIL_ON,
        "--output", str(output),
    ]  # fmt: skip


def pair_record(artifact: dict[str, Any]) -> dict[str, Any]:
    """The radar-imu pair record of one check run, with its exit status."""

    for pair in artifact.get("pairs", []):
        if pair["pair"] != PAIRS or pair["status"] == "skipped":
            continue
        verdict_given = pair["status"] in ("pass", "warn", "fail")
        axes = pair.get("axes", [])
        partial = int(pair.get("coverage") == "partial")
        return {
            "status": pair["status"],
            "coverage": pair.get("coverage"),
            "reason_code": pair.get("reason_code"),
            "exit_status": check_exit_status(pair["status"], partial, FAIL_ON),
            "judged_axes": [a["name"] for a in axes if verdict_given and a["name"] in AXIS_INDEX],
            "unchecked_axes": [u["name"] for u in pair.get("unchecked_axes", [])],
            "time_offset": pair.get("time_offset"),
            "axes": {
                a["name"]: {
                    "candidate_error": a.get("candidate_error"),
                    "estimate_std": a.get("estimate_std"),
                    "tolerance": a.get("tolerance"),
                    "status": a.get("status"),
                }
                for a in axes
            },
        }
    return {}


def passes(record: dict[str, Any]) -> bool:
    return record.get("status") == "pass"


def detected(record: dict[str, Any], axis: str) -> bool:
    """A turned case is detected when the pair fails on the axis that carries the error."""

    return record.get("status") == "fail" and record["axes"].get(axis, {}).get("status") == "fail"


def judged_miss(record: dict[str, Any], axis: str) -> bool:
    """A turned case whose turned axis was judged but did not fail."""

    return axis in record.get("judged_axes", []) and not detected(record, axis)


def verdicts(scores: dict[str, dict[str, Any]], expected_bags: int) -> dict[str, Any]:
    """The two pre-registered claims over every (bag, candidate) case."""

    reference = [scores[b].get("reference", {}) for b in scores]
    turned = [(scores[b].get(v, {}), axis) for b in scores for v, axis in VARIANTS.items() if axis]
    fails = sum(r.get("status") == "fail" for r in reference)
    pass_fraction = sum(passes(r) for r in reference) / expected_bags
    unflagged = sum(r.get("exit_status") == 0 for r, _axis in turned if r)
    missed = sum(judged_miss(r, axis) for r, axis in turned if r)
    detecting = sum(
        any(detected(scores[b].get(v, {}), axis) for v, axis in VARIANTS.items() if axis)
        for b in scores
        if scores[b].get("reference")
    )
    complete = len(reference) == expected_bags and all(reference) and all(r for r, _axis in turned)
    honest = complete and fails <= MAX_REFERENCE_FAILS
    honest = honest and pass_fraction >= MIN_REFERENCE_PASS_FRACTION
    detects = complete and unflagged <= MAX_TURNED_UNFLAGGED and missed <= MAX_JUDGED_TURNS_MISSED
    detects = detects and detecting / expected_bags >= MIN_DETECTING_BAG_FRACTION
    return {
        "honesty": {
            "verdict": "supported" if honest else "refuted",
            "reference_fails": fails,
            "reference_pass_fraction": pass_fraction,
        },
        "detection": {
            "verdict": "supported" if detects else "refuted",
            "turned_unflagged": unflagged,
            "judged_turns_missed": missed,
            "detecting_bag_fraction": detecting / expected_bags,
        },
        "complete": complete,
    }


def run_bag(name: str, bag: Path, candidates: dict[str, Path], work: Path, runs: Path) -> dict:
    cache = work / f"cache-{name}"
    result: dict[str, Any] = {"exit_status": {}}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for variant, candidate in candidates.items():
        output = runs / f"{name}_{variant}.json"
        completed = subprocess.run(
            check_command(bag, candidate, output, cache), capture_output=True, text=True, env=env
        )
        (runs / f"{name}_{variant}.log").write_text(completed.stdout + completed.stderr)
        result["exit_status"][variant] = completed.returncode
        artifact = json.loads(output.read_text()) if output.exists() else {"pairs": []}
        result[variant] = pair_record(artifact)
    shutil.rmtree(cache, ignore_errors=True)
    (runs / f"{name}_score.json").write_text(json.dumps(result, indent=1, default=float))
    return result


class RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a pinned URL through HTTP range requests (curl)."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.position = url, size, 0
        self._start, self._data = -1, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = base + offset
        return self.position

    def _fetch(self, start: int) -> None:
        end = min(start + CHUNK_BYTES, self.size) - 1
        for _attempt in range(10):
            done = subprocess.run(
                ["curl", "-sSfL", "-r", f"{start}-{end}", self.url], capture_output=True
            )
            if done.returncode == 0 and len(done.stdout) == end - start + 1:
                self._start, self._data = start, done.stdout
                return
        raise OSError(f"range {start}-{end} of the archive could not be read")

    def read(self, size: int = -1) -> bytes:
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        out = bytearray()
        while size > 0:
            if not self._start <= self.position < self._start + len(self._data):
                self._fetch(self.position)
            chunk = self._data[self.position - self._start : self.position - self._start + size]
            out += chunk
            self.position += len(chunk)
            size -= len(chunk)
        return bytes(out)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def archive_url() -> str:
    """The direct download URL of the pinned archive (past Google Drive's scan notice)."""

    page = subprocess.run(
        ["curl", "-sSfL", ARCHIVE_PAGE.format(id=ARCHIVE_ID)], capture_output=True, text=True
    ).stdout
    found = re.search(r'name="uuid" value="([^"]+)"', page)
    uuid = f"&uuid={found.group(1)}" if found else ""
    return ARCHIVE_PAGE.format(id=ARCHIVE_ID) + f"&confirm=t{uuid}"


def archive_bags() -> tuple[zipfile.ZipFile, list[zipfile.ZipInfo]]:
    remote = RemoteFile(archive_url(), ARCHIVE_SIZE)
    head = remote.read(4)
    remote.seek(0)
    if head != b"PK\x03\x04":
        raise SystemExit("the archive URL did not serve the pinned zip")
    archive = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=CHUNK_BYTES))
    members = sorted(
        (item for item in archive.infolist() if item.filename.endswith(".bag")),
        key=lambda item: item.filename,
    )
    return archive, members


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calib", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bags", type=Path)
    source.add_argument("--archive", action="store_true")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calib_sha256 = hashlib.sha256(args.calib.read_bytes()).hexdigest()
    candidates = write_candidates(args.calib, args.output_dir / "candidates")
    scores: dict[str, dict[str, Any]] = {}
    if args.bags is not None:
        bags = sorted(args.bags.rglob("*.bag"))
        expected = len(bags)
        for bag in bags:
            scores[bag.stem] = run_bag(bag.stem, bag, candidates, args.work_dir, args.output_dir)
    else:
        archive, members = archive_bags()
        expected = EXPECTED_ARCHIVE_BAGS
        for member in members:
            name = Path(member.filename).stem
            target = args.work_dir / f"{name}.bag"
            try:
                with archive.open(member) as stream, target.open("wb") as out:
                    shutil.copyfileobj(stream, out, CHUNK_BYTES)
                scores[name] = run_bag(name, target, candidates, args.work_dir, args.output_dir)
            except Exception as error:  # a bag that cannot be run leaves its cases empty
                scores[name] = {"error": repr(error), **{v: {} for v in VARIANTS}}
            target.unlink(missing_ok=True)
    summary = {
        "calib_sha256": calib_sha256,
        "bags": list(scores),
        "scores": scores,
        "verdicts": verdicts(scores, expected),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    shutil.rmtree(args.work_dir, ignore_errors=True)
    print(json.dumps(summary["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
