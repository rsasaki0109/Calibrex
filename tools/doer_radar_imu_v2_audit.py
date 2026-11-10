#!/usr/bin/env python3
"""The pre-registered radar-imu check audit v2: longer multi-radar recordings.

An evaluation harness, not part of the Calibrex package. v1
(``tools/doer_radar_imu_audit.py``, the IROS 2021 archive) was refuted on coverage: on
recordings of 84-206 s, 8 of 9 bags judged no axis. v2 runs the same frozen method on
Doer & Trommer's JGN 2022 multi-radar archive (three TI IWR6843AOP radars, left, right
and centre, and an ADIS16448 IMU; four recordings of 219-368 s).

For every bag it runs ``calibrex check --pairs radar-imu --fail-on partial`` with four
candidates written from the authors' calibration of the three radars (as calibrated, and
with every radar's ``T_radar_imu`` turned 3 deg about its own x, y and z axes), sharing
one estimator cache, and scores each radar's pair record.

Two sources:

* ``--bags DIR --calib NAME=PATH ...``: every ``*.bag`` below a local folder, with the
  named calibration files (the development run);
* ``--archive``: the JGN 2022 archive, pinned by its Google Drive id and size, read over
  HTTP range requests; its ``calib_*.yaml`` files are the reference and each bag member
  is unpacked alone, checked, and deleted.

::

    python tools/doer_radar_imu_v2_audit.py --archive --work-dir WORK --output-dir RUNS

See docs/benchmarks/doer_radar_imu_v2_preregistration.yaml.
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
sys.path.insert(0, str(ROOT))

from tools.doer_radar_imu_audit import RemoteFile  # noqa: E402

from calibrex.check.verdict import check_exit_status  # noqa: E402

ARCHIVE_ID = "1zq9g0S9Rn6jrlmgwsmzBNgnSFr-m9NqL"
ARCHIVE_NAME = "multi_radar_inertial_datasets_jgn_2022.zip"
ARCHIVE_SIZE = 135085976
ARCHIVE_PAGE = "https://drive.usercontent.google.com/download?id={id}&export=download"
EXPECTED_BAGS = 4
EXPECTED_RADARS = 3
IMU_TOPIC = "/sensor_platform/imu"
PAIRS = "radar-imu"
FAIL_ON = "partial"
TURN_DEG = 3.0
VARIANTS: dict[str, str | None] = {"reference": None, "rx3": "roll", "ry3": "pitch", "rz3": "yaw"}
AXIS_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}
CHUNK_BYTES = 32 << 20
_CALIB = re.compile(r"calib_([A-Za-z0-9]+)\.yaml$")

MAX_REFERENCE_FAILS = 0
MIN_REFERENCE_PASS_FRACTION = 0.25
MIN_REFERENCE_JUDGED_FRACTION = 0.5
MAX_TURNED_UNFLAGGED = 0
MAX_JUDGED_TURNS_MISSED = 0
MIN_DETECTING_CASE_FRACTION = 0.5


def radar_frame(name: str) -> str:
    return f"radar_{name}"


def write_candidates(calibs: dict[str, dict[str, Any]], directory: Path) -> dict[str, Path]:
    """The reference and three turned copies as ``slac.check_frames`` files (T_imu_radar).

    A turn of ``d`` about a radar's axis ``k`` is ``T_radar_imu' = R_delta T_radar_imu``,
    applied to every radar at once.
    """

    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for variant, axis in VARIANTS.items():
        frames = []
        for name, calib in sorted(calibs.items()):
            radar_to_imu = Rotation.from_quat(
                [calib["q_b_r_x"], calib["q_b_r_y"], calib["q_b_r_z"], calib["q_b_r_w"]]
            )
            rotation = radar_to_imu
            if axis is not None:
                turn = np.zeros(3)
                turn[AXIS_INDEX[axis]] = np.radians(TURN_DEG)
                rotation = (Rotation.from_rotvec(turn) * radar_to_imu.inv()).inv()
            frames.append(
                {
                    "name": radar_frame(name),
                    "parent": "imu",
                    "translation_m": [float(calib[f"l_b_r_{k}"]) for k in "xyz"],
                    "rotation_quat_xyzw": [float(v) for v in rotation.as_quat()],
                }
            )
        path = directory / f"{variant}.yaml"
        path.write_text(
            yaml.safe_dump({"schema_version": "slac.check_frames/v0.1", "frames": frames}),
            encoding="utf-8",
        )
        paths[variant] = path
    return paths


def check_command(
    bag: Path, candidate: Path, output: Path, cache: Path, calibs: dict[str, dict[str, Any]]
) -> list[str]:
    frame_maps: list[str] = []
    for name, calib in sorted(calibs.items()):
        frame_maps += ["--frame-map", f"{calib['topic_radar_scan']}={radar_frame(name)}"]
    return [
        sys.executable,
        "-c",
        "import sys; from calibrex.cli.main import main; sys.exit(main())",
        "check", str(bag),
        "--tf", str(candidate),
        "--pairs", PAIRS,
        *frame_maps,
        "--frame-map", f"{IMU_TOPIC}=imu",
        "--cache-dir", str(cache),
        "--fail-on", FAIL_ON,
        "--output", str(output),
    ]  # fmt: skip


def pair_records(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each radar's radar-imu pair record, keyed by the radar's frame, with its exit status."""

    records: dict[str, dict[str, Any]] = {}
    for pair in artifact.get("pairs", []):
        if pair["pair"] != PAIRS or pair["status"] == "skipped":
            continue
        verdict_given = pair["status"] in ("pass", "warn", "fail")
        axes = pair.get("axes", [])
        partial = int(pair.get("coverage") == "partial")
        records[pair["sensors"][0]] = {
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
    return records


def detected(record: dict[str, Any], axis: str) -> bool:
    """A turned case is detected when the pair fails on the axis that carries the error."""

    return record.get("status") == "fail" and record["axes"].get(axis, {}).get("status") == "fail"


def verdicts(scores: dict[str, dict[str, Any]], radars: list[str]) -> dict[str, Any]:
    """The three pre-registered claims over every (bag, radar, candidate) case."""

    frames = [radar_frame(name) for name in radars]
    expected = EXPECTED_BAGS * EXPECTED_RADARS
    reference = [scores[b].get("reference", {}).get(f, {}) for b in scores for f in frames]
    turned = [
        (scores[b].get(v, {}).get(f, {}), axis)
        for b in scores
        for v, axis in VARIANTS.items()
        if axis
        for f in frames
    ]
    fails = sum(r.get("status") == "fail" for r in reference)
    passed = sum(r.get("status") == "pass" for r in reference) / expected
    judged = sum(bool(r.get("judged_axes")) for r in reference) / expected
    unflagged = sum(r.get("exit_status") == 0 for r, _axis in turned if r)
    missed = sum(
        axis in r.get("judged_axes", []) and not detected(r, axis) for r, axis in turned if r
    )
    detecting = (
        sum(
            any(
                detected(scores[b].get(v, {}).get(f, {}), axis)
                for v, axis in VARIANTS.items()
                if axis
            )
            for b in scores
            for f in frames
        )
        / expected
    )
    complete = (
        len(scores) == EXPECTED_BAGS
        and len(radars) == EXPECTED_RADARS
        and len(reference) == expected
        and all(reference)
        and all(r for r, _axis in turned)
    )
    honest = complete and fails <= MAX_REFERENCE_FAILS and passed >= MIN_REFERENCE_PASS_FRACTION
    covered = complete and judged >= MIN_REFERENCE_JUDGED_FRACTION
    detects = complete and unflagged <= MAX_TURNED_UNFLAGGED and missed <= MAX_JUDGED_TURNS_MISSED
    detects = detects and detecting >= MIN_DETECTING_CASE_FRACTION
    return {
        "honesty": {
            "verdict": "supported" if honest else "refuted",
            "reference_fails": fails,
            "reference_pass_fraction": passed,
        },
        "coverage": {
            "verdict": "supported" if covered else "refuted",
            "reference_judged_fraction": judged,
        },
        "detection": {
            "verdict": "supported" if detects else "refuted",
            "turned_unflagged": unflagged,
            "judged_turns_missed": missed,
            "detecting_case_fraction": detecting,
        },
        "complete": complete,
    }


def run_bag(
    name: str,
    bag: Path,
    candidates: dict[str, Path],
    calibs: dict[str, dict[str, Any]],
    work: Path,
    runs: Path,
) -> dict:
    cache = work / f"cache-{name}"
    result: dict[str, Any] = {"exit_status": {}}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for variant, candidate in candidates.items():
        output = runs / f"{name}_{variant}.json"
        completed = subprocess.run(
            check_command(bag, candidate, output, cache, calibs),
            capture_output=True,
            text=True,
            env=env,
        )
        (runs / f"{name}_{variant}.log").write_text(completed.stdout + completed.stderr)
        result["exit_status"][variant] = completed.returncode
        artifact = json.loads(output.read_text()) if output.exists() else {"pairs": []}
        result[variant] = pair_records(artifact)
    shutil.rmtree(cache, ignore_errors=True)
    (runs / f"{name}_score.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def archive_url() -> str:
    """The direct download URL of the pinned archive (past Google Drive's scan notice)."""

    page = subprocess.run(
        ["curl", "-sSfL", ARCHIVE_PAGE.format(id=ARCHIVE_ID)], capture_output=True, text=True
    ).stdout
    found = re.search(r'name="uuid" value="([^"]+)"', page)
    uuid = f"&uuid={found.group(1)}" if found else ""
    return ARCHIVE_PAGE.format(id=ARCHIVE_ID) + f"&confirm=t{uuid}"


def open_archive() -> zipfile.ZipFile:
    remote = RemoteFile(archive_url(), ARCHIVE_SIZE)
    if remote.read(4) != b"PK\x03\x04":
        raise SystemExit("the archive URL did not serve the pinned zip")
    remote.seek(0)
    return zipfile.ZipFile(io.BufferedReader(remote, buffer_size=CHUNK_BYTES))


def top_level(member: zipfile.ZipInfo) -> bool:
    """A member directly in the archive's top folder (not in the results folders)."""

    return len(Path(member.filename).parts) == 2


def archive_members(archive: zipfile.ZipFile) -> tuple[dict[str, bytes], list[zipfile.ZipInfo]]:
    """The top folder's ``calib_NAME.yaml`` texts by NAME, and its bags in name order."""

    texts: dict[str, bytes] = {}
    bags: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        if not top_level(member):
            continue
        found = _CALIB.search(member.filename)
        if found:
            texts[found.group(1)] = archive.read(member)
        elif member.filename.endswith(".bag"):
            bags.append(member)
    return texts, sorted(bags, key=lambda item: item.filename)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bags", type=Path)
    source.add_argument("--archive", action="store_true")
    parser.add_argument("--calib", action="append", default=[], help="NAME=PATH (with --bags)")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts: dict[str, bytes] = {}
    archive: zipfile.ZipFile | None = None
    bag_members: list[zipfile.ZipInfo] = []
    if args.archive:
        archive = open_archive()
        texts, bag_members = archive_members(archive)
    else:
        for item in args.calib:
            name, _, path = item.partition("=")
            texts[name] = Path(path).read_bytes()
    calibs = {name: yaml.safe_load(text.decode("utf-8")) for name, text in texts.items()}
    calib_sha256 = {name: hashlib.sha256(text).hexdigest() for name, text in texts.items()}
    candidates = write_candidates(calibs, args.output_dir / "candidates")
    scores: dict[str, dict[str, Any]] = {}
    if archive is None:
        for bag in sorted(args.bags.rglob("*.bag")):
            scores[bag.stem] = run_bag(
                bag.stem, bag, candidates, calibs, args.work_dir, args.output_dir
            )
    else:
        for member in bag_members:
            name = Path(member.filename).stem
            target = args.work_dir / f"{name}.bag"
            try:
                with archive.open(member) as stream, target.open("wb") as out:
                    shutil.copyfileobj(stream, out, CHUNK_BYTES)
                scores[name] = run_bag(
                    name, target, candidates, calibs, args.work_dir, args.output_dir
                )
            except Exception as error:  # a bag that cannot be run leaves its cases empty
                scores[name] = {"error": repr(error), **{v: {} for v in VARIANTS}}
            target.unlink(missing_ok=True)
    radars = sorted(calibs)
    summary = {
        "calib_sha256": calib_sha256,
        "radars": radars,
        "bags": list(scores),
        "scores": scores,
        "verdicts": verdicts(scores, radars),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    shutil.rmtree(args.work_dir, ignore_errors=True)
    print(json.dumps(summary["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
