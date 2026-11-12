#!/usr/bin/env python3
"""The pre-registered imu-lidar check audit on Snail-Radar (hand-held, e-bike and SUV).

An evaluation harness, not part of the Calibrex package. Snail-Radar (Huai et al., IJRR;
Open Database License) records a Hesai Pandar XT32 LiDAR with a Bynav X36D IMU on three
platforms, and from November 2023 also an Xsens MTi-3 IMU. Its per-sequence folder zips
on OneDrive are read over HTTP range requests; only ``xt32/*.pcd``, ``x36d/imu.txt``,
``mti3dk/imu.txt`` and ``tf_static.launch`` are taken out.

Each sequence is rebuilt as a ROS 2 bag:

* ``/hesai/pandar`` (``PandarXT-32``): every ``POINT_STRIDE``-th firing of each scan (all
  32 rings of a firing kept), with x, y, z and the per-point absolute ``timestamp``;
* ``/x36d/imu_raw`` (``x36d``) and ``/mti3dk/imu`` (``mti3dk``) as recorded;

and only its first ``MAX_DURATION_S`` seconds of scans (with the IMU samples up to one
second past the last scan) are written.

``calibrex check --pairs imu-lidar --fail-on partial`` then runs with four candidates
from the sequence's ``tf_static.launch`` (as calibrated, and with the LiDAR turned 3 deg
about its own x, y and z axes, which turns its pair with every IMU), sharing one
estimator cache, and each (sequence, IMU) pair record is scored.

::

    python tools/snail_lidar_imu_audit.py --sequences 20230921/data4 --work-dir W --output-dir R

See docs/benchmarks/snail_lidar_imu_preregistration.yaml.
"""

from __future__ import annotations

import argparse
import base64
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

SHARE_URL = "https://1drv.ms/f/c/60208caf9367dbb1/ErHbZ5OvjCAggGAHDQAAAAABCBUW1vutI7GYt95u7EB-Mg"
API = "https://my.microsoftpersonalcontent.com/_api/v2.0"
TOKEN_URL = "https://api-badgerp.svc.ms/v1.0/token"
TOKEN_APP = "5cbed6ac-a083-4e14-b191-b4ba07653de2"  # OneDrive's public web app id
LIDAR_TOPIC = "/hesai/pandar"
LIDAR_FRAME = "PandarXT-32"
IMUS = {"/x36d/imu_raw": ("x36d", "x36d/imu.txt"), "/mti3dk/imu": ("mti3dk", "mti3dk/imu.txt")}
POINT_STRIDE = 4
MAX_DURATION_S = 300.0  # the first 300 s of every sequence
PAIRS = "imu-lidar"
FAIL_ON = "partial"
TURN_DEG = 3.0
VARIANTS: dict[str, str | None] = {"reference": None, "rx3": "roll", "ry3": "pitch", "rz3": "yaw"}
AXIS_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}
CHUNK_BYTES = 32 << 20

# Gates (docs/benchmarks/snail_lidar_imu_preregistration.yaml); a case is one
# (sequence, IMU) pair.
MAX_REFERENCE_ROTATION_FAILS = 0
MIN_REFERENCE_PASS_FRACTION = 0.0
MIN_REFERENCE_JUDGED_FRACTION = 0.0
MAX_JUDGED_TURNS_MISSED = 0
MIN_DETECTING_CASE_FRACTION = 0.0
_TF = re.compile(r'args="([^"]+)"')


def _curl_json(args: list[str]) -> Any:
    done = subprocess.run(["curl", "-sSf", *args], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def onedrive_item(path: str) -> dict[str, Any]:
    """A fresh drive item (with its short-lived download URL) of the share, by path."""

    token = _curl_json(
        ["-X", "POST", TOKEN_URL, "-H", "Content-Type: application/json",
         "-d", json.dumps({"appId": TOKEN_APP})]
    )["token"]  # fmt: skip
    head = ["-H", f"Authorization: Badger {token}", "-H", "Prefer: autoredeem"]
    share = base64.urlsafe_b64encode(SHARE_URL.encode()).decode().rstrip("=")
    node = _curl_json([*head, f"{API}/shares/u!{share}/driveitem"])
    for part in path.split("/"):
        drive = node["parentReference"]["driveId"]
        children = _curl_json([*head, f"{API}/drives/{drive}/items/{node['id']}/children?$top=999"])
        node = next(child for child in children["value"] if child["name"] == part)
    return node


def open_sequence(sequence: str) -> tuple[zipfile.ZipFile, int]:
    """The sequence's folder zip (``DAY/dataN.zip``) over range requests, and its size."""

    item = onedrive_item(f"{sequence}.zip")
    remote = RemoteFile(item["@content.downloadUrl"], int(item["size"]))
    if remote.read(4) != b"PK\x03\x04":
        raise SystemExit(f"{sequence}.zip is not a zip")
    remote.seek(0)
    return zipfile.ZipFile(io.BufferedReader(remote, buffer_size=CHUNK_BYTES)), int(item["size"])


def static_transforms(text: str) -> dict[str, tuple[np.ndarray, Rotation]]:
    """``T_parent_child`` of every ``static_transform_publisher`` line, by child frame."""

    found: dict[str, tuple[np.ndarray, Rotation]] = {}
    for line in text.splitlines():
        match = _TF.search(line)
        if not match:
            continue
        args = match.group(1).split()
        values = [float(v) for v in args[:7]]
        found[args[8].lstrip("/")] = (np.array(values[:3]), Rotation.from_quat(values[3:7]))
    return found


def _pcd_points(raw: bytes) -> np.ndarray:
    """x, y, z (float32) and the absolute timestamp (float64) of a binary XT32 PCD."""

    end = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    header = raw[:end].decode("ascii")
    names = re.search(r"FIELDS (.*)", header).group(1).split()  # type: ignore[union-attr]
    sizes = [int(v) for v in re.search(r"SIZE (.*)", header).group(1).split()]  # type: ignore[union-attr]
    kinds = re.search(r"TYPE (.*)", header).group(1).split()  # type: ignore[union-attr]
    count = int(re.search(r"POINTS (\d+)", header).group(1))  # type: ignore[union-attr]
    dtype = np.dtype([(n, f"<{k.lower()}{s}") for n, s, k in zip(names, sizes, kinds, strict=True)])
    return np.frombuffer(raw[end : end + dtype.itemsize * count], dtype)


POINT_FIELDS = [("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1), ("timestamp", 12, 8, 1)]
POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("timestamp", "<f8")])


def thinned_scan(points: np.ndarray) -> np.ndarray:
    """Every ``POINT_STRIDE``-th firing (points sharing one timestamp) of a scan."""

    _, firing = np.unique(points["timestamp"], return_inverse=True)
    kept = points[(firing % POINT_STRIDE) == 0]
    out = np.empty(len(kept), POINT_DTYPE)
    for name in POINT_DTYPE.names or ():
        out[name] = kept[name]
    return out


def _imu_rows(text: bytes) -> np.ndarray:
    rows = np.loadtxt(io.StringIO(text.decode("ascii")), ndmin=2)
    return rows[:, :7]  # time, ax, ay, az, gx, gy, gz


def build_bag(archive: zipfile.ZipFile, bag: Path) -> dict[str, Any]:
    """Rebuild the sequence's LiDAR and IMU streams as a ROS 2 bag; returns what was read."""

    from calibrex.data.ros_cdr_writer import encode_imu, encode_pointcloud2
    from calibrex.data.rosbag2_writer import Rosbag2Writer

    names = {item.filename: item for item in archive.infolist()}
    imus = {
        topic: _imu_rows(archive.read(names[member]))
        for topic, (_frame, member) in IMUS.items()
        if member in names
    }
    scans = sorted(
        (item for name, item in names.items() if re.fullmatch(r"xt32/[0-9.]+\.pcd", name)),
        key=lambda item: item.header_offset,
    )
    imu_events = sorted(
        (float(row[0]), topic, index)
        for topic, rows in imus.items()
        for index, row in enumerate(rows)
    )
    first_s = float(Path(scans[0].filename).stem)
    scans = [item for item in scans if float(Path(item.filename).stem) <= first_s + MAX_DURATION_S]
    last_s = float(Path(scans[-1].filename).stem)
    imu_events = [event for event in imu_events if event[0] <= last_s + 1.0]
    cursor = 0
    written = 0
    with Rosbag2Writer(bag) as writer:
        writer.add_topic(LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2")
        for topic in imus:
            writer.add_topic(topic, "sensor_msgs/msg/Imu")

        def flush_imu(until_s: float) -> None:
            nonlocal cursor
            while cursor < len(imu_events) and imu_events[cursor][0] <= until_s:
                stamp_s, topic, index = imu_events[cursor]
                row = imus[topic][index]
                ns = round(stamp_s * 1e9)
                writer.write(
                    topic,
                    ns,
                    encode_imu(
                        frame_id=IMUS[topic][0],
                        timestamp_ns=ns,
                        angular_velocity=(float(row[4]), float(row[5]), float(row[6])),
                        linear_acceleration=(float(row[1]), float(row[2]), float(row[3])),
                    ),
                )
                cursor += 1

        for item in scans:
            stamp_s = float(Path(item.filename).stem)
            flush_imu(stamp_s)
            points = thinned_scan(_pcd_points(archive.read(item)))
            ns = round(stamp_s * 1e9)
            writer.write(
                LIDAR_TOPIC,
                ns,
                encode_pointcloud2(
                    frame_id=LIDAR_FRAME,
                    timestamp_ns=ns,
                    fields=POINT_FIELDS,
                    point_step=POINT_DTYPE.itemsize,
                    data=points.tobytes(),
                ),
            )
            written += 1
        flush_imu(float("inf"))
    counts = {topic: sum(event[1] == topic for event in imu_events) for topic in imus}
    return {"scans": written, "seconds": last_s - first_s, "imu_samples": counts}


def write_candidates(
    transforms: dict[str, tuple[np.ndarray, Rotation]], imu_frames: list[str], directory: Path
) -> dict[str, Path]:
    """The reference and three turned copies as ``slac.check_frames`` files.

    Every frame hangs off ``base_link`` as in ``tf_static.launch``. A turn of ``d`` about
    the LiDAR's axis ``k`` is ``T_base_lidar' = T_base_lidar R_delta``: it turns the
    LiDAR against every IMU at once.
    """

    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for variant, axis in VARIANTS.items():
        frames = []
        for frame in [LIDAR_FRAME, *imu_frames]:
            translation, rotation = transforms[frame]
            if frame == LIDAR_FRAME and axis is not None:
                turn = np.zeros(3)
                turn[AXIS_INDEX[axis]] = np.radians(TURN_DEG)
                rotation = rotation * Rotation.from_rotvec(turn)
            frames.append(
                {
                    "name": frame,
                    "parent": "base_link",
                    "translation_m": [float(v) for v in translation],
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
    bag: Path, candidate: Path, output: Path, cache: Path, imu_topics: list[str]
) -> list[str]:
    frame_maps = ["--frame-map", f"{LIDAR_TOPIC}={LIDAR_FRAME}"]
    for topic in imu_topics:
        frame_maps += ["--frame-map", f"{topic}={IMUS[topic][0]}"]
    return [
        sys.executable,
        "-c",
        "import sys; from calibrex.cli.main import main; sys.exit(main())",
        "check", str(bag),
        "--tf", str(candidate),
        "--pairs", PAIRS,
        *frame_maps,
        "--max-duration-s", f"{MAX_DURATION_S:g}",
        "--cache-dir", str(cache),
        "--fail-on", FAIL_ON,
        "--output", str(output),
    ]  # fmt: skip


def pair_records(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each IMU's imu-lidar pair record, keyed by the IMU's frame, with its exit status."""

    records: dict[str, dict[str, Any]] = {}
    for pair in artifact.get("pairs", []):
        if pair["pair"] != PAIRS or pair["status"] == "skipped":
            continue
        verdict_given = pair["status"] in ("pass", "warn", "fail")
        axes = pair.get("axes", [])
        partial = int(pair.get("coverage") == "partial")
        imu = next((s for s in pair["sensors"] if s != LIDAR_FRAME), pair["sensors"][0])
        records[imu] = {
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


def rotation_failed(record: dict[str, Any]) -> bool:
    """The pair fails a rotation axis of the candidate (translation is not scored)."""

    return any(record["axes"].get(axis, {}).get("status") == "fail" for axis in AXIS_INDEX)


def rotation_passed(record: dict[str, Any]) -> bool:
    """At least one rotation axis is judged and every judged rotation axis passes."""

    judged = record.get("judged_axes", [])
    return bool(judged) and all(record["axes"][axis]["status"] == "pass" for axis in judged)


def cases(scores: dict[str, dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    """Every (sequence, IMU) record of one candidate; an IMU the sequence has but the
    check did not report counts as an empty record."""

    found = []
    for score in scores.values():
        for imu in score.get("imus", []):
            found.append(score.get(variant, {}).get(imu, {}))
    return found


def verdicts(scores: dict[str, dict[str, Any]], expected_cases: int) -> dict[str, Any]:
    """The three pre-registered claims over every (sequence, IMU, candidate) case."""

    reference = cases(scores, "reference")
    fails = sum(rotation_failed(r) for r in reference if r)
    passed = sum(rotation_passed(r) for r in reference if r) / max(expected_cases, 1)
    judged = sum(bool(r.get("judged_axes")) for r in reference if r) / max(expected_cases, 1)
    missed = 0
    detecting_cases = 0
    for score in scores.values():
        for imu in score.get("imus", []):
            hit = False
            for variant, axis in VARIANTS.items():
                record = score.get(variant, {}).get(imu, {})
                if axis is None or not record:
                    continue
                if axis in record.get("judged_axes", []) and not detected(record, axis):
                    missed += 1
                hit = hit or detected(record, axis)
            detecting_cases += hit
    detecting = detecting_cases / max(expected_cases, 1)
    complete = len(reference) == expected_cases and all(
        cases(scores, variant) and all(cases(scores, variant)) for variant in VARIANTS
    )
    honest = complete and fails <= MAX_REFERENCE_ROTATION_FAILS
    honest = honest and passed >= MIN_REFERENCE_PASS_FRACTION
    covered = complete and judged >= MIN_REFERENCE_JUDGED_FRACTION
    detects = complete and missed <= MAX_JUDGED_TURNS_MISSED
    detects = detects and detecting >= MIN_DETECTING_CASE_FRACTION
    return {
        "honesty": {
            "verdict": "supported" if honest else "refuted",
            "reference_rotation_fails": fails,
            "reference_pass_fraction": passed,
        },
        "coverage": {
            "verdict": "supported" if covered else "refuted",
            "reference_judged_fraction": judged,
        },
        "detection": {
            "verdict": "supported" if detects else "refuted",
            "judged_turns_missed": missed,
            "detecting_case_fraction": detecting,
        },
        "cases": len(reference),
        "expected_cases": expected_cases,
        "complete": complete,
    }


def run_sequence(sequence: str, work: Path, runs: Path) -> dict[str, Any]:
    name = sequence.replace("/", "_")
    archive, size = open_sequence(sequence)
    transforms = static_transforms(archive.read("tf_static.launch").decode("utf-8"))
    bag = work / name
    shutil.rmtree(bag, ignore_errors=True)
    built = build_bag(archive, bag)
    imu_topics = [topic for topic in IMUS if topic in built["imu_samples"]]
    candidates = write_candidates(
        transforms, [IMUS[t][0] for t in imu_topics], runs / "candidates" / name
    )
    cache = work / f"cache-{name}"
    result: dict[str, Any] = {
        "zip_bytes": size,
        "built": built,
        "imus": [IMUS[t][0] for t in imu_topics],
        "exit_status": {},
    }
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for variant, candidate in candidates.items():
        output = runs / f"{name}_{variant}.json"
        completed = subprocess.run(
            check_command(bag, candidate, output, cache, imu_topics),
            capture_output=True,
            text=True,
            env=env,
        )
        (runs / f"{name}_{variant}.log").write_text(completed.stdout + completed.stderr)
        result["exit_status"][variant] = completed.returncode
        artifact = json.loads(output.read_text()) if output.exists() else {"pairs": []}
        result[variant] = pair_records(artifact)
    shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(bag, ignore_errors=True)
    (runs / f"{name}_score.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", nargs="+", required=True, help="DAY/dataN ...")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="sequences run at once")
    parser.add_argument(
        "--expected-cases", type=int, default=0, help="(sequence, IMU) cases the plan expects"
    )
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def one(sequence: str) -> dict[str, Any]:
        try:
            return run_sequence(sequence, args.work_dir, args.output_dir)
        except Exception as error:  # a sequence that cannot be run leaves its cases empty
            return {"error": repr(error), **{v: {} for v in VARIANTS}}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        scores = dict(zip(args.sequences, pool.map(one, args.sequences), strict=True))
    expected = args.expected_cases or sum(len(s.get("imus", [])) for s in scores.values())
    summary = {
        "sequences": args.sequences,
        "scores": scores,
        "verdicts": verdicts(scores, expected),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
