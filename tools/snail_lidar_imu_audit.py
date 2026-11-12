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
    python tools/snail_lidar_imu_audit.py --evaluation --jobs 3 --work-dir W --output-dir R

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
# (sequence, IMU) pair.  Fractions are the development rate / 1.5, rounded down to 0.25.
MAX_REFERENCE_ROTATION_FAILS = 0
MIN_REFERENCE_PASS_FRACTION = 0.5  # development 0.75
MIN_REFERENCE_JUDGED_FRACTION = 0.5  # development 1.0
MAX_JUDGED_TURNS_MISSED = 0
MIN_DETECTING_CASE_FRACTION = 0.5  # development 1.0
DEVELOPMENT_SPENT = ("20230921/data4", "20231019/data2", "20240113/data5")
REFERENCE_FITTING = "20230921/data5"  # the X36D-XT32 rotation was fitted on it
MTI_FROM_DAY = "20231105"  # the MTi-3 is recorded from this day on
# The unseen sequences, pinned by the zip's size and OneDrive eTag (metadata only).
EVALUATION: dict[str, tuple[int, str]] = {
    "20230920/data2": (25033261305, '"{34938F8F-6EB5-49B2-AE9D-C818477208EA},6"'),
    "20230921/data3": (23035087302, '"{1F6D146D-4F3B-47B8-9FD2-989A18975973},6"'),
    "20231007/data2": (9972191149, '"{2122AAC9-F795-45E5-B3E9-3FB5637B6B1A},6"'),
    "20231019/data1": (6962331607, '"{BC3B658F-33EC-4927-8770-8EE5F5670537},6"'),
    "20231105/data2": (7102582878, '"{9380060A-3FB4-407A-B19D-0EA6F024603E},6"'),
    "20231105/data3": (7193500349, '"{D83C2EEB-9B70-4DAA-927B-FA6B45D87688},6"'),
    "20231105/data4": (5374912072, '"{49928D35-51F5-4241-AD59-41E4FFBFBF80},6"'),
    "20231105/data5": (5530969277, '"{0E382E09-57DC-42C8-8A0A-49A33D9AF2D7},6"'),
    "20231105_aft/data4": (9287469203, '"{B8F50626-13A7-4924-ABC0-883CF0B5EB01},6"'),
    "20231105_aft/data5": (6582319327, '"{EC237446-290B-491E-AFAB-38852A781709},6"'),
    "20231109/data3": (8053057985, '"{54919921-BA46-4003-B7B4-E135291819C1},6"'),
    "20231109/data4": (8043548274, '"{5E42B5D0-4416-4F18-8534-BD51AB09717B},6"'),
    "20231201/data2": (17075255586, '"{15A81429-478C-46D3-A8EF-B7967B7BB849},6"'),
    "20231201/data3": (17260104425, '"{8F64CB7C-DAE2-4D26-9803-4CE6B9C3321A},6"'),
    "20231208/data4": (8750824258, '"{22EF2DD9-9510-4672-8AE3-4FC064D43062},6"'),
    "20231208/data5": (14442417452, '"{3CEAC9C7-EC6F-44DF-A595-D0C03E02E285},6"'),
    "20231213/data2": (13591333515, '"{BCE20F08-CE94-4E26-8813-4685C677C3B6},6"'),
    "20231213/data3": (11705309080, '"{40C59D03-39B8-4287-8B90-51D19715A12C},6"'),
    "20231213/data4": (6769226080, '"{5B1CE686-EAD6-4FBA-82DB-69A2C66A1FE1},6"'),
    "20231213/data5": (6505811711, '"{ACA63689-CEBE-41AE-B86F-B1BBADE7FCB4},6"'),
    "20240113/data2": (16526178246, '"{F5AC630B-14EA-454A-B2AA-E4F831904A68},6"'),
    "20240113/data3": (16368978716, '"{76B7E645-F1AD-4524-A120-168D86356A9D},6"'),
    "20240115/data2": (24165399551, '"{E2FD442B-B1D9-47FC-B523-313651B8A6A6},6"'),
    "20240115/data3": (8765886287, '"{F55FC8D0-2FCF-43E7-BC25-FC59BCEE7571},6"'),
    "20240116/data2": (21964365294, '"{2B936414-E49A-4277-BC9A-BB2A804D35FC},6"'),
    "20240116/data4": (25013830505, '"{85180E72-22C4-4722-B4CC-818578AE1452},6"'),
    "20240116/data5": (8185481886, '"{59828FB6-2503-42DC-8956-5CEB2EC98F58},6"'),
    "20240116_eve/data3": (18208252397, '"{02882A02-A5A4-4475-B17A-0E866C13FA35},6"'),
    "20240116_eve/data4": (12196350969, '"{B30651EB-D739-40E1-AB1E-0D84054F5521},6"'),
    "20240116_eve/data5": (6294526883, '"{1C5868ED-5106-43B1-912B-2FF55FAAFF99},6"'),
    "20240123/data2": (27621464463, '"{439FD84D-5D0C-402B-A848-74D144F0C3B4},6"'),
    "20240123/data3": (8891208303, '"{39113311-2A1B-44EC-B7F0-CB27E4FC6C22},6"'),
}
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
    pin = EVALUATION.get(sequence)
    if pin is not None and (int(item["size"]), item["eTag"]) != pin:
        raise SystemExit(f"{sequence}.zip is not the pinned file: {item['size']} {item['eTag']}")
    remote = RemoteFile(item["@content.downloadUrl"], int(item["size"]))
    if remote.read(4) != b"PK\x03\x04":
        raise SystemExit(f"{sequence}.zip is not a zip")
    remote.seek(0)
    return zipfile.ZipFile(io.BufferedReader(remote, buffer_size=CHUNK_BYTES)), int(item["size"])


def expected_cases(sequences: list[str]) -> int:
    """(sequence, IMU) cases: the X36D everywhere, the MTi-3 from ``MTI_FROM_DAY`` on."""

    return sum(1 + (sequence[:8] >= MTI_FROM_DAY) for sequence in sequences)


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
    chosen = parser.add_mutually_exclusive_group(required=True)
    chosen.add_argument("--sequences", nargs="+", help="DAY/dataN ... (development)")
    chosen.add_argument(
        "--evaluation", action="store_true", help="the pre-registered unseen sequences, once"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="sequences run at once")
    parser.add_argument(
        "--expected-cases", type=int, default=0, help="(sequence, IMU) cases the plan expects"
    )
    args = parser.parse_args(argv)
    if args.evaluation:
        args.sequences = list(EVALUATION)
        args.expected_cases = expected_cases(args.sequences)
    elif set(args.sequences) & set(EVALUATION):
        raise SystemExit("an evaluation sequence is read only by --evaluation")
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
