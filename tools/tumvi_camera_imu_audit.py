#!/usr/bin/env python3
"""The pre-registered camera-imu check audit on unseen TUM-VI corridor sequences.

An evaluation harness, not part of the Calibrex package. TUM-VI (Schubert et al., IROS
2018; CC BY 4.0) records a hand-held rig with two 190 deg fisheye cameras and a BMI160
IMU, calibrated with Kalibr. For each sequence in ``SESSIONS`` it downloads the 512x512
calibrated bag (pinned by size and ETag), runs ``calibrex check --pairs camera-imu`` four
times with four candidates written from Kalibr's camchain (as calibrated, and with
T_cam_imu turned 2 deg about the camera's x, y and z axes), records each camera's
``camera-imu`` verdict, and deletes the bag::

    python tools/tumvi_camera_imu_audit.py --camchain CAMCHAIN --work-dir WORK --output-dir RUNS

See docs/benchmarks/tumvi_camera_imu_preregistration.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BAGS = "https://cdn2.vision.in.tum.de/tumvi/calibrated/512_16"
SESSIONS: dict[str, tuple[int, str]] = {
    # sequence: (bag size in bytes, ETag)
    "corridor1": (6305344038, "177d3ea26-56a0fcb5fca46"),
    "corridor2": (7128726908, "1a8e7bd7c-56a0fd84cbc47"),
    "corridor3": (6107616861, "16c0ad65d-56a0fe89c410d"),
    "corridor4": (2029099748, "78f19ae4-56a0ff42e9514"),
    "corridor5": (6225138858, "1730c14aa-56a0ffad65f07"),
}
CAMCHAIN_SHA256 = "3a0c0285f4d0a3a1678053b2e5912e3e250e78ac09bd009f6d7d8bba669d5703"
"""Kalibr's camchain (dso/camchain.yaml of TUM-VI's dataset-calib-cam1_512_16 export)."""
CAMERAS = ("cam0", "cam1")
ERROR_DEG = 2.0
VARIANTS: dict[str, str | None] = {"kalibr": None, "rx2": "roll", "ry2": "pitch", "rz2": "yaw"}
"""Each candidate and the check axis its error lies on (camera x, y, z: roll, pitch, yaw)."""
AXIS_VECTORS = {"roll": (1.0, 0.0, 0.0), "pitch": (0.0, 1.0, 0.0), "yaw": (0.0, 0.0, 1.0)}
PAIRS = "camera-imu"
DOWNLOAD_ATTEMPTS = 20

MAX_KALIBR_FAILS = 0
MIN_KALIBR_JUDGED_FRACTION = 0.5
MAX_PERTURBED_PASSES = 0
MIN_PERTURBED_DETECTED_FRACTION = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_candidates(camchain: Path, out: Path) -> dict[str, Path]:
    """Kalibr's camchain, and copies with T_cam_imu turned ``ERROR_DEG`` about one camera axis.

    The turn is applied in the camera frame, R' = R_delta R and t' = R_delta t, so the
    candidate's error against Kalibr is ``ERROR_DEG`` about that camera axis and nothing
    else (check's rotation error is rotvec(R_candidate R_estimate^T), in the camera frame).
    """

    if sha256(camchain) != CAMCHAIN_SHA256:
        raise SystemExit(f"{camchain}: not the pinned Kalibr camchain")
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, axis in VARIANTS.items():
        chain = yaml.safe_load(camchain.read_text(encoding="utf-8"))
        if axis is not None:
            delta = Rotation.from_rotvec(np.radians(ERROR_DEG) * np.array(AXIS_VECTORS[axis]))
            for camera in CAMERAS:
                transform = np.array(chain[camera]["T_cam_imu"], dtype=float)
                transform[:3, :3] = delta.as_matrix() @ transform[:3, :3]
                transform[:3, 3] = delta.as_matrix() @ transform[:3, 3]
                chain[camera]["T_cam_imu"] = transform.tolist()
        path = out / f"camchain_{name}.yaml"
        path.write_text(yaml.safe_dump(chain, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


def check_command(bag: Path, candidate: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import sys; from calibrex.cli.main import main; sys.exit(main())",
        "check",
        str(bag),
        "--tf",
        str(candidate),
        "--pairs",
        PAIRS,
        "--no-cache",
        "--fail-on",
        "never",
        "--output",
        str(output),
    ]


def download(session: str, target: Path) -> None:
    """Fetch the pinned bag, resuming a cut-off transfer from the bytes already on disk.

    curl's own ``--retry`` restarts a transfer from where that call began, so a bag longer
    than what one connection carries would never finish; each call here resumes instead.
    """

    size, etag = SESSIONS[session]
    url = f"{BAGS}/dataset-{session}_512_16.bag"
    for attempt in range(DOWNLOAD_ATTEMPTS):
        if target.exists() and target.stat().st_size >= size:
            break
        if attempt:
            time.sleep(min(60, 5 * attempt))
        subprocess.run(
            ["curl", "-sSf", "-C", "-", "-H", f'If-Match: "{etag}"', "-o", str(target), url],
            check=False,
        )  # fmt: skip
    got = target.stat().st_size if target.exists() else 0
    if got != size:
        raise SystemExit(f"{url}: {got} bytes, pinned {size}")


def imu_records(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each camera's camera-imu pair record, keyed cam0/cam1."""

    records = {}
    for pair in artifact["pairs"]:
        if pair["pair"] != "camera-imu":
            continue
        camera = pair["sensors"][0].split("/")[0]
        # an axis counts as judged only when check gave the pair a verdict
        verdict_given = pair["status"] in ("pass", "warn", "fail")
        axes = pair.get("axes", [])
        rotation = [a for a in axes if a["name"] in AXIS_VECTORS]
        records[camera] = {
            "status": pair["status"],
            "reason_code": pair.get("reason_code"),
            "judged_axes": [a["name"] for a in rotation if verdict_given],
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
    """A perturbed case is detected when the pair fails on the axis that carries the error."""

    return record.get("status") == "fail" and record["axes"].get(axis, {}).get("status") == "fail"


def verdicts(scores: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """The two pre-registered claims over every (sequence, camera, candidate) case."""

    kalibr = [scores[s]["kalibr"].get(c, {}) for s in scores for c in CAMERAS]
    perturbed = [
        (scores[s][v].get(c, {}), axis)
        for s in scores
        for v, axis in VARIANTS.items()
        if axis is not None
        for c in CAMERAS
    ]
    expected = len(SESSIONS) * len(CAMERAS)
    expected_perturbed = expected * (len(VARIANTS) - 1)
    kalibr_fails = sum(r.get("status") == "fail" for r in kalibr)
    kalibr_judged = sum(bool(r.get("judged_axes")) for r in kalibr) / expected
    perturbed_passes = sum(r.get("status") == "pass" for r, _axis in perturbed)
    perturbed_detected = sum(detected(r, axis) for r, axis in perturbed if r) / expected_perturbed
    complete = len(kalibr) == expected and all(kalibr) and all(r for r, _axis in perturbed)
    honest = complete and kalibr_fails <= MAX_KALIBR_FAILS
    honest = honest and kalibr_judged >= MIN_KALIBR_JUDGED_FRACTION
    detects = complete and perturbed_passes <= MAX_PERTURBED_PASSES
    detects = detects and perturbed_detected >= MIN_PERTURBED_DETECTED_FRACTION
    return {
        "honesty": {
            "verdict": "supported" if honest else "refuted",
            "kalibr_fails": kalibr_fails,
            "kalibr_judged_fraction": kalibr_judged,
        },
        "detection": {
            "verdict": "supported" if detects else "refuted",
            "perturbed_passes": perturbed_passes,
            "perturbed_detected_fraction": perturbed_detected,
        },
        "complete": complete,
    }


def run_session(session: str, candidates: dict[str, Path], work: Path, runs: Path) -> dict:
    bag = work / f"dataset-{session}_512_16.bag"
    download(session, bag)
    result: dict[str, Any] = {}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for name, candidate in candidates.items():
        output = runs / f"{session}_{name}.json"
        completed = subprocess.run(
            check_command(bag, candidate, output), capture_output=True, text=True, env=env
        )
        (runs / f"{session}_{name}.log").write_text(completed.stdout + completed.stderr)
        artifact = json.loads(output.read_text()) if output.exists() else {"pairs": []}
        result[name] = imu_records(artifact)
    bag.unlink()
    (runs / f"{session}_score.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camchain", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = write_candidates(args.camchain, args.output_dir / "candidates")
    scores = {}
    for session in SESSIONS:
        try:
            scores[session] = run_session(session, candidates, args.work_dir, args.output_dir)
        except Exception as error:  # a sequence that cannot be run leaves its cases empty
            scores[session] = {"error": repr(error), **{name: {} for name in VARIANTS}}
        partial = args.work_dir / f"dataset-{session}_512_16.bag"
        partial.unlink(missing_ok=True)
    summary = {"scores": scores, "verdicts": verdicts(scores)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    shutil.rmtree(args.work_dir, ignore_errors=True)
    print(json.dumps(summary["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
