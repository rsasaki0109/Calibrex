#!/usr/bin/env python3
"""The pre-registered lidar-wheel_odometry audit on unseen NCLT sessions.

An evaluation harness, not part of the Calibrex package. For each session in ``SESSIONS``
it converts one window (``tools/nclt_to_rosbag2.py``, the archives pinned by ETag), runs
``calibrex check --pairs lidar-wheel_odometry`` on it with NCLT's own calibration as the
candidate (the bag's ``/tf_static``), scores the estimate in the evidence artifact against
that calibration, and deletes the bag. The calibration is never an input of the estimate:
the lidar-wheel estimator reads only the LiDAR scans and the wheel odometry, and the
candidate is used only for the verdict.

    python tools/nclt_lidar_wheel_audit.py --work-dir WORK --output-dir RUNS

See docs/benchmarks/nclt_lidar_wheel_preregistration.yaml.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.nclt_to_rosbag2 import S3, convert, reference_base_velodyne

SESSIONS: dict[str, tuple[int, str, int, str]] = {
    # session: (sensor archive size, ETag, velodyne archive size, ETag)
    "2012-02-02": (
        124485782,
        "f5ac6cee5a59aefda93b8c992ca986b1-15",
        18540266768,
        "4587eb8e8bbcd6cfa81d14776e3576e6-2211",
    ),
    "2012-05-11": (
        108371461,
        "2cbeca60df2ba3b594e09f8493ec5ffb-13",
        16110380065,
        "2a6319a48627dadcd1712c56e0ebab4f-1921",
    ),
    "2012-08-04": (
        101652700,
        "d47f951b004c9688d132a7e094b84a35-13",
        15755320735,
        "8bf1feda8a224b8b76b91beb3f63603a-1879",
    ),
    "2012-11-04": (
        101352381,
        "bf8b0a1ade72e50889510d5ac0d5f217-13",
        14963938140,
        "e039ccb001059fbb209a1109bbd778fe-1784",
    ),
    "2013-02-23": (
        100961203,
        "31a54a49ccfef4748ec2dddb7551cf71-13",
        17174854426,
        "5990494ddb54faa32d3ce83616582023-2048",
    ),
}
START_S = 60.0
"""The window starts this long after the session's first Velodyne packet."""
DURATION_S = 3600.0
"""One hour: twice the development window (2012-01-08), and within every session's data."""
DECIMATE = 4
PAIR = "lidar-wheel_odometry"
AXES = ("roll", "pitch", "yaw")

MAX_ROTATION_ERROR_DEG = 0.5
MAX_POSITION_ERROR_M = 0.20
SESSIONS_REQUIRED = len(SESSIONS)

SIGMA_K = 3.0
ROTATION_FLOOR_DEG = 0.5
FAIL_FACTOR = 2.0
"""calibrex check's defaults: an axis fails beyond FAIL_FACTOR x max(SIGMA_K x std, floor)."""
MIN_CONVERGED_SESSIONS = 3


def check_command(bag: Path, output: Path, evidence: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "calibrex.cli.main",
        "check",
        str(bag),
        "--vehicle-frame",
        "base_link",
        "--topic-kind",
        "/wheel_odom=wheel",
        "--pairs",
        PAIR,
        "--no-cache",
        "--output",
        str(output),
        "--evidence-dir",
        str(evidence),
    ]


def head(url: str) -> tuple[int, str]:
    """``(size, ETag)`` of an archive, from a HEAD request."""

    text = subprocess.run(["curl", "-sSfI", url], check=True, capture_output=True, text=True).stdout
    fields = {}
    for line in text.splitlines():
        name, _, value = line.partition(":")
        fields[name.strip().lower()] = value.strip().strip('"')
    return int(fields["content-length"]), fields["etag"]


def verify_archives(session: str) -> None:
    sensor_size, sensor_etag, velodyne_size, velodyne_etag = SESSIONS[session]
    for url, size, etag in (
        (f"{S3}/sensor_data/{session}_sen.tar.gz", sensor_size, sensor_etag),
        (f"{S3}/velodyne_data/{session}_vel.tar.gz", velodyne_size, velodyne_etag),
    ):
        if head(url) != (size, etag):
            raise SystemExit(f"{url}: not the pinned archive ({size} bytes, ETag {etag})")


def score_session(evidence: dict[str, Any], pair_record: dict[str, Any]) -> dict[str, Any]:
    """Errors of the estimate against NCLT's calibration, and the check's verdict on it."""

    from scipy.spatial.transform import Rotation

    from calibrex.check.estimators import rotation_errors_deg

    reference = reference_base_velodyne()
    converged = evidence.get("solver_status") == "converged"
    record: dict[str, Any] = {
        "solver_status": evidence.get("solver_status"),
        "policy_status": evidence.get("policy_status"),
        "check_status": pair_record.get("status"),
        "judged_axes": [axis["name"] for axis in pair_record.get("axes", [])],
        "converged": converged,
    }
    quaternion = evidence.get("rotation_quat_xyzw")
    if not converged or quaternion is None:
        return {**record, "accurate": False, "beyond_fail_bound": []}
    estimate = Rotation.from_quat(quaternion)
    error = estimate * Rotation.from_matrix(reference[:3, :3]).inv()
    rotation_error = math.degrees(error.magnitude())
    lever = evidence.get("lever_m")
    lever_error = None if lever is None else abs(float(lever) - float(reference[0, 3]))
    stds = {item["name"]: item["std_reported"] for item in evidence.get("parameters", [])}
    errors = rotation_errors_deg(reference[:3, :3], estimate.as_matrix())
    axis_errors = dict(zip(AXES, (float(value) for value in errors), strict=True))
    bounds = {
        name: FAIL_FACTOR * max(SIGMA_K * float(stds[name]), ROTATION_FLOOR_DEG) for name in AXES
    }
    record.update(
        rotation_error_deg=rotation_error,
        axis_errors_deg=axis_errors,
        fail_bounds_deg=bounds,
        beyond_fail_bound=[name for name in AXES if abs(axis_errors[name]) > bounds[name]],
        errors_in_stds={name: abs(axis_errors[name]) / float(stds[name]) for name in AXES},
        lever_m=lever,
        lever_error_m=lever_error,
        speed_scale=evidence.get("speed_scale"),
        time_offset_s=evidence.get("time_offset_s"),
        stds=stds,
        accurate=rotation_error < MAX_ROTATION_ERROR_DEG
        and lever_error is not None
        and lever_error < MAX_POSITION_ERROR_M,
    )
    return record


def verdicts(scores: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    accurate = [session for session, item in scores.items() if item.get("accurate")]
    converged = [session for session, item in scores.items() if item.get("converged")]
    failed = [session for session, item in scores.items() if item.get("check_status") == "fail"]
    beyond = [session for session, item in scores.items() if item.get("beyond_fail_bound")]
    judged = [session for session, item in scores.items() if item.get("judged_axes")]
    honest = not failed and not beyond and len(converged) >= MIN_CONVERGED_SESSIONS
    return {
        "accuracy": {
            "verdict": "supported" if len(accurate) >= SESSIONS_REQUIRED else "refuted",
            "accurate_sessions": accurate,
            "required": SESSIONS_REQUIRED,
        },
        "check_honesty": {
            "verdict": "supported" if honest else "refuted",
            "failed_sessions": failed,
            "sessions_beyond_fail_bound": beyond,
            "converged_sessions": converged,
            "sessions_with_judged_axes": judged,
        },
    }


def run_session(session: str, work: Path, runs: Path) -> dict[str, Any]:
    verify_archives(session)
    bag = work / f"{session}_bag"
    if bag.exists():
        shutil.rmtree(bag)
    sensor_size, sensor_etag, _velodyne_size, velodyne_etag = SESSIONS[session]
    conversion = convert(
        session,
        bag,
        work,
        start_s=START_S,
        duration_s=DURATION_S,
        decimate=DECIMATE,
        etags=(sensor_etag, velodyne_etag),
    )
    if (work / f"{session}_sen.tar.gz").stat().st_size != sensor_size:
        raise SystemExit(f"{session}: the sensor archive is not the pinned size")
    output = runs / f"{session}_check.json"
    evidence_dir = runs / f"{session}_evidence"
    completed = subprocess.run(
        check_command(bag, output, evidence_dir),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    (runs / f"{session}_check.log").write_text(completed.stdout + completed.stderr)
    shutil.rmtree(bag)
    import yaml

    if not output.exists():
        raise RuntimeError(f"calibrex check wrote no artifact (exit {completed.returncode})")
    artifact = json.loads(output.read_text())
    (pair_record,) = [item for item in artifact["pairs"] if item["pair"] == PAIR]
    evidence_files = sorted(evidence_dir.glob(f"{PAIR}_*.yaml"))
    evidence = yaml.safe_load(evidence_files[0].read_text()) if evidence_files else {}
    score = {"session": session, "conversion": conversion, **score_session(evidence, pair_record)}
    (runs / f"{session}_score.json").write_text(json.dumps(score, indent=1, default=float))
    return score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    for session in SESSIONS:
        try:
            scores[session] = run_session(session, args.work_dir, args.output_dir)
        except Exception as error:  # a session that cannot be run counts as a failure
            scores[session] = {"session": session, "error": repr(error), "accurate": False}
        (args.work_dir / f"{session}_sen.tar.gz").unlink(missing_ok=True)
    summary = {"scores": scores, "verdicts": verdicts(scores)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
