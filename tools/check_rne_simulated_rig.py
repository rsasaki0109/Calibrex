"""Check a bag simulated by Robot Native Engine: true mountings pass, a turned sensor fails.

Robot Native Engine's example 149 (``calibrex_rig_bag``) simulates a hand-held rig in a
room and writes ``rig.mcap`` with ``good.yaml`` (the true mountings in the ``lidar``
frame) and one candidate per sensor turned 3 degrees. Recorded with
``--sensors imu,lidar2`` it holds a LiDAR, a second LiDAR and an IMU. This script runs
``calibrex check --pairs imu-lidar,lidar-lidar`` on three candidates and fails unless
every pair gets the verdict in :data:`EXPECTED`:

* a pair that should pass passes, with roll, pitch and yaw judged and every judged axis
  within its tolerance;
* a pair that should fail fails, and its rotation error (the norm of the roll, pitch and
  yaw errors) is between 2 and 4 degrees, the 3 degree turn.

It is what the ``RNE simulated rig`` workflow runs; locally::

    cargo run --release -p calibrex_rig_bag --example 149_calibrex_rig_bag -- \\
        --out /tmp/rne_rig --duration-s 20 --sensors imu,lidar2  # in RobotNativeEngine
    python tools/check_rne_simulated_rig.py /tmp/rne_rig

The data is simulated: a pass shows the estimator and verdict path work end to end on a
recording with a known truth, not how accurate they are on hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROTATION_AXES = ("roll", "pitch", "yaw")
PAIRS = "imu-lidar,lidar-lidar"
TURN_RANGE_DEG = (2.0, 4.0)

PairKey = tuple[str, frozenset[str]]
IMU_LIDAR: PairKey = ("imu-lidar", frozenset({"imu", "lidar"}))
IMU_LIDAR2: PairKey = ("imu-lidar", frozenset({"imu", "lidar2"}))
LIDAR_LIDAR: PairKey = ("lidar-lidar", frozenset({"lidar", "lidar2"}))

EXPECTED: dict[str, dict[PairKey, str]] = {
    "good.yaml": {IMU_LIDAR: "pass", IMU_LIDAR2: "pass", LIDAR_LIDAR: "pass"},
    # The IMU turned about the LiDAR z axis: both IMU pairs fail, the LiDARs agree.
    "turned_imu.yaml": {IMU_LIDAR: "fail", IMU_LIDAR2: "fail", LIDAR_LIDAR: "pass"},
    # lidar2 turned about its own z axis: its pairs fail, imu-lidar still passes.
    "turned_lidar2.yaml": {IMU_LIDAR: "pass", IMU_LIDAR2: "fail", LIDAR_LIDAR: "fail"},
}


def run_check(bag: Path, frames: Path, extra: list[str]) -> dict[str, Any]:
    """Run ``calibrex check`` on one candidate and return its JSON result."""

    command = [
        sys.executable,
        "-m",
        "calibrex.cli.main",
        "check",
        str(bag),
        "--tf",
        str(frames),
        "--pairs",
        PAIRS,
        "--fail-on",
        "never",
        "--json",
        "--quiet",
        *extra,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result: dict[str, Any] = json.loads(completed.stdout)
    return result


def judged_pairs(result: dict[str, Any]) -> dict[PairKey, dict[str, Any]]:
    """The pairs that got a verdict, by pair name and sensor frames."""

    return {
        (str(pair["pair"]), frozenset(pair["frames"])): pair
        for pair in result["pairs"]
        if pair.get("verdict") is not None
    }


def axes_of(pair: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(axis["name"]): axis for axis in pair.get("axes", [])}


def rotation_error_deg(axes: dict[str, dict[str, Any]]) -> float:
    return math.sqrt(
        sum(float(axes[name]["candidate_error"]) ** 2 for name in ROTATION_AXES if name in axes)
    )


def label(key: PairKey) -> str:
    return f"{key[0]} ({' / '.join(sorted(key[1]))})"


def check_pair(
    candidate: str, key: PairKey, pair: dict[str, Any], expected: str, problems: list[str]
) -> None:
    where = f"{candidate}: {label(key)}"
    verdict = str(pair["verdict"])
    axes = axes_of(pair)
    if verdict != expected:
        problems.append(f"{where}: verdict {verdict}, expected {expected}")
        return
    if expected == "pass":
        for name in ROTATION_AXES:
            if name not in axes:
                problems.append(f"{where}: {name} was not judged")
        for name, axis in axes.items():
            if axis["status"] != "pass" or abs(axis["candidate_error"]) > axis["tolerance"]:
                problems.append(
                    f"{where}: {name} {axis['candidate_error']:.3f}/{axis['tolerance']} "
                    f"{axis['status']}"
                )
    else:
        error = rotation_error_deg(axes)
        low, high = TURN_RANGE_DEG
        if not low <= error <= high:
            problems.append(
                f"{where}: rotation error {error:.3f} deg, expected the 3 deg turn "
                f"({low}-{high} deg)"
            )


def summary_line(candidate: str, key: PairKey, pair: dict[str, Any]) -> str:
    judged = "; ".join(
        f"{name} {axis['candidate_error']:.3f}/{axis['tolerance']} {axis['status']}"
        for name, axis in axes_of(pair).items()
    )
    return f"{candidate}: {label(key)} {pair['verdict']} ({judged})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rig_dir", type=Path, help="example 149 output: rig.mcap, good.yaml, ...")
    parser.add_argument(
        "--cache-dir", type=Path, help="calibrex check estimator cache (default: none)"
    )
    args = parser.parse_args(argv)

    bag = args.rig_dir / "rig.mcap"
    for path in (bag, *(args.rig_dir / name for name in EXPECTED)):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    # The estimates do not depend on the candidate: later checks reuse the first's.
    cache = ["--cache-dir", str(args.cache_dir)] if args.cache_dir else []
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as evidence:
        for candidate, expected in EXPECTED.items():
            result = run_check(bag, args.rig_dir / candidate, [*cache, "--evidence-dir", evidence])
            pairs = judged_pairs(result)
            for key, verdict in expected.items():
                pair = pairs.get(key)
                if pair is None:
                    problems.append(f"{candidate}: {label(key)} was not judged")
                    continue
                print(summary_line(candidate, key, pair))
                check_pair(candidate, key, pair, verdict, problems)
    for problem in problems:
        print(f"FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
