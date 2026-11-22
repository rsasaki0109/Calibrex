"""Check a bag simulated by Robot Native Engine: true mountings pass, a wrong one fails.

Robot Native Engine's example 149 (``calibrex_rig_bag``) simulates a hand-held rig in a
room and writes ``rig.mcap`` with ``good.yaml`` (the true mountings in the ``lidar``
frame), one candidate per sensor turned 3 degrees and ``shifted_imu.yaml`` (the IMU moved
5 cm). This script runs ``calibrex check`` on the candidates of a profile and fails unless
every pair gets the verdict the profile expects (:data:`PROFILES`):

``pairs`` (the default)
    A 20 s bag recorded with ``--sensors imu,lidar2,camera``: imu-lidar for both LiDARs,
    lidar-lidar and camera-lidar for both LiDARs, on the true mountings and each sensor
    turned.
``lever-arm``
    A 90 s bag recorded with ``--sensors imu --rotation-speed 2``, whose faster rotation
    makes the IMU's lever arm observable: imu-lidar on the true mounting, the IMU turned
    and the IMU shifted.

The verdicts are checked as follows:

* a pair that should pass passes, with every judged axis within its tolerance and the
  profile's axes judged (camera-lidar: at least one rotation axis, as its edge objective
  may leave an axis unchecked);
* a pair that should fail fails, and its rotation error (the norm of the roll, pitch and
  yaw errors) is between 2 and 4 degrees, the 3 degree turn;
* a pair whose lever arm should fail fails, its translation error is between 3 and 7 cm
  (the 5 cm shift) and every judged rotation axis passes.

It is what the ``RNE simulated rig`` workflow runs; locally, in RobotNativeEngine::

    cargo run --release -p calibrex_rig_bag --example 149_calibrex_rig_bag -- \\
        --out /tmp/rne_rig --duration-s 20 --sensors imu,lidar2,camera
    python tools/check_rne_simulated_rig.py /tmp/rne_rig
    cargo run --release -p calibrex_rig_bag --example 149_calibrex_rig_bag -- \\
        --out /tmp/rne_lever --duration-s 90 --sensors imu --rotation-speed 2
    python tools/check_rne_simulated_rig.py /tmp/rne_lever --profile lever-arm

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROTATION_AXES = ("roll", "pitch", "yaw")
TRANSLATION_AXES = ("x", "y", "z")
TURN_RANGE_DEG = (2.0, 4.0)
SHIFT_RANGE_M = (0.03, 0.07)

PairKey = tuple[str, frozenset[str]]
Expected = Literal["pass", "fail", "fail-shift"]
IMU_LIDAR: PairKey = ("imu-lidar", frozenset({"imu", "lidar"}))
IMU_LIDAR2: PairKey = ("imu-lidar", frozenset({"imu", "lidar2"}))
LIDAR_LIDAR: PairKey = ("lidar-lidar", frozenset({"lidar", "lidar2"}))
CAMERA_LIDAR: PairKey = ("camera-lidar", frozenset({"camera", "lidar"}))
CAMERA_LIDAR2: PairKey = ("camera-lidar", frozenset({"camera", "lidar2"}))
CAMERA_PAIRS = frozenset({"camera-lidar"})


@dataclass(frozen=True)
class Profile:
    """The pairs to check, the axes a passing pair must judge, and the expected verdicts."""

    pairs: str
    required_axes: tuple[str, ...]
    expected: dict[str, dict[PairKey, Expected]]


PROFILES: dict[str, Profile] = {
    "pairs": Profile(
        pairs="imu-lidar,lidar-lidar,camera-lidar",
        required_axes=ROTATION_AXES,
        expected={
            "good.yaml": {
                IMU_LIDAR: "pass",
                IMU_LIDAR2: "pass",
                LIDAR_LIDAR: "pass",
                CAMERA_LIDAR: "pass",
                CAMERA_LIDAR2: "pass",
            },
            # The IMU turned about the LiDAR z axis: both IMU pairs fail, the rest agree.
            "turned_imu.yaml": {
                IMU_LIDAR: "fail",
                IMU_LIDAR2: "fail",
                LIDAR_LIDAR: "pass",
                CAMERA_LIDAR: "pass",
                CAMERA_LIDAR2: "pass",
            },
            # lidar2 turned about its own z axis: its pairs fail, the others still pass.
            "turned_lidar2.yaml": {
                IMU_LIDAR: "pass",
                IMU_LIDAR2: "fail",
                LIDAR_LIDAR: "fail",
                CAMERA_LIDAR: "pass",
                CAMERA_LIDAR2: "fail",
            },
            # The camera turned about its optical x axis: only the camera pairs fail.
            "turned_camera.yaml": {
                IMU_LIDAR: "pass",
                IMU_LIDAR2: "pass",
                LIDAR_LIDAR: "pass",
                CAMERA_LIDAR: "fail",
                CAMERA_LIDAR2: "fail",
            },
        },
    ),
    "lever-arm": Profile(
        pairs="imu-lidar",
        required_axes=(*ROTATION_AXES, *TRANSLATION_AXES),
        expected={
            "good.yaml": {IMU_LIDAR: "pass"},
            "turned_imu.yaml": {IMU_LIDAR: "fail"},
            # The IMU moved 5 cm along the LiDAR x axis: only the lever arm is wrong.
            "shifted_imu.yaml": {IMU_LIDAR: "fail-shift"},
        },
    ),
}


def run_check(bag: Path, frames: Path, pairs: str, extra: list[str]) -> dict[str, Any]:
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
        pairs,
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


def error_norm(axes: dict[str, dict[str, Any]], names: tuple[str, ...]) -> float:
    return math.sqrt(
        sum(float(axes[name]["candidate_error"]) ** 2 for name in names if name in axes)
    )


def label(key: PairKey) -> str:
    return f"{key[0]} ({' / '.join(sorted(key[1]))})"


def _out_of_tolerance(
    where: str, axes: dict[str, dict[str, Any]], names: tuple[str, ...]
) -> list[str]:
    return [
        f"{where}: {name} {axis['candidate_error']:.3f}/{axis['tolerance']:.3g} {axis['status']}"
        for name, axis in axes.items()
        if name in names
        and (axis["status"] != "pass" or abs(axis["candidate_error"]) > axis["tolerance"])
    ]


def check_pair(
    candidate: str,
    key: PairKey,
    pair: dict[str, Any],
    expected: Expected,
    required_axes: tuple[str, ...],
    problems: list[str],
) -> None:
    where = f"{candidate}: {label(key)}"
    verdict = str(pair["verdict"])
    axes = axes_of(pair)
    wanted = "fail" if expected == "fail-shift" else expected
    if verdict != wanted:
        problems.append(f"{where}: verdict {verdict}, expected {wanted}")
        return
    if expected == "pass":
        if key[0] in CAMERA_PAIRS:
            if not any(name in axes for name in ROTATION_AXES):
                problems.append(f"{where}: no rotation axis was judged")
        else:
            problems += [
                f"{where}: {name} was not judged" for name in required_axes if name not in axes
            ]
        problems += _out_of_tolerance(where, axes, tuple(axes))
    elif expected == "fail":
        error = error_norm(axes, ROTATION_AXES)
        low, high = TURN_RANGE_DEG
        if not low <= error <= high:
            problems.append(
                f"{where}: rotation error {error:.3f} deg, expected the 3 deg turn "
                f"({low}-{high} deg)"
            )
    else:
        error = error_norm(axes, TRANSLATION_AXES)
        low, high = SHIFT_RANGE_M
        if not low <= error <= high:
            problems.append(
                f"{where}: translation error {error:.4f} m, expected the 5 cm shift "
                f"({low}-{high} m)"
            )
        problems += _out_of_tolerance(where, axes, ROTATION_AXES)


def summary_line(candidate: str, key: PairKey, pair: dict[str, Any]) -> str:
    judged = "; ".join(
        f"{name} {axis['candidate_error']:.3f}/{axis['tolerance']:.3g} {axis['status']}"
        for name, axis in axes_of(pair).items()
    )
    return f"{candidate}: {label(key)} {pair['verdict']} ({judged})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rig_dir", type=Path, help="example 149 output: rig.mcap, good.yaml, ...")
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="pairs", help="what the bag was recorded for"
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="calibrex check estimator cache (default: none)"
    )
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]

    bag = args.rig_dir / "rig.mcap"
    for path in (bag, *(args.rig_dir / name for name in profile.expected)):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    # The estimates do not depend on the candidate: later checks reuse the first's.
    cache = ["--cache-dir", str(args.cache_dir)] if args.cache_dir else []
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as evidence:
        for candidate, expected in profile.expected.items():
            result = run_check(
                bag, args.rig_dir / candidate, profile.pairs, [*cache, "--evidence-dir", evidence]
            )
            pairs = judged_pairs(result)
            for key, verdict in expected.items():
                pair = pairs.get(key)
                if pair is None:
                    problems.append(f"{candidate}: {label(key)} was not judged")
                    continue
                print(summary_line(candidate, key, pair))
                check_pair(candidate, key, pair, verdict, profile.required_axes, problems)
    for problem in problems:
        print(f"FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
