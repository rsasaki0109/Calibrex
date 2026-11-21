"""Check a bag simulated by Robot Native Engine: the true mounting passes, a turned one fails.

Robot Native Engine's example 149 (``calibrex_rig_bag``) simulates a hand-held LiDAR + IMU
rig in a room and writes ``rig.mcap`` with ``good.yaml`` (the true ``lidar -> imu``
mounting) and ``turned.yaml`` (the IMU turned 3 degrees about the LiDAR z axis). This
script runs ``calibrex check --pairs imu-lidar`` on both and fails unless:

* ``good.yaml`` passes, with roll, pitch and yaw judged and each within its tolerance;
* ``turned.yaml`` fails, on yaw, with a yaw error between 2 and 4 degrees, and roll and
  pitch still pass.

It is what the ``RNE simulated rig`` workflow runs; locally::

    cargo run --release -p calibrex_rig_bag --example 149_calibrex_rig_bag -- \\
        --out /tmp/rne_rig --duration-s 20           # in a RobotNativeEngine checkout
    python tools/check_rne_simulated_rig.py /tmp/rne_rig

The data is simulated: a pass shows the estimator and verdict path work end to end on a
recording with a known truth, not how accurate they are on hardware.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROTATION_AXES = ("roll", "pitch", "yaw")


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
        "imu-lidar",
        "--fail-on",
        "never",
        "--json",
        "--quiet",
        *extra,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result: dict[str, Any] = json.loads(completed.stdout)
    return result


def imu_lidar_axes(result: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    """The imu-lidar pair's verdict and its axes by name."""

    pairs = [
        pair
        for pair in result["pairs"]
        if pair.get("pair") == "imu-lidar" and pair.get("verdict") is not None
    ]
    if len(pairs) != 1:
        raise SystemExit(f"expected one judged imu-lidar pair, found {len(pairs)}")
    pair = pairs[0]
    return str(pair["verdict"]), {str(axis["name"]): axis for axis in pair.get("axes", [])}


def expect(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


def check_good(result: dict[str, Any], problems: list[str]) -> None:
    verdict, axes = imu_lidar_axes(result)
    expect(verdict == "pass", f"good.yaml: imu-lidar verdict {verdict}, expected pass", problems)
    for name in ROTATION_AXES:
        axis = axes.get(name)
        if axis is None:
            problems.append(f"good.yaml: {name} was not judged")
            continue
        expect(
            axis["status"] == "pass" and abs(axis["candidate_error"]) <= axis["tolerance"],
            f"good.yaml: {name} {axis['candidate_error']:.3f}/{axis['tolerance']} {axis['status']}",
            problems,
        )


def check_turned(result: dict[str, Any], problems: list[str]) -> None:
    verdict, axes = imu_lidar_axes(result)
    expect(verdict == "fail", f"turned.yaml: imu-lidar verdict {verdict}, expected fail", problems)
    yaw = axes.get("yaw")
    if yaw is None:
        problems.append("turned.yaml: yaw was not judged")
    else:
        expect(
            yaw["status"] == "fail" and 2.0 <= abs(yaw["candidate_error"]) <= 4.0,
            f"turned.yaml: yaw {yaw['candidate_error']:.3f} deg {yaw['status']}, "
            "expected a fail near the 3 deg turn",
            problems,
        )
    for name in ("roll", "pitch"):
        axis = axes.get(name)
        expect(
            axis is not None and axis["status"] == "pass",
            f"turned.yaml: {name} should still pass ({axis and axis['status']})",
            problems,
        )


def summary_line(label: str, result: dict[str, Any]) -> str:
    verdict, axes = imu_lidar_axes(result)
    judged = "; ".join(
        f"{name} {axis['candidate_error']:.3f}/{axis['tolerance']} {axis['status']}"
        for name in ROTATION_AXES
        if (axis := axes.get(name)) is not None
    )
    return f"{label}: imu-lidar {verdict} ({judged})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rig_dir", type=Path, help="example 149 output: rig.mcap, good.yaml, ...")
    parser.add_argument(
        "--cache-dir", type=Path, help="calibrex check estimator cache (default: none)"
    )
    args = parser.parse_args(argv)

    bag = args.rig_dir / "rig.mcap"
    for path in (bag, args.rig_dir / "good.yaml", args.rig_dir / "turned.yaml"):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    # The estimate does not depend on the candidate: the second check reuses the first's.
    cache = ["--cache-dir", str(args.cache_dir)] if args.cache_dir else []
    with tempfile.TemporaryDirectory() as evidence:
        extra = [*cache, "--evidence-dir", evidence]
        good = run_check(bag, args.rig_dir / "good.yaml", extra)
        turned = run_check(bag, args.rig_dir / "turned.yaml", extra)

    problems: list[str] = []
    check_good(good, problems)
    check_turned(turned, problems)
    print(summary_line("good.yaml", good))
    print(summary_line("turned.yaml", turned))
    for problem in problems:
        print(f"FAIL {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
