"""Retain independent synthetic LiDAR-IMU draws and assess formal uncertainty.

Run from a checkout: python tools/evaluate_p0_uncertainty.py --output-dir DIR.
Use --producer synthetic_bag to rebuild odometry, maps and correspondences on
EVERY draw. Known planes are a cheaper conditional-model control, not a bag test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.evaluation.uncertainty_coverage import (
    CoverageAxisObservation,
    CoverageTrial,
    UncertaintyCoverageReport,
    summarize_coverage,
)

BLOCKS = ("imu_rotation", "gyro_bias", "lever_arm", "imu_clock_offset", "accel_bias", "gravity")
AXES = [f"{block}[{i}]" for block in BLOCKS for i in range(1 if block == "imu_clock_offset" else 3)]


def main() -> None:
    """Run a fixed cohort, saving its protocol before any fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--producer", choices=("known_planes", "synthetic_bag"), default="known_planes"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(41001, 41021)))
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--map-rounds", type=int, default=4)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("seeds must be unique and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "src/calibrex").rglob("*.py")), Path(__file__).resolve()]
    digests = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    protocol = {
        "producer": args.producer,
        "seeds": args.seeds,
        "nominal_coverage": 0.95,
        "control_std_scale": 0.1,
        "duration_s": args.duration_s,
        "map_rounds": args.map_rounds,
        "source_sha256": digests,
        "git_commit": commit,
        "command": sys.argv,
    }
    _write(args.output_dir / "protocol.json", protocol)
    trials = []
    for seed in args.seeds:
        try:
            if args.producer == "known_planes":
                from calibrex.evaluation.continuous_time_lidar_imu_joint import (
                    run_synthetic_lidar_imu_joint_recovery,
                )

                result = run_synthetic_lidar_imu_joint_recovery(seed=seed, command=sys.argv)
                axes = [
                    CoverageAxisObservation(axis=f"{b.name}[{i}]", error=e, std=s)
                    for b in result.parameters
                    for i, (e, s) in enumerate(zip(b.error, b.std, strict=True))
                ]
                status = result.fit_status
            else:
                from calibrex.data.livox_ros2 import LivoxStreamProfile
                from calibrex.data.synthetic_lidar_imu_bag import (
                    SyntheticLidarImuBagSpec,
                    write_synthetic_lidar_imu_bag,
                )
                from calibrex.evaluation.lidar_imu_bag_calibration import (
                    LidarImuBagOptions,
                    calibrate_lidar_imu_bag,
                )

                spec = SyntheticLidarImuBagSpec(seed=seed, duration_s=args.duration_s)
                bag = write_synthetic_lidar_imu_bag(args.output_dir / f"bag-{seed}", spec)
                _write(args.output_dir / f"spec-{seed}.json", asdict(spec))
                profile = LivoxStreamProfile(
                    "synthetic", bag.lidar_topic, bag.imu_topic, "offset_time", "offset_s", "mps2"
                )
                result = calibrate_lidar_imu_bag(
                    bag.path,
                    profile,
                    LidarImuBagOptions(max_rounds=args.map_rounds),
                    dataset_family="synthetic",
                    dataset_license="synthetic",
                    command=sys.argv,
                )
                truth = spec.truth
                values = dict(
                    zip(
                        BLOCKS,
                        (
                            truth.imu_rotation_vector_rad,
                            truth.gyro_bias_rad_s,
                            truth.lever_arm_m,
                            [truth.clock_offset_s],
                            truth.accel_bias_m_s2,
                            truth.gravity_world_m_s2,
                        ),
                        strict=True,
                    )
                )
                axes = []
                for block in result.blocks:
                    # Odometry world gravity is not aligned to the synthetic world.
                    if block.name == "gravity":
                        continue
                    error = np.asarray(block.estimate) - values[block.name]
                    if block.name == "imu_rotation":
                        error = (
                            Rotation.from_rotvec(values[block.name]).inv()
                            * Rotation.from_rotvec(block.estimate)
                        ).as_rotvec()
                    axes.extend(
                        CoverageAxisObservation(axis=f"{block.name}[{i}]", error=e, std=s)
                        for i, (e, s) in enumerate(zip(error, block.formal_std, strict=True))
                    )
                status = result.rounds[-1].fit_status
            path = args.output_dir / f"result-{seed}.json"
            result.save(path)
            trial = CoverageTrial(
                seed=seed,
                fit_status=status,
                policy_status=result.policy_status,
                axes=axes,
                artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        except Exception as exc:
            trial = CoverageTrial(
                seed=seed,
                fit_status="exception",
                policy_status="failed",
                failure=f"{type(exc).__name__}: {exc}",
            )
        trials.append(trial)
        _write(args.output_dir / f"trial-{seed}.json", trial.model_dump(mode="json"))
        print(
            f"seed={seed} status={trial.fit_status}"
            f" policy={trial.policy_status} failure={trial.failure}",
            flush=True,
        )
    report = UncertaintyCoverageReport(
        nominal_coverage=0.95,
        control_std_scale=0.1,
        producer=args.producer,
        source_sha256=digests,
        command=sys.argv,
        git_commit=commit,
        trials=trials,
        axes=summarize_coverage(
            trials,
            [a for a in AXES if args.producer != "synthetic_bag" or not a.startswith("gravity[")],
        ),
        limitations=[
            "Synthetic-bag gravity components are excluded because world gauges are unaligned.",
            "Development evidence on independent noise draws of one fixed synthetic "
            "trajectory, not public accuracy.",
            "Marginal Gaussian intervals; per-axis Wilson intervals are not "
            "simultaneous family-wise guarantees.",
            "All returned fits, including stalled fits and failed quality gates, "
            "remain in coverage; exceptions remain in the denominator.",
            "No interval inflation is fitted on these evaluation draws; the 0.1x std "
            "control is reported separately.",
            "Known-plane runs condition on the plane map; synthetic_bag runs rebuild "
            "the complete front end on every independent recording.",
        ],
    )
    _write(args.output_dir / "coverage.json", report.model_dump(mode="json"))
    _write(args.output_dir / "coverage.schema.json", report.model_json_schema())


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
