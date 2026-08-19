"""Opt-in integration test for empirical uncertainty on official KITTI inputs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.empirical_uncertainty import load_empirical_se3_uncertainty
from calibrex.core.validation import validate_file
from calibrex.data.kitti_benchmark import KITTI_RAW_0005_SEQUENCE_ID
from calibrex.data.kitti_camera_lidar_problem import build_kitti_raw_camera_lidar_problem
from calibrex.data.probabilistic_correspondence_from_problem import (
    build_probabilistic_correspondence_from_problem,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    "CALIBREX_KITTI_RAW_0005" not in os.environ
    or "CALIBREX_KITTI_DEPTH_PROVIDER" not in os.environ,
    reason=(
        "set CALIBREX_KITTI_RAW_0005 to the official sequence and "
        "CALIBREX_KITTI_DEPTH_PROVIDER to a frozen depth-provider artifact"
    ),
)
def test_official_kitti_empirical_uncertainty_ground_truth_workflow(
    tmp_path: Path,
) -> None:
    sequence = Path(os.environ["CALIBREX_KITTI_RAW_0005"])
    depth_provider = Path(os.environ["CALIBREX_KITTI_DEPTH_PROVIDER"])
    assert sequence.name == KITTI_RAW_0005_SEQUENCE_ID

    problem_path = tmp_path / "problem.yaml"
    problem = build_kitti_raw_camera_lidar_problem(
        sequence,
        depth_provider,
        command=("pytest", "kitti-empirical-uncertainty"),
    )
    problem.save(problem_path)

    correspondence_path = tmp_path / "correspondence.yaml"
    build_probabilistic_correspondence_from_problem(
        problem_path,
        transform_source="initial",
        max_points_per_frame=120,
        depth_relative_gate=0.35,
        seed=20260819,
    ).save(correspondence_path)

    result_dir = tmp_path / "resamples"
    output = tmp_path / "uncertainty.yaml"
    exit_code = main(
        [
            "camera-lidar",
            "empirical-uncertainty",
            str(correspondence_path),
            str(problem_path),
            "--result-dir",
            str(result_dir),
            "--output",
            str(output),
            "--block-length",
            "4",
            "--resample-count",
            "8",
            "--target-coverage",
            "0.6",
            "--seed",
            "20260819",
            "--json",
        ]
    )

    assert exit_code == 0, "expected pass or warn, not fail"
    assert validate_file(output).valid

    artifact = load_empirical_se3_uncertainty(output)
    assert artifact.policy_status in {"pass", "warn"}
    assert artifact.reference_transform is not None
    assert artifact.coverage_score is not None
    assert artifact.overconfidence_control is not None
    assert artifact.provenance.source_sha256
    assert len(list(result_dir.glob("*.yaml"))) == 8
