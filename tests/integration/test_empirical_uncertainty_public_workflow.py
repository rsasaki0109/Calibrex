"""Integration test: empirical SE(3) uncertainty — stability-only public workflow.

This test exercises the full CLI path of
``calibrex camera-lidar empirical-uncertainty --stability-only`` on a
synthetic correspondence set whose scale and structure match the KITTI-0005
public benchmark (20 frames, 8 point correspondences per frame, realistic
noise, KITTI-like camera / LiDAR frame names).

Independent ground-truth depth data is not required: the ``--stability-only``
flag bypasses coverage assessment and yields an INCONCLUSIVE policy — which
is the correct honest report when no independent reference transform exists
for a public sequence.

Acceptance criteria (from development roadmap 2026-08-19):
- Artifact is schema-valid (``slac.empirical_se3_uncertainty/v0.1``).
- ``policy_status == "inconclusive"`` with a non-empty ``policy_reason``.
- All six tangent-axis intervals are present and non-negative.
- Per-resample refit artifacts are written to the result directory.
- Provenance captures generator, seed, block_length, resample_count,
  fit_block_ratio, and correspondence / problem digests.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.camera_lidar_artifacts import (
    CameraLidarArtifactProvenance,
    CameraLidarCalibrationProblem,
    CameraLidarObservationBinding,
)
from calibrex.core.empirical_uncertainty import load_empirical_se3_uncertainty
from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
)
from calibrex.core.result import TransformResult
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference

pytestmark = pytest.mark.integration

_FRAME_COUNT = 20
_TRUTH = SE3(
    (0.27, -0.04, 0.08),
    (0.0, 0.0, math.sin(math.radians(1.5)), math.cos(math.radians(1.5))),
)
_INITIAL = SE3((0.25, -0.03, 0.07), (0.0, 0.0, 0.0, 1.0))
_CAMERA = DepthCameraIntrinsics(
    width=1392,
    height=512,
    fx=707.1,
    fy=707.1,
    cx=608.0,
    cy=185.0,
)
_BASE_POINTS = (
    (-4.0, -1.2, 18.0),
    (-2.0,  1.4, 15.0),
    ( 0.5, -2.0, 20.0),
    ( 3.0,  1.0, 22.0),
    (-3.5,  0.8, 25.0),
    ( 2.5, -0.6, 14.0),
    ( 0.3,  2.2, 19.0),
    ( 4.0, -1.5, 28.0),
)


def _build_frames(
    noise_std: float = 1.5,
    *,
    pose_jitter: bool = True,
) -> list[ProbabilisticCorrespondenceFrame]:
    rng = random.Random(2026_08_19)
    frames: list[ProbabilisticCorrespondenceFrame] = []
    for i in range(_FRAME_COUNT):
        if pose_jitter:
            offset = SE3(
                (rng.gauss(0.0, 0.01), rng.gauss(0.0, 0.01), rng.gauss(0.0, 0.01)),
                (
                    rng.gauss(0.0, math.radians(0.3)),
                    rng.gauss(0.0, math.radians(0.3)),
                    rng.gauss(0.0, math.radians(0.3)),
                    1.0,
                ),
            )
        else:
            offset = SE3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        truth_i = _TRUTH.compose(offset)
        points = [
            (p[0] + 0.03 * i, p[1] - 0.01 * i, p[2])
            for p in _BASE_POINTS
        ]
        correspondences = []
        for idx, point in enumerate(points):
            cam = truth_i.transform_point(point)
            correspondences.append(
                ProbabilisticImageCorrespondence(
                    correspondence_id=str(idx),
                    point_lidar_m=list(point),
                    image_mean_px=[
                        _CAMERA.fx * cam[0] / cam[2] + _CAMERA.cx + rng.gauss(0.0, noise_std),
                        _CAMERA.fy * cam[1] / cam[2] + _CAMERA.cy + rng.gauss(0.0, noise_std),
                    ],
                    image_covariance_px2=[1.0, 0.0, 0.0, 1.0],
                    outlier_probability=0.02,
                    reliability=0.98,
                )
            )
        frames.append(
            ProbabilisticCorrespondenceFrame(
                frame_id=f"kitti-0005-{i:010d}",
                capture_time_ns=i * 100_000_000,
                camera_frame="camera_02",
                lidar_frame="velodyne",
                intrinsics=_CAMERA,
                correspondences=correspondences,
            )
        )
    return frames


def _write_inputs(
    root: Path,
    *,
    noise_std: float = 1.5,
    reference_transform: TransformResult | None = None,
    initial_transform: TransformResult | None = None,
    pose_jitter: bool = True,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    frames = _build_frames(noise_std=noise_std, pose_jitter=pose_jitter)
    artifact = ProbabilisticCorrespondenceArtifact(
        artifact_id="kitti-0005-stability-fixture",
        dataset_id="kitti-0005-synthetic",
        split_id="all",
        dataset_license_spdx="CC-BY-NC-SA-3.0",
        provider=CorrespondenceProviderIdentity(
            provider="fixture",
            model="exact-projection",
            version="1",
            source_repository="https://example.test/fixture",
            source_commit="a" * 40,
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="calibrex-integration-test",
            generator_version="1",
            input_sha256={"fixture": "b" * 64},
        ),
    )
    default_initial = TransformResult(
        parent="camera_02",
        child="velodyne",
        translation_m=list(_INITIAL.translation_m),
        rotation_quat_xyzw=list(_INITIAL.rotation_quat_xyzw),
    )
    initial = initial_transform if initial_transform is not None else default_initial
    reference = reference_transform if reference_transform is not None else initial
    problem = CameraLidarCalibrationProblem(
        problem_id="kitti-0005-stability-problem",
        dataset_id="kitti-0005-synthetic",
        dataset_family="KITTI",
        sequence_id="2011_09_30_drive_0005_sync",
        depth_provider_path="/tmp/depth-provider.yaml",
        depth_provider_sha256="c" * 64,
        observations=[
            CameraLidarObservationBinding(
                frame_id=f.frame_id,
                split_id="all",
                depth_observation_frame_id=f.frame_id,
                lidar=DepthFileReference(
                    path=f"/tmp/{f.frame_id}.npy",
                    sha256="d" * 64,
                    size_bytes=1,
                    encoding="npy_float64_xyz",
                ),
                lidar_capture_time_ns=f.capture_time_ns,
            )
            for f in frames
        ],
        reference_transform_camera_lidar=reference,
        initial_transform_camera_lidar=initial,
        time_convention="synchronized",
        rotation_bound_deg=5.0,
        translation_bound_m=0.5,
        provenance=CameraLidarArtifactProvenance(
            generator="calibrex-integration-test",
            generator_version="1",
            source_sha256="e" * 64,
        ),
    )
    correspondence_path = root / "correspondence.yaml"
    problem_path = root / "problem.yaml"
    artifact.save(correspondence_path)
    problem.save(problem_path)
    return correspondence_path, problem_path


def test_empirical_uncertainty_stability_only_public_workflow(
    tmp_path: Path,
) -> None:
    """End-to-end CLI test for the stability-only public workflow.

    Mirrors the acceptance criteria for roadmap issue #1:
    run ``calibrex camera-lidar empirical-uncertainty --stability-only``
    on a public-scale dataset and confirm INCONCLUSIVE reporting with
    schema-valid provenance.
    """
    inputs_dir = tmp_path / "inputs"
    result_dir = tmp_path / "resamples"
    output = tmp_path / "uncertainty.yaml"

    correspondence_path, problem_path = _write_inputs(inputs_dir)

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
            "0.9",
            "--stability-only",
            "--json",
        ]
    )

    assert exit_code == 0, f"CLI exited with {exit_code}"
    assert validate_file(output).valid, "artifact is not schema-valid"

    artifact = load_empirical_se3_uncertainty(output)
    assert artifact.policy_status == "inconclusive"
    assert artifact.policy_reason
    assert "coverage cannot be assessed" in artifact.policy_reason

    assert artifact.reference_transform is None
    assert artifact.observed_coverage_translation is None
    assert artifact.observed_coverage_rotation is None
    assert artifact.overconfidence_control is None

    assert len(artifact.axis_intervals) == 6
    for interval in artifact.axis_intervals:
        assert interval.half_width >= 0.0, (
            f"non-negative interval expected for axis {interval.axis}"
        )

    resample_files = sorted(result_dir.glob("*.yaml"))
    assert len(resample_files) == 8, (
        f"expected 8 per-resample refits, got {len(resample_files)}"
    )

    assert artifact.seed == 0
    assert artifact.block_length == 4
    assert artifact.resample_count == 8
    prov = artifact.provenance
    assert prov.source_sha256  # correspondence + problem digest recorded


def test_empirical_uncertainty_ground_truth_public_workflow(
    tmp_path: Path,
) -> None:
    """End-to-end CLI test for the public workflow with reference truth.

    The fixture declares injected ``_TRUTH`` as the vendor reference so
    coverage assessment is exercised without requiring external depth artifacts.
    """
    inputs_dir = tmp_path / "inputs"
    result_dir = tmp_path / "resamples"
    output = tmp_path / "uncertainty.yaml"
    truth_transform = TransformResult(
        parent="camera_02",
        child="velodyne",
        translation_m=list(_TRUTH.translation_m),
        rotation_quat_xyzw=list(_TRUTH.rotation_quat_xyzw),
    )
    initial_transform = TransformResult(
        parent="camera_02",
        child="velodyne",
        translation_m=list(_INITIAL.translation_m),
        rotation_quat_xyzw=list(_TRUTH.rotation_quat_xyzw),
    )
    correspondence_path, problem_path = _write_inputs(
        inputs_dir,
        noise_std=2.0,
        reference_transform=truth_transform,
        initial_transform=initial_transform,
    )

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
            "3",
            "--json",
        ]
    )

    assert exit_code == 0, f"CLI exited with {exit_code}"
    assert validate_file(output).valid, "artifact is not schema-valid"

    artifact = load_empirical_se3_uncertainty(output)
    assert artifact.policy_status in {"pass", "warn"}
    assert artifact.reference_transform is not None
    assert artifact.observed_coverage_translation is not None
    assert artifact.observed_coverage_rotation is not None
    assert artifact.overconfidence_control is not None
    assert artifact.coverage_score is not None
    assert artifact.coverage_score >= 0.45
    assert artifact.reference_rotation_error_deg is not None
    assert artifact.reference_translation_error_m is not None

    assert len(artifact.axis_intervals) == 6
    for interval in artifact.axis_intervals:
        assert interval.half_width >= 0.0

    resample_files = sorted(result_dir.glob("*.yaml"))
    assert len(resample_files) == 8
