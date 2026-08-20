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
from calibrex.core.empirical_uncertainty import (
    load_empirical_se3_uncertainty,
)
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
from calibrex.evaluation.empirical_se3_uncertainty import (
    _ROTATION_AXES,
    _TRANSLATION_AXES,
    TANGENT_AXES,
    _axis_intervals,
    _central_half_width,
    _delta_to_reference,
    _estimate_delta,
    _form_blocks,
    _mean_se3,
    _observed_coverage,
    _overconfidence_control,
    _percentile,
    _policy,
    _sample_splits,
    run_empirical_se3_uncertainty,
)
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
)

TRUTH = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
CAMERA = DepthCameraIntrinsics(
    width=1280,
    height=720,
    fx=600.0,
    fy=600.0,
    cx=640.0,
    cy=360.0,
)
BASE_POINTS = (
    (-2.0, -1.0, 8.0),
    (-1.0, 1.0, 7.0),
    (0.5, -1.5, 9.0),
    (2.0, 1.0, 10.0),
    (-2.5, 0.5, 12.0),
    (1.5, -0.5, 6.0),
    (0.2, 1.8, 11.0),
    (2.8, -1.2, 13.0),
)


def test_percentile_uses_linear_interpolation() -> None:
    values = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 5.0
    assert _percentile(values, 0.5) == 3.0
    assert _percentile(values, 0.25) == 2.0
    assert _central_half_width([-1.0, 1.0], 0.8) == pytest.approx(0.8)


def test_block_formation_and_splits_are_deterministic_and_contiguous() -> None:
    frames = _frames(noise_std=0.0, n=12)
    blocks = _form_blocks(frames, 4)
    assert len(blocks) == 3
    assert [len(item.frame_ids) for item in blocks] == [4, 4, 4]
    for block in blocks:
        assert block.capture_time_start_ns <= block.capture_time_end_ns
    first = _sample_splits(3, 4, 0.6, seed=11)
    second = _sample_splits(3, 4, 0.6, seed=11)
    assert first == second
    for fit, hold in first:
        assert set(fit).isdisjoint(hold)
        assert sorted(fit + hold) == [0, 1, 2]


def test_mean_se3_and_tangent_deltas() -> None:
    estimates = [
        SE3(
            (0.10 + value, -0.05, 0.02),
            (0.0, 0.0, math.sin(value), math.cos(value)),
        )
        for value in (-0.1, 0.0, 0.1)
    ]
    mean = _mean_se3(estimates)
    assert mean.translation_m[0] == pytest.approx(0.10)
    deltas = [_estimate_delta(estimate, mean) for estimate in estimates]
    rotation_deltas = [item[1][2] for item in deltas]
    assert rotation_deltas[0] == pytest.approx(-2.0 * math.degrees(0.1), abs=1.0e-6)
    assert rotation_deltas[2] == pytest.approx(2.0 * math.degrees(0.1), abs=1.0e-6)
    reference_deltas = [_delta_to_reference(estimate, TRUTH) for estimate in estimates]
    assert reference_deltas[1][0] == (pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.0))
    assert reference_deltas[1][1] == (pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.0))


def test_observed_coverage_hits_target_on_unbiased_injected_truth() -> None:
    target = 0.8
    rng = random.Random(2026)
    estimates = [
        SE3(
            (
                TRUTH.translation_m[0] + rng.gauss(0.0, 0.01),
                TRUTH.translation_m[1] + rng.gauss(0.0, 0.01),
                TRUTH.translation_m[2] + rng.gauss(0.0, 0.01),
            ),
            (
                rng.gauss(0.0, math.radians(0.5)),
                rng.gauss(0.0, math.radians(0.5)),
                rng.gauss(0.0, math.radians(0.5)),
                1.0,
            ),
        )
        for _ in range(2000)
    ]
    mean = _mean_se3(estimates)
    estimate_deltas = [_estimate_delta(estimate, mean) for estimate in estimates]
    intervals = _axis_intervals(estimate_deltas, target)
    reference_deltas = [_delta_to_reference(estimate, TRUTH) for estimate in estimates]
    observed_translation = _observed_coverage(
        reference_deltas, intervals, _TRANSLATION_AXES
    )
    observed_rotation = _observed_coverage(
        reference_deltas, intervals, _ROTATION_AXES
    )
    assert observed_translation == pytest.approx(target, abs=0.05)
    assert observed_rotation == pytest.approx(target, abs=0.05)


def test_overconfident_control_and_policy_are_strict() -> None:
    target = 0.8
    rng = random.Random(7)
    center_estimates = [
        SE3(
            (
                TRUTH.translation_m[0] + rng.gauss(0.0, 0.01),
                TRUTH.translation_m[1] + rng.gauss(0.0, 0.01),
                TRUTH.translation_m[2] + rng.gauss(0.0, 0.01),
            ),
            (
                rng.gauss(0.0, math.radians(0.5)),
                rng.gauss(0.0, math.radians(0.5)),
                rng.gauss(0.0, math.radians(0.5)),
                1.0,
            ),
        )
        for _ in range(2000)
    ]
    intervals = _axis_intervals(
        [_estimate_delta(estimate, _mean_se3(center_estimates)) for estimate in center_estimates],
        target,
    )
    center_deltas = [
        _delta_to_reference(estimate, TRUTH) for estimate in center_estimates
    ]
    control = _overconfidence_control(center_deltas, intervals, 0.1, target)
    assert control.refuted is True
    status, reason = _policy(0.8, control, target)
    assert status == "pass"

    degenerate_estimates = [SE3(TRUTH.translation_m, TRUTH.rotation_quat_xyzw)] * 20
    degenerate_intervals = _axis_intervals(
        [_estimate_delta(estimate, TRUTH) for estimate in degenerate_estimates],
        target,
    )
    degenerate_deltas = [
        _delta_to_reference(estimate, TRUTH) for estimate in degenerate_estimates
    ]
    degenerate_control = _overconfidence_control(
        degenerate_deltas, degenerate_intervals, 0.1, target
    )
    assert degenerate_control.refuted is False
    status, reason = _policy(
        1.0,
        degenerate_control,
        target,
        axis_intervals=degenerate_intervals,
    )
    assert status == "warn"
    assert "degenerate" in reason


def test_pipeline_produces_schema_valid_artifact(tmp_path: Path) -> None:
    correspondence_path, problem_path = _write_inputs(
        tmp_path, noise_std=2.0, per_frame_rot_deg=0.3, per_frame_trans_m=0.01
    )
    artifact = run_empirical_se3_uncertainty(
        correspondence_path,
        problem_path,
        result_directory=tmp_path / "resamples",
        command="pytest empirical uncertainty",
        block_length=4,
        target_coverage=0.8,
        resample_count=8,
        seed=3,
        fit_block_ratio=0.6,
        options=_fast_options(),
    )
    assert artifact.resampling_method == "seeded_block_subsampling"
    assert artifact.sampling_unit == "temporal_block"
    assert artifact.tangent_ordering == list(TANGENT_AXES)
    assert artifact.policy_status in {"pass", "warn", "fail", "inconclusive"}
    assert artifact.coverage_score is not None
    assert artifact.observed_coverage_joint is not None
    assert len(artifact.iterations) == 8
    assert len(artifact.blocks) == 5
    assert artifact.overconfidence_control is not None
    assert artifact.reference_transform is not None

    output = tmp_path / "uncertainty.yaml"
    artifact.save(output)
    assert validate_file(output).valid
    loaded = load_empirical_se3_uncertainty(output)
    assert loaded.uncertainty_id == artifact.uncertainty_id


def test_pipeline_intervals_widen_with_more_noise(tmp_path: Path) -> None:
    low_path, low_problem = _write_inputs(
        tmp_path, "low", noise_std=2.0, per_frame_rot_deg=0.3, per_frame_trans_m=0.01
    )
    high_path, high_problem = _write_inputs(
        tmp_path, "high", noise_std=6.0, per_frame_rot_deg=0.3, per_frame_trans_m=0.01
    )
    low = run_empirical_se3_uncertainty(
        low_path,
        low_problem,
        result_directory=tmp_path / "low-resamples",
        command="pytest",
        block_length=4,
        target_coverage=0.8,
        resample_count=6,
        seed=1,
        fit_block_ratio=0.6,
        options=_fast_options(),
    )
    high = run_empirical_se3_uncertainty(
        high_path,
        high_problem,
        result_directory=tmp_path / "high-resamples",
        command="pytest",
        block_length=4,
        target_coverage=0.8,
        resample_count=6,
        seed=1,
        fit_block_ratio=0.6,
        options=_fast_options(),
    )
    assert high.interval_halfwidth_translation_m > low.interval_halfwidth_translation_m
    assert high.interval_halfwidth_rotation_deg > low.interval_halfwidth_rotation_deg


def test_pipeline_is_deterministic_given_seed(tmp_path: Path) -> None:
    correspondence_path, problem_path = _write_inputs(
        tmp_path, noise_std=2.0, per_frame_rot_deg=0.3, per_frame_trans_m=0.01
    )
    first = run_empirical_se3_uncertainty(
        correspondence_path,
        problem_path,
        result_directory=tmp_path / "resamples-1",
        command="pytest",
        block_length=4,
        target_coverage=0.8,
        resample_count=6,
        seed=5,
        fit_block_ratio=0.6,
        options=_fast_options(),
    )
    second = run_empirical_se3_uncertainty(
        correspondence_path,
        problem_path,
        result_directory=tmp_path / "resamples-2",
        command="pytest",
        block_length=4,
        target_coverage=0.8,
        resample_count=6,
        seed=5,
        fit_block_ratio=0.6,
        options=_fast_options(),
    )
    assert first.axis_intervals == second.axis_intervals
    assert first.observed_coverage_translation == second.observed_coverage_translation
    assert first.policy_status == second.policy_status
    assert [item.iteration_id for item in first.iterations] == [
        item.iteration_id for item in second.iterations
    ]


def test_pipeline_without_reference_is_inconclusive(tmp_path: Path) -> None:
    correspondence_path, problem_path = _write_inputs(
        tmp_path,
        noise_std=2.0,
        per_frame_rot_deg=0.3,
        per_frame_trans_m=0.01,
    )
    artifact = run_empirical_se3_uncertainty(
        correspondence_path,
        problem_path,
        result_directory=tmp_path / "resamples",
        command="pytest",
        block_length=4,
        target_coverage=0.8,
        resample_count=6,
        seed=0,
        fit_block_ratio=0.6,
        assess_coverage=False,
        options=_fast_options(),
    )
    assert artifact.policy_status == "inconclusive"
    assert artifact.reference_transform is None
    assert artifact.observed_coverage_translation is None
    assert artifact.observed_coverage_rotation is None
    assert artifact.overconfidence_control is None
    assert "coverage cannot be assessed" in artifact.policy_reason


def test_pipeline_degenerate_exact_data_fails_overconfidence(tmp_path: Path) -> None:
    correspondence_path, problem_path = _write_inputs(tmp_path, noise_std=0.0)
    artifact = run_empirical_se3_uncertainty(
        correspondence_path,
        problem_path,
        result_directory=tmp_path / "resamples",
        command="pytest",
        block_length=4,
        target_coverage=0.8,
        resample_count=6,
        seed=0,
        fit_block_ratio=0.6,
        options=_fast_options(),
    )
    assert artifact.policy_status == "warn"
    assert "degenerate" in artifact.policy_reason


def test_pipeline_rejects_too_few_successful_refits(tmp_path: Path) -> None:
    correspondence_path, problem_path = _write_inputs(tmp_path, noise_std=2.0)
    impossible_options = ProbabilisticCameraLidarRefinementOptions(
        minimum_train_correspondences=1000,
        minimum_holdout_correspondences=1000,
        max_evaluations=60,
    )
    with pytest.raises(ValueError, match="successful resample refits"):
        run_empirical_se3_uncertainty(
            correspondence_path,
            problem_path,
            result_directory=tmp_path / "resamples",
            command="pytest",
            block_length=4,
            target_coverage=0.8,
            resample_count=4,
            seed=0,
            fit_block_ratio=0.6,
            options=impossible_options,
        )


def test_cli_empirical_uncertainty_writes_schema_valid_artifact(tmp_path: Path) -> None:
    correspondence_path, problem_path = _write_inputs(
        tmp_path, noise_std=2.0, per_frame_rot_deg=0.3, per_frame_trans_m=0.01
    )
    output = tmp_path / "uncertainty.yaml"
    exit_code = main(
        [
            "camera-lidar",
            "empirical-uncertainty",
            str(correspondence_path),
            str(problem_path),
            "--result-dir",
            str(tmp_path / "cli-resamples"),
            "--output",
            str(output),
            "--block-length",
            "4",
            "--resample-count",
            "6",
            "--target-coverage",
            "0.8",
            "--json",
        ]
    )
    assert exit_code in {0, 2}
    assert validate_file(output).valid
    assert (tmp_path / "cli-resamples").exists()
    resample_files = list((tmp_path / "cli-resamples").glob("*.yaml"))
    assert len(resample_files) == 6


def _fast_options() -> ProbabilisticCameraLidarRefinementOptions:
    return ProbabilisticCameraLidarRefinementOptions(
        initial_rotation_step_deg=0.5,
        initial_translation_step_m=0.05,
        minimum_rotation_step_deg=0.05,
        minimum_translation_step_m=0.005,
        max_evaluations=400,
    )


def _frames(
    noise_std: float,
    n: int = 20,
    per_frame_rot_deg: float = 0.0,
    per_frame_trans_m: float = 0.0,
    seed: int = 123,
) -> list[ProbabilisticCorrespondenceFrame]:
    rng = random.Random(seed)
    frames: list[ProbabilisticCorrespondenceFrame] = []
    for frame_index in range(n):
        offset = SE3(
            (
                rng.gauss(0.0, per_frame_trans_m),
                rng.gauss(0.0, per_frame_trans_m),
                rng.gauss(0.0, per_frame_trans_m),
            ),
            (
                rng.gauss(0.0, math.radians(per_frame_rot_deg)),
                rng.gauss(0.0, math.radians(per_frame_rot_deg)),
                rng.gauss(0.0, math.radians(per_frame_rot_deg)),
                1.0,
            ),
        )
        frame_truth = TRUTH.compose(offset)
        points = [
            (
                point[0] + 0.02 * frame_index,
                point[1] - 0.01 * frame_index,
                point[2],
            )
            for point in BASE_POINTS
        ]
        correspondences = []
        for index, point in enumerate(points):
            camera_point = frame_truth.transform_point(point)
            correspondences.append(
                ProbabilisticImageCorrespondence(
                    correspondence_id=str(index),
                    point_lidar_m=list(point),
                    image_mean_px=[
                        CAMERA.fx * camera_point[0] / camera_point[2]
                        + CAMERA.cx
                        + rng.gauss(0.0, noise_std),
                        CAMERA.fy * camera_point[1] / camera_point[2]
                        + CAMERA.cy
                        + rng.gauss(0.0, noise_std),
                    ],
                    image_covariance_px2=[1.0, 0.0, 0.0, 1.0],
                    outlier_probability=0.01,
                    reliability=0.99,
                )
            )
        frames.append(
            ProbabilisticCorrespondenceFrame(
                frame_id=f"frame-{frame_index:03d}",
                capture_time_ns=frame_index,
                camera_frame="camera",
                lidar_frame="lidar",
                intrinsics=CAMERA,
                correspondences=correspondences,
            )
        )
    return frames


def _artifact(
    frames: list[ProbabilisticCorrespondenceFrame],
) -> ProbabilisticCorrespondenceArtifact:
    return ProbabilisticCorrespondenceArtifact(
        artifact_id="synthetic-uncertainty",
        dataset_id="synthetic",
        split_id="test",
        dataset_license_spdx="CC0-1.0",
        provider=CorrespondenceProviderIdentity(
            provider="fixture",
            model="exact",
            version="1",
            source_repository="https://example.test/provider",
            source_commit="a" * 40,
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="pytest",
            generator_version="1",
            input_sha256={"fixture": "b" * 64},
        ),
    )


def _problem(
    frames: list[ProbabilisticCorrespondenceFrame],
) -> CameraLidarCalibrationProblem:
    initial = SE3((0.05, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    initial_transform = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=list(initial.translation_m),
        rotation_quat_xyzw=list(initial.rotation_quat_xyzw),
    )
    reference_transform = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=list(TRUTH.translation_m),
        rotation_quat_xyzw=list(TRUTH.rotation_quat_xyzw),
    )
    return CameraLidarCalibrationProblem(
        problem_id="synthetic-uncertainty-problem",
        dataset_id="synthetic",
        dataset_family="synthetic",
        sequence_id="test",
        depth_provider_path="/tmp/depth-provider.yaml",
        depth_provider_sha256="c" * 64,
        observations=[
            CameraLidarObservationBinding(
                frame_id=frame.frame_id,
                split_id="test",
                depth_observation_frame_id=frame.frame_id,
                lidar=DepthFileReference(
                    path=f"/tmp/{frame.frame_id}.npy",
                    sha256="d" * 64,
                    size_bytes=1,
                    encoding="npy_float64_xyz",
                ),
                lidar_capture_time_ns=frame.capture_time_ns,
            )
            for frame in frames
        ],
        reference_transform_camera_lidar=reference_transform,
        initial_transform_camera_lidar=initial_transform,
        time_convention="synchronized synthetic observations",
        rotation_bound_deg=2.0,
        translation_bound_m=0.25,
        provenance=CameraLidarArtifactProvenance(
            generator="pytest",
            generator_version="1",
            source_sha256="e" * 64,
        ),
    )


def _write_inputs(
    root: Path,
    tag: str = "inputs",
    *,
    noise_std: float,
    per_frame_rot_deg: float = 0.0,
    per_frame_trans_m: float = 0.0,
) -> tuple[Path, Path]:
    directory = root / tag
    directory.mkdir(parents=True, exist_ok=True)
    frames = _frames(
        noise_std,
        per_frame_rot_deg=per_frame_rot_deg,
        per_frame_trans_m=per_frame_trans_m,
    )
    correspondence_path = directory / "correspondence.yaml"
    problem_path = directory / "problem.yaml"
    _artifact(frames).save(correspondence_path)
    _problem(frames).save(problem_path)
    return correspondence_path, problem_path