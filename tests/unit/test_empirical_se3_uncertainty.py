from __future__ import annotations

import math
import random
from concurrent.futures import Future
from dataclasses import replace
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
from calibrex.core.io import read_mapping
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import TransformResult
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference
from calibrex.evaluation.empirical_se3_uncertainty import (
    _ROTATION_AXES,
    _TRANSLATION_AXES,
    TANGENT_AXES,
    _axis_intervals,
    _bounded_refits,
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


def _parallel_inputs(directory: Path) -> tuple[Path, Path]:
    frames = _frames(1.8, n=8, per_frame_rot_deg=0.3, per_frame_trans_m=0.01)
    for frame in frames:
        frame.correspondences = [
            item.model_copy(update={"correspondence_id": f"{repeat}-{item.correspondence_id}"})
            for repeat in range(8) for item in frame.correspondences
        ]
        frame.intrinsics = frame.intrinsics.model_copy(update={
            "distortion_model": "radial-tangential", "distortion": [0.0002, 0.0, 0.0, 0.0],
        })
    correspondence, problem = directory / "correspondence.json", directory / "problem.json"
    _artifact(frames).save(correspondence)
    _problem(frames).save(problem)
    return correspondence, problem


def _semantic_refits(folder: Path) -> dict[str, object]:
    values = {}
    for path in sorted(folder.glob("*.yaml")):
        assert validate_file(path).valid
        value = read_mapping(path)
        assert value["provenance"]["git_commit"] == "1" * 40
        value["provenance"].pop("created_at")
        if "iterations" in value:
            for item in value["iterations"]:
                filename = folder / (item["iteration_id"] + ".yaml")
                digest = sha256_path(filename)
                assert digest == item["output_sha256"]
                assert digest == value["provenance"]["resample_output_sha256"][item["iteration_id"]]
                item["output_sha256"] = "<HASH>"
            value["provenance"]["resample_output_sha256"] = {}
        values[path.name] = value
    return values


def test_spawn_refits_preserve_every_result_interval_and_parent_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrex.evaluation import empirical_se3_uncertainty as uncertainty

    correspondence, problem = _parallel_inputs(tmp_path)
    calls = []

    def source_commit() -> str:
        calls.append(True)
        return "1" * 40

    monkeypatch.setattr(uncertainty, "git_commit", source_commit)
    for workers in (1, 2):
        folder = tmp_path / str(workers)
        result = run_empirical_se3_uncertainty(
            correspondence, problem, result_directory=folder, command="pytest",
            block_length=2, resample_count=4, seed=3, workers=workers,
            options=_fast_options(),
        )
        result.save(folder / "uncertainty.yaml")
        assert len(result.iterations) == 4
        assert result.reference_transform is not None
        assert result.overconfidence_control is not None
    assert len(calls) == 2  # No Git subprocess is repeated in a refit process.
    assert _semantic_refits(tmp_path / "1") == _semantic_refits(tmp_path / "2")
    assert uncertainty._WORKER_CONTEXT is None


def test_parallel_failed_refits_are_retained_without_forming_intervals(tmp_path: Path) -> None:
    correspondence, problem = _parallel_inputs(tmp_path)
    folder = tmp_path / "failed"
    with pytest.raises(ValueError, match="fewer than two successful"):
        run_empirical_se3_uncertainty(
            correspondence, problem, result_directory=folder, command="pytest",
            block_length=2, resample_count=2, workers=2,
            options=replace(_fast_options(), minimum_train_correspondences=10**9),
        )
    files = list(folder.glob("resample-*.yaml"))
    assert len(files) == 2
    assert all(validate_file(path).valid for path in files)
    assert all(read_mapping(path)["status"] == "insufficient_correspondences" for path in files)


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_budget_is_rejected_before_input_loading(workers: int) -> None:
    with pytest.raises(ValueError, match="workers must be at least one"):
        run_empirical_se3_uncertainty(
            "absent.json", "absent.json", result_directory="unused", command="pytest",
            workers=workers,
        )


def test_refit_queue_is_bounded_and_cancels_unconsumed_work() -> None:
    class Executor:
        def __init__(self) -> None:
            self.futures: list[Future[int]] = []

        def submit(self, function: object, task: tuple[int, object]) -> Future[int]:
            future: Future[int] = Future()
            self.futures.append(future)
            # Deliberately leave later work unresolved. Closing the iterator
            # must cancel it instead of scheduling the entire requested run.
            if task[0] == 0:
                future.set_result(0)
            return future

    executor = Executor()
    results = _bounded_refits(executor, [([0], [1])] * 1000, 2)
    assert next(results) == 0
    assert len(executor.futures) == 4
    results.close()
    assert all(future.cancelled() for future in executor.futures[1:])


def test_worker_exception_is_propagated_without_retrying_in_serial() -> None:
    class Executor:
        def __init__(self) -> None:
            self.calls = 0

        def submit(self, function: object, task: object) -> Future[object]:
            self.calls += 1
            future: Future[object] = Future()
            future.set_exception(RuntimeError("worker failed"))
            return future

    executor = Executor()
    with pytest.raises(RuntimeError, match="worker failed"):
        list(_bounded_refits(executor, [([0], [1])] * 20, 2))
    assert executor.calls == 4


def test_cli_parallel_refits_record_requested_workers(tmp_path: Path) -> None:
    correspondence, problem = _parallel_inputs(tmp_path)
    output, folder = tmp_path / "uncertainty.yaml", tmp_path / "resamples"
    assert main([
        "camera-lidar", "empirical-uncertainty", str(correspondence), str(problem),
        "--output", str(output), "--result-dir", str(folder), "--workers", "2",
        "--block-length", "2", "--resample-count", "2", "--stability-only", "--json",
    ]) == 0
    result = load_empirical_se3_uncertainty(output)
    assert result.provenance.command[-2:] == ["--workers", "2"]
    assert len(result.iterations) == 2
    assert result.policy_status == "inconclusive"


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
