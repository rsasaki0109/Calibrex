import math
from pathlib import Path

import numpy as np
import pytest

from calibrex.cli.main import main
from calibrex.core.camera_lidar_artifacts import (
    CameraLidarArtifactProvenance,
    CameraLidarCalibrationProblem,
    CameraLidarObservationBinding,
    load_bullseye_plot,
    load_calibration_candidate_trace,
    load_camera_lidar_benchmark_protocol,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import (
    TransformEstimateProvenance,
    TransformResult,
)
from calibrex.data.depth import (
    DepthCameraIntrinsics,
    DepthImageTransform,
    DepthMapObservation,
    DepthProviderArtifact,
    DepthProviderEnvironment,
    DepthProviderIdentity,
    default_depth_provider_provenance,
    depth_file_reference,
)
from calibrex.evaluation.borer_rotation_benchmark import (
    build_borer_rotation_protocol,
    build_borer_six_dof_protocol,
    load_borer_problem,
    run_borer_rotation_benchmark,
)
from calibrex.evaluation.borer_six_dof_benchmark import (
    run_borer_six_dof_benchmark,
)


def _write_problem(tmp_path: Path) -> Path:
    width, height = 48, 36
    fx = fy = 38.0
    cx, cy = 23.5, 17.5
    depth = np.full((height, width), np.nan, dtype=np.float32)
    points = []
    for v in range(4, 32):
        for u in range(5, 43):
            distance = (
                8.0
                + 0.05 * u
                + 1.2 * math.sin(0.41 * u)
                + 0.7 * math.cos(0.33 * v)
            )
            ray = np.asarray(
                [(u - cx) / fx, (v - cy) / fy, 1.0],
                dtype=float,
            )
            ray /= np.linalg.norm(ray)
            points.append(ray * distance)
            depth[v, u] = distance
    image_path = tmp_path / "000000.png"
    depth_path = tmp_path / "000000.depth.npy"
    lidar_path = tmp_path / "000000.lidar.npy"
    image_path.write_bytes(b"fixture-image")
    np.save(depth_path, depth, allow_pickle=False)
    np.save(lidar_path, np.asarray(points, dtype=np.float32), allow_pickle=False)
    depth_provider = DepthProviderArtifact(
        artifact_id="synthetic-provider",
        provider=DepthProviderIdentity(
            provider="synthetic",
            model="exact-range",
            version="v0.1",
            source_repository="https://example.test/synthetic",
            source_commit="0123456789abcdef",
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
        ),
        environment=DepthProviderEnvironment(execution_mode="imported"),
        scale_convention="metric_range",
        observations=[
            DepthMapObservation(
                frame_id="000000",
                capture_time_ns=0,
                source_frame="camera0",
                image=depth_file_reference(image_path, encoding="fixture"),
                depth=depth_file_reference(depth_path, encoding="npy_float32"),
                intrinsics=DepthCameraIntrinsics(
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                ),
                image_transform=DepthImageTransform(
                    source_width=width,
                    source_height=height,
                    output_width=width,
                    output_height=height,
                    scale_x=1.0,
                    scale_y=1.0,
                    interpolation="none",
                ),
                scale_convention="metric_range",
                scale_to_m=1.0,
                valid_fraction=float(np.isfinite(depth).mean()),
            )
        ],
        provenance=default_depth_provider_provenance(
            generator_version="fixture",
            input_manifest_sha256="1" * 64,
            output_manifest_sha256="2" * 64,
        ),
    )
    provider_path = tmp_path / "depth_provider.yaml"
    depth_provider.save(provider_path)
    provider_digest = sha256_path(provider_path)
    assert provider_digest is not None
    transform = TransformResult(
        parent="camera0",
        child="lidar0",
        translation_m=[0.0, 0.0, 0.0],
        rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        provenance=TransformEstimateProvenance(
            producer="dataset_provider",
            execution_mode="dataset_reference",
            evidence_level="synthetic_truth",
        ),
    )
    problem = CameraLidarCalibrationProblem(
        problem_id="synthetic-d2d",
        dataset_id="synthetic-d2d",
        dataset_family="synthetic",
        sequence_id="fixture",
        depth_provider_path=str(provider_path),
        depth_provider_sha256=provider_digest,
        observations=[
            CameraLidarObservationBinding(
                frame_id="000000",
                split_id="test",
                depth_observation_frame_id="000000",
                lidar=depth_file_reference(
                    lidar_path,
                    encoding="npy_xyz_float32",
                ),
                lidar_capture_time_ns=0,
            )
        ],
        reference_transform_camera_lidar=transform,
        initial_transform_camera_lidar=transform,
        time_convention="synchronized synthetic capture",
        rotation_bound_deg=6.0,
        translation_bound_m=1.0,
        provenance=CameraLidarArtifactProvenance(
            generator="fixture",
            generator_version="v0.1",
            source_sha256="3" * 64,
        ),
    )
    problem_path = tmp_path / "problem.yaml"
    problem.save(problem_path)
    return problem_path


def test_borer_problem_loader_verifies_and_loads_arrays(tmp_path: Path) -> None:
    loaded = load_borer_problem(_write_problem(tmp_path))

    assert len(loaded.observations) == 1
    assert loaded.observations[0].depth_map.shape == (36, 48)
    assert loaded.observations[0].lidar_points.shape[1] == 3


def test_six_dof_protocol_freezes_paired_fibonacci_perturbations(
    tmp_path: Path,
) -> None:
    problem_path = _write_problem(tmp_path)

    protocol = build_borer_six_dof_protocol(
        problem_path,
        perturbation_count=8,
        rotation_magnitude_deg=0.5,
        translation_magnitude_m=0.25,
    )

    assert protocol.degrees_of_freedom == "six_dof"
    assert protocol.optimizer == "bounded_se3_pattern_search/v0.1"
    for perturbation in protocol.perturbations:
        assert np.linalg.norm(perturbation.rotation_deg_xyz) == pytest.approx(0.5)
        assert np.linalg.norm(perturbation.translation_m_xyz) == pytest.approx(0.25)


def test_six_dof_protocol_cli_writes_valid_artifact(tmp_path: Path) -> None:
    problem_path = _write_problem(tmp_path)
    output = tmp_path / "six-dof.yaml"

    status = main(
        [
            "camera-lidar",
            "freeze-six-dof-protocol",
            str(problem_path),
            "--output",
            str(output),
            "--perturbation-count",
            "4",
            "--rotation-deg",
            "0.5",
            "--translation-m",
            "0.25",
            "--json",
        ]
    )

    assert status == 0
    protocol = load_camera_lidar_benchmark_protocol(output)
    assert protocol.degrees_of_freedom == "six_dof"
    assert len(protocol.perturbations) == 4


def test_six_dof_benchmark_retains_schema_valid_traces(tmp_path: Path) -> None:
    problem_path = _write_problem(tmp_path)
    protocol = build_borer_six_dof_protocol(
        problem_path,
        perturbation_count=2,
        rotation_magnitude_deg=0.2,
        translation_magnitude_m=0.05,
        histogram_bins=20,
        min_visible_points=100,
        rotation_bound_deg=1.0,
        translation_bound_m=0.2,
        initial_rotation_step_deg=0.5,
        initial_translation_step_m=0.05,
        minimum_rotation_step_deg=0.5,
        minimum_translation_step_m=0.05,
        max_evaluations=40,
    )
    protocol_path = tmp_path / "six-dof-protocol.yaml"
    protocol.save(protocol_path)

    definition, benchmark = run_borer_six_dof_benchmark(
        problem_path,
        protocol_path,
        trace_directory=tmp_path / "six-dof-traces",
        command="calibrex camera-lidar benchmark-six-dof fixture",
        bootstrap_samples=100,
        workers=2,
    )

    assert len(definition.trials) == 4
    summary = benchmark.method_summaries["native_borer_d2d_six_dof"]
    assert summary.trial_count == 2
    traces = sorted((tmp_path / "six-dof-traces").glob("*.trace.yaml"))
    assert len(traces) == 2
    trace = load_calibration_candidate_trace(traces[0])
    assert set(trace.evaluations[0].parameters) == {
        "roll_delta_deg",
        "pitch_delta_deg",
        "yaw_delta_deg",
        "tx_delta_m",
        "ty_delta_m",
        "tz_delta_m",
    }

    status = main(
        [
            "camera-lidar",
            "benchmark-six-dof",
            str(problem_path),
            str(protocol_path),
            "--trace-dir",
            str(tmp_path / "six-dof-traces"),
            "--definition-output",
            str(tmp_path / "six-dof-definition.yaml"),
            "--output",
            str(tmp_path / "six-dof-benchmark.yaml"),
            "--bootstrap-samples",
            "100",
            "--workers",
            "2",
            "--resume",
            "--json",
        ]
    )

    assert status == 0


def test_borer_rotation_benchmark_retains_trials_and_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem_path = _write_problem(tmp_path)
    protocol = build_borer_rotation_protocol(
        problem_path,
        perturbation_count=6,
        rotation_magnitude_deg=3.0,
        histogram_bins=20,
        min_visible_points=100,
        bound_deg=6.0,
        initial_step_deg=2.0,
        minimum_step_deg=0.25,
        max_evaluations=100,
    )
    protocol_path = tmp_path / "protocol.yaml"
    protocol.save(protocol_path)

    definition, benchmark = run_borer_rotation_benchmark(
        problem_path,
        protocol_path,
        trace_directory=tmp_path / "traces",
        command="calibrex camera-lidar benchmark-rotation fixture",
        bootstrap_samples=100,
        workers=2,
    )

    assert len(definition.protocol.splits) == 6
    assert len(definition.trials) == 12
    assert benchmark.provenance.data_verified is True
    assert benchmark.method_summaries["native_borer_d2d_rotation"].trial_count == 6
    assert (
        benchmark.method_summaries["native_borer_d2d_rotation"]
        .metrics["hit"]
        .distribution.count
        == 6
    )
    trace_paths = sorted((tmp_path / "traces").glob("*.trace.yaml"))
    assert len(trace_paths) == 6
    trace = load_calibration_candidate_trace(trace_paths[0])
    assert trace.problem_sha256 == protocol.problem_sha256
    assert trace.evaluations

    def fail_if_recomputed(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"completed trace was recomputed: {args}, {kwargs}")

    monkeypatch.setattr(
        "calibrex.evaluation.borer_rotation_benchmark."
        "BorerRotationOnlySolver.solve",
        fail_if_recomputed,
    )
    resumed_definition, resumed_benchmark = run_borer_rotation_benchmark(
        problem_path,
        protocol_path,
        trace_directory=tmp_path / "traces",
        command="calibrex camera-lidar benchmark-rotation fixture --resume",
        bootstrap_samples=100,
        workers=2,
        resume=True,
    )

    assert [trial.metrics for trial in resumed_definition.trials] == [
        trial.metrics for trial in definition.trials
    ]
    assert resumed_benchmark.method_summaries == benchmark.method_summaries


def test_borer_rotation_cli_freezes_and_executes_protocol(tmp_path: Path) -> None:
    problem_path = _write_problem(tmp_path)
    protocol_path = tmp_path / "protocol.yaml"
    definition_path = tmp_path / "definition.json"
    benchmark_path = tmp_path / "benchmark.json"

    freeze_status = main(
        [
            "camera-lidar",
            "freeze-rotation-protocol",
            str(problem_path),
            "--output",
            str(protocol_path),
            "--perturbation-count",
            "2",
            "--rotation-deg",
            "2",
            "--histogram-bins",
            "20",
            "--min-visible-points",
            "100",
            "--bound-deg",
            "4",
            "--initial-step-deg",
            "2",
            "--minimum-step-deg",
            "0.5",
            "--max-evaluations",
            "60",
            "--json",
        ]
    )
    benchmark_status = main(
        [
            "camera-lidar",
            "benchmark-rotation",
            str(problem_path),
            str(protocol_path),
            "--trace-dir",
            str(tmp_path / "cli-traces"),
            "--definition-output",
            str(definition_path),
            "--output",
            str(benchmark_path),
            "--bootstrap-samples",
            "10",
            "--workers",
            "2",
            "--required-hit-rate",
            "0",
            "--json",
        ]
    )

    assert freeze_status == 0
    assert benchmark_status == 0
    assert protocol_path.is_file()
    assert definition_path.is_file()
    assert benchmark_path.is_file()
    bullseye_path = tmp_path / "benchmark_bullseye.svg"
    bullseye_artifact_path = tmp_path / "benchmark_bullseye.json"
    assert bullseye_path.is_file()
    artifact = load_bullseye_plot(bullseye_artifact_path)
    assert artifact.svg_sha256 == sha256_path(bullseye_path)
    assert len(artifact.points) == 2
