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
from calibrex.core.external_run import load_external_run
from calibrex.core.provenance import sha256_path
from calibrex.core.result import (
    TransformEstimateProvenance,
    TransformResult,
)
from calibrex.core.validation import validate_file
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
                source_frame="camera_2",
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
        parent="camera_2",
        child="lidar",
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


def _estimate_args(problem: Path, output: Path) -> list[str]:
    return [
        "camera-lidar", "estimate-six-dof", str(problem), "--output", str(output),
        "--minimum-rotation-step-deg", "0.25", "--minimum-translation-step-m", "0.1", "--json",
    ]


def test_single_six_dof_estimate_saves_standard_result_and_exact_argv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from calibrex.core.result import load_result
    from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions, BorerSixDofSolver

    folder = tmp_path / "path with spaces"
    folder.mkdir()
    problem = _write_problem(folder)
    output = folder / "result.json"
    args = _estimate_args(problem, output)
    assert main(args) == 0
    result = load_result(output)
    assert validate_file(output, kind="result").valid
    assert result.production_valid
    assert result.run.status == "success"
    assert result.quality.grade == "warn"
    assert result.observability.rank is None
    assert result.run.provenance["command"] == ["calibrex", *args]
    assert result.run.provenance["data_verified"] is True
    assert result.run.provenance["reference_used_for_optimization"] is False
    key = "T_camera_2_lidar"
    assert result.candidate_extrinsics[key].provenance.role_in_comparison == "candidate"
    assert result.reference_extrinsics[key].provenance.role_in_comparison == "selected_reference"
    assert result.transforms[key].provenance.evidence_level == "algorithmically_refined"
    assert result.transforms[key].quality.std_rotation_deg == []
    assert result.transforms[key].covariance is None
    assert result.metrics["final_d2d_mutual_information"].holdout is None
    assert result.metrics["final_d2d_mutual_information"].grade == "warn"
    loaded = load_borer_problem(problem)
    expected = BorerSixDofSolver().solve(
        loaded.observations, loaded.problem.initial_transform_camera_lidar.as_se3(),
        BorerSixDofOptions(minimum_rotation_step_deg=0.25, minimum_translation_step_m=0.1),
    ).as_dict()
    actual = dict(result.run.provenance["solver_result"])
    expected.pop("projection_backend")
    actual.pop("projection_backend")
    assert actual == expected
    assert '"status": "converged"' in capsys.readouterr().out


@pytest.mark.parametrize("extra, status, output_pose", [
    (["--max-evaluations", "13", "--minimum-rotation-step-deg", "0.01"],
     "max_evaluations", True),
    (["--min-visible-points", "2000"], "insufficient_observations", False),
])
def test_single_estimate_nonconvergence_preserves_evidence_without_success(
    tmp_path: Path, extra: list[str], status: str, output_pose: bool,
) -> None:
    from calibrex.core.result import load_result

    problem = _write_problem(tmp_path)
    output = tmp_path / "result.yaml"
    assert main([*_estimate_args(problem, output), *extra]) == 2
    result = load_result(output)
    assert validate_file(output, kind="result").valid
    assert result.run.status == ("warning" if output_pose else "failed")
    assert result.run.provenance["solver_result"]["status"] == status
    assert bool(result.transforms) is output_pose
    assert bool(result.candidate_extrinsics)
    if not output_pose:
        assert result.quality.grade == "fail"
        assert result.metrics["final_d2d_mutual_information"].train is None
        assert result.quality.blocking_failures


def test_single_estimate_uses_prior_and_ignores_reference_values(tmp_path: Path) -> None:
    from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
    from calibrex.pipelines.camera_lidar_six_dof import run_camera_lidar_six_dof_estimate
    from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions, apply_local_se3_delta

    path = _write_problem(tmp_path)
    problem = load_camera_lidar_problem(path)
    prior = apply_local_se3_delta(
        problem.initial_transform_camera_lidar.as_se3(), [0.5, -0.25, 0.0], [0.1, 0.0, 0.0],
    )
    problem.initial_transform_camera_lidar.translation_m = list(prior.translation_m)
    problem.initial_transform_camera_lidar.rotation_quat_xyzw = list(prior.rotation_quat_xyzw)
    problem.save(path)
    options = BorerSixDofOptions(max_evaluations=13)
    first = run_camera_lidar_six_dof_estimate(path, options=options, command=["fixture"])
    problem.reference_transform_camera_lidar.translation_m = [1000.0, -200.0, 50.0]
    problem.reference_transform_camera_lidar.rotation_quat_xyzw = [0.5, 0.5, 0.5, 0.5]
    problem.save(path)
    second = run_camera_lidar_six_dof_estimate(path, options=options, command=["fixture"])
    assert first.run.provenance["solver_result"] == second.run.provenance["solver_result"]
    assert first.transforms == second.transforms
    initial_pose = second.run.provenance["solver_result"]["initial_transform_camera_lidar"]
    assert initial_pose == prior.as_dict()
    assert first.run.config_sha256 == second.run.config_sha256
    assert first.run.dataset_sha256 != second.run.dataset_sha256
    assert first.reference_extrinsics != second.reference_extrinsics


def test_single_estimate_holdout_is_excluded_from_search_and_ranges(tmp_path: Path) -> None:
    from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
    from calibrex.data.depth import load_depth_provider
    from calibrex.pipelines.camera_lidar_six_dof import run_camera_lidar_six_dof_estimate
    from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions

    path = _write_problem(tmp_path)
    problem = load_camera_lidar_problem(path)
    options = BorerSixDofOptions(max_evaluations=13)
    baseline = run_camera_lidar_six_dof_estimate(path, options=options, command=["fixture"])
    provider_path = Path(problem.depth_provider_path)
    provider = load_depth_provider(provider_path)
    extreme_depth = tmp_path / "held-out.depth.npy"
    np.save(extreme_depth, np.load(tmp_path / "000000.depth.npy") * 1000.0 + 500.0)
    provider.observations.append(provider.observations[0].model_copy(update={
        "frame_id": "held-out",
        "depth": depth_file_reference(extreme_depth, encoding="npy_float32"),
    }))
    provider.save(provider_path)
    problem.depth_provider_sha256 = sha256_path(provider_path) or ""
    problem.observations.append(problem.observations[0].model_copy(update={
        "frame_id": "held-out", "depth_observation_frame_id": "held-out",
    }))
    problem.save(path)
    result = run_camera_lidar_six_dof_estimate(
        path, options=options,
        holdout_frame_ids=["held-out"], command=["fixture"],
    )
    config = result.run.provenance["configuration"]
    assert config["train_frame_ids"] == ["000000"]
    assert config["holdout_frame_ids"] == ["held-out"]
    trace = result.run.provenance["solver_result"]
    assert trace == baseline.run.provenance["solver_result"]
    assert config["options"] == baseline.run.provenance["configuration"]["options"]
    assert all(item["evaluated_frame_count"] == 1 for item in trace["trace"])
    final = result.run.provenance["final_holdout_evaluation"]
    assert [item["frame_id"] for item in final["frames"]] == ["held-out"]
    metric = result.metrics["final_d2d_mutual_information"]
    assert metric.train is not None and metric.train > 0.0
    assert metric.holdout == 0.0
    invalid_splits = [
        (["unknown"], "unknown holdout"), (["000000", "held-out"], "training frame"),
    ]
    for ids, message in invalid_splits:
        with pytest.raises(ValueError, match=message):
            run_camera_lidar_six_dof_estimate(path, holdout_frame_ids=ids, command=["fixture"])


def test_single_estimate_can_reuse_only_a_matching_output_pose(tmp_path: Path) -> None:
    from calibrex.core.result import load_result

    path = _write_problem(tmp_path)
    initial_path = tmp_path / "first.yaml"
    output = tmp_path / "second.yaml"
    assert main(_estimate_args(path, initial_path)) == 0
    initial = load_result(initial_path)
    key = "T_camera_2_lidar"
    # Use a distinct prior so this catches ignoring --initial-result.
    initial.transforms[key].translation_m = [0.25, -0.125, 0.05]
    initial.save(initial_path)
    assert main([*_estimate_args(path, output), "--initial-result", str(initial_path)]) == 0
    result = load_result(output)
    assert result.candidate_extrinsics[key].as_se3() == initial.transforms[key].as_se3()
    recorded_digest = result.run.provenance["input_digests"]["initial_result_sha256"]
    assert recorded_digest == sha256_path(initial_path)
    assert result.run.provenance["initial_source"] == str(initial_path)
    # A reference alone is not an output pose usable as an implicit prior.
    initial.transforms = {}
    initial.save(initial_path)
    rejected = tmp_path / "rejected.yaml"
    assert main([*_estimate_args(path, rejected), "--initial-result", str(initial_path)]) == 2
    assert not rejected.exists()


@pytest.mark.parametrize("fault, message", [
    ("lidar_digest", "digest mismatch"),
    ("translation_bound", "exceeds the problem"),
    ("frame_mismatch", "same camera and LiDAR frames"),
    ("nonfinite_pose", "coordinates must be finite"),
    ("nonfinite_options", "Out of range float"),
])
def test_single_estimate_rejects_bad_input_before_saving(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fault: str, message: str,
) -> None:
    from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem

    path = _write_problem(tmp_path)
    problem = load_camera_lidar_problem(path)
    extra = []
    if fault == "lidar_digest":
        lidar = Path(problem.observations[0].lidar.path)
        lidar.write_bytes(lidar.read_bytes() + b"tampered")
    elif fault == "translation_bound":
        problem.translation_bound_m = 0.25
    elif fault == "frame_mismatch":
        problem.reference_transform_camera_lidar.child = "other-lidar"
    elif fault == "nonfinite_pose":
        problem.initial_transform_camera_lidar.translation_m = [float("nan"), 0.0, 0.0]
    else:
        extra = ["--improvement-tolerance", "nan"]
    problem.save(path)
    output = tmp_path / "rejected.yaml"
    assert main([*_estimate_args(path, output), *extra]) == 2
    assert not output.exists()
    assert message in capsys.readouterr().err


def test_single_estimate_missing_cuda_is_actionable_and_cpu_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import builtins

    original_import = builtins.__import__

    def without_cuda(name: str, *args: object, **kwargs: object) -> object:
        if name == "calibrex.solvers.cuda_depth_to_depth_adapter":
            raise ImportError("optional dependency absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cuda)
    path = _write_problem(tmp_path)
    output = tmp_path / "cuda.yaml"
    assert main([*_estimate_args(path, output), "--projection-backend", "cuda"]) == 2
    assert not output.exists()
    assert "calibrex[cuda]" in capsys.readouterr().err
    assert main(_estimate_args(path, tmp_path / "numpy.yaml")) == 0


def test_single_estimate_cuda_matches_cpu_result_trace_and_holdout(tmp_path: Path) -> None:
    from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
    from calibrex.core.result import load_result

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")
    path = _write_problem(tmp_path)
    problem = load_camera_lidar_problem(path)
    problem.observations.append(problem.observations[0].model_copy(update={"frame_id": "held-out"}))
    problem.save(path)
    results = []
    for backend in ("numpy", "cuda"):
        output = tmp_path / f"{backend}.yaml"
        assert main([
            *_estimate_args(path, output), "--projection-backend", backend,
            "--holdout-frame-id", "held-out",
        ]) == 0
        assert validate_file(output, kind="result").valid
        results.append(load_result(output))
    cpu, gpu = results
    assert gpu.transforms == cpu.transforms
    assert gpu.metrics == cpu.metrics
    cpu_holdout = cpu.run.provenance["final_holdout_evaluation"]
    assert gpu.run.provenance["final_holdout_evaluation"] == cpu_holdout
    cpu_trace = dict(cpu.run.provenance["solver_result"])
    gpu_trace = dict(gpu.run.provenance["solver_result"])
    cpu_trace.pop("projection_backend")
    gpu_trace.pop("projection_backend")
    assert cpu_trace == gpu_trace
    identity = gpu.run.provenance["projection_backend"]
    assert "calibrex.cuda_depth_pair_projector/v0.3;cupy=" in identity
    assert ";histogram_batch_frames=32" in identity
    assert ";device=" in identity


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
    assert protocol.provenance.command == [
        "calibrex",
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
        "--histogram-bins",
        "32",
        "--min-visible-points",
        "64",
        "--rotation-bound-deg",
        "2.0",
        "--translation-bound-m",
        "1.0",
        "--initial-rotation-step-deg",
        "0.25",
        "--initial-translation-step-m",
        "0.1",
        "--minimum-rotation-step-deg",
        "0.01",
        "--minimum-translation-step-m",
        "0.005",
        "--max-evaluations",
        "800",
    ]


def test_external_baseline_audit_retains_not_executed_release_gate(
    tmp_path: Path,
) -> None:
    problem_path = _write_problem(tmp_path)
    protocol = build_borer_six_dof_protocol(
        problem_path,
        perturbation_count=4,
        rotation_magnitude_deg=0.5,
        translation_magnitude_m=0.25,
    )
    protocol_path = tmp_path / "external-audit-protocol.yaml"
    protocol.save(protocol_path)
    output_directory = tmp_path / "external-audit"

    status = main(
        [
            "camera-lidar",
            "audit-external-baselines",
            str(problem_path),
            str(protocol_path),
            "--output-dir",
            str(output_directory),
            "--koide-source-commit",
            "a" * 40,
            "--unicalib-source-commit",
            "b" * 40,
            "--json",
        ]
    )

    assert status == 2
    koide_path = output_directory / "koide.external-run.yaml"
    unicalib_path = output_directory / "unicalib.external-run.yaml"
    assert validate_file(koide_path, "external-run").valid
    assert validate_file(unicalib_path, "external-run").valid
    assert load_external_run(koide_path).status == "not_executed"
    unicalib = load_external_run(unicalib_path)
    assert unicalib.status == "not_executed"
    assert unicalib.train_data_isolation.declared is False
    assert unicalib.parsed_outputs.external_metrics_comparable is False


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
    assert trace.initial_transform_camera_lidar.parent == "camera_2"
    assert trace.initial_transform_camera_lidar.child == "lidar"
    assert trace.output_transform_camera_lidar.parent == "camera_2"
    assert trace.output_transform_camera_lidar.child == "lidar"

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
    definition_text = (tmp_path / "six-dof-definition.yaml").read_text(
        encoding="utf-8"
    )
    assert "--projection-backend numpy" in definition_text
    assert "--bootstrap-samples 100" in definition_text


def test_cuda_benchmark_requires_one_worker_before_loading_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cuda projection requires --workers 1"):
        run_borer_six_dof_benchmark(
            tmp_path / "missing-problem.yaml", tmp_path / "missing-protocol.yaml",
            trace_directory=tmp_path / "traces", command="fixture",
            projection_backend="cuda", workers=2,
        )


def test_cuda_dependency_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from calibrex.evaluation.borer_six_dof_benchmark import _projection_runtime

    original_import = builtins.__import__

    def without_cuda(name: str, *args: object, **kwargs: object) -> object:
        if name == "calibrex.solvers.cuda_depth_to_depth_adapter":
            raise ImportError("optional dependency absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cuda)
    # Default execution still works without importing CuPy.
    projector, identity = _projection_runtime("numpy")
    assert callable(projector)
    assert "numpy_depth_pair_projector" in identity
    with pytest.raises(ValueError, match=r"calibrex\[cuda\]"):
        _projection_runtime("cuda")


def test_cuda_cli_benchmark_records_backend_and_rejects_mixed_resume(tmp_path: Path) -> None:
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")
    problem_path = _write_problem(tmp_path)
    protocol = build_borer_six_dof_protocol(
        problem_path, perturbation_count=1, rotation_magnitude_deg=0.2,
        translation_magnitude_m=0.05, min_visible_points=100, max_evaluations=20,
    )
    protocol_path = tmp_path / "cuda-protocol.yaml"
    protocol.save(protocol_path)
    trace_dir = tmp_path / "cuda-traces"
    args = [
        "camera-lidar", "benchmark-six-dof", str(problem_path), str(protocol_path),
        "--trace-dir", str(trace_dir), "--definition-output", str(tmp_path / "definition.yaml"),
        "--output", str(tmp_path / "benchmark.yaml"), "--bootstrap-samples", "10",
        "--projection-backend", "cuda", "--resume", "--json",
    ]
    assert main(args) == 0
    # Cache capacity is performance provenance; changed budgets cannot reuse
    # the earlier timing records, while an explicit default remains compatible.
    assert main([*args, "--cuda-cache-mib", "256"]) == 0
    assert main([*args, "--cuda-cache-mib", "0"]) == 2
    trace = load_calibration_candidate_trace(next(trace_dir.glob("*.trace.yaml")))
    assert ";projection_backend=calibrex.cuda_depth_pair_projector/v0.3;cupy=" in trace.solver
    assert ";histogram_batch_frames=32" in trace.solver
    assert ";device=" in trace.solver
    assert validate_file(tmp_path / "definition.yaml", kind="benchmark-definition").valid
    assert validate_file(tmp_path / "benchmark.yaml", kind="benchmark").valid
    assert main(args) == 0
    with pytest.raises(ValueError, match="identity mismatch"):
        run_borer_six_dof_benchmark(
            problem_path, protocol_path, trace_directory=trace_dir, command="fixture",
            projection_backend="numpy", resume=True, bootstrap_samples=10,
        )


@pytest.mark.parametrize("extra", [
    ["--cuda-cache-mib", "0"],
    ["--projection-backend", "cuda", "--cuda-cache-mib", "-1"],
])
def test_single_estimate_rejects_cache_flags_before_loading_inputs(
    tmp_path: Path, extra: list[str], capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "rejected.yaml"
    assert main([*_estimate_args(tmp_path / "missing.yaml", output), *extra]) == 2
    assert not output.exists()
    error = capsys.readouterr().err
    assert "CUDA cache settings" in error or "non-negative" in error


def test_single_estimate_cuda_budget_is_applied_and_recorded(tmp_path: Path) -> None:
    from calibrex.core.result import load_result

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")
    path = _write_problem(tmp_path)
    results = []
    for budget_mib in (0, 1):
        output = tmp_path / f"cache-{budget_mib}.yaml"
        assert main([
            *_estimate_args(path, output), "--projection-backend", "cuda",
            "--cuda-cache-mib", str(budget_mib),
        ]) == 0
        result = load_result(output)
        assert validate_file(output, kind="result").valid
        statistics = result.run.provenance["projection_transfer_statistics"]
        assert statistics["cache_budget_bytes"] == budget_mib * 1024 * 1024
        assert statistics["prepared_observation_count"] == budget_mib
        assert statistics["evaluation_input_upload_calls"] == (0 if budget_mib else 13)
        identity = result.run.provenance["projection_backend"]
        assert f";cache_bytes={budget_mib * 1024 * 1024}" in identity
        trace = dict(result.run.provenance["solver_result"])
        trace.pop("projection_backend")
        results.append((result.transforms, result.metrics, trace))
    assert results[0] == results[1]


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
    protocol = load_camera_lidar_benchmark_protocol(protocol_path)
    assert protocol.provenance.command == [
        "calibrex",
        "camera-lidar",
        "freeze-rotation-protocol",
        str(problem_path),
        "--output",
        str(protocol_path),
        "--perturbation-count",
        "2",
        "--rotation-deg",
        "2.0",
        "--histogram-bins",
        "20",
        "--min-visible-points",
        "100",
        "--bound-deg",
        "4.0",
        "--initial-step-deg",
        "2.0",
        "--minimum-step-deg",
        "0.5",
        "--max-evaluations",
        "60",
    ]
    assert protocol_path.is_file()
    assert definition_path.is_file()
    assert benchmark_path.is_file()
    bullseye_path = tmp_path / "benchmark_bullseye.svg"
    bullseye_artifact_path = tmp_path / "benchmark_bullseye.json"
    assert bullseye_path.is_file()
    artifact = load_bullseye_plot(bullseye_artifact_path)
    assert artifact.svg_sha256 == sha256_path(bullseye_path)
    assert len(artifact.points) == 2
