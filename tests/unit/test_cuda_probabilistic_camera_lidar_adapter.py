"""Real-device complete-trace parity and bounded resident snapshot checks."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp
from tests.unit.test_probabilistic_camera_lidar_refiner import _artifact, _problem

from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceFrame,
    ProbabilisticImageCorrespondence,
    load_probabilistic_refinement_result,
)
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics
from calibrex.evaluation.probabilistic_refinement_ablation import (
    run_probabilistic_refinement_ablation,
)
from calibrex.solvers import cuda_probabilistic_camera_lidar_adapter as adapter
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions as Options,
)
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefiner as Refiner,
)
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    _PreparedPoseEvaluation,
    project_camera_point,
)


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")


def frames():
    rng = np.random.default_rng(20261019)
    result = []
    for i in range(9):
        kind = ("pinhole", "mei", "double_sphere")[i % 3]
        intr = DepthCameraIntrinsics(
            width=1280,
            height=720,
            fx=600.0,
            fy=610.0,
            cx=640.0,
            cy=360.0,
            projection=kind,
            xi=0.7 if i % 3 else None,
            alpha=0.6 if i % 3 == 2 else None,
            distortion_model="radial-tangential",
            distortion=[0.01, -0.001, 0.0001, -0.0002],
        )
        frame = ProbabilisticCorrespondenceFrame(
            frame_id=f"frame-{i:03d}",
            capture_time_ns=i,
            camera_frame="camera",
            lidar_frame="lidar",
            intrinsics=intr,
            correspondences=[
                ProbabilisticImageCorrespondence(
                    correspondence_id=str(j),
                    point_lidar_m=p.tolist(),
                    image_mean_px=[0.0, 0.0],
                    image_covariance_px2=[1.5, 0.125, 0.125, 2.0],
                    outlier_probability=(j % 5) * 0.1,
                    reliability=0.2 + (j % 4) * 0.25,
                )
                for j, p in enumerate(rng.uniform((-4.0, -2.0, 2.0), (4.0, 2.0, 12.0), (128, 3)))
            ],
        )
        for c in frame.correspondences:
            c.image_mean_px = list(project_camera_point(tuple(c.point_lidar_m), frame))
        result.append(frame)
    return result


@pytest.mark.parametrize(
    "switches",
    [{}, {"use_covariance": False}, {"use_outlier_probability": False}, {"use_reliability": False}],
)
def test_complete_solver_trace_matches_numpy(switches):
    data = frames()
    initial = SE3((0.04, -0.03, 0.02), (0.0, 0.0, math.sin(0.002), math.cos(0.002)))
    options = Options(max_evaluations=400, **switches)
    expected = Refiner().solve(data, initial, options)
    actual = Refiner(evaluation_backend="cuda").solve(data, initial, options)
    assert actual.execution.cuda_calls > 0
    assert actual.execution.numpy_fallback_calls == 0
    assert replace(actual, execution=None) == expected


@pytest.mark.parametrize("case", ["k3", "missing_xi", "missing_alpha", "budget", "tiny"])
def test_conservative_native_fallback_keeps_values_and_errors(case, monkeypatch):
    data = frames()
    if case == "k3":
        for f in data:
            f.intrinsics.distortion.append(0.002)
    elif case == "missing_xi":
        data[1].intrinsics.xi = None
    elif case == "missing_alpha":
        data[2].intrinsics.alpha = None
    elif case == "budget":
        monkeypatch.setattr(adapter, "_POSE_CAPTURE_BUDGET_BYTES", 0)
    elif case == "tiny":
        for f in data:
            f.correspondences = f.correspondences[:8]
    options = Options()
    native = _PreparedPoseEvaluation(data, options)
    runtime = adapter.CudaProbabilisticPoseRuntime()
    prepared = runtime.prepare(data, options)
    if case.startswith("missing"):
        with pytest.raises(ValueError):
            native.evaluate(SE3.identity())
        with pytest.raises(ValueError):
            prepared.evaluate(SE3.identity())
    else:
        assert prepared.evaluate(SE3.identity()) == native.evaluate(SE3.identity())
    assert runtime.execution().cuda_calls == 0
    assert runtime.execution().numpy_fallback_calls == 1


def test_snapshot_and_device_inputs_reused_across_uncertainty_variants():
    data = frames()
    runtime = adapter.CudaProbabilisticPoseRuntime()
    options = Options()
    first = runtime.prepare(data, options)
    expected = _PreparedPoseEvaluation(data, options).evaluate(SE3.identity())
    assert first.evaluate(SE3.identity()) == expected
    before = runtime.execution()
    second = runtime.prepare(data, replace(options, use_covariance=False))
    assert second.evaluate(SE3.identity()) == _PreparedPoseEvaluation(
        data, replace(options, use_covariance=False)
    ).evaluate(SE3.identity())
    delta = runtime.execution(before)
    assert delta.upload_bytes == 0
    assert delta.calls == delta.cuda_calls == 1
    assert runtime._device_cache_bytes <= adapter._POSE_CAPTURE_BUDGET_BYTES
    assert runtime._snapshot_bytes <= adapter._POSE_CAPTURE_BUDGET_BYTES
    data[0].correspondences[0].point_lidar_m[:] = [50.0, 50.0, 50.0]
    data[0].intrinsics.fx = 100.0
    assert first.evaluate(SE3.identity()) == expected
    alien = data[0].model_copy(deep=True)
    with pytest.raises(ValueError, match="different objects"):
        runtime.prepare([alien], options)


def test_projection_boundaries_and_overflow_match_native():
    data = frames()[:1]
    # Strict near-plane and image tests include both accepted and rejected rows.
    data[0].intrinsics.distortion_model = "none"
    points = [
        [0.0, 0.0, 1e-9],
        [0.0, 0.0, np.nextafter(1e-9, math.inf)],
        [-640.0 / 600.0, 0.0, 1.0],
        [640.0 / 600.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    for c, p in zip(data[0].correspondences, points, strict=False):
        c.point_lidar_m = p
    native = _PreparedPoseEvaluation(data, Options())
    prepared = adapter.CudaProbabilisticPoseRuntime().prepare(data, Options())
    assert prepared.evaluate(SE3.identity()) == native.evaluate(SE3.identity())
    data[0].intrinsics.distortion_model = "radial-tangential"
    data[0].correspondences[6].point_lidar_m = [1e60, 0.0, 1.0]
    native = _PreparedPoseEvaluation(data, Options())
    runtime = adapter.CudaProbabilisticPoseRuntime()
    prepared = runtime.prepare(data, Options())
    with pytest.raises(OverflowError):
        native.evaluate(SE3.identity())
    with pytest.raises(OverflowError):
        prepared.evaluate(SE3.identity())
    assert runtime.execution().cuda_calls == runtime.execution().numpy_fallback_calls == 1


def test_ablation_writes_schema_valid_execution_lineage_and_native_options(tmp_path: Path):
    data = frames()
    correspondence, problem = tmp_path / "correspondence.json", tmp_path / "problem.json"
    _artifact(data).save(correspondence)
    _problem(data).save(problem)
    definition, _ = run_probabilistic_refinement_ablation(
        correspondence,
        problem,
        result_directory=tmp_path / "results",
        command="pytest CUDA ablation",
        split_seeds=(0, 1),
        bootstrap_samples=10,
        evaluation_backend="cuda",
        base_options=Options(max_evaluations=13),
    )
    assert len(definition.trials) == 8
    outputs = list((tmp_path / "results").glob("*.yaml"))
    assert len(outputs) == 8
    uploads = []
    for output in outputs:
        assert validate_file(output, "probabilistic-refinement-result").valid
        value = load_probabilistic_refinement_result(output)
        assert value.provenance.execution.cuda_calls > 0
        assert "evaluation_backend" not in value.options
        uploads.append(value.provenance.execution.upload_bytes)
    assert 0 in uploads


@pytest.mark.parametrize("limit", [13, 14, 25, 37])
def test_pose_batches_preserve_partial_evaluation_limits(limit):
    data = frames()
    options = Options(max_evaluations=limit)
    initial = SE3((0.02, 0.01, -0.03), (0.0, 0.0, 0.0, 1.0))
    native = Refiner().solve(data, initial, options)
    cuda = Refiner(evaluation_backend="cuda").solve(data, initial, options)
    assert replace(cuda, execution=None) == native
    assert len(cuda.trace) <= limit
    assert cuda.execution.kernel_launches < cuda.execution.cuda_calls


def test_pose_batch_scratch_budget_chunks_without_dropping_candidates(monkeypatch):
    data = frames()
    total = sum(len(f.correspondences) for f in data)
    monkeypatch.setattr(adapter, "_POSE_CAPTURE_BUDGET_BYTES", total * (384 + 48))
    runtime = adapter.CudaProbabilisticPoseRuntime()
    prepared = runtime.prepare(data, Options())
    poses = tuple(SE3((0.01 * i, -0.005 * i, 0.0), (0.0, 0.0, 0.0, 1.0)) for i in range(5))
    native = _PreparedPoseEvaluation(data, Options()).evaluate_many(poses)
    assert prepared.evaluate_many(poses) == native
    stats = runtime.execution()
    assert stats.cuda_calls == stats.calls == 5
    assert stats.kernel_launches == 3
    assert stats.pose_upload_bytes == 5 * 11 * 8
    assert len(prepared._batch_out) == 2
