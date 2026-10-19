"""Real-device raw-score bounds, native decisions, and owned-data limits."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp
from tests.unit.test_levinson_thrun_online_solver import _frames_for_truths, _options, _transform

from calibrex.core.geometry import SE3
from calibrex.solvers import cuda_online_edge_adapter as adapter
from calibrex.solvers import levinson_thrun_online_solver as online


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires a CUDA device")


@pytest.mark.parametrize("shape", [(1, 1), (3, 17), (48, 64)])
@pytest.mark.parametrize(
    "radius,gamma,alpha",
    [(0, 0.98, 0.0), (1, 0.9999999999999999, 1.0), (32, 0.98, 1 / 3), (2, 1e-308, 0.5)],
)
def test_image_spilloff_preserves_native_float_bits(shape, radius, gamma, alpha):
    rng = np.random.default_rng(918)
    image = rng.uniform(-255.0, 255.0, shape)[:, ::-1]
    expected = online.levinson_edge_response(image, alpha=alpha, gamma=gamma, max_radius=radius)
    actual = online.levinson_edge_response(
        image, alpha=alpha, gamma=gamma, max_radius=radius, evaluation_backend="cuda"
    )
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_image_device_budget_uses_native_arithmetic(monkeypatch):
    monkeypatch.setattr(adapter, "_OWNED_DEVICE_LIMIT", 128)
    image = np.arange(64.0).reshape(8, 8)
    actual = online.levinson_edge_response(image, evaluation_backend="cuda")
    expected = online.levinson_edge_response(image)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_real_device_preparation_overlaps_cpu_loaders_without_changing_frames():
    from calibrex.solvers.levinson_online_preparation import (
        LevinsonOnlineImage,
        prepare_levinson_online_frames,
    )

    source = _frames()

    def load(index):
        frame = source[index]
        return LevinsonOnlineImage(
            frame.frame_id,
            frame.edge_response,
            frame.lidar_points,
            frame.discontinuity_weights,
            frame.camera,
        )

    cpu = prepare_levinson_online_frames(list(range(3)), load, workers=3, max_radius=8)
    gpu = prepare_levinson_online_frames(
        list(range(3)), load, workers=3, evaluation_backend="cuda", max_radius=8
    )
    for left, right in zip(cpu, gpu, strict=True):
        assert left.frame_id == right.frame_id
        for first, second in (
            (left.edge_response, right.edge_response),
            (left.lidar_points, right.lidar_points),
            (left.discontinuity_weights, right.discontinuity_weights),
        ):
            np.testing.assert_array_equal(first.view(np.uint64), second.view(np.uint64))


def _frames(kind: str = "random", count: int = 128) -> list[online.LevinsonOnlineFrame]:
    rng = np.random.default_rng(405)
    camera = online.LevinsonCameraModel(
        64,
        48,
        22.0,
        24.0,
        30.0,
        20.0,
        axes="forward_x_left_y_up_z" if kind == "forward" else "optical_z",
        projection="opencv_fisheye" if kind == "fisheye" else "pinhole",
    )
    frames = []
    for index in range(3):
        points = rng.normal(0.0, 0.5, (2 * count, 3))[::2]
        points[:, 2] += 5.0
        weights = rng.uniform(0.0, 2.0, 2 * count)[::2]
        image = rng.uniform(0.0, 255.0, (48, 64))[:, ::-1]
        if kind == "constant":
            image[:] = 17.0
        if kind == "zero":
            image[:] = 0.0
        if kind == "near_tie":
            image[:] = 1.0 + rng.uniform(0.0, 1e-12, image.shape)
        if kind == "tiny_scores":
            image *= 1e-112
            weights *= 1e-200
        if kind == "borders":
            points[:4] = [
                (0.0, 0.0, 1e-6),
                (-30.0 / 22.0, 0.0, 1.0),
                ((63.0 - 30.0) / 22.0, 0.0, 1.0),
                (0.0, (47.0 - 20.0) / 24.0, 1.0),
            ]
        if kind == "forward":
            points = np.column_stack((points[:, 2], -points[:, 0], -points[:, 1]))
        if kind == "large":
            points *= 1e13
        if kind == "strided":
            points = np.asfortranarray(points)
        frames.append(online.LevinsonOnlineFrame(str(index), image, points, weights, camera))
    return frames


@pytest.mark.parametrize(
    "kind",
    [
        "random",
        "constant",
        "zero",
        "borders",
        "forward",
        "fisheye",
        "small",
        "strided",
        "large",
        "near_tie",
        "tiny_scores",
        "tiny_grid",
    ],
)
def test_complete_step_decisions_are_native(kind):
    frames = _frames(kind, 17 if kind == "small" else 128)
    options = online.LevinsonThrunOnlineOptions(
        window_size=3,
        min_train_frames=3,
        min_holdout_frames=0,
        translation_grid_step_m=1e-300 if kind == "tiny_grid" else 0.01,
        rotation_grid_step_deg=1e-300 if kind == "tiny_grid" else 0.05,
    )
    pose = SE3((1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)) if kind == "tiny_grid" else SE3.identity()
    expected = online._evaluate_online_step(frames, pose, options)
    runtime = adapter.CudaOnlineEdgeRuntime(frames)
    actual = online._evaluate_online_step(runtime.frames, pose, options, scorer=runtime)
    assert actual == expected
    execution = runtime.execution()
    assert execution.grid_pose_evaluations == 728
    assert execution.max_device_snapshot_bytes <= execution.device_snapshot_limit_bytes
    assert execution.max_scratch_bytes < execution.owned_device_limit_bytes
    if kind in {"constant", "near_tie", "tiny_grid"}:
        assert execution.native_comparison_poses > 0
    if kind == "tiny_scores":
        assert execution.native_comparison_poses == 728
    if kind == "borders":
        assert execution.native_boundary_poses > 0
    if kind in {"fisheye", "small", "large"}:
        assert execution.cuda_frame_pose_evaluations == 0
        assert execution.native_frame_pose_evaluations == 3 * 728


@pytest.mark.parametrize("count", [32, 33, 65, 257])
def test_native_raw_scores_are_enclosed_by_device_bounds(count):
    runtime = adapter.CudaOnlineEdgeRuntime(_frames("borders", count))
    frame = runtime.frames[0]
    data = runtime._device_frame(frame)
    assert data is not None
    rng = np.random.default_rng(904)
    poses = [online._transform_from_parameters(value) for value in rng.normal(0.0, 0.03, (65, 6))]
    rotations = cp.asarray(
        np.asarray([online._rotation_matrix(pose) for pose in poses]).reshape(-1, 3)
    )
    translations = cp.asarray(np.asarray([pose.translation_m for pose in poses]))
    stats = runtime._score_chunk(data, rotations, translations)
    checked = 0
    for pose, (raw, error, boundary) in zip(poses, stats, strict=True):
        native = online.evaluate_levinson_objective([frame], pose).raw_objective
        if not boundary:
            assert raw - error <= native <= raw + error
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("kind", ["optimum", "offset", "sudden", "gradual"])
def test_full_calibration_controls_and_curvature_are_native(kind):
    if kind == "optimum":
        truths = [SE3.identity()] * 12
    elif kind == "offset":
        truths = [_transform(0.02, -0.01, 0.01, 0.05, -0.05, 0.2)] * 16
    elif kind == "sudden":
        truths = [SE3.identity()] * 10 + [_transform(0.1, 0.0, 0.0, 0.0, 0.0, 0.25)] * 10
    else:
        truths = [_transform(0.0, 0.0, 0.0, 0.02 * i, 0.02 * i, 0.02 * i) for i in range(20)]
    frames = _frames_for_truths(truths)
    options = _options(tracking_enabled=True)
    native = online.LevinsonThrunOnlineSolver().solve(frames, SE3.identity(), options)
    actual = online.LevinsonThrunOnlineSolver(evaluation_backend="cuda").solve(
        frames,
        SE3.identity(),
        options,
    )
    assert replace(actual, execution=None) == native
    value = actual.as_dict()
    execution = value.pop("execution")
    assert value == native.as_dict()
    assert actual.execution is not None
    assert actual.execution.grid_pose_evaluations == 728 * len(actual.timeline)
    assert actual.execution.cuda_frame_pose_evaluations > 0
    assert execution == asdict(actual.execution)


def test_snapshots_are_private_and_only_owned_frames_can_be_evaluated():
    source = _frames()
    expected = online._evaluate_online_step(
        source, SE3.identity(), online.LevinsonThrunOnlineOptions()
    )
    runtime = adapter.CudaOnlineEdgeRuntime(source)
    identity = runtime.execution().capture_sha256
    source[0].edge_response[:] = 0.0
    source[0].lidar_points[:] = 3.0
    source[0].discontinuity_weights[:] = 0.0
    assert source[0].edge_response.flags.writeable
    assert not runtime.frames[0].edge_response.flags.writeable
    assert not runtime.frames[0].lidar_points.flags.writeable
    actual = online._evaluate_online_step(
        runtime.frames,
        SE3.identity(),
        online.LevinsonThrunOnlineOptions(),
        scorer=runtime,
    )
    assert actual == expected
    assert runtime.execution().capture_sha256 == identity
    assert adapter.CudaOnlineEdgeRuntime(source).execution().capture_sha256 != identity
    center = online.evaluate_levinson_objective(runtime.frames, SE3.identity())
    with pytest.raises(ValueError, match="owned input snapshot"):
        runtime.raw_scores(source, [SE3.identity()], center)
    with pytest.raises(ValueError, match="728"):
        runtime.raw_scores(runtime.frames, [SE3.identity()] * 729, center)


@pytest.mark.parametrize("budget", [4096, 58000])
def test_device_budget_fallback_or_eviction_preserves_decisions(monkeypatch, budget):
    monkeypatch.setattr(adapter, "_SNAPSHOT_LIMIT", budget)
    frames = _frames()
    runtime = adapter.CudaOnlineEdgeRuntime(frames)
    options = online.LevinsonThrunOnlineOptions()
    expected = online._evaluate_online_step(frames, SE3.identity(), options)
    for _ in range(2):
        assert (
            online._evaluate_online_step(
                runtime.frames,
                SE3.identity(),
                options,
                scorer=runtime,
            )
            == expected
        )
    execution = runtime.execution()
    assert execution.max_device_snapshot_bytes <= budget
    if budget == 4096:
        assert execution.cuda_frame_pose_evaluations == 0
    else:
        assert execution.image_upload_bytes > sum(frame.edge_response.nbytes for frame in frames)


def test_cuda_failure_is_not_silently_retried_on_cpu(monkeypatch):
    runtime = adapter.CudaOnlineEdgeRuntime(_frames())

    def fail(*args):
        raise RuntimeError("CUDA execution probe")

    monkeypatch.setattr(runtime, "_score_chunk", fail)
    with pytest.raises(RuntimeError, match="CUDA execution probe"):
        online._evaluate_online_step(
            runtime.frames,
            SE3.identity(),
            online.LevinsonThrunOnlineOptions(),
            scorer=runtime,
        )


def test_device_unavailable_is_an_explicit_error(monkeypatch):
    monkeypatch.setattr(cp.cuda.runtime, "getDeviceCount", lambda: 0)
    with pytest.raises(ValueError, match="available CUDA device"):
        adapter.CudaOnlineEdgeRuntime(_frames())
