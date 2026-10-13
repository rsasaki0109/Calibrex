"""Numerical and recovery checks that run only with the optional CUDA adapter."""

import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cupy")

import cupy as cp

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    DepthToDepthOptions,
    evaluate_depth_to_depth_mi,
    project_depth_pairs,
)
from calibrex.solvers.borer_six_dof_solver import (
    BorerSixDofOptions,
    BorerSixDofSolver,
    apply_local_se3_delta,
)
from calibrex.solvers.cuda_depth_to_depth_adapter import (
    CUDA_DEPTH_PAIR_PROJECTOR_VERSION,
    CudaDepthPairProjector,
)
from calibrex.solvers.depth_to_depth_runtime import DEFAULT_CUDA_CACHE_BYTES


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip("requires an NVIDIA CUDA device")


@pytest.mark.parametrize("projection", ["pinhole", "double_sphere", "mei"])
@pytest.mark.parametrize("use_z_buffer", [True, False])
def test_cuda_projection_matches_numpy(projection: str, use_z_buffer: bool) -> None:
    camera = DepthToDepthCameraModel(
        width=96,
        height=80,
        fx=58.0,
        fy=57.0,
        cx=47.5,
        cy=39.5,
        projection=projection,  # type: ignore[arg-type]
        xi=0.35 if projection == "double_sphere" else 1.4,
        alpha=0.55,
        distortion=(0.01, 0.02, 0.001, -0.002),
    )
    rng = np.random.default_rng(20260811)
    points = rng.uniform((-12.0, -8.0, -4.0), (12.0, 8.0, 25.0), size=(20_000, 3))
    points[1] = points[0]
    depth = rng.uniform(0.1, 80.0, size=(camera.height, camera.width))
    depth[::11, ::13] = np.nan
    depth[::7, ::19] = 0.0
    observation = DepthToDepthObservation(projection, depth, points, camera)
    # Exercise both prepared snapshots and the bounded uncached path.
    cuda = CudaDepthPairProjector([observation])
    uncached = CudaDepthPairProjector([observation], cache_bytes=0)
    assert uncached.cached_input_bytes == 0
    assert cuda.cached_input_bytes == (
        points.nbytes + observation.depth_map.nbytes + observation.lidar_range_m.nbytes
    )
    for delta in [(0.0, 0.0, 0.0), (1.2, -0.7, 0.4), (-2.0, 1.0, -0.5)]:
        transform = apply_local_se3_delta(SE3.identity(), delta, (0.12, -0.04, 0.08))
        expected = project_depth_pairs(observation, transform, use_z_buffer=use_z_buffer)
        for projector in (cuda, uncached):
            actual = projector(observation, transform, use_z_buffer=use_z_buffer)
            assert actual.projected_count_before_visibility == (
                expected.projected_count_before_visibility
            )
            for name in ("camera_depth", "lidar_range_m", "pixel_u", "pixel_v"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        options = DepthToDepthOptions(
            min_visible_points=4,
            use_z_buffer=use_z_buffer,
            camera_depth_range=(0.0, 80.0),
            lidar_range_m=(0.0, 35.0),
        )
        expected_histogram = np.histogram2d(
            expected.camera_depth,
            expected.lidar_range_m,
            bins=options.histogram_bins,
            range=((0.0, 80.0), (0.0, 35.0)),
        )[0]
        actual_histogram = cuda.histogram_depth_pairs(
            observation,
            transform,
            use_z_buffer=use_z_buffer,
            bins=options.histogram_bins,
            camera_range=(0.0, 80.0),
            lidar_range=(0.0, 35.0),
        )
        np.testing.assert_array_equal(actual_histogram.histogram, expected_histogram)
        assert actual_histogram.visible_point_count == expected.camera_depth.size
        assert evaluate_depth_to_depth_mi([observation], transform, options, projector=cuda) == (
            evaluate_depth_to_depth_mi([observation], transform, options)
        )


def test_cuda_visibility_ties_use_first_input_not_smallest_lidar_range() -> None:
    camera = DepthToDepthCameraModel(16, 12, 8.0, 8.0, 8.0, 6.0)
    # Equal camera ranges with different LiDAR ranges, both at pixel (8, 6).
    points = np.asarray([[-1.01, 0.0, 2.0], [-0.99, 0.0, 2.0]])
    transform = SE3(translation_m=(1.0, 0.0, 0.0), rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0))
    observation = DepthToDepthObservation("ties", np.ones((12, 16)), points, camera)
    cuda = CudaDepthPairProjector([observation, observation])
    assert cuda.cached_input_bytes == (
        points.nbytes + observation.depth_map.nbytes + observation.lidar_range_m.nbytes
    )
    for _ in range(8):
        actual = cuda(observation, transform)
        assert actual.projected_count_before_visibility == 2
        np.testing.assert_array_equal(actual.lidar_range_m, observation.lidar_range_m[:1])
        np.testing.assert_array_equal(actual.pixel_u, [8])
        np.testing.assert_array_equal(actual.pixel_v, [6])


def test_cuda_rounding_visibility_and_invalid_depth_gates() -> None:
    camera = DepthToDepthCameraModel(8, 8, 1.0, 1.0, 4.0, 4.0)
    u = np.asarray([0.5, 1.5, -0.5, 7.5, -0.5001, 7.4999, 3.5, 4.5])
    points = np.column_stack((u - 4.0, np.zeros(len(u)), np.ones(len(u))))
    points = np.concatenate((points, [[0.0, 0.0, -1.0], [1.0e100, 0.0, 1.0]]))
    depth = np.ones((8, 8))
    depth[4, 0] = np.nan
    depth[4, 2] = -1.0
    observation = DepthToDepthObservation("boundaries", depth, points, camera)
    cuda = CudaDepthPairProjector([observation])
    for use_z_buffer in (True, False):
        expected = project_depth_pairs(observation, SE3.identity(), use_z_buffer=use_z_buffer)
        actual = cuda(observation, SE3.identity(), use_z_buffer=use_z_buffer)
        assert actual.projected_count_before_visibility == (
            expected.projected_count_before_visibility
        )
        np.testing.assert_array_equal(actual.pixel_u, expected.pixel_u)
        np.testing.assert_array_equal(actual.lidar_range_m, expected.lidar_range_m)


def test_cuda_empty_cloud_and_no_projected_points() -> None:
    camera = DepthToDepthCameraModel(16, 12, 8.0, 8.0, 7.5, 5.5)
    cuda = CudaDepthPairProjector()
    for points in (np.empty((0, 3)), np.asarray([[0.0, 0.0, -1.0]])):
        observation = DepthToDepthObservation("empty", np.ones((12, 16)), points, camera)
        actual = cuda(observation, SE3.identity())
        assert actual.projected_count_before_visibility == 0
        assert actual.camera_depth.size == 0
        assert actual.pixel_u.dtype == np.int64
    with pytest.raises(ValueError, match="cache_bytes"):
        CudaDepthPairProjector(cache_bytes=-1)


@pytest.mark.parametrize("use_z_buffer", [True, False])
def test_cuda_histogram_matches_numpy_edges_and_denominators(use_z_buffer: bool) -> None:
    camera = DepthToDepthCameraModel(8, 8, 1.0, 1.0, 4.0, 4.0)
    # Exact edges, immediately adjacent values, points outside histogram ranges,
    # and zero LiDAR range made projectable by a camera translation.
    z = np.asarray(
        [0.0, 0.5, 1.0, np.nextafter(2.0, 0.0), 2.0, 3.0, 8.0, 9.0, np.nextafter(9.0, np.inf), 10.0]
    )
    points = np.column_stack((np.zeros(len(z)), np.zeros(len(z)), z))
    transform = SE3(translation_m=(0.0, 0.0, 1.0), rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0))
    for value in (np.nan, np.inf, -1.0, 0.0, 0.5, 1.0, 2.0, 8.0, 9.0, np.nextafter(9.0, np.inf)):
        depth = np.full((8, 8), value)
        observation = DepthToDepthObservation("edges", depth, points, camera)
        cuda = CudaDepthPairProjector([observation])
        pairs = project_depth_pairs(observation, transform, use_z_buffer=use_z_buffer)
        expected = np.histogram2d(
            pairs.camera_depth,
            pairs.lidar_range_m,
            bins=8,
            range=((1.0, 9.0), (1.0, 9.0)),
        )[0]
        for projector in (cuda, CudaDepthPairProjector(cache_bytes=0)):
            actual = projector.histogram_depth_pairs(
                observation,
                transform,
                use_z_buffer=use_z_buffer,
                bins=8,
                camera_range=(1.0, 9.0),
                lidar_range=(1.0, 9.0),
            )
            np.testing.assert_array_equal(actual.histogram, expected)
            assert actual.visible_point_count == pairs.camera_depth.size
            assert (
                actual.projected_count_before_visibility == pairs.projected_count_before_visibility
            )


def test_cuda_gpu_histogram_evaluator_does_not_copy_feature_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = DepthToDepthCameraModel(16, 12, 8.0, 8.0, 8.0, 6.0)
    points = np.asarray([[u / 4.0, v / 4.0, 2.0] for v in range(-3, 4) for u in range(-5, 6)])
    observation = DepthToDepthObservation("fused", np.ones((12, 16)), points, camera)
    cuda = CudaDepthPairProjector([observation])
    expected = evaluate_depth_to_depth_mi([observation], SE3.identity())

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("fused evaluation must only transfer the histogram and counts")

    monkeypatch.setattr(cuda, "_project_pairs", forbidden)
    assert evaluate_depth_to_depth_mi([observation], SE3.identity(), projector=cuda) == expected


def test_cuda_six_dof_recovery_trace_matches_numpy() -> None:
    camera = DepthToDepthCameraModel(64, 48, 45.0, 45.0, 31.5, 23.5)
    depth = np.full((48, 64), np.nan)
    points = []
    for v in range(5, 43):
        for u in range(6, 58):
            distance = (
                8.0 + 0.03 * u + 0.02 * v + 1.7 * math.sin(0.31 * u) + 1.1 * math.cos(0.27 * v)
            )
            ray = np.asarray([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0])
            ray /= np.linalg.norm(ray)
            points.append(ray * distance)
            depth[v, u] = distance
    observation = DepthToDepthObservation("recovery", depth, np.asarray(points), camera)
    cuda = CudaDepthPairProjector([observation])
    initial = apply_local_se3_delta(SE3.identity(), (1.0, 0.0, 0.0), (0.10, 0.0, 0.0))
    settings = BorerSixDofOptions(
        rotation_bound_deg=2.0,
        translation_bound_m=0.2,
        initial_rotation_step_deg=0.5,
        initial_translation_step_m=0.05,
        minimum_rotation_step_deg=0.125,
        minimum_translation_step_m=0.0125,
        max_evaluations=160,
        d2d=DepthToDepthOptions(histogram_bins=20, min_visible_points=100),
    )
    cpu_result = BorerSixDofSolver().solve([observation], initial, settings)
    gpu_result = BorerSixDofSolver().solve(
        [observation],
        initial,
        replace(settings, projector=cuda, projection_backend=cuda.identity()),
    )
    assert gpu_result.trace == cpu_result.trace
    assert gpu_result.transform_camera_lidar == cpu_result.transform_camera_lidar
    assert gpu_result.final_evaluation == cpu_result.final_evaluation
    assert gpu_result.projection_backend.startswith(CUDA_DEPTH_PAIR_PROJECTOR_VERSION)
    assert ";cupy=" in gpu_result.projection_backend
    assert ";device=" in gpu_result.projection_backend
    assert ";fp64;fmad=false" in gpu_result.projection_backend


@pytest.mark.parametrize("cache_frames", [0, 1, 2])
def test_cuda_cache_budget_counts_actual_input_transfers_and_preserves_results(
    cache_frames: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = DepthToDepthCameraModel(48, 36, 30.0, 30.0, 23.5, 17.5)
    rng = np.random.default_rng(991)
    observations = tuple(
        DepthToDepthObservation(
            str(index), rng.uniform(1.0, 20.0, (36, 48)),
            rng.uniform((-4.0, -3.0, 2.0), (4.0, 3.0, 15.0), (1000, 3)), camera,
        ) for index in range(2)
    )
    input_ids = {
        id(array) for observation in observations
        for array in (observation.lidar_points, observation.depth_map, observation.lidar_range_m)
    }
    copied_sizes = []
    original_asarray = cp.asarray

    def observe_copies(value: object, *args: object, **kwargs: object) -> object:
        if id(value) in input_ids and isinstance(value, np.ndarray):
            copied_sizes.append(value.nbytes)
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", observe_copies)
    frame_bytes = sum(array.nbytes for array in (
        observations[0].lidar_points, observations[0].depth_map, observations[0].lidar_range_m,
    ))
    cuda = CudaDepthPairProjector(observations, cache_bytes=cache_frames * frame_bytes)
    before = cuda.transfer_statistics()
    assert before["prepared_observation_count"] == cache_frames
    assert before["preparation_input_upload_bytes"] == sum(copied_sizes)
    assert sum(copied_sizes) == cache_frames * frame_bytes
    for _ in range(2):
        assert evaluate_depth_to_depth_mi(observations, SE3.identity(), projector=cuda) == (
            evaluate_depth_to_depth_mi(observations, SE3.identity())
        )
    after = cuda.transfer_statistics()
    assert after["evaluation_input_upload_calls"] == 2 * (2 - cache_frames)
    assert after["evaluation_input_upload_bytes"] == 2 * (2 - cache_frames) * frame_bytes
    assert sum(copied_sizes) == (
        after["preparation_input_upload_bytes"] + after["evaluation_input_upload_bytes"]
    )
    assert len(copied_sizes) == 3 * (
        after["preparation_input_upload_calls"] + after["evaluation_input_upload_calls"]
    )
    expected = BorerSixDofSolver().solve(
        observations, SE3.identity(), BorerSixDofOptions(max_evaluations=20),
    )
    actual = BorerSixDofSolver().solve(
        observations, SE3.identity(),
        BorerSixDofOptions(max_evaluations=20, projector=cuda, projection_backend=cuda.identity()),
    )
    assert replace(actual, projection_backend=expected.projection_backend) == expected


def test_cuda_cache_skips_large_frame_and_retains_later_fitting_frame() -> None:
    camera = DepthToDepthCameraModel(16, 12, 8.0, 8.0, 7.5, 5.5)
    small = DepthToDepthObservation("small", np.ones((12, 16)), np.ones((10, 3)), camera)
    large = DepthToDepthObservation("large", np.ones((12, 16)), np.ones((500, 3)), camera)
    later = DepthToDepthObservation("later", np.ones((12, 16)), np.ones((10, 3)), camera)
    size = small.depth_map.nbytes + small.lidar_points.nbytes + small.lidar_range_m.nbytes
    cuda = CudaDepthPairProjector([small, large, later, small], cache_bytes=2 * size)
    for observation in (small, later):
        cuda(observation, SE3.identity())
    assert cuda.transfer_statistics()["evaluation_input_upload_calls"] == 0
    cuda(large, SE3.identity())
    assert cuda.transfer_statistics()["evaluation_input_upload_calls"] == 1
    assert cuda.transfer_statistics()["prepared_observation_count"] == 2


def test_cuda_cache_identity_preserves_default_and_distinguishes_changed_budget() -> None:
    default = CudaDepthPairProjector()
    explicit_default = CudaDepthPairProjector(cache_bytes=DEFAULT_CUDA_CACHE_BYTES)
    no_cache = CudaDepthPairProjector(cache_bytes=0)
    assert default.identity() == explicit_default.identity()
    assert ";cache_bytes=" not in default.identity()
    assert no_cache.identity() == default.identity() + ";cache_bytes=0"


def test_cuda_cache_allocation_failure_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = DepthToDepthCameraModel(16, 12, 8.0, 8.0, 7.5, 5.5)
    observation = DepthToDepthObservation("oom", np.ones((12, 16)), np.ones((10, 3)), camera)

    def fail_upload(*args: object, **kwargs: object) -> object:
        raise cp.cuda.memory.OutOfMemoryError(4096, 0, 1024)

    monkeypatch.setattr(CudaDepthPairProjector, "_upload", fail_upload)
    with pytest.raises(ValueError, match="lower --cuda-cache-mib"):
        CudaDepthPairProjector([observation])


@pytest.mark.parametrize("cache_bytes", [0, DEFAULT_CUDA_CACHE_BYTES])
@pytest.mark.parametrize("z_buffer", [False, True])
def test_cuda_batched_mixed_frames_preserve_histograms_and_bound_transfers(
    cache_bytes: int, z_buffer: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(313)
    frames = []
    for index in range(35):
        kind = ("pinhole", "double_sphere", "mei")[index % 3]
        camera = DepthToDepthCameraModel(
            10 + index % 7, 8 + index % 5, 6., 5., 4., 3.,
            projection=kind, xi=0.35 if kind == "double_sphere" else 1.4,  # type: ignore[arg-type]
            alpha=0.55, distortion=(0.01, 0.02, 0.001, -0.002),
        )
        points = rng.uniform((-4., -3., -2.), (4., 3., 8.), size=(50 + index, 3))
        points[1] = points[0]
        if index in (1, 33):
            points = np.empty((0, 3))
        depth = rng.uniform(0.5, 12., size=(camera.height, camera.width))
        depth[::3, ::4] = np.nan
        if index == 10:
            depth[:] = np.nan
        frames.append(DepthToDepthObservation(str(index), depth, points, camera))
    frames = tuple(frames)
    cuda = CudaDepthPairProjector(frames, cache_bytes=cache_bytes)
    cp.cuda.get_current_stream().synchronize()
    downloads = []
    small_uploads = []
    asnumpy, asarray = cp.asnumpy, cp.asarray

    def observe_download(value, *args, **kwargs):
        downloads.append(value.shape)
        return asnumpy(value, *args, **kwargs)

    def observe_upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.shape in ((12,), (18,)):
            small_uploads.append(value.shape)
        return asarray(value, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", observe_download)
    monkeypatch.setattr(cp, "asarray", observe_upload)
    options = DepthToDepthOptions(
        histogram_bins=8, min_visible_points=4, use_z_buffer=z_buffer,
        camera_depth_range=(1., 9.), lidar_range_m=(1., 9.),
    )
    for delta in ((0., 0., 0.), (0.3, -0.2, 0.1)):
        pose = apply_local_se3_delta(SE3.identity(), delta, (0.1, -0.04, 0.02))
        expected = evaluate_depth_to_depth_mi(frames, pose, options)
        with cp.cuda.Stream(non_blocking=True):
            actual = evaluate_depth_to_depth_mi(frames, pose, options, projector=cuda)
        assert actual == expected
    assert downloads == [(32, 66), (3, 66)] * 2
    assert small_uploads == [(12,), (18,)] * 2
    statistics = cuda.transfer_statistics()
    assert statistics['histogram_download_calls'] == 4
    assert statistics['histogram_download_bytes'] == 2 * 35 * 66 * 8
    assert statistics['evaluation_input_upload_calls'] == (66 if cache_bytes == 0 else 0)


def test_cuda_batch_partial_iteration_failure_and_retry_have_no_retained_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = DepthToDepthCameraModel(16, 12, 8., 8., 7.5, 5.5)
    frame = DepthToDepthObservation(
        "same", np.ones((12, 16)), np.array([[float(i) / 10, 0., 2.] for i in range(70)]),
        camera,
    )
    frames = (frame,) * 33
    cuda = CudaDepthPairProjector(frames)
    arguments = {
        'use_z_buffer': True, 'bins': 8, 'camera_range': (0.5, 2.), 'lidar_range': (1., 10.),
    }
    iterator = cuda.histogram_depth_pairs_batch(frames, SE3.identity(), **arguments)
    next(iterator)
    assert cuda.transfer_statistics()['histogram_download_calls'] == 1
    iterator.close()
    original = cp.asnumpy

    def fail(*args, **kwargs):
        raise RuntimeError("download failure")

    monkeypatch.setattr(cp, "asnumpy", fail)
    with pytest.raises(RuntimeError, match="download failure"):
        tuple(cuda.histogram_depth_pairs_batch(frames, SE3.identity(), **arguments))
    monkeypatch.setattr(cp, "asnumpy", original)
    expected = cuda.histogram_depth_pairs(frame, SE3.identity(), **arguments)
    actual = tuple(cuda.histogram_depth_pairs_batch(frames, SE3.identity(), **arguments))
    assert len(actual) == 33
    for item in actual:
        np.testing.assert_array_equal(item.histogram, expected.histogram)
        assert item.visible_point_count == expected.visible_point_count
        assert item.projected_count_before_visibility == expected.projected_count_before_visibility


@pytest.mark.parametrize("bins,ranges", [(0, (1., 9.)), (True, (1., 9.)), (8, (9., 1.))])
def test_cuda_batch_validates_histogram_configuration(bins, ranges) -> None:
    with pytest.raises(ValueError, match="histogram"):
        tuple(CudaDepthPairProjector().histogram_depth_pairs_batch(
            [], SE3.identity(), use_z_buffer=True, bins=bins,
            camera_range=ranges, lidar_range=(1., 9.),
        ))
