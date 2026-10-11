"""GPU checks for conservative pruning, persistent scratch, and fallbacks."""

from itertools import product

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
    evaluate_depth_to_depth_mi_batch,
)
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta
from calibrex.solvers.cuda_depth_to_depth_adapter import CudaDepthPairProjector
from calibrex.solvers.cuda_depth_to_depth_batch import conservative_pinhole_mask


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires an NVIDIA CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("requires an NVIDIA CUDA device")


def mixed_frames():
    rng = np.random.default_rng(622)
    frames = []
    for index, kind in enumerate(("pinhole", "double_sphere", "mei", "pinhole", "pinhole")):
        camera = DepthToDepthCameraModel(
            40+index*2, 32+index, 23., 24., 19.5, 15.5, projection=kind,
            xi=.35 if kind == "double_sphere" else 1.4, alpha=.55,
            distortion=(.01, .02, .001, -.002),
        )
        points = rng.uniform((-20., -20., -20.), (20., 20., 25.), (1600+index*100, 3))
        points[1] = points[0]
        points[3] = [-1.01, 0., 2.]
        points[4] = [-.99, 0., 2.]
        if index == 3:
            points = np.empty((0, 3))
        depth = rng.uniform(.1, 15., (camera.height, camera.width))
        depth[::3, ::5] = np.nan
        depth[::7, ::4] = 0.
        if index == 4:
            depth[:] = np.nan
        frames.append(DepthToDepthObservation(str(index), depth, points, camera))
    return tuple(frames)


@pytest.mark.parametrize("z_buffer", [False, True])
def test_gpu_pruning_and_scratch_reuse_match_numpy_for_all_bound_corners(z_buffer):
    frames = mixed_frames()
    initial = SE3((1., 0., 0.), (0., 0., 0., 1.))
    cuda = CudaDepthPairProjector(frames)
    cuda.prepare_pose_search(frames, initial, rotation_bound_deg=2., translation_bound_m=1.)
    batch = cuda._pose_batch
    assert batch is not None
    assert batch.retained_points < batch.original_points
    assert cuda.cached_input_bytes+batch.input_bytes <= cuda.cache_budget_bytes
    assert batch.workspace_bytes <= 1024**3
    for frame, inputs in zip(frames, batch.inputs, strict=True):
        mask = conservative_pinhole_mask(frame, initial, 2., 1.)
        np.testing.assert_array_equal(cp.asnumpy(inputs.points), frame.lidar_points[mask])
        np.testing.assert_array_equal(cp.asnumpy(inputs.lidar_range), frame.lidar_range_m[mask])
    corners = tuple(product((-1., 1.), repeat=3))
    poses = tuple(apply_local_se3_delta(initial, tuple(2.*v for v in rotation), translation)
                  for rotation in corners for translation in corners)
    poses += (initial,)*3  # Partial final device chunk and previously occupied pixels.
    options = DepthToDepthOptions(
        histogram_bins=16, min_visible_points=4, use_z_buffer=z_buffer,
        camera_depth_range=(1., 12.), lidar_range_m=(1., 24.),
    )
    expected = tuple(evaluate_depth_to_depth_mi(frames, pose, options) for pose in poses)
    for stream in (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)):
        with stream:
            actual = evaluate_depth_to_depth_mi_batch(frames, poses, options, projector=cuda)
            assert actual == expected
    stats = cuda.transfer_statistics()
    assert stats["histogram_download_calls"] == 2*((len(poses)+batch.capacity-1)//batch.capacity)
    assert stats["histogram_download_bytes"] == 2*len(poses)*len(frames)*(16*16+2)*8
    assert stats["evaluation_input_upload_calls"] == 0


def test_outside_search_and_different_frames_use_full_inputs():
    frames = mixed_frames()
    cuda = CudaDepthPairProjector(frames)
    cuda.prepare_pose_search(frames, SE3.identity(), rotation_bound_deg=2., translation_bound_m=1.)
    assert cuda._pose_batch is not None
    outside = apply_local_se3_delta(SE3.identity(), (50., 0., 0.), (4., 0., 0.))
    for selected, poses in ((frames, (outside,)), (frames[::-1], (SE3.identity(),))):
        assert evaluate_depth_to_depth_mi_batch(selected, poses, projector=cuda) == tuple(
            evaluate_depth_to_depth_mi(selected, pose) for pose in poses
        )


@pytest.mark.parametrize("cache_bytes,workspace", [(0, 1024**3), (268435456, 0), (268435456, 1)])
def test_missing_cache_or_scratch_budget_falls_back_exactly(cache_bytes, workspace):
    frames = mixed_frames()
    cuda = CudaDepthPairProjector(
        frames, cache_bytes=cache_bytes, candidate_workspace_bytes=workspace,
    )
    cuda.prepare_pose_search(frames, SE3.identity(), rotation_bound_deg=2., translation_bound_m=1.)
    assert cuda._pose_batch is None
    poses = (SE3.identity(),)*5
    assert evaluate_depth_to_depth_mi_batch(frames, poses, projector=cuda) == tuple(
        evaluate_depth_to_depth_mi(frames, pose) for pose in poses
    )


def test_candidate_allocation_failure_falls_back_and_download_failure_retries(monkeypatch):
    frames = mixed_frames()
    cuda = CudaDepthPairProjector(frames)
    empty = cp.empty

    def oom(*args, **kwargs):
        raise cp.cuda.memory.OutOfMemoryError(4096, 0, 1024)

    monkeypatch.setattr(cp, "empty", oom)
    cuda.prepare_pose_search(frames, SE3.identity(), rotation_bound_deg=2., translation_bound_m=1.)
    assert cuda._pose_batch is None
    monkeypatch.setattr(cp, "empty", empty)
    cuda.prepare_pose_search(frames, SE3.identity(), rotation_bound_deg=2., translation_bound_m=1.)
    assert cuda._pose_batch is not None
    asnumpy = cp.asnumpy

    def fail(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(cp, "asnumpy", fail)
    with pytest.raises(RuntimeError, match="download failed"):
        evaluate_depth_to_depth_mi_batch(frames, (SE3.identity(),), projector=cuda)
    assert cuda._pose_batch is None
    monkeypatch.setattr(cp, "asnumpy", asnumpy)
    assert evaluate_depth_to_depth_mi_batch(frames, (SE3.identity(),), projector=cuda) == (
        evaluate_depth_to_depth_mi(frames, SE3.identity()),
    )


def test_pruning_keeps_half_pixel_boundaries_and_large_finite_points():
    camera = DepthToDepthCameraModel(8, 8, 1., 1., 4., 4.)
    u = np.asarray([-.5, .5, 1.5, 7.5, np.nextafter(-.5, -np.inf), 1.e100])
    points = np.column_stack((u-4., np.zeros(len(u)), np.ones(len(u))))
    frame = DepthToDepthObservation("edges", np.ones((8, 8)), points, camera)
    cuda = CudaDepthPairProjector((frame,))
    cuda.prepare_pose_search(
        (frame,), SE3.identity(), rotation_bound_deg=.01, translation_bound_m=.01,
    )
    assert cuda._pose_batch is not None
    mask = conservative_pinhole_mask(frame, SE3.identity(), .01, .01)
    assert np.all(mask[:5])
    np.testing.assert_array_equal(cp.asnumpy(cuda._pose_batch.inputs[0].points), points[mask])
    assert evaluate_depth_to_depth_mi_batch((frame,), (SE3.identity(),), projector=cuda) == (
        evaluate_depth_to_depth_mi((frame,), SE3.identity()),
    )
