"""Compiled CPU images, interpolation and complete calibration remain native."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("numba")
from tests.unit.test_levinson_thrun_online_solver import _frames_for_truths, _options, _transform

from calibrex.core.geometry import SE3
from calibrex.solvers import levinson_thrun_online_solver as online
from calibrex.solvers import numba_online_edge_adapter as adapter


def test_cpu_backend_and_images_work_with_cuda_imports_blocked():
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'cupy':
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
import numpy as np
from calibrex.core.geometry import SE3
from calibrex.solvers.levinson_thrun_online_solver import (
    LevinsonThrunOnlineSolver, levinson_edge_response,
)
result = LevinsonThrunOnlineSolver(evaluation_backend='numba').solve([], SE3.identity())
assert result.execution.backend == 'numba'
image = np.arange(16.0).reshape(4, 4)
actual = levinson_edge_response(image, evaluation_backend='numba', max_radius=3)
native = levinson_edge_response(image, max_radius=3)
np.testing.assert_array_equal(actual.view(np.uint64), native.view(np.uint64))
assert 'cupy' not in sys.modules
assert 'calibrex.solvers.cuda_online_edge_adapter' not in sys.modules
"""
    value = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False
    )
    assert value.returncode == 0, value.stdout + value.stderr


@pytest.mark.parametrize("shape", [(1, 1), (3, 17), (48, 64)])
@pytest.mark.parametrize(
    "radius,gamma,alpha",
    [(0, 0.98, 0.0), (1, 0.9999999999999999, 1.0), (32, 0.98, 1 / 3), (2, 1e-308, 0.5)],
)
def test_separable_image_preserves_native_float_bits(shape, radius, gamma, alpha):
    image = np.random.default_rng(918).uniform(-255.0, 255.0, shape)[:, ::-1]
    expected = online.levinson_edge_response(image, alpha=alpha, gamma=gamma, max_radius=radius)
    actual = online.levinson_edge_response(
        image, alpha=alpha, gamma=gamma, max_radius=radius, evaluation_backend="numba"
    )
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def _frame(kind, axes):
    rng = np.random.default_rng(405)
    points = rng.normal(0.0, 0.5, (258, 3))[::2]
    points[:, 2] += 5.0
    points[:7] = (
        (0, 0, 1e-6),
        (0, 0, np.nextafter(1e-6, np.inf)),
        (-30 / 22, 0, 1),
        ((63 - 30) / 22, 0, 1),
        (0, (47 - 20) / 24, 1),
        (0, 0, -1),
        (0, 0, 0),
    )
    weights = rng.uniform(0.0, 2.0, 258)[::2]
    image = rng.uniform(0.0, 255.0, (48, 64))[:, ::-1]
    if axes == "forward_x_left_y_up_z":
        points = np.column_stack((points[:, 2], -points[:, 0], -points[:, 1]))
    if kind == "fortran":
        points = np.asfortranarray(points)
    elif kind == "reversed":
        points, weights = points[::-1], weights[::-1]
    elif kind == "column_stride":
        storage = np.zeros((len(points), 6))
        storage[:, ::2] = points
        points = storage[:, ::2]
    elif kind == "empty":
        points, weights = points[:0], weights[:0]
    elif kind == "tiny":
        weights *= 1e-310
        image *= 1e-12
    elif kind == "zero":
        weights[:] = 0.0
    camera = online.LevinsonCameraModel(
        64,
        48,
        22.0,
        24.0,
        30.0,
        20.0,
        axes=axes,
        projection="opencv_fisheye" if kind == "fisheye" else "pinhole",
        distortion=(0.03, -0.001, 0.0, 0.0),
    )
    return online.LevinsonOnlineFrame("frame", image, points, weights, camera)


@pytest.mark.parametrize("axes", ["optical_z", "forward_x_left_y_up_z"])
@pytest.mark.parametrize(
    "kind", ["strided", "fortran", "reversed", "column_stride", "empty", "tiny", "zero", "fisheye"]
)
def test_interpolation_support_and_full_statistics_are_bitwise_native(kind, axes):
    frame = _frame(kind, axes)
    runtime = adapter.NumbaOnlineEdgeRuntime()
    for pose in (SE3.identity(), _transform(0.01, -0.03, 0.02, 0.07, -0.05, 0.08)):
        expected = online._project_response(frame, pose)
        actual = runtime.project(frame, pose)
        for first, second in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(first.view(np.uint64), second.view(np.uint64))
        assert runtime.evaluate([frame], pose) == online.evaluate_levinson_objective([frame], pose)
    counts = runtime.execution()
    assert counts.objective_evaluations == 2
    assert counts.point_rows == 2 * len(frame.lidar_points)
    if kind == "fisheye":
        assert counts.numba_frame_pose_evaluations == 0
        assert counts.native_frame_pose_evaluations == 4


@pytest.mark.parametrize("kind", ["optimum", "offset", "sudden", "gradual"])
def test_complete_tracking_holdout_controls_and_curvature_are_native(kind):
    if kind == "optimum":
        truths = [SE3.identity()] * 12
    elif kind == "offset":
        truths = [_transform(0.02, -0.01, 0.01, 0.05, -0.05, 0.2)] * 16
    elif kind == "sudden":
        truths = [SE3.identity()] * 10 + [_transform(0.1, 0.0, 0.0, 0.0, 0.0, 0.25)] * 10
    else:
        truths = [_transform(0, 0, 0, 0.02 * i, 0.02 * i, 0.02 * i) for i in range(20)]
    frames = _frames_for_truths(truths)
    options = _options(tracking_enabled=True)
    original = online._project_response
    native = online.LevinsonThrunOnlineSolver().solve(frames, SE3.identity(), options)
    actual = online.LevinsonThrunOnlineSolver(evaluation_backend="numba").solve(
        frames, SE3.identity(), options
    )
    assert replace(actual, execution=None) == native
    assert online._project_response is original
    assert {k: v for k, v in actual.as_dict().items() if k != "execution"} == native.as_dict()
    execution = actual.execution
    assert execution.backend == "numba"
    assert execution.grid_pose_evaluations == 728 * len(actual.timeline)
    assert execution.objective_evaluations == 729 * len(actual.timeline) + 88
    assert execution.numba_frame_pose_evaluations > 0
    assert execution.native_frame_pose_evaluations == 0


def test_parallel_numba_preparation_preserves_order_and_frames():
    from calibrex.solvers.levinson_online_preparation import (
        LevinsonOnlineImage,
        prepare_levinson_online_frames,
    )

    def load(index):
        frame = _frame("strided", "optical_z")
        return LevinsonOnlineImage(
            str(index),
            frame.edge_response,
            frame.lidar_points,
            frame.discontinuity_weights,
            frame.camera,
        )

    native = prepare_levinson_online_frames(list(range(5)), load, workers=3, max_radius=8)
    actual = prepare_levinson_online_frames(
        list(range(5)), load, workers=3, max_radius=8, evaluation_backend="numba"
    )
    assert [frame.frame_id for frame in actual] == [frame.frame_id for frame in native]
    for first, second in zip(actual, native, strict=True):
        np.testing.assert_array_equal(
            first.edge_response.view(np.uint64), second.edge_response.view(np.uint64)
        )


def test_compilation_or_execution_failure_is_not_silently_retried(monkeypatch):
    def fail(*args):
        raise RuntimeError("compiled CPU execution probe")

    monkeypatch.setattr(adapter, "_interpolate", fail)
    with pytest.raises(RuntimeError, match="execution probe"):
        adapter.NumbaOnlineEdgeRuntime().evaluate([_frame("strided", "optical_z")], SE3.identity())
    monkeypatch.setattr(adapter, "_image_response", fail)
    with pytest.raises(RuntimeError, match="execution probe"):
        online.levinson_edge_response(np.ones((8, 8)), evaluation_backend="numba")
