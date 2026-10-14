"""Fused map operations preserve the independent previous GPU implementation."""

from __future__ import annotations

import tarfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_lidar_map_adapter import CudaMapRuntime
from calibrex.solvers.lidar_lidar_map_solver import MapSample, MapSampleOptions


@pytest.fixture(scope="module")
def previous() -> ModuleType:
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        count = 0
    if not count:
        pytest.skip("requires a CUDA device")
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/cuda_fixed_neighbors/sources-prior.tar.gz"
    )
    with tarfile.open(path) as archive:
        stream = archive.extractfile("src/calibrex/solvers/cuda_lidar_map_adapter.py")
        assert stream is not None
        source = stream.read()
    module = ModuleType("_previous_cuda_map")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("count", [17, 4097])
@pytest.mark.parametrize("precomputed", [False, True])
def test_fused_residuals_and_normal_systems_keep_previous_gpu_bytes(
    previous: ModuleType,
    count: int,
    precomputed: bool,
) -> None:
    rng = np.random.default_rng(705)
    world = rng.normal(size=(1003, 3))
    world[:501, 2] *= 0.01
    points = world[rng.integers(len(world), size=count)] + rng.normal(0, 0.02, (count, 3))
    pose = np.eye(4)
    pose[:3, 3] = [0.3, -0.1, 0.2]
    sample = MapSample(0.0, 0, pose, points, world)
    options = MapSampleOptions()
    if precomputed:
        sample.prepare(options)
    old = previous.CudaMapRuntime([sample], options)
    current = CudaMapRuntime([sample], options)
    for transform in (
        np.eye(4),
        np.array(
            [
                [1, 0, 0, 0.04],
                [0, 1, 0, -0.03],
                [0, 0, 1, 0.02],
                [0, 0, 0, 1],
            ]
        ),
    ):
        left = old.residuals(sample, transform)
        right = current.residuals(sample, transform)
        assert right.tobytes() == left.tobytes()
        expected = old.normal_equations([sample], transform)
        actual = current.normal_equations([sample], transform)
        for a, b in zip(actual, expected, strict=True):
            assert a.tobytes() == b.tobytes()
    assert current.statistics()["gpu_map_fused_residual_rows"] == 4 * count
    assert current.statistics()["gpu_map_fused_system_rows"] > 0
    assert current.statistics()["gpu_map_fused_kernel_calls"] > 0


def test_fused_empty_and_all_unmatched_rows(previous: ModuleType) -> None:
    world = np.random.default_rng(706).normal(size=(79, 3))
    options = MapSampleOptions()
    for points in (np.empty((0, 3)), np.full((31, 3), 1e5)):
        sample = MapSample(0, 0, np.eye(4), points, world)
        old = previous.CudaMapRuntime([sample], options)
        current = CudaMapRuntime([sample], options)
        assert current.residuals(sample, np.eye(4)).shape == (0,)
        for a, b in zip(
            current.normal_equations([sample], np.eye(4)),
            old.normal_equations([sample], np.eye(4)),
            strict=True,
        ):
            assert a.tobytes() == b.tobytes()
        assert current.statistics()["gpu_map_fused_system_rows"] == 0
