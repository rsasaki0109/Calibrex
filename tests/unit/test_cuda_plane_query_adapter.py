"""Actual-device plane identities, native boundary repair and captured map ownership."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.data import livox
from calibrex.data.adaptive_voxel import (
    AdaptivePlaneQueryContext,
    AdaptiveVoxelPlane,
    AdaptiveVoxelPlaneMap,
    AdaptiveVoxelPolicy,
    _SparsePlaneCells,
    build_adaptive_voxel_plane_map,
)
from calibrex.evaluation.packed_lidar_observations import _PackedObservationBuilder
from calibrex.solvers import cuda_plane_query_adapter as adapter


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        count = 0
    if not count:
        pytest.skip("requires a CUDA device")


def _contexts(strategy="hybrid"):
    rng = np.random.default_rng(1203)
    records = [livox.LivoxPointRecord((*p, 0.0), (0.1, -0.2, 0.9))
               for p in rng.uniform(-3, 3, (1000, 3)).tolist()]
    uniform = (livox.VoxelPlaneQueryContext(livox.build_voxel_plane_map(records, 1.0))
               if strategy != "adaptive" else None)
    adaptive = (AdaptivePlaneQueryContext(build_adaptive_voxel_plane_map(
        records, AdaptiveVoxelPolicy(min_points_per_voxel=1),
    )) if strategy != "uniform" else None)
    return uniform, adaptive


@pytest.mark.parametrize("strategy", ["uniform", "adaptive", "hybrid"])
@pytest.mark.parametrize("count", [64, 4097, 8193])
@pytest.mark.parametrize("gate", [0.01, 0.5, 1.5])
def test_random_associations_and_complete_packed_columns_match_cpu(strategy, count, gate):
    uniform, adaptive = _contexts(strategy)
    gpu = adapter.CudaPlaneQueries.capture(uniform, adaptive)
    assert gpu is not None
    points = np.random.default_rng(730).uniform(-4, 4, (count, 3))
    result = gpu.query(points, voxel_size_m=1., correspondence_gate_m=gate)
    assert result is not None
    for context, actual in zip((uniform, adaptive), result, strict=True):
        if context is None:
            assert actual is None
            continue
        kwargs = {"correspondence_gate_m": gate}
        if context is uniform:
            kwargs["voxel_size_m"] = 1.0
        expected = context._nearest_positions(points, **kwargs)
        np.testing.assert_array_equal(actual, expected)
    builder = _PackedObservationBuilder.capture(
        [tuple(p) for p in points.tolist()], uniform, adaptive,
    )
    assert builder is not None
    poses = np.tile([0., 0., 0., 0., 0., 0., 1.], (count, 1))
    kwargs = {"voxel_size_m": 1., "correspondence_gate_m": gate}
    cpu = builder.build(np.arange(count), poses, points, **kwargs)
    actual = builder.build(np.arange(count), poses, points, _position_query=gpu.query, **kwargs)
    assert cpu is not None and actual is not None
    for name in ("points", "plane_points", "normals", "poses", "sqrt_weights"):
        np.testing.assert_array_equal(getattr(cpu.arrays, name).view(np.uint64),
                                      getattr(actual.arrays, name).view(np.uint64))
    assert gpu.query_calls == 2
    assert gpu.uniform_rows == (2*count if uniform is not None else 0)
    assert gpu.adaptive_rows == (2*count if adaptive is not None else 0)


@pytest.mark.parametrize("gate", [0., .5, math.nextafter(.5, 0.), math.nextafter(.5, math.inf),
                                  float("inf"), float("nan")])
def test_uniform_first_tie_invalid_nearest_normal_and_exact_gate(gate):
    planes = {
        (-1, 0, 0): livox._PlaneGeometry((-.5, 0., 0.), (1., 0., 0.)),
        (0, 0, 0): livox._PlaneGeometry((.5, 0., 0.), None),
        (1, 0, 0): livox._PlaneGeometry((1.5, 0., 0.), (0., 1., 0.)),
        (3, 0, 0): livox._PlaneGeometry((.51, 0., 0.), (0., 0., 1.)),
    }
    context = livox.VoxelPlaneQueryContext(planes)
    gpu = adapter.CudaPlaneQueries.capture(context, None)
    points = np.asarray([(0., 0., 0.), (.5, 0., 0.), (1., 0., 0.),
                         (math.nextafter(0., 1.), 0., 0.)]*20)
    assert gpu is not None
    result = gpu.query(points, voxel_size_m=1., correspondence_gate_m=gate)
    np.testing.assert_array_equal(result[0], context._nearest_positions(
        points, voxel_size_m=1., correspondence_gate_m=gate,
    ))


def test_adaptive_ties_subnormal_distances_single_neighbor_and_inclusive_gate():
    for centroids in ((-.5, .5, math.nextafter(1., math.inf)), (1e-200,), (0.5,)):
        cells = [AdaptiveVoxelPlane(0, (math.floor(x), 0, 0), (x, 0., 0.),
                                    (0., 0., 1.), 3, 1., 1.) for x in centroids]
        planes = AdaptiveVoxelPlaneMap(tuple(cells), (0., 0., 0.),
            {0: _SparsePlaneCells({c.key:c for c in cells})})
        context = AdaptivePlaneQueryContext(planes)
        gpu = adapter.CudaPlaneQueries.capture(None, context)
        assert gpu is not None
        points = np.asarray([(0., 0., 0.), (.5, 0., 0.), (math.nextafter(0., 1.), 0., 0.)]*32)
        for gate in (.5, math.nextafter(.5, 0.), math.nextafter(.5, math.inf), 1e-200):
            result = gpu.query(points, voxel_size_m=1., correspondence_gate_m=gate)
            np.testing.assert_array_equal(result[1], context._nearest_positions(
                points, correspondence_gate_m=gate,
            ))


def test_captured_geometry_parallel_calls_and_callers_context_are_preserved():
    uniform, adaptive = _contexts()
    gpu = adapter.CudaPlaneQueries.capture(uniform, adaptive)
    points = np.random.default_rng(43).normal(size=(4097, 3))
    kwargs = {"voxel_size_m": 1., "correspondence_gate_m": .5}
    expected = gpu.query(points, **kwargs)
    device = cp.cuda.runtime.getDevice()
    with cp.cuda.Stream(non_blocking=True) as stream:
        result = gpu.query(points, **kwargs)
        assert cp.cuda.get_current_stream().ptr == stream.ptr
        assert cp.cuda.runtime.getDevice() == device
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: gpu.query(points, **kwargs), range(6)))
    for actual in (result, *results):
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
    assert gpu.query_calls == 8
    assert gpu.query_upload_bytes == 8*points.nbytes*2


def test_unsupported_maps_points_budgets_and_public_hooks_keep_cpu_path(monkeypatch):
    assert adapter.CudaPlaneQueries.capture(None, None) is None
    uniform, adaptive = _contexts()
    monkeypatch.setattr(adapter, "_MAX_MAP_BYTES", 1)
    assert adapter.CudaPlaneQueries.capture(uniform, adaptive) is None
    monkeypatch.undo()
    gpu = adapter.CudaPlaneQueries.capture(uniform, adaptive)
    assert gpu.query(np.zeros((63, 3)), voxel_size_m=1., correspondence_gate_m=1.) is None
    assert gpu.query(np.full((64, 3), np.nan), voxel_size_m=1., correspondence_gate_m=1.) is None
    assert gpu.query(np.zeros((64, 3)), voxel_size_m=0., correspondence_gate_m=1.) is None
    monkeypatch.setattr(adapter, "_MAX_QUERY_BYTES", 1)
    assert gpu.query(np.zeros((64, 3)), voxel_size_m=1., correspondence_gate_m=1.) is None
    monkeypatch.undo()
    points = [(0., 0., 0.)]*64
    builder = _PackedObservationBuilder.capture(points, uniform, adaptive)
    monkeypatch.setattr(uniform, "nearest_matches", lambda *a, **k: [])
    assert builder.build(
        np.arange(64), np.zeros((64, 7)), np.zeros((64, 3)),
        voxel_size_m=1., correspondence_gate_m=1., _position_query=gpu.query,
    ) is None
    assert gpu.query_calls == 0
