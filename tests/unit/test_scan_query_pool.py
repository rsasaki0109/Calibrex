"""Reusable CPU workers retain exact native rows and close after failures."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from scipy.spatial import cKDTree

from calibrex.solvers import scan_to_scan_odometry as solver


@pytest.mark.parametrize("k", [1, 2, 10, 17])
@pytest.mark.parametrize("count", [4095, 4096, 8201])
def test_pooled_query_rows_preserve_native_ties_gates_and_strided_inputs(
    k: int, count: int,
) -> None:
    rng = np.random.default_rng(891)
    target = rng.normal(size=(127, 3))
    target[-1] = target[0]
    storage = rng.normal(size=(count, 6))
    source = storage[:, ::2]
    source[:2] = target[0]
    tree = cKDTree(target)
    with solver._CpuQueries(4) as queries:
        for gate in (float("inf"), .5, np.nextafter(.5, 0.)):
            expected = tree.query(source, k=k, distance_upper_bound=gate, workers=1)
            actual = queries.query(tree, source, k=k, distance_upper_bound=gate)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
        empty = queries.query(tree, source[:0], k=k)
        assert empty[0].shape == ((0,) if k == 1 else (0, k))
        assert empty[1].dtype == np.dtype(np.intp)


def test_parallel_pca_preserves_every_row_including_repeated_neighborhoods() -> None:
    rng = np.random.default_rng(892)
    target = rng.normal(size=(127, 3))
    target[-1] = target[0]
    indices = rng.integers(0, len(target), size=(9001, 10), dtype=np.intp)
    indices[100:200] = indices[:100]
    indices[200:300] = 0
    options = solver.ScanOdometryOptions(query_workers=4)
    expected = solver._normals_from_neighbours(target, indices, 10, options)
    with solver._CpuQueries(4) as queries:
        actual = queries.normals(target, indices, 10, options)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_small_batches_and_serial_queries_create_no_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("small and serial work created a thread pool")

    monkeypatch.setattr(solver, "ThreadPoolExecutor", forbidden)
    points = np.arange(60., dtype=float).reshape(-1, 3)
    tree = cKDTree(points)
    for workers in (1, 8):
        with solver._CpuQueries(workers) as queries:
            _, indices = queries.query(tree, points, k=10)
            queries.normals(points, indices, 10, solver.ScanOdometryOptions())


def test_failed_parallel_pca_joins_workers_and_rejects_closed_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {thread.ident for thread in threading.enumerate()}

    def fail(*args: object) -> object:
        raise RuntimeError("intentional PCA failure")

    monkeypatch.setattr(solver, "_normals_from_neighbours", fail)
    context = solver._CpuQueries(4)
    points = np.zeros((20, 3))
    with pytest.raises(RuntimeError, match="intentional PCA failure"), context:
        context.normals(points, np.zeros((9000, 10), dtype=np.intp), 10,
                        solver.ScanOdometryOptions(query_workers=4))
    assert not [thread for thread in threading.enumerate()
                if thread.name.startswith("calibrex-scan") and thread.ident not in before]
    with pytest.raises(RuntimeError, match="closed"):
        context.query(cKDTree(points), points, k=1)
