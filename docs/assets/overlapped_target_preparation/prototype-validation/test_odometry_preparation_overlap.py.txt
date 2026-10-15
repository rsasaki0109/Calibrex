"""Dependency-driven overlap, caller callbacks and bounded executor lifetime."""

from __future__ import annotations

from threading import Event, get_ident

import numpy as np
import pytest

from calibrex.solvers import scan_to_scan_odometry as solver


def _cloud() -> np.ndarray:
    return np.random.default_rng(82).normal(size=(120, 3))


class _Target:
    def __init__(self) -> None:
        self.sources: list[np.ndarray] = []

    def register(self, source: np.ndarray, initial: np.ndarray) -> solver.ScanRegistration:
        self.sources.append(source.copy())
        return solver.ScanRegistration(initial.copy(), True, 1, len(source), 0., 1., "fake_cuda")

    def statistics(self) -> dict[str, int]:
        return {"prepared_targets": 1}


def test_target_and_source_overlap_without_moving_callbacks_off_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_started, source_started = Event(), Event()
    caller = get_ident()
    prepared = _Target()
    worker_threads: list[int] = []
    target_clouds: list[np.ndarray] = []

    def factory(target: np.ndarray, options: solver.ScanOdometryOptions) -> _Target:
        worker_threads.append(get_ident())
        target_clouds.append(target.copy())
        target_started.set()
        assert source_started.wait(5), "source work did not overlap target preparation"
        return prepared

    def model(time: float, offsets: np.ndarray) -> np.ndarray:
        assert get_ident() == caller
        assert target_started.wait(5), "target preparation was not scheduled"
        source_started.set()
        return np.broadcast_to(np.eye(3), (len(offsets), 3, 3))

    monkeypatch.setattr(solver, "_cuda_odometry", lambda: (factory, "fake_cuda"))
    options = solver.ScanOdometryOptions(voxel_size_m=.1, local_map_scans=3)
    odometry = solver.IncrementalScanOdometry(options, rotation_model=model, backend="cuda")
    cloud = _cloud()
    offsets = np.linspace(-.01, .01, len(cloud))
    odometry.add(cloud, 1., offsets)
    expected_target = solver._local_map(odometry._recent, odometry.poses[-1], options)
    odometry.add(cloud, 1.1, offsets)
    executor = odometry._preparation_executor
    assert executor is not None
    result = odometry.result()
    assert odometry._preparation_executor is None and executor._shutdown
    assert worker_threads and all(thread != caller for thread in worker_threads)
    np.testing.assert_array_equal(target_clouds[0], expected_target)
    assert len(prepared.sources) == 1 + options.deskew_iterations
    for source in prepared.sources:
        np.testing.assert_array_equal(source, solver.preprocess_scan(cloud, options))
    assert result.execution_statistics == {"prepared_targets": 1}

    # A snapshot closes the idle pool; continuing the same odometry creates a new one.
    odometry.add(cloud, 1.2, offsets)
    assert odometry._preparation_executor is not None
    assert odometry._preparation_executor is not executor
    assert odometry.result().execution_statistics == {"prepared_targets": 2}


def test_preparation_exception_reaches_caller_and_result_releases_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(target: np.ndarray, options: solver.ScanOdometryOptions) -> _Target:
        raise ValueError("target preparation failed")

    monkeypatch.setattr(solver, "_cuda_odometry", lambda: (fail, "fake_cuda"))
    odometry = solver.IncrementalScanOdometry(backend="cuda")
    odometry.add(_cloud(), 1.)
    with pytest.raises(ValueError, match="target preparation failed"):
        odometry.add(_cloud(), 1.1)
    executor = odometry._preparation_executor
    assert executor is not None
    odometry.result()
    assert odometry._preparation_executor is None and executor._shutdown


def test_native_cpu_path_creates_no_preparation_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU path created a background executor")

    monkeypatch.setattr(solver, "ThreadPoolExecutor", forbidden)
    odometry = solver.IncrementalScanOdometry()
    odometry.add(_cloud(), 1.)
    odometry.add(_cloud(), 1.1)
    assert odometry._preparation_executor is None
    assert odometry.result().execution_backend == "numpy_scipy"
