"""Live CUDA parity, fallback, cache ownership and stream lifetime contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_resident_odometry_adapter import (
    ResidentGeometrySession,
    _DeviceGrid,
    _DeviceGyro,
    _ResidentTarget,
)
from calibrex.solvers.cuda_scan_odometry_adapter import CudaOdometryTarget, _CapturedSource
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, _GyroRotationModel
from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    ScanOdometryOptions,
    _quantize_geometry,
)


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        count = 0
    if not count:
        pytest.skip("requires a CUDA device")


def _options() -> ScanOdometryOptions:
    return ScanOdometryOptions(
        voxel_size_m=0.12,
        min_range_m=0.0,
        max_range_m=100,
        geometry_quantum_m=1e-9,
        motion_translation_quantum_m=1e-6,
        motion_rotation_quantum_rad=1e-7,
        min_correspondences=50,
    )


def _scene() -> np.ndarray:
    rng = np.random.default_rng(42)
    parts = []
    for axis in range(3):
        face = rng.uniform(-4, 4, (500, 3))
        face[:, axis] = 3.0
        parts.append(face)
    return np.concatenate(parts)


@pytest.mark.parametrize("kind", ["ordinary", "overflow", "domain", "nonfinite", "empty", "crop"])
def test_resident_voxels_preserve_native_order_and_fallback(kind: str) -> None:
    from calibrex.solvers.scan_to_scan_odometry import preprocess_scan

    points = _scene()
    if kind == "overflow":
        points = np.repeat([[1.0, 2.0, 3.0]], 33, axis=0)
    elif kind == "domain":
        points = points + 1e6
    elif kind == "nonfinite":
        points[0] = np.nan
    elif kind == "empty":
        points = points[:0]
    elif kind == "crop":
        points = np.array([[1.0, 0, 0], [-1.0, 0, 0], [100.0, 0, 0], [101.0, 0, 0]])
    options = _options()
    with ResidentGeometrySession() as session:
        geometry = session._geometry
        assert geometry is not None
        actual = cp.asnumpy(geometry.preprocess(cp.asarray(points), options))
        np.testing.assert_array_equal(actual, preprocess_scan(points, options))
        assert geometry.voxel_fallbacks == int(kind in ("overflow", "domain", "nonfinite"))


@pytest.mark.parametrize("k", [1, 2, 11, 17])
def test_certified_grid_and_complete_search_match_native_neighbours(k: int) -> None:
    rng = np.random.default_rng(34)
    points = rng.uniform(-0.8, 0.8, (160, 3))
    queries = np.concatenate([points[:20] + 1e-4, rng.uniform(-10, 10, (20, 3))])
    with ResidentGeometrySession():
        grid = _DeviceGrid(cp.asarray(points))
        distances, indexes = grid.query(cp.asarray(queries), k)
        expected_d, expected_i = cKDTree(points).query(queries, k=k)
        np.testing.assert_array_equal(cp.asnumpy(indexes).reshape(-1), expected_i.reshape(-1))
        np.testing.assert_allclose(
            cp.asnumpy(distances).reshape(-1), expected_d.reshape(-1), rtol=3e-15, atol=1e-15
        )
        assert int(grid.arrays[4][0].get()) >= 20
        assert grid._host is None and grid._cpu is None


def test_grid_worklist_grows_past_the_original_chunk_without_dropping_queries() -> None:
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 2.0, 0]])
    queries = np.tile([0.1, 0.1, 0.0], (65537, 1))
    with ResidentGeometrySession():
        grid = _DeviceGrid(cp.asarray(points))
        distances, indexes = grid.query(cp.asarray(queries), 2)
        assert len(grid.arrays[4]) == 65538
        np.testing.assert_array_equal(cp.asnumpy(indexes), np.tile([0, 1], (65537, 1)))
        assert np.isfinite(cp.asnumpy(distances)).all()
        assert grid.query_rows == 65537


def test_resident_target_repairs_ties_and_grid_domain_without_losing_quality() -> None:
    options = replace(_options(), normal_neighbours=4, min_correspondences=3)
    for points, fallback in (
        (np.repeat(np.array([[float(i), 0, 0] for i in range(20)]), 2, axis=0), False),
        (_scene() + 1e6, True),
    ):
        # The internal device entry receives geometry canonicalized by the
        # resident preprocessing/transform kernels, just as the native entry
        # canonicalizes its host source before capturing it.
        points = _quantize_geometry(points, options.geometry_quantum_m)
        with ResidentGeometrySession() as session:
            target = _ResidentTarget(cp.asarray(points), options, session.identity)
            actual = target._register_captured(_CapturedSource(None, cp.asarray(points)), np.eye(4))
            expected = CudaOdometryTarget(points, options).register(points, np.eye(4))
            np.testing.assert_array_equal(actual.transform, expected.transform)
            assert (actual.iterations, actual.correspondences, actual.converged) == (
                expected.iterations,
                expected.correspondences,
                expected.converged,
            )
            assert target.statistics()["gpu_grid_native_fallbacks"] == int(fallback)
            assert target.statistics()["native_tree_captures"] > 0


@pytest.mark.parametrize("local_map", [1, 3])
@pytest.mark.parametrize("deskew", ["none", "rigid", "gyro"])
def test_resident_temporal_passes_match_native_cuda_and_reuse_scans(
    local_map: int,
    deskew: str,
) -> None:
    scene = _scene()
    options = replace(_options(), local_map_scans=local_map)
    times = [0.1, 0.2, 0.4, 0.5]
    scans = []
    for time_s in times:
        rotation = Rotation.from_rotvec(np.array([0.02, -0.01, 0.015]) * time_s).as_matrix()
        scans.append((scene - np.array([0.1, 0.03, -0.02]) * time_s) @ rotation)
    offsets = None if deskew == "none" else np.linspace(0, 0.09, len(scene))
    model = None
    if deskew == "gyro":
        imu_times = np.linspace(0, 1, 201)
        rates = np.stack(
            [0.02 + 0.01 * imu_times, -0.01 * imu_times, 0.015 + 0.03 * imu_times], axis=1
        )
        model = _GyroRotationModel(
            GyroSeries(imu_times, rates),
            Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix(),
            np.array([0.001, -0.002, 0.003]),
            0.003,
        )
    expected = IncrementalScanOdometry(options, rotation_model=model, backend="cuda")
    for time_s, scan in zip(times, scans, strict=True):
        expected.add(scan, time_s, offsets)
    reference = expected.result()
    with ResidentGeometrySession() as session:
        for _ in range(2):
            actual = session.create_odometry(options, rotation_model=model, backend="cuda")
            for time_s, scan in zip(times, scans, strict=True):
                actual.add(scan, time_s, offsets)
            result = actual.result()
            np.testing.assert_array_equal(result.poses, reference.poses)
            for left, right in zip(result.registrations, reference.registrations, strict=True):
                assert (left.converged, left.iterations, left.correspondences) == (
                    right.converged,
                    right.iterations,
                    right.correspondences,
                )
                np.testing.assert_allclose(
                    [left.rmse_m, left.hessian_min_eigenvalue],
                    [right.rmse_m, right.hessian_min_eigenvalue],
                    rtol=1e-8,
                    atol=1e-9,
                )
        assert session.statistics()["source_captures"] == len(scans)
        assert session.statistics()["source_cache_hits"] == len(scans)
        assert session.statistics()["source_validation_rows"] == sum(map(len, scans))
        assert bool(session.statistics()["gyro_rows"]) == (deskew == "gyro")
        assert session.statistics()["gyro_cached_validations"] == (
            2 * (len(scans) - 1) if deskew == "gyro" else 0
        )


def test_session_ownership_and_exception_restore_the_callers_stream() -> None:
    options = _options()
    outer = cp.cuda.Stream(non_blocking=True)
    with outer:
        session = ResidentGeometrySession()
        with session:
            inside = cp.cuda.get_current_stream().ptr
            assert inside != outer.ptr
            odometry = session.create_odometry(options, backend="cuda")
            odometry.add(_scene(), 0.1)
            with pytest.raises(RuntimeError, match="test abort"), ResidentGeometrySession():
                raise RuntimeError("test abort")
            assert cp.cuda.get_current_stream().ptr == inside
        assert cp.cuda.get_current_stream().ptr == outer.ptr
        with pytest.raises(ValueError, match="active session"):
            odometry.add(_scene(), 0.2)
        with pytest.raises(ValueError, match="re-entered"), session:
            pass
        assert session._geometry is not None and not session._geometry.scans


def test_cache_captures_content_and_offsets_without_aliasing_host_mutations() -> None:
    with ResidentGeometrySession() as session:
        geometry = session._geometry
        assert geometry is not None
        points = _scene()
        offsets = np.linspace(0, 0.09, len(points))
        capture = geometry.capture(points, offsets)
        captured, captured_offsets = capture.points, capture.offsets
        cp.cuda.get_current_stream().synchronize()
        saved = points.copy()
        assert geometry.capture(points.copy(), offsets.copy()) is capture
        points[0] += 0.001
        assert geometry.capture(points, offsets).points is not captured
        offsets[0] += 0.001
        assert geometry.capture(saved, offsets).offsets is not captured_offsets
        np.testing.assert_array_equal(cp.asnumpy(captured), saved)
        assert capture.finite_points
        assert capture.offset_bounds == (0.0, 0.09)


def test_arbitrary_rotation_callbacks_keep_the_native_path() -> None:
    def callback(time_s: float, offsets: np.ndarray) -> np.ndarray:
        return np.tile(np.eye(3), (len(offsets), 1, 1))

    with ResidentGeometrySession() as session:
        actual = session.create_odometry(_options(), rotation_model=callback, backend="cuda")
        assert isinstance(actual, IncrementalScanOdometry)
        assert session.statistics()["native_model_fallbacks"] == 1


def _gyro_model() -> _GyroRotationModel:
    times = np.linspace(0, 1, 201)
    rates = np.stack([0.02 + 0.01 * times, -0.01 * times, 0.015 + 0.03 * times], axis=1)
    return _GyroRotationModel(
        GyroSeries(times, rates),
        Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix(),
        np.array([0.001, -0.002, 0.003]),
        0.003,
    )


def test_gyro_uses_capture_validation_without_device_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _scene()
    offsets = np.linspace(-0.02, 0.09, len(points))
    model = _gyro_model()
    expected = model.rotate_points(0.2, offsets, points)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("gyro capture validation launched a device reduction")

    with ResidentGeometrySession() as session:
        geometry = session._geometry
        assert geometry is not None
        capture = geometry.capture(points, offsets)
        gyro = _DeviceGyro(model, geometry)
        with monkeypatch.context() as patch:
            for name in ("min", "max", "isfinite"):
                patch.setattr(cp, name, unexpected)
            actual = gyro.rotate(0.2, capture)
        np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-14, atol=1e-14)
        assert geometry.gyro_cached_validations == 1
        assert geometry.gyro_fallbacks == 0


@pytest.mark.parametrize(
    ("start", "offsets", "first", "last"),
    [
        (0.0, [0.0, 0.1, 0.2], 0.0, 0.2),
        (0.0, [-1e-15, 0.0, 0.1], 0.0, 1.0),
        (1.0, [0.0, np.nextafter(0.0, 1.0), 0.0], 0.0, 1.0),
        (1.0, [0.0, 2 * np.finfo(float).eps, 0.0], 0.0, 1.0),
        (2.0**50, [-0.03125, 0.03125, 0.0625], 2.0**50, 2.0**50 + 0.25),
        (np.finfo(float).max, [0.0, np.finfo(float).max, -1.0], 0.0, np.finfo(float).max),
        (float("nan"), [0.0, 0.1, 0.2], 0.0, 1.0),
        (float("inf"), [0.0, 0.1, 0.2], 0.0, 1.0),
    ],
)
def test_capture_coverage_matches_device_timestamp_rounding(
    start: float,
    offsets: list[float],
    first: float,
    last: float,
) -> None:
    with ResidentGeometrySession() as session:
        geometry = session._geometry
        assert geometry is not None
        capture = geometry.capture(np.ones((len(offsets), 3)), np.array(offsets))
        queries = cp.asnumpy(start + capture.offsets)
        expected = bool(
            np.isfinite(start)
            and first <= start <= last
            and queries.min() >= first
            and queries.max() <= last
        )
        assert capture.covers(start, first, last) == expected


@pytest.mark.parametrize(
    "kind",
    ["before", "after", "spans-before", "spans-after", "nan", "infinity", "empty", "model"],
)
def test_cached_gyro_validation_keeps_native_fallbacks(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _scene()
    offsets = np.linspace(0.0, 0.09, len(points))
    scan_time = 0.2
    model = _gyro_model()
    if kind == "before":
        scan_time = -1.0
    elif kind == "after":
        scan_time = 2.0
    elif kind == "spans-before":
        offsets[0] = -1.0
    elif kind == "spans-after":
        offsets[-1] = 2.0
    elif kind in ("nan", "infinity"):
        points[0, 0] = np.nan if kind == "nan" else np.inf
    elif kind == "empty":
        points, offsets = points[:0], offsets[:0]
    elif kind == "model":
        model = _GyroRotationModel(
            model.gyro,
            model.rotation,
            np.array([2.0, 0.0, 0.0]),
            model.time_offset_s,
        )
    calls = []

    def fallback(
        self: _GyroRotationModel,
        time_s: float,
        times: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        assert self is model and time_s == scan_time
        np.testing.assert_array_equal(times, offsets)
        np.testing.assert_array_equal(values, points)
        calls.append(time_s)
        return np.full(values.shape, 7.0)

    with ResidentGeometrySession() as session:
        geometry = session._geometry
        assert geometry is not None
        capture = geometry.capture(points, offsets)
        monkeypatch.setattr(_GyroRotationModel, "rotate_points", fallback)
        actual = _DeviceGyro(model, geometry).rotate(scan_time, capture)
        np.testing.assert_array_equal(cp.asnumpy(actual), np.full(points.shape, 7.0))
        assert calls == [scan_time]
        assert geometry.gyro_fallbacks == 1


def test_capture_validation_tracks_mutated_content_and_uncached_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrex.solvers import cuda_resident_odometry_adapter as resident

    monkeypatch.setattr(resident, "_CACHE_BYTES", 1)
    points = _scene()
    offsets = np.linspace(0.0, 0.09, len(points))
    with ResidentGeometrySession() as session:
        geometry = session._geometry
        assert geometry is not None
        first = geometry.capture(points, offsets)
        assert first.finite_points and first.offset_bounds == (0.0, 0.09)
        assert not geometry.scans
        points[0, 0] = np.nan
        offsets[-1] = 0.5
        second = geometry.capture(points, offsets)
        assert not second.finite_points and second.offset_bounds == (0.0, 0.5)
        assert first.finite_points and first.offset_bounds == (0.0, 0.09)
        assert geometry.source_validation_rows == 2 * len(points)
