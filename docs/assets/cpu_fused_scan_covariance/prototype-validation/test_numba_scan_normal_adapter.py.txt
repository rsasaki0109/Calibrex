"""Fused covariance retains native arithmetic, lazy cache semantics and provenance."""

from __future__ import annotations

import builtins
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    ScanOdometryOptions,
    TargetNormals,
    _normals_from_neighbours,
    estimate_normals,
    register_point_to_plane,
    run_scan_to_scan_odometry,
)


def test_native_normals_do_not_import_numba_and_explicit_missing_dependency_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name == "numba" or "numba_scan_normal_adapter" in name:
            raise ImportError("optional compiler unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    points = np.random.default_rng(31).normal(size=(30, 3))
    estimate_normals(points, cKDTree(points), ScanOdometryOptions())
    assert run_scan_to_scan_odometry([]).execution_backend == "numpy_scipy"
    with pytest.raises(ValueError, match=r"calibrex\[numba\]"):
        IncrementalScanOdometry(ScanOdometryOptions(normal_backend="numba"))


def test_invalid_normal_backend_and_gpu_mixture_are_rejected() -> None:
    with pytest.raises(ValueError, match="normal backend"):
        ScanOdometryOptions(normal_backend="invalid")  # type: ignore[arg-type]
    options = ScanOdometryOptions(normal_backend="numba")
    with pytest.raises(ValueError, match="CPU option"):
        IncrementalScanOdometry(options, backend="cuda")
    with pytest.raises(ValueError, match="CPU option"):
        register_point_to_plane(np.empty((0, 3)), np.empty((0, 3)), np.eye(4), options,
                                backend="cuda")


@pytest.mark.parametrize("k", [2, 3, 4, 10, 31])
@pytest.mark.parametrize("scale", [0., 1e-100, 1., 1e100])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_covariance_and_pca_are_exact_for_scaled_degenerate_and_strided_inputs(
    k: int, scale: float, layout: str,
) -> None:
    pytest.importorskip("numba")
    from calibrex.solvers.numba_scan_normal_adapter import NumbaNormalPreparer, _covariance

    rng = np.random.default_rng(192)
    points = rng.normal(size=(80, 3)) * scale
    if layout == "F":
        points = np.asfortranarray(points)
    elif layout == "strided":
        points = np.repeat(points, 2, axis=1)[:, ::2]
    indices = rng.integers(0, len(points), size=(127, k), dtype=np.intp)
    neighbours = points[indices]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    expected_covariance = np.einsum("nki,nkj->nij", centered, centered) / k
    np.testing.assert_array_equal(_covariance(points, indices), expected_covariance)
    options = ScanOdometryOptions(normal_neighbours=k, normal_backend="numba")
    preparer = NumbaNormalPreparer()
    actual = preparer.prepare(points, indices, k, options)
    expected = _normals_from_neighbours(points, indices, k, options)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert preparer.statistics() == {
        "numba_normal_covariance_calls": 1, "numba_normal_covariance_rows": len(indices),
        "native_normal_covariance_fallback_rows": 0,
    }


def _scene() -> np.ndarray:
    rng = np.random.default_rng(412)
    parts = []
    for axis in range(3):
        face = rng.uniform(-5., 5., (700, 3))
        face[:, axis] = 4.
        parts.append(face)
    points = np.vstack(parts)
    return np.vstack((points, points[:50]))


def test_lazy_normals_use_native_tie_order_and_compute_each_row_once() -> None:
    pytest.importorskip("numba")
    points = _scene()
    tree = cKDTree(points)
    options = ScanOdometryOptions(normal_backend="numba", query_workers=2)
    expected = estimate_normals(points, tree, replace(options, normal_backend="numpy"))
    holder = TargetNormals(points, tree, options)
    holder.ensure(np.arange(100, dtype=np.intp))
    holder.ensure(np.arange(80, 130, dtype=np.intp))
    holder.ensure(np.arange(130, dtype=np.intp))
    for a, b in zip((holder.normals[:130], holder.planar[:130]), expected, strict=True):
        np.testing.assert_array_equal(a, b[:130])
    assert holder.statistics()["numba_normal_covariance_rows"] == 130
    assert holder.statistics()["numba_normal_covariance_calls"] == 2
    assert "fastmath=false;parallel=false;eigh=numpy;source_sha256=" in holder.execution_backend


def test_float32_keeps_native_rounding_and_reports_fallback() -> None:
    pytest.importorskip("numba")
    from calibrex.solvers.numba_scan_normal_adapter import NumbaNormalPreparer

    points = _scene().astype(np.float32)
    _, indices = cKDTree(points).query(points[:50], k=10)
    options = ScanOdometryOptions(normal_backend="numba")
    holder = NumbaNormalPreparer()
    actual = holder.prepare(points, indices, 10, options)
    expected = _normals_from_neighbours(points, indices, 10, options)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert holder.statistics() == {
        "numba_normal_covariance_calls": 0, "numba_normal_covariance_rows": 0,
        "native_normal_covariance_fallback_rows": 50,
    }


def test_complete_odometry_keeps_native_results_without_numerical_grids() -> None:
    pytest.importorskip("numba")
    target = _scene()
    scans = []
    for index in range(4):
        rotation = Rotation.from_rotvec(np.array([.002, -.003, .004]) * index).as_matrix()
        scans.append((target - np.array([.03, -.02, .01]) * index) @ rotation)
    options = ScanOdometryOptions(voxel_size_m=.08, local_map_scans=3, query_workers=2)
    expected = run_scan_to_scan_odometry(scans, options, times_s=[0., .1, .2, .4])
    actual = run_scan_to_scan_odometry(scans, replace(options, normal_backend="numba"),
                                       times_s=[0., .1, .2, .4])
    for a, b in zip(actual.poses, expected.poses, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(actual.registrations, expected.registrations, strict=True):
        np.testing.assert_array_equal(a.transform, b.transform)
        assert (
            a.converged, a.iterations, a.correspondences, a.rmse_m, a.hessian_min_eigenvalue,
        ) == (
            b.converged, b.iterations, b.correspondences, b.rmse_m, b.hessian_min_eigenvalue,
        )
        assert a.execution_backend == actual.execution_backend
    assert actual.execution_statistics["numba_normal_covariance_rows"] > 0
    assert actual.execution_statistics["native_normal_covariance_fallback_rows"] == 0
    assert expected.execution_statistics == {}


def test_precomputed_normals_report_native_execution() -> None:
    points = _scene()
    options = ScanOdometryOptions(normal_backend="numba")
    normals = estimate_normals(points, cKDTree(points), replace(options, normal_backend="numpy"))
    result = register_point_to_plane(points, points, np.eye(4), options, target_normals=normals)
    assert result.execution_backend == "numpy_scipy"


def test_joint_cli_forwards_compiled_normals_and_records_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object,
) -> None:
    from pathlib import Path

    from calibrex.cli.main import main

    seen: dict[str, object] = {}

    def fake_run(_bag: object, _profile: object, options: object, **kwargs: object) -> None:
        seen.update(kwargs)
        seen["options"] = options
        raise ValueError("intentional test stop")

    monkeypatch.setattr("calibrex.check.estimators.detect_point_time", lambda *args: None)
    monkeypatch.setattr("calibrex.evaluation.lidar_imu_bag_calibration.calibrate_lidar_imu_bag",
                        fake_run)
    assert main([
        "imu-lidar", "joint", "public.bag", "--lidar-topic", "/lidar", "--imu-topic", "/imu",
        "--dataset-family", "public", "--dataset-license", "test-only",
        "--odometry-normal-backend", "numba", "--odometry-query-workers", "4",
        "--output", str(Path(str(tmp_path)) / "result.json"),
    ]) == 2
    options = seen["options"].rotation.windowing.odometry
    assert options.normal_backend == "numba" and options.query_workers == 4
    command = seen["command"]
    assert command[command.index("--odometry-normal-backend") + 1] == "numba"
