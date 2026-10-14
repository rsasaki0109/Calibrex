"""Actual-device voxel parity, source summation order, boundaries and fallback."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_scan_odometry_adapter import CudaScanPreprocessor
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions, preprocess_scan


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


@pytest.mark.parametrize("quantum", [0., 1e-9, .125])
@pytest.mark.parametrize("voxel", [.03, .3, 5.])
def test_gpu_voxels_match_native_bits_and_lexicographic_order(quantum: float, voxel: float) -> None:
    rng = np.random.default_rng(216)
    points = rng.normal(size=(4097, 7)) * 3
    points[::11, :3] = points[::11, :3].round(1)
    points[::17, :3] = 0
    options = ScanOdometryOptions(voxel_size_m=voxel, geometry_quantum_m=quantum,
                                  min_range_m=0, max_range_m=6)
    prepared = CudaScanPreprocessor()
    np.testing.assert_array_equal(prepared(points, options), preprocess_scan(points, options))
    assert prepared.statistics()["gpu_preprocess_input_rows"] == len(points)
    assert prepared.statistics()["gpu_preprocess_kernel_calls"] == 2
    assert prepared.statistics()["native_preprocess_calls"] == 0


def test_gpu_voxels_preserve_input_addition_order_for_unequal_magnitudes() -> None:
    points = np.array([[1e-8, 1e-8, 0.]] * 200 + [[1e10, 1., 0.]]
                      + [[1e-8, 1e-8, 0.]] * 200)
    options = ScanOdometryOptions(voxel_size_m=1e11, min_range_m=0, max_range_m=np.inf)
    actual = CudaScanPreprocessor()(points, options)
    expected = preprocess_scan(points, options)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("bound", [np.nextafter(.5, 0), .5, np.nextafter(.5, 1)])
def test_gpu_crop_and_voxel_boundaries_keep_native_membership(bound: float) -> None:
    points = np.array([[bound, 0., 0.], [-bound, 0., 0.], [0., bound, 0.], [0., 0., bound]])
    for options in (ScanOdometryOptions(min_range_m=.5, max_range_m=.5, voxel_size_m=.5),
                    ScanOdometryOptions(min_range_m=0, max_range_m=1, voxel_size_m=.5)):
        np.testing.assert_array_equal(CudaScanPreprocessor()(points, options),
                                      preprocess_scan(points, options))


def test_unsupported_input_and_empty_scans_use_native_fallback() -> None:
    prepared = CudaScanPreprocessor()
    options = ScanOdometryOptions(min_range_m=0)
    points = np.array([[np.nan, 0., 0.], [1., 2., 3.], [np.inf, 0., 0.]])
    with np.errstate(invalid="ignore"):
        np.testing.assert_array_equal(prepared(points, options), preprocess_scan(points, options))
    np.testing.assert_array_equal(prepared(np.empty((0, 3)), options), np.empty((0, 3)))
    wide = np.array([[1e20, 0., 0.], [1e20, 0., 0.]])
    with np.errstate(invalid="ignore"):
        settings = replace(options, max_range_m=np.inf)
        np.testing.assert_array_equal(prepared(wide, settings), preprocess_scan(wide, settings))
    assert prepared.statistics()["native_preprocess_calls"] == 3
    assert prepared.statistics()["gpu_preprocess_calls"] == 0
