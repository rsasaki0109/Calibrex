"""Builtin sum float bytes across shared scalar-field and probe evaluation."""

import sys

import numpy as np
import pytest

from calibrex.graph import _ordered_sums as sums
from calibrex.graph import _rgbd_batches as batches
from calibrex.graph import _rgbd_field_columns as fields
from calibrex.solvers import _tum_geometry as geometry


@pytest.mark.parametrize("rows,columns", [(256, 1), (256, 8), (257, 64), (1000, 8), (65536, 8)])
@pytest.mark.parametrize("seed", [0, 41, 795])
def test_native_row_sums_keep_every_builtin_float_bit(rows, columns, seed):
    rng = np.random.default_rng(seed)
    products = rng.normal(size=(rows, columns)) * np.exp2(
        rng.integers(-100, 100, (rows, columns))
    )
    products[0] = -0.0
    expected = np.asarray([sum(row) for row in products.tolist()], dtype=np.float64)
    actual = sums._ordered_float_sums(products)
    if sys.implementation.name != "cpython" or sys.version_info[:2] not in (
        (3, 10), (3, 11), (3, 12),
    ):
        assert actual is None
    else:
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("version", [(3, 10), (3, 11)])
def test_legacy_mode_preserves_left_to_right_rounding(version):
    products = np.tile((1e16, 1.0, -1e16, -0.0, 0.0, 1e-200, -1e-200, -0.0), (512, 1))
    expected = []
    for row in products.tolist():
        total = 0.0
        for value in row:
            total += value
        expected.append(total)
    actual = sums._ordered_float_sums(products, version=version)
    assert actual.tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("rows", [0, 1, 63, 255, 256, 511, 512, 1000])
def test_both_callers_keep_all_products_and_ordered_scalar_fallback(rows):
    weights = np.random.default_rng(41).uniform(0.0, 1.0, (rows, 8))
    offsets = (0.01, -0.03, 0.0, -0.0, 1e-20, 0.07, -0.02, 0.05)
    products = weights * np.asarray(offsets)
    expected = np.asarray([sum(row) for row in products.tolist()], dtype=np.float64)
    for caller in (batches._weighted_sums, geometry._weighted_sums):
        assert caller(weights, offsets).tobytes() == expected.tobytes()


def test_probe_and_scalar_fields_share_the_same_compensation_and_tie_bits():
    weights = np.random.default_rng(401).normal(size=(129, 8))
    offsets = np.random.default_rng(712).normal(size=(9, 8))
    products = weights[None, :, :] * offsets[:, None, :]
    expected = np.asarray([sum(row) for row in products.reshape((-1, 8)).tolist()])
    assert fields._weighted_probes(weights, offsets).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "change", [
        "small", "large", "wide", "empty", "dtype",
        "subclass", "nan", "inf", "magnitude", "version",
    ],
)
def test_unusual_products_keep_the_original_scalar_path(change):
    products = np.ones((256, 8))
    version = (3, 12)
    if change == "small":
        products = products[:255]
    elif change == "large":
        products = np.ones((65537, 1))
    elif change == "wide":
        products = np.ones((256, 65))
    elif change == "empty":
        products = np.ones((256, 0))
    elif change == "dtype":
        products = products.astype(np.float32)
    elif change == "subclass":
        products = products.view(type("Array", (np.ndarray,), {}))
    elif change in ("nan", "inf", "magnitude"):
        products[0, 0] = {"nan": float("nan"), "inf": float("inf"), "magnitude": 1e101}[change]
    else:
        version = (3, 99)
    assert sums._ordered_float_sums(products, version=version) is None
