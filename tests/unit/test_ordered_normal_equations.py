import math
import random
import struct

import numpy as np
import pytest

from calibrex.graph.lidar_point_to_plane import _ordered_normal_equations
from calibrex.solvers import fixed_trajectory_se3_solver as solver


def _scalar(residuals, rows, options):
    hessian = [[0.0] * 6 for _ in range(6)]
    gradient = [0.0] * 6
    for residual, row in zip(residuals, rows, strict=True):
        weight = solver._robust_weight(residual, options)
        for left in range(6):
            gradient[left] += weight * row[left] * residual
            for right in range(6):
                hessian[left][right] += weight * row[left] * row[right]
    return hessian, gradient


def _bits(result):
    hessian, gradient = result
    return struct.pack("<42d", *(value for row in hessian for value in row), *gradient)


@pytest.mark.parametrize("count", [0, 63, 64, 65, 4096, 4097])
@pytest.mark.parametrize("loss", ["none", "huber"])
def test_weighted_normal_equations_preserve_original_product_and_sum_order(count, loss):
    rng = random.Random(274)
    rows = [[rng.uniform(-1.0e5, 1.0e5) for _ in range(6)] for _ in range(count)]
    residuals = [rng.choice([0.0, -0.0, 1.0, -1.0, 1.0e16, -1.0e16]) for _ in rows]
    options = solver.FixedTrajectorySe3SolverOptions(robust_loss=loss)
    assert _bits(solver._weighted_normal_equations(residuals, rows, options)) == _bits(
        _scalar(residuals, rows, options)
    )


def test_unweighted_normal_equations_and_cancellation_keep_zero_seed_and_chunk_carry():
    rows = np.tile(np.asarray([[1.0, -0.0, 1.0e16, -1.0e16, -1.0, 0.0]]), (4097, 1))
    residuals = np.tile([1.0e16, 1.0, -1.0e16], 1366)[:4097]
    actual = _ordered_normal_equations(rows, residuals)
    expected = _scalar(residuals.tolist(), rows.tolist(),
                       solver.FixedTrajectorySe3SolverOptions(robust_loss="none"))
    assert _bits(actual) == _bits(expected)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_normal_equations_keep_original_path(value):
    rows = [[value, 1.0, 2.0, 3.0, 4.0, 5.0]] * 65
    residuals = [1.0] * 65
    options = solver.FixedTrajectorySe3SolverOptions()
    assert _bits(solver._weighted_normal_equations(residuals, rows, options)) == _bits(
        _scalar(residuals, rows, options)
    )
