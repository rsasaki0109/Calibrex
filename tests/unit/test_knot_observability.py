"""Local rank diagnostics separate missing columns from coupled null modes."""
from __future__ import annotations

import numpy as np
import pytest

from calibrex.core.knot_observability import analyze_knot_information


def test_zero_columns_are_named_with_global_knot_indices():
    h = np.eye(12)
    h[6:, 6:] = 0.
    result = analyze_knot_information(h, (4, 5))
    assert result.numerical_rank == 6
    assert result.numerical_nullity == 6
    assert result.zero_information_parameters == [
        f"knot[5].{axis}" for axis in ("tx", "ty", "tz", "rx", "ry", "rz")
    ]
    assert sum(result.nullspace_participation.values()) == pytest.approx(6.)


def test_relative_measurements_have_coupled_nullspace_without_zero_columns():
    jacobian = np.column_stack((np.eye(6), -np.eye(6)))
    result = analyze_knot_information(jacobian.T @ jacobian, (0, 1))
    assert result.numerical_rank == 6
    assert result.zero_information_parameters == []
    np.testing.assert_allclose(list(result.nullspace_participation.values()), .5)
    # An absolute prior removes the gauge; local measurements alone still lack it.
    prior = np.diag([1.] * 6 + [0.] * 6)
    assert analyze_knot_information(jacobian.T @ jacobian + prior, (0, 1)).numerical_rank == 12


def test_rank_and_nullspace_are_invariant_to_independent_coordinate_units():
    rng = np.random.default_rng(19)
    jacobian = rng.normal(size=(8, 12))
    h = jacobian.T @ jacobian
    units = np.logspace(-12, 12, 12)
    reference = analyze_knot_information(h, (0, 1))
    rescaled = analyze_knot_information(h * units[:, None] * units[None, :], (0, 1))
    assert reference.numerical_rank == rescaled.numerical_rank == 8
    np.testing.assert_allclose(
        rescaled.scaled_eigenvalues, reference.scaled_eigenvalues, atol=1e-14,
    )
    np.testing.assert_allclose(list(rescaled.nullspace_participation.values()),
                               list(reference.nullspace_participation.values()), atol=1e-13)


def test_weak_but_independent_information_is_not_lost_to_units():
    result = analyze_knot_information(np.diag(np.logspace(-28, 28, 6)), (0,))
    assert result.numerical_rank == 6
    assert result.numerical_nullity == 0


@pytest.mark.parametrize("matrix", [np.eye(5), np.full((6, 6), np.nan), -np.eye(6),
                                    np.eye(6) + 2 * np.eye(6, k=1)])
def test_invalid_information_is_rejected(matrix):
    with pytest.raises(ValueError):
        analyze_knot_information(matrix, (0,))
