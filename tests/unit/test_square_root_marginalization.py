"""Residual-space elimination preserves information lost in normal subtraction."""

import numpy as np
import pytest

from calibrex.core.continuous_time_marginalization import schur_complement_normal
from calibrex.core.square_root_marginalization import marginalize_residual_prefix


def test_weak_retained_information_survives_normal_equation_cancellation():
    j = np.array([[1., 1.], [0., 1e-9]])
    r = np.array([0., 1.])
    h, g = schur_complement_normal(j.T @ j, j.T @ r, 1)
    assert h[0, 0] == 0.  # The input normal matrix has already lost the weak term.
    assert g[0] == 1e-9
    result = marginalize_residual_prefix(j, r, 1)
    np.testing.assert_allclose(result.jacobian.T @ result.jacobian, [[1e-18]], atol=0.)
    np.testing.assert_allclose(result.jacobian.T @ result.residual, [1e-9], atol=0.)
    optimum = np.linalg.lstsq(result.jacobian, -result.residual, rcond=None)[0]
    np.testing.assert_allclose(optimum, [-1e9])
    assert result.numerical_rank == 1


@pytest.mark.parametrize("coupled", [False, True])
def test_complete_profiled_cost_including_constant_matches_lstsq(coupled):
    rng = np.random.default_rng(872)
    e = rng.normal(size=(12, 3))
    if coupled:
        e[:, 2] = e[:, 0] + 2 * e[:, 1]
    retained = rng.normal(size=(12, 2))
    r = rng.normal(size=12)
    result = marginalize_residual_prefix(np.column_stack((e, retained)), r, 3)
    assert result.numerical_rank == (2 if coupled else 3)
    for x in rng.normal(size=(5, 2)):
        shifted = retained @ x + r
        old = np.linalg.lstsq(e, -shifted, rcond=None)[0]
        expected = np.linalg.norm(e @ old + shifted) ** 2
        actual = np.linalg.norm(result.jacobian @ x + result.residual) ** 2
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_extreme_independent_column_units_preserve_rank_and_retained_quadratic():
    j = np.array([[1., 0., .2], [0., 1., .3], [0., 0., .4]])
    r = np.array([1., 2., 3.])
    reference = marginalize_residual_prefix(j, r, 2)
    j[:, :2] *= np.array([1e-320, 1e300])
    result = marginalize_residual_prefix(j, r, 2)
    assert result.numerical_rank == 2
    assert result.zero_information_dimension == 0
    np.testing.assert_allclose(result.jacobian, reference.jacobian)
    np.testing.assert_allclose(result.residual, reference.residual)


@pytest.mark.parametrize("rows, eliminated", [(3, 0), (3, 2), (0, 2)])
def test_absent_eliminated_columns_preserve_factors_and_copy_inputs(rows, eliminated):
    j = np.zeros((rows, 3))
    j[:, -1] = 2.
    r = np.ones(rows)
    result = marginalize_residual_prefix(j, r, eliminated)
    np.testing.assert_array_equal(result.jacobian, j[:, eliminated:])
    np.testing.assert_array_equal(result.residual, r)
    assert result.numerical_rank == 0
    assert result.zero_information_dimension == eliminated
    assert not np.shares_memory(result.jacobian, j)
    assert not np.shares_memory(result.residual, r)


def test_elimination_spanning_all_rows_leaves_zero_cost():
    result = marginalize_residual_prefix(np.array([[1., 0., 1.], [0., 1., 2.]]),
                                        np.ones(2), 2)
    assert result.jacobian.shape == (0, 1)
    assert result.residual.shape == (0,)
    assert result.numerical_rank == 2


@pytest.mark.parametrize("j,r,dim", [
    (np.eye(2).astype(complex), np.ones(2), 1),
    (np.eye(2), np.ones(2).astype(complex), 1),
    (np.array([[np.inf, 0.]]), np.ones(1), 1),
    (np.eye(2), np.array([np.nan, 0.]), 1),
    (np.ones(2), np.ones(2), 1),
    (np.eye(2), np.ones(3), 1),
    (np.eye(2), np.ones(2), -1),
    (np.eye(2), np.ones(2), 2),
    (np.eye(2), np.ones(2), 0.5),
])
def test_invalid_input_is_rejected(j, r, dim):
    with pytest.raises(ValueError):
        marginalize_residual_prefix(j, r, dim)


@pytest.mark.parametrize('rows, eliminated', [(4, 6), (30, 3), (300, 6)])
@pytest.mark.parametrize('coupled', [False, True])
def test_implicit_projection_matches_full_basis_quadratic(rows, eliminated, coupled):
    rng = np.random.default_rng(591)
    j, r = rng.normal(size=(rows, eliminated + 2)), rng.normal(size=rows)
    if coupled:
        j[:, eliminated - 1] = j[:, 0] + j[:, 1]
    e = j[:, :eliminated].copy()
    e /= np.max(np.abs(e), axis=0)
    e /= np.linalg.norm(e, axis=0)
    basis, values, _ = np.linalg.svd(e, full_matrices=True)
    threshold = 60 * np.finfo(float).eps * max(rows, eliminated) * values[0]
    rank = np.count_nonzero(values > threshold)
    reference_j, reference_r = basis[:, rank:].T @ j[:, eliminated:], basis[:, rank:].T @ r
    result = marginalize_residual_prefix(j, r, eliminated)
    assert result.numerical_rank == rank
    assert result.residual.shape == (rows - rank,)
    for x in rng.normal(size=(5, 2)):
        assert np.linalg.norm(result.jacobian @ x + result.residual)**2 == pytest.approx(
            np.linalg.norm(reference_j @ x + reference_r)**2, rel=1e-12, abs=1e-12,
        )
