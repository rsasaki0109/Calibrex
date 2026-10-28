"""Bound prior storage without dropping linear or constant least-squares terms."""
import numpy as np
import pytest

from calibrex.core.square_root_marginalization import compress_residual_factor


@pytest.mark.parametrize('dependent', [False, True])
def test_compression_preserves_complete_cost_and_bounds_rows(dependent):
    rng = np.random.default_rng(403)
    j = rng.normal(size=(80, 4))
    if dependent:
        j[:, 3] = j[:, 0] + j[:, 1]
    r = rng.normal(size=80)
    compressed, offset = compress_residual_factor(j, r)
    assert compressed.shape == (5, 4)
    assert offset.shape == (5,)
    for x in [np.zeros(4), *rng.normal(size=(8, 4))]:
        assert np.linalg.norm(compressed @ x + offset)**2 == pytest.approx(
            np.linalg.norm(j @ x + r)**2, rel=1e-13,
        )
    np.testing.assert_allclose(compressed.T @ offset, j.T @ r, atol=1e-13)


def test_weak_independent_information_and_irreducible_constant_are_preserved():
    j = np.zeros((30, 2))
    j[0, 0], j[1, 1] = 1., 1e-150
    r = np.zeros(30)
    r[1], r[2] = 1., 3.
    compressed, offset = compress_residual_factor(j, r)
    assert (compressed.T @ compressed)[1, 1] == pytest.approx(1e-300, rel=1e-13, abs=0.)
    assert (compressed.T @ offset)[1] == pytest.approx(1e-150, rel=1e-13, abs=0.)
    # The weak coordinate can explain row 1, but never the independent constant.
    assert np.linalg.norm(compressed @ np.array([0., -1e150]) + offset)**2 == pytest.approx(9.)


def test_repeated_compression_is_bounded_and_matches_all_original_rows():
    rng = np.random.default_rng(29)
    compressed, offset = np.zeros((0, 3)), np.zeros(0)
    originals, residuals = [], []
    for _ in range(20):
        j, r = rng.normal(size=(10, 3)), rng.normal(size=10)
        originals.append(j)
        residuals.append(r)
        compressed, offset = compress_residual_factor(
            np.vstack((compressed, j)), np.concatenate((offset, r)),
        )
        assert len(offset) <= 4
    j, r = np.vstack(originals), np.concatenate(residuals)
    for x in rng.normal(size=(5, 3)):
        assert np.linalg.norm(compressed @ x + offset)**2 == pytest.approx(
            np.linalg.norm(j @ x + r)**2, rel=1e-13,
        )


@pytest.mark.parametrize('rows, columns', [(0, 2), (2, 2), (5, 0)])
def test_empty_short_and_constant_only_factors(rows, columns):
    j, r = np.zeros((rows, columns)), np.ones(rows)
    compressed, offset = compress_residual_factor(j, r)
    assert len(offset) <= columns + 1
    assert np.linalg.norm(offset)**2 == pytest.approx(float(rows))
    assert not np.shares_memory(compressed, j)
    assert not np.shares_memory(offset, r)


@pytest.mark.parametrize('j,r', [
    (np.ones(2), np.ones(2)), (np.eye(2), np.zeros(3)),
    (np.eye(2, dtype=complex), np.zeros(2)),
    (np.eye(2), np.zeros(2, dtype=complex)),
    (np.full((5, 1), np.inf), np.zeros(5)),
    (np.ones((5, 1)), np.full(5, np.nan)),
])
def test_invalid_factors_are_rejected(j, r):
    with pytest.raises(ValueError):
        compress_residual_factor(j, r)
