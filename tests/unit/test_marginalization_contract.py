"""Public elimination APIs validate dimensions before numerical operations."""
import numpy as np
import pytest

from calibrex.core.continuous_time_marginalization import (
    marginalize_knot_prefix,
    schur_complement_normal,
    schur_complement_normal_with_evidence,
)
from calibrex.core.square_root_marginalization import marginalize_residual_prefix


@pytest.mark.parametrize('function', [schur_complement_normal,
                                     schur_complement_normal_with_evidence,
                                     marginalize_residual_prefix])
@pytest.mark.parametrize('dimension', [True, False, np.bool_(True), 0., 1., .5,
                                       float('nan'), float('inf'), -1])
def test_invalid_dimension_types_fail_explicitly(function, dimension):
    with pytest.raises(ValueError, match=r'dimension.*integer'):
        function(np.eye(2), np.zeros(2), dimension)


@pytest.mark.parametrize('count', [True, np.bool_(False), 0., .5, float('nan'),
                                  float('inf'), -1])
def test_knot_counts_are_validated_before_multiplication(count):
    with pytest.raises(ValueError, match=r'knot count.*integer'):
        marginalize_knot_prefix(np.eye(12), np.zeros(12), eliminated_knot_count=count)


@pytest.mark.parametrize('dimension', [2, 10**500, np.uint64(2**64-1)])
def test_oversized_dimensions_fail_before_overflow_or_slicing(dimension):
    with pytest.raises(ValueError, match='dimension'):
        schur_complement_normal(np.eye(2), np.zeros(2), dimension)
    with pytest.raises(ValueError, match='dimension'):
        marginalize_residual_prefix(np.eye(2), np.zeros(2), dimension)
    with pytest.raises(ValueError, match='dimension'):
        marginalize_knot_prefix(np.eye(12), np.zeros(12), eliminated_knot_count=dimension)


@pytest.mark.parametrize('dimension', [0, 1])
def test_asymmetric_information_is_rejected_even_without_elimination(dimension):
    with pytest.raises(ValueError, match='symmetric'):
        schur_complement_normal(np.array([[1., 2.], [0., 1.]]), np.zeros(2), dimension)


@pytest.mark.parametrize('size', [0, 3])
def test_zero_elimination_preserves_valid_information_without_aliasing(size):
    h, g = np.eye(size), np.arange(size, dtype=float)
    retained, gradient, evidence = schur_complement_normal_with_evidence(h, g, np.int64(0))
    np.testing.assert_array_equal(retained, h)
    np.testing.assert_array_equal(gradient, g)
    assert not np.shares_memory(retained, h)
    assert not np.shares_memory(gradient, g)
    assert evidence.eliminated_dimension == 0


def test_numpy_integer_dimensions_remain_supported():
    h, g = schur_complement_normal(np.eye(3), np.ones(3), np.int64(1))
    np.testing.assert_array_equal(h, np.eye(2))
    np.testing.assert_array_equal(g, np.ones(2))
    h, g = marginalize_knot_prefix(np.eye(12), np.ones(12), eliminated_knot_count=np.int64(1))
    np.testing.assert_array_equal(h, np.eye(6))
    np.testing.assert_array_equal(g, np.ones(6))
    result = marginalize_residual_prefix(np.eye(3), np.ones(3), np.int64(1))
    assert result.numerical_rank == 1
