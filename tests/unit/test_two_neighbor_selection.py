"""Two-candidate ordering preserves exact NumPy distance/index tie semantics."""

from __future__ import annotations

import numpy as np
import pytest

from calibrex.solvers._exact_neighbors import _ExactNeighbors, _first_of_two


@pytest.mark.parametrize("count", [0, 1, 1024, 125000])
def test_pair_selection_matches_full_sort_including_float_bits(count: int) -> None:
    rng = np.random.default_rng(20261013)
    indices = rng.integers(0, 17, size=(count, 2), dtype=np.int64)
    squared = rng.uniform(0.0, 100.0, size=(count, 2))
    special = np.array(
        [
            [0.0, -0.0],
            [-0.0, 0.0],
            [np.inf, np.inf],
            [np.nan, 2.0],
            [1.0, np.nan],
            [np.nan, np.nan],
            [1.0, 1.0],
            [np.nextafter(1.0, 0.0), 1.0],
            [1.0, np.nextafter(1.0, np.inf)],
        ]
    )
    limit = min(count, len(special))
    squared[:limit] = special[:limit]
    if count > 16:
        squared[16::13, 1] = squared[16::13, 0]
    expected = np.lexsort((indices, squared), axis=1)[:, 0]
    first = _first_of_two(indices, squared)
    actual = np.where(first, 0, 1)
    np.testing.assert_array_equal(actual, expected)
    selected = np.where(first, squared[:, 0], squared[:, 1])
    expected_values = np.take_along_axis(squared, expected[:, None], axis=1)[:, 0]
    np.testing.assert_array_equal(selected.view(np.uint64), expected_values.view(np.uint64))


@pytest.mark.parametrize("count", [1, 2, 4])
def test_spatial_queries_keep_duplicate_ties_distances_and_owned_outputs(count: int) -> None:
    reference = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [np.nextafter(0.5, 0.0), 0.0, 0.0]])
    squared = np.sum((points[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    expected = np.argsort(squared, axis=1, kind="stable")[:, :count]
    search = _ExactNeighbors(reference)
    indices, distances = search.query(points, count)
    assert indices.flags.owndata and distances.flags.owndata
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(
        distances.view(np.uint64), np.take_along_axis(squared, expected, axis=1).view(np.uint64)
    )
    indices[:] = -1
    distances[:] = -1.0
    next_indices, next_distances = search.query(points, count)
    np.testing.assert_array_equal(next_indices, expected)
    np.testing.assert_array_equal(
        next_distances.view(np.uint64),
        np.take_along_axis(squared, expected, axis=1).view(np.uint64),
    )
