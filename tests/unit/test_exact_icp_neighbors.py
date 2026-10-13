from __future__ import annotations

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers import chen_medioni_point_to_plane_icp_solver as plane
from calibrex.solvers import robust_point_to_point_icp_solver as point
from calibrex.solvers._exact_neighbors import _ExactNeighbors


def _original_query(reference, queries, count):
    squared = np.sum((queries[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    count = min(count, len(reference))
    indices = (
        np.argmin(squared, axis=1)[:, None] if count == 1 else
        np.argsort(squared, axis=1, kind="stable")[:, :count]
    )
    return indices, np.take_along_axis(squared, indices, axis=1)


@pytest.mark.parametrize("count", [1, 3, 12, 101])
@pytest.mark.parametrize("seed", [0, 44])
def test_random_neighbors_preserve_indices_and_distance_bits(count, seed):
    rng = np.random.default_rng(seed)
    reference, queries = rng.normal(size=(101, 3)), rng.normal(size=(63, 3))
    actual = _ExactNeighbors(reference).query(queries, count)
    expected = _original_query(reference, queries, count)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("count", [1, 3, 12])
def test_duplicate_and_grid_boundary_ties_use_original_index(count):
    values = np.asarray([(x, y, z) for x in (-1., 0., 1.)
                         for y in (-1., 0., 1.) for z in (-1., 0., 1.)])
    reference = np.concatenate((values[::-1], values, values))
    queries = np.asarray([(0., 0., 0.), (.5, .5, .5), (1., .5, 0.),
                          (np.nextafter(.5, 1.), .5, .5)])
    actual = _ExactNeighbors(reference).query(queries, count)
    for left, right in zip(actual, _original_query(reference, queries, count), strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("exponent", [-170, -155, -150, -50, 0, 50, 145, 151, 200])
@pytest.mark.parametrize("count", [1, 7])
def test_underflow_overflow_and_wide_scales_preserve_numpy(exponent, count):
    rng = np.random.default_rng(59)
    reference = rng.normal(size=(37, 3)) * 10.**exponent
    queries = rng.normal(size=(23, 3)) * 10.**exponent
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        actual = _ExactNeighbors(reference).query(queries, count)
        expected = _original_query(reference, queries, count)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("count", [1, 3])
def test_nonfinite_fallback_preserves_native_nan_order(count):
    reference = np.asarray([(1., 0., 0.), (np.nan, 2., 0.), (np.inf, 0., 0.)])
    queries = np.asarray([(0., 0., 0.), (np.inf, 1., 0.)])
    with np.errstate(invalid="ignore"):
        actual = _ExactNeighbors(reference).query(queries, count)
        expected = _original_query(reference, queries, count)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


def test_reference_is_private_and_empty_queries_are_supported():
    reference = np.asarray([(1., 0., 0.), (2., 0., 0.)])
    search = _ExactNeighbors(reference)
    reference[:] = 42.
    indices, squared = search.nearest(np.asarray([(0., 0., 0.)]))
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_array_equal(squared, [1.])
    assert search.query(np.empty((0, 3)), 2)[0].shape == (0, 2)
    with pytest.raises(ValueError):
        _ExactNeighbors(np.empty((0, 3))).nearest(np.ones((1, 3)))


@pytest.mark.parametrize("mutual", [False, True])
def test_mutual_check_gates_and_stable_trimming_match_original(mutual):
    target = np.asarray([(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.),
                         (-1., 0., 0.), (0., 0., 2.)])
    source = np.asarray([(0., 0., 0.), (0., 0., 0.), (-.5, 0., 0.),
                         (.5, 0., 0.), (0., 0., 1.)])
    options = point.RobustPointToPointIcpOptions(
        correspondence_distance_m=1., trim_fraction=.6, mutual_correspondences=mutual)
    squared = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    ti = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(source)), ti])
    mask = distances <= 1.
    if mutual:
        mask &= np.argmin(squared, axis=0)[ti] == np.arange(len(source))
    selected = np.flatnonzero(mask)
    selected = selected[np.argsort(distances[selected], kind="stable")
                        [:max(1, int(np.ceil(.6 * len(selected))))]]
    expected = (source[selected], target[ti[selected]], distances[selected], selected, ti[selected])
    for left, right in zip(point._correspondences(source, target, options), expected, strict=True):
        np.testing.assert_array_equal(left, right)


class _OriginalNeighbors:
    def __init__(self, reference):
        self.reference = reference

    def query(self, queries, count):
        return _original_query(self.reference, queries, count)

    def nearest(self, queries):
        indices, squared = self.query(queries, 1)
        return indices[:, 0], squared[:, 0]


@pytest.mark.parametrize("solver", ["point", "plane"])
def test_full_solver_payload_matches_exhaustive_with_all_diagnostics(monkeypatch, solver):
    from tests.unit.test_chen_medioni_point_to_plane_icp_solver import _three_plane_clouds
    from tests.unit.test_robust_point_to_point_icp_solver import _clouds

    truth = SE3((.05, -.03, .02), (.002, -.003, .004, .99998))
    options = point.RobustPointToPointIcpOptions(correspondence_distance_m=.5)
    source, target = _clouds(truth, count=160) if solver == "point" else _three_plane_clouds(truth)

    def run():
        if solver == "point":
            return point.RobustPointToPointIcpSolver().solve(
                source, target, options=options).as_dict()
        return plane.ChenMedioniPointToPlaneIcpSolver().solve(
            source, target, common_options=options).as_dict()

    measured = run()
    monkeypatch.setattr(point, "_ExactNeighbors", _OriginalNeighbors)
    monkeypatch.setattr(plane, "_ExactNeighbors", _OriginalNeighbors)
    assert measured == run()
