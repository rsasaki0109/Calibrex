"""Exact beam assignment, including rounded ties and unusual center orders."""

from __future__ import annotations

import numpy as np
import pytest

from calibrex.evaluation import camera_lidar_edge as edge


@pytest.mark.parametrize(
    "centers",
    [
        np.linspace(-25.0, 15.0, 64),
        np.array([-1.0, 1.0]),
        np.array([4.0]),
        np.array([1.0, -1.0, 2.0]),
        np.array([-1.0, -1.0, 1.0]),
        np.array([0.0, 1e-14, 2e-14]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ],
)
def test_neighbors_preserve_dense_first_index_ties(centers):
    rng = np.random.default_rng(906)
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    elevation = np.concatenate(
        (
            rng.uniform(-90.0, 90.0, 10_000),
            centers,
            midpoints,
            np.nextafter(midpoints, -np.inf),
            np.nextafter(midpoints, np.inf),
            [-1e20, 1e20],
        )
    )
    # Test both the bounded physical domain and the unusual-input fallback.
    for values in (elevation[:-2], elevation):
        expected = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        np.testing.assert_array_equal(edge._nearest_ring_indices(values, centers), expected)
    assert edge._nearest_ring_indices(np.empty(0), centers).size == 0


def test_complete_depth_discontinuities_preserve_dense_reference(monkeypatch):
    rng = np.random.default_rng(603)
    elevation = np.repeat(np.linspace(-23.0, 7.0, 32), 1000)
    azimuth = np.tile(np.linspace(-np.pi, np.pi, 1000, endpoint=False), 32)
    radius = rng.uniform(5.0, 35.0, elevation.size)
    horizontal = radius * np.cos(np.radians(elevation))
    points = np.column_stack(
        (
            horizontal * np.cos(azimuth),
            horizontal * np.sin(azimuth),
            radius * np.sin(np.radians(elevation)),
        )
    ).astype(np.float32)
    options = {"min_jump_m": 0.3, "exponent": 0.5, "max_points": 2000, "min_ring_fraction": 0.5}
    actual = edge.lidar_depth_edges(points, **options)
    monkeypatch.setattr(
        edge,
        "_nearest_ring_indices",
        lambda values, centers: np.abs(values[:, None] - centers[None, :]).argmin(axis=1),
    )
    expected = edge.lidar_depth_edges(points, **options)
    assert len(actual[0]) == 2000
    for left, right in zip(actual[:2], expected[:2], strict=True):
        np.testing.assert_array_equal(left, right)
    assert actual[2:] == expected[2:]
