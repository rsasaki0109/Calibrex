"""Bitwise tests of the max-frontier identity, including image borders."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from calibrex.evaluation import camera_lidar_edge as edge


def _frontier(pooled, step, iterations):
    spread, front = pooled.copy(), pooled.copy()
    for _ in range(iterations):
        front = step * ndimage.maximum_filter(front, size=3)
        np.maximum(spread, front, out=spread)
    return spread


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (3, 4), (17, 19), (120, 160)])
@pytest.mark.parametrize("iterations", [1, 2, 3, 4, 7, 8, 15, 20, 21, 32, 65])
@pytest.mark.parametrize("gamma", [0.0, 0.1, 0.98, 1.0, 1.01])
def test_spill_keeps_every_float32_bit(shape, iterations, gamma):
    random = np.random.default_rng(20261021)
    image = random.uniform(0, 1, shape).astype(np.float32)
    image.flat[0] = 1.0
    image.flat[-1] = 0.0
    before = image.copy()
    step = np.float32(gamma * gamma)
    actual = edge._edge_spill(image, step, iterations)
    expected = _frontier(image, step, iterations)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("background", [None, 20.0])
@pytest.mark.parametrize("shape,width", [((241, 321), 960), ((481, 643), 240)])
def test_full_edge_image_keeps_original_pixels(monkeypatch, background, shape, width):
    image = np.random.default_rng(23).integers(0, 256, shape, dtype=np.uint8)
    options = {"max_width": width, "gamma": 0.98, "radius": 40, "background_sigma": background}
    actual = edge.image_edge_response(image, **options)
    monkeypatch.setattr(edge, "_edge_spill", _frontier)
    expected = edge.image_edge_response(image, **options)
    assert actual is not None and expected is not None
    assert actual[1] == expected[1]
    np.testing.assert_array_equal(actual[0].view(np.uint32), expected[0].view(np.uint32))


def test_nonfinite_values_retain_the_original_frontier():
    for values, step in [
        (np.array([[1, np.nan], [0, 1]], np.float32), np.float32(0.9604)),
        (np.zeros((2, 2), np.float32), np.float32(np.inf)),
    ]:
        with np.errstate(invalid="ignore"):
            actual = edge._edge_spill(values, step, 20)
            expected = _frontier(values, step, 20)
        np.testing.assert_array_equal(actual, expected)
