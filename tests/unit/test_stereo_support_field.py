"""Missing/occluding boundaries must not cause finite objective jumps."""

import numpy as np
import pytest

from calibrex.solvers.stereo_support_field import build_support_field, support_residuals
from calibrex.solvers.stereo_translation_refiner import StereoSurfaceObservation


def make_observation(kind):
    depth = np.full((60, 80), 10.0)
    if kind == "hole":
        depth[20:40, 30:50] = np.nan
    else:
        depth[:, 30:] = 20.0
    return StereoSurfaceObservation("test", depth, np.zeros((1, 3)), (100, 100, 40, 30), 0.5)


def cost(u, obs, field):
    camera = np.array([[(u - 40) * 0.1, 0, 10]])
    a, b = support_residuals(camera, obs, field, 0.5, 16.0)
    return float(np.sum(np.sqrt(1 + a * a) - 1) + np.sum(np.sqrt(1 + b * b) - 1))


@pytest.mark.parametrize("kind", ["hole", "step"])
def test_boundary_cost_is_continuous_and_missing_support_is_not_free(kind):
    obs = make_observation(kind)
    field = build_support_field(obs, 2.0)
    for u in np.arange(24, 35):
        assert abs(cost(u - 1e-7, obs, field) - cost(u + 1e-7, obs, field)) < 1e-4
    assert cost(30, obs, field) > 15
    assert cost(24, obs, field) == pytest.approx(0, abs=1e-10)
    assert field.confidence[30, 30] == 0


def test_empty_surface_has_zero_observation_weight():
    obs = make_observation("hole")
    obs.depth_m[:] = np.nan
    field = build_support_field(obs, 2.0)
    a, b = support_residuals(np.array([[0, 0, 10.0]]), obs, field, 0.5, 16.0)
    np.testing.assert_array_equal(a, [0.0])
    np.testing.assert_allclose(b, [16.0])
