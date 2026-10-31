import numpy as np
import pytest

pytest.importorskip("cv2")

from tools.diagnose_stereo_ambiguity import patch_costs, stats


def test_known_shift_has_zero_cost_at_frozen_disparity():
    left = np.random.default_rng(5).integers(0, 255, (40, 100), dtype=np.uint8)
    right = np.roll(left, -12, axis=1)
    cost, texture, valid = patch_costs(
        left, right, np.array([40.0]), np.array([20.0]), np.array([12.0])
    )
    assert valid.all() and cost[0, 4] == 0 and texture[0] > 5
    assert np.min(cost[0, [0, 1, 2, 6, 7, 8]]) > 0


def test_flat_patch_is_ambiguous_and_empty_population_explicit():
    im = np.full((40, 100), 128, dtype=np.uint8)
    cost, texture, valid = patch_costs(im, im, np.array([40.0]), np.array([20.0]), np.array([12.0]))
    assert stats(cost, texture, valid)["ambiguous_fraction"] == 1
    assert stats(cost, texture, ~valid)["count"] == 0
