"""The fixed mask removes unsupported depth without changing retained measurements."""

import numpy as np
from tools.compare_stereo_cycle_filter import mask_depth


def test_cycle_boundary_and_missing_reverse_do_not_pass():
    depth = np.array([[2.0, 3.0, 4.0, 5.0, np.nan]])
    original = depth.copy()
    result = mask_depth(depth, np.array([[0.0, 1.0, 1.0001, np.nan, 0.0]]))
    np.testing.assert_array_equal(result[:, :2], depth[:, :2])
    assert np.isnan(result[:, 2:]).all()
    np.testing.assert_array_equal(depth, original)
