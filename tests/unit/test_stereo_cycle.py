"""Round-trip correspondence must sample the right domain with the correct sign."""

import numpy as np
from tools.diagnose_stereo_cycle import round_trip_error, stats


def test_round_trip_uses_right_coordinate_and_preserves_missing():
    left = np.full((1, 8), -1.0)
    right = np.full((1, 8), -1.0)
    left[0, 4] = 2.0
    right[0, 2] = 2.25
    left[0, 5] = 2.0  # Reverse pixel 3 is invalid.
    left[0, 0] = 2.0  # Outside the right image, never clamped into validity.
    error = round_trip_error(left, right)
    assert error[0, 4] == 0.25
    assert np.isfinite(error).sum() == 1
    assert np.isnan(error[0, 5]) and np.isnan(error[0, 0])


def test_empty_population_has_no_invented_zero_error():
    s = stats(np.array([]), np.array([]))
    assert s["count"] == 0
    assert s["median_z_error_m"] is None
    assert s["median_disparity_error_px"] is None
