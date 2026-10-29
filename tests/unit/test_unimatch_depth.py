import numpy as np
from tools.unimatch_depth import depth_from_prediction


def test_horizontal_stereo_metric_depth_and_missing_support():
    pl = np.array([[100.0, 0, 20, 0], [0, 100, 10, 0], [0, 0, 1, 0]])
    pr = pl.copy()
    pr[0, 3] = -50
    disparity = np.full((24, 48), 5.0)
    disparity[12, 25] = np.nan
    disparity[12, 26] = 0
    result = depth_from_prediction(disparity, pl, pr)
    np.testing.assert_allclose(result[10, 15:30], 10.0, atol=1e-5)
    assert np.isnan(result[12, 25:27]).all()
    assert np.isnan(result[:, :5]).all()
