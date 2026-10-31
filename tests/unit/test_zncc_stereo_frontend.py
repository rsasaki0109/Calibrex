import numpy as np
import pytest

pytest.importorskip("cv2")

from tools.zncc_stereo_frontend import disparity_zncc


def test_known_shift_survives_gain_and_offset():
    rng = np.random.default_rng(120)
    left = rng.uniform(40, 170, (40, 200))
    right = np.zeros_like(left)
    right[:, :-12] = left[:, 12:] * 1.2 + 15
    d = disparity_zncc(left, right)[8:-8, 145:-8]
    assert np.isfinite(d).mean() > 0.95
    np.testing.assert_allclose(d[np.isfinite(d)], 12, atol=0.06)


def test_textureless_images_produce_no_depth():
    image = np.full((30, 180), 100.0)
    assert np.isnan(disparity_zncc(image, image)).all()


def test_repeated_texture_is_not_unique():
    tile = np.random.default_rng(7).uniform(0, 255, (30, 8))
    image = np.tile(tile, (1, 24))
    d = disparity_zncc(image, image)
    assert np.isnan(d[:, 140:-8]).all()
