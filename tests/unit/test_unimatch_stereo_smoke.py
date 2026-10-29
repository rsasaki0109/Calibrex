"""Pixel geometry must survive learned-model padding without disparity rescaling."""

import numpy as np
import pytest
from tools.unimatch_stereo_smoke import padding


@pytest.mark.parametrize("shape", [(375, 1242), (384, 1248), (1, 1)])
def test_padding_preserves_pixel_coordinates_and_disparity(shape):
    h, w = shape
    left, right, top, bottom = padding(h, w)
    image = np.arange(h * w).reshape(h, w)
    padded = np.pad(image, ((top, bottom), (left, right)), mode="edge")
    assert padded.shape[0] % 32 == padded.shape[1] % 32 == 0
    np.testing.assert_array_equal(padded[top : top + h, left : left + w], image)
    disparity = np.full(padded.shape, 12.5)
    np.testing.assert_array_equal(disparity[top : top + h, left : left + w], np.full(shape, 12.5))
