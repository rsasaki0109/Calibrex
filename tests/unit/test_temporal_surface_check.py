import numpy as np
from tools.temporal_surface_check import classify


def test_projection_categories_and_occlusion_sign():
    points = np.array([[0, 0, 5.0], [100, 0, 5.0], [0, 0, -1.0]])
    args = [np.eye(4), [10, 10, 5, 5], 120]
    np.testing.assert_array_equal(classify(points, np.full((12, 12), 5.0), *args), [3, 1, 0])
    assert classify(points[:1], np.full((12, 12), 3.0), *args)[0] == 4
    assert classify(points[:1], np.full((12, 12), 10.0), *args)[0] == 5
    assert classify(points[:1], np.full((12, 12), np.nan), *args)[0] == 2


def test_target_from_source_translation_and_nontrivial_depth():
    transform = np.eye(4)
    transform[2, 3] = 1.0
    result = classify(
        np.array([[0, 0, 5.0]]), np.full((12, 12), 6.0), transform, [10, 10, 5, 5], 120
    )
    assert result[0] == 3
