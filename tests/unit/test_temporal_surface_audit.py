import cv2
import numpy as np
from tools.audit_temporal_surface_inputs import image_pair


def test_blank_images_do_not_produce_motion_support(tmp_path):
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in paths:
        assert cv2.imwrite(str(path), np.zeros((80, 100), dtype=np.uint8))
    result = image_pair(*paths, [100, 100, 50, 40])
    assert result["matches"] == result["essential_inliers"] == result["cheirality_inliers"] == 0
    assert result["cells"] == [0, 0]
