"""Fisheye (equidistant) projection and stereo rectification on Hilti 2022's alphasense."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from calibrex.evaluation.fisheye_stereo import (
    EquidistantCamera,
    estimate_vertical_offset,
    read_kalibr_fisheye_pair,
    rectify_fisheye_pair,
    with_right_row_shift,
)

CAMCHAIN = Path("data/public/hilti_2022/calib_3_cam0-1-camchain-imucam.yaml")
# Hilti 2022's alphasense cam0 and cam1 (Kalibr camchain), written out so the test does
# not need the dataset
CAM0 = EquidistantCamera(
    351.31400364193297, 351.4911744656785, 367.8522793375995, 253.8402144980996,
    (-0.03696737352869157, -0.008917880497032812, 0.008912969593422046, -0.0037685977496087313),
    720, 540,
)  # fmt: skip
CAM1 = EquidistantCamera(
    352.6489794433894, 352.8586498571586, 347.8170010310082, 270.5806692485468,
    (-0.039086652082708805, -0.005525347047415151, 0.004398151558986798, -0.0019701263170917808),
    720, 540,
)  # fmt: skip
CAM1_FROM_CAM0 = np.array(
    [
        [0.9999761426988956, 0.004698109260098521, -0.005063773535634404, 0.10817339479208792],
        [-0.004705552944957792, 0.9999878643586437, -0.0014590774217762541, 0.0005128409082424196],
        [0.005056857178348414, 0.0014828704666000705, 0.9999861145489266, 0.0006919599620310546],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _points(count: int = 400) -> np.ndarray:
    """Points in front of cam0, up to 60 deg off its axis, 1 to 20 m away."""

    rng = np.random.default_rng(3)
    angle = np.radians(rng.uniform(0.0, 60.0, count))
    azimuth = rng.uniform(-np.pi, np.pi, count)
    distance = rng.uniform(1.0, 20.0, count)
    rays = np.c_[np.sin(angle) * np.cos(azimuth), np.sin(angle) * np.sin(azimuth), np.cos(angle)]
    return rays * distance[:, None]


def test_unprojecting_a_projected_point_gives_its_ray_back() -> None:
    points = _points()
    u, v, front = CAM0.project(points)
    assert front.all()
    rays = CAM0.unproject(u, v)
    expected = points / np.linalg.norm(points, axis=1, keepdims=True)
    np.testing.assert_allclose(rays, expected, atol=1e-9)


def test_rectified_rows_agree_and_the_maps_point_back_to_the_raw_pixels() -> None:
    rectified = rectify_fisheye_pair(CAM0, CAM1, CAM1_FROM_CAM0)
    assert rectified.baseline_m == pytest.approx(0.1082, abs=1e-3)
    stereo = rectified.stereo
    points = _points()
    in_rectified = points @ rectified.rectified_from_left.T
    in_view = in_rectified[:, 2] > 0.5
    homogeneous = np.c_[in_rectified[in_view], np.ones(int(in_view.sum()))]
    left = homogeneous @ stereo.left.T
    right = homogeneous @ stereo.right.T
    u_left, v_left = left[:, 0] / left[:, 2], left[:, 1] / left[:, 2]
    u_right, v_right = right[:, 0] / right[:, 2], right[:, 1] / right[:, 2]
    # rectified: the same point is on the same row in both images, base / Z apart
    np.testing.assert_allclose(v_left, v_right, atol=1e-9)
    np.testing.assert_allclose(
        u_left - u_right, stereo.base_px_m / in_rectified[in_view, 2], rtol=1e-9
    )

    inside = (u_left > 1) & (u_left < 718) & (v_left > 1) & (v_left < 538)
    inside &= (u_right > 1) & (u_right < 718)
    raw = points[in_view][inside]
    for camera_map, camera, cam_points, u, v in (
        (rectified.left_map, CAM0, raw, u_left[inside], v_left[inside]),
        (
            rectified.right_map,
            CAM1,
            raw @ CAM1_FROM_CAM0[:3, :3].T + CAM1_FROM_CAM0[:3, 3],
            u_right[inside],
            v_right[inside],
        ),
    ):
        mapped_row = ndimage.map_coordinates(camera_map[0], [v, u], order=1)
        mapped_col = ndimage.map_coordinates(camera_map[1], [v, u], order=1)
        raw_u, raw_v, _ = camera.project(cam_points)
        # the maps are sampled per pixel and interpolated: well under a tenth of a pixel
        np.testing.assert_allclose(mapped_col, raw_u, atol=0.05)
        np.testing.assert_allclose(mapped_row, raw_v, atol=0.05)


def test_a_rectified_image_samples_the_raw_one() -> None:
    rectified = rectify_fisheye_pair(CAM0, CAM1, CAM1_FROM_CAM0)
    # an image whose value is its raw column: rectified, each pixel holds the raw column
    # its map points to
    raw = np.tile(np.arange(720, dtype=np.float64), (540, 1))
    image = rectified.rectify_left(raw)
    centre = (270, 360)
    assert image[centre] == pytest.approx(rectified.left_map[1][centre], abs=1e-6)


@pytest.mark.skipif(not CAMCHAIN.exists(), reason="Hilti 2022 calibration not downloaded")
def test_the_kalibr_camchain_gives_the_same_pair() -> None:
    left, right, right_from_left = read_kalibr_fisheye_pair(CAMCHAIN)
    assert left == CAM0 and right == CAM1
    np.testing.assert_allclose(right_from_left, CAM1_FROM_CAM0)


def test_a_vertical_offset_of_the_right_image_is_measured_and_removed() -> None:
    rectified = rectify_fisheye_pair(CAM0, CAM1, CAM1_FROM_CAM0)
    rng = np.random.default_rng(5)
    texture = ndimage.gaussian_filter(rng.uniform(0, 255, (540, 720)), 1.5) * 4.0
    # the right image: the scene 12 px to the left (disparity) and 0.7 px higher
    right = ndimage.shift(texture, (-0.7, -12.0), order=1, mode="nearest")
    offset = estimate_vertical_offset([(texture, right)], patches_per_pair=120)
    assert offset == pytest.approx(-0.7, abs=0.1)

    # with the offset removed, rectified row r of the right image takes row r - 0.7: the
    # content that sat 0.7 px higher is back on the left image's row
    raw = np.tile(np.arange(540, dtype=np.float64)[:, None], (1, 720))
    shifted = with_right_row_shift(rectified, -0.7)
    centre = (270, 360)
    assert shifted.rectify_right(raw)[centre] == pytest.approx(
        ndimage.map_coordinates(rectified.right_map[0], [[269.3], [360.0]], order=1)[0], abs=1e-3
    )
