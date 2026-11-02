"""Overlay images of the camera-lidar check: projection, colours, depth order, encoding."""

from __future__ import annotations

import sys
import zlib
from io import BytesIO

import numpy as np
import pytest

from calibrex.evaluation.camera_lidar_edge import EdgeCamera
from calibrex.visualization.camera_lidar_check_overlay import (
    depth_colors,
    encode_overlay,
    encode_png,
    overlay_rgb,
)

CAMERA = EdgeCamera(100.0, 100.0, 50.0, 40.0)
IDENTITY = np.eye(4)
GRAY = np.full((80, 100), 200, dtype=np.uint8)


def test_a_point_is_drawn_where_it_projects_in_its_depth_colour() -> None:
    point = np.array([[0.5, -0.2, 5.0]])  # u = 60, v = 36
    image = overlay_rgb(GRAY, point, CAMERA, IDENTITY)
    assert image.shape == (80, 100, 3)
    expected = depth_colors(np.array([5.0]))[0]
    for row, column in ((36, 60), (36, 61), (37, 60), (37, 61)):
        np.testing.assert_array_equal(image[row, column], expected)
    background = np.uint8(round(200 * 0.7))
    np.testing.assert_array_equal(image[10, 10], [background] * 3)


def test_the_transform_moves_the_points() -> None:
    shifted = np.eye(4)
    shifted[0, 3] = 0.2  # 0.2 m at 5 m: 4 px to the right
    point = np.array([[0.5, -0.2, 5.0]])
    image = overlay_rgb(GRAY, point, CAMERA, shifted)
    expected = depth_colors(np.array([5.0]))[0]
    np.testing.assert_array_equal(image[36, 64], expected)
    assert not np.array_equal(image[36, 60], expected)


def test_the_nearest_point_wins_and_points_behind_or_outside_are_dropped() -> None:
    near, far = depth_colors(np.array([2.0, 50.0]))
    points = np.array(
        [
            [10.0, -4.0, 50.0],  # far, at u = 70, v = 32
            [0.4, -0.16, 2.0],  # near, same pixel
            [0.0, 0.0, -3.0],  # behind the camera
            [9.0, 0.0, 1.0],  # outside the image
        ]
    )
    image = overlay_rgb(GRAY, points, CAMERA, IDENTITY)
    np.testing.assert_array_equal(image[32, 70], near)
    assert not np.any(np.all(image == far, axis=2))
    assert np.count_nonzero(np.any(image != image[0, 0], axis=2)) == 4


def test_colours_run_from_red_near_to_blue_far() -> None:
    near, middle, far = depth_colors(np.array([0.5, 9.0, 500.0]))
    assert near[0] > 200 and near[2] < 80
    assert far[2] > 200 and far[0] < 120
    assert tuple(middle) not in {tuple(near), tuple(far)}


def test_large_images_are_downscaled_to_the_maximum_side() -> None:
    gray = np.zeros((1500, 2000), dtype=np.uint8)
    camera = EdgeCamera(1000.0, 1000.0, 1000.0, 750.0)
    image = overlay_rgb(gray, np.array([[0.0, 0.0, 10.0]]), camera, IDENTITY, max_side=960)
    assert image.shape == (500, 666, 3)
    assert np.any(image[250, 333] != 0)


def test_png_encoder_round_trips_and_pillow_writes_jpeg() -> None:
    rgb = np.random.default_rng(1).integers(0, 255, (7, 5, 3), dtype=np.uint8)
    png = encode_png(rgb)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    start = png.index(b"IDAT") + 4
    length = int.from_bytes(png[start - 8 : start - 4], "big")
    rows = np.frombuffer(zlib.decompress(png[start : start + length]), dtype=np.uint8)
    np.testing.assert_array_equal(rows.reshape(7, 16)[:, 1:].reshape(7, 5, 3), rgb)
    Image = pytest.importorskip("PIL.Image")
    np.testing.assert_array_equal(np.asarray(Image.open(BytesIO(png))), rgb)
    image = encode_overlay("overlay_candidate", rgb)
    assert (image.media_type, image.suffix) == ("image/jpeg", ".jpg")
    assert image.data.startswith(b"\xff\xd8")


def test_without_pillow_the_overlay_is_a_png(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "PIL", None)
    image = encode_overlay("overlay_estimate", np.zeros((3, 4, 3), dtype=np.uint8))
    assert (image.role, image.media_type, image.suffix) == ("overlay_estimate", "image/png", ".png")
    assert image.data.startswith(b"\x89PNG")
