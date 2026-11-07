"""Fisheye rays beyond 90 deg: their angle from the optical axis, and the tracker's cap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from calibrex.evaluation.visual_rotation import CameraModel, VisualRotationOptions

# TUM-VI's cam0 (Kalibr, 512 x 512 equidistant): the image corners are about 114 deg off axis
TUMVI = CameraModel(
    190.978, 190.973, 254.932, 256.897, "equidistant", (0.00348, 0.000715, -0.00205, 0.000203)
)


def test_incidence_of_an_equidistant_lens_reaches_past_90_deg() -> None:
    angles = np.degrees(TUMVI.incidence_rad(np.array([[254.932, 256.897], [511.0, 511.0]])))
    assert angles[0] == pytest.approx(0.0, abs=1e-6)
    assert angles[1] > 110.0
    # theta_d = theta (1 + k1 theta^2 + ...): the inverse reproduces a known angle
    theta = math.radians(100.0)
    k = TUMVI.distortion
    theta_d = theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)
    pixel = np.array([[TUMVI.cx + TUMVI.fx * theta_d, TUMVI.cy]])
    assert np.degrees(TUMVI.incidence_rad(pixel))[0] == pytest.approx(100.0, abs=1e-6)


def test_perspective_coordinates_flip_beyond_90_deg() -> None:
    """Why the cap exists: normalize() puts a 114 deg corner on the opposite side."""

    corner = TUMVI.normalize(np.array([[511.0, 511.0]]))[0]
    assert corner[0] < 0 and corner[1] < 0


def test_pinhole_incidence_matches_the_ray_angle() -> None:
    camera = CameraModel(500.0, 500.0, 320.0, 240.0)
    pixel = np.array([[320.0 + 500.0, 240.0]])  # x/z = 1: 45 deg
    assert np.degrees(camera.incidence_rad(pixel))[0] == pytest.approx(45.0)


def test_the_cap_is_off_by_default() -> None:
    assert VisualRotationOptions().max_incidence_deg is None
