"""Unit tests for odometry SE(3) interpolation helpers."""

from __future__ import annotations

import math

import pytest

from calibrex.core.geometry import SE3, interpolate_se3, slerp_quaternion_xyzw
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack


def test_slerp_identity_endpoints() -> None:
    left = (0.0, 0.0, 0.0, 1.0)
    right = (0.0, 0.0, 0.70710678, 0.70710678)
    assert slerp_quaternion_xyzw(left, left, 0.25) == pytest.approx(left)
    assert slerp_quaternion_xyzw(left, right, 1.0) == pytest.approx(right)


def test_slerp_antipodal_quaternions_take_shortest_arc() -> None:
    left = (0.0, 0.0, 0.0, 1.0)
    antipodal = (0.0, 0.0, 0.0, -1.0)
    midpoint = slerp_quaternion_xyzw(left, antipodal, 0.5)
    assert midpoint == pytest.approx(left)


def test_interpolate_se3_blends_translation_and_rotation() -> None:
    left = SE3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    right = SE3((2.0, 0.0, 0.0), (0.0, 0.0, 0.70710678, 0.70710678))
    mid = interpolate_se3(left, right, 0.5)
    assert mid.translation_m == pytest.approx((1.0, 0.0, 0.0))
    assert mid.rotation_quat_xyzw[2] == pytest.approx(0.3826834, abs=1.0e-5)


def test_odometry_track_interpolates_between_samples() -> None:
    track = OdometryTrack(
        [
            OdometryPoseSample(0, SE3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
            OdometryPoseSample(100, SE3((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        ]
    )
    pose, clamped, extrapolation_s = track.interpolate(50)
    assert clamped is False
    assert extrapolation_s == 0.0
    assert pose.translation_m == pytest.approx((0.5, 0.0, 0.0))
    assert track.clamp_count == 0


def test_odometry_track_clamps_out_of_range_and_counts() -> None:
    track = OdometryTrack(
        [
            OdometryPoseSample(100, SE3((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
            OdometryPoseSample(200, SE3((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        ]
    )
    early_pose, early_clamped, early_extrap = track.interpolate(0)
    late_pose, late_clamped, late_extrap = track.interpolate(500)
    assert early_clamped is True
    assert late_clamped is True
    assert early_extrap == pytest.approx(100 / 1_000_000_000)
    assert late_extrap == pytest.approx(300 / 1_000_000_000)
    assert early_pose.translation_m == pytest.approx((1.0, 0.0, 0.0))
    assert late_pose.translation_m == pytest.approx((2.0, 0.0, 0.0))
    assert track.clamp_count == 2


def test_odometry_track_slerp_rotation_midpoint() -> None:
    yaw_90 = (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
    track = OdometryTrack(
        [
            OdometryPoseSample(0, SE3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
            OdometryPoseSample(100, SE3((0.0, 0.0, 0.0), yaw_90)),
        ]
    )
    pose, _clamped, extrapolation_s = track.interpolate(50)
    assert extrapolation_s == 0.0
    assert pose.rotation_quat_xyzw[2] == pytest.approx(math.sin(math.pi / 8.0), abs=1.0e-6)
