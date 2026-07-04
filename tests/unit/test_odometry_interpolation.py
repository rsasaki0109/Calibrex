"""Unit tests for odometry SE(3) interpolation helpers."""

from __future__ import annotations

import math
from itertools import pairwise

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


def _reference_linear_scan_interpolate(
    samples: tuple[OdometryPoseSample, ...],
    timestamp_ns: int,
) -> tuple[SE3, bool, float]:
    if not samples:
        return SE3.identity(), False, 0.0
    first_ns = samples[0].timestamp_ns
    last_ns = samples[-1].timestamp_ns
    if timestamp_ns <= first_ns:
        extrapolation_s = max(0.0, (first_ns - timestamp_ns) / 1_000_000_000)
        return samples[0].pose, timestamp_ns < first_ns, extrapolation_s
    if timestamp_ns >= last_ns:
        extrapolation_s = max(0.0, (timestamp_ns - last_ns) / 1_000_000_000)
        return samples[-1].pose, timestamp_ns > last_ns, extrapolation_s
    for left, right in pairwise(samples):
        if left.timestamp_ns <= timestamp_ns <= right.timestamp_ns:
            span_ns = right.timestamp_ns - left.timestamp_ns
            if span_ns <= 0:
                return left.pose, False, 0.0
            alpha = (timestamp_ns - left.timestamp_ns) / span_ns
            return interpolate_se3(left.pose, right.pose, alpha), False, 0.0
    return samples[-1].pose, True, 0.0


def test_odometry_track_bisect_matches_linear_scan() -> None:
    samples = [
        OdometryPoseSample(100, SE3((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        OdometryPoseSample(200, SE3((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        OdometryPoseSample(350, SE3((3.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        OdometryPoseSample(500, SE3((5.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
    ]
    track = OdometryTrack(samples)
    query_times = [0, 50, 100, 150, 200, 275, 350, 450, 500, 600]
    expected_clamp_count = 0
    for timestamp_ns in query_times:
        pose_b, clamped_b, extrap_b = track.interpolate(timestamp_ns)
        pose_r, clamped_r, extrap_r = _reference_linear_scan_interpolate(
            track.samples,
            timestamp_ns,
        )
        assert pose_b.translation_m == pytest.approx(pose_r.translation_m)
        assert pose_b.rotation_quat_xyzw == pytest.approx(pose_r.rotation_quat_xyzw)
        assert clamped_b == clamped_r
        assert extrap_b == pytest.approx(extrap_r)
        if clamped_r:
            expected_clamp_count += 1
    assert track.clamp_count == expected_clamp_count
    assert track.interpolation_count == len(query_times)
