import math

import pytest

from calibrex.core.capture_time import (
    ConstantBodyTwist,
    LidarCaptureTimePolicy,
    TimedLidarPoint,
    deskew_lidar_points_to_reference,
)
from calibrex.core.geometry import SE3


def test_point_capture_time_uses_calibrex_clock_convention() -> None:
    policy = LidarCaptureTimePolicy(
        point_offset_unit="nanoseconds",
        stamp_reference="scan_start",
        sensor_time_offset_sec=0.012,
    )

    assert policy.point_reference_time_sec(10.0, 8_000_000) == pytest.approx(10.020)
    assert policy.as_dict()["time_convention"] == ("sensor_time + dt_sensor = reference_time")


def test_translation_deskew_recovers_static_point_at_camera_time() -> None:
    result = deskew_lidar_points_to_reference(
        [TimedLidarPoint("early", (10.0, 1.0, 0.0), 0.0)],
        message_stamp_sec=4.0,
        reference_time_sec=4.1,
        policy=LidarCaptureTimePolicy("seconds", "scan_start"),
        twist=ConstantBodyTwist((2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        transform_body_lidar=SE3.identity(),
    )

    assert result.points[0].position_lidar_at_reference_m == pytest.approx((9.8, 1.0, 0.0))
    assert result.maximum_abs_deskew_sec == pytest.approx(0.1)


def test_rotation_deskew_uses_each_points_own_capture_time() -> None:
    result = deskew_lidar_points_to_reference(
        [
            TimedLidarPoint("early", (1.0, 0.0, 0.0), -0.1),
            TimedLidarPoint("at-camera", (1.0, 0.0, 0.0), 0.0),
        ],
        message_stamp_sec=3.0,
        reference_time_sec=3.0,
        policy=LidarCaptureTimePolicy("seconds", "scan_end"),
        twist=ConstantBodyTwist((0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2.0)),
        transform_body_lidar=SE3.identity(),
    )

    expected = (math.cos(math.pi / 20.0), -math.sin(math.pi / 20.0), 0.0)
    assert result.points[0].position_lidar_at_reference_m == pytest.approx(expected)
    assert result.points[1].position_lidar_at_reference_m == pytest.approx((1.0, 0.0, 0.0))


def test_nonidentity_lidar_mount_is_composed_during_deskew() -> None:
    result = deskew_lidar_points_to_reference(
        [TimedLidarPoint("point", (4.0, 0.0, 0.0), 0.0)],
        message_stamp_sec=0.0,
        reference_time_sec=0.2,
        policy=LidarCaptureTimePolicy("seconds", "scan_start"),
        twist=ConstantBodyTwist((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        transform_body_lidar=SE3((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )

    assert result.points[0].position_lidar_at_reference_m == pytest.approx((3.8, 0.0, 0.0))


def test_empty_scan_retains_declared_policy() -> None:
    policy = LidarCaptureTimePolicy("seconds", "scan_midpoint", 0.005)
    result = deskew_lidar_points_to_reference(
        [],
        message_stamp_sec=1.0,
        reference_time_sec=1.0,
        policy=policy,
        twist=ConstantBodyTwist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        transform_body_lidar=SE3.identity(),
    )

    assert result.points == ()
    assert result.minimum_capture_time_sec is None
    assert result.policy is policy
