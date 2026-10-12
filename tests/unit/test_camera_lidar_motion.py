"""Moving LiDAR points to the image time with the rig's measured velocity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.data.ros_cdr_writer import encode_imu, encode_odometry, encode_twist_stamped
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.evaluation.camera_lidar_motion import (
    LidarMotion,
    lidar_motion_from_body,
    move_frames_to_image_time,
    move_to_image_time,
    read_lidar_motion,
)

T_BODY_LIDAR = np.eye(4)
T_BODY_LIDAR[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.4]).as_matrix()
T_BODY_LIDAR[:3, 3] = [1.0, -0.5, 0.8]


def _frame(
    points: np.ndarray, *, gap_s: float, offsets: np.ndarray | None = None
) -> CameraLidarFrame:
    return CameraLidarFrame(
        time_s=10.0,
        image_time_s=10.0 + gap_s,
        image=np.zeros((4, 4), dtype=np.uint8),
        points=np.asarray(points, dtype=np.float32),
        cloud_frame_id="lidar",
        image_frame_id="camera",
        point_time_offsets_s=offsets,
    )


def _constant(omega: list[float], velocity: list[float] | None) -> LidarMotion:
    times = np.asarray([9.0, 10.0, 11.0])
    return LidarMotion(
        source_topic="/odom",
        kind="twist" if velocity is not None else "gyro",
        times_s=times,
        angular_rad_s=np.tile(omega, (3, 1)),
        linear_m_s=np.tile(velocity, (3, 1)) if velocity is not None else None,
    )


def test_lidar_velocity_is_the_body_velocity_at_the_lidar_origin() -> None:
    omega_body = np.asarray([0.1, -0.3, 0.5])
    velocity_body = np.asarray([4.0, 0.2, -0.1])
    motion = lidar_motion_from_body(
        "/odom", "twist", [1.0], [omega_body], [velocity_body], T_BODY_LIDAR
    )
    rotation, lever = T_BODY_LIDAR[:3, :3], T_BODY_LIDAR[:3, 3]
    np.testing.assert_allclose(motion.angular_rad_s[0], rotation.T @ omega_body)
    np.testing.assert_allclose(
        motion.linear_m_s[0],  # type: ignore[index]
        rotation.T @ (velocity_body + np.cross(omega_body, lever)),
    )
    gyro = lidar_motion_from_body("/imu", "gyro", [1.0], [omega_body], None, T_BODY_LIDAR)
    assert gyro.linear_m_s is None


def test_velocity_is_interpolated_and_absent_far_from_every_sample() -> None:
    motion = lidar_motion_from_body(
        "/odom",
        "twist",
        [2.0, 1.0],  # out of order on purpose: samples are sorted by time
        [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        np.eye(4),
    )
    found = motion.velocity_at(1.25)
    assert found is not None
    np.testing.assert_allclose(found[0], [0.0, 0.0, 1.25])
    np.testing.assert_allclose(found[1], [1.25, 0.0, 0.0])  # type: ignore[arg-type]
    assert motion.velocity_at(2.2) is not None
    assert motion.velocity_at(2.3) is None  # more than 0.25 s past the last sample
    assert motion.velocity_at(0.7) is None


def test_points_are_moved_to_where_the_lidar_saw_them_at_the_image_time() -> None:
    omega, velocity, gap = [0.0, 0.0, np.radians(30.0)], [5.0, 0.0, 0.0], 0.04
    points = np.asarray([[10.0, 0.0, 0.0], [0.0, 8.0, 1.0]])
    result = move_to_image_time(_frame(points, gap_s=gap), _constant(omega, velocity))
    assert result is not None
    moved, rotation_deg, translation_m = result
    # the scan was taken 40 ms before the image: undo the motion over -40 ms
    turn = Rotation.from_rotvec(np.asarray(omega) * -gap).as_matrix()
    expected = points @ turn.T + np.asarray(velocity) * -gap
    np.testing.assert_allclose(moved.points, expected, atol=1.0e-5)
    assert rotation_deg == pytest.approx(30.0 * gap)
    assert translation_m == pytest.approx(5.0 * gap)
    assert moved.time_s == 10.0 and moved.image_time_s == 10.0 + gap


def test_each_point_is_moved_by_its_own_measurement_time() -> None:
    omega = [0.0, 0.0, np.radians(90.0)]
    points = np.asarray([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    # the second point was measured 50 ms after the reference time, at the image time
    frame = _frame(points, gap_s=0.05, offsets=np.asarray([0.0, 0.05]))
    result = move_to_image_time(frame, _constant(omega, None))
    assert result is not None
    moved = result[0].points
    np.testing.assert_allclose(moved[1], points[1], atol=1.0e-5)
    turn = Rotation.from_rotvec(np.asarray(omega) * -0.05).as_matrix()
    np.testing.assert_allclose(moved[0], turn @ points[0], atol=1.0e-5)


def test_frames_without_a_motion_sample_are_kept_and_counted() -> None:
    motion = _constant([0.0, 0.0, 0.1], [1.0, 0.0, 0.0])
    near = _frame(np.ones((3, 3)), gap_s=0.02)
    far = CameraLidarFrame(
        time_s=20.0,
        image_time_s=20.02,
        image=near.image,
        points=near.points,
        cloud_frame_id="lidar",
        image_frame_id="camera",
    )
    moved, summary = move_frames_to_image_time([near, far], motion)
    assert moved[1] is far
    assert (summary.frames_moved, summary.frames_without_motion) == (1, 1)
    assert summary.kind == "twist" and not summary.per_point_time
    assert summary.median_translation_m == pytest.approx(0.02)


def test_motion_is_read_from_odometry_twist_and_imu_topics(tmp_path: Path) -> None:
    omega, velocity = (0.0, 0.1, 0.2), (3.0, 0.0, 0.0)
    bag = tmp_path / "bag"
    with Rosbag2Writer(bag) as writer:
        writer.add_topic("/odom", "nav_msgs/msg/Odometry")
        writer.add_topic("/twist", "geometry_msgs/msg/TwistStamped")
        writer.add_topic("/imu", "sensor_msgs/msg/Imu")
        for tick in range(5):
            stamp = 1_000_000_000 + tick * 100_000_000
            writer.write(
                "/odom",
                stamp,
                encode_odometry(
                    frame_id="odom",
                    child_frame_id="base_link",
                    timestamp_ns=stamp,
                    position=(0.0, 0.0, 0.0),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    linear_velocity=velocity,
                    angular_velocity=omega,
                ),
            )
            writer.write(
                "/twist",
                stamp,
                encode_twist_stamped(
                    frame_id="base_link", timestamp_ns=stamp, linear=velocity, angular=omega
                ),
            )
            writer.write(
                "/imu",
                stamp,
                encode_imu(
                    frame_id="base_link",
                    timestamp_ns=stamp,
                    angular_velocity=omega,
                    linear_acceleration=(0.0, 0.0, 9.81),
                ),
            )
    expected = lidar_motion_from_body(
        "/odom", "twist", [1.0], [omega], [velocity], T_BODY_LIDAR
    )
    for topic, message_type in (
        ("/odom", "nav_msgs/msg/Odometry"),
        ("/twist", "geometry_msgs/msg/TwistStamped"),
    ):
        motion = read_lidar_motion(bag, topic, message_type, T_BODY_LIDAR)
        assert motion is not None and motion.kind == "twist"
        assert motion.times_s.tolist() == pytest.approx([1.0, 1.1, 1.2, 1.3, 1.4])
        np.testing.assert_allclose(motion.angular_rad_s[2], expected.angular_rad_s[0])
        np.testing.assert_allclose(motion.linear_m_s[2], expected.linear_m_s[0])  # type: ignore[index]
    gyro = read_lidar_motion(bag, "/imu", "sensor_msgs/msg/Imu", T_BODY_LIDAR)
    assert gyro is not None and gyro.kind == "gyro" and gyro.linear_m_s is None
    np.testing.assert_allclose(gyro.angular_rad_s[0], expected.angular_rad_s[0])
    with pytest.raises(ValueError, match="not a velocity source"):
        read_lidar_motion(bag, "/imu", "sensor_msgs/msg/NavSatFix", T_BODY_LIDAR)
