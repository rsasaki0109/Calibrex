"""The recording-motion explanation attached to ``no_judgeable_axes``."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from calibrex.check.motion_summary import (
    motion_note,
    recording_motion_note,
    rotation_excursions_deg,
)
from calibrex.data.ros_cdr_writer import encode_imu
from calibrex.data.rosbag2_writer import Rosbag2Writer


def test_excursions_integrate_angular_rate_per_axis() -> None:
    t = np.linspace(0.0, 10.0, 1001)
    rate = np.zeros((t.size, 3))
    rate[:, 2] = np.radians(5.0)  # 5 deg/s about z for 10 s
    duration, excursions = rotation_excursions_deg(t, rate)

    assert duration == pytest.approx(10.0)
    assert excursions[2] == pytest.approx(50.0, abs=0.1)
    assert excursions[0] == 0.0 and excursions[1] == 0.0


def test_short_static_recording_asks_for_time_and_named_axes() -> None:
    note = motion_note(36.0, (4.0, 5.0, 1.5))

    assert "too short or too static" in note
    assert "36 s" in note and "z 1.5 deg" in note
    assert "about 60 s or more" in note
    assert "IMU x/y/z axes" in note


def test_long_dynamic_recording_blames_the_estimator_not_the_length() -> None:
    note = motion_note(120.0, (40.0, 50.0, 90.0))

    assert "too short" not in note
    assert "excitation looks sufficient" in note


def test_single_weak_axis_is_named_alone() -> None:
    note = motion_note(90.0, (40.0, 50.0, 2.0))

    assert "IMU z axis" in note and "about 60 s" not in note


def test_too_few_samples_give_no_excursion() -> None:
    assert rotation_excursions_deg([0.0], np.zeros((1, 3))) == (0.0, (0.0, 0.0, 0.0))


@pytest.mark.parametrize('encoding', ['cdr', 'json', 'protobuf', ''])
def test_recording_motion_note_requires_cdr_even_when_payload_bytes_are_decodable(
    tmp_path, encoding,
):
    bag = tmp_path / 'motion'
    with Rosbag2Writer(bag) as writer:
        writer.add_topic('/imu', 'sensor_msgs/msg/Imu')
        for timestamp_ns in (1_000_000_000, 2_000_000_000, 3_000_000_000):
            writer.write('/imu', timestamp_ns, encode_imu(
                frame_id='imu', timestamp_ns=timestamp_ns,
                angular_velocity=(0.0, 0.0, 0.1), linear_acceleration=(0.0, 0.0, 9.8),
            ))
    with sqlite3.connect(next(bag.glob('*.db3'))) as db:
        db.execute('UPDATE topics SET serialization_format = ?', (encoding,))
    note = recording_motion_note(bag, '/imu')
    if encoding == 'cdr':
        assert note is not None
        assert '2 s' in note and 'z 11.5 deg' in note
    else:
        assert note is None


def test_rates_are_given_in_the_frame_of_the_errors() -> None:
    from calibrex.check.motion_summary import rotation_rates_dps

    gyro = np.zeros((100, 3))
    gyro[:, 2] = np.radians(30.0)  # the IMU turns about its z axis
    # the camera's y axis is the IMU's z axis (R_cam_imu maps imu z onto cam y)
    rotation_cam_imu = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    assert rotation_rates_dps(gyro, rotation_cam_imu) == pytest.approx((0.0, 30.0, 0.0))
    assert rotation_rates_dps(np.zeros((0, 3)), np.eye(3)) == (0.0, 0.0, 0.0)


def test_advice_names_the_other_two_axes_when_the_rig_did_not_rotate_about_them() -> None:
    from calibrex.check.motion_summary import rotation_axis_advice

    advice = rotation_axis_advice("pitch", "cam0", (2.0, 31.0, 3.0), camera=True)

    # a pitch (y) error shows only in rotation about x or z, never about y itself
    assert advice.startswith("pitch, rotate the rig about the x or z axis of cam0")
    assert "tilt it up and down for x, or roll it for z" in advice
    assert "x 2, y 31, z 3 deg/s" in advice
    lidar = rotation_axis_advice("yaw", "lidar", (2.0, 3.0, 31.0))
    assert "about the x or y axis of lidar:" in lidar and "tilt" not in lidar


def test_advice_does_not_ask_for_motion_the_recording_already_has() -> None:
    from calibrex.check.motion_summary import rotation_axis_advice

    # TUM-VI corridor4: pitch unchecked although the rig rotated about x and z
    advice = rotation_axis_advice("pitch", "cam0", (34.0, 43.0, 27.0), camera=True)

    assert advice.startswith("pitch, other motion is not what is missing")
    assert "x 34, y 43, z 27 deg/s" in advice
    assert "not expected to help" in advice and "tilt" not in advice


def test_a_short_recording_with_the_right_motion_asks_for_more_of_it() -> None:
    from calibrex.check.motion_summary import rotation_axis_advice

    # LiU GoPro playground1: 32 s, the rig rotated about every axis
    advice = rotation_axis_advice("roll", "cam0", (12.0, 22.0, 10.0), duration_s=32.0)

    assert advice.startswith("roll, record the same motion for longer")
    assert "only for 32 s; record about 60 s or more" in advice
    assert "not expected to help" not in advice
    long = rotation_axis_advice("roll", "cam0", (12.0, 22.0, 10.0), duration_s=140.0)
    assert "not expected to help" in long
