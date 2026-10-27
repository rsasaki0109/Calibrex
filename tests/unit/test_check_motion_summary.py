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
