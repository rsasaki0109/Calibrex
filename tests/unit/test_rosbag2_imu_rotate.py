from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr_writer import encode_header_only, encode_imu
from calibrex.data.rosbag2 import iter_messages, read_imu_messages
from calibrex.data.rosbag2_imu_rotate import (
    rotate_imu_in_bag,
    rotate_imu_payload,
    rotation_from_axis_angle_deg,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer

OMEGA = (0.1, -0.2, 0.3)
ACC = (0.5, 0.25, 9.8)


def _bag(path: Path, *, orientation_known: bool = False) -> Path:
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/imu", "sensor_msgs/msg/Imu")
        writer.add_topic("/other", "std_msgs/msg/Header")
        for i in range(5):
            ns = 1_000_000_000 + i * 10_000_000
            writer.write(
                "/imu",
                ns,
                encode_imu(
                    frame_id="imu_link",
                    timestamp_ns=ns,
                    angular_velocity=OMEGA,
                    linear_acceleration=ACC,
                    orientation_known=orientation_known,
                ),
            )
        writer.write("/other", 1_000_000_000, encode_header_only("x"))
    return path


def test_payload_rotation_matches_the_matrix() -> None:
    rotation = rotation_from_axis_angle_deg((0, 0, 1), 2.0)
    payload = encode_imu(
        frame_id="a_longer_frame", timestamp_ns=5, angular_velocity=OMEGA, linear_acceleration=ACC
    )
    out = rotate_imu_payload(payload, rotation)
    assert len(out) == len(payload)
    # header, orientation and covariances are untouched: only the two vectors differ
    differing = [i for i, (a, b) in enumerate(zip(payload, out, strict=True)) if a != b]
    assert differing
    assert max(differing) - min(differing) > 8  # angular and linear vectors


def test_rotate_bag_rotates_only_imu_vectors(tmp_path: Path) -> None:
    source = _bag(tmp_path / "src")
    rotation = rotation_from_axis_angle_deg((1, 2, 3), 1.5)
    summary = rotate_imu_in_bag(source, tmp_path / "dst", rotation)
    assert summary.imu_messages_rotated == 5
    assert summary.messages_copied == 6
    assert summary.rotation_angle_deg == pytest.approx(1.5, abs=1e-9)
    original = list(read_imu_messages(source))
    rotated = list(read_imu_messages(tmp_path / "dst"))
    assert len(rotated) == len(original) == 5
    for before, after in zip(original, rotated, strict=True):
        assert np.allclose(after.angular_velocity, rotation @ np.array(before.angular_velocity))
        assert np.allclose(
            after.linear_acceleration, rotation @ np.array(before.linear_acceleration)
        )
        # a rotation preserves norms: what the IMU measures in magnitude is unchanged
        assert np.linalg.norm(after.linear_acceleration) == pytest.approx(
            np.linalg.norm(before.linear_acceleration)
        )
        assert after.frame_id == before.frame_id
    other = [p for c, _, p in iter_messages(tmp_path / "dst") if c.topic == "/other"]
    assert other == [encode_header_only("x")]


def test_known_orientation_is_rotated_and_stays_unit(tmp_path: Path) -> None:
    source = _bag(tmp_path / "src", orientation_known=True)
    rotate_imu_in_bag(source, tmp_path / "dst", rotation_from_axis_angle_deg((0, 1, 0), 90.0))
    quat = np.array(next(iter(read_imu_messages(tmp_path / "dst"))).orientation_xyzw)
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    assert quat[1] == pytest.approx(math.sin(math.radians(45.0)))


def test_duration_limit_and_errors(tmp_path: Path) -> None:
    source = _bag(tmp_path / "src")
    summary = rotate_imu_in_bag(source, tmp_path / "short", np.eye(3), max_duration_s=0.025)
    assert summary.imu_messages_rotated == 3
    with pytest.raises(DatasetError):
        rotate_imu_in_bag(source, tmp_path / "bad", np.eye(3), imu_topics=["/other"])
    with pytest.raises(ValueError):
        rotate_imu_in_bag(source, tmp_path / "bad2", np.eye(3) * 2.0)
