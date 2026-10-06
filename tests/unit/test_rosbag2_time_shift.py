from __future__ import annotations

from pathlib import Path

import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr_writer import encode_header_only, encode_imu
from calibrex.data.rosbag2 import iter_messages, read_imu_messages
from calibrex.data.rosbag2_time_shift import shift_header_stamp, shift_stamps_in_bag
from calibrex.data.rosbag2_writer import Rosbag2Writer

OMEGA = (0.1, -0.2, 0.3)
ACC = (0.5, 0.25, 9.8)
T0 = 1_700_000_000_999_000_000


def _bag(path: Path) -> Path:
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/imu", "sensor_msgs/msg/Imu")
        writer.add_topic("/other", "std_msgs/msg/Header")
        for i in range(5):
            ns = T0 + i * 10_000_000
            writer.write(
                "/imu",
                ns,
                encode_imu(
                    frame_id="imu_link",
                    timestamp_ns=ns,
                    angular_velocity=OMEGA,
                    linear_acceleration=ACC,
                ),
            )
        writer.write("/other", T0, encode_header_only("x"))
    return path


def _stamp_ns(data: bytes) -> int:
    return int.from_bytes(data[4:8], "little", signed=True) * 10**9 + int.from_bytes(
        data[8:12], "little"
    )


def test_header_stamp_shift_crosses_the_second_boundary_and_changes_only_the_stamp() -> None:
    payload = encode_imu(
        frame_id="a_frame", timestamp_ns=T0, angular_velocity=OMEGA, linear_acceleration=ACC
    )
    out = shift_header_stamp(payload, 5_000_000)
    assert out is not None and len(out) == len(payload)
    assert out[12:] == payload[12:]
    assert _stamp_ns(out) - _stamp_ns(payload) == 5_000_000
    back = shift_header_stamp(payload, -5_000_000)
    assert back is not None and _stamp_ns(payload) - _stamp_ns(back) == 5_000_000


def test_zero_stamp_is_left_alone_and_negative_epoch_is_refused() -> None:
    zero = encode_imu(frame_id="a", timestamp_ns=0, angular_velocity=OMEGA, linear_acceleration=ACC)
    assert shift_header_stamp(zero, 1_000) is None
    early = encode_imu(
        frame_id="a", timestamp_ns=1_000, angular_velocity=OMEGA, linear_acceleration=ACC
    )
    with pytest.raises(DatasetError):
        shift_header_stamp(early, -1_000_000)


def test_shift_bag_moves_only_the_chosen_topic_header_stamps(tmp_path: Path) -> None:
    source = _bag(tmp_path / "src")
    summary = shift_stamps_in_bag(source, tmp_path / "dst", 0.0075, topics=["/imu"])
    assert summary.messages_shifted == 5 and summary.messages_copied == 6
    original = list(read_imu_messages(source))
    shifted = list(read_imu_messages(tmp_path / "dst"))
    for before, after in zip(original, shifted, strict=True):
        assert after.timestamp_ns - before.timestamp_ns == 7_500_000
        assert after.angular_velocity == before.angular_velocity
    # the recorder's log time and the other topic are untouched
    log_before = [(c.topic, t) for c, t, _ in iter_messages(source)]
    log_after = [(c.topic, t) for c, t, _ in iter_messages(tmp_path / "dst")]
    assert log_before == log_after
    other = [p for c, _, p in iter_messages(tmp_path / "dst") if c.topic == "/other"]
    assert other == [encode_header_only("x")]


def test_log_time_option_duration_limit_and_errors(tmp_path: Path) -> None:
    source = _bag(tmp_path / "src")
    shift_stamps_in_bag(source, tmp_path / "dst", -0.002, topics=["/imu"], shift_log_time=True)
    log = [t for c, t, _ in iter_messages(tmp_path / "dst") if c.topic == "/imu"]
    assert log[0] == T0 - 2_000_000
    short = shift_stamps_in_bag(
        source, tmp_path / "short", 0.001, topics=["/imu"], max_duration_s=0.025
    )
    assert short.messages_shifted == 3
    with pytest.raises(DatasetError):
        shift_stamps_in_bag(source, tmp_path / "bad", 0.001, topics=["/nope"])
    with pytest.raises(ValueError):
        shift_stamps_in_bag(source, tmp_path / "bad2", 0.001, topics=[])
