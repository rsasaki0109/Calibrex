"""camera-imu tracks images about 0.1 s apart, whatever the camera's frame rate.

The rotation solver compares tracked rotations with the gyro over intervals of at most
``max_step_s`` (0.25 s). The fixed stride of 4 was chosen for the 40 Hz Hilti cameras; on
a 10 Hz camera it put tracked images 0.4 s apart, and no interval was left. The stride now
follows the frame rate (40 Hz keeps 4), and a camera too slow for any interval is skipped
with a reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from calibrex.check import estimators
from calibrex.check.estimators import CheckSkipError, camera_tracking_stride
from calibrex.data.ros_cdr_writer import encode_compressed_image, encode_image
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.evaluation.camera_imu_rotation import (
    CameraImuRunOptions,
    camera_frame_interval_s,
    tracking_stride,
)


@pytest.mark.parametrize(
    ("rate_hz", "stride"),
    [(40.0, 4), (30.0, 3), (20.0, 2), (15.0, 2), (10.0, 1), (5.0, 1)],
)
def test_the_stride_tracks_images_about_a_tenth_of_a_second_apart(
    rate_hz: float, stride: int
) -> None:
    assert tracking_stride(1.0 / rate_hz) == stride


def test_an_unknown_rate_keeps_the_hilti_stride() -> None:
    assert tracking_stride(None) == CameraImuRunOptions.frame_stride == 4


def _bag(path: Path, rate_hz: float, *, compressed: bool = False, count: int = 30) -> Path:
    topic = "/cam/image_raw/compressed" if compressed else "/cam/image_raw"
    kind = "sensor_msgs/msg/CompressedImage" if compressed else "sensor_msgs/msg/Image"
    with Rosbag2Writer(path) as writer:
        writer.add_topic(topic, kind)
        for index in range(count):
            # jittered stamps: the median ignores one dropped frame
            stamp = 10**9 + round((index + (1 if index > 15 else 0)) / rate_hz * 1e9)
            payload = (
                encode_compressed_image(
                    frame_id="cam", timestamp_ns=stamp, format="png", data=b"\x89PNG"
                )
                if compressed
                else encode_image(
                    frame_id="cam",
                    timestamp_ns=stamp,
                    height=2,
                    width=2,
                    encoding="mono8",
                    step=2,
                    data=bytes(4),
                )
            )
            writer.write(topic, stamp, payload)
    return path


@pytest.mark.parametrize("compressed", [False, True])
def test_the_frame_interval_is_read_from_the_image_stamps(tmp_path: Path, compressed: bool) -> None:
    bag = _bag(tmp_path / "bag", 10.0, compressed=compressed)
    topic = "/cam/image_raw/compressed" if compressed else "/cam/image_raw"
    assert camera_frame_interval_s(bag, topic) == pytest.approx(0.1, abs=1e-6)
    assert camera_frame_interval_s(tmp_path / "missing", topic) is None


def _context(bag: Path, topic: str) -> Any:
    return estimators.PairContext(
        bag=bag,
        pair=None,  # type: ignore[arg-type]
        topic_types={topic: "sensor_msgs/msg/Image"},
        sensor_topics={"camera": (topic,)},
        candidate=np.eye(4),
        sources=[],
        controls=estimators.RunControls(),
    )


def test_the_check_picks_the_stride_and_says_so(tmp_path: Path) -> None:
    ten = camera_tracking_stride(
        _context(_bag(tmp_path / "ten", 10.0), "/cam/image_raw"), "/cam/image_raw"
    )
    forty = camera_tracking_stride(
        _context(_bag(tmp_path / "forty", 40.0), "/cam/image_raw"), "/cam/image_raw"
    )
    assert ten == (1, "tracking every image (10.0 Hz)")
    assert forty == (4, "tracking every 4. image (40.0 Hz), about 10 Hz")


def test_a_camera_slower_than_the_solver_interval_is_skipped(tmp_path: Path) -> None:
    ctx = _context(_bag(tmp_path / "slow", 3.0), "/cam/image_raw")
    with pytest.raises(CheckSkipError) as skipped:
        camera_tracking_stride(ctx, "/cam/image_raw")
    assert skipped.value.code == "unsupported_sensor"
    assert "3.0 Hz" in str(skipped.value) and "at least 4 Hz" in str(skipped.value)
