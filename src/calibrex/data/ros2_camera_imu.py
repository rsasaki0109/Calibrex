"""Gray images and IMU samples from ROS 2 bags, without ROS.

Used for camera-IMU calibration: images come from a ``sensor_msgs/msg/Image``
topic (``mono8``, ``mono16``, ``rgb8``, ``bgr8``, ``rgba8``, ``bgra8``) and
are returned as 8-bit gray arrays with their header time in seconds.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.progress import emit_tick
from calibrex.data import ros_cdr
from calibrex.data.livox_ros2 import STANDARD_GRAVITY_MPS2, ImuSamples
from calibrex.data.rosbag2 import iter_messages

GrayImage: TypeAlias = NDArray[np.uint8]
_CHANNELS = {"mono8": 1, "mono16": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}


def load_ros2_imu(
    bag_dir: str | Path,
    topic: str,
    *,
    acceleration_unit: Literal["mps2", "g"] = "mps2",
) -> ImuSamples:
    """Read every ``sensor_msgs/msg/Imu`` sample of a topic into SI units, time-sorted."""

    times: list[float] = []
    gyro: list[tuple[float, float, float]] = []
    accel: list[tuple[float, float, float]] = []
    for _, timestamp_ns, payload in iter_messages(bag_dir, topics={topic}):
        message = ros_cdr.decode_ros2_imu(topic, timestamp_ns, payload)
        times.append(message.timestamp_ns * 1.0e-9)
        gyro.append(message.angular_velocity)
        accel.append(message.linear_acceleration)
    if len(times) < 2:
        raise ValueError(f"{bag_dir} has fewer than two IMU samples on {topic}")
    order = np.argsort(times, kind="stable")
    scale = STANDARD_GRAVITY_MPS2 if acceleration_unit == "g" else 1.0
    stamps = np.asarray(times, dtype=np.float64)[order]
    keep = np.r_[True, np.diff(stamps) > 0.0]
    return ImuSamples(
        times_s=stamps[keep],
        gyro_rps=np.asarray(gyro, dtype=np.float64)[order][keep],
        accel_mps2=np.asarray(accel, dtype=np.float64)[order][keep] * scale,
    )


def to_gray(data: bytes, *, height: int, width: int, step: int, encoding: str) -> GrayImage:
    """Convert raw ``sensor_msgs/Image`` bytes to an 8-bit gray image."""

    if encoding not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    channels = _CHANNELS[encoding]
    depth = 2 if encoding == "mono16" else 1
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    pixels = rows[:, : width * channels * depth]
    if encoding == "mono16":
        return np.asarray(pixels.view("<u2").reshape(height, width) >> 8, dtype=np.uint8)
    image = pixels.reshape(height, width, channels)
    if channels == 1:
        return np.ascontiguousarray(image[:, :, 0])
    color = image[:, :, :3].astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if encoding.startswith("bgr"):
        weights = weights[::-1]
    return np.asarray(np.clip(color @ weights, 0, 255), dtype=np.uint8)


def iter_gray_images(
    bag_dir: str | Path,
    topic: str,
    *,
    stride: int = 1,
    max_seconds: float | None = None,
) -> Iterator[tuple[float, GrayImage]]:
    """Yield ``(header time in s, gray image)`` for every ``stride``-th image."""

    first: float | None = None
    for index, (_, timestamp_ns, payload) in enumerate(iter_messages(bag_dir, topics={topic})):
        if index % stride:
            continue
        message = ros_cdr.decode_ros2_image(topic, timestamp_ns, payload)
        emit_tick(index + 1)
        time_s = message.timestamp_ns * 1.0e-9
        first = time_s if first is None else first
        if max_seconds is not None and time_s - first > max_seconds:
            return
        assert message.data is not None
        yield time_s, to_gray(
            bytes(message.data),
            height=message.height,
            width=message.width,
            step=message.step,
            encoding=message.encoding,
        )
