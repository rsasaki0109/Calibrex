"""Gray images and IMU samples from ROS 2 bags, without ROS.

Used for camera-IMU calibration: images come from a ``sensor_msgs/msg/Image``
topic (``mono8``, ``mono16``, ``rgb8``, ``bgr8``, ``rgba8``, ``bgra8``) or a
``sensor_msgs/msg/CompressedImage`` topic (JPEG or PNG, see
:mod:`calibrex.data.compressed_image`) and are returned as 8-bit gray arrays with
their header time in seconds.
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
from calibrex.data.rosbag2 import iter_messages, require_cdr_connection

GrayImage: TypeAlias = NDArray[np.uint8]
IMAGE_MESSAGE_TYPE = "sensor_msgs/msg/Image"
COMPRESSED_IMAGE_MESSAGE_TYPE = "sensor_msgs/msg/CompressedImage"
_CHANNELS = {"mono8": 1, "mono16": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
# A check reads one IMU topic for every camera paired with it; the last reads are kept,
# keyed by the bag's files (name, size, modification time), the topic and the unit.
_IMU_READS: dict[tuple[object, ...], ImuSamples] = {}
_IMU_READS_KEPT = 2


def load_ros2_imu(
    bag_dir: str | Path,
    topic: str,
    *,
    acceleration_unit: Literal["mps2", "g"] = "mps2",
) -> ImuSamples:
    """Read every ``sensor_msgs/msg/Imu`` sample of a topic into SI units, time-sorted.

    A repeat read of an unchanged bag returns the samples of the last read; their arrays
    are read-only.
    """

    key = (_bag_files_key(Path(bag_dir)), topic, acceleration_unit)
    cached = _IMU_READS.get(key)
    if cached is None:
        cached = _read_ros2_imu(bag_dir, topic, acceleration_unit)
        for array in (cached.times_s, cached.gyro_rps, cached.accel_mps2):
            array.setflags(write=False)
        _IMU_READS[key] = cached
        while len(_IMU_READS) > _IMU_READS_KEPT:
            _IMU_READS.pop(next(iter(_IMU_READS)))
    return cached


def _bag_files_key(root: Path) -> tuple[object, ...]:
    files = sorted(item for item in root.iterdir() if item.is_file()) if root.is_dir() else [root]
    return (
        str(root.resolve()),
        *((item.name, item.stat().st_size, item.stat().st_mtime_ns) for item in files),
    )


def _read_ros2_imu(
    bag_dir: str | Path, topic: str, acceleration_unit: Literal["mps2", "g"]
) -> ImuSamples:
    times: list[float] = []
    gyro: list[tuple[float, float, float]] = []
    accel: list[tuple[float, float, float]] = []
    for connection, timestamp_ns, payload in iter_messages(bag_dir, topics={topic}):
        require_cdr_connection(connection)
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
    return color_to_gray(image[:, :, :3], bgr=encoding.startswith("bgr"))


def color_to_gray(color: NDArray[np.uint8], *, bgr: bool) -> GrayImage:
    """8-bit gray of an ``(H, W, 3)`` 8-bit colour image (ITU-R 601 luma weights)."""

    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if bgr:
        weights = weights[::-1]
    return np.asarray(np.clip(color.astype(np.float32) @ weights, 0, 255), dtype=np.uint8)


def image_header(
    message_type: str, topic: str, timestamp_ns: int, payload: bytes
) -> tuple[int, str]:
    """``(stamp in ns, frame id)`` of an Image or CompressedImage payload, pixels not decoded.

    The stamp is the header stamp, or the bag time when the header stamp is zero.
    """

    if message_type == COMPRESSED_IMAGE_MESSAGE_TYPE:
        compressed = ros_cdr.decode_ros2_compressed_image(
            topic, timestamp_ns, payload, include_data=False
        )
        return compressed.timestamp_ns, compressed.frame_id
    image = ros_cdr.decode_ros2_image(topic, timestamp_ns, payload, include_data=False)
    return image.timestamp_ns, image.frame_id


def decode_gray_message(
    message_type: str, topic: str, timestamp_ns: int, payload: bytes
) -> tuple[int, str, GrayImage]:
    """``(stamp in ns, frame id, 8-bit gray image)`` of an Image or CompressedImage payload."""

    if message_type == COMPRESSED_IMAGE_MESSAGE_TYPE:
        from calibrex.data.compressed_image import decode_compressed_gray

        compressed = ros_cdr.decode_ros2_compressed_image(topic, timestamp_ns, payload)
        assert compressed.data is not None
        try:
            gray = decode_compressed_gray(bytes(compressed.data), compressed.format)
        except ValueError as exc:
            raise ValueError(f"{topic} at {compressed.timestamp_ns} ns: {exc}") from exc
        return compressed.timestamp_ns, compressed.frame_id, gray
    if message_type != IMAGE_MESSAGE_TYPE:
        raise ValueError(f"{topic} is {message_type}, not an image topic")
    image = ros_cdr.decode_ros2_image(topic, timestamp_ns, payload)
    assert image.data is not None
    return (
        image.timestamp_ns,
        image.frame_id,
        to_gray(
            bytes(image.data),
            height=image.height,
            width=image.width,
            step=image.step,
            encoding=image.encoding,
        ),
    )


def iter_gray_images(
    bag_dir: str | Path,
    topic: str,
    *,
    stride: int = 1,
    max_seconds: float | None = None,
) -> Iterator[tuple[float, GrayImage]]:
    """Yield ``(header time in s, gray image)`` for every ``stride``-th image."""

    first: float | None = None
    messages = iter_messages(bag_dir, topics={topic})
    for index, (connection, timestamp_ns, payload) in enumerate(messages):
        if index % stride:
            continue
        require_cdr_connection(connection)
        stamp_ns, _frame = image_header(connection.message_type, topic, timestamp_ns, payload)
        emit_tick(index + 1)
        time_s = stamp_ns * 1.0e-9
        first = time_s if first is None else first
        if max_seconds is not None and time_s - first > max_seconds:
            return
        _stamp, _frame, gray = decode_gray_message(
            connection.message_type, topic, timestamp_ns, payload
        )
        yield time_s, gray
