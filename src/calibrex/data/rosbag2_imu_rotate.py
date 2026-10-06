"""Rotate the IMU of a rosbag2 as if the IMU had been remounted (known-bad control).

A physically remounted IMU measures the same angular velocity and specific
force expressed in rotated axes. :func:`rotate_imu_in_bag` copies a bag message
by message and, for every ``sensor_msgs/msg/Imu`` message of the chosen topics,
replaces the angular velocity and linear acceleration vectors ``v`` with
``R v``. Every other byte of every message is copied unchanged, so the only
change a calibration check can see is the IMU extrinsic. The orientation
quaternion is rotated to match (``q' = q * q_R``) when the message carries a
valid one (``orientation_covariance[0] >= 0``); covariances are left as they
are (they are zero or isotropic on the IMUs this is used with).

The point of the control is a *known* extrinsic change: remounting by ``R``
changes the IMU-LiDAR rotation estimate by the rotation ``R`` (up to the
convention of the frame the estimate is expressed in), so a drift detector
that sees the original and the rotated bag must flag it with a magnitude of
about ``|R|``.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import CdrReader
from calibrex.data.rosbag2 import iter_messages, list_rosbag2_connections
from calibrex.data.rosbag2_writer import Rosbag2Writer

IMU_MESSAGE_TYPE = "sensor_msgs/msg/Imu"


def rotation_from_axis_angle_deg(axis: Sequence[float], angle_deg: float) -> NDArray[np.float64]:
    """Rotation matrix of ``angle_deg`` degrees about ``axis`` (Rodrigues)."""

    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or norm == 0.0:
        raise ValueError("axis must be a non-zero 3-vector")
    unit = vector / norm
    theta = math.radians(angle_deg)
    skew = np.array([[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]])
    return np.asarray(
        np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew),
        dtype=float,
    )


def _quaternion_from_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit quaternion ``(x, y, z, w)`` of a rotation matrix."""

    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
                0.25 * s,
            ]
        )
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k])
        quat = np.zeros(4)
        quat[i] = 0.25 * s
        quat[j] = (matrix[j, i] + matrix[i, j]) / s
        quat[k] = (matrix[k, i] + matrix[i, k]) / s
        quat[3] = (matrix[k, j] - matrix[j, k]) / s
    return np.asarray(quat / np.linalg.norm(quat), dtype=float)


def _quaternion_multiply(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def rotate_imu_payload(data: bytes, rotation: NDArray[np.float64]) -> bytes:
    """Return the CDR payload of an ``Imu`` message with its vectors rotated by ``rotation``."""

    matrix = np.asarray(rotation, dtype=float)
    reader = CdrReader(data)
    reader.read_int32()
    reader.read_uint32()
    reader.read_string()
    reader.align(8)
    orientation_at = reader.offset
    orientation = reader.read_float64_array(4)
    covariance = reader.read_float64_array(9)
    reader.align(8)
    angular_at = reader.offset
    reader.read_float64_array(3)
    reader.read_float64_array(9)
    reader.align(8)
    acceleration_at = reader.offset
    endian = "<" if reader.little_endian else ">"
    buffer = bytearray(data)
    for at in (angular_at, acceleration_at):
        vector = np.array(struct.unpack_from(f"{endian}3d", buffer, at))
        struct.pack_into(f"{endian}3d", buffer, at, *(matrix @ vector))
    if covariance[0] >= 0.0 and any(orientation):
        rotated = _quaternion_multiply(
            np.asarray(orientation, dtype=float), _quaternion_from_matrix(matrix)
        )
        struct.pack_into(f"{endian}4d", buffer, orientation_at, *rotated)
    return bytes(buffer)


@dataclass(frozen=True)
class RotateImuSummary:
    """What :func:`rotate_imu_in_bag` wrote."""

    source: str
    destination: str
    imu_topics: tuple[str, ...]
    imu_messages_rotated: int
    messages_copied: int
    rotation_matrix: tuple[tuple[float, ...], ...]
    rotation_angle_deg: float
    max_duration_s: float | None


def rotate_imu_in_bag(
    source: str | Path,
    destination: str | Path,
    rotation: NDArray[np.float64],
    *,
    imu_topics: Sequence[str] | None = None,
    max_duration_s: float | None = None,
    overwrite: bool = False,
) -> RotateImuSummary:
    """Copy ``source`` to ``destination`` with the IMU vectors rotated by ``rotation``.

    ``imu_topics`` defaults to every ``sensor_msgs/msg/Imu`` topic. With
    ``max_duration_s`` only messages within that many seconds of the first
    message are copied (to keep a copy of a large bag small; an estimator run
    with the same ``--max-duration-s`` sees the same data).
    """

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9):
        raise ValueError("rotation must be a 3x3 rotation matrix")
    connections = [c for c, _ in list_rosbag2_connections(source)]
    imu = {c.topic for c in connections if c.message_type == IMU_MESSAGE_TYPE}
    if imu_topics is not None:
        unknown = sorted(set(imu_topics) - imu)
        if unknown:
            raise DatasetError(f"not an Imu topic of {source}: {', '.join(unknown)}")
        imu = set(imu_topics)
    if not imu:
        raise DatasetError(f"{source} has no {IMU_MESSAGE_TYPE} topic")
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(matrix)) - 1.0) / 2.0))))
    rotated = copied = 0
    first_ns: int | None = None
    with Rosbag2Writer(destination, overwrite=overwrite) as writer:
        for connection in connections:
            latched = "durability: 1" in (connection.offered_qos_profiles or "")
            writer.add_topic(connection.topic, connection.message_type, latched=latched)
        for connection, timestamp_ns, payload in iter_messages(source):
            if first_ns is None:
                first_ns = timestamp_ns
            if max_duration_s is not None and timestamp_ns - first_ns > max_duration_s * 1e9:
                break  # messages arrive in timestamp order: nothing later is in range
            if connection.topic in imu:
                payload = rotate_imu_payload(payload, matrix)
                rotated += 1
            writer.write(connection.topic, timestamp_ns, payload)
            copied += 1
    return RotateImuSummary(
        source=str(source),
        destination=str(destination),
        imu_topics=tuple(sorted(imu)),
        imu_messages_rotated=rotated,
        messages_copied=copied,
        rotation_matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        rotation_angle_deg=angle,
        max_duration_s=max_duration_s,
    )
