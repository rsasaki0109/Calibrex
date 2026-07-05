"""Minimal CDR (XCDR1) reader for ROS 2 serialized messages."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from slac.core.exceptions import DatasetError
from slac.data.ros_messages import (
    ImuMessage,
    OdometryMessage,
    PointCloud2Message,
    PointField,
    decode_point_time_offsets,
    decode_pointcloud_payload,
    require_numpy,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class CdrReader:
    """Cursor over a CDR-encoded ROS 2 message payload.

    XCDR1 uses a 4-byte encapsulation header: byte 0 is the encapsulation
    identifier (``0x00`` for CDR / PL_CDR), byte 1 selects endianness
    (``0``/``2`` big-endian, ``1``/``3`` little-endian), and bytes 2-3 are
    options. ROS 2 middleware typically emits ``00 01 00 00`` for
    little-endian CDR.
    """

    def __init__(self, data: bytes) -> None:
        if len(data) < 4:
            msg = "truncated CDR payload (missing encapsulation header)"
            raise DatasetError(msg)
        header = data[:4]
        if header[0] != 0:
            hex_header = " ".join(f"{byte:02x}" for byte in header)
            msg = f"unsupported CDR encapsulation header {hex_header}"
            raise DatasetError(msg)
        endian_byte = header[1]
        if endian_byte in (1, 3):
            self._little_endian = True
        elif endian_byte in (0, 2):
            self._little_endian = False
        else:
            hex_header = " ".join(f"{byte:02x}" for byte in header)
            msg = f"unsupported CDR encapsulation header {hex_header}"
            raise DatasetError(msg)
        self._endian = "<" if self._little_endian else ">"
        self._data = data
        self._offset = 4

    @property
    def little_endian(self) -> bool:
        """Return whether the payload uses little-endian byte order."""

        return self._little_endian

    @property
    def offset(self) -> int:
        """Return the current read cursor."""

        return self._offset

    def align(self, alignment: int) -> None:
        """Advance the cursor to the next ``alignment``-byte boundary.

        XCDR1 alignment is relative to the first payload byte after the
        4-byte encapsulation header (absolute offset 4).
        """

        if alignment <= 1:
            return
        relative_offset = self._offset - 4
        padding = (-relative_offset) % alignment
        self._offset += padding

    def read_bool(self) -> bool:
        """Read a CDR boolean (uint8)."""

        value = self.read_uint8()
        return bool(value)

    def read_uint8(self) -> int:
        """Read an aligned uint8."""

        self.align(1)
        if self._offset >= len(self._data):
            msg = "truncated CDR payload"
            raise DatasetError(msg)
        value = self._data[self._offset]
        self._offset += 1
        return int(value)

    def read_int32(self) -> int:
        """Read an aligned int32."""

        self.align(4)
        if self._offset + 4 > len(self._data):
            msg = "truncated CDR payload"
            raise DatasetError(msg)
        (value,) = struct.unpack_from(f"{self._endian}i", self._data, self._offset)
        self._offset += 4
        return int(value)

    def read_uint32(self) -> int:
        """Read an aligned uint32."""

        self.align(4)
        if self._offset + 4 > len(self._data):
            msg = "truncated CDR payload"
            raise DatasetError(msg)
        (value,) = struct.unpack_from(f"{self._endian}I", self._data, self._offset)
        self._offset += 4
        return int(value)

    def read_float64(self) -> float:
        """Read an aligned float64."""

        self.align(8)
        if self._offset + 8 > len(self._data):
            msg = "truncated CDR payload"
            raise DatasetError(msg)
        (value,) = struct.unpack_from(f"{self._endian}d", self._data, self._offset)
        self._offset += 8
        return float(value)

    def read_string(self) -> str:
        """Read a length-prefixed UTF-8 string (length includes the NUL terminator)."""

        length = self.read_uint32()
        if length == 0:
            return ""
        if self._offset + length > len(self._data):
            msg = "truncated CDR string"
            raise DatasetError(msg)
        raw = self._data[self._offset : self._offset + length]
        self._offset += length
        if raw[-1:] == b"\x00":
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")

    def read_byte_sequence(self) -> bytes:
        """Read a length-prefixed byte sequence."""

        length = self.read_uint32()
        if length == 0:
            return b""
        if self._offset + length > len(self._data):
            msg = "truncated CDR byte sequence"
            raise DatasetError(msg)
        payload = self._data[self._offset : self._offset + length]
        self._offset += length
        return payload

    def read_float64_array(self, count: int) -> tuple[float, ...]:
        """Read ``count`` float64 values with per-element alignment."""

        values: list[float] = []
        for _ in range(count):
            values.append(self.read_float64())
        return tuple(values)


def decode_ros2_pointcloud2(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    point_time_field: str | None = None,
) -> PointCloud2Message:
    """Decode a CDR-encoded ``sensor_msgs/msg/PointCloud2`` message."""

    numpy_module = require_numpy(extra_name="rosbag2")
    reader = CdrReader(data)
    stamp_secs = reader.read_int32()
    stamp_nsecs = reader.read_uint32()
    frame_id = reader.read_string()
    height = reader.read_uint32()
    width = reader.read_uint32()

    field_count = reader.read_uint32()
    fields: list[PointField] = []
    for _ in range(field_count):
        name = reader.read_string()
        field_offset = reader.read_uint32()
        datatype = reader.read_uint8()
        reader.align(4)
        count = reader.read_uint32()
        fields.append(
            PointField(name=name, offset=field_offset, datatype=datatype, count=count)
        )

    is_bigendian = reader.read_bool()
    reader.align(4)
    point_step = reader.read_uint32()
    _row_step = reader.read_uint32()
    payload = reader.read_byte_sequence()
    _is_dense = reader.read_bool()

    if height * width:
        point_count = height * width
    elif point_step:
        point_count = len(payload) // point_step
    else:
        point_count = 0
    xyz, intensity = decode_pointcloud_payload(
        numpy_module,
        payload=payload,
        fields=fields,
        point_step=point_step,
        point_count=point_count,
        is_bigendian=is_bigendian,
    )
    point_time_offsets_s: Any = None
    if point_time_field is not None:
        point_time_offsets_s = decode_point_time_offsets(
            numpy_module,
            payload=payload,
            fields=fields,
            point_step=point_step,
            point_count=point_count,
            is_bigendian=is_bigendian,
            field_name=point_time_field,
            topic=topic,
        )
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return PointCloud2Message(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        width=width,
        height=height,
        point_step=point_step,
        fields=tuple(fields),
        xyz=xyz,
        intensity=intensity,
        point_time_offsets_s=point_time_offsets_s,
    )


def decode_ros2_imu(topic: str, timestamp_ns: int, data: bytes) -> ImuMessage:
    """Decode a CDR-encoded ``sensor_msgs/msg/Imu`` message."""

    reader = CdrReader(data)
    stamp_secs = reader.read_int32()
    stamp_nsecs = reader.read_uint32()
    frame_id = reader.read_string()

    ori_x = reader.read_float64()
    ori_y = reader.read_float64()
    ori_z = reader.read_float64()
    ori_w = reader.read_float64()
    orientation_covariance = reader.read_float64_array(9)

    ang_x = reader.read_float64()
    ang_y = reader.read_float64()
    ang_z = reader.read_float64()
    angular_velocity_covariance = reader.read_float64_array(9)

    acc_x = reader.read_float64()
    acc_y = reader.read_float64()
    acc_z = reader.read_float64()
    linear_acceleration_covariance = reader.read_float64_array(9)

    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return ImuMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        orientation_xyzw=(ori_x, ori_y, ori_z, ori_w),
        orientation_covariance=orientation_covariance,
        angular_velocity=(ang_x, ang_y, ang_z),
        angular_velocity_covariance=angular_velocity_covariance,
        linear_acceleration=(acc_x, acc_y, acc_z),
        linear_acceleration_covariance=linear_acceleration_covariance,
    )


def decode_ros2_odometry(topic: str, timestamp_ns: int, data: bytes) -> OdometryMessage:
    """Decode a CDR-encoded ``nav_msgs/msg/Odometry`` message."""

    reader = CdrReader(data)
    stamp_secs = reader.read_int32()
    stamp_nsecs = reader.read_uint32()
    frame_id = reader.read_string()
    child_frame_id = reader.read_string()

    pos_x = reader.read_float64()
    pos_y = reader.read_float64()
    pos_z = reader.read_float64()
    ori_x = reader.read_float64()
    ori_y = reader.read_float64()
    ori_z = reader.read_float64()
    ori_w = reader.read_float64()
    pose_covariance = reader.read_float64_array(36)

    lin_x = reader.read_float64()
    lin_y = reader.read_float64()
    lin_z = reader.read_float64()
    ang_x = reader.read_float64()
    ang_y = reader.read_float64()
    ang_z = reader.read_float64()
    twist_covariance = reader.read_float64_array(36)

    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return OdometryMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        child_frame_id=child_frame_id,
        position=(pos_x, pos_y, pos_z),
        orientation_xyzw=(ori_x, ori_y, ori_z, ori_w),
        linear_velocity=(lin_x, lin_y, lin_z),
        angular_velocity=(ang_x, ang_y, ang_z),
        pose_covariance=pose_covariance,
        twist_covariance=twist_covariance,
    )
