"""ROS 2 CDR (XCDR1) encoders for the message types ``calibrex`` writes into bags.

The counterpart of :mod:`calibrex.data.ros_cdr`: byte-for-byte encoders for
``sensor_msgs/PointCloud2``, ``Imu``, ``NavSatFix``, ``Image``, ``CameraInfo``,
``nav_msgs/Odometry``, ``geometry_msgs/TwistStamped`` and
``TwistWithCovarianceStamped`` and ``tf2_msgs/TFMessage``. They need no ROS install,
and the decoders in ``ros_cdr`` read them back. Little-endian is the default (what ROS 2 emits);
big-endian is supported for tests of the reader.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

Transform = tuple[str, str, tuple[float, float, float], tuple[float, float, float, float]]
"""``(parent_frame, child_frame, translation, quaternion_xyzw)`` of a TFMessage entry."""

# sensor_msgs/PointField datatypes
POINT_FIELD_UINT8 = 2
POINT_FIELD_FLOAT32 = 7
POINT_FIELD_FLOAT64 = 8

PointFieldSpec = tuple[str, int, int, int]
"""``(name, byte_offset, datatype, count)`` of one PointCloud2 field."""


class CdrWriter:
    """Minimal ROS 2 CDR (XCDR1) encoder."""

    def __init__(self, *, little_endian: bool = True) -> None:
        endian_byte = 1 if little_endian else 0
        self._buf = bytearray([0, endian_byte, 0, 0])
        self._little = little_endian
        self._endian = "<" if little_endian else ">"

    def align(self, alignment: int) -> None:
        if alignment <= 1:
            return
        relative_offset = len(self._buf) - 4
        padding = (-relative_offset) % alignment
        self._buf.extend(b"\x00" * padding)

    def write_int8(self, value: int) -> None:
        self.align(1)
        self._buf.extend(struct.pack(f"{self._endian}b", value))

    def write_uint16(self, value: int) -> None:
        self.align(2)
        self._buf.extend(struct.pack(f"{self._endian}H", value))

    def write_int32(self, value: int) -> None:
        self.align(4)
        self._buf.extend(struct.pack(f"{self._endian}i", value))

    def write_uint32(self, value: int) -> None:
        self.align(4)
        self._buf.extend(struct.pack(f"{self._endian}I", value))

    def write_uint64(self, value: int) -> None:
        self.align(8)
        self._buf.extend(struct.pack(f"{self._endian}Q", value))

    def write_uint8(self, value: int) -> None:
        self.align(1)
        self._buf.append(value & 0xFF)

    def write_float32(self, value: float) -> None:
        self.align(4)
        self._buf.extend(struct.pack(f"{self._endian}f", value))

    def write_float64(self, value: float) -> None:
        self.align(8)
        self._buf.extend(struct.pack(f"{self._endian}d", value))

    def write_bool(self, value: bool) -> None:
        self.write_uint8(1 if value else 0)

    def write_string(self, value: str) -> None:
        encoded = value.encode("utf-8") + b"\x00"
        self.write_uint32(len(encoded))
        self._buf.extend(encoded)

    def write_byte_sequence(self, value: bytes) -> None:
        self.write_uint32(len(value))
        self._buf.extend(value)

    def write_float64_array(self, values: Sequence[float]) -> None:
        for item in values:
            self.write_float64(item)

    def finish(self) -> bytes:
        return bytes(self._buf)


def _write_header(writer: CdrWriter, frame_id: str, timestamp_ns: int) -> None:
    writer.write_int32(timestamp_ns // 1_000_000_000)
    writer.write_uint32(timestamp_ns % 1_000_000_000)
    writer.write_string(frame_id)


def encode_header_only(frame_id: str, *, secs: int = 1, nsecs: int = 2) -> bytes:
    """A stamped-message prefix; enough for header frame id reads."""

    writer = CdrWriter()
    writer.write_int32(secs)
    writer.write_uint32(nsecs)
    writer.write_string(frame_id)
    writer.write_uint32(0)
    return writer.finish()


def encode_tf_message(
    transforms: Sequence[Transform],
    *,
    secs: int = 5,
    nsecs: int = 7,
    little_endian: bool = True,
) -> bytes:
    """CDR-encode a ``tf2_msgs/msg/TFMessage`` (parent, child, translation, xyzw)."""

    writer = CdrWriter(little_endian=little_endian)
    writer.write_uint32(len(transforms))
    for parent, child, translation, quaternion in transforms:
        writer.write_int32(secs)
        writer.write_uint32(nsecs)
        writer.write_string(parent)
        writer.write_string(child)
        for value in translation:
            writer.write_float64(value)
        for value in quaternion:
            writer.write_float64(value)
    return writer.finish()


def encode_navsatfix(
    *,
    frame_id: str = "gps",
    secs: int = 9,
    nsecs: int = 3,
    status: int = 0,
    service: int = 1,
    latitude: float = 35.5,
    longitude: float = 139.25,
    altitude: float = 42.0,
    covariance: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0),
    covariance_type: int = 2,
    little_endian: bool = True,
) -> bytes:
    """CDR-encode a ``sensor_msgs/msg/NavSatFix``."""

    writer = CdrWriter(little_endian=little_endian)
    writer.write_int32(secs)
    writer.write_uint32(nsecs)
    writer.write_string(frame_id)
    writer.write_int8(status)
    writer.write_uint16(service)
    writer.write_float64(latitude)
    writer.write_float64(longitude)
    writer.write_float64(altitude)
    writer.write_float64_array(covariance)
    writer.write_uint8(covariance_type)
    return writer.finish()


def encode_imu(
    *,
    frame_id: str,
    timestamp_ns: int,
    angular_velocity: tuple[float, float, float],
    linear_acceleration: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    orientation_known: bool = False,
) -> bytes:
    """CDR-encode a ``sensor_msgs/msg/Imu``.

    Without a known orientation, ``orientation_covariance[0]`` is ``-1`` (the ROS
    convention for "no orientation estimate").
    """

    writer = CdrWriter()
    _write_header(writer, frame_id, timestamp_ns)
    for value in orientation_xyzw:
        writer.write_float64(value)
    cov = [0.0] * 9
    if not orientation_known:
        cov[0] = -1.0
    writer.write_float64_array(cov)
    for value in angular_velocity:
        writer.write_float64(value)
    writer.write_float64_array([0.0] * 9)
    for value in linear_acceleration:
        writer.write_float64(value)
    writer.write_float64_array([0.0] * 9)
    return writer.finish()


def encode_odometry(
    *,
    frame_id: str,
    child_frame_id: str,
    timestamp_ns: int,
    position: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pose_covariance: Sequence[float] | None = None,
    twist_covariance: Sequence[float] | None = None,
) -> bytes:
    """CDR-encode a ``nav_msgs/msg/Odometry`` (pose in ``frame_id``, twist in the child frame)."""

    writer = CdrWriter()
    _write_header(writer, frame_id, timestamp_ns)
    writer.write_string(child_frame_id)
    for value in position:
        writer.write_float64(value)
    for value in orientation_xyzw:
        writer.write_float64(value)
    writer.write_float64_array(pose_covariance if pose_covariance is not None else [0.0] * 36)
    for value in linear_velocity:
        writer.write_float64(value)
    for value in angular_velocity:
        writer.write_float64(value)
    writer.write_float64_array(twist_covariance if twist_covariance is not None else [0.0] * 36)
    return writer.finish()


def encode_twist_stamped(
    *,
    frame_id: str,
    timestamp_ns: int,
    linear: tuple[float, float, float],
    angular: tuple[float, float, float],
    covariance: Sequence[float] | None = None,
) -> bytes:
    """CDR-encode a ``geometry_msgs/msg/TwistStamped``.

    With ``covariance`` the message is a ``TwistWithCovarianceStamped`` instead.
    """

    writer = CdrWriter()
    _write_header(writer, frame_id, timestamp_ns)
    for value in linear:
        writer.write_float64(value)
    for value in angular:
        writer.write_float64(value)
    if covariance is not None:
        writer.write_float64_array(covariance)
    return writer.finish()


def encode_pointcloud2(
    *,
    frame_id: str,
    timestamp_ns: int,
    fields: Sequence[PointFieldSpec],
    point_step: int,
    data: bytes,
    height: int = 1,
    is_dense: bool = True,
) -> bytes:
    """CDR-encode a little-endian ``sensor_msgs/msg/PointCloud2`` from packed point bytes."""

    if point_step <= 0 or len(data) % point_step:
        raise ValueError("point data length must be a multiple of point_step")
    count = len(data) // point_step
    if height <= 0 or count % height:
        raise ValueError("the point count must be a multiple of height")
    width = count // height
    writer = CdrWriter()
    _write_header(writer, frame_id, timestamp_ns)
    writer.write_uint32(height)
    writer.write_uint32(width)
    writer.write_uint32(len(fields))
    for name, offset, datatype, field_count in fields:
        writer.write_string(name)
        writer.write_uint32(offset)
        writer.write_uint8(datatype)
        writer.align(4)
        writer.write_uint32(field_count)
    writer.write_bool(False)
    writer.align(4)
    writer.write_uint32(point_step)
    writer.write_uint32(point_step * width)
    writer.write_byte_sequence(data)
    writer.write_bool(is_dense)
    return writer.finish()


def encode_image(
    *,
    frame_id: str,
    timestamp_ns: int,
    height: int,
    width: int,
    encoding: str,
    step: int,
    data: bytes,
    is_bigendian: bool = False,
) -> bytes:
    """CDR-encode a ``sensor_msgs/msg/Image`` from packed pixel bytes."""

    if len(data) != step * height:
        raise ValueError("image data length must equal step * height")
    writer = CdrWriter()
    _write_header(writer, frame_id, timestamp_ns)
    writer.write_uint32(height)
    writer.write_uint32(width)
    writer.write_string(encoding)
    writer.write_bool(is_bigendian)
    writer.write_uint32(step)
    writer.write_byte_sequence(data)
    return writer.finish()


def encode_camera_info(
    *,
    frame_id: str,
    timestamp_ns: int,
    height: int,
    width: int,
    k: Sequence[float],
    distortion_model: str = "plumb_bob",
    d: Sequence[float] = (),
    r: Sequence[float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    p: Sequence[float] | None = None,
) -> bytes:
    """CDR-encode a ``sensor_msgs/msg/CameraInfo`` (no binning, empty ROI).

    ``p`` defaults to the rectified projection ``[K | 0]`` of ``k``.
    """

    if len(k) != 9 or len(r) != 9:
        raise ValueError("k and r must have 9 values")
    projection = (
        list(p)
        if p is not None
        else [k[0], k[1], k[2], 0.0, k[3], k[4], k[5], 0.0, k[6], k[7], k[8], 0.0]
    )
    if len(projection) != 12:
        raise ValueError("p must have 12 values")
    writer = CdrWriter()
    _write_header(writer, frame_id, timestamp_ns)
    writer.write_uint32(height)
    writer.write_uint32(width)
    writer.write_string(distortion_model)
    writer.write_uint32(len(d))
    writer.write_float64_array(d)
    writer.write_float64_array(k)
    writer.write_float64_array(r)
    writer.write_float64_array(projection)
    for _ in range(6):  # binning_x, binning_y, roi x_offset, y_offset, height, width
        writer.write_uint32(0)
    writer.write_bool(False)
    return writer.finish()
