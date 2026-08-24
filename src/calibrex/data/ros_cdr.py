"""Minimal CDR (XCDR1) reader for ROS 2 serialized messages."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_messages import (
    MAX_CAMERA_INFO_DISTORTION_COEFFICIENTS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_ENCODING_LENGTH,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    MAX_RADAR_PAYLOAD_BYTES,
    MAX_RADAR_RETURNS,
    MAX_ROS_STRING_BYTES,
    RADAR_SCAN_ROS2_TYPE,
    CameraInfoMessage,
    ImageMessage,
    ImuMessage,
    LivoxCustomMessage,
    OdometryMessage,
    PointCloud2Message,
    PointField,
    RadarReturn,
    RadarScanMessage,
    RegionOfInterest,
    decode_point_time_offsets,
    decode_pointcloud_payload,
    filter_nonfinite_pointcloud_rows,
    require_numpy,
    validate_camera_info_metadata,
    validate_image_metadata,
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

    @property
    def remaining(self) -> int:
        """Return the number of unread serialized bytes."""

        return len(self._data) - self._offset

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

    def read_uint64(self) -> int:
        """Read an aligned uint64."""

        self.align(8)
        if self._offset + 8 > len(self._data):
            msg = "truncated CDR payload"
            raise DatasetError(msg)
        (value,) = struct.unpack_from(f"{self._endian}Q", self._data, self._offset)
        self._offset += 8
        return int(value)

    def read_float32(self) -> float:
        """Read an aligned float32."""

        self.align(4)
        if self._offset + 4 > len(self._data):
            msg = "truncated CDR payload"
            raise DatasetError(msg)
        (value,) = struct.unpack_from(f"{self._endian}f", self._data, self._offset)
        self._offset += 4
        return float(value)

    def read_float64(self) -> float:
        """Read an aligned float64."""

        self.align(8)
        if self._offset + 8 > len(self._data):
            msg = "truncated CDR payload"
            raise DatasetError(msg)
        (value,) = struct.unpack_from(f"{self._endian}d", self._data, self._offset)
        self._offset += 8
        return float(value)

    def read_string(self, *, max_length: int | None = None) -> str:
        """Read a length-prefixed UTF-8 string (length includes the NUL terminator)."""

        length = self.read_uint32()
        if length == 0:
            raise DatasetError("invalid zero-length CDR string")
        if max_length is not None and length - 1 > max_length:
            raise DatasetError(
                f"CDR string length {length - 1} exceeds safe bound {max_length}"
            )
        if self._offset + length > len(self._data):
            msg = "truncated CDR string"
            raise DatasetError(msg)
        raw = self._data[self._offset : self._offset + length]
        self._offset += length
        if raw[-1:] != b"\x00":
            raise DatasetError("CDR string is missing its NUL terminator")
        raw = raw[:-1]
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise DatasetError("invalid UTF-8 CDR string") from exc

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

    def read_bounded_byte_sequence(
        self,
        *,
        max_length: int,
        copy: bool = True,
    ) -> tuple[int, bytes | None]:
        """Read a bounded byte sequence, optionally without copying its bytes."""

        length = self.read_uint32()
        if length > max_length:
            raise DatasetError(
                f"CDR byte sequence length {length} exceeds safe bound {max_length}"
            )
        if self._offset + length > len(self._data):
            raise DatasetError("truncated CDR byte sequence")
        payload = (
            self._data[self._offset : self._offset + length] if copy else None
        )
        self._offset += length
        return int(length), payload

    def read_float64_array(self, count: int) -> tuple[float, ...]:
        """Read ``count`` float64 values with per-element alignment."""

        values: list[float] = []
        for _ in range(count):
            values.append(self.read_float64())
        return tuple(values)


def decode_ros2_radar_scan(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    max_returns: int = MAX_RADAR_RETURNS,
    max_payload_bytes: int = MAX_RADAR_PAYLOAD_BYTES,
) -> RadarScanMessage:
    """Deserialize a ROS 2 CDR ``radar_msgs/msg/RadarScan`` payload.

    The pinned upstream definition is ``Header header; RadarReturn[] returns``;
    each return is exactly five aligned ``float32`` values.  The sequence count
    and complete serialized payload are checked before iterating so hostile
    bags cannot trigger unbounded allocation or partial normalization.
    """

    if len(data) > max_payload_bytes:
        raise DatasetError(
            f"radar_msgs/msg/RadarScan payload length {len(data)} exceeds safe "
            f"bound {max_payload_bytes}"
        )
    reader = CdrReader(data)
    stamp_secs = reader.read_int32()
    stamp_nsecs = reader.read_uint32()
    if stamp_nsecs >= 1_000_000_000:
        raise DatasetError(
            "radar_msgs/msg/RadarScan header nanoseconds are outside [0, 1e9)"
        )
    frame_id = reader.read_string(max_length=MAX_ROS_STRING_BYTES)
    count = reader.read_uint32()
    if count > max_returns:
        raise DatasetError(
            f"radar_msgs/msg/RadarScan return count {count} exceeds safe bound "
            f"{max_returns}"
        )
    required_bytes = int(count) * 5 * 4
    if required_bytes > reader.remaining:
        raise DatasetError(
            "truncated radar_msgs/msg/RadarScan return sequence: "
            f"declared {count} returns need {required_bytes} bytes, "
            f"only {reader.remaining} remain"
        )
    returns = tuple(
        RadarReturn(
            range=reader.read_float32(),
            azimuth=reader.read_float32(),
            elevation=reader.read_float32(),
            doppler_velocity=reader.read_float32(),
            amplitude=reader.read_float32(),
        )
        for _ in range(int(count))
    )
    if reader.remaining:
        raise DatasetError(
            "radar_msgs/msg/RadarScan payload has "
            f"{reader.remaining} trailing byte(s)"
        )
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return RadarScanMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        returns=returns,
        source_spec=RADAR_SCAN_ROS2_TYPE,
        header_stamp_ns=header_stamp_ns,
    )


def decode_ros2_radar(
    topic: str,
    timestamp_ns: int,
    data: bytes,
) -> RadarScanMessage:
    """Compatibility alias for the ROS 2 RadarScan decoder."""

    return decode_ros2_radar_scan(topic, timestamp_ns, data)


def decode_ros2_image(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    include_data: bool = True,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT,
    max_data_bytes: int = MAX_IMAGE_BYTES,
) -> ImageMessage:
    """Deserialize and validate a ROS 2 CDR ``sensor_msgs/msg/Image`` payload."""

    reader = CdrReader(data)
    stamp_secs = reader.read_int32()
    stamp_nsecs = reader.read_uint32()
    if stamp_nsecs >= 1_000_000_000:
        raise DatasetError("sensor_msgs/msg/Image header nanoseconds are outside [0, 1e9)")
    frame_id = reader.read_string(max_length=MAX_ROS_STRING_BYTES)
    height = reader.read_uint32()
    width = reader.read_uint32()
    encoding = reader.read_string(max_length=MAX_IMAGE_ENCODING_LENGTH)
    is_bigendian = reader.read_uint8()
    step = reader.read_uint32()
    data_length, image_data = reader.read_bounded_byte_sequence(
        max_length=max_data_bytes,
        copy=include_data,
    )
    validate_image_metadata(
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=is_bigendian,
        step=step,
        data_length=data_length,
        max_width=max_width,
        max_height=max_height,
        max_data_bytes=max_data_bytes,
    )
    if reader.remaining:
        raise DatasetError(
            f"sensor_msgs/msg/Image payload has {reader.remaining} trailing byte(s)"
        )
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return ImageMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        height=int(height),
        width=int(width),
        encoding=encoding,
        is_bigendian=bool(is_bigendian),
        step=int(step),
        data_length=int(data_length),
        data=image_data,
        header_stamp_ns=header_stamp_ns,
    )


def decode_ros2_camera_info(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT,
    max_distortion_coefficients: int = MAX_CAMERA_INFO_DISTORTION_COEFFICIENTS,
) -> CameraInfoMessage:
    """Deserialize and validate a ROS 2 CDR ``sensor_msgs/msg/CameraInfo``."""

    reader = CdrReader(data)
    stamp_secs = reader.read_int32()
    stamp_nsecs = reader.read_uint32()
    if stamp_nsecs >= 1_000_000_000:
        raise DatasetError(
            "sensor_msgs/msg/CameraInfo header nanoseconds are outside [0, 1e9)"
        )
    frame_id = reader.read_string(max_length=MAX_ROS_STRING_BYTES)
    height = reader.read_uint32()
    width = reader.read_uint32()
    distortion_model = reader.read_string(max_length=MAX_ROS_STRING_BYTES)
    distortion_count = reader.read_uint32()
    if distortion_count > max_distortion_coefficients:
        raise DatasetError("sensor_msgs/msg/CameraInfo D sequence is too long")
    d = reader.read_float64_array(int(distortion_count))
    k = reader.read_float64_array(9)
    r = reader.read_float64_array(9)
    p = reader.read_float64_array(12)
    binning_x = reader.read_uint32()
    binning_y = reader.read_uint32()
    roi_x = reader.read_uint32()
    roi_y = reader.read_uint32()
    roi_height = reader.read_uint32()
    roi_width = reader.read_uint32()
    roi_do_rectify = reader.read_uint8()
    if roi_do_rectify not in (0, 1):
        raise DatasetError(
            "sensor_msgs/msg/CameraInfo ROI do_rectify must be 0 or 1"
        )
    roi = RegionOfInterest(
        x_offset=roi_x,
        y_offset=roi_y,
        height=roi_height,
        width=roi_width,
        do_rectify=bool(roi_do_rectify),
    )
    validate_camera_info_metadata(
        height=int(height),
        width=int(width),
        distortion_model=distortion_model,
        d=d,
        k=k,
        r=r,
        p=p,
        binning_x=int(binning_x),
        binning_y=int(binning_y),
        roi=roi,
        max_width=max_width,
        max_height=max_height,
        max_distortion_coefficients=max_distortion_coefficients,
    )
    if reader.remaining:
        raise DatasetError(
            f"sensor_msgs/msg/CameraInfo payload has {reader.remaining} trailing byte(s)"
        )
    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    return CameraInfoMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        height=int(height),
        width=int(width),
        distortion_model=distortion_model,
        d=d,
        k=k,
        r=r,
        p=p,
        binning_x=int(binning_x),
        binning_y=int(binning_y),
        roi=roi,
        header_stamp_ns=header_stamp_ns,
    )


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
    raw_point_count = int(xyz.shape[0])
    xyz, intensity, point_time_offsets_s, nonfinite_xyz_count = (
        filter_nonfinite_pointcloud_rows(
            numpy_module, xyz, intensity, point_time_offsets_s
        )
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
        raw_point_count=raw_point_count,
        nonfinite_xyz_count=nonfinite_xyz_count,
    )


def decode_ros2_livox_custommsg(
    topic: str,
    timestamp_ns: int,
    data: bytes,
) -> LivoxCustomMessage:
    """Decode a ROS 2 Livox ``CustomMsg`` payload.

    ``CustomPoint`` has 19 data bytes, but XCDR1 aligns each sequence element
    to its four-byte maximum alignment, so consecutive elements start 20
    bytes apart.  The final element has no required trailing padding.  The
    decoder validates the declared sequence before creating a numpy view;
    truncated custom messages are rejected rather than silently shortened.
    """

    numpy_module = require_numpy(extra_name="rosbag2")
    reader = CdrReader(data)
    stamp_secs = reader.read_int32()
    stamp_nsecs = reader.read_uint32()
    frame_id = reader.read_string()
    timebase_ns = reader.read_uint64()
    point_num = reader.read_uint32()
    lidar_id = reader.read_uint8()
    for _ in range(3):
        reader.read_uint8()
    array_len = reader.read_uint32()
    point_count = int(array_len)
    point_offset = reader.offset
    point_step = 20
    expected_end = (
        point_offset + (point_count - 1) * point_step + 19 if point_count else point_offset
    )
    if expected_end > len(data):
        msg = (
            "truncated livox_interfaces/msg/CustomMsg point sequence: "
            f"declared {point_count} points needs {point_count * point_step} bytes, "
            f"only {max(0, len(data) - point_offset)} remain"
        )
        raise DatasetError(msg)

    header_stamp_ns = int(stamp_secs) * 1_000_000_000 + int(stamp_nsecs)
    if point_count == 0:
        empty = numpy_module.empty((0, 3), dtype=numpy_module.float64)
        return LivoxCustomMessage(
            topic=topic,
            timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
            frame_id=frame_id,
            timebase_ns=int(timebase_ns),
            point_num=int(point_num),
            lidar_id=int(lidar_id),
            xyz=empty,
            intensity=None,
            offset_time_ns=None,
            line=None,
        )

    byteorder = "<" if reader.little_endian else ">"
    point_dtype = numpy_module.dtype(
        {
            "names": [
                "offset_time",
                "x",
                "y",
                "z",
                "reflectivity",
                "tag",
                "line",
            ],
            "formats": [
                f"{byteorder}u4",
                f"{byteorder}f4",
                f"{byteorder}f4",
                f"{byteorder}f4",
                "u1",
                "u1",
                "u1",
            ],
            "offsets": [0, 4, 8, 12, 16, 17, 18],
            "itemsize": point_step,
        }
    )
    point_payload = data[point_offset:expected_end]
    if len(point_payload) < point_count * point_step:
        point_payload += b"\x00"
    structured = numpy_module.frombuffer(
        point_payload,
        dtype=point_dtype,
        count=point_count,
    )
    xyz = numpy_module.stack(
        [
            structured["x"].astype(numpy_module.float64),
            structured["y"].astype(numpy_module.float64),
            structured["z"].astype(numpy_module.float64),
        ],
        axis=1,
    )
    return LivoxCustomMessage(
        topic=topic,
        timestamp_ns=header_stamp_ns if header_stamp_ns else timestamp_ns,
        frame_id=frame_id,
        timebase_ns=int(timebase_ns),
        point_num=int(point_num),
        lidar_id=int(lidar_id),
        xyz=xyz,
        intensity=structured["reflectivity"].astype(numpy_module.float64),
        offset_time_ns=structured["offset_time"].astype(numpy_module.int64),
        line=structured["line"].astype(numpy_module.int64),
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
