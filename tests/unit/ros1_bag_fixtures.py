"""Minimal ROS 1 bag (v2.0) writer and message serializer for tests.

The serializer is generic: it walks a ``message_definition`` text (the same text a
real bag stores in each connection record) and writes the ROS 1 wire format from a
plain ``dict`` of values, defaulting everything that is not given. Nothing here
imports ROS.
"""

from __future__ import annotations

import bz2
import struct
from pathlib import Path
from typing import Any

from calibrex.data.ros1_cdr import Ros1ToCdrTranscoder

_SEP = "=" * 80 + "\n"

HEADER = "std_msgs/Header"
_HEADER_DEF = "uint32 seq\ntime stamp\nstring frame_id\n"
_TRANSFORM_DEFS = {
    "geometry_msgs/Vector3": "float64 x\nfloat64 y\nfloat64 z\n",
    "geometry_msgs/Quaternion": "float64 x\nfloat64 y\nfloat64 z\nfloat64 w\n",
    "geometry_msgs/Point": "float64 x\nfloat64 y\nfloat64 z\n",
    "geometry_msgs/Transform": (
        "geometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n"
    ),
    "geometry_msgs/Pose": "geometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n",
    "geometry_msgs/Twist": "geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular\n",
    "geometry_msgs/TransformStamped": (
        "Header header\nstring child_frame_id\ngeometry_msgs/Transform transform\n"
    ),
    "geometry_msgs/PoseWithCovariance": "geometry_msgs/Pose pose\nfloat64[36] covariance\n",
    "geometry_msgs/TwistWithCovariance": "geometry_msgs/Twist twist\nfloat64[36] covariance\n",
    "sensor_msgs/NavSatStatus": (
        "int8 STATUS_NO_FIX=-1\nint8 STATUS_FIX=0\nint8 status\nuint16 service\n"
    ),
    "sensor_msgs/PointField": (
        "uint8 INT8=1\nuint8 FLOAT32=7\nstring name\nuint32 offset\nuint8 datatype\nuint32 count\n"
    ),
    "sensor_msgs/RegionOfInterest": (
        "uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\nbool do_rectify\n"
    ),
    "livox_ros_driver/CustomPoint": (
        "uint32 offset_time\nfloat32 x\nfloat32 y\nfloat32 z\n"
        "uint8 reflectivity\nuint8 tag\nuint8 line\n"
    ),
}
_ROOT_DEFS = {
    "sensor_msgs/Imu": (
        "Header header\ngeometry_msgs/Quaternion orientation\nfloat64[9] orientation_covariance\n"
        "geometry_msgs/Vector3 angular_velocity\nfloat64[9] angular_velocity_covariance\n"
        "geometry_msgs/Vector3 linear_acceleration\nfloat64[9] linear_acceleration_covariance\n",
        ["geometry_msgs/Quaternion", "geometry_msgs/Vector3"],
    ),
    "sensor_msgs/PointCloud2": (
        "Header header\nuint32 height\nuint32 width\nsensor_msgs/PointField[] fields\n"
        "bool is_bigendian\nuint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense\n",
        ["sensor_msgs/PointField"],
    ),
    "sensor_msgs/Image": (
        "Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\n"
        "uint32 step\nuint8[] data\n",
        [],
    ),
    "sensor_msgs/CompressedImage": ("Header header\nstring format\nuint8[] data\n", []),
    "sensor_msgs/CameraInfo": (
        "Header header\nuint32 height\nuint32 width\nstring distortion_model\nfloat64[] D\n"
        "float64[9] K\nfloat64[9] R\nfloat64[12] P\nuint32 binning_x\nuint32 binning_y\n"
        "sensor_msgs/RegionOfInterest roi\n",
        ["sensor_msgs/RegionOfInterest"],
    ),
    "sensor_msgs/NavSatFix": (
        "Header header\nsensor_msgs/NavSatStatus status\nfloat64 latitude\nfloat64 longitude\n"
        "float64 altitude\nfloat64[9] position_covariance\nuint8 position_covariance_type\n",
        ["sensor_msgs/NavSatStatus"],
    ),
    "nav_msgs/Odometry": (
        "Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\n"
        "geometry_msgs/TwistWithCovariance twist\n",
        [
            "geometry_msgs/PoseWithCovariance",
            "geometry_msgs/Pose",
            "geometry_msgs/Point",
            "geometry_msgs/Quaternion",
            "geometry_msgs/TwistWithCovariance",
            "geometry_msgs/Twist",
            "geometry_msgs/Vector3",
        ],
    ),
    "geometry_msgs/TwistStamped": (
        "Header header\ngeometry_msgs/Twist twist\n",
        ["geometry_msgs/Twist", "geometry_msgs/Vector3"],
    ),
    "tf2_msgs/TFMessage": (
        "geometry_msgs/TransformStamped[] transforms\n",
        [
            "geometry_msgs/TransformStamped",
            "geometry_msgs/Transform",
            "geometry_msgs/Vector3",
            "geometry_msgs/Quaternion",
        ],
    ),
    "livox_ros_driver/CustomMsg": (
        "Header header\nuint64 timebase\nuint32 point_num\nuint8 lidar_id\nuint8[3] rsvd\n"
        "livox_ros_driver/CustomPoint[] points\n",
        ["livox_ros_driver/CustomPoint"],
    ),
}


def message_definition(ros1_type: str) -> str:
    """The full ``message_definition`` text of one supported type, as a bag stores it."""

    body, deps = _ROOT_DEFS[ros1_type]
    text = body
    for dep in [*deps, HEADER]:
        dep_body = _HEADER_DEF if dep == HEADER else _TRANSFORM_DEFS[dep]
        text += f"{_SEP}MSG: {dep}\n{dep_body}"
    return text


_FORMATS = {
    "bool": "<B",
    "int8": "<b",
    "uint8": "<B",
    "byte": "<b",
    "char": "<B",
    "int16": "<h",
    "uint16": "<H",
    "int32": "<i",
    "uint32": "<I",
    "int64": "<q",
    "uint64": "<Q",
    "float32": "<f",
    "float64": "<d",
}


def serialize(ros1_type: str, values: dict[str, Any] | None = None) -> bytes:
    """Serialize ``values`` as ``ros1_type`` in the ROS 1 wire format."""

    transcoder = Ros1ToCdrTranscoder(ros1_type, message_definition(ros1_type))
    out = bytearray()
    _write_struct(transcoder, transcoder._root, values or {}, out)
    return bytes(out)


def _write_struct(
    transcoder: Ros1ToCdrTranscoder, name: str, values: dict[str, Any], out: bytearray
) -> None:
    struct_ = transcoder._structs[name]
    for field in struct_.fields:
        value = values.get(field.name)
        if field.array is None:
            _write_value(transcoder, struct_, field.type_name, value, out)
            continue
        if field.array < 0:
            items = list(value or [])
            out += struct.pack("<I", len(items))
        else:
            items = list(value) if value is not None else [None] * field.array
            if field.type_name in _FORMATS and value is None:
                items = [0] * field.array
        if isinstance(value, bytes | bytearray) and field.type_name in ("uint8", "byte", "char"):
            out += bytes(value)
            continue
        for item in items:
            _write_value(transcoder, struct_, field.type_name, item, out)


def _write_value(
    transcoder: Ros1ToCdrTranscoder, owner: Any, type_name: str, value: Any, out: bytearray
) -> None:
    if type_name in _FORMATS:
        out += struct.pack(_FORMATS[type_name], value or 0)
    elif type_name == "string":
        raw = (value or "").encode("utf-8")
        out += struct.pack("<I", len(raw)) + raw
    elif type_name in ("time", "duration"):
        seconds, nanoseconds = value or (0, 0)
        out += struct.pack("<II", seconds, nanoseconds)
    else:
        nested = transcoder._qualify(type_name, owner.name)
        _write_struct(transcoder, nested, value or {}, out)


def _header(fields: dict[str, bytes]) -> bytes:
    body = b""
    for name, value in fields.items():
        raw = name.encode("ascii") + b"=" + value
        body += struct.pack("<I", len(raw)) + raw
    return body


def _record(fields: dict[str, bytes], data: bytes) -> bytes:
    header = _header(fields)
    return struct.pack("<I", len(header)) + header + struct.pack("<I", len(data)) + data


def _time(timestamp_ns: int) -> bytes:
    return struct.pack("<II", timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000)


class Ros1BagWriter:
    """Write a bag; ``compression`` is ``none``, ``bz2`` or ``lz4``.

    ``indexed=False`` leaves ``index_pos`` at zero and drops the index section, the
    way an interrupted recording looks. ``chunk_messages`` splits chunks so tests
    can exercise chunk skipping.
    """

    def __init__(
        self,
        path: Path,
        *,
        compression: str = "none",
        chunk_messages: int = 1000,
        indexed: bool = True,
    ) -> None:
        self._path = path
        self._compression = compression
        self._chunk_messages = chunk_messages
        self._indexed = indexed
        self._connections: dict[str, tuple[int, str, bytes]] = {}
        self._pending: list[tuple[int, int, bytes]] = []
        self._chunks: list[tuple[bytes, int, int, int, dict[int, int]]] = []
        self._index_records: list[bytes] = []

    def add_connection(self, topic: str, ros1_type: str, definition: str | None = None) -> None:
        if topic in self._connections:
            return
        text = definition if definition is not None else message_definition(ros1_type)
        data = _header(
            {
                "topic": topic.encode(),
                "type": ros1_type.encode(),
                "md5sum": b"0" * 32,
                "message_definition": text.encode(),
            }
        )
        self._connections[topic] = (len(self._connections), ros1_type, data)

    def write(self, topic: str, timestamp_ns: int, payload: bytes) -> None:
        conn_id = self._connections[topic][0]
        self._pending.append((conn_id, timestamp_ns, payload))
        if len(self._pending) >= self._chunk_messages:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        body = b""
        index: dict[int, list[tuple[int, int]]] = {}
        seen: set[int] = set()
        for conn_id, timestamp_ns, payload in self._pending:
            if conn_id not in seen:
                seen.add(conn_id)
                topic = next(t for t, c in self._connections.items() if c[0] == conn_id)
                body += _record(
                    {"op": b"\x07", "conn": struct.pack("<I", conn_id), "topic": topic.encode()},
                    self._connections[topic][2],
                )
            index.setdefault(conn_id, []).append((timestamp_ns, len(body)))
            body += _record(
                {"op": b"\x02", "conn": struct.pack("<I", conn_id), "time": _time(timestamp_ns)},
                payload,
            )
        stamps = [timestamp_ns for _, timestamp_ns, _ in self._pending]
        self._chunks.append(
            (
                body,
                min(stamps),
                max(stamps),
                0,
                {conn: len(items) for conn, items in index.items()},
            )
        )
        self._index_records.append(
            b"".join(
                _record(
                    {
                        "op": b"\x04",
                        "ver": struct.pack("<I", 1),
                        "conn": struct.pack("<I", conn_id),
                        "count": struct.pack("<I", len(items)),
                    },
                    b"".join(_time(t) + struct.pack("<I", offset) for t, offset in items),
                )
                for conn_id, items in index.items()
            )
        )
        self._pending = []

    def _compress(self, body: bytes) -> bytes:
        if self._compression == "none":
            return body
        if self._compression == "bz2":
            return bz2.compress(body)
        if self._compression == "lz4":
            import lz4.frame

            result: bytes = lz4.frame.compress(body)
            return result
        raise ValueError(self._compression)

    def close(self) -> None:
        self._flush()
        parts: list[bytes] = []
        offset = 13 + 4096
        chunk_infos: list[bytes] = []
        for number, (body, start, end, _unused, counts) in enumerate(self._chunks):
            chunk_pos = offset
            record = _record(
                {
                    "op": b"\x05",
                    "compression": self._compression.encode(),
                    "size": struct.pack("<I", len(body)),
                },
                self._compress(body),
            )
            index_records = self._index_records[number]
            parts.append(record + index_records)
            offset += len(record) + len(index_records)
            chunk_infos.append(
                _record(
                    {
                        "op": b"\x06",
                        "ver": struct.pack("<I", 1),
                        "chunk_pos": struct.pack("<Q", chunk_pos),
                        "start_time": _time(start),
                        "end_time": _time(end),
                        "count": struct.pack("<I", len(counts)),
                    },
                    b"".join(struct.pack("<II", c, n) for c, n in counts.items()),
                )
            )
        index_pos = offset if self._indexed else 0
        tail = b""
        if self._indexed:
            for topic, (conn_id, _type, data) in self._connections.items():
                tail += _record(
                    {"op": b"\x07", "conn": struct.pack("<I", conn_id), "topic": topic.encode()},
                    data,
                )
            tail += b"".join(chunk_infos)
        header = _header(
            {
                "op": b"\x03",
                "index_pos": struct.pack("<Q", index_pos),
                "conn_count": struct.pack("<I", len(self._connections)),
                "chunk_count": struct.pack("<I", len(self._chunks)),
            }
        )
        base = 4 + len(header) + 4
        pad = 4096 - base
        header_record = (
            struct.pack("<I", len(header)) + header + struct.pack("<I", pad) + b" " * pad
        )
        self._path.write_bytes(b"#ROSBAG V2.0\n" + header_record + b"".join(parts) + tail)

    def __enter__(self) -> Ros1BagWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
