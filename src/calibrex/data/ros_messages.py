"""Shared ROS message decoding helpers for bag readers.

Point-cloud field layout and numpy decoding are shared between the ROS 1 and
ROS 2 bag readers. Numpy is imported lazily so mypy CI (without numpy) stays
clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from calibrex.core.exceptions import DatasetError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# sensor_msgs/PointField datatype enum -> numpy format string (little endian base).
POINTFIELD_NUMPY = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}


@dataclass(frozen=True)
class PointField:
    """A single ``sensor_msgs/PointField`` descriptor."""

    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class PointCloud2Message:
    """A decoded ``sensor_msgs/PointCloud2`` message.

    ``xyz`` is an ``(N, 3)`` float64 numpy array; ``intensity`` is an ``(N,)``
    float64 array when the cloud carries an ``intensity`` field, else ``None``.
    ``point_time_offsets_s`` holds per-point capture offsets relative to the
    message/header stamp in seconds when requested at decode time.
    """

    topic: str
    timestamp_ns: int
    frame_id: str
    width: int
    height: int
    point_step: int
    fields: tuple[PointField, ...]
    xyz: np.ndarray
    intensity: np.ndarray | None
    point_time_offsets_s: np.ndarray | None = None
    raw_point_count: int | None = None
    nonfinite_xyz_count: int = 0

    @property
    def point_count(self) -> int:
        """Return the number of decoded points."""

        return int(self.xyz.shape[0])


@dataclass(frozen=True)
class LivoxCustomMessage:
    """A decoded Livox ``CustomMsg`` LiDAR payload.

    The normalized representation is shared by the ROS 1 and ROS 2 adapters.
    ``offset_time_ns`` is the raw Livox per-point offset from ``timebase``;
    callers must explicitly map that sensor clock into their recording clock
    before using it for deskew.
    """

    topic: str
    timestamp_ns: int
    frame_id: str
    timebase_ns: int
    point_num: int
    lidar_id: int
    xyz: np.ndarray
    intensity: np.ndarray | None
    offset_time_ns: np.ndarray | None
    line: np.ndarray | None

    @property
    def point_count(self) -> int:
        """Return the number of decoded points."""

        return int(self.xyz.shape[0])

    @property
    def point_time_reference_ns(self) -> int:
        """Return Livox's point-time base, with a header fallback."""

        return self.timebase_ns if self.timebase_ns else self.timestamp_ns


@dataclass(frozen=True)
class OdometryMessage:
    """A decoded ``nav_msgs/msg/Odometry`` message."""

    topic: str
    timestamp_ns: int
    frame_id: str
    child_frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    pose_covariance: tuple[float, ...]
    twist_covariance: tuple[float, ...]


@dataclass(frozen=True)
class ImuMessage:
    """A decoded ``sensor_msgs/msg/Imu`` message.

    Covariance arrays are stored in full row-major 3x3 layout (9 float64 values
    each) as emitted by ROS 2 CDR; index ``0``, ``4``, and ``8`` are the
  diagonal entries for orientation, angular velocity, and linear acceleration.
    """

    topic: str
    timestamp_ns: int
    frame_id: str
    orientation_xyzw: tuple[float, float, float, float]
    orientation_covariance: tuple[float, ...]
    angular_velocity: tuple[float, float, float]
    angular_velocity_covariance: tuple[float, ...]
    linear_acceleration: tuple[float, float, float]
    linear_acceleration_covariance: tuple[float, ...]


def require_numpy(*, extra_name: str) -> Any:
    """Return the numpy module or raise a clear ``DatasetError``."""

    try:
        import numpy as numpy_module
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        msg = f"decoding ROS bag messages requires numpy; install slac[{extra_name}]"
        raise DatasetError(msg) from exc
    return numpy_module


def filter_nonfinite_pointcloud_rows(
    np_mod: Any,
    xyz: np.ndarray,
    intensity: np.ndarray | None,
    point_time_offsets_s: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    """Drop PointCloud2 rows whose xyz coordinates are not finite.

    Organized ROS point clouds commonly use NaN rows for missing returns.  The
    calibration core requires finite coordinates, so the adapter removes those
    rows while keeping optional intensity and per-point time arrays aligned.
    The returned count preserves an audit trail for the discarded rows.
    """

    valid = np_mod.isfinite(xyz).all(axis=1)
    nonfinite_count = int(np_mod.count_nonzero(~valid))
    filtered_xyz = xyz[valid]
    filtered_intensity = intensity[valid] if intensity is not None else None
    filtered_point_time = (
        point_time_offsets_s[valid] if point_time_offsets_s is not None else None
    )
    return filtered_xyz, filtered_intensity, filtered_point_time, nonfinite_count


def decode_pointcloud_payload(
    np_mod: Any,
    *,
    payload: bytes,
    fields: list[PointField],
    point_step: int,
    point_count: int,
    is_bigendian: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode xyz (+ optional intensity) arrays from a PointCloud2 data payload."""

    byteorder = ">" if is_bigendian else "<"
    by_name = {field.name: field for field in fields}
    required = ("x", "y", "z")
    if not all(name in by_name for name in required):
        msg = "PointCloud2 message does not carry x/y/z fields"
        raise DatasetError(msg)

    wanted = [*required]
    if "intensity" in by_name:
        wanted.append("intensity")

    names: list[str] = []
    formats: list[str] = []
    offsets: list[int] = []
    for name in wanted:
        point_field = by_name[name]
        base = POINTFIELD_NUMPY.get(point_field.datatype)
        if base is None:
            msg = f"unsupported PointField datatype {point_field.datatype} for field {name!r}"
            raise DatasetError(msg)
        names.append(name)
        formats.append(f"{byteorder}{base}")
        offsets.append(point_field.offset)

    dtype = np_mod.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step}
    )
    usable = min(point_count, len(payload) // point_step) if point_step else 0
    structured = np_mod.frombuffer(payload, dtype=dtype, count=usable)
    xyz = np_mod.stack(
        [
            structured["x"].astype(np_mod.float64),
            structured["y"].astype(np_mod.float64),
            structured["z"].astype(np_mod.float64),
        ],
        axis=1,
    )
    intensity = (
        structured["intensity"].astype(np_mod.float64) if "intensity" in names else None
    )
    return xyz, intensity


_POINT_TIME_SECONDS_DATATYPES = frozenset({7, 8})  # FLOAT32, FLOAT64
_POINT_TIME_NANOSECONDS_DATATYPES = frozenset({5, 6})  # INT32, UINT32


def decode_point_time_offsets(
    np_mod: Any,
    *,
    payload: bytes,
    fields: list[PointField],
    point_step: int,
    point_count: int,
    is_bigendian: bool,
    field_name: str,
    topic: str,
) -> np.ndarray:
    """Decode per-point capture-time offsets in seconds relative to the header stamp."""

    byteorder = ">" if is_bigendian else "<"
    by_name = {field.name: field for field in fields}
    if field_name not in by_name:
        available = ", ".join(sorted(by_name))
        msg = (
            f"PointCloud2 on topic {topic!r} has no per-point time field "
            f"{field_name!r}; available fields: {available}"
        )
        raise DatasetError(msg)

    point_field = by_name[field_name]
    base = POINTFIELD_NUMPY.get(point_field.datatype)
    if base is None:
        msg = (
            f"unsupported PointField datatype {point_field.datatype} for per-point "
            f"time field {field_name!r} on topic {topic!r}"
        )
        raise DatasetError(msg)

    dtype = np_mod.dtype(
        {
            "names": [field_name],
            "formats": [f"{byteorder}{base}"],
            "offsets": [point_field.offset],
            "itemsize": point_step,
        }
    )
    usable = min(point_count, len(payload) // point_step) if point_step else 0
    structured = np_mod.frombuffer(payload, dtype=dtype, count=usable)
    raw = structured[field_name]
    if point_field.datatype in _POINT_TIME_SECONDS_DATATYPES:
        offsets_s = raw.astype(np_mod.float64)
    elif point_field.datatype in _POINT_TIME_NANOSECONDS_DATATYPES:
        offsets_s = raw.astype(np_mod.float64) * 1.0e-9
    else:
        msg = (
            f"per-point time field {field_name!r} on topic {topic!r} has unsupported "
            f"datatype {point_field.datatype}; expected float seconds or integer "
            "nanoseconds"
        )
        raise DatasetError(msg)
    return cast("np.ndarray", offsets_s)
