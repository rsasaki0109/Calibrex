"""Shared ROS message decoding helpers for bag readers.

Point-cloud field layout and numpy decoding are shared between the ROS 1 and
ROS 2 bag readers. Numpy is imported lazily so mypy CI (without numpy) stays
clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

    @property
    def point_count(self) -> int:
        """Return the number of decoded points."""

        return int(self.xyz.shape[0])


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


def require_numpy(*, extra_name: str) -> Any:
    """Return the numpy module or raise a clear ``DatasetError``."""

    try:
        import numpy as numpy_module
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        msg = f"decoding ROS bag messages requires numpy; install calibrex[{extra_name}]"
        raise DatasetError(msg) from exc
    return numpy_module


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
