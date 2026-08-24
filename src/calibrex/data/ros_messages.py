"""Shared ROS message decoding helpers for bag readers.

Point-cloud field layout and numpy decoding are shared between the ROS 1 and
ROS 2 bag readers. Numpy is imported lazily so mypy CI (without numpy) stays
clean.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from calibrex.core.exceptions import DatasetError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


# ``radar_msgs`` is an external ROS message package.  Keep its exact source
# identity at this adapter boundary so normalized data can carry a reproducible
# definition reference without importing ROS (or the package itself).  These
# commits are the upstream repositories used to verify the wire definitions:
# ROS 2 ``radar_msgs/msg/RadarScan`` and ROS 1 ``radar_msgs/RadarScan``.
RADAR_MSGS_ROS2_SPEC_COMMIT = "47d2f26906ef38fa15ada352aea6b5aad547781d"
RADAR_MSGS_ROS1_SPEC_COMMIT = "bfd6d5487405500957dfa1b39a7e513ec172536a"
# Short aliases are kept for adapter authors who refer to the upstream
# identity as a package commit rather than a specification commit.
RADAR_MSGS_ROS2_COMMIT = RADAR_MSGS_ROS2_SPEC_COMMIT
RADAR_MSGS_ROS1_COMMIT = RADAR_MSGS_ROS1_SPEC_COMMIT
RADAR_MSGS_ROS2_VERSION = "0.2.2"
RADAR_MSGS_ROS1_VERSION = "0.2.2"
RADAR_MSGS_LICENSE = "Apache-2.0"
RADAR_MSGS_ROS2_SPEC_URL = (
    "https://github.com/ros-perception/radar_msgs/tree/"
    f"{RADAR_MSGS_ROS2_SPEC_COMMIT}/msg"
)
RADAR_MSGS_ROS1_SPEC_URL = (
    "https://github.com/ros-perception/radar_msgs/tree/"
    f"{RADAR_MSGS_ROS1_SPEC_COMMIT}/msg"
)
RADAR_SCAN_ROS2_TYPE = "radar_msgs/msg/RadarScan"
RADAR_SCAN_ROS1_TYPE = "radar_msgs/RadarScan"
RADAR_MSGS_PROVENANCE: dict[str, str] = {
    "package": "radar_msgs",
    "license": RADAR_MSGS_LICENSE,
    "ros2_type": RADAR_SCAN_ROS2_TYPE,
    "ros2_version": RADAR_MSGS_ROS2_VERSION,
    "ros2_commit": RADAR_MSGS_ROS2_SPEC_COMMIT,
    "ros2_url": RADAR_MSGS_ROS2_SPEC_URL,
    "ros1_type": RADAR_SCAN_ROS1_TYPE,
    "ros1_version": RADAR_MSGS_ROS1_VERSION,
    "ros1_commit": RADAR_MSGS_ROS1_SPEC_COMMIT,
    "ros1_url": RADAR_MSGS_ROS1_SPEC_URL,
}

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


MAX_RADAR_RETURNS = 100_000
MAX_RADAR_PAYLOAD_BYTES = 8 * 1024 * 1024
# Explicit aliases make the safety policy discoverable to adapter callers.
MAX_RADAR_RETURN_COUNT = MAX_RADAR_RETURNS
MAX_RADAR_BYTES = MAX_RADAR_PAYLOAD_BYTES
RADAR_AZIMUTH_MIN_RAD = -math.pi
RADAR_AZIMUTH_MAX_RAD = math.pi
RADAR_ELEVATION_MIN_RAD = -0.5 * math.pi
RADAR_ELEVATION_MAX_RAD = 0.5 * math.pi


@dataclass(frozen=True)
class RadarReturn:
    """One normalized ``radar_msgs/RadarReturn`` value.

    The official ROS 1 and ROS 2 definitions intentionally contain exactly
    these five ``float32`` fields, in this order.  Values are represented as
    Python ``float`` after decoding, but validation retains the wire-level
    semantics: range is non-negative, angles are bounded to the normal radar
    spherical domain, and every value must be finite.
    """

    range: float
    azimuth: float
    elevation: float
    doppler_velocity: float
    amplitude: float

    def __post_init__(self) -> None:
        values = (
            ("range", self.range),
            ("azimuth", self.azimuth),
            ("elevation", self.elevation),
            ("doppler_velocity", self.doppler_velocity),
            ("amplitude", self.amplitude),
        )
        for name, value in values:
            if not math.isfinite(float(value)):
                raise DatasetError(f"RadarReturn {name} must be finite")
        if self.range < 0.0:
            raise DatasetError("RadarReturn range must be non-negative")
        if not RADAR_AZIMUTH_MIN_RAD <= self.azimuth <= RADAR_AZIMUTH_MAX_RAD:
            raise DatasetError(
                "RadarReturn azimuth must lie within [-pi, pi] radians"
            )
        if not RADAR_ELEVATION_MIN_RAD <= self.elevation <= RADAR_ELEVATION_MAX_RAD:
            raise DatasetError(
                "RadarReturn elevation must lie within [-pi/2, pi/2] radians"
            )

    @property
    def values(self) -> tuple[float, float, float, float, float]:
        """Return the fields in the exact official wire order."""

        return (
            float(self.range),
            float(self.azimuth),
            float(self.elevation),
            float(self.doppler_velocity),
            float(self.amplitude),
        )

    @property
    def range_m(self) -> float:
        """Return ``range`` with an explicit SI-unit spelling."""

        return float(self.range)

    @property
    def azimuth_rad(self) -> float:
        """Return ``azimuth`` with an explicit SI-unit spelling."""

        return float(self.azimuth)

    @property
    def elevation_rad(self) -> float:
        """Return ``elevation`` with an explicit SI-unit spelling."""

        return float(self.elevation)

    @property
    def doppler_velocity_mps(self) -> float:
        """Return ``doppler_velocity`` with an explicit SI-unit spelling."""

        return float(self.doppler_velocity)

    @property
    def amplitude_db(self) -> float:
        """Return ``amplitude`` with an explicit SI-unit spelling."""

        return float(self.amplitude)


@dataclass(frozen=True)
class RadarScanMessage:
    """ROS-independent normalized ``radar_msgs/RadarScan`` message.

    ``returns`` is a tuple to make the bounded, immutable normalized record
    safe to pass across adapter boundaries.  ``source_spec`` identifies the
    exact upstream message definition used by the serializer decoder.
    """

    topic: str
    timestamp_ns: int
    frame_id: str
    returns: tuple[RadarReturn, ...]
    source_spec: str
    header_stamp_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "returns", tuple(self.returns))
        if len(self.returns) > MAX_RADAR_RETURNS:
            raise DatasetError(
                f"RadarScan contains {len(self.returns)} returns; "
                f"safe bound is {MAX_RADAR_RETURNS}"
            )
        if self.source_spec not in {RADAR_SCAN_ROS1_TYPE, RADAR_SCAN_ROS2_TYPE}:
            raise DatasetError(f"unsupported RadarScan source spec: {self.source_spec!r}")

    @property
    def return_count(self) -> int:
        """Return the number of radar returns in this scan."""

        return len(self.returns)

    @property
    def radar_return_count(self) -> int:
        """Compatibility alias used by capture evidence serializers."""

        return self.return_count

    @property
    def radar_returns(self) -> tuple[RadarReturn, ...]:
        """Return the normalized sequence under the message-package spelling."""

        return self.returns

    @property
    def source_spec_provenance(self) -> dict[str, str]:
        """Return the pinned upstream definition identity for this message."""

        if self.source_spec == RADAR_SCAN_ROS2_TYPE:
            return {
                "package": "radar_msgs",
                "type": RADAR_SCAN_ROS2_TYPE,
                "version": RADAR_MSGS_ROS2_VERSION,
                "commit": RADAR_MSGS_ROS2_SPEC_COMMIT,
                "license": RADAR_MSGS_LICENSE,
                "url": RADAR_MSGS_ROS2_SPEC_URL,
            }
        return {
            "package": "radar_msgs",
            "type": RADAR_SCAN_ROS1_TYPE,
            "version": RADAR_MSGS_ROS1_VERSION,
            "commit": RADAR_MSGS_ROS1_SPEC_COMMIT,
            "license": RADAR_MSGS_LICENSE,
            "url": RADAR_MSGS_ROS1_SPEC_URL,
        }

    @property
    def provenance(self) -> dict[str, str]:
        """Compatibility spelling for source definition provenance."""

        return self.source_spec_provenance

    @property
    def duplicate_return_count(self) -> int:
        """Return exact duplicate returns beyond their first occurrence.

        ``RadarReturn`` has no identifier, so equality of all five official
        fields is the only deterministic duplicate diagnostic available here.
        """

        counts: dict[tuple[float, float, float, float, float], int] = {}
        for value in self.returns:
            counts[value.values] = counts.get(value.values, 0) + 1
        return sum(max(0, count - 1) for count in counts.values())

    @property
    def duplicate_count(self) -> int:
        """Compatibility alias for the exact duplicate-return diagnostic."""

        return self.duplicate_return_count

    @staticmethod
    def _span(values: tuple[float, ...]) -> float | None:
        return max(values) - min(values) if values else None

    @property
    def range_min_m(self) -> float | None:
        return min((value.range_m for value in self.returns), default=None)

    @property
    def range_max_m(self) -> float | None:
        return max((value.range_m for value in self.returns), default=None)

    @property
    def azimuth_span_rad(self) -> float | None:
        return self._span(tuple(value.azimuth_rad for value in self.returns))

    @property
    def elevation_span_rad(self) -> float | None:
        return self._span(tuple(value.elevation_rad for value in self.returns))

    @property
    def doppler_min_mps(self) -> float | None:
        return min((value.doppler_velocity_mps for value in self.returns), default=None)

    @property
    def doppler_max_mps(self) -> float | None:
        return max((value.doppler_velocity_mps for value in self.returns), default=None)

    @property
    def doppler_span_mps(self) -> float | None:
        return self._span(
            tuple(value.doppler_velocity_mps for value in self.returns)
        )

    @property
    def diversity_status(self) -> str:
        """Classify bounded geometric/velocity excitation for readiness.

        This is an intake diagnostic, not a calibration quality score.  A
        single return (or a scan with no meaningful range/angle/velocity
        variation) is weak evidence and must not silently pass a required
        radar stream.
        """

        if not self.returns:
            return "empty"
        if len(self.returns) < 2:
            return "weak"
        spans = (
            (self.range_max_m or 0.0) - (self.range_min_m or 0.0),
            self.azimuth_span_rad or 0.0,
            self.elevation_span_rad or 0.0,
            self.doppler_span_mps or 0.0,
        )
        # Any clearly non-degenerate spherical/velocity dimension is enough
        # for a "strong" intake sample; the solver still owns its own gates.
        return "strong" if any(
            (spans[0] >= 0.5, spans[1] >= 0.05, spans[2] >= 0.05, spans[3] >= 0.1)
        ) else "weak"

    def diversity_summary(self) -> dict[str, object]:
        """Return JSON-safe evidence used by capture readiness and reports."""

        unique_count = self.return_count - self.duplicate_return_count
        return {
            "return_count": self.return_count,
            "unique_return_count": unique_count,
            "duplicate_return_count": self.duplicate_return_count,
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "range_span_m": (
                self.range_max_m - self.range_min_m
                if self.range_min_m is not None and self.range_max_m is not None
                else None
            ),
            "azimuth_span_rad": self.azimuth_span_rad,
            "elevation_span_rad": self.elevation_span_rad,
            "doppler_min_mps": self.doppler_min_mps,
            "doppler_max_mps": self.doppler_max_mps,
            "doppler_span_mps": self.doppler_span_mps,
            "status": self.diversity_status,
            "frame_id": self.frame_id or None,
            "source_spec": self.source_spec,
        }

    @property
    def diversity(self) -> dict[str, object]:
        """Compatibility spelling for :meth:`diversity_summary`."""

        return self.diversity_summary()

    def to_xyz(self) -> tuple[tuple[float, float, float], ...]:
        """Convert spherical radar returns to Cartesian coordinates in-frame."""

        return tuple(
            (
                value.range_m
                * math.cos(value.elevation_rad)
                * math.cos(value.azimuth_rad),
                value.range_m
                * math.cos(value.elevation_rad)
                * math.sin(value.azimuth_rad),
                value.range_m * math.sin(value.elevation_rad),
            )
            for value in self.returns
        )

    @property
    def xyz(self) -> tuple[tuple[float, float, float], ...]:
        """Return :meth:`to_xyz` for callers that use geometry-style naming."""

        return self.to_xyz()


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


@dataclass(frozen=True)
class ImageMessage:
    """A validated ``sensor_msgs/Image`` payload without ROS dependencies.

    The adapter deliberately stores image bytes only when the caller asks for
    them.  Inventory/readiness paths can therefore validate dimensions,
    encoding, row stride, and the declared byte count without decoding pixels
    or retaining a second full image buffer.
    """

    topic: str
    timestamp_ns: int
    frame_id: str
    height: int
    width: int
    encoding: str
    is_bigendian: bool
    step: int
    data_length: int
    data: bytes | None = None
    header_stamp_ns: int | None = None

    @property
    def byte_count(self) -> int:
        """Return the validated serialized image byte count."""

        return self.data_length

    @property
    def pixel_count(self) -> int:
        """Return the number of pixels represented by the image dimensions."""

        return self.width * self.height


@dataclass(frozen=True)
class RegionOfInterest:
    """The ROS ``sensor_msgs/RegionOfInterest`` value object."""

    x_offset: int
    y_offset: int
    height: int
    width: int
    do_rectify: bool


@dataclass(frozen=True)
class CameraInfoMessage:
    """A validated ``sensor_msgs/CameraInfo`` payload.

    ``d``, ``k``, ``r``, and ``p`` retain the ROS message naming so this
    record can be passed to adapters without an intermediate ROS package.
    The matrix properties below provide descriptive aliases for callers that
    do not use the ROS field names.
    """

    topic: str
    timestamp_ns: int
    frame_id: str
    height: int
    width: int
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]
    binning_x: int
    binning_y: int
    roi: RegionOfInterest
    header_stamp_ns: int | None = None

    @property
    def distortion(self) -> tuple[float, ...]:
        """Return distortion coefficients under the common adapter spelling."""

        return self.d

    @property
    def intrinsic_matrix(self) -> tuple[float, ...]:
        """Return the row-major 3x3 intrinsic matrix ``K``."""

        return self.k

    @property
    def rectification_matrix(self) -> tuple[float, ...]:
        """Return the row-major 3x3 rectification matrix ``R``."""

        return self.r

    @property
    def projection_matrix(self) -> tuple[float, ...]:
        """Return the row-major 3x4 projection matrix ``P``."""

        return self.p

    @property
    def is_calibrated(self) -> bool:
        """Return whether the intrinsic and projection focal lengths are usable."""

        return (
            len(self.k) == 9
            and len(self.p) == 12
            and math.isfinite(self.k[0])
            and math.isfinite(self.k[4])
            and math.isfinite(self.p[0])
            and math.isfinite(self.p[5])
            and self.k[0] > 0.0
            and self.k[4] > 0.0
            and self.p[0] > 0.0
            and self.p[5] > 0.0
        )

    @property
    def calibration_status(self) -> str:
        """Return a readiness-facing label without changing ROS semantics.

        ROS uses an all-zero ``K`` matrix (in particular ``K[0] == 0``) to
        represent a valid, but uncalibrated, CameraInfo message.  Parsing and
        readiness therefore remain separate: this property reports the latter
        while the decoder still preserves the complete message.
        """

        return "calibrated" if self.is_calibrated else "uncalibrated"


# A bounded parser must reject hostile dimensions before multiplying them or
# entering a sequence loop.  These limits are intentionally generous for
# automotive cameras while keeping inventory safe for untrusted bags.
MAX_IMAGE_WIDTH = 32_768
MAX_IMAGE_HEIGHT = 32_768
MAX_IMAGE_BYTES = 512 * 1024 * 1024
MAX_IMAGE_ENCODING_LENGTH = 128
MAX_CAMERA_INFO_DISTORTION_COEFFICIENTS = 128
MAX_ROS_STRING_BYTES = 1 * 1024 * 1024

_IMAGE_ENCODING_LAYOUT: dict[str, tuple[int, int]] = {
    "mono8": (1, 1),
    "mono16": (2, 1),
    "8uc1": (1, 1),
    "8sc1": (1, 1),
    "8uc2": (1, 2),
    "8sc2": (1, 2),
    "8uc3": (1, 3),
    "8sc3": (1, 3),
    "8uc4": (1, 4),
    "8sc4": (1, 4),
    "16uc1": (2, 1),
    "16sc1": (2, 1),
    "16uc2": (2, 2),
    "16sc2": (2, 2),
    "16uc3": (2, 3),
    "16sc3": (2, 3),
    "16uc4": (2, 4),
    "16sc4": (2, 4),
    "32sc1": (4, 1),
    "32sc2": (4, 2),
    "32sc3": (4, 3),
    "32sc4": (4, 4),
    "32fc1": (4, 1),
    "32fc2": (4, 2),
    "32fc3": (4, 3),
    "32fc4": (4, 4),
    "64fc1": (8, 1),
    "64fc2": (8, 2),
    "64fc3": (8, 3),
    "64fc4": (8, 4),
    "rgb8": (1, 3),
    "bgr8": (1, 3),
    "rgba8": (1, 4),
    "bgra8": (1, 4),
    "rgb16": (2, 3),
    "bgr16": (2, 3),
    "rgba16": (2, 4),
    "bgra16": (2, 4),
    "yuv422": (2, 2),
    "bayer_rggb8": (1, 1),
    "bayer_bggr8": (1, 1),
    "bayer_gbrg8": (1, 1),
    "bayer_grbg8": (1, 1),
    "bayer_rggb16": (2, 1),
    "bayer_bggr16": (2, 1),
    "bayer_gbrg16": (2, 1),
    "bayer_grbg16": (2, 1),
}
_GENERIC_IMAGE_ENCODING = re.compile(r"^(8|16|32|64)(U|S|F)C([1-9][0-9]*)$", re.IGNORECASE)


def image_encoding_layout(encoding: str) -> tuple[int, int]:
    """Return ``(bytes_per_channel, channels)`` for a standard ROS encoding.

    Unknown encodings are rejected rather than being marked supported merely
    because the declared byte array fits the dimensions.  This is the key
    fail-closed invariant used by capture readiness.
    """

    normalized = encoding.strip().lower()
    layout = _IMAGE_ENCODING_LAYOUT.get(normalized)
    if layout is not None:
        return layout
    match = _GENERIC_IMAGE_ENCODING.fullmatch(encoding.strip())
    if match is not None:
        bits = int(match.group(1))
        channels = int(match.group(3))
        return bits // 8, channels
    raise DatasetError(f"unsupported sensor_msgs/Image encoding {encoding!r}")


def validate_image_metadata(
    *,
    height: int,
    width: int,
    encoding: str,
    is_bigendian: int | bool,
    step: int,
    data_length: int,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT,
    max_data_bytes: int = MAX_IMAGE_BYTES,
) -> None:
    """Validate Image dimensions, encoding, stride, and payload size.

    The function performs no image decode and allocates no buffer.  It is
    shared by ROS 1 and ROS 2 adapters so both serializers have identical
    readiness semantics.
    """

    if not 0 < height <= max_height:
        raise DatasetError(f"sensor_msgs/Image height {height} exceeds safe bounds")
    if not 0 < width <= max_width:
        raise DatasetError(f"sensor_msgs/Image width {width} exceeds safe bounds")
    if not encoding or len(encoding.encode("utf-8")) > MAX_IMAGE_ENCODING_LENGTH:
        raise DatasetError("sensor_msgs/Image encoding is empty or too long")
    if int(is_bigendian) not in (0, 1):
        raise DatasetError("sensor_msgs/Image is_bigendian must be 0 or 1")
    if step <= 0:
        raise DatasetError("sensor_msgs/Image step must be positive")
    if data_length < 0 or data_length > max_data_bytes:
        raise DatasetError(
            f"sensor_msgs/Image data length {data_length} exceeds safe bound {max_data_bytes}"
        )
    bytes_per_channel, channels = image_encoding_layout(encoding)
    minimum_step = width * bytes_per_channel * channels
    if step < minimum_step:
        raise DatasetError(
            f"sensor_msgs/Image step {step} is smaller than encoded row size {minimum_step}"
        )
    expected_length = step * height
    if expected_length > max_data_bytes:
        raise DatasetError(
            f"sensor_msgs/Image declared payload {expected_length} exceeds safe bound "
            f"{max_data_bytes}"
        )
    if data_length != expected_length:
        raise DatasetError(
            f"sensor_msgs/Image data length {data_length} does not equal step*height "
            f"({expected_length})"
        )


def validate_camera_info_metadata(
    *,
    height: int,
    width: int,
    distortion_model: str,
    d: tuple[float, ...],
    k: tuple[float, ...],
    r: tuple[float, ...],
    p: tuple[float, ...],
    binning_x: int,
    binning_y: int,
    roi: RegionOfInterest,
    max_width: int = MAX_IMAGE_WIDTH,
    max_height: int = MAX_IMAGE_HEIGHT,
    max_distortion_coefficients: int = MAX_CAMERA_INFO_DISTORTION_COEFFICIENTS,
) -> None:
    """Validate CameraInfo dimensions, matrices, finite values, and ROI."""

    if not 0 <= height <= max_height or not 0 <= width <= max_width:
        raise DatasetError("sensor_msgs/CameraInfo dimensions are outside safe bounds")
    if len(distortion_model.encode("utf-8")) > MAX_ROS_STRING_BYTES:
        raise DatasetError("sensor_msgs/CameraInfo distortion_model is too long")
    if len(d) > max_distortion_coefficients:
        raise DatasetError("sensor_msgs/CameraInfo distortion coefficient sequence is too long")
    if len(k) != 9 or len(r) != 9 or len(p) != 12:
        raise DatasetError("sensor_msgs/CameraInfo D/K/R/P arrays have invalid lengths")
    for name, values in (("D", d), ("K", k), ("R", r), ("P", p)):
        if not all(math.isfinite(value) for value in values):
            raise DatasetError(f"sensor_msgs/CameraInfo {name} contains non-finite values")
    if binning_x < 0 or binning_y < 0:
        raise DatasetError("sensor_msgs/CameraInfo binning values must be non-negative")
    if roi.x_offset + roi.width > width or roi.y_offset + roi.height > height:
        raise DatasetError("sensor_msgs/CameraInfo ROI lies outside image dimensions")


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
