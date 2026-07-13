"""Sensor capture-time semantics and constant-twist LiDAR deskew primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from calibrex.core.geometry import SE3, Vector3, quaternion_xyzw_from_rotation_matrix

PointOffsetUnit = Literal["seconds", "nanoseconds"]
StampReference = Literal["scan_start", "scan_midpoint", "scan_end"]


@dataclass(frozen=True)
class LidarCaptureTimePolicy:
    """Declare how a LiDAR point timestamp relates to its message stamp."""

    point_offset_unit: PointOffsetUnit
    stamp_reference: StampReference
    sensor_time_offset_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.point_offset_unit not in {"seconds", "nanoseconds"}:
            raise ValueError("point_offset_unit must be seconds or nanoseconds")
        if self.stamp_reference not in {"scan_start", "scan_midpoint", "scan_end"}:
            raise ValueError("stamp_reference must be scan_start, scan_midpoint, or scan_end")
        if not math.isfinite(self.sensor_time_offset_sec):
            raise ValueError("sensor_time_offset_sec must be finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> LidarCaptureTimePolicy:
        """Parse a schema-friendly policy mapping with strict enum validation."""

        point_offset_unit = value.get("point_offset_unit")
        stamp_reference = value.get("stamp_reference")
        sensor_time_offset_sec = value.get("sensor_time_offset_sec", 0.0)
        if not isinstance(point_offset_unit, str) or not isinstance(stamp_reference, str):
            raise ValueError("capture-time policy requires string unit and stamp reference")
        if not isinstance(sensor_time_offset_sec, int | float):
            raise ValueError("sensor_time_offset_sec must be numeric")
        return cls(
            point_offset_unit=cast(PointOffsetUnit, point_offset_unit),
            stamp_reference=cast(StampReference, stamp_reference),
            sensor_time_offset_sec=float(sensor_time_offset_sec),
        )

    def point_reference_time_sec(
        self, message_stamp_sec: float, point_offset: float | int
    ) -> float:
        """Return reference-clock capture time for one point.

        Point offsets are signed relative to the message stamp. The declared
        stamp-reference label is provenance; it does not silently change the
        numeric offset supplied by the frontend.
        """

        scale = 1.0 if self.point_offset_unit == "seconds" else 1.0e-9
        return message_stamp_sec + float(point_offset) * scale + self.sensor_time_offset_sec

    def as_dict(self) -> dict[str, object]:
        return {
            "point_offset_unit": self.point_offset_unit,
            "stamp_reference": self.stamp_reference,
            "sensor_time_offset_sec": self.sensor_time_offset_sec,
            "time_convention": "sensor_time + dt_sensor = reference_time",
            "point_offset_convention": "signed offset relative to message stamp",
        }


@dataclass(frozen=True)
class TimedLidarPoint:
    """LiDAR-frame point with a capture offset relative to its message stamp."""

    point_id: str
    position_lidar_m: Vector3
    capture_offset: float | int


@dataclass(frozen=True)
class ConstantBodyTwist:
    """Body linear/angular velocity expressed in the body reference frame."""

    linear_velocity_body_mps: Vector3
    angular_velocity_body_radps: Vector3

    def as_dict(self) -> dict[str, object]:
        return {
            "linear_velocity_body_mps": list(self.linear_velocity_body_mps),
            "angular_velocity_body_radps": list(self.angular_velocity_body_radps),
        }


@dataclass(frozen=True)
class DeskewedLidarPoint:
    point_id: str
    position_lidar_at_reference_m: Vector3
    capture_time_reference_sec: float
    deskew_delta_sec: float


@dataclass(frozen=True)
class LidarDeskewResult:
    points: tuple[DeskewedLidarPoint, ...]
    reference_time_sec: float
    minimum_capture_time_sec: float | None
    maximum_capture_time_sec: float | None
    maximum_abs_deskew_sec: float
    policy: LidarCaptureTimePolicy
    twist: ConstantBodyTwist
    transform_body_lidar: SE3

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_time_sec": self.reference_time_sec,
            "minimum_capture_time_sec": self.minimum_capture_time_sec,
            "maximum_capture_time_sec": self.maximum_capture_time_sec,
            "maximum_abs_deskew_sec": self.maximum_abs_deskew_sec,
            "point_count": len(self.points),
            "policy": self.policy.as_dict(),
            "twist": self.twist.as_dict(),
            "transform_body_lidar": self.transform_body_lidar.as_dict(),
            "points": [point.__dict__ for point in self.points],
            "method": "constant_body_twist_per_point_deskew/v0.2",
        }


def deskew_lidar_points_to_reference(
    points: Sequence[TimedLidarPoint],
    *,
    message_stamp_sec: float,
    reference_time_sec: float,
    policy: LidarCaptureTimePolicy,
    twist: ConstantBodyTwist,
    transform_body_lidar: SE3,
) -> LidarDeskewResult:
    """Move sequential LiDAR returns to one reference-clock instant."""

    output: list[DeskewedLidarPoint] = []
    capture_times: list[float] = []
    for point in points:
        capture_time = policy.point_reference_time_sec(message_stamp_sec, point.capture_offset)
        delta = reference_time_sec - capture_time
        position_body_capture = transform_body_lidar.transform_point(point.position_lidar_m)
        position_body_reference = _static_point_at_later_body_frame(
            position_body_capture, twist, delta
        )
        position_lidar_reference = transform_body_lidar.inverse().transform_point(
            position_body_reference
        )
        output.append(
            DeskewedLidarPoint(
                point.point_id,
                position_lidar_reference,
                capture_time,
                delta,
            )
        )
        capture_times.append(capture_time)
    return LidarDeskewResult(
        points=tuple(output),
        reference_time_sec=reference_time_sec,
        minimum_capture_time_sec=min(capture_times) if capture_times else None,
        maximum_capture_time_sec=max(capture_times) if capture_times else None,
        maximum_abs_deskew_sec=max(
            (abs(reference_time_sec - value) for value in capture_times), default=0.0
        ),
        policy=policy,
        twist=twist,
        transform_body_lidar=transform_body_lidar,
    )


def _static_point_at_later_body_frame(
    point_body_capture: Vector3,
    twist: ConstantBodyTwist,
    delta_sec: float,
) -> Vector3:
    # T_body(reference)_body(capture): inverse of forward body motion.
    rotation = _axis_angle_rotation(twist.angular_velocity_body_radps, -delta_sec)
    rotated = rotation.transform_point(point_body_capture)
    velocity = twist.linear_velocity_body_mps
    return (
        rotated[0] - velocity[0] * delta_sec,
        rotated[1] - velocity[1] * delta_sec,
        rotated[2] - velocity[2] * delta_sec,
    )


def _axis_angle_rotation(angular_velocity: Vector3, delta_sec: float) -> SE3:
    wx, wy, wz = angular_velocity
    speed = math.sqrt(wx * wx + wy * wy + wz * wz)
    if speed < 1.0e-15 or delta_sec == 0.0:
        return SE3.identity()
    x, y, z = wx / speed, wy / speed, wz / speed
    angle = speed * delta_sec
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1.0 - cosine
    matrix = (
        cosine + x * x * one_minus,
        x * y * one_minus - z * sine,
        x * z * one_minus + y * sine,
        y * x * one_minus + z * sine,
        cosine + y * y * one_minus,
        y * z * one_minus - x * sine,
        z * x * one_minus - y * sine,
        z * y * one_minus + x * sine,
        cosine + z * z * one_minus,
    )
    return SE3((0.0, 0.0, 0.0), quaternion_xyzw_from_rotation_matrix(matrix))
