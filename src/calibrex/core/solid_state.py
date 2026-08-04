"""Schema-validatable metadata for solid-state LiDAR calibration.

Solid-state sensors do not share the single, implicit "scan" semantics that
are often assumed by spinning LiDAR tooling.  This module keeps those
assumptions explicit: the scan pattern, the point-time convention, the
integration window, the intrinsic-calibration source, and the temperature
condition travel with a calibration result.

The models are deliberately ROS-independent.  ROS/MCAP/Livox adapters may
populate them, but the core result contract does not depend on a middleware
message type.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SOLID_STATE_CONTEXT_SCHEMA_VERSION: Literal[
    "slac.solid_state_lidar_context/v0.1"
] = "slac.solid_state_lidar_context/v0.1"

SolidStateArchitecture = Literal[
    "solid_state",
    "mems",
    "flash",
    "opa",
    "hybrid",
    "unknown",
]
SolidStateScanPattern = Literal[
    "repetitive",
    "non_repetitive",
    "frame",
    "unknown",
]
PointTimeReference = Literal[
    "message_stamp",
    "timebase",
    "sweep_start",
    "sweep_end",
    "mid_sweep",
    "absolute_sensor_time",
    "unknown",
]
PointTimeUnit = Literal["seconds", "nanoseconds", "unknown"]
IntrinsicCalibrationSource = Literal[
    "manufacturer",
    "estimated",
    "imported",
    "unknown",
]
IntrinsicCalibrationModel = Literal[
    "manufacturer_native",
    "beam_table",
    "angular_resolution",
    "flash_projection",
    "unknown",
]
TemperatureSource = Literal[
    "sensor_telemetry",
    "ambient",
    "external",
    "unknown",
]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class StrictModel(BaseModel):
    """Base model with stable, explicit fields."""

    model_config = ConfigDict(extra="forbid")


class SolidStateIntrinsicCalibration(StrictModel):
    """Origin and representation of the sensor's internal calibration."""

    source: IntrinsicCalibrationSource = "unknown"
    model: IntrinsicCalibrationModel = "unknown"
    artifact_path: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    parameters: dict[str, float] = Field(default_factory=dict)


class SolidStateTemperatureCondition(StrictModel):
    """Temperature attached to a calibration or capture interval."""

    sensor_c: float | None = None
    ambient_c: float | None = None
    min_c: float | None = None
    max_c: float | None = None
    source: TemperatureSource = "unknown"
    compensation_applied: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> SolidStateTemperatureCondition:
        if self.min_c is not None and self.max_c is not None and self.max_c < self.min_c:
            raise ValueError("solid-state temperature max_c must be >= min_c")
        return self


class SolidStateLidarProfile(StrictModel):
    """Static and acquisition-semantic metadata for one solid-state LiDAR."""

    architecture: SolidStateArchitecture = "unknown"
    scan_pattern: SolidStateScanPattern = "unknown"
    integration_window_s: float | None = Field(default=None, ge=0.0)
    point_time_reference: PointTimeReference = "unknown"
    point_time_unit: PointTimeUnit = "unknown"
    point_time_available: bool = False
    point_time_field: str | None = Field(
        default=None,
        description="source field carrying per-point capture-time offsets, when applicable",
    )
    intrinsic_calibration: SolidStateIntrinsicCalibration = Field(
        default_factory=SolidStateIntrinsicCalibration
    )
    temperature: SolidStateTemperatureCondition | None = None
    firmware_version: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_point_time_semantics(self) -> SolidStateLidarProfile:
        if self.point_time_available and (
            self.point_time_reference == "unknown" or self.point_time_unit == "unknown"
        ):
            raise ValueError(
                "point_time_available requires point_time_reference and point_time_unit"
            )
        return self


class SolidStateEvaluationConfig(StrictModel):
    """Evaluation gates specific to non-repetitive/solid-state acquisition."""

    enabled: bool = True
    require_point_time: bool = False
    require_temperature: bool = False
    require_independent_holdout: bool = False
    min_holdout_windows: int = Field(default=2, ge=1)
    min_temperature_span_c: float = Field(default=5.0, ge=0.0)
    min_distance_span_m: float = Field(default=5.0, ge=0.0)
    min_fov_azimuth_deg: float = Field(default=20.0, ge=0.0, le=360.0)
    min_fov_elevation_deg: float = Field(default=5.0, ge=0.0, le=180.0)
    distance_bins_m: list[float] = Field(
        default_factory=lambda: [0.0, 10.0, 20.0, 40.0, 80.0]
    )
    min_points_per_distance_bin: int = Field(default=20, ge=0)
    min_known_bad_detectable_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("distance_bins_m")
    @classmethod
    def validate_distance_bins(cls, value: list[float]) -> list[float]:
        if len(value) < 2:
            raise ValueError("distance_bins_m requires at least two edges")
        if value[0] < 0.0 or any(right <= left for left, right in pairwise(value)):
            raise ValueError("distance_bins_m must be non-negative and strictly increasing")
        return value


class SolidStateCaptureWindow(StrictModel):
    """Observed timing and operating conditions for one capture interval."""

    start_timestamp_ns: int | None = None
    end_timestamp_ns: int | None = None
    reference_timestamp_ns: int | None = None
    point_time_field: str | None = None
    point_time_offsets_available: bool = False
    point_time_deskew_applied: bool = False
    point_time_min_s: float | None = None
    point_time_max_s: float | None = None
    integration_window_s: float | None = Field(default=None, ge=0.0)
    capture_span_s: float | None = Field(default=None, ge=0.0)
    temperature: SolidStateTemperatureCondition | None = None
    point_count: int | None = Field(default=None, ge=0)
    range_min_m: float | None = Field(default=None, ge=0.0)
    range_max_m: float | None = Field(default=None, ge=0.0)
    fov_azimuth_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    fov_elevation_deg: float | None = Field(default=None, ge=0.0, le=180.0)

    @model_validator(mode="after")
    def validate_window(self) -> SolidStateCaptureWindow:
        if (
            self.start_timestamp_ns is not None
            and self.end_timestamp_ns is not None
            and self.end_timestamp_ns < self.start_timestamp_ns
        ):
            raise ValueError("solid-state capture end timestamp must be >= start timestamp")
        if (
            self.point_time_min_s is not None
            and self.point_time_max_s is not None
            and self.point_time_max_s < self.point_time_min_s
        ):
            raise ValueError("solid-state point-time max must be >= min")
        if (
            self.range_min_m is not None
            and self.range_max_m is not None
            and self.range_max_m < self.range_min_m
        ):
            raise ValueError("solid-state range_max_m must be >= range_min_m")
        return self


class SolidStateProvenance(StrictModel):
    """Provenance for solid-state metadata and operating-condition claims."""

    source_paths: list[str] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(default_factory=dict)
    tool_name: str = "calibrex"
    tool_version: str | None = None
    git_commit: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if not _SHA256_RE.fullmatch(digest)]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class SolidStateLidarCalibrationContext(StrictModel):
    """Result-level solid-state context for one calibration run."""

    schema_version: Literal[
        "slac.solid_state_lidar_context/v0.1"
    ] = SOLID_STATE_CONTEXT_SCHEMA_VERSION
    sensors: dict[str, SolidStateLidarProfile]
    capture_windows: dict[str, SolidStateCaptureWindow] = Field(default_factory=dict)
    provenance: SolidStateProvenance

    @field_validator("sensors")
    @classmethod
    def require_sensor_profiles(
        cls, value: dict[str, SolidStateLidarProfile]
    ) -> dict[str, SolidStateLidarProfile]:
        if not value:
            raise ValueError("solid-state context requires at least one sensor profile")
        return value


def build_solid_state_context(
    profiles: Mapping[str, SolidStateLidarProfile],
    *,
    config_sha256: str | None = None,
    dataset_sha256: str | None = None,
    source_paths: list[str] | None = None,
    tool_version: str | None = None,
    git_commit: str | None = None,
    notes: list[str] | None = None,
) -> SolidStateLidarCalibrationContext | None:
    """Build a result context from configured solid-state sensor profiles.

    ``None`` is returned when the config contains no solid-state profile, so
    existing spinning-LiDAR and camera/IMU results keep their historical
    shape apart from the optional result field.
    """

    if not profiles:
        return None
    source_sha256: dict[str, str] = {}
    if config_sha256 is not None:
        source_sha256["config"] = config_sha256
    if dataset_sha256 is not None:
        source_sha256["dataset"] = dataset_sha256
    return SolidStateLidarCalibrationContext(
        sensors=dict(profiles),
        provenance=SolidStateProvenance(
            source_paths=list(source_paths or []),
            source_sha256=source_sha256,
            tool_version=tool_version,
            git_commit=git_commit,
            notes=list(notes or []),
        ),
    )


def solid_state_context_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for solid-state context."""

    return SolidStateLidarCalibrationContext.model_json_schema()
