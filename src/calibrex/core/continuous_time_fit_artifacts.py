"""Schema-valid continuous-time trajectory measurement and fit artifacts."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel, TransformResult

CONTINUOUS_TIME_MEASUREMENTS_SCHEMA_VERSION: Literal[
    "slac.continuous_time_trajectory_measurements/v0.1"
] = "slac.continuous_time_trajectory_measurements/v0.1"
CONTINUOUS_TIME_FIT_SCHEMA_VERSION: Literal[
    "slac.continuous_time_trajectory_fit/v0.1"
] = "slac.continuous_time_trajectory_fit/v0.1"


class ContinuousTimeFitProvenance(StrictModel):
    """Generator and immutable input lineage for trajectory fits."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measurements_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TrajectoryMeasurementProvenance(StrictModel):
    """Generator and immutable input lineage for measurement sets."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TrajectoryPoseMeasurementArtifact(StrictModel):
    """A measured world pose at a timestamp."""

    measurement_id: str
    timestamp_sec: float
    pose_world_body: TransformResult
    weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def check_measurement(self) -> TrajectoryPoseMeasurementArtifact:
        """Reject non-finite or mistyped pose measurements."""

        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("pose measurement timestamp must be finite")
        return self


class TrajectoryPointMeasurementArtifact(StrictModel):
    """A body-frame point observed at a timestamp against a world target."""

    measurement_id: str
    timestamp_sec: float
    point_body_m: list[float] = Field(min_length=3, max_length=3)
    target_world_m: list[float] = Field(min_length=3, max_length=3)
    weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def check_measurement(self) -> TrajectoryPointMeasurementArtifact:
        """Reject non-finite or mistyped point measurements."""

        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("point measurement timestamp must be finite")
        values = (
            *self.point_body_m,
            *self.target_world_m,
            self.weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("point measurement values must be finite")
        return self


class TrajectoryPointToPlaneMeasurementArtifact(StrictModel):
    """A body-frame LiDAR return against a world plane at a timestamp."""

    measurement_id: str
    timestamp_sec: float
    point_body_m: list[float] = Field(min_length=3, max_length=3)
    plane_point_world_m: list[float] = Field(min_length=3, max_length=3)
    plane_normal_world: list[float] = Field(min_length=3, max_length=3)
    weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def check_measurement(self) -> TrajectoryPointToPlaneMeasurementArtifact:
        """Reject non-finite or degenerate plane measurements."""

        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("point-to-plane timestamp must be finite")
        values = (
            *self.point_body_m,
            *self.plane_point_world_m,
            *self.plane_normal_world,
            self.weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("point-to-plane values must be finite")
        normal_norm = math.sqrt(sum(value * value for value in self.plane_normal_world))
        if normal_norm <= 1.0e-12:
            raise ValueError("point-to-plane normal must be non-zero")
        return self


class TrajectoryImuGyroSampleArtifact(StrictModel):
    """A body-frame gyro reading at a timestamp."""

    timestamp_sec: float
    omega_body_rad_s: list[float] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def check_sample(self) -> TrajectoryImuGyroSampleArtifact:
        """Reject non-finite gyro samples."""

        if not math.isfinite(self.timestamp_sec):
            raise ValueError("gyro sample timestamp must be finite")
        if not all(math.isfinite(value) for value in self.omega_body_rad_s):
            raise ValueError("gyro sample rates must be finite")
        return self


class TrajectoryImuPreintegrationMeasurementArtifact(StrictModel):
    """Gyro samples spanning two timestamps for a rotation pre-integration."""

    measurement_id: str
    gyro_samples: list[TrajectoryImuGyroSampleArtifact] = Field(min_length=2)
    weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def check_measurement(self) -> TrajectoryImuPreintegrationMeasurementArtifact:
        """Reject empty IDs or non-increasing gyro timestamps."""

        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        timestamps = [item.timestamp_sec for item in self.gyro_samples]
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("gyro sample timestamps must be strictly increasing")
        return self


class TrajectoryImuLeverArmMeasurementArtifact(StrictModel):
    """Gravity-compensated specific force used to observe the IMU lever arm."""

    measurement_id: str
    timestamp_sec: float
    accel_body_m_s2: list[float] = Field(min_length=3, max_length=3)
    weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def check_measurement(self) -> TrajectoryImuLeverArmMeasurementArtifact:
        """Reject non-finite lever-arm measurements."""

        if not self.measurement_id:
            raise ValueError("measurement_id must not be empty")
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("lever-arm timestamp must be finite")
        if not all(math.isfinite(value) for value in self.accel_body_m_s2):
            raise ValueError("lever-arm acceleration must be finite")
        return self


class ContinuousTimeTrajectoryMeasurements(StrictModel):
    """Measurements constraining a continuous-time trajectory fit."""

    schema_version: Literal[
        "slac.continuous_time_trajectory_measurements/v0.1"
    ] = CONTINUOUS_TIME_MEASUREMENTS_SCHEMA_VERSION
    measurements_id: str
    world_frame: str
    body_frame: str
    pose_measurements: list[TrajectoryPoseMeasurementArtifact] = Field(
        default_factory=list
    )
    point_measurements: list[TrajectoryPointMeasurementArtifact] = Field(
        default_factory=list
    )
    point_to_plane_measurements: list[TrajectoryPointToPlaneMeasurementArtifact] = (
        Field(default_factory=list)
    )
    imu_preintegration_measurements: list[
        TrajectoryImuPreintegrationMeasurementArtifact
    ] = Field(default_factory=list)
    imu_lever_arm_measurements: list[TrajectoryImuLeverArmMeasurementArtifact] = Field(
        default_factory=list
    )
    initial_gyro_bias_rad_s: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )
    initial_lever_arm_body_m: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )
    estimate_gyro_bias: bool = False
    estimate_lever_arm: bool = False
    estimate_imu_clock_offset: bool = False
    estimate_accel_bias: bool = False
    estimate_gravity: bool = False
    estimate_gyro_scale: bool = False
    estimate_accel_scale: bool = False
    initial_imu_clock_offset_sec: float = 0.0
    initial_accel_bias_body_m_s2: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )
    initial_gravity_world_m_s2: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        min_length=3,
        max_length=3,
    )
    initial_gyro_scale: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        min_length=3,
        max_length=3,
    )
    initial_accel_scale: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        min_length=3,
        max_length=3,
    )
    provenance: TrajectoryMeasurementProvenance

    @model_validator(mode="after")
    def check_measurements(self) -> ContinuousTimeTrajectoryMeasurements:
        """Require at least one measurement and consistent frame IDs."""

        if (
            not self.pose_measurements
            and not self.point_measurements
            and not self.point_to_plane_measurements
            and not self.imu_preintegration_measurements
            and not self.imu_lever_arm_measurements
        ):
            raise ValueError("measurements set requires at least one measurement")
        identifiers = (
            [item.measurement_id for item in self.pose_measurements]
            + [item.measurement_id for item in self.point_measurements]
            + [item.measurement_id for item in self.point_to_plane_measurements]
            + [item.measurement_id for item in self.imu_preintegration_measurements]
            + [item.measurement_id for item in self.imu_lever_arm_measurements]
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("measurement IDs must be unique")
        if not all(math.isfinite(value) for value in self.initial_gyro_bias_rad_s):
            raise ValueError("initial gyro bias must be finite")
        if not all(math.isfinite(value) for value in self.initial_lever_arm_body_m):
            raise ValueError("initial lever arm must be finite")
        if not math.isfinite(self.initial_imu_clock_offset_sec):
            raise ValueError("initial IMU clock offset must be finite")
        if not all(math.isfinite(value) for value in self.initial_accel_bias_body_m_s2):
            raise ValueError("initial accelerometer bias must be finite")
        if not all(math.isfinite(value) for value in self.initial_gravity_world_m_s2):
            raise ValueError("initial gravity must be finite")
        if self.estimate_gyro_bias and not self.imu_preintegration_measurements:
            raise ValueError(
                "gyro bias estimation requires IMU pre-integration measurements"
            )
        if self.estimate_lever_arm and not self.imu_lever_arm_measurements:
            raise ValueError("lever-arm estimation requires IMU lever-arm measurements")
        if self.estimate_imu_clock_offset and not (
            self.imu_preintegration_measurements or self.imu_lever_arm_measurements
        ):
            raise ValueError("IMU clock-offset estimation requires IMU measurements")
        if self.estimate_accel_bias and not self.imu_lever_arm_measurements:
            raise ValueError(
                "accelerometer-bias estimation requires IMU lever-arm measurements"
            )
        if self.estimate_gravity and not self.imu_lever_arm_measurements:
            raise ValueError("gravity estimation requires IMU lever-arm measurements")
        if not all(math.isfinite(value) for value in self.initial_gyro_scale):
            raise ValueError("initial gyro scale must be finite")
        if not all(math.isfinite(value) for value in self.initial_accel_scale):
            raise ValueError("initial accelerometer scale must be finite")
        if any(value <= 0.0 for value in self.initial_gyro_scale):
            raise ValueError("initial gyro scale must be positive")
        if any(value <= 0.0 for value in self.initial_accel_scale):
            raise ValueError("initial accelerometer scale must be positive")
        if self.estimate_gyro_scale and not self.imu_preintegration_measurements:
            raise ValueError(
                "gyro scale estimation requires IMU pre-integration measurements"
            )
        if self.estimate_accel_scale and not self.imu_lever_arm_measurements:
            raise ValueError(
                "accelerometer scale estimation requires IMU lever-arm measurements"
            )
        for item in self.pose_measurements:
            if (
                item.pose_world_body.parent != self.world_frame
                or item.pose_world_body.child != self.body_frame
            ):
                raise ValueError("pose measurement frame IDs are inconsistent")
        return self

    def save(self, path: str | Path) -> None:
        """Save the measurements set as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


class ContinuousTimeFittedKnot(StrictModel):
    """One fitted trajectory knot with its initial value."""

    knot_index: int = Field(ge=0)
    timestamp_sec: float
    initial_transform_world_body: TransformResult
    transform_world_body: TransformResult

    @model_validator(mode="after")
    def check_knot(self) -> ContinuousTimeFittedKnot:
        """Reject non-finite knot timestamps."""

        if not math.isfinite(self.timestamp_sec):
            raise ValueError("fitted knot timestamp must be finite")
        return self


class ContinuousTimeTrajectoryFitResultArtifact(StrictModel):
    """Schema-valid result of a sparse continuous-time trajectory fit."""

    schema_version: Literal[
        "slac.continuous_time_trajectory_fit/v0.1"
    ] = CONTINUOUS_TIME_FIT_SCHEMA_VERSION
    fit_id: str
    trajectory_id: str
    interpolation: Literal["screw_linear"]
    status: Literal["converged", "max_iterations", "singular_system"]
    iterations: int = Field(ge=0)
    final_objective: float
    final_point_rmse: float | None = Field(default=None, ge=0.0)
    final_point_to_plane_rmse: float | None = Field(default=None, ge=0.0)
    final_pose_rmse: float | None = Field(default=None, ge=0.0)
    final_imu_rotation_rmse_rad: float | None = Field(default=None, ge=0.0)
    final_lever_arm_rmse_m_s2: float | None = Field(default=None, ge=0.0)
    gyro_bias_rad_s: list[float] | None = Field(default=None, min_length=3, max_length=3)
    lever_arm_body_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    imu_clock_offset_sec: float | None = None
    accel_bias_body_m_s2: list[float] | None = Field(default=None, min_length=3, max_length=3)
    gravity_world_m_s2: list[float] | None = Field(default=None, min_length=3, max_length=3)
    gyro_scale: list[float] | None = Field(default=None, min_length=3, max_length=3)
    accel_scale: list[float] | None = Field(default=None, min_length=3, max_length=3)
    max_step_translation_m: float = Field(ge=0.0)
    max_step_rotation_deg: float = Field(ge=0.0)
    gradient_norm: float = Field(ge=0.0)
    nonzero_jacobian_blocks: int = Field(ge=0)
    knots: list[ContinuousTimeFittedKnot] = Field(min_length=2)
    provenance: ContinuousTimeFitProvenance

    @model_validator(mode="after")
    def check_result(self) -> ContinuousTimeTrajectoryFitResultArtifact:
        """Require ordered fitted knots with consistent indices."""

        indices = [item.knot_index for item in self.knots]
        if indices != list(range(len(indices))):
            raise ValueError("fitted knot indices must be contiguous from zero")
        timestamps = [item.timestamp_sec for item in self.knots]
        if any(
            right <= left
            for left, right in pairwise(timestamps)
        ):
            raise ValueError("fitted knot timestamps must be strictly increasing")
        return self

    def save(self, path: str | Path) -> None:
        """Save the fit result as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def continuous_time_measurements_json_schema() -> dict[str, Any]:
    """Return the JSON schema for trajectory measurement sets."""

    return ContinuousTimeTrajectoryMeasurements.model_json_schema()


def continuous_time_fit_json_schema() -> dict[str, Any]:
    """Return the JSON schema for trajectory fit results."""

    return ContinuousTimeTrajectoryFitResultArtifact.model_json_schema()


def load_continuous_time_measurements(
    path: str | Path,
) -> ContinuousTimeTrajectoryMeasurements:
    """Load and validate a trajectory measurement set."""

    return ContinuousTimeTrajectoryMeasurements.model_validate(
        read_mapping(Path(path))
    )