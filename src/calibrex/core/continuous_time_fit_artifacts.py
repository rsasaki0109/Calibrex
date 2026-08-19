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
    provenance: TrajectoryMeasurementProvenance

    @model_validator(mode="after")
    def check_measurements(self) -> ContinuousTimeTrajectoryMeasurements:
        """Require at least one measurement and consistent frame IDs."""

        if not self.pose_measurements and not self.point_measurements:
            raise ValueError("measurements set requires at least one measurement")
        identifiers = [item.measurement_id for item in self.pose_measurements] + [
            item.measurement_id for item in self.point_measurements
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("measurement IDs must be unique")
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
    final_pose_rmse: float | None = Field(default=None, ge=0.0)
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