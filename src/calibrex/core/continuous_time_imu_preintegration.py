"""Schema-valid continuous-time IMU pre-integration recovery evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import write_mapping
from calibrex.core.result import StrictModel

CONTINUOUS_TIME_IMU_PREINTEGRATION_SCHEMA_VERSION: Literal[
    "slac.continuous_time_imu_preintegration/v0.1"
] = "slac.continuous_time_imu_preintegration/v0.1"


class ContinuousTimeImuPreintegrationProvenance(StrictModel):
    """Seeded synthetic recovery lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    seed: int
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ContinuousTimeImuPreintegrationArtifact(StrictModel):
    """Holdout and known-bad evidence for knot-path IMU rotation factors."""

    schema_version: Literal[
        "slac.continuous_time_imu_preintegration/v0.1"
    ] = CONTINUOUS_TIME_IMU_PREINTEGRATION_SCHEMA_VERSION
    recovery_id: str
    interpolation: Literal["screw_linear"]
    seed: int
    knot_count: int = Field(ge=2)
    train_measurement_count: int = Field(ge=1)
    holdout_measurement_count: int = Field(ge=1)
    fit_status: Literal["converged", "max_iterations", "singular_system"]
    max_knot_rotation_error_rad: float = Field(ge=0.0)
    gyro_bias_error_rad_s: float = Field(ge=0.0)
    train_imu_rotation_rmse_rad: float = Field(ge=0.0)
    holdout_imu_rotation_rmse_rad: float = Field(ge=0.0)
    known_bad_holdout_rmse_rad: float = Field(ge=0.0)
    known_bad_rmse_delta_rad: float
    known_bad_gyro_bias_rad_s: tuple[float, float, float]
    policy_status: Literal["pass", "fail"]
    policy_reason: str
    provenance: ContinuousTimeImuPreintegrationProvenance

    def save(self, path: str | Path) -> None:
        """Save the recovery artifact as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def continuous_time_imu_preintegration_json_schema() -> dict[str, Any]:
    """Return the JSON schema for IMU pre-integration recovery artifacts."""

    return ContinuousTimeImuPreintegrationArtifact.model_json_schema()
