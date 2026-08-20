"""Schema-valid continuous-time IMU diagonal intrinsics recovery evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import write_mapping
from calibrex.core.result import StrictModel

CONTINUOUS_TIME_IMU_INTRINSICS_SCHEMA_VERSION: Literal[
    "slac.continuous_time_imu_intrinsics/v0.1"
] = "slac.continuous_time_imu_intrinsics/v0.1"


class ContinuousTimeImuIntrinsicsProvenance(StrictModel):
    """Seeded synthetic recovery lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    seed: int
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ContinuousTimeImuIntrinsicsArtifact(StrictModel):
    """Holdout and known-bad evidence for diagonal IMU scale factors."""

    schema_version: Literal[
        "slac.continuous_time_imu_intrinsics/v0.1"
    ] = CONTINUOUS_TIME_IMU_INTRINSICS_SCHEMA_VERSION
    recovery_id: str
    interpolation: Literal["screw_linear"]
    seed: int
    knot_count: int = Field(ge=2)
    train_gyro_measurement_count: int = Field(ge=1)
    train_accel_measurement_count: int = Field(ge=1)
    holdout_gyro_measurement_count: int = Field(ge=1)
    holdout_accel_measurement_count: int = Field(ge=1)
    fit_status: Literal["converged", "max_iterations", "singular_system"]
    gyro_scale_error: float = Field(ge=0.0)
    accel_scale_error: float = Field(ge=0.0)
    train_imu_rotation_rmse_rad: float = Field(ge=0.0)
    holdout_imu_rotation_rmse_rad: float = Field(ge=0.0)
    train_lever_arm_rmse_m_s2: float = Field(ge=0.0)
    holdout_lever_arm_rmse_m_s2: float = Field(ge=0.0)
    known_bad_holdout_imu_rmse_rad: float = Field(ge=0.0)
    known_bad_imu_rmse_delta_rad: float
    known_bad_gyro_scale_delta: tuple[float, float, float]
    policy_status: Literal["pass", "fail"]
    policy_reason: str
    provenance: ContinuousTimeImuIntrinsicsProvenance

    def save(self, path: str | Path) -> None:
        """Save the recovery artifact as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def continuous_time_imu_intrinsics_json_schema() -> dict[str, Any]:
    """Return the JSON schema for IMU intrinsics recovery artifacts."""

    return ContinuousTimeImuIntrinsicsArtifact.model_json_schema()
