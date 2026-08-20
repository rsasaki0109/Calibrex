"""Schema-valid continuous-time IMU accelerometer-bias and gravity recovery."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import write_mapping
from calibrex.core.result import StrictModel

CONTINUOUS_TIME_IMU_ACCEL_BIAS_SCHEMA_VERSION: Literal[
    "slac.continuous_time_imu_accel_bias/v0.1"
] = "slac.continuous_time_imu_accel_bias/v0.1"


class ContinuousTimeImuAccelBiasProvenance(StrictModel):
    """Seeded synthetic recovery lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    seed: int
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ContinuousTimeImuAccelBiasArtifact(StrictModel):
    """Holdout and known-bad evidence for accelerometer bias and gravity."""

    schema_version: Literal[
        "slac.continuous_time_imu_accel_bias/v0.1"
    ] = CONTINUOUS_TIME_IMU_ACCEL_BIAS_SCHEMA_VERSION
    recovery_id: str
    interpolation: Literal["screw_linear"]
    seed: int
    knot_count: int = Field(ge=2)
    train_measurement_count: int = Field(ge=1)
    holdout_measurement_count: int = Field(ge=1)
    fit_status: Literal["converged", "max_iterations", "singular_system"]
    max_knot_translation_error_m: float = Field(ge=0.0)
    accel_bias_error_m_s2: float = Field(ge=0.0)
    gravity_error_m_s2: float = Field(ge=0.0)
    train_lever_arm_rmse_m_s2: float = Field(ge=0.0)
    holdout_lever_arm_rmse_m_s2: float = Field(ge=0.0)
    known_bad_holdout_rmse_m_s2: float = Field(ge=0.0)
    known_bad_rmse_delta_m_s2: float
    known_bad_accel_bias_body_m_s2: tuple[float, float, float]
    policy_status: Literal["pass", "fail"]
    policy_reason: str
    provenance: ContinuousTimeImuAccelBiasProvenance

    def save(self, path: str | Path) -> None:
        """Save the recovery artifact as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def continuous_time_imu_accel_bias_json_schema() -> dict[str, Any]:
    """Return the JSON schema for IMU accelerometer-bias recovery artifacts."""

    return ContinuousTimeImuAccelBiasArtifact.model_json_schema()
