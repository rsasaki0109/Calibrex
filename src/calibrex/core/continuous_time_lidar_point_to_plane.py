"""Schema-valid continuous-time LiDAR point-to-plane recovery evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import write_mapping
from calibrex.core.result import StrictModel

CONTINUOUS_TIME_LIDAR_POINT_TO_PLANE_SCHEMA_VERSION: Literal[
    "slac.continuous_time_lidar_point_to_plane/v0.1"
] = "slac.continuous_time_lidar_point_to_plane/v0.1"


class ContinuousTimeLidarPointToPlaneProvenance(StrictModel):
    """Seeded synthetic recovery lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    seed: int
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ContinuousTimeLidarPointToPlaneArtifact(StrictModel):
    """Holdout and known-bad evidence for knot-path point-to-plane factors."""

    schema_version: Literal[
        "slac.continuous_time_lidar_point_to_plane/v0.1"
    ] = CONTINUOUS_TIME_LIDAR_POINT_TO_PLANE_SCHEMA_VERSION
    recovery_id: str
    interpolation: Literal["screw_linear"]
    seed: int
    knot_count: int = Field(ge=2)
    train_measurement_count: int = Field(ge=1)
    holdout_measurement_count: int = Field(ge=1)
    fit_status: Literal["converged", "max_iterations", "singular_system"]
    max_knot_error: float = Field(ge=0.0)
    train_point_to_plane_rmse_m: float = Field(ge=0.0)
    holdout_point_to_plane_rmse_m: float = Field(ge=0.0)
    known_bad_holdout_rmse_m: float = Field(ge=0.0)
    known_bad_rmse_delta_m: float
    known_bad_translation_m: tuple[float, float, float]
    policy_status: Literal["pass", "fail"]
    policy_reason: str
    provenance: ContinuousTimeLidarPointToPlaneProvenance

    def save(self, path: str | Path) -> None:
        """Save the recovery artifact as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def continuous_time_lidar_point_to_plane_json_schema() -> dict[str, Any]:
    """Return the JSON schema for LiDAR point-to-plane recovery artifacts."""

    return ContinuousTimeLidarPointToPlaneArtifact.model_json_schema()
