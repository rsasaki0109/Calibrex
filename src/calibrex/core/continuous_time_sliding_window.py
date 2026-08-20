"""Schema-valid sliding-window marginalization recovery evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import write_mapping
from calibrex.core.result import StrictModel

CONTINUOUS_TIME_SLIDING_WINDOW_SCHEMA_VERSION: Literal[
    "slac.continuous_time_sliding_window/v0.1"
] = "slac.continuous_time_sliding_window/v0.1"


class ContinuousTimeSlidingWindowProvenance(StrictModel):
    """Seeded synthetic recovery lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    seed: int
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ContinuousTimeSlidingWindowArtifact(StrictModel):
    """Holdout and known-bad evidence for sliding-window marginalization."""

    schema_version: Literal[
        "slac.continuous_time_sliding_window/v0.1"
    ] = CONTINUOUS_TIME_SLIDING_WINDOW_SCHEMA_VERSION
    recovery_id: str
    interpolation: Literal["screw_linear"]
    seed: int
    knot_count: int = Field(ge=2)
    window_count: int = Field(ge=2)
    overlap_knot_count: int = Field(ge=1)
    train_measurement_count: int = Field(ge=1)
    holdout_measurement_count: int = Field(ge=1)
    batch_fit_status: Literal["converged", "max_iterations", "singular_system"]
    sliding_fit_status: Literal["converged", "max_iterations", "singular_system"]
    max_overlap_translation_error_m: float = Field(ge=0.0)
    max_overlap_rotation_error_rad: float = Field(ge=0.0)
    batch_holdout_rmse_m: float = Field(ge=0.0)
    sliding_holdout_rmse_m: float = Field(ge=0.0)
    known_bad_overlap_translation_error_m: float = Field(ge=0.0)
    known_bad_rmse_delta_m: float
    known_bad_skip_marginalization: bool
    policy_status: Literal["pass", "fail"]
    policy_reason: str
    provenance: ContinuousTimeSlidingWindowProvenance

    def save(self, path: str | Path) -> None:
        """Save the recovery artifact as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def continuous_time_sliding_window_json_schema() -> dict[str, Any]:
    """Return the JSON schema for sliding-window recovery artifacts."""

    return ContinuousTimeSlidingWindowArtifact.model_json_schema()
