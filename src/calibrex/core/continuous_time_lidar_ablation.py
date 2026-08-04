"""Schema-valid ablation evidence for continuous-time LiDAR refinement."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.result import Grade, StrictModel

CONTINUOUS_TIME_LIDAR_ABLATION_SCHEMA_VERSION: Literal[
    "slac.continuous_time_lidar_ablation/v0.1"
] = "slac.continuous_time_lidar_ablation/v0.1"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ContinuousTimeLidarAblationPolicy(StrictModel):
    """Declared controls shared by all ablation variants."""

    same_capture_windows: bool = True
    same_temporal_holdout: bool = True
    same_solver_budget: bool = True
    baseline_definition: str


class ContinuousTimeLidarAblationVariant(StrictModel):
    """One schema-valid continuous-time variant result."""

    id: str
    config_path: str
    config_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    result_path: str | None = None
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    voxel_strategy: Literal["uniform", "adaptive"]
    outlier_policy: Literal["none", "mad"]
    status: str
    estimated_time_offset_sec: float | None = None
    final_train_rmse_m: float | None = None
    final_holdout_rmse_m: float | None = None
    observability_rank: int | None = Field(default=None, ge=0, le=6)
    outlier_rejected_count: int | None = Field(default=None, ge=0)
    quality_grade: Grade | None = None


class ContinuousTimeLidarAblationManifest(StrictModel):
    """Reproducible comparison table for baseline and SOTA-inspired variants."""

    schema_version: Literal[
        "slac.continuous_time_lidar_ablation/v0.1"
    ] = CONTINUOUS_TIME_LIDAR_ABLATION_SCHEMA_VERSION
    tool: str
    tool_version: str
    base_config_path: str
    base_config_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    policy: ContinuousTimeLidarAblationPolicy
    variants: list[ContinuousTimeLidarAblationVariant] = Field(min_length=1)

    @field_validator("base_config_sha256")
    @classmethod
    def validate_base_hash(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("base_config_sha256 must be a SHA-256 digest")
        return value


def continuous_time_lidar_ablation_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for the ablation manifest."""

    return ContinuousTimeLidarAblationManifest.model_json_schema()
