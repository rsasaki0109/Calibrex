"""Schema-valid continuous-time LiDAR-pair refinement results."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.result import StrictModel, TransformResult

CONTINUOUS_TIME_LIDAR_PAIR_SCHEMA_VERSION: Literal[
    "slac.continuous_time_lidar_pair_result/v0.1"
] = "slac.continuous_time_lidar_pair_result/v0.1"

ContinuousTimeLidarPairArtifactStatus = Literal[
    "converged",
    "max_iterations",
    "insufficient_constraints",
    "rejected",
]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ContinuousTimeLidarPairOptionsArtifact(StrictModel):
    """Profile and fixed-extrinsic solver settings recorded in the result."""

    initial_time_offset_sec: float
    max_abs_time_offset_sec: float = Field(ge=0.0)
    initial_time_step_sec: float = Field(gt=0.0)
    minimum_time_step_sec: float = Field(gt=0.0)
    max_iterations: int = Field(ge=1)
    min_correspondences: int = Field(ge=1)
    max_odometry_extrapolation_s: float = Field(ge=0.0)
    max_source_records: int | None = Field(default=None, ge=1)
    max_target_points_per_split: int | None = Field(default=None, ge=1)
    sampling_seed: int = 0
    holdout_start_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    outlier_policy: Literal["none", "mad"]
    outlier_mad_scale: float = Field(ge=0.0)
    outlier_min_threshold_m: float = Field(ge=0.0)
    outlier_min_inlier_fraction: float = Field(gt=0.0, le=1.0)
    voxel_strategy: Literal["uniform", "adaptive"]
    adaptive_use_uniform_fallback: bool = True
    adaptive_range_reference_m: float = Field(gt=0.0)
    adaptive_range_exponent: float = Field(ge=0.0)
    adaptive_min_voxel_size_m: float = Field(gt=0.0)
    adaptive_max_voxel_size_m: float = Field(gt=0.0)
    adaptive_min_points_per_voxel: int = Field(ge=1)
    correspondence_refinement_iterations: int = Field(ge=0)
    fixed_extrinsic_max_iterations: int = Field(ge=0)
    fixed_extrinsic_robust_loss: Literal["none", "huber"]


class ContinuousTimeLidarPairObservability(StrictModel):
    """Local six-DoF evidence at the selected profile solution."""

    rank: int | None = Field(default=None, ge=0, le=6)
    condition_number: float | None = Field(default=None, ge=0.0)
    weak_directions: list[str] = Field(default_factory=list)
    residual_count: int = Field(default=0, ge=0)


class ContinuousTimeLidarPairIterationArtifact(StrictModel):
    """One recorded clock-profile iteration."""

    iteration: int = Field(ge=1)
    candidate_offsets_sec: list[float] = Field(min_length=1)
    candidate_train_rmse_m: list[float | None] = Field(default_factory=list)
    candidate_holdout_rmse_m: list[float | None] = Field(default_factory=list)
    selected_offset_sec: float
    train_rmse_m: float | None = Field(default=None, ge=0.0)
    holdout_rmse_m: float | None = Field(default=None, ge=0.0)
    accepted: bool


class ContinuousTimeLidarPairProvenance(StrictModel):
    """Input digests and generation metadata."""

    source_paths: list[str] = Field(min_length=2)
    source_sha256: dict[str, str] = Field(min_length=2)
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


class ContinuousTimeLidarPairArtifact(StrictModel):
    """Result of joint profile refinement with an externally supplied trajectory."""

    schema_version: Literal[
        "slac.continuous_time_lidar_pair_result/v0.1"
    ] = CONTINUOUS_TIME_LIDAR_PAIR_SCHEMA_VERSION
    config_path: str
    dataset_path: str
    source_sensor: str
    target_sensor: str
    odometry_topic: str
    variable: str
    trajectory_model: Literal["piecewise_se3_fixed_odometry"]
    time_offset_sign_convention: str
    initial_transform: TransformResult
    refined_transform: TransformResult
    initial_time_offset_sec: float
    estimated_time_offset_sec: float
    initial_train_rmse_m: float | None = Field(default=None, ge=0.0)
    final_train_rmse_m: float | None = Field(default=None, ge=0.0)
    final_holdout_rmse_m: float | None = Field(default=None, ge=0.0)
    train_correspondence_count: int = Field(ge=0)
    holdout_correspondence_count: int = Field(ge=0)
    outlier_rejected_count: int = Field(ge=0)
    observability: ContinuousTimeLidarPairObservability
    status: ContinuousTimeLidarPairArtifactStatus
    reason: str
    options: ContinuousTimeLidarPairOptionsArtifact
    iterations: list[ContinuousTimeLidarPairIterationArtifact] = Field(default_factory=list)
    provenance: ContinuousTimeLidarPairProvenance


def continuous_time_lidar_pair_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for continuous-time LiDAR results."""

    return ContinuousTimeLidarPairArtifact.model_json_schema()
