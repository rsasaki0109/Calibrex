"""Schema-valid cross-dataset evidence for solid-state LiDAR calibration."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.result import Grade, StrictModel

SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION_V0_1: Literal[
    "slac.solid_state_cross_dataset_benchmark_config/v0.1"
] = "slac.solid_state_cross_dataset_benchmark_config/v0.1"
SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION: Literal[
    "slac.solid_state_cross_dataset_benchmark_config/v0.2"
] = "slac.solid_state_cross_dataset_benchmark_config/v0.2"
SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION_V0_1: Literal[
    "slac.solid_state_cross_dataset_benchmark/v0.1"
] = "slac.solid_state_cross_dataset_benchmark/v0.1"
SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION: Literal[
    "slac.solid_state_cross_dataset_benchmark/v0.2"
] = "slac.solid_state_cross_dataset_benchmark/v0.2"

SolidStateCrossDatasetBenchmarkConfigSchemaVersion = Literal[
    "slac.solid_state_cross_dataset_benchmark_config/v0.1",
    "slac.solid_state_cross_dataset_benchmark_config/v0.2",
]
SolidStateCrossDatasetBenchmarkSchemaVersion = Literal[
    "slac.solid_state_cross_dataset_benchmark/v0.1",
    "slac.solid_state_cross_dataset_benchmark/v0.2",
]

BenchmarkReferenceMode = Literal[
    "absolute_extrinsic",
    "identity_control",
    "trajectory_only",
    "none",
]
BenchmarkDatasetStatus = Literal["scored", "inconclusive", "failed", "unavailable"]
BenchmarkWinner = Literal["adaptive", "uniform", "tie", "inconclusive"]
BenchmarkFailureCategory = Literal[
    "none",
    "max_iterations",
    "insufficient_constraints",
    "rejected",
    "missing_result",
    "invalid_artifact",
    "rank_deficient",
    "insufficient_holdout",
    "not_run",
    "unknown",
]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class SolidStateCrossDatasetBenchmarkProtocol(StrictModel):
    """The controls and primary metric shared by each dataset comparison."""

    name: str
    solver: Literal["continuous_time_lidar_pair"]
    variants: list[str] = Field(min_length=2)
    same_capture_windows: bool = True
    same_temporal_holdout: bool = True
    same_solver_budget: bool = True
    bounded_replay_allowed: bool = True
    bounded_replay_note: str
    primary_metric: Literal["final_holdout_rmse_m"] = "final_holdout_rmse_m"
    lower_is_better: bool = True
    minimum_holdout_correspondences: int = Field(default=6, ge=1)
    split_ids: list[str] = Field(default_factory=lambda: ["default"], min_length=1)
    holdout_start_fractions: dict[str, float] = Field(
        default_factory=lambda: {"default": 0.8}
    )
    seed_values: list[int] = Field(default_factory=lambda: [0], min_length=1)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_samples: int = Field(default=2000, ge=100)
    bootstrap_seed: int = 0
    minimum_scored_replicates: int = Field(default=1, ge=1)
    require_converged: bool = False

    @field_validator("holdout_start_fractions")
    @classmethod
    def validate_holdout_start_fractions(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(not 0.0 < fraction < 1.0 for fraction in value.values()):
            raise ValueError("holdout_start_fractions must map names to fractions in (0, 1)")
        return value


class SolidStateCrossDatasetBenchmarkSpecReplicate(StrictModel):
    """One explicitly materialized repeated evaluation declared by the user."""

    id: str
    split_id: str
    seed: int
    ablation_manifest_path: str
    notes: list[str] = Field(default_factory=list)


class SolidStateCrossDatasetBenchmarkSpecDataset(StrictModel):
    """One input declaration for the cross-dataset aggregator."""

    id: str
    name: str
    family: str
    dataset_manifest_path: str
    config_path: str
    ablation_manifest_path: str
    reference_mode: BenchmarkReferenceMode
    absolute_extrinsic_ground_truth: bool = False
    identity_control: bool = False
    independent_temporal_holdout: bool = True
    replicates: list[SolidStateCrossDatasetBenchmarkSpecReplicate] = Field(
        default_factory=list
    )
    notes: list[str] = Field(default_factory=list)


class SolidStateCrossDatasetBenchmarkSpec(StrictModel):
    """User-authored cross-dataset benchmark declaration."""

    schema_version: SolidStateCrossDatasetBenchmarkConfigSchemaVersion = (
        SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION
    )
    protocol: SolidStateCrossDatasetBenchmarkProtocol
    datasets: list[SolidStateCrossDatasetBenchmarkSpecDataset] = Field(min_length=1)


class SolidStateCrossDatasetBenchmarkVariant(StrictModel):
    """The comparable metrics copied from one continuous-time artifact."""

    id: str
    voxel_strategy: Literal["uniform", "adaptive"]
    outlier_policy: Literal["none", "mad"]
    result_path: str | None = None
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    optimization_status: str
    quality_grade: Grade | None = None
    estimated_time_offset_sec: float | None = None
    final_train_rmse_m: float | None = Field(default=None, ge=0.0)
    final_holdout_rmse_m: float | None = Field(default=None, ge=0.0)
    train_correspondence_count: int = Field(default=0, ge=0)
    holdout_correspondence_count: int = Field(default=0, ge=0)
    observability_rank: int | None = Field(default=None, ge=0, le=6)
    outlier_rejected_count: int = Field(default=0, ge=0)
    reason: str | None = None
    failure_category: BenchmarkFailureCategory = "none"


class SolidStateCrossDatasetBenchmarkConfidenceInterval(StrictModel):
    """A deterministic bootstrap interval for a reported benchmark statistic."""

    confidence_level: float = Field(gt=0.0, lt=1.0)
    lower: float
    upper: float
    statistic: Literal["mean_holdout_improvement_percent"]


class SolidStateCrossDatasetBenchmarkReplicate(StrictModel):
    """One paired split/seed comparison used by the dataset aggregate."""

    id: str
    split_id: str
    seed: int
    ablation_manifest_path: str
    ablation_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-fA-F]{64}$"
    )
    status: BenchmarkDatasetStatus
    variants: list[SolidStateCrossDatasetBenchmarkVariant] = Field(min_length=1)
    holdout_improvement_percent: float | None = None
    winner: BenchmarkWinner = "inconclusive"
    failure_categories: list[BenchmarkFailureCategory] = Field(default_factory=list)
    comparison_note: str
    notes: list[str] = Field(default_factory=list)


class SolidStateCrossDatasetBenchmarkDataset(StrictModel):
    """One dataset-level comparison and its evidence classification."""

    id: str
    name: str
    family: str
    dataset_manifest_path: str
    dataset_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-fA-F]{64}$"
    )
    config_path: str
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    ablation_manifest_path: str
    ablation_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-fA-F]{64}$"
    )
    reference_mode: BenchmarkReferenceMode
    absolute_extrinsic_ground_truth: bool = False
    identity_control: bool = False
    independent_temporal_holdout: bool = True
    status: BenchmarkDatasetStatus
    variants: list[SolidStateCrossDatasetBenchmarkVariant] = Field(min_length=1)
    adaptive_variant_id: str | None = None
    uniform_variant_id: str | None = None
    holdout_improvement_percent: float | None = None
    winner: BenchmarkWinner = "inconclusive"
    comparison_note: str
    replicates: list[SolidStateCrossDatasetBenchmarkReplicate] = Field(
        default_factory=list
    )
    replicate_count: int = Field(default=0, ge=0)
    scored_replicate_count: int = Field(default=0, ge=0)
    adaptive_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    holdout_improvement_mean_percent: float | None = None
    holdout_improvement_median_percent: float | None = None
    holdout_improvement_ci: SolidStateCrossDatasetBenchmarkConfidenceInterval | None = None
    failure_categories: list[BenchmarkFailureCategory] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SolidStateCrossDatasetBenchmarkAggregate(StrictModel):
    """Summary counts retained alongside the per-dataset table."""

    dataset_count: int = Field(ge=0)
    scored_dataset_count: int = Field(ge=0)
    inconclusive_dataset_count: int = Field(ge=0)
    adaptive_wins: int = Field(ge=0)
    uniform_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    mean_holdout_improvement_percent: float | None = None
    median_holdout_improvement_percent: float | None = None
    total_replicate_count: int = Field(default=0, ge=0)
    scored_replicate_count: int = Field(default=0, ge=0)
    adaptive_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    holdout_improvement_ci: SolidStateCrossDatasetBenchmarkConfidenceInterval | None = None
    failure_categories: list[BenchmarkFailureCategory] = Field(default_factory=list)
    conclusion: str


class SolidStateCrossDatasetBenchmarkProvenance(StrictModel):
    """Digests for the declarations and evidence manifests used to aggregate."""

    source_paths: list[str] = Field(min_length=1)
    source_sha256: dict[str, str] = Field(min_length=1)
    tool_name: str = "tools/run_solid_state_cross_dataset_benchmark.py"
    tool_version: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if _SHA256_RE.fullmatch(digest) is None]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class SolidStateCrossDatasetBenchmarkManifest(StrictModel):
    """Reproducible cross-dataset comparison artifact."""

    schema_version: SolidStateCrossDatasetBenchmarkSchemaVersion = (
        SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION
    )
    tool: str = "tools/run_solid_state_cross_dataset_benchmark.py"
    tool_version: str
    spec_path: str
    spec_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    protocol: SolidStateCrossDatasetBenchmarkProtocol
    datasets: list[SolidStateCrossDatasetBenchmarkDataset] = Field(min_length=1)
    aggregate: SolidStateCrossDatasetBenchmarkAggregate
    provenance: SolidStateCrossDatasetBenchmarkProvenance


def solid_state_cross_dataset_benchmark_config_json_schema() -> dict[str, Any]:
    """Return the standalone schema for the benchmark declaration."""

    return SolidStateCrossDatasetBenchmarkSpec.model_json_schema()


def solid_state_cross_dataset_benchmark_json_schema() -> dict[str, Any]:
    """Return the standalone schema for the generated benchmark result."""

    return SolidStateCrossDatasetBenchmarkManifest.model_json_schema()
