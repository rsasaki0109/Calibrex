"""Schema-valid ground-truth evidence for the solid-state LiDAR pair solver."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field

from calibrex.core.result import StrictModel, TransformResult

SOLID_STATE_SYNTHETIC_BENCHMARK_SCHEMA_VERSION: Literal[
    "slac.solid_state_synthetic_benchmark/v0.1"
] = "slac.solid_state_synthetic_benchmark/v0.1"

SyntheticBenchmarkCaseType = Literal["reference", "known_bad"]
SyntheticBenchmarkExpectedOutcome = Literal["pass", "fail"]
SyntheticBenchmarkConclusion = Literal["pass", "fail", "inconclusive"]


class SolidStateSyntheticBenchmarkProtocol(StrictModel):
    """Frozen construction and evaluation policy for the synthetic scene."""

    name: str
    solver: Literal["continuous_time_lidar_pair"]
    scene: Literal["three_plane_motion_fixture"]
    frame_convention: Literal["T_parent_child"] = "T_parent_child"
    point_time_convention: str
    train_fraction: float = Field(gt=0.0, lt=1.0)
    known_bad_control: str
    notes: list[str] = Field(default_factory=list)


class SolidStateSyntheticBenchmarkThresholds(StrictModel):
    """Absolute parameter-error gates for the reference case."""

    max_rotation_error_deg: float = Field(gt=0.0)
    max_translation_error_m: float = Field(gt=0.0)
    max_time_offset_error_sec: float = Field(gt=0.0)
    require_converged: bool = True


class SolidStateSyntheticBenchmarkCase(StrictModel):
    """One truth-backed reference or deliberately failing control."""

    id: str
    case_type: SyntheticBenchmarkCaseType
    expected_outcome: SyntheticBenchmarkExpectedOutcome
    status: Literal[
        "converged",
        "max_iterations",
        "insufficient_constraints",
        "rejected",
    ]
    initial_transform: TransformResult
    true_transform: TransformResult
    estimated_transform: TransformResult
    true_time_offset_sec: float
    estimated_time_offset_sec: float
    initial_rotation_error_deg: float = Field(ge=0.0)
    final_rotation_error_deg: float = Field(ge=0.0)
    initial_translation_error_m: float = Field(ge=0.0)
    final_translation_error_m: float = Field(ge=0.0)
    initial_time_offset_error_sec: float = Field(ge=0.0)
    final_time_offset_error_sec: float = Field(ge=0.0)
    initial_train_rmse_m: float | None = Field(default=None, ge=0.0)
    final_train_rmse_m: float | None = Field(default=None, ge=0.0)
    final_holdout_rmse_m: float | None = Field(default=None, ge=0.0)
    train_correspondence_count: int = Field(ge=0)
    holdout_correspondence_count: int = Field(ge=0)
    observability_rank: int | None = Field(default=None, ge=0, le=6)
    iterations: int = Field(ge=0)
    gate_passed: bool
    expected_outcome_detected: bool
    failure_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SolidStateSyntheticBenchmarkAggregate(StrictModel):
    """Aggregate gate outcome without hiding individual controls."""

    reference_case_count: int = Field(ge=0)
    reference_pass_count: int = Field(ge=0)
    known_bad_case_count: int = Field(ge=0)
    known_bad_detected_count: int = Field(ge=0)
    all_expected_outcomes_detected: bool
    conclusion: SyntheticBenchmarkConclusion
    notes: list[str] = Field(default_factory=list)


class SolidStateSyntheticBenchmarkProvenance(StrictModel):
    """Digest-bound provenance for generated synthetic evidence."""

    generator: str
    generator_version: str
    source_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    command: list[str] = Field(default_factory=list)
    random_seed: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: list[str] = Field(default_factory=list)


class SolidStateSyntheticBenchmarkArtifact(StrictModel):
    """Result of a deterministic solid-state LiDAR ground-truth benchmark."""

    schema_version: Literal[
        "slac.solid_state_synthetic_benchmark/v0.1"
    ] = SOLID_STATE_SYNTHETIC_BENCHMARK_SCHEMA_VERSION
    benchmark_id: str
    protocol: SolidStateSyntheticBenchmarkProtocol
    thresholds: SolidStateSyntheticBenchmarkThresholds
    cases: list[SolidStateSyntheticBenchmarkCase] = Field(min_length=2)
    aggregate: SolidStateSyntheticBenchmarkAggregate
    provenance: SolidStateSyntheticBenchmarkProvenance


def solid_state_synthetic_benchmark_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for synthetic benchmark artifacts."""

    return SolidStateSyntheticBenchmarkArtifact.model_json_schema()
