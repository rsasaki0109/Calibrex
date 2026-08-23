"""Schema-valid development-only Camera--LiDAR provider support comparison."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from calibrex.core.camera_lidar_confidence_calibration import (
    CameraLidarConfidenceCalibrationCandidate,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
)
from calibrex.core.result import StrictModel

CAMERA_LIDAR_PROVIDER_SUPPORT_COMPARISON_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_provider_support_comparison/v0.1"
] = "slac.camera_lidar_provider_support_comparison/v0.1"


class CameraLidarProviderSupportProtocol(StrictModel):
    """Common development protocol required from every compared calibration."""

    dataset_id: str
    sequence_id: str
    split_id: Literal["development"] = "development"
    problem_id: str
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_dataset_ids_excluded: list[str] = Field(min_length=1)
    confidence_thresholds: list[float] = Field(min_length=2)
    calibration_holdout_ratio: float = Field(gt=0.0, lt=1.0)
    calibration_split_seeds: list[int] = Field(min_length=1)
    minimum_frame_correspondence_count: int = Field(ge=1)
    minimum_frame_support_rate: float = Field(gt=0.0, le=1.0)
    minimum_full_rank_frame_rate: float = Field(gt=0.0, le=1.0)
    development_reprojection_inlier_threshold_px: float = Field(gt=0.0)
    minimum_geometric_inlier_rate: float = Field(gt=0.0, le=1.0)
    minimum_frame_geometric_inlier_count: int = Field(ge=1)
    minimum_frame_geometric_support_rate: float = Field(gt=0.0, le=1.0)
    reference_pose_used_for_development_calibration: Literal[True] = True
    runtime_holdout_used_for_pose_selection: Literal[False] = False
    evaluation_data_used_for_selection: Literal[False] = False

    @model_validator(mode="after")
    def check_lists(self) -> CameraLidarProviderSupportProtocol:
        """Keep frozen threshold, seed, and exclusion sets deterministic."""

        if self.confidence_thresholds != sorted(self.confidence_thresholds):
            raise ValueError("provider comparison thresholds must be ascending")
        if len(self.confidence_thresholds) != len(set(self.confidence_thresholds)):
            raise ValueError("provider comparison thresholds must be unique")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.confidence_thresholds
        ):
            raise ValueError("provider comparison thresholds must be finite in [0, 1]")
        if len(self.calibration_split_seeds) != len(
            set(self.calibration_split_seeds)
        ):
            raise ValueError("provider comparison split seeds must be unique")
        if len(self.evaluation_dataset_ids_excluded) != len(
            set(self.evaluation_dataset_ids_excluded)
        ):
            raise ValueError("provider comparison exclusions must be unique")
        if self.dataset_id in self.evaluation_dataset_ids_excluded:
            raise ValueError("development dataset cannot be an excluded evaluation set")
        return self


class CameraLidarProviderSupportCandidate(StrictModel):
    """One provider calibration and its support-first diagnostic operating point."""

    candidate_id: str
    calibration_id: str
    calibration_path: str
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correspondence_artifact_id: str
    correspondence_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: CorrespondenceProviderIdentity
    calibration_status: Literal["locked", "rejected"]
    passing_thresholds: list[float] = Field(default_factory=list)
    runtime_eligible: bool
    diagnostic: CameraLidarConfidenceCalibrationCandidate
    diagnostic_selection_rule_id: Literal[
        "selected_lock_or_geometry_then_support/v0.1"
    ] = "selected_lock_or_geometry_then_support/v0.1"
    worst_split_geometric_inlier_count: int = Field(ge=0)
    diagnostic_rank: int = Field(ge=1)
    pareto_dominated_by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_eligibility(self) -> CameraLidarProviderSupportCandidate:
        """Never make a rejected confidence calibration runtime eligible."""

        expected_eligible = self.calibration_status == "locked"
        if self.runtime_eligible != expected_eligible:
            raise ValueError("runtime eligibility differs from calibration status")
        if expected_eligible != bool(self.passing_thresholds):
            raise ValueError("locked providers require passing confidence thresholds")
        train = self.diagnostic.minimum_train_geometric_inlier_count_across_seeds
        holdout = self.diagnostic.minimum_holdout_geometric_inlier_count_across_seeds
        if train is None or holdout is None:
            raise ValueError("provider comparison requires geometric diagnostics")
        if self.worst_split_geometric_inlier_count != min(train, holdout):
            raise ValueError("worst split geometric count differs from diagnostics")
        return self


class CameraLidarProviderSupportComparisonProvenance(StrictModel):
    """Exact calibration sources and generator identity."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(min_length=2)
    source_sha256: dict[str, str] = Field(min_length=2)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def check_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        """Require lowercase SHA-256 provenance."""

        invalid = [
            key
            for key, digest in value.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(f"invalid provider comparison SHA-256 values: {invalid}")
        return value


class CameraLidarProviderSupportComparisonArtifact(StrictModel):
    """Development-only comparison that cannot select a rejected provider."""

    schema_version: Literal[
        "slac.camera_lidar_provider_support_comparison/v0.1"
    ] = CAMERA_LIDAR_PROVIDER_SUPPORT_COMPARISON_SCHEMA_VERSION
    comparison_id: str
    status: Literal["locked", "rejected"]
    protocol: CameraLidarProviderSupportProtocol
    selection_rule_id: Literal[
        "runtime_eligible_then_geometry_support/v0.1"
    ] = "runtime_eligible_then_geometry_support/v0.1"
    candidates: list[CameraLidarProviderSupportCandidate] = Field(min_length=2)
    selected_candidate_id: str | None = None
    diagnostic_leader_candidate_id: str
    release_sota_claim_allowed: Literal[False] = False
    provenance: CameraLidarProviderSupportComparisonProvenance

    @model_validator(mode="after")
    def check_decision(self) -> CameraLidarProviderSupportComparisonArtifact:
        """Bind status, runtime selection, ranks, and diagnostic leader."""

        identifiers = [item.candidate_id for item in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("provider comparison candidate IDs must be unique")
        ranks = sorted(item.diagnostic_rank for item in self.candidates)
        if ranks != list(range(1, len(self.candidates) + 1)):
            raise ValueError("provider comparison ranks must be contiguous")
        leader = next(
            (item for item in self.candidates if item.diagnostic_rank == 1), None
        )
        if leader is None or leader.candidate_id != self.diagnostic_leader_candidate_id:
            raise ValueError("diagnostic leader must be the rank-one candidate")
        eligible = {item.candidate_id for item in self.candidates if item.runtime_eligible}
        if self.status == "rejected":
            if eligible or self.selected_candidate_id is not None:
                raise ValueError("rejected provider comparison cannot select runtime input")
        elif self.selected_candidate_id not in eligible:
            raise ValueError("locked provider comparison must select an eligible candidate")
        for item in self.candidates:
            invalid = set(item.pareto_dominated_by) - set(identifiers)
            if invalid or item.candidate_id in item.pareto_dominated_by:
                raise ValueError("invalid provider comparison dominance reference")
        return self

    def save(self, path: str | Path) -> None:
        """Save the provider support comparison as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def load_camera_lidar_provider_support_comparison(
    path: str | Path,
) -> CameraLidarProviderSupportComparisonArtifact:
    """Load and validate a provider support comparison."""

    return CameraLidarProviderSupportComparisonArtifact.model_validate(
        read_mapping(Path(path))
    )


def camera_lidar_provider_support_comparison_json_schema() -> dict[str, Any]:
    """Return the provider support comparison JSON schema."""

    return CameraLidarProviderSupportComparisonArtifact.model_json_schema()
