"""Schema-valid development-only Camera--LiDAR confidence calibration."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from calibrex.core.camera_lidar_correspondence_quality import (
    CameraLidarDistributionSummary,
    CameraLidarPoseObservability,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.probabilistic_correspondence import CorrespondenceProviderIdentity
from calibrex.core.result import StrictModel

CAMERA_LIDAR_CONFIDENCE_CALIBRATION_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_confidence_calibration/v0.2"
] = "slac.camera_lidar_confidence_calibration/v0.2"


class CameraLidarConfidenceCalibrationCandidate(StrictModel):
    """Support, geometry, and observability gates for one confidence threshold."""

    minimum_confidence: float = Field(ge=0.0, le=1.0)
    accepted_correspondence_count: int = Field(ge=0)
    accepted_correspondence_rate: float = Field(ge=0.0, le=1.0)
    frames_meeting_minimum_count: int = Field(ge=0)
    frames_meeting_minimum_rate: float = Field(ge=0.0, le=1.0)
    full_rank_frame_count: int = Field(ge=0)
    full_rank_frame_rate: float = Field(ge=0.0, le=1.0)
    minimum_train_correspondence_count_across_seeds: int = Field(ge=0)
    minimum_holdout_correspondence_count_across_seeds: int = Field(ge=0)
    aggregate_observability: CameraLidarPoseObservability
    geometric_inlier_count: int | None = Field(default=None, ge=0)
    geometric_inlier_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    frames_meeting_geometric_minimum_count: int | None = Field(
        default=None, ge=0
    )
    frames_meeting_geometric_minimum_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    minimum_train_geometric_inlier_count_across_seeds: int | None = Field(
        default=None, ge=0
    )
    minimum_holdout_geometric_inlier_count_across_seeds: int | None = Field(
        default=None, ge=0
    )
    reprojection_error_px: CameraLidarDistributionSummary | None = None
    gate_pass: bool
    gate_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_gate(self) -> CameraLidarConfidenceCalibrationCandidate:
        """Keep the pass bit equivalent to an empty hard-gate reason list."""

        if self.gate_pass == bool(self.gate_reasons):
            raise ValueError("gate_pass must be true exactly when gate_reasons is empty")
        geometry = (
            self.geometric_inlier_count,
            self.geometric_inlier_rate,
            self.frames_meeting_geometric_minimum_count,
            self.frames_meeting_geometric_minimum_rate,
            self.minimum_train_geometric_inlier_count_across_seeds,
            self.minimum_holdout_geometric_inlier_count_across_seeds,
            self.reprojection_error_px,
        )
        if any(item is not None for item in geometry) and any(
            item is None for item in geometry
        ):
            raise ValueError("candidate geometric diagnostics are incomplete")
        if self.geometric_inlier_count is not None:
            if self.geometric_inlier_count > self.accepted_correspondence_count:
                raise ValueError("geometric inliers exceed accepted correspondences")
            expected_rate = (
                self.geometric_inlier_count / self.accepted_correspondence_count
                if self.accepted_correspondence_count
                else 0.0
            )
            if not math.isclose(
                self.geometric_inlier_rate or 0.0,
                expected_rate,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("geometric inlier rate does not match counts")
        return self


class CameraLidarLockedRefinementOptions(StrictModel):
    """Exact provider and solver thresholds frozen after development selection."""

    minimum_confidence: float = Field(ge=0.0, le=1.0)
    holdout_ratio: float = Field(gt=0.0, lt=1.0)
    evaluation_split_seeds: list[int] = Field(min_length=1)
    minimum_train_correspondences: int = Field(ge=4)
    minimum_holdout_correspondences: int = Field(ge=4)
    rotation_bound_deg: float = Field(gt=0.0)
    translation_bound_m: float = Field(gt=0.0)
    initial_rotation_step_deg: float = Field(gt=0.0)
    initial_translation_step_m: float = Field(gt=0.0)
    minimum_rotation_step_deg: float = Field(gt=0.0)
    minimum_translation_step_m: float = Field(gt=0.0)
    max_evaluations: int = Field(ge=13)
    cauchy_scale: float = Field(gt=0.0)
    use_covariance: Literal[True] = True
    use_outlier_probability: Literal[True] = True
    use_reliability: Literal[True] = True
    acceptance_policy_id: Literal[
        "initializer_preserving_fit_only/v0.1"
    ] = "initializer_preserving_fit_only/v0.1"
    minimum_absolute_train_objective_improvement: float = Field(ge=0.0)
    minimum_relative_train_objective_improvement: float = Field(ge=0.0)
    maximum_train_correspondence_loss_fraction: float = Field(ge=0.0, lt=1.0)
    maximum_accepted_bound_fraction: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def check_steps(self) -> CameraLidarLockedRefinementOptions:
        """Require optimizer steps to stay within their correction bounds."""

        if not (
            self.minimum_rotation_step_deg
            <= self.initial_rotation_step_deg
            <= self.rotation_bound_deg
        ):
            raise ValueError("locked rotation steps and bound are inconsistent")
        if not (
            self.minimum_translation_step_m
            <= self.initial_translation_step_m
            <= self.translation_bound_m
        ):
            raise ValueError("locked translation steps and bound are inconsistent")
        if len(self.evaluation_split_seeds) != len(
            set(self.evaluation_split_seeds)
        ):
            raise ValueError("evaluation split seeds must be unique")
        return self


class CameraLidarConfidenceCalibrationProvenance(StrictModel):
    """Inputs and generator identity for a development threshold lock."""

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
        invalid = [
            key
            for key, digest in value.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(f"invalid source SHA-256 values: {invalid}")
        return value


class CameraLidarConfidenceCalibrationArtifact(StrictModel):
    """Development-only sweep and immutable runtime option selection."""

    schema_version: Literal[
        "slac.camera_lidar_confidence_calibration/v0.1",
        "slac.camera_lidar_confidence_calibration/v0.2",
    ] = CAMERA_LIDAR_CONFIDENCE_CALIBRATION_SCHEMA_VERSION
    status: Literal["locked", "rejected"] | None = None
    calibration_id: str
    dataset_id: str
    sequence_id: str
    split_id: Literal["development"]
    problem_id: str
    problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correspondence_artifact_id: str
    correspondence_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: CorrespondenceProviderIdentity
    evaluation_dataset_ids_excluded: list[str] = Field(min_length=1)
    selection_rule_id: Literal[
        "highest_threshold_passing_support_and_observability/v0.1",
        "highest_threshold_passing_support_observability_and_geometry/v0.2",
    ] = "highest_threshold_passing_support_and_observability/v0.1"
    calibration_holdout_ratio: float = Field(gt=0.0, lt=1.0)
    calibration_split_seeds: list[int] = Field(min_length=1)
    minimum_frame_correspondence_count: int = Field(ge=1)
    minimum_frame_support_rate: float = Field(gt=0.0, le=1.0)
    minimum_full_rank_frame_rate: float = Field(gt=0.0, le=1.0)
    development_reprojection_inlier_threshold_px: float | None = Field(
        default=None, gt=0.0
    )
    minimum_geometric_inlier_rate: float | None = Field(
        default=None, gt=0.0, le=1.0
    )
    minimum_frame_geometric_inlier_count: int | None = Field(
        default=None, ge=1
    )
    minimum_frame_geometric_support_rate: float | None = Field(
        default=None, gt=0.0, le=1.0
    )
    candidates: list[CameraLidarConfidenceCalibrationCandidate] = Field(
        min_length=2
    )
    selected_minimum_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    locked_refinement_options: CameraLidarLockedRefinementOptions | None = None
    reference_pose_used_for_development_calibration: Literal[True] = True
    runtime_holdout_used_for_pose_selection: Literal[False] = False
    release_sota_claim_allowed: Literal[False] = False
    provenance: CameraLidarConfidenceCalibrationProvenance

    @model_validator(mode="after")
    def check_selection(self) -> CameraLidarConfidenceCalibrationArtifact:
        """Require a deterministic highest-passing threshold and no eval leakage."""

        thresholds = [item.minimum_confidence for item in self.candidates]
        if thresholds != sorted(thresholds) or len(thresholds) != len(set(thresholds)):
            raise ValueError("confidence candidates must be unique and ascending")
        passing = [item.minimum_confidence for item in self.candidates if item.gate_pass]
        is_v2 = self.schema_version.endswith("/v0.2")
        if is_v2:
            if self.status is None:
                raise ValueError("v0.2 confidence calibration requires status")
            if self.selection_rule_id != (
                "highest_threshold_passing_support_observability_and_geometry/v0.2"
            ):
                raise ValueError("v0.2 confidence calibration requires geometry rule")
            geometric_settings = (
                self.development_reprojection_inlier_threshold_px,
                self.minimum_geometric_inlier_rate,
                self.minimum_frame_geometric_inlier_count,
                self.minimum_frame_geometric_support_rate,
            )
            if any(item is None for item in geometric_settings):
                raise ValueError("v0.2 geometric gate settings are incomplete")
            if any(
                item.geometric_inlier_count is None for item in self.candidates
            ):
                raise ValueError("v0.2 candidates require geometric diagnostics")
        elif self.status is not None:
            raise ValueError("v0.1 confidence calibration cannot declare status")
        if self.status == "rejected":
            if passing:
                raise ValueError("a rejected calibration cannot have passing candidates")
            if (
                self.selected_minimum_confidence is not None
                or self.locked_refinement_options is not None
            ):
                raise ValueError("a rejected calibration cannot contain a runtime lock")
        else:
            if not passing:
                raise ValueError("a locked calibration requires a passing candidate")
            if (
                self.selected_minimum_confidence is None
                or self.locked_refinement_options is None
            ):
                raise ValueError("a locked calibration requires selected runtime options")
            if not math.isclose(
                self.selected_minimum_confidence,
                max(passing),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "selected confidence is not the highest passing candidate"
                )
            if not math.isclose(
                self.locked_refinement_options.minimum_confidence,
                self.selected_minimum_confidence,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("locked confidence differs from selected confidence")
        if len(self.calibration_split_seeds) != len(
            set(self.calibration_split_seeds)
        ):
            raise ValueError("calibration split seeds must be unique")
        if self.dataset_id in self.evaluation_dataset_ids_excluded:
            raise ValueError("development dataset cannot be listed as excluded evaluation")
        if len(self.evaluation_dataset_ids_excluded) != len(
            set(self.evaluation_dataset_ids_excluded)
        ):
            raise ValueError("excluded evaluation dataset IDs must be unique")
        return self

    def save(self, path: str | Path) -> None:
        """Save this calibration lock as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def load_camera_lidar_confidence_calibration(
    path: str | Path,
) -> CameraLidarConfidenceCalibrationArtifact:
    """Load and validate a Camera--LiDAR development threshold lock."""

    return CameraLidarConfidenceCalibrationArtifact.model_validate(
        read_mapping(Path(path))
    )


def camera_lidar_confidence_calibration_json_schema() -> dict[str, Any]:
    """Return the standalone JSON Schema for development threshold locks."""

    return CameraLidarConfidenceCalibrationArtifact.model_json_schema()
