"""Schema-valid Camera--LiDAR correspondence quality diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.probabilistic_correspondence import CorrespondenceProviderIdentity
from calibrex.core.result import StrictModel, TransformResult

CAMERA_LIDAR_CORRESPONDENCE_QUALITY_SCHEMA_VERSION: Literal[
    "slac.camera_lidar_correspondence_quality/v0.1"
] = "slac.camera_lidar_correspondence_quality/v0.1"

CameraLidarQualityGrade = Literal["pass", "warn", "fail"]
CameraLidarQualityPartition = Literal["train", "holdout"]
CameraLidarQualityPoseRole = Literal["initializer", "candidate", "selected"]


class CameraLidarDistributionSummary(StrictModel):
    """Deterministic scalar distribution summary."""

    count: int = Field(ge=0)
    minimum: float | None = None
    p10: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    maximum: float | None = None
    mean: float | None = None

    @model_validator(mode="after")
    def check_empty_state(self) -> CameraLidarDistributionSummary:
        """Require scalar values exactly when the summary is non-empty."""

        values = (
            self.minimum,
            self.p10,
            self.median,
            self.p90,
            self.p95,
            self.maximum,
            self.mean,
        )
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("an empty distribution cannot contain summary values")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("a non-empty distribution requires all summary values")
        return self


class CameraLidarCorrespondenceGateCounts(StrictModel):
    """Mutually exclusive correspondence outcomes at one tested pose."""

    total_count: int = Field(ge=0)
    below_confidence_count: int = Field(ge=0)
    behind_camera_count: int = Field(ge=0)
    invalid_projection_count: int = Field(ge=0)
    outside_image_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)

    @model_validator(mode="after")
    def check_partition(self) -> CameraLidarCorrespondenceGateCounts:
        """Require every correspondence to have exactly one gate outcome."""

        classified = (
            self.below_confidence_count
            + self.behind_camera_count
            + self.invalid_projection_count
            + self.outside_image_count
            + self.accepted_count
        )
        if classified != self.total_count:
            raise ValueError("correspondence gate counts do not sum to total_count")
        return self


class CameraLidarImageCoverage(StrictModel):
    """Accepted projection coverage on a fixed image grid."""

    grid_columns: int = Field(ge=1)
    grid_rows: int = Field(ge=1)
    occupied_cell_count: int = Field(ge=0)
    occupancy_ratio: float = Field(ge=0.0, le=1.0)
    normalized_bounding_box_area: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_occupancy(self) -> CameraLidarImageCoverage:
        """Keep occupied cells and ratio consistent."""

        cell_count = self.grid_columns * self.grid_rows
        if self.occupied_cell_count > cell_count:
            raise ValueError("occupied image cells exceed the configured grid")
        expected = self.occupied_cell_count / cell_count
        if abs(self.occupancy_ratio - expected) > 1.0e-12:
            raise ValueError("image occupancy ratio does not match occupied cells")
        return self


class CameraLidarPoseObservability(StrictModel):
    """Scale-normalized local six-DoF information diagnostics."""

    parameter_order: list[Literal["rx", "ry", "rz", "tx", "ty", "tz"]]
    parameter_scales: list[float] = Field(min_length=6, max_length=6)
    factor_count: int = Field(ge=0)
    singular_values: list[float] = Field(min_length=6, max_length=6)
    rank: int = Field(ge=0, le=6)
    condition_number: float | None = Field(default=None, ge=1.0)
    axis_information_fraction: dict[str, float]
    weak_axes: list[Literal["rx", "ry", "rz", "tx", "ty", "tz"]]

    @model_validator(mode="after")
    def check_axes(self) -> CameraLidarPoseObservability:
        """Require one deterministic entry for every pose axis."""

        expected = ["rx", "ry", "rz", "tx", "ty", "tz"]
        if self.parameter_order != expected:
            raise ValueError("pose observability parameter order is not canonical")
        if set(self.axis_information_fraction) != set(expected):
            raise ValueError("axis information fractions must cover all six axes")
        if any(not 0.0 <= value <= 1.0 for value in self.axis_information_fraction.values()):
            raise ValueError("axis information fractions must be in [0, 1]")
        if self.rank < 6 and self.condition_number is not None:
            raise ValueError("rank-deficient information cannot declare a condition number")
        return self


class CameraLidarCorrespondenceFrameQuality(StrictModel):
    """Pose-dependent quality evidence for one capture frame."""

    frame_id: str
    capture_time_ns: int
    partition: CameraLidarQualityPartition
    gates: CameraLidarCorrespondenceGateCounts
    confidence: CameraLidarDistributionSummary
    accepted_confidence: CameraLidarDistributionSummary
    lidar_range_m: CameraLidarDistributionSummary
    accepted_camera_depth_m: CameraLidarDistributionSummary
    reprojection_error_px: CameraLidarDistributionSummary
    image_coverage: CameraLidarImageCoverage
    observability: CameraLidarPoseObservability
    grade: CameraLidarQualityGrade
    gate_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_frame_counts(self) -> CameraLidarCorrespondenceFrameQuality:
        """Bind distribution counts to the gate counts."""

        if self.confidence.count != self.gates.total_count:
            raise ValueError("confidence count differs from correspondence total")
        if self.lidar_range_m.count != self.gates.total_count:
            raise ValueError("LiDAR range count differs from correspondence total")
        accepted_count = self.gates.accepted_count
        for summary in (
            self.accepted_confidence,
            self.accepted_camera_depth_m,
            self.reprojection_error_px,
        ):
            if summary.count != accepted_count:
                raise ValueError("accepted distribution count differs from gate count")
        if self.grade == "pass" and self.gate_reasons:
            raise ValueError("a passing frame cannot contain gate reasons")
        if self.grade != "pass" and not self.gate_reasons:
            raise ValueError("a non-passing frame requires gate reasons")
        return self


class CameraLidarCorrespondenceQualitySummary(StrictModel):
    """Aggregate support, rejection, and observability statistics."""

    frame_count: int = Field(ge=1)
    train_frame_count: int = Field(ge=0)
    holdout_frame_count: int = Field(ge=0)
    total_correspondence_count: int = Field(ge=0)
    accepted_correspondence_count: int = Field(ge=0)
    accepted_correspondence_rate: float = Field(ge=0.0, le=1.0)
    train_accepted_correspondence_count: int = Field(ge=0)
    holdout_accepted_correspondence_count: int = Field(ge=0)
    minimum_train_correspondence_count: int = Field(ge=0)
    minimum_holdout_correspondence_count: int = Field(ge=0)
    frames_meeting_minimum_count: int = Field(ge=0)
    frames_meeting_minimum_rate: float = Field(ge=0.0, le=1.0)
    full_rank_frame_count: int = Field(ge=0)
    aggregate_observability: CameraLidarPoseObservability
    rejection_counts: dict[str, int]
    grade: CameraLidarQualityGrade
    gate_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_summary(self) -> CameraLidarCorrespondenceQualitySummary:
        """Check aggregate count and grade invariants."""

        if self.train_frame_count + self.holdout_frame_count != self.frame_count:
            raise ValueError("train and holdout frame counts do not sum to frame_count")
        if self.accepted_correspondence_count > self.total_correspondence_count:
            raise ValueError("accepted correspondences exceed total correspondences")
        if self.frames_meeting_minimum_count > self.frame_count:
            raise ValueError("frames meeting minimum exceed frame_count")
        if self.full_rank_frame_count > self.frame_count:
            raise ValueError("full-rank frame count exceeds frame_count")
        expected_rate = (
            self.accepted_correspondence_count / self.total_correspondence_count
            if self.total_correspondence_count
            else 0.0
        )
        if abs(self.accepted_correspondence_rate - expected_rate) > 1.0e-12:
            raise ValueError("accepted correspondence rate does not match counts")
        expected_frame_rate = self.frames_meeting_minimum_count / self.frame_count
        if abs(self.frames_meeting_minimum_rate - expected_frame_rate) > 1.0e-12:
            raise ValueError("frame support rate does not match counts")
        if self.grade == "pass" and self.gate_reasons:
            raise ValueError("a passing summary cannot contain gate reasons")
        if self.grade != "pass" and not self.gate_reasons:
            raise ValueError("a non-passing summary requires gate reasons")
        return self


class CameraLidarCorrespondenceQualityProvenance(StrictModel):
    """Digest-bound correspondence and refinement-result lineage."""

    source_paths: list[str] = Field(min_length=2)
    source_sha256: dict[str, str] = Field(min_length=2)
    generator: str = "calibrex.evaluation.camera_lidar_correspondence_quality"
    generator_version: str = "v0.1"
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [
            key
            for key, digest in value.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class CameraLidarCorrespondenceQualityArtifact(StrictModel):
    """Post-hoc, pose-dependent quality report for probabilistic correspondences."""

    schema_version: Literal[
        "slac.camera_lidar_correspondence_quality/v0.1"
    ] = CAMERA_LIDAR_CORRESPONDENCE_QUALITY_SCHEMA_VERSION
    report_id: str
    dataset_id: str
    split_id: str
    correspondence_artifact_id: str
    correspondence_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    refinement_result_id: str
    refinement_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: CorrespondenceProviderIdentity
    pose_role: CameraLidarQualityPoseRole
    transform_camera_lidar: TransformResult
    holdout_used_for_selection: Literal[False] = False
    analysis_scope: Literal["posthoc_all_result_frames"] = "posthoc_all_result_frames"
    minimum_confidence: float = Field(ge=0.0, le=1.0)
    use_covariance: bool
    use_outlier_probability: bool
    use_reliability: bool
    minimum_frame_correspondence_count: int = Field(ge=1)
    summary: CameraLidarCorrespondenceQualitySummary
    frames: list[CameraLidarCorrespondenceFrameQuality] = Field(min_length=1)
    provenance: CameraLidarCorrespondenceQualityProvenance

    @model_validator(mode="after")
    def check_frames(self) -> CameraLidarCorrespondenceQualityArtifact:
        """Require one unique frame report and matching summary totals."""

        frame_ids = [item.frame_id for item in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("correspondence quality frame IDs must be unique")
        if self.summary.frame_count != len(self.frames):
            raise ValueError("quality summary frame_count differs from frames")
        return self

    def save(self, path: str | Path) -> None:
        """Save this report as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def load_camera_lidar_correspondence_quality(
    path: str | Path,
) -> CameraLidarCorrespondenceQualityArtifact:
    """Load and validate a Camera--LiDAR correspondence quality report."""

    return CameraLidarCorrespondenceQualityArtifact.model_validate(
        read_mapping(Path(path))
    )


def camera_lidar_correspondence_quality_json_schema() -> dict[str, Any]:
    """Return the standalone JSON Schema for quality reports."""

    return CameraLidarCorrespondenceQualityArtifact.model_json_schema()
