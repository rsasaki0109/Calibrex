"""Schema-valid probabilistic 2D--3D correspondence artifacts.

Learned feature extraction stays outside the Calibrex core.  The core accepts
only explicit correspondence distributions, provider lineage, and immutable
provenance so a PnP implementation can be evaluated without importing a
training framework.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel, TransformResult
from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference

PROBABILISTIC_CORRESPONDENCE_SCHEMA_VERSION: Literal[
    "slac.probabilistic_correspondence/v0.1"
] = "slac.probabilistic_correspondence/v0.1"
PROBABILISTIC_PNP_RESULT_SCHEMA_VERSION: Literal[
    "slac.probabilistic_pnp_result/v0.1"
] = "slac.probabilistic_pnp_result/v0.1"
PROBABILISTIC_REFINEMENT_RESULT_SCHEMA_VERSION: Literal[
    "slac.probabilistic_refinement_result/v0.1"
] = "slac.probabilistic_refinement_result/v0.1"


class CorrespondenceTrainingDeclaration(StrictModel):
    """Training-corpus relationship for a learned correspondence provider."""

    training_datasets: list[str] = Field(min_length=1)
    evaluation_overlap: Literal["none", "possible", "confirmed", "unknown"]
    test_data_used_for_online_refinement: bool
    evidence: str


class CorrespondenceProviderIdentity(StrictModel):
    """Identity and redistribution boundary of the external provider."""

    provider: str
    model: str
    version: str
    source_repository: str
    source_commit: str
    license_spdx: str
    checkpoint: DepthFileReference | None = None
    checkpoint_redistribution: str
    training: CorrespondenceTrainingDeclaration | None = None


class ProbabilisticImageCorrespondence(StrictModel):
    """One LiDAR point and its image-plane probability distribution."""

    correspondence_id: str
    point_lidar_m: list[float] = Field(min_length=3, max_length=3)
    image_mean_px: list[float] = Field(min_length=2, max_length=2)
    image_covariance_px2: list[float] = Field(min_length=4, max_length=4)
    outlier_probability: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_finite_distribution(self) -> ProbabilisticImageCorrespondence:
        """Require finite coordinates and a positive-definite covariance."""

        values = (
            self.point_lidar_m
            + self.image_mean_px
            + self.image_covariance_px2
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("correspondence coordinates and covariance must be finite")
        a, b, c, d = self.image_covariance_px2
        if not math.isclose(b, c, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError("image covariance must be symmetric")
        if a <= 0.0 or d <= 0.0 or a * d - b * c <= 0.0:
            raise ValueError("image covariance must be positive definite")
        return self


class ProbabilisticCorrespondenceFrame(StrictModel):
    """One synchronized set of probabilistic image/LiDAR correspondences."""

    frame_id: str
    capture_time_ns: int
    camera_frame: str
    lidar_frame: str
    intrinsics: DepthCameraIntrinsics
    correspondences: list[ProbabilisticImageCorrespondence] = Field(min_length=4)

    @model_validator(mode="after")
    def check_unique_ids(self) -> ProbabilisticCorrespondenceFrame:
        """Require unique correspondence IDs within a frame."""

        identifiers = [item.correspondence_id for item in self.correspondences]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("correspondence IDs must be unique within a frame")
        return self


class ProbabilisticCorrespondenceProvenance(StrictModel):
    """Generator, command, and input lineage for correspondence output."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def check_input_digests(self) -> ProbabilisticCorrespondenceProvenance:
        """Require all declared input hashes to be lowercase SHA-256 values."""

        invalid = [
            name
            for name, digest in self.input_sha256.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(
                "invalid provenance SHA-256 for inputs: " + ", ".join(invalid)
            )
        return self


class ProbabilisticCorrespondenceArtifact(StrictModel):
    """Provider-neutral correspondence means, uncertainty, and reliability."""

    schema_version: Literal[
        "slac.probabilistic_correspondence/v0.1"
    ] = PROBABILISTIC_CORRESPONDENCE_SCHEMA_VERSION
    artifact_id: str
    dataset_id: str
    split_id: str
    dataset_license_spdx: str | None = None
    provider: CorrespondenceProviderIdentity
    frames: list[ProbabilisticCorrespondenceFrame] = Field(min_length=1)
    provenance: ProbabilisticCorrespondenceProvenance
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_unique_frames(self) -> ProbabilisticCorrespondenceArtifact:
        """Require unique frame IDs."""

        identifiers = [item.frame_id for item in self.frames]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("probabilistic correspondence frame IDs must be unique")
        return self

    def save(self, path: str | Path) -> None:
        """Save this artifact as schema-valid YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


class ProbabilisticPnpToolIdentity(StrictModel):
    """External pose-estimator identity and license boundary."""

    name: str
    version: str | None = None
    source_url: str
    api_reference: str
    license_spdx: str
    execution_mode: Literal["optional_import", "subprocess", "container"]


class ProbabilisticPnpResultProvenance(StrictModel):
    """Immutable input and execution lineage for a PnP result."""

    generator: str
    generator_version: str
    command: list[str] = Field(default_factory=list)
    correspondence_artifact_id: str
    correspondence_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initialization_artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ProbabilisticPnpResultArtifact(StrictModel):
    """Schema-valid output of uncertainty-aware robust pose estimation."""

    schema_version: Literal[
        "slac.probabilistic_pnp_result/v0.1"
    ] = PROBABILISTIC_PNP_RESULT_SCHEMA_VERSION
    result_id: str
    status: Literal[
        "converged",
        "unavailable",
        "failed",
        "insufficient_correspondences",
        "unsupported_camera",
    ]
    method: str
    frame_id: str
    initial_transform_camera_lidar: TransformResult | None = None
    transform_camera_lidar: TransformResult | None = None
    selected_correspondence_count: int = Field(ge=0)
    ransac_inlier_count: int = Field(ge=0)
    probabilistic_inlier_count: int = Field(ge=0)
    weighted_reprojection_rmse_px: float | None = Field(default=None, ge=0.0)
    mean_mahalanobis_error: float | None = Field(default=None, ge=0.0)
    reason: str
    provider: CorrespondenceProviderIdentity
    tool: ProbabilisticPnpToolIdentity
    options: dict[str, int | float | str | bool]
    warnings: list[str] = Field(default_factory=list)
    provenance: ProbabilisticPnpResultProvenance

    @model_validator(mode="after")
    def check_pose_status(self) -> ProbabilisticPnpResultArtifact:
        """Require a pose exactly when the adapter reports convergence."""

        if (self.transform_camera_lidar is not None) != (
            self.status == "converged"
        ):
            raise ValueError("only a converged PnP result may contain a transform")
        if self.ransac_inlier_count > self.selected_correspondence_count:
            raise ValueError("RANSAC inlier count exceeds selected correspondences")
        if self.probabilistic_inlier_count > self.selected_correspondence_count:
            raise ValueError(
                "probabilistic inlier count exceeds selected correspondences"
            )
        return self

    def save(self, path: str | Path) -> None:
        """Save this result as schema-valid YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


class ProbabilisticRefinementEvaluationArtifact(StrictModel):
    """Schema-valid robust multi-frame reprojection evidence."""

    objective: float
    weighted_reprojection_rmse_px: float | None = Field(default=None, ge=0.0)
    mean_mahalanobis_error: float | None = Field(default=None, ge=0.0)
    valid_correspondence_count: int = Field(ge=0)
    skipped_correspondence_count: int = Field(ge=0)


class ProbabilisticRefinementIterationArtifact(StrictModel):
    """One retained bounded pose candidate."""

    evaluation: int = Field(gt=0)
    delta_rotation_deg_xyz: list[float] = Field(min_length=3, max_length=3)
    delta_translation_m_xyz: list[float] = Field(min_length=3, max_length=3)
    objective: float
    accepted: bool


class ProbabilisticRefinementProvenance(StrictModel):
    """D2D initializer and learned correspondence lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    correspondence_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initialization_problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initialization_trace_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ProbabilisticRefinementResultArtifact(StrictModel):
    """Schema-valid multi-frame refinement with explicit initialization."""

    schema_version: Literal[
        "slac.probabilistic_refinement_result/v0.1"
    ] = PROBABILISTIC_REFINEMENT_RESULT_SCHEMA_VERSION
    result_id: str
    status: Literal[
        "converged",
        "max_evaluations",
        "insufficient_correspondences",
        "missing_holdout",
        "at_bound",
    ]
    reason: str
    correspondence_artifact_id: str
    initialization_problem_id: str
    initialization_trace_id: str | None = None
    initialization_trace_status: str | None = None
    initialization_trace_hit: bool | None = None
    initialization_source: Literal[
        "problem_initial_transform",
        "d2d_candidate_trace",
    ] = "problem_initial_transform"
    provider: CorrespondenceProviderIdentity
    initial_transform_camera_lidar: TransformResult
    transform_camera_lidar: TransformResult
    reference_transform_camera_lidar: TransformResult | None = None
    initial_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    final_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    initial_translation_error_m: float | None = Field(default=None, ge=0.0)
    final_translation_error_m: float | None = Field(default=None, ge=0.0)
    train_frame_ids: list[str]
    holdout_frame_ids: list[str]
    initial_train_evaluation: ProbabilisticRefinementEvaluationArtifact
    final_train_evaluation: ProbabilisticRefinementEvaluationArtifact
    initial_holdout_evaluation: ProbabilisticRefinementEvaluationArtifact
    final_holdout_evaluation: ProbabilisticRefinementEvaluationArtifact
    trace: list[ProbabilisticRefinementIterationArtifact]
    options: dict[str, int | float | str | bool]
    method: Literal[
        "d2d_initialized_probabilistic_multiframe_refinement/v0.1",
        "probabilistic_multiframe_refinement/v0.2",
    ] = "probabilistic_multiframe_refinement/v0.2"
    provenance: ProbabilisticRefinementProvenance

    @model_validator(mode="after")
    def check_refinement(self) -> ProbabilisticRefinementResultArtifact:
        """Require disjoint frame evidence and contiguous trace."""

        if set(self.train_frame_ids).intersection(self.holdout_frame_ids):
            raise ValueError("train and holdout frame IDs must be disjoint")
        reference_values = (
            self.reference_transform_camera_lidar,
            self.initial_rotation_error_deg,
            self.final_rotation_error_deg,
            self.initial_translation_error_m,
            self.final_translation_error_m,
        )
        if any(value is not None for value in reference_values) and any(
            value is None for value in reference_values
        ):
            raise ValueError(
                "reference transform and all four pose errors are one block"
            )
        if (
            self.initialization_source == "d2d_candidate_trace"
            and (
                self.initialization_trace_id is None
                or self.initialization_trace_status is None
                or self.initialization_trace_hit is None
                or self.provenance.initialization_trace_sha256 is None
            )
        ):
            raise ValueError(
                "D2D trace initialization requires trace identity and digest"
            )
        if (
            self.initialization_source == "problem_initial_transform"
            and (
                self.initialization_trace_id is not None
                or self.initialization_trace_status is not None
                or self.initialization_trace_hit is not None
                or self.provenance.initialization_trace_sha256 is not None
            )
        ):
            raise ValueError(
                "problem-pose initialization cannot declare a D2D trace"
            )
        evaluations = [item.evaluation for item in self.trace]
        if evaluations and evaluations != list(range(1, len(evaluations) + 1)):
            raise ValueError("probabilistic refinement trace must be contiguous")
        return self

    def save(self, path: str | Path) -> None:
        """Save this refinement result as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def probabilistic_correspondence_json_schema() -> dict[str, Any]:
    """Return the static JSON-schema source for correspondence artifacts."""

    return ProbabilisticCorrespondenceArtifact.model_json_schema()


def probabilistic_pnp_result_json_schema() -> dict[str, Any]:
    """Return the static JSON-schema source for probabilistic PnP results."""

    return ProbabilisticPnpResultArtifact.model_json_schema()


def probabilistic_refinement_result_json_schema() -> dict[str, Any]:
    """Return the static JSON schema for multi-frame refinement results."""

    return ProbabilisticRefinementResultArtifact.model_json_schema()


def load_probabilistic_correspondence(
    path: str | Path,
) -> ProbabilisticCorrespondenceArtifact:
    """Load and validate a probabilistic correspondence artifact."""

    return ProbabilisticCorrespondenceArtifact.model_validate(
        read_mapping(Path(path))
    )


def load_probabilistic_pnp_result(
    path: str | Path,
) -> ProbabilisticPnpResultArtifact:
    """Load and validate a probabilistic PnP result artifact."""

    return ProbabilisticPnpResultArtifact.model_validate(read_mapping(Path(path)))


def load_probabilistic_refinement_result(
    path: str | Path,
) -> ProbabilisticRefinementResultArtifact:
    """Load and validate a multi-frame refinement result."""

    return ProbabilisticRefinementResultArtifact.model_validate(
        read_mapping(Path(path))
    )
