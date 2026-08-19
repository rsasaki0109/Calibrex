"""Schema-valid empirical SE(3) uncertainty evidence.

The core deliberately avoids presenting an optimizer Hessian as a covariance.
This artifact instead records the sampling distribution produced by
deterministic block resampling of a native calibration family: tangent-space
intervals, observed coverage against injected truth, and an overconfident
interval control.  Weak or unobservable directions are retained and labelled
rather than silently dropped.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel, TransformResult

EMPIRICAL_SE3_UNCERTAINTY_SCHEMA_VERSION: Literal[
    "slac.empirical_se3_uncertainty/v0.1"
] = "slac.empirical_se3_uncertainty/v0.1"

TANGENT_AXES: tuple[str, ...] = (
    "translation_x_m",
    "translation_y_m",
    "translation_z_m",
    "rotation_x_deg",
    "rotation_y_deg",
    "rotation_z_deg",
)

UncertaintyAxis = Literal[
    "translation_x_m",
    "translation_y_m",
    "translation_z_m",
    "rotation_x_deg",
    "rotation_y_deg",
    "rotation_z_deg",
]


class EmpiricalUncertaintyProvenance(StrictModel):
    """Generator, command, and immutable input lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(default_factory=dict)
    resample_output_sha256: dict[str, str] = Field(default_factory=dict)
    data_verified: bool
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def check_digests(self) -> EmpiricalUncertaintyProvenance:
        """Require all declared hashes to be lowercase SHA-256 values."""

        sources = list(self.source_sha256.items()) + list(
            self.resample_output_sha256.items()
        )
        invalid = [
            name
            for name, digest in sources
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid:
            raise ValueError(
                "invalid empirical uncertainty SHA-256: " + ", ".join(invalid)
            )
        return self


class EmpiricalBlock(StrictModel):
    """One contiguous temporal block that is never split across splits."""

    block_id: str
    frame_ids: list[str] = Field(min_length=1)
    capture_time_start_ns: int
    capture_time_end_ns: int

    @model_validator(mode="after")
    def check_time_order(self) -> EmpiricalBlock:
        """Require a non-decreasing capture-time range."""

        if self.capture_time_end_ns < self.capture_time_start_ns:
            raise ValueError("block capture-time range must be non-decreasing")
        return self


class EmpiricalResampleIteration(StrictModel):
    """One block resample refit with its retained solver status."""

    iteration_id: str
    status: Literal["success", "failed"]
    solver_status: str
    train_block_ids: list[str] = Field(min_length=1)
    holdout_block_ids: list[str] = Field(min_length=1)
    transform_camera_lidar: TransformResult | None = None
    output_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    reason: str | None = None

    @model_validator(mode="after")
    def check_pose_status(self) -> EmpiricalResampleIteration:
        """Require a transform exactly when the iteration succeeded."""

        if (self.transform_camera_lidar is not None) != (
            self.status == "success"
        ):
            raise ValueError(
                "successful resample iterations must retain their transform"
            )
        return self


class EmpiricalAxisInterval(StrictModel):
    """Central interval for one SE(3) tangent axis."""

    axis: UncertaintyAxis
    unit: str
    lower: float
    upper: float
    half_width: float = Field(ge=0.0)
    weak_direction: bool

    @model_validator(mode="after")
    def check_interval(self) -> EmpiricalAxisInterval:
        """Require an ordered interval consistent with its half width."""

        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("interval bounds must be finite")
        if self.upper < self.lower:
            raise ValueError("interval upper bound must not precede lower")
        if not math.isclose(
            self.half_width, (self.upper - self.lower) / 2.0, rel_tol=1.0e-9
        ):
            raise ValueError("interval half width must equal half the width")
        return self


class OverconfidenceControlResult(StrictModel):
    """Result of applying the reported interval at a shrunken scale."""

    control_id: str
    scale: float = Field(gt=0.0, lt=1.0)
    coverage_translation: float = Field(ge=0.0, le=1.0)
    coverage_rotation: float = Field(ge=0.0, le=1.0)
    coverage_joint: float = Field(ge=0.0, le=1.0)
    refuted: bool
    policy_status: Literal["pass", "fail"]
    reason: str

    @model_validator(mode="after")
    def check_refutation(self) -> OverconfidenceControlResult:
        """Require the refuted flag to match the policy status."""

        if self.refuted and self.policy_status != "pass":
            raise ValueError("refuted overconfidence control must pass its policy")
        if not self.refuted and self.policy_status != "fail":
            raise ValueError(
                "unrefuted overconfidence control must fail its policy"
            )
        return self


class EmpiricalSe3UncertaintyArtifact(StrictModel):
    """Sampling-based SE(3) uncertainty evidence for one calibration family."""

    schema_version: Literal[
        "slac.empirical_se3_uncertainty/v0.1"
    ] = EMPIRICAL_SE3_UNCERTAINTY_SCHEMA_VERSION
    uncertainty_id: str
    calibration_family: str
    transform_convention: Literal["T_parent_child"] = "T_parent_child"
    parent_frame: str
    child_frame: str
    tangent_ordering: list[str] = Field(min_length=6, max_length=6)
    tangent_origin: Literal["resample_mean"] = "resample_mean"
    sampling_unit: Literal["temporal_block"] = "temporal_block"
    block_length: int = Field(ge=1)
    block_count: int = Field(ge=1)
    resampling_method: Literal["seeded_block_subsampling"] = (
        "seeded_block_subsampling"
    )
    resample_count: int = Field(ge=2)
    seed: int
    target_coverage: float = Field(gt=0.0, lt=1.0)
    fit_block_ratio: float = Field(gt=0.0, lt=1.0)
    blocks: list[EmpiricalBlock] = Field(min_length=2)
    iterations: list[EmpiricalResampleIteration] = Field(min_length=2)
    sample_ids: list[str] = Field(min_length=2)
    mean_estimate_transform: TransformResult
    reference_transform: TransformResult | None = None
    reference_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    reference_translation_error_m: float | None = Field(default=None, ge=0.0)
    axis_intervals: list[EmpiricalAxisInterval] = Field(
        min_length=6, max_length=6
    )
    observed_coverage_translation: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    observed_coverage_rotation: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    observed_coverage_joint: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    coverage_score: float | None = Field(default=None, ge=0.0, le=1.0)
    interval_halfwidth_translation_m: float = Field(ge=0.0)
    interval_halfwidth_rotation_deg: float = Field(ge=0.0)
    weak_directions: list[str] = Field(default_factory=list)
    overconfidence_control: OverconfidenceControlResult | None = None
    policy_status: Literal["pass", "warn", "fail", "inconclusive"]
    policy_reason: str
    warnings: list[str] = Field(default_factory=list)
    provenance: EmpiricalUncertaintyProvenance

    @model_validator(mode="after")
    def check_consistency(self) -> EmpiricalSe3UncertaintyArtifact:
        """Enforce the declared tangent convention and policy semantics."""

        if list(self.tangent_ordering) != list(TANGENT_AXES):
            raise ValueError(
                "tangent_ordering must match the frozen SE(3) axis convention"
            )
        interval_axes = [item.axis for item in self.axis_intervals]
        if interval_axes != list(TANGENT_AXES):
            raise ValueError("axis intervals must match the tangent ordering")
        iteration_ids = [item.iteration_id for item in self.iterations]
        if self.sample_ids != iteration_ids:
            raise ValueError("sample_ids must match iteration order")
        if len(iteration_ids) != len(set(iteration_ids)):
            raise ValueError("resample iteration IDs must be unique")
        block_ids = [item.block_id for item in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("temporal block IDs must be unique")
        declared_weak = set(self.weak_directions)
        flagged_weak = {
            item.axis for item in self.axis_intervals if item.weak_direction
        }
        if declared_weak != flagged_weak:
            raise ValueError("weak_directions must match interval flags")
        reference_present = self.reference_transform is not None
        coverage_present = (
            self.observed_coverage_translation is not None
            and self.observed_coverage_rotation is not None
            and self.observed_coverage_joint is not None
            and self.coverage_score is not None
        )
        if reference_present:
            if not coverage_present:
                raise ValueError(
                    "reference truth requires all observed coverage fields"
                )
            assert self.observed_coverage_translation is not None
            assert self.observed_coverage_rotation is not None
            assert self.observed_coverage_joint is not None
            assert self.coverage_score is not None
            expected_score = min(
                self.observed_coverage_translation,
                self.observed_coverage_rotation,
            )
            if not math.isclose(
                self.coverage_score, expected_score, rel_tol=1.0e-9
            ):
                raise ValueError(
                    "coverage_score must be the minimum family coverage"
                )
            if self.overconfidence_control is None:
                raise ValueError(
                    "reference truth requires an overconfidence control"
                )
            if self.policy_status == "inconclusive":
                raise ValueError(
                    "reference truth cannot produce an inconclusive policy"
                )
        else:
            if coverage_present:
                raise ValueError(
                    "observed coverage requires a declared reference transform"
                )
            if self.overconfidence_control is not None:
                raise ValueError(
                    "overconfidence control requires a reference transform"
                )
            if self.policy_status != "inconclusive":
                raise ValueError(
                    "no reference truth forces an inconclusive policy"
                )
            if (
                self.reference_rotation_error_deg is not None
                or self.reference_translation_error_m is not None
            ):
                raise ValueError(
                    "reference errors require a reference transform"
                )
        return self

    def save(self, path: str | Path) -> None:
        """Save this uncertainty artifact as schema-valid YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def empirical_se3_uncertainty_json_schema() -> dict[str, Any]:
    """Return the static JSON schema for empirical SE(3) uncertainty."""

    return EmpiricalSe3UncertaintyArtifact.model_json_schema()


def load_empirical_se3_uncertainty(
    path: str | Path,
) -> EmpiricalSe3UncertaintyArtifact:
    """Load and validate an empirical SE(3) uncertainty artifact."""

    return EmpiricalSe3UncertaintyArtifact.model_validate(read_mapping(Path(path)))