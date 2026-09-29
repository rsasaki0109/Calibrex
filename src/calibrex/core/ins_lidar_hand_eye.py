"""Schema-valid INS--LiDAR trajectory hand-eye calibration evidence.

The artifact records, per DoF of ``T_ins_lidar`` (plus the clock offset and
the reference trajectory scale): the estimate, its analytic and block-jackknife
standard deviations, whether the data, a declared prior, or nothing constrains
it, whether a known-bad perturbation is detected on held-out time blocks, and,
when the dataset ships one, the difference to a reference calibration.  The
reference calibration is evaluation-only and never enters the fit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

INS_LIDAR_HAND_EYE_SCHEMA_VERSION: Literal["slac.ins_lidar_hand_eye/v0.1"] = (
    "slac.ins_lidar_hand_eye/v0.1"
)
InsLidarDofName = Literal["roll", "pitch", "yaw", "x", "y", "z", "time_offset", "reference_scale"]
InsLidarDofStatus = Literal["estimated", "prior", "unobservable"]
InsLidarPolicyStatus = Literal["pass", "warn", "fail", "inconclusive"]


class InsLidarHandEyeProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    drive_ids: list[str] = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_license: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InsLidarKnownBadControl(StrictModel):
    """Held-out chi-square increase after moving one DoF by a known-bad amount."""

    amount: float
    unit: Literal["deg", "m", "s"]
    holdout_delta_chi2: float
    detected: bool


class InsLidarDofRecord(StrictModel):
    """One DoF of the estimate with its uncertainty and evidence."""

    name: InsLidarDofName
    unit: Literal["deg", "m", "s", "ratio"]
    value: float
    std_analytic: float = Field(ge=0.0)
    std_jackknife: float | None = Field(default=None, ge=0.0)
    std_reported: float = Field(ge=0.0)
    status: InsLidarDofStatus
    reference_value: float | None = None
    error_to_reference: float | None = None
    known_bad_control: InsLidarKnownBadControl | None = None

    @model_validator(mode="after")
    def check_reported_std(self) -> InsLidarDofRecord:
        """Never report less uncertainty than either estimate provides."""

        floor = max(self.std_analytic, self.std_jackknife or 0.0)
        if self.std_reported + 1e-15 < floor:
            raise ValueError("std_reported must be the larger of the analytic and jackknife std")
        return self


class InsLidarResidualSummary(StrictModel):
    """Residual norms of ``(A X)^-1 X B`` on one motion subset."""

    motion_count: int = Field(ge=0)
    rotation_rmse_deg: float | None = None
    translation_rmse_m: float | None = None
    rotation_median_deg: float | None = None
    translation_median_m: float | None = None


class InsLidarOdometrySummary(StrictModel):
    """LiDAR-only odometry quality; the reference never enters it."""

    frame_count: int = Field(ge=0)
    missing_frames: list[str] = Field(default_factory=list)
    registration_count: int = Field(ge=0)
    converged_registrations: int = Field(ge=0)
    median_correspondences: float | None = None
    median_registration_rmse_m: float | None = None


class InsLidarTransform(StrictModel):
    """``T_ins_lidar`` in ``T_parent_child`` convention."""

    parent_frame: str
    child_frame: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)


class InsLidarHandEyeArtifact(StrictModel):
    """Evidence for one INS--LiDAR trajectory hand-eye calibration run."""

    schema_version: Literal["slac.ins_lidar_hand_eye/v0.1"] = INS_LIDAR_HAND_EYE_SCHEMA_VERSION
    method: Literal["joint_trajectory_hand_eye/v0.1"] = "joint_trajectory_hand_eye/v0.1"
    solver_status: Literal["converged", "insufficient_motions"]
    policy_status: InsLidarPolicyStatus
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_dofs: list[InsLidarDofName]
    transform: InsLidarTransform | None = None
    time_offset_s: float
    reference_scale: float
    dofs: list[InsLidarDofRecord]
    options: dict[str, Any]
    odometry: InsLidarOdometrySummary
    block_duration_s: float = Field(gt=0.0)
    train_blocks: list[int]
    holdout_blocks: list[int]
    jackknife_fits: int = Field(ge=0)
    train: InsLidarResidualSummary
    holdout: InsLidarResidualSummary
    reference_calibration: str | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance: InsLidarHandEyeProvenance

    @model_validator(mode="after")
    def check_policy(self) -> InsLidarHandEyeArtifact:
        """Keep calibrated DoFs and the pass verdict honest."""

        estimated = sorted(item.name for item in self.dofs if item.status != "unobservable")
        if sorted(self.calibrated_dofs) != estimated:
            raise ValueError("calibrated_dofs must list exactly the observable or prior DoFs")
        if set(self.train_blocks) & set(self.holdout_blocks):
            raise ValueError("train and holdout blocks must be disjoint")
        extrinsic = {"roll", "pitch", "yaw", "x", "y", "z"}
        unobservable = {item.name for item in self.dofs if item.status == "unobservable"}
        if self.policy_status == "pass" and (unobservable & extrinsic):
            raise ValueError("a pass verdict requires every extrinsic DoF to be constrained")
        if self.solver_status != "converged" and self.policy_status in {"pass", "warn"}:
            raise ValueError("an unsolved run cannot pass or warn")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def ins_lidar_hand_eye_json_schema() -> dict[str, Any]:
    """Return the INS--LiDAR hand-eye artifact JSON schema."""

    return InsLidarHandEyeArtifact.model_json_schema()


def load_ins_lidar_hand_eye(path: str | Path) -> InsLidarHandEyeArtifact:
    """Load and validate an INS--LiDAR hand-eye artifact."""

    return InsLidarHandEyeArtifact.model_validate(read_mapping(Path(path)))
