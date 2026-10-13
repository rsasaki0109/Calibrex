"""Schema-valid LiDAR-LiDAR extrinsic evidence from map-based registration.

The extrinsic is ``T_reference_target`` (``T_parent_child``): it maps target
LiDAR points into the reference LiDAR frame.  Rotation DoFs are reported as
xyz Euler angles of that rotation; their std and differences to a reference
are small rotations about the reference LiDAR's axes, which stay meaningful
for any mounting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

LIDAR_LIDAR_EXTRINSIC_SCHEMA_VERSION: Literal["slac.lidar_lidar_extrinsic/v0.1"] = (
    "slac.lidar_lidar_extrinsic/v0.1"
)
LidarLidarDofName = Literal["roll", "pitch", "yaw", "x", "y", "z"]
LidarLidarPolicyStatus = Literal["pass", "warn", "fail", "inconclusive"]


class LidarLidarProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    sequence_ids: list[str] = Field(min_length=1)
    reference_topic: str
    target_topic: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_digest_scope: str
    dataset_license: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LidarLidarControl(StrictModel):
    """Held-out chi-square increase after shifting one DoF."""

    amount: float
    unit: Literal["deg", "m"]
    holdout_delta_chi2: float
    detected: bool


class LidarLidarDofRecord(StrictModel):
    """One DoF of ``T_reference_target`` with its uncertainty and evidence."""

    name: LidarLidarDofName
    unit: Literal["deg", "m"]
    value: float
    std_analytic: float = Field(ge=0.0)
    std_jackknife: float | None = Field(default=None, ge=0.0)
    std_reported: float = Field(ge=0.0)
    status: Literal["estimated", "unobservable"]
    reference_value: float | None = None
    error_to_reference: float | None = None
    known_bad_control: LidarLidarControl | None = None

    @model_validator(mode="after")
    def check_reported_std(self) -> LidarLidarDofRecord:
        """Never report less uncertainty than either estimate provides."""

        if self.std_reported + 1e-15 < max(self.std_analytic, self.std_jackknife or 0.0):
            raise ValueError("std_reported must be the larger of the analytic and jackknife std")
        return self


class LidarLidarTransform(StrictModel):
    """``T_reference_target`` as translation and quaternion."""

    parent_frame: str
    child_frame: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)


class LidarLidarSamples(StrictModel):
    """How the target scans were sampled and matched."""

    reference_scans: int = Field(ge=0)
    samples: int = Field(ge=0)
    train_samples: int = Field(ge=0)
    holdout_samples: int = Field(ge=0)
    train_correspondences: int = Field(ge=0)
    holdout_correspondences: int = Field(ge=0)


class LidarLidarExtrinsicArtifact(StrictModel):
    """Evidence for one LiDAR-LiDAR extrinsic calibration run."""

    schema_version: Literal["slac.lidar_lidar_extrinsic/v0.1"] = (
        LIDAR_LIDAR_EXTRINSIC_SCHEMA_VERSION
    )
    method: Literal["map_registration/v0.1"] = "map_registration/v0.1"
    solver_status: Literal["converged", "max_iterations", "insufficient_samples"]
    policy_status: LidarLidarPolicyStatus
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_dofs: list[LidarLidarDofName]
    transform: LidarLidarTransform | None = None
    dofs: list[LidarLidarDofRecord]
    residual_sigma_m: float | None = None
    holdout_median_residual_m: float | None = None
    reference_holdout_delta_chi2: float | None = Field(
        default=None,
        description=(
            "Held-out chi-square increase when the reference transform replaces the "
            "estimate; large values mean the held-out data reject the reference."
        ),
    )
    samples: LidarLidarSamples
    train_blocks: list[int] = Field(default_factory=list)
    holdout_blocks: list[int] = Field(default_factory=list)
    jackknife_fits: int = Field(ge=0)
    options: dict[str, Any]
    reference: str | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance: LidarLidarProvenance

    @model_validator(mode="after")
    def check_policy(self) -> LidarLidarExtrinsicArtifact:
        """Keep calibrated DoFs and the pass verdict honest."""

        estimated = sorted(item.name for item in self.dofs if item.status == "estimated")
        if sorted(self.calibrated_dofs) != estimated:
            raise ValueError("calibrated_dofs must list exactly the estimated DoFs")
        if self.policy_status == "pass" and len(estimated) != 6:
            raise ValueError("a pass verdict requires all six DoFs to be estimated")
        if self.solver_status == "insufficient_samples" and self.policy_status in {"pass", "warn"}:
            raise ValueError("an unsolved run cannot pass or warn")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def lidar_lidar_extrinsic_json_schema() -> dict[str, Any]:
    """Return the LiDAR-LiDAR extrinsic artifact JSON schema."""

    return LidarLidarExtrinsicArtifact.model_json_schema()


def load_lidar_lidar_extrinsic(path: str | Path) -> LidarLidarExtrinsicArtifact:
    """Load and validate a LiDAR-LiDAR extrinsic artifact."""

    return LidarLidarExtrinsicArtifact.model_validate(read_mapping(Path(path)))
