"""Schema-valid IMU-LiDAR rotation, clock-offset, and gyro-bias evidence.

The rotation is that of ``T_lidar_imu`` (``T_parent_child``: IMU vectors
expressed in the LiDAR frame), reported as xyz Euler angles; the clock offset
follows ``t_imu = t_lidar + dt``.  The translation of ``T_lidar_imu`` is not
estimated by this method and is never implied by it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

IMU_LIDAR_ROTATION_SCHEMA_VERSION: Literal["slac.imu_lidar_rotation/v0.1"] = (
    "slac.imu_lidar_rotation/v0.1"
)
ImuLidarRotationDofName = Literal[
    "roll", "pitch", "yaw", "time_offset", "gyro_bias_x", "gyro_bias_y", "gyro_bias_z"
]
ImuLidarPolicyStatus = Literal["pass", "warn", "fail", "inconclusive"]


class ImuLidarRotationProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    sequence_ids: list[str] = Field(min_length=1)
    stream_profile: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_digest_scope: str
    dataset_license: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ImuLidarKnownBadControl(StrictModel):
    """Held-out chi-square increase after shifting one quantity."""

    amount: float
    unit: Literal["deg", "s"]
    holdout_delta_chi2: float
    detected: bool


class ImuLidarRotationDofRecord(StrictModel):
    """One estimated quantity with its uncertainty and evidence."""

    name: ImuLidarRotationDofName
    unit: Literal["deg", "s", "rad/s"]
    value: float
    std_analytic: float = Field(ge=0.0)
    std_jackknife: float | None = Field(default=None, ge=0.0)
    std_reported: float = Field(ge=0.0)
    status: Literal["estimated", "unobservable"]
    reference_value: float | None = None
    error_to_reference: float | None = None
    known_bad_control: ImuLidarKnownBadControl | None = None

    @model_validator(mode="after")
    def check_reported_std(self) -> ImuLidarRotationDofRecord:
        """Never report less uncertainty than either estimate provides."""

        if self.std_reported + 1e-15 < max(self.std_analytic, self.std_jackknife or 0.0):
            raise ValueError("std_reported must be the larger of the analytic and jackknife std")
        return self


class ImuLidarWindowSummary(StrictModel):
    """How the LiDAR stream was cut into windows and rate intervals."""

    scans_read: int = Field(ge=0)
    odometry_segments: int = Field(ge=0)
    unreliable_registrations: int = Field(ge=0)
    windows: int = Field(ge=0)
    rate_intervals: int = Field(ge=0)
    imu_samples: int = Field(ge=0)


class ImuLidarRotationArtifact(StrictModel):
    """Evidence for one IMU-LiDAR rotation calibration run."""

    schema_version: Literal["slac.imu_lidar_rotation/v0.1"] = IMU_LIDAR_ROTATION_SCHEMA_VERSION
    method: Literal["angular_rate_alignment/v0.1"] = "angular_rate_alignment/v0.1"
    solver_status: Literal["converged", "insufficient_intervals"]
    policy_status: ImuLidarPolicyStatus
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_dofs: list[ImuLidarRotationDofName]
    rotation_quat_xyzw: list[float] | None = Field(default=None, min_length=4, max_length=4)
    time_offset_s: float
    gyro_bias_rps: list[float] = Field(min_length=3, max_length=3)
    dofs: list[ImuLidarRotationDofRecord]
    options: dict[str, Any]
    windows: ImuLidarWindowSummary
    train_windows: int = Field(ge=0)
    holdout_windows: int = Field(ge=0)
    jackknife_fits: int = Field(ge=0)
    train_median_rate_residual_rps: float | None = None
    holdout_median_rate_residual_rps: float | None = None
    reference: str | None = None
    deskew_passes: list[dict[str, Any]] = Field(default_factory=list)
    deskew_feedback_ratio: float | None = Field(
        default=None,
        description=(
            "Fraction of a deliberate extrinsic error, injected into the gyro deskew, "
            "that survives in the estimate: near 0 means the estimate is not steered by "
            "its own deskew model, near 1 means it merely confirms it."
        ),
    )
    limitations: list[str] = Field(default_factory=list)
    provenance: ImuLidarRotationProvenance

    @model_validator(mode="after")
    def check_policy(self) -> ImuLidarRotationArtifact:
        """Keep calibrated DoFs and the pass verdict honest."""

        estimated = sorted(item.name for item in self.dofs if item.status == "estimated")
        if sorted(self.calibrated_dofs) != estimated:
            raise ValueError("calibrated_dofs must list exactly the estimated DoFs")
        if self.policy_status == "pass" and not {"roll", "pitch", "yaw"} <= set(estimated):
            raise ValueError("a pass verdict requires roll, pitch, and yaw to be estimated")
        if self.solver_status != "converged" and self.policy_status in {"pass", "warn"}:
            raise ValueError("an unsolved run cannot pass or warn")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def imu_lidar_rotation_json_schema() -> dict[str, Any]:
    """Return the IMU-LiDAR rotation artifact JSON schema."""

    return ImuLidarRotationArtifact.model_json_schema()


def load_imu_lidar_rotation(path: str | Path) -> ImuLidarRotationArtifact:
    """Load and validate an IMU-LiDAR rotation artifact."""

    return ImuLidarRotationArtifact.model_validate(read_mapping(Path(path)))
