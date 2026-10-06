"""Schema-valid GNSS antenna lever-arm and clock-offset evidence for a LiDAR.

The lever arm is the GNSS antenna phase centre expressed in the LiDAR frame;
the clock offset follows ``t_gnss = t_lidar + dt``.  Each quantity carries its
analytic and window-jackknife standard deviation, whether the data constrain
it, whether held-out windows detect a known-bad shift, and, when available,
the difference to a reference such as a CAD offset.  The reference never
enters the fit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.excitation import AxisExcitation
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

GNSS_LIDAR_LEVER_ARM_SCHEMA_VERSION: Literal["slac.gnss_lidar_lever_arm/v0.1"] = (
    "slac.gnss_lidar_lever_arm/v0.1"
)
GnssLidarDofName = Literal["x", "y", "z", "time_offset"]
GnssLidarPolicyStatus = Literal["pass", "warn", "fail", "inconclusive"]


class GnssLidarLeverArmProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    sequence_ids: list[str] = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_digest_scope: str
    dataset_license: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GnssLidarKnownBadControl(StrictModel):
    """Held-out chi-square increase after shifting one quantity."""

    amount: float
    unit: Literal["m", "s"]
    holdout_delta_chi2: float
    detected: bool


class GnssLidarDofRecord(StrictModel):
    """One estimated quantity with its uncertainty and evidence."""

    name: GnssLidarDofName
    unit: Literal["m", "s"]
    value: float
    std_analytic: float = Field(ge=0.0)
    std_jackknife: float | None = Field(default=None, ge=0.0)
    std_reported: float = Field(ge=0.0)
    status: Literal["estimated", "unobservable"]
    reference_value: float | None = None
    error_to_reference: float | None = None
    known_bad_control: GnssLidarKnownBadControl | None = None
    excitation: AxisExcitation | None = Field(
        default=None,
        description="translation axes only: why it is (un)observable and what would change it",
    )

    @model_validator(mode="after")
    def check_reported_std(self) -> GnssLidarDofRecord:
        """Never report less uncertainty than either estimate provides."""

        if self.std_reported + 1e-15 < max(self.std_analytic, self.std_jackknife or 0.0):
            raise ValueError("std_reported must be the larger of the analytic and jackknife std")
        return self


class GnssLidarSegmentSummary(StrictModel):
    """How the LiDAR stream was cut into reliable, GNSS-covered windows."""

    scans_read: int = Field(ge=0)
    scans_in_gnss_coverage: int = Field(ge=0)
    odometry_segments: int = Field(ge=0)
    unreliable_registrations: int = Field(ge=0)
    windows: int = Field(ge=0)
    gnss_epochs_used: int = Field(ge=0)
    gnss_epochs_rejected: int = Field(ge=0)


class GnssLidarLeverArmArtifact(StrictModel):
    """Evidence for one GNSS-LiDAR lever-arm calibration run."""

    schema_version: Literal["slac.gnss_lidar_lever_arm/v0.1"] = GNSS_LIDAR_LEVER_ARM_SCHEMA_VERSION
    method: Literal["windowed_variable_projection/v0.1"] = "windowed_variable_projection/v0.1"
    solver_status: Literal["converged", "insufficient_windows"]
    policy_status: GnssLidarPolicyStatus
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_dofs: list[GnssLidarDofName]
    lever_arm_frame: str
    lever_arm_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    time_offset_s: float
    dofs: list[GnssLidarDofRecord]
    options: dict[str, Any]
    segments: GnssLidarSegmentSummary
    train_windows: int = Field(ge=0)
    holdout_windows: int = Field(ge=0)
    jackknife_fits: int = Field(ge=0)
    holdout_median_residual_m: float | None = None
    train_median_residual_m: float | None = None
    reference: str | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance: GnssLidarLeverArmProvenance

    @model_validator(mode="after")
    def check_policy(self) -> GnssLidarLeverArmArtifact:
        """Keep calibrated DoFs and the pass verdict honest."""

        estimated = sorted(item.name for item in self.dofs if item.status == "estimated")
        if sorted(self.calibrated_dofs) != estimated:
            raise ValueError("calibrated_dofs must list exactly the estimated DoFs")
        lever = {"x", "y", "z"}
        if self.policy_status == "pass" and not lever <= set(estimated):
            raise ValueError("a pass verdict requires every lever-arm component to be estimated")
        if self.solver_status != "converged" and self.policy_status in {"pass", "warn"}:
            raise ValueError("an unsolved run cannot pass or warn")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def gnss_lidar_lever_arm_json_schema() -> dict[str, Any]:
    """Return the GNSS-LiDAR lever-arm artifact JSON schema."""

    return GnssLidarLeverArmArtifact.model_json_schema()


def load_gnss_lidar_lever_arm(path: str | Path) -> GnssLidarLeverArmArtifact:
    """Load and validate a GNSS-LiDAR lever-arm artifact."""

    return GnssLidarLeverArmArtifact.model_validate(read_mapping(Path(path)))
