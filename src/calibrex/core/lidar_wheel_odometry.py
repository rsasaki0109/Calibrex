"""Schema-valid LiDAR-to-wheel-odometry calibration evidence.

The rotation is that of ``T_vehicle_lidar`` in the vehicle frame the wheel
odometry reports in (x forward, z up).  The speed scale multiplies the wheel
speed (``v_true = s v_wheel``, a tyre-radius error), the clock offset follows
``t_wheel = t_lidar + dt``, and the lever is the LiDAR's distance ahead of
the axle the vehicle turns about.  Rotation std and reference differences are
small rotations about the vehicle axes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

LIDAR_WHEEL_SCHEMA_VERSION: Literal["slac.lidar_wheel_odometry/v0.1"] = (
    "slac.lidar_wheel_odometry/v0.1"
)
LidarWheelParameter = Literal["roll", "pitch", "yaw", "lever", "speed_scale", "time_offset"]


class LidarWheelProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    sequence_ids: list[str] = Field(min_length=1)
    wheel_source: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_license: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LidarWheelControl(StrictModel):
    """Held-out chi-square increase after a known-bad shift."""

    amount: float
    unit: Literal["deg", "m", "ratio", "s"]
    holdout_delta_chi2: float
    detected: bool


class LidarWheelRecord(StrictModel):
    """One estimated parameter."""

    name: LidarWheelParameter
    unit: Literal["deg", "m", "ratio", "s"]
    value: float
    std_analytic: float = Field(ge=0.0)
    std_jackknife: float | None = Field(default=None, ge=0.0)
    std_reported: float = Field(ge=0.0)
    status: Literal["estimated", "unobservable"]
    known_bad_control: LidarWheelControl | None = None

    @model_validator(mode="after")
    def check_reported_std(self) -> LidarWheelRecord:
        """Never report less uncertainty than either estimate provides."""

        if self.std_reported + 1e-15 < max(self.std_analytic, self.std_jackknife or 0.0):
            raise ValueError("std_reported must be the larger of the analytic and jackknife std")
        return self


class LidarWheelReference(StrictModel):
    """A reference rotation and the estimate's difference from it."""

    name: str
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    difference_deg: list[float] = Field(min_length=3, max_length=3)


class LidarWheelOdometryArtifact(StrictModel):
    """Evidence for one LiDAR-to-wheel-odometry calibration run."""

    schema_version: Literal["slac.lidar_wheel_odometry/v0.1"] = LIDAR_WHEEL_SCHEMA_VERSION
    method: Literal["wheel_odometry_motion/v0.1"] = "wheel_odometry_motion/v0.1"
    solver_status: Literal["converged", "insufficient_motions"]
    policy_status: Literal["pass", "warn", "fail", "inconclusive"]
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_parameters: list[LidarWheelParameter]
    rotation_quat_xyzw: list[float] | None = Field(default=None, min_length=4, max_length=4)
    lever_m: float | None = None
    speed_scale: float | None = None
    time_offset_s: float | None = None
    parameters: list[LidarWheelRecord]
    references: list[LidarWheelReference] = Field(default_factory=list)
    motions: int = Field(ge=0)
    train_motions: int = Field(ge=0)
    holdout_motions: int = Field(ge=0)
    jackknife_fits: int = Field(ge=0)
    train_median_normalized_residual: float | None = None
    holdout_median_normalized_residual: float | None = None
    options: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)
    provenance: LidarWheelProvenance

    @model_validator(mode="after")
    def check_policy(self) -> LidarWheelOdometryArtifact:
        """Keep calibrated parameters and the pass verdict honest."""

        estimated = sorted(item.name for item in self.parameters if item.status == "estimated")
        if sorted(self.calibrated_parameters) != estimated:
            raise ValueError("calibrated_parameters must list exactly the estimated parameters")
        needed = {"roll", "pitch", "yaw", "speed_scale", "time_offset"}
        if self.policy_status == "pass" and not needed <= set(estimated):
            raise ValueError(
                "a pass verdict requires roll, pitch, yaw, speed_scale, and time_offset"
            )
        if self.solver_status != "converged" and self.policy_status in {"pass", "warn"}:
            raise ValueError("an unsolved run cannot pass or warn")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def lidar_wheel_odometry_json_schema() -> dict[str, Any]:
    """Return the LiDAR-wheel-odometry artifact JSON schema."""

    return LidarWheelOdometryArtifact.model_json_schema()


def load_lidar_wheel_odometry(path: str | Path) -> LidarWheelOdometryArtifact:
    """Load and validate a LiDAR-wheel-odometry artifact."""

    return LidarWheelOdometryArtifact.model_validate(read_mapping(Path(path)))
