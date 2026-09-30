"""Schema-valid GNSS-IMU lever arm composed from GNSS-LiDAR and IMU-LiDAR evidence.

The lever arm is the GNSS antenna position in the IMU frame,
``p_imu = R_lidar_imu^T (p_lidar - t_lidar_imu)``, where ``p_lidar`` is the
GNSS-LiDAR lever arm and ``(R_lidar_imu, t_lidar_imu)`` is ``T_lidar_imu``
from the IMU-LiDAR rotation and translation artifacts.  Each input is pinned
by its SHA-256, and the std of every axis is propagated from the inputs'
reported std.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

GNSS_IMU_LEVER_ARM_SCHEMA_VERSION: Literal["slac.gnss_imu_lever_arm/v0.1"] = (
    "slac.gnss_imu_lever_arm/v0.1"
)


class GnssImuInput(StrictModel):
    """One composed input artifact."""

    role: Literal["gnss_lidar_lever_arm", "imu_lidar_rotation", "imu_lidar_translation"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_status: str


class GnssImuAxisRecord(StrictModel):
    """One axis of the composed lever arm."""

    name: Literal["x", "y", "z"]
    value_m: float
    std_m: float = Field(ge=0.0)
    status: Literal["estimated", "unobservable"]
    reference_m: float | None = None
    error_to_reference_m: float | None = None


class GnssImuLeverArmArtifact(StrictModel):
    """The GNSS antenna in the IMU frame, composed from pinned evidence."""

    schema_version: Literal["slac.gnss_imu_lever_arm/v0.1"] = GNSS_IMU_LEVER_ARM_SCHEMA_VERSION
    method: Literal["composition/v0.1"] = "composition/v0.1"
    policy_status: Literal["pass", "inconclusive"]
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_dofs: list[Literal["x", "y", "z"]]
    lever_arm_m: list[float] = Field(min_length=3, max_length=3)
    axes: list[GnssImuAxisRecord] = Field(min_length=3, max_length=3)
    inputs: list[GnssImuInput] = Field(min_length=3, max_length=3)
    reference: str | None = None
    observable_std_m: float = Field(gt=0.0)
    limitations: list[str] = Field(default_factory=list)
    generator: str
    generator_version: str
    command: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def check_policy(self) -> GnssImuLeverArmArtifact:
        """Keep calibrated axes and the pass verdict honest."""

        estimated = sorted(item.name for item in self.axes if item.status == "estimated")
        if sorted(self.calibrated_dofs) != estimated:
            raise ValueError("calibrated_dofs must list exactly the estimated axes")
        if self.policy_status == "pass" and estimated != ["x", "y", "z"]:
            raise ValueError("a pass verdict requires x, y, and z to be estimated")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def gnss_imu_lever_arm_json_schema() -> dict[str, Any]:
    """Return the GNSS-IMU lever-arm artifact JSON schema."""

    return GnssImuLeverArmArtifact.model_json_schema()


def load_gnss_imu_lever_arm(path: str | Path) -> GnssImuLeverArmArtifact:
    """Load and validate a GNSS-IMU lever-arm artifact."""

    return GnssImuLeverArmArtifact.model_validate(read_mapping(Path(path)))
