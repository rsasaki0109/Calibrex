"""Schema-valid sensor-to-vehicle rotation evidence from non-holonomic motion.

The rotation is that of ``T_vehicle_sensor``: the vehicle frame is x forward,
y left, z up, defined by the vehicle's own motion (no side slip, no vertical
velocity, turns about z).  Rotation std and reference differences are small
rotations about the vehicle axes; the reported roll, pitch, and yaw values
are xyz Euler angles.  The translation is not estimated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

VEHICLE_FRAME_ROTATION_SCHEMA_VERSION: Literal["slac.vehicle_frame_rotation/v0.1"] = (
    "slac.vehicle_frame_rotation/v0.1"
)
VehicleDofName = Literal["roll", "pitch", "yaw"]
VehiclePolicyStatus = Literal["pass", "warn", "fail", "inconclusive"]


class VehicleFrameProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    sequence_ids: list[str] = Field(min_length=1)
    sensor: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_license: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VehicleFrameControl(StrictModel):
    """Held-out chi-square increase after rotating the estimate about one axis."""

    amount_deg: float
    holdout_delta_chi2: float
    detected: bool


class VehicleFrameReference(StrictModel):
    """A reference rotation and the estimate's difference from it."""

    name: str
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    difference_deg: list[float] = Field(
        min_length=3,
        max_length=3,
        description=(
            "small rotation about the vehicle x, y, z axes from the reference to the estimate"
        ),
    )
    holdout_delta_chi2: float | None = None


class VehicleFrameDofRecord(StrictModel):
    """One rotation axis with its uncertainty and evidence."""

    name: VehicleDofName
    unit: Literal["deg"] = "deg"
    value: float
    std_analytic: float = Field(ge=0.0)
    std_jackknife: float | None = Field(default=None, ge=0.0)
    std_reported: float = Field(ge=0.0)
    status: Literal["estimated", "unobservable"]
    known_bad_control: VehicleFrameControl | None = None

    @model_validator(mode="after")
    def check_reported_std(self) -> VehicleFrameDofRecord:
        """Never report less uncertainty than either estimate provides."""

        if self.std_reported + 1e-15 < max(self.std_analytic, self.std_jackknife or 0.0):
            raise ValueError("std_reported must be the larger of the analytic and jackknife std")
        return self


class VehicleFrameRotationArtifact(StrictModel):
    """Evidence for one sensor-to-vehicle rotation calibration run."""

    schema_version: Literal["slac.vehicle_frame_rotation/v0.1"] = (
        VEHICLE_FRAME_ROTATION_SCHEMA_VERSION
    )
    method: Literal["non_holonomic_motion/v0.1"] = "non_holonomic_motion/v0.1"
    sensor_modality: Literal["lidar", "ins", "trajectory"]
    solver_status: Literal["converged", "insufficient_motions"]
    policy_status: VehiclePolicyStatus
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_dofs: list[VehicleDofName]
    rotation_quat_xyzw: list[float] | None = Field(default=None, min_length=4, max_length=4)
    lever_m: float | None = Field(
        default=None,
        description="nuisance: sensor distance ahead of the axle about which the vehicle turns",
    )
    dofs: list[VehicleFrameDofRecord]
    references: list[VehicleFrameReference] = Field(default_factory=list)
    motions: int = Field(ge=0)
    train_motions: int = Field(ge=0)
    holdout_motions: int = Field(ge=0)
    train_blocks: list[int] = Field(default_factory=list)
    holdout_blocks: list[int] = Field(default_factory=list)
    jackknife_fits: int = Field(ge=0)
    train_median_normalized_residual: float | None = None
    holdout_median_normalized_residual: float | None = None
    options: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)
    provenance: VehicleFrameProvenance

    @model_validator(mode="after")
    def check_policy(self) -> VehicleFrameRotationArtifact:
        """Keep calibrated DoFs and the pass verdict honest."""

        estimated = sorted(item.name for item in self.dofs if item.status == "estimated")
        if sorted(self.calibrated_dofs) != estimated:
            raise ValueError("calibrated_dofs must list exactly the estimated DoFs")
        if self.policy_status == "pass" and estimated != ["pitch", "roll", "yaw"]:
            raise ValueError("a pass verdict requires roll, pitch, and yaw to be estimated")
        if self.solver_status != "converged" and self.policy_status in {"pass", "warn"}:
            raise ValueError("an unsolved run cannot pass or warn")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def vehicle_frame_rotation_json_schema() -> dict[str, Any]:
    """Return the vehicle-frame rotation artifact JSON schema."""

    return VehicleFrameRotationArtifact.model_json_schema()


def load_vehicle_frame_rotation(path: str | Path) -> VehicleFrameRotationArtifact:
    """Load and validate a vehicle-frame rotation artifact."""

    return VehicleFrameRotationArtifact.model_validate(read_mapping(Path(path)))
