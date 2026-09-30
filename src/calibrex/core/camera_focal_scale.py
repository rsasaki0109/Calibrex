"""Schema-valid, target-free camera focal-length check against the gyro.

Rate ratios are camera/gyro angular-rate ratios about the camera x, y, z
axes, with the camera-IMU rotation, clock offset, and gyro bias taken from a
digest-pinned ``slac.imu_lidar_rotation/v0.1`` camera artifact.  The focal
estimates are ``fx_used k_y`` and ``fy_used k_x``.  ``k_z`` (about the
optical axis) does not depend on the focal length and serves as a control.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

CAMERA_FOCAL_SCALE_SCHEMA_VERSION: Literal["slac.camera_focal_scale/v0.1"] = (
    "slac.camera_focal_scale/v0.1"
)


class CameraFocalProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    sequence_ids: list[str] = Field(min_length=1)
    image_topic: str
    rotation_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_license: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CameraFocalScaleArtifact(StrictModel):
    """Focal-length estimates from camera/gyro rate ratios, with evidence."""

    schema_version: Literal["slac.camera_focal_scale/v0.1"] = CAMERA_FOCAL_SCALE_SCHEMA_VERSION
    method: Literal["gyro_rate_ratio/v0.1"] = "gyro_rate_ratio/v0.1"
    policy_status: Literal["pass", "warn", "fail", "inconclusive"]
    policy_reasons: list[str] = Field(min_length=1)
    rate_ratio: list[float] = Field(min_length=3, max_length=3)
    rate_ratio_std: list[float] = Field(min_length=3, max_length=3)
    holdout_rate_ratio: list[float] = Field(min_length=3, max_length=3)
    fx_used_px: float = Field(gt=0.0)
    fy_used_px: float = Field(gt=0.0)
    fx_estimate_px: float = Field(gt=0.0)
    fy_estimate_px: float = Field(gt=0.0)
    fx_std_px: float = Field(ge=0.0)
    fy_std_px: float = Field(ge=0.0)
    intervals: int = Field(ge=0)
    options: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)
    provenance: CameraFocalProvenance

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def camera_focal_scale_json_schema() -> dict[str, Any]:
    """Return the camera focal-scale artifact JSON schema."""

    return CameraFocalScaleArtifact.model_json_schema()


def load_camera_focal_scale(path: str | Path) -> CameraFocalScaleArtifact:
    """Load and validate a camera focal-scale artifact."""

    return CameraFocalScaleArtifact.model_validate(read_mapping(Path(path)))
