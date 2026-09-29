"""Schema-valid IMU-LiDAR translation (IMU lever arm) evidence.

The translation is that of ``T_lidar_imu`` (``T_parent_child``): the IMU
origin expressed in the LiDAR frame, in metres.  The rotation and the clock
offset (``t_imu = t_lidar + dt``) are inputs, taken from a
``slac.imu_lidar_rotation/v0.1`` artifact whose digest is recorded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.imu_lidar_rotation import ImuLidarPolicyStatus
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

IMU_LIDAR_TRANSLATION_SCHEMA_VERSION: Literal["slac.imu_lidar_translation/v0.1"] = (
    "slac.imu_lidar_translation/v0.1"
)
ImuLidarTranslationAxis = Literal["x", "y", "z"]


class ImuLidarTranslationProvenance(StrictModel):
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


class ImuLidarRotationInput(StrictModel):
    """The calibrated rotation, clock offset, and gyro bias the lever arm builds on."""

    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy_status: ImuLidarPolicyStatus
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    time_offset_s: float
    gyro_bias_rps: list[float] = Field(min_length=3, max_length=3)


class ImuLidarTranslationControl(StrictModel):
    """Held-out chi-square increase after shifting one axis of the lever arm."""

    amount: float
    unit: Literal["m"] = "m"
    holdout_delta_chi2: float
    detected: bool


class ImuLidarTranslationAxisRecord(StrictModel):
    """One lever-arm axis with its uncertainty and evidence."""

    name: ImuLidarTranslationAxis
    unit: Literal["m"] = "m"
    value: float
    std_analytic: float = Field(ge=0.0)
    std_jackknife: float | None = Field(default=None, ge=0.0)
    std_segment_sensitivity: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Largest change of this axis when the fit is repeated with other segment "
            "durations; it bounds modelling error that the jackknife cannot see."
        ),
    )
    std_reported: float = Field(ge=0.0)
    status: Literal["estimated", "unobservable"]
    reference_value: float | None = None
    error_to_reference: float | None = None
    known_bad_control: ImuLidarTranslationControl | None = None

    @model_validator(mode="after")
    def check_reported_std(self) -> ImuLidarTranslationAxisRecord:
        """Never report less uncertainty than any estimate provides."""

        floor = max(
            self.std_analytic, self.std_jackknife or 0.0, self.std_segment_sensitivity or 0.0
        )
        if self.std_reported + 1e-15 < floor:
            raise ValueError(
                "std_reported must be the largest of the analytic, jackknife, and "
                "segment-sensitivity std"
            )
        return self


class ImuLidarSegmentFit(StrictModel):
    """The lever arm refit with another segment duration."""

    segment_duration_s: float = Field(gt=0.0)
    translation_m: list[float] = Field(min_length=3, max_length=3)


class ImuLidarTranslationWindows(StrictModel):
    """How the odometry and IMU were cut into windows and segments."""

    windows: int = Field(ge=0)
    segments: int = Field(ge=0)
    scan_rows: int = Field(ge=0)
    imu_samples: int = Field(ge=0)


class ImuLidarTranslationArtifact(StrictModel):
    """Evidence for one IMU-LiDAR lever-arm calibration run."""

    schema_version: Literal["slac.imu_lidar_translation/v0.1"] = (
        IMU_LIDAR_TRANSLATION_SCHEMA_VERSION
    )
    method: Literal["accelerometer_lever_arm/v0.1"] = "accelerometer_lever_arm/v0.1"
    solver_status: Literal["converged", "insufficient_segments"]
    policy_status: ImuLidarPolicyStatus
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_dofs: list[ImuLidarTranslationAxis]
    translation_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    rotation_input: ImuLidarRotationInput
    axes: list[ImuLidarTranslationAxisRecord]
    segment_sensitivity: list[ImuLidarSegmentFit] = Field(default_factory=list)
    options: dict[str, Any]
    windows: ImuLidarTranslationWindows
    train_windows: int = Field(ge=0)
    holdout_windows: int = Field(ge=0)
    jackknife_fits: int = Field(ge=0)
    train_median_position_residual_m: float | None = None
    holdout_median_position_residual_m: float | None = None
    gravity_norm_median_mps2: float | None = Field(
        default=None,
        description="Median norm of the per-window gravity fit; a unit or scale sanity check.",
    )
    reference: str | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance: ImuLidarTranslationProvenance

    @model_validator(mode="after")
    def check_policy(self) -> ImuLidarTranslationArtifact:
        """Keep calibrated axes and the pass verdict honest."""

        estimated = sorted(item.name for item in self.axes if item.status == "estimated")
        if sorted(self.calibrated_dofs) != estimated:
            raise ValueError("calibrated_dofs must list exactly the estimated axes")
        if self.policy_status == "pass" and estimated != ["x", "y", "z"]:
            raise ValueError("a pass verdict requires x, y, and z to be estimated")
        if self.solver_status != "converged" and self.policy_status in {"pass", "warn"}:
            raise ValueError("an unsolved run cannot pass or warn")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def imu_lidar_translation_json_schema() -> dict[str, Any]:
    """Return the IMU-LiDAR translation artifact JSON schema."""

    return ImuLidarTranslationArtifact.model_json_schema()


def load_imu_lidar_translation(path: str | Path) -> ImuLidarTranslationArtifact:
    """Load and validate an IMU-LiDAR translation artifact."""

    return ImuLidarTranslationArtifact.model_validate(read_mapping(Path(path)))
