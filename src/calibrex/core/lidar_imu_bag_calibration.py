"""Schema-valid joint LiDAR-IMU calibration of a recorded bag."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import write_mapping
from calibrex.core.result import StrictModel

LIDAR_IMU_BAG_CALIBRATION_SCHEMA_VERSION: Literal["slac.lidar_imu_bag_calibration/v0.1"] = (
    "slac.lidar_imu_bag_calibration/v0.1"
)

LidarImuBagBlock = Literal[
    "imu_rotation",
    "gyro_bias",
    "lever_arm",
    "imu_clock_offset",
    "accel_bias",
    "gravity",
]
LidarImuBagFamily = Literal["gyro", "accel"]
LidarImuBagPolicyStatus = Literal["pass", "warn", "fail", "inconclusive"]


class LidarImuBagProvenance(StrictModel):
    """Where the calibration came from."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    dataset_license: str
    bag: str
    lidar_topic: str
    imu_topic: str
    point_time_field: str | None = None
    input_sha256: str
    input_digest_scope: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LidarImuBagInitialEstimate(StrictModel):
    """The angular-rate estimate that seeds the joint fit (an independent method)."""

    method: Literal["angular_rate_alignment/v0.1"] = "angular_rate_alignment/v0.1"
    policy_status: str
    rotation_vector_rad: list[float] = Field(min_length=3, max_length=3)
    clock_offset_s: float
    gyro_bias_rad_s: list[float] = Field(min_length=3, max_length=3)


class LidarImuBagBlockRecord(StrictModel):
    """One estimated calibration block.

    ``formal_std`` comes from the fit's covariance with the plane map held
    fixed, so it understates the uncertainty that the map itself carries;
    ``last_round_change`` is how far the block moved in the final map round.
    """

    name: LidarImuBagBlock
    unit: str
    estimate: list[float]
    formal_std: list[float]
    last_round_change: list[float]
    controls_detected: int = Field(ge=0)
    controls: int = Field(ge=0)
    verdict: Literal["estimated", "weak"]
    reason: str


class LidarImuBagControlRecord(StrictModel):
    """A signed known-bad perturbation of one block, scored on held-out IMU samples."""

    parameter: LidarImuBagBlock
    axis: int | None = None
    sign: Literal[-1, 1]
    magnitude: float = Field(gt=0.0)
    unit: str
    family: LidarImuBagFamily
    delta_chi2: float
    detected: bool


class LidarImuBagSeparabilityRecord(StrictModel):
    """Largest absolute correlation between two estimated blocks."""

    block_a: LidarImuBagBlock
    block_b: LidarImuBagBlock
    max_abs_correlation: float = Field(ge=0.0)
    entangled: bool


class LidarImuBagRound(StrictModel):
    """One map round: re-associate LiDAR returns to the plane map, then refit."""

    round: int = Field(ge=0)
    map_planes: int = Field(ge=0)
    plane_returns: int = Field(ge=0)
    fit_status: Literal["converged", "max_iterations", "singular_system"]
    iterations: int = Field(ge=1)
    rotation_change_deg: float = Field(ge=0.0)


class LidarImuBagCalibrationArtifact(StrictModel):
    """Joint B-spline LiDAR-IMU calibration of one bag segment, with its evidence.

    The transform is ``T_lidar_imu`` (IMU frame in the LiDAR frame) and the
    clock offset follows ``t_imu = t_lidar + imu_clock_offset_s``.
    """

    schema_version: Literal["slac.lidar_imu_bag_calibration/v0.1"] = (
        LIDAR_IMU_BAG_CALIBRATION_SCHEMA_VERSION
    )
    method: Literal["cubic_bspline_joint/v0.1"] = "cubic_bspline_joint/v0.1"
    policy_status: LidarImuBagPolicyStatus
    policy_reasons: list[str] = Field(min_length=1)
    t_lidar_imu_translation_m: list[float] = Field(min_length=3, max_length=3)
    t_lidar_imu_rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    imu_clock_offset_s: float
    blocks: list[LidarImuBagBlockRecord]
    initial: LidarImuBagInitialEstimate
    rotation_difference_from_initial_deg: float = Field(ge=0.0)
    segment_start_s: float
    segment_duration_s: float = Field(gt=0.0)
    knot_spacing_s: float = Field(gt=0.0)
    control_points: int = Field(ge=4)
    scans: int = Field(ge=1)
    train_imu_samples: int = Field(ge=1)
    holdout_imu_samples: int = Field(ge=1)
    noise_sigmas: dict[str, float]
    rounds: list[LidarImuBagRound] = Field(min_length=1)
    converged: bool
    train_whitened_rms: dict[str, float]
    holdout_whitened_rms: dict[str, float]
    control_delta_chi2_threshold: float = Field(gt=0.0)
    controls: list[LidarImuBagControlRecord]
    entangled_correlation_threshold: float = Field(gt=0.0, le=1.0)
    separability: list[LidarImuBagSeparabilityRecord]
    options: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)
    provenance: LidarImuBagProvenance

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def lidar_imu_bag_calibration_json_schema() -> dict[str, Any]:
    """Return the JSON schema for joint LiDAR-IMU bag calibrations."""

    return LidarImuBagCalibrationArtifact.model_json_schema()
