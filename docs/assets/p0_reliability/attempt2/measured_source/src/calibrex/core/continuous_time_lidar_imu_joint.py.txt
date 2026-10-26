"""Schema-valid joint continuous-time LiDAR-IMU calibration evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.fit_diagnostics import FitDiagnostics
from calibrex.core.io import write_mapping
from calibrex.core.result import StrictModel

CONTINUOUS_TIME_LIDAR_IMU_JOINT_SCHEMA_VERSION: Literal[
    "slac.continuous_time_lidar_imu_joint/v0.1"
] = "slac.continuous_time_lidar_imu_joint/v0.1"

LidarImuJointParameterName = Literal[
    "imu_rotation",
    "gyro_bias",
    "lever_arm",
    "imu_clock_offset",
    "accel_bias",
    "gravity",
]
LidarImuJointFamily = Literal["gyro", "accel"]


class ContinuousTimeLidarImuJointProvenance(StrictModel):
    """Seeded synthetic recovery lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    seed: int
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class LidarImuJointParameterRecord(StrictModel):
    """One jointly estimated calibration block against the injected truth.

    ``error`` is ``estimate - truth`` per axis; for ``imu_rotation`` it is the
    rotation vector of ``R_truth^T R_estimate`` (right perturbation, rad).
    ``std`` is the marginal standard deviation from the fit with the knots
    marginalized, and ``max_normalized_error`` is ``max |error / std|``.
    """

    name: LidarImuJointParameterName
    unit: str
    truth: list[float]
    initial: list[float]
    estimate: list[float]
    error: list[float]
    std: list[float]
    error_norm: float = Field(ge=0.0)
    error_budget: float = Field(gt=0.0)
    max_normalized_error: float = Field(ge=0.0)
    within_budget: bool


class LidarImuJointControlRecord(StrictModel):
    """A signed known-bad perturbation of one fitted parameter on holdout data.

    ``delta_chi2`` is the change of the whitened holdout chi-square of the
    family the parameter enters (gyro pre-integration or specific force).
    """

    parameter: LidarImuJointParameterName
    axis: int | None = None
    sign: Literal[-1, 1]
    magnitude: float = Field(gt=0.0)
    unit: str
    family: LidarImuJointFamily
    baseline_holdout_rmse: float = Field(ge=0.0)
    perturbed_holdout_rmse: float = Field(ge=0.0)
    delta_chi2: float
    detected: bool


class LidarImuJointSeparabilityRecord(StrictModel):
    """Largest absolute correlation between two estimated blocks."""

    block_a: LidarImuJointParameterName
    block_b: LidarImuJointParameterName
    max_abs_correlation: float = Field(ge=0.0)
    entangled: bool


class ContinuousTimeLidarImuJointArtifact(StrictModel):
    """Joint LiDAR-IMU recovery with holdout, signed controls and separability.

    ``interpolation`` names the trajectory model: ``screw_linear`` knots with
    gyro pre-integration windows (``gyro_residual`` in rad), or a C2
    ``cubic_bspline`` with per-sample angular-rate residuals (rad/s).  The
    trajectory errors are taken at the knots (screw-linear) or on a dense time
    grid against the continuous truth (B-spline).
    """

    schema_version: Literal[
        "slac.continuous_time_lidar_imu_joint/v0.1"
    ] = CONTINUOUS_TIME_LIDAR_IMU_JOINT_SCHEMA_VERSION
    recovery_id: str
    data_source: Literal["synthetic"]
    interpolation: Literal["screw_linear", "cubic_bspline"]
    seed: int
    knot_count: int = Field(ge=2)
    knot_spacing_sec: float = Field(gt=0.0)
    gyro_residual: Literal["preintegrated_window_rad", "angular_rate_rad_s"]
    holdout_scheme: str
    lidar_measurement_count: int = Field(ge=1)
    train_gyro_count: int = Field(ge=1)
    holdout_gyro_count: int = Field(ge=1)
    train_accel_count: int = Field(ge=1)
    holdout_accel_count: int = Field(ge=1)
    fit_status: Literal["converged", "max_iterations", "singular_system", "stalled"]
    fit_diagnostics: FitDiagnostics | None = None
    iterations: int = Field(ge=1)
    max_trajectory_translation_error_m: float = Field(ge=0.0)
    max_trajectory_rotation_error_deg: float = Field(ge=0.0)
    parameters: list[LidarImuJointParameterRecord]
    train_gyro_rmse: float = Field(ge=0.0)
    holdout_gyro_rmse: float = Field(ge=0.0)
    train_accel_rmse_m_s2: float = Field(ge=0.0)
    holdout_accel_rmse_m_s2: float = Field(ge=0.0)
    oracle_holdout_gyro_rmse: float = Field(ge=0.0)
    oracle_holdout_accel_rmse_m_s2: float = Field(ge=0.0)
    holdout_to_oracle_ratio_limit: float = Field(gt=0.0)
    control_delta_chi2_threshold: float = Field(gt=0.0)
    controls: list[LidarImuJointControlRecord]
    entangled_correlation_threshold: float = Field(gt=0.0, le=1.0)
    separability: list[LidarImuJointSeparabilityRecord]
    policy_status: Literal["pass", "fail"]
    policy_reason: str
    provenance: ContinuousTimeLidarImuJointProvenance

    def save(self, path: str | Path) -> None:
        """Save the recovery artifact as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )


def continuous_time_lidar_imu_joint_json_schema() -> dict[str, Any]:
    """Return the JSON schema for joint LiDAR-IMU recovery artifacts."""

    return ContinuousTimeLidarImuJointArtifact.model_json_schema()
