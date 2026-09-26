"""Schema-valid evidence for an externally produced Camera--IMU calibration.

An external tool such as Kalibr reports ``T_cam_imu`` and ``timeshift_cam_imu``
together with its own fitness metrics. Those metrics are not comparable across
tools, so Calibrex re-evaluates the imported candidate on a separate motion
recording with a disjoint temporal holdout and signed known-bad controls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex import __version__
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel

CAMERA_IMU_MOTION_RECORDING_SCHEMA_VERSION: Literal["slac.camera_imu_motion_recording/v0.1"] = (
    "slac.camera_imu_motion_recording/v0.1"
)
EXTERNAL_CAMERA_IMU_EVIDENCE_SCHEMA_VERSION: Literal["slac.external_camera_imu_evidence/v0.1"] = (
    "slac.external_camera_imu_evidence/v0.1"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

ExternalCameraImuEvidenceStatus = Literal["pass", "warn", "fail", "inconclusive", "blocked"]


class CameraOrientationSample(StrictModel):
    """Camera orientation ``R_world_camera`` on the camera clock."""

    timestamp_sec: float
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)


class ImuGyroSample(StrictModel):
    """Body-frame gyroscope sample on the IMU clock."""

    timestamp_sec: float
    omega_rad_s: list[float] = Field(min_length=3, max_length=3)


class CameraImuMotionRecordingProvenance(StrictModel):
    """Lineage of a Camera--IMU motion recording."""

    producer: str
    producer_version: str
    source: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    seed: int | None = None
    synthetic: bool = False


class CameraImuMotionRecording(StrictModel):
    """Camera orientations and IMU gyro samples used only for evaluation.

    The recording must not be the input the external tool was fitted on; the
    evaluator rejects a recording whose bytes match a declared fitting input.
    """

    schema_version: Literal["slac.camera_imu_motion_recording/v0.1"] = (
        CAMERA_IMU_MOTION_RECORDING_SCHEMA_VERSION
    )
    recording_id: str
    camera_name: str
    imu_name: str = "imu"
    time_convention: str = (
        "camera and IMU timestamps are on their own clocks; t_imu = t_cam + timeshift_cam_imu"
    )
    camera_orientations: list[CameraOrientationSample] = Field(min_length=2)
    imu_gyro: list[ImuGyroSample] = Field(min_length=2)
    provenance: CameraImuMotionRecordingProvenance

    @model_validator(mode="after")
    def require_monotonic_timestamps(self) -> CameraImuMotionRecording:
        """Reject unsorted or duplicated timestamps instead of reordering them."""

        for name, stamps in (
            ("camera_orientations", [item.timestamp_sec for item in self.camera_orientations]),
            ("imu_gyro", [item.timestamp_sec for item in self.imu_gyro]),
        ):
            if any(later <= earlier for earlier, later in pairwise(stamps)):
                msg = f"{name} timestamps must be strictly increasing"
                raise ValueError(msg)
        return self

    def save(self, path: str | Path) -> None:
        """Write the recording as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


class ExternalRunReference(StrictModel):
    """Digest-bound identity of the evaluated external-run artifact."""

    path: str
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    run_id: str | None = None
    adapter_name: str | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    source_commit: str | None = None
    license_spdx: str | None = None
    status: str | None = None
    training_isolation_declared: bool = False
    fitting_input_sha256: list[str] = Field(default_factory=list)


class RecordingReference(StrictModel):
    """Digest-bound identity of the evaluation recording."""

    path: str
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    recording_id: str | None = None
    camera_orientation_count: int = Field(default=0, ge=0)
    imu_sample_count: int = Field(default=0, ge=0)
    synthetic: bool | None = None


class CameraImuCandidate(StrictModel):
    """Imported candidate expressed in the Kalibr convention."""

    transform_name: str
    convention: Literal["T_cam_imu"] = "T_cam_imu"
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    translation_m: list[float] = Field(min_length=3, max_length=3)
    time_offset_name: str | None = None
    timeshift_cam_imu_sec: float
    time_offset_declared: bool


class CameraImuEvidenceThresholds(StrictModel):
    """Gate defaults; a program may tighten them before seeing results."""

    holdout_fraction_start: float = Field(default=0.35, gt=0.0, lt=1.0)
    holdout_fraction_end: float = Field(default=0.65, gt=0.0, lt=1.0)
    min_train_intervals: int = Field(default=20, ge=1)
    min_holdout_intervals: int = Field(default=20, ge=1)
    max_holdout_rate_rmse_rad_s: float = Field(default=0.05, gt=0.0)
    min_rate_excitation_rad_s: float = Field(default=0.15, ge=0.0)
    min_angular_acceleration_rad_s2: float = Field(default=0.5, ge=0.0)
    rotation_control_deg: float = Field(default=2.0, gt=0.0)
    time_control_sec: float = Field(default=0.01, gt=0.0)
    control_min_rmse_increase_rad_s: float = Field(default=0.005, ge=0.0)
    control_min_rmse_ratio: float = Field(default=1.2, ge=1.0)
    min_control_detection_fraction: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_ordered_holdout(self) -> CameraImuEvidenceThresholds:
        """Reject an empty or inverted holdout span."""

        if self.holdout_fraction_end <= self.holdout_fraction_start:
            msg = "holdout_fraction_end must exceed holdout_fraction_start"
            raise ValueError(msg)
        return self


class CameraImuTemporalSplit(StrictModel):
    """Contiguous middle-block holdout with a guard gap on both sides."""

    holdout_start_sec: float
    holdout_end_sec: float
    guard_sec: float = Field(ge=0.0)
    train_interval_count: int = Field(ge=0)
    holdout_interval_count: int = Field(ge=0)
    excluded_interval_count: int = Field(ge=0)


class CameraImuExcitation(StrictModel):
    """Holdout excitation needed for rotation and time to be observable."""

    min_rate_excitation_rad_s: float = Field(ge=0.0)
    rate_excitation_eigen_rad_s: list[float] = Field(min_length=3, max_length=3)
    rms_angular_acceleration_rad_s2: float = Field(ge=0.0)


class CameraImuControl(StrictModel):
    """One signed known-bad perturbation of the imported candidate."""

    control_id: str
    kind: Literal["rotation", "time"]
    axis: Literal["x", "y", "z"] | None = None
    signed_magnitude: float
    unit: Literal["deg", "s"]
    holdout_rate_rmse_rad_s: float = Field(ge=0.0)
    rmse_increase_rad_s: float
    detected: bool


class ExternalCameraImuEvidenceProvenance(StrictModel):
    """Lineage of the evidence artifact itself."""

    producer: Literal["calibrex"] = "calibrex"
    tool_name: str = "calibrex.external-run.evaluate-camera-imu"
    tool_version: str = __version__
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = Field(default_factory=git_commit)
    command: list[str] = Field(default_factory=list)
    input_sha256: dict[str, str] = Field(default_factory=dict)


class ExternalCameraImuEvidenceArtifact(StrictModel):
    """Holdout and known-bad evidence for an imported ``T_cam_imu`` candidate."""

    schema_version: Literal["slac.external_camera_imu_evidence/v0.1"] = (
        EXTERNAL_CAMERA_IMU_EVIDENCE_SCHEMA_VERSION
    )
    evaluation_id: str
    camera_name: str
    external_run: ExternalRunReference
    recording: RecordingReference
    candidate: CameraImuCandidate | None = None
    thresholds: CameraImuEvidenceThresholds
    split: CameraImuTemporalSplit | None = None
    estimated_gyro_bias_rad_s: list[float] | None = Field(default=None, min_length=3, max_length=3)
    train_rate_rmse_rad_s: float | None = Field(default=None, ge=0.0)
    holdout_rate_rmse_rad_s: float | None = Field(default=None, ge=0.0)
    excitation: CameraImuExcitation | None = None
    controls: list[CameraImuControl] = Field(default_factory=list)
    control_detection_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ExternalCameraImuEvidenceStatus
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    external_metrics_used: Literal[False] = False
    provenance: ExternalCameraImuEvidenceProvenance

    def save(self, path: str | Path) -> None:
        """Write the evidence artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def camera_imu_motion_recording_json_schema() -> dict[str, Any]:
    """Return the JSON schema for Camera--IMU motion recordings."""

    return CameraImuMotionRecording.model_json_schema()


def external_camera_imu_evidence_json_schema() -> dict[str, Any]:
    """Return the JSON schema for external Camera--IMU evidence artifacts."""

    return ExternalCameraImuEvidenceArtifact.model_json_schema()


def load_camera_imu_motion_recording(path: str | Path) -> CameraImuMotionRecording:
    """Load and validate a Camera--IMU motion recording."""

    return CameraImuMotionRecording.model_validate(read_mapping(Path(path)))


def load_external_camera_imu_evidence(
    path: str | Path,
) -> ExternalCameraImuEvidenceArtifact:
    """Load and validate an external Camera--IMU evidence artifact."""

    return ExternalCameraImuEvidenceArtifact.model_validate(read_mapping(Path(path)))
