"""Shared immutable contract for the first cross-dataset INS--LiDAR rotation audit."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tools.score_ins_lidar_sota import CaseSpec

PROTOCOL_ID = "ins-lidar-rotation-kitti-boreas-v1"
METHOD_ID = "calibrex_native_trajectory_hand_eye"
ERROR_MAX_DEG = 0.5
UNOBSERVED_FRACTION_MAX = 0.5
CONTROL_MISS_FRACTION_MAX = 0.0
CLAIM = (
    "On the four pre-registered KITTI raw and Boreas recording windows not previously "
    "read by Calibrex INS-LiDAR methods, LiDAR-only odometry followed by the default "
    "native joint trajectory hand-eye fit produces a scorable rotation estimate for "
    "every recording; every data-estimated rotation axis is within 0.5 deg of its "
    "dataset-provided reference calibration; at least half of the three rotation "
    "axes per recording are observed on average in each dataset family; and every "
    "observed-axis 1 deg held-out control is detected with delta chi-square at least 9. "
    "Dataset calibrations are reference estimates, not independent metrology. "
    "No translation, full-extrinsic or absolute clock-accuracy claim is made."
)
EXPECTED_CASES = (
    ("2011_09_29_0071", "kitti_raw", "kitti-platform"),
    ("2011_09_30_0020", "kitti_raw", "kitti-platform"),
    ("2011_09_30_0027", "kitti_raw", "kitti-platform"),
    ("boreas-2020-12-01-13-26", "boreas", "boreas-platform"),
)
SCOPE = {
    "modalities": ["ins", "lidar"],
    "quantities": ["rotation"],
    "category": "trajectory_hand_eye_per_axis",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetadataPin(_Strict):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InputFile(_Strict):
    path: str
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AcquisitionReceipt(_Strict):
    schema_version: Literal["calibrex.ins_lidar_audit_acquisition/v0.1"] = (
        "calibrex.ins_lidar_audit_acquisition/v0.1"
    )
    case_id: str
    dataset_family: Literal["kitti_raw", "boreas"]
    verified: Literal[True]
    sequence_dir: str
    files: list[InputFile] = Field(min_length=1)
    source_records: list[dict[str, Any]]
    provenance: dict[str, Any]


class RunManifest(_Strict):
    schema_version: Literal["calibrex.ins_lidar_sota_run/v0.1"] = "calibrex.ins_lidar_sota_run/v0.1"
    preregistration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    cases: list[CaseSpec]
    input_data_verified: bool = False
    input_receipts: dict[str, dict[str, str]]
    attempts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    acquisition_errors: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_population(self) -> RunManifest:
        identities = tuple((c.case_id, c.dataset_family, c.rig_id) for c in self.cases)
        if identities != EXPECTED_CASES:
            raise ValueError("run dropped or substituted a declared case")
        if set(self.input_receipts) - {c.case_id for c in self.cases}:
            raise ValueError("undeclared input receipt")
        return self


class EvaluationCase(_Strict):
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    dataset_family: Literal["kitti_raw", "boreas"]
    rig_id: str

    @property
    def drive_ids(self) -> list[str]:
        if self.dataset_family == "boreas":
            return [self.case_id]
        date, _, drive = self.case_id.rpartition("_")
        return [f"{date}_drive_{drive}_sync"]


class Thresholds(_Strict):
    maximum_reference_difference_deg: Literal[0.5] = ERROR_MAX_DEG
    maximum_unobserved_axis_fraction: Literal[0.5] = UNOBSERVED_FRACTION_MAX
    maximum_control_miss_fraction: Literal[0.0] = CONTROL_MISS_FRACTION_MAX
    maximum_failed_recording_fraction: Literal[0.0] = 0.0
    minimum_dataset_families: Literal[2] = 2
    minimum_independent_rigs: Literal[2] = 2


class AcquisitionPolicy(_Strict):
    max_frames: Literal[600] = 600
    minimum_free_disk_bytes: Literal[21474836480] = 20 * 1024**3
    maximum_additional_raw_bytes: Literal[17179869184] = 16 * 1024**3
    object_attempts: Literal[2] = 2
    method_attempts: Literal[1] = 1
    method_timeout_seconds: Literal[1800] = 1800
    boreas_workers: Literal[4] = 4
    max_point_object_bytes: Literal[16777216] = 16 * 1024**2
    max_reference_object_bytes: Literal[134217728] = 128 * 1024**2


class InsLidarPreregistration(_Strict):
    schema_version: Literal["calibrex.ins_lidar_sota_preregistration/v0.1"] = (
        "calibrex.ins_lidar_sota_preregistration/v0.1"
    )
    protocol_id: Literal["ins-lidar-rotation-kitti-boreas-v1"] = PROTOCOL_ID
    claim_text: str = CLAIM
    scope: dict[str, Any] = Field(default_factory=lambda: dict(SCOPE))
    frozen_code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    metadata: dict[Literal["kitti_raw", "boreas"], MetadataPin]
    evaluation: list[EvaluationCase]
    calibration_archive_sha256: dict[str, str]
    thresholds: Thresholds = Field(default_factory=Thresholds)
    acquisition: AcquisitionPolicy = Field(default_factory=AcquisitionPolicy)
    development: dict[str, Any]
    exposure: dict[str, Any]
    run_recipe: dict[str, Any]
    provenance: dict[str, Any]

    @model_validator(mode="after")
    def check_frozen_contract(self) -> InsLidarPreregistration:
        if (
            self.claim_text != CLAIM
            or self.scope != SCOPE
            or set(self.metadata) != {"kitti_raw", "boreas"}
        ):
            raise ValueError("claim or metadata family differs from the frozen contract")
        cases = tuple((c.case_id, c.dataset_family, c.rig_id) for c in self.evaluation)
        if cases != EXPECTED_CASES:
            raise ValueError("evaluation cases/order differ from the frozen contract")
        for pin in self.metadata.values():
            if Path(pin.path).is_absolute() or ".." in Path(pin.path).parts:
                raise ValueError("metadata must be a repository-relative path")
        if set(self.calibration_archive_sha256) != {"2011_09_29", "2011_09_30"}:
            raise ValueError("both KITTI calibration dates require pins")
        for digest in self.calibration_archive_sha256.values():
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid calibration archive SHA-256")
        return self


def load_preregistration(path: Path, source_root: Path) -> InsLidarPreregistration:
    """Validate the real pre-registration and both digest-pinned metadata inventories."""
    result = InsLidarPreregistration.model_validate(yaml.safe_load(path.read_text("utf-8")))
    for pin in result.metadata.values():
        actual = hashlib.sha256((source_root / pin.path).read_bytes()).hexdigest()
        if actual != pin.sha256:
            raise ValueError(f"metadata pin mismatch: {pin.path}")
    return result


def verify_frozen_source(
    prereg: InsLidarPreregistration, source_root: Path, *, require_main: bool
) -> str:
    """Refuse evaluation before the freeze is on main or after source changes."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=source_root, text=True).strip()

    head = git("rev-parse", "HEAD")
    git("cat-file", "-e", prereg.frozen_code_commit + "^{commit}")
    if git("diff", prereg.frozen_code_commit, head, "--", "src", "tools"):
        raise ValueError("src/tools differ from frozen_code_commit")
    if git("status", "--porcelain", "--", "src", "tools"):
        raise ValueError("uncommitted src/tools changes")
    if require_main and head != git("rev-parse", "main"):
        raise ValueError("evaluation must use the merged main commit")
    return head
