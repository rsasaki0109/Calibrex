"""The frozen contract of the second cross-dataset INS-LiDAR rotation audit (v2).

v1 (``ins_lidar_sota_protocol``) was refuted at 6/8. Its accuracy gate was already per
estimated axis and held (every estimated axis within 0.29 deg), and every control on an
estimated axis was detected. It failed because it also required a scorable estimate for
every recording, and counted a recording with no observable axis as a failed recording
and as a missed control. v2 gates what the method judges and how much it judges:

* a recording where the method reports no observable rotation axis is not a failure; it
  counts in coverage, with all three axes unobserved;
* agreement and controls are gated on the recordings with at least one estimated axis;
* a method failure (no converged result, a crash, a timeout) is still a failure;
* a recording that could not be acquired leaves the audit incomplete, not refuted.

No numeric gate is looser than v1's.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tools.ins_lidar_sota_protocol import EvaluationCase, MetadataPin
from tools.score_ins_lidar_sota import CaseSpec

PROTOCOL_ID = "ins-lidar-rotation-kitti-boreas-v2"
METHOD_ID = "calibrex_native_trajectory_hand_eye"
ERROR_MAX_DEG = 0.5
UNOBSERVED_FRACTION_MAX = 0.5
CONTROL_MISS_FRACTION_MAX = 0.0
METHOD_FAILURE_FRACTION_MAX = 0.0
CLAIM = (
    "On the ten pre-registered KITTI raw and Boreas recording windows not previously read "
    "by Calibrex INS-LiDAR methods, LiDAR-only odometry followed by the default native "
    "joint trajectory hand-eye fit completes on every recording; every rotation axis it "
    "reports as data-estimated is within 0.5 deg of its dataset-provided reference "
    "calibration; at least half of the three rotation axes per recording are estimated on "
    "average in each dataset family; and every 1 deg held-out control on an estimated axis "
    "is detected with delta chi-square at least 9. A recording on which the method reports "
    "no estimated axis is not a failure: its three axes count as unobserved. Dataset "
    "calibrations are reference estimates, not independent metrology. No translation, "
    "full-extrinsic or absolute clock-accuracy claim is made."
)
EXPECTED_CASES = (
    ("2011_09_30_0028", "kitti_raw", "kitti-platform"),
    ("2011_09_30_0033", "kitti_raw", "kitti-platform"),
    ("2011_09_30_0034", "kitti_raw", "kitti-platform"),
    ("2011_10_03_0027", "kitti_raw", "kitti-platform"),
    ("2011_10_03_0034", "kitti_raw", "kitti-platform"),
    ("2011_10_03_0042", "kitti_raw", "kitti-platform"),
    ("2011_10_03_0047", "kitti_raw", "kitti-platform"),
    ("boreas-2020-12-18-13-44", "boreas", "boreas-platform"),
    ("boreas-2021-01-15-12-17", "boreas", "boreas-platform"),
    ("boreas-2021-01-19-15-08", "boreas", "boreas-platform"),
)
DEVELOPMENT_CASES = (
    ("2011_09_29_0071", "kitti_raw", "kitti-platform"),
    ("2011_09_30_0020", "kitti_raw", "kitti-platform"),
    ("2011_09_30_0027", "kitti_raw", "kitti-platform"),
    ("boreas-2020-12-01-13-26", "boreas", "boreas-platform"),
)
"""v1's evaluation recordings: spent, so development data here."""
SCOPE = {
    "modalities": ["ins", "lidar"],
    "quantities": ["rotation"],
    "category": "trajectory_hand_eye_per_judged_axis",
}
CALIBRATION_DATES = ("2011_09_30", "2011_10_03")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Thresholds(_Strict):
    maximum_reference_difference_deg: Literal[0.5] = ERROR_MAX_DEG
    maximum_unobserved_axis_fraction: Literal[0.5] = UNOBSERVED_FRACTION_MAX
    maximum_control_miss_fraction: Literal[0.0] = CONTROL_MISS_FRACTION_MAX
    maximum_method_failure_fraction: Literal[0.0] = METHOD_FAILURE_FRACTION_MAX
    minimum_dataset_families: Literal[2] = 2
    minimum_independent_rigs: Literal[2] = 2


class AcquisitionPolicy(_Strict):
    """What one case may read, and how the run survives a restart."""

    max_frames: Literal[600] = 600
    minimum_free_disk_bytes: Literal[6442450944] = 6 * 1024**3
    object_attempts: Literal[2] = 2
    method_attempts: Literal[1] = 1
    method_timeout_seconds: Literal[1800] = 1800
    boreas_workers: Literal[4] = 4
    max_point_object_bytes: Literal[16777216] = 16 * 1024**2
    max_reference_object_bytes: Literal[134217728] = 128 * 1024**2
    delete_raw_after_method: Literal[True] = True
    rerun_interrupted_method: Literal[True] = True


class RunManifest(_Strict):
    schema_version: Literal["calibrex.ins_lidar_sota_run/v0.2"] = "calibrex.ins_lidar_sota_run/v0.2"
    preregistration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    development: bool = False
    cases: list[CaseSpec]
    input_receipts: dict[str, dict[str, Any]]
    attempts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    interrupted_attempts: list[dict[str, Any]] = Field(default_factory=list)
    acquisition_errors: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_population(self) -> RunManifest:
        identities = tuple((c.case_id, c.dataset_family, c.rig_id) for c in self.cases)
        if identities != (DEVELOPMENT_CASES if self.development else EXPECTED_CASES):
            raise ValueError("run dropped or substituted a declared case")
        if set(self.input_receipts) - {c.case_id for c in self.cases}:
            raise ValueError("undeclared input receipt")
        return self


class InsLidarV2Preregistration(_Strict):
    schema_version: Literal["calibrex.ins_lidar_sota_preregistration/v0.2"] = (
        "calibrex.ins_lidar_sota_preregistration/v0.2"
    )
    protocol_id: Literal["ins-lidar-rotation-kitti-boreas-v2"] = PROTOCOL_ID
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
    def check_frozen_contract(self) -> InsLidarV2Preregistration:
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
        if set(self.calibration_archive_sha256) != set(CALIBRATION_DATES):
            raise ValueError("both KITTI calibration dates require pins")
        for digest in self.calibration_archive_sha256.values():
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid calibration archive SHA-256")
        return self


def load_preregistration(path: Path, source_root: Path) -> InsLidarV2Preregistration:
    """Validate the real pre-registration and both digest-pinned metadata inventories."""

    result = InsLidarV2Preregistration.model_validate(yaml.safe_load(path.read_text("utf-8")))
    for pin in result.metadata.values():
        actual = hashlib.sha256((source_root / pin.path).read_bytes()).hexdigest()
        if actual != pin.sha256:
            raise ValueError(f"metadata pin mismatch: {pin.path}")
    return result
