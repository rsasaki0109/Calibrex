"""Pilot-ready handoff and verification for a real Koide execution.

This module intentionally does not run Koide.  It records the exact external
execution request, the official dataset handoff, the frame-disjoint split and
known-bad controls, and the acceptance policy used by
``calibrex camera-lidar benchmark-koide-pilot``.  A request is *not* evidence:
the verifier remains ``BLOCKED`` until an operator supplies digest-locked
inputs, a successful external-run artifact, and the native ``calib.json``.

The native format adapter and the frozen pilot evaluator live in
``calibrex.importers.koide`` and ``calibrex.evaluation.koide_pilot``.  Keeping
this boundary separate makes it impossible for a handoff/checklist to be
mistaken for an executed benchmark.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from calibrex import __version__
from calibrex.core.external_run import ExternalCalibrationRunArtifact, load_external_run
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.koide_handoff import (
    KOIDE_EXECUTION_LOCK_SCHEMA_VERSION,
    KOIDE_LOCK_HUMBLE_IMAGE,
    KOIDE_LOCK_SOURCE_COMMIT,
    KOIDE_LOCK_SOURCE_REPOSITORY,
    load_koide_execution_lock,
)
from calibrex.core.koide_readiness import load_koide_readiness
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel
from calibrex.data.kitti_benchmark import (
    KITTI_RAW_0005_BENCHMARK_FRAME_IDS,
    KITTI_RAW_0005_DATASET_ID,
    KITTI_RAW_0005_SEQUENCE_ID,
    KITTI_RAW_OFFICIAL_URL,
    load_kitti_benchmark_input,
    verify_kitti_benchmark_input,
)
from calibrex.evaluation.koide_pilot import (
    KoidePilotArtifact,
    KoidePilotThresholds,
    load_koide_pilot,
)
from calibrex.importers.koide import parse_koide_calib_json

KOIDE_REAL_PILOT_REQUEST_SCHEMA_VERSION: Literal["slac.koide_real_pilot_request/v0.1"] = (
    "slac.koide_real_pilot_request/v0.1"
)
KOIDE_REAL_PILOT_VERIFICATION_SCHEMA_VERSION: Literal["slac.koide_real_pilot_verification/v0.1"] = (
    "slac.koide_real_pilot_verification/v0.1"
)
KOIDE_REAL_PILOT_FINALIZATION_SCHEMA_VERSION: Literal["slac.koide_real_pilot_finalization/v0.1"] = (
    "slac.koide_real_pilot_finalization/v0.1"
)

KOIDE_OFFICIAL_EXAMPLE_URL = "https://koide3.github.io/direct_visual_lidar_calibration/example/"
KOIDE_OFFICIAL_DOCKER_URL = "https://koide3.github.io/direct_visual_lidar_calibration/docker/"
KOIDE_LOCK_RELATIVE_PATH = "koide_execution_lock.yaml"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

KoideRequestArtifactRole = Literal[
    "dataset_archive",
    "dataset_sequence",
    "input_manifest",
    "config",
    "readiness",
    "native_calib_json",
    "external_run",
    "pilot",
    "execution_log",
    "environment",
    "execution_evidence",
]
KoideRequestArtifactStatus = Literal["missing", "supplied", "verified"]


class KoideRealArtifactRef(StrictModel):
    """Optional digest reference supplied by the external runner."""

    role: KoideRequestArtifactRole
    path: str | None = None
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    size_bytes: int | None = Field(default=None, ge=0)
    required: bool = True
    status: KoideRequestArtifactStatus = "missing"

    @model_validator(mode="after")
    def validate_supply_state(self) -> KoideRealArtifactRef:
        present = self.path is not None or self.sha256 is not None or self.size_bytes is not None
        if self.status == "missing" and present:
            raise ValueError("missing artifact references must not contain path, digest, or size")
        if self.status in {"supplied", "verified"} and (
            self.path is None or self.sha256 is None or self.size_bytes is None
        ):
            raise ValueError("supplied artifact references require path, SHA-256, and size")
        return self


class KoideRealDatasetPin(StrictModel):
    """Official real-data request; raw data are never redistributed here."""

    dataset_id: str = KITTI_RAW_0005_DATASET_ID
    sequence_id: str = KITTI_RAW_0005_SEQUENCE_ID
    official_record_url: str = KITTI_RAW_OFFICIAL_URL
    license_url: str = "https://www.cvlibs.net/datasets/kitti/"
    license_name: Literal["CC BY-NC-SA 3.0"] = "CC BY-NC-SA 3.0"
    download_requires_registration: Literal[True] = True
    license_statement: str = (
        "Official KITTI page states Creative Commons Attribution-NonCommercial-"
        "ShareAlike 3.0 and academic-use restrictions; operator must review the "
        "current terms before download or publication"
    )
    source_archive_name: str = "2011_09_26_drive_0005_sync.zip"
    source_archive_url: str | None = None
    archive_metadata_status: Literal[
        "not_published_by_official_record", "operator_observed", "official_record"
    ] = "not_published_by_official_record"
    archive_metadata_note: str = (
        "The official KITTI raw-data record requires login and does not publish a "
        "per-archive byte size or SHA-256 checksum. Do not copy size/checksum values "
        "from an unauthorised mirror into an official claim."
    )
    local_sequence_path: str | None = None
    archive_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    archive_size_bytes: int | None = Field(default=None, ge=0)
    checksum_status: Literal["not_yet_observed", "observed"] = "not_yet_observed"
    required_streams: list[str] = Field(
        default_factory=lambda: [
            "image_02/data/*.png",
            "velodyne_points/data/*.bin",
            "image_02/timestamps.txt",
            "velodyne_points/timestamps.txt",
            "calib_cam_to_cam.txt",
            "calib_velo_to_cam.txt",
        ],
        min_length=2,
    )

    @model_validator(mode="after")
    def bind_checksum_status(self) -> KoideRealDatasetPin:
        observed = self.archive_sha256 is not None and self.archive_size_bytes is not None
        if self.checksum_status == "observed" and not observed:
            raise ValueError("observed dataset checksum requires archive SHA-256 and size")
        if self.checksum_status == "not_yet_observed" and observed:
            raise ValueError("dataset checksum fields require checksum_status='observed'")
        if self.archive_metadata_status == "official_record" and not observed:
            raise ValueError(
                "official dataset archive metadata requires an authoritative SHA-256 and size"
            )
        if self.archive_metadata_status == "operator_observed" and not observed:
            raise ValueError("operator-observed archive metadata requires SHA-256 and size")
        if self.archive_metadata_status == "not_published_by_official_record" and observed:
            raise ValueError(
                "observed archive metadata must declare operator_observed or official_record"
            )
        return self


class KoideSplitManifest(StrictModel):
    """Predeclared frame-disjoint fitting and holdout split."""

    split_id: str = "kitti_raw_0005_frame_disjoint_v0_1"
    policy: str = (
        "fixed timestamp-order selection from KITTI raw 0005; candidate is frozen "
        "before scoring; holdout is never supplied to Koide fitting"
    )
    frame_ids: list[str] = Field(default_factory=lambda: list(KITTI_RAW_0005_BENCHMARK_FRAME_IDS))
    fitting_frame_ids: list[str] = Field(
        default_factory=lambda: list(KITTI_RAW_0005_BENCHMARK_FRAME_IDS[:16])
    )
    holdout_frame_ids: list[str] = Field(
        default_factory=lambda: list(KITTI_RAW_0005_BENCHMARK_FRAME_IDS[16:])
    )
    frame_ids_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    fitting_frame_ids_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    holdout_frame_ids_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    candidate_frozen_before_holdout: Literal[True] = True
    holdout_independent: Literal[True] = True

    @model_validator(mode="after")
    def validate_disjoint_split(self) -> KoideSplitManifest:
        all_ids = set(self.frame_ids)
        fitting = set(self.fitting_frame_ids)
        holdout = set(self.holdout_frame_ids)
        if not all_ids or len(all_ids) != len(self.frame_ids):
            raise ValueError("split frame_ids must be non-empty and unique")
        if not fitting or not holdout:
            raise ValueError("both fitting and holdout frame IDs are required")
        if fitting & holdout:
            raise ValueError("fitting and holdout frame IDs must be disjoint")
        if fitting | holdout != all_ids:
            raise ValueError("fitting and holdout IDs must cover frame_ids exactly")
        if any(len(item) != 10 or not item.isdigit() for item in all_ids):
            raise ValueError("KITTI frame IDs must be ten-digit decimal strings")
        expected = {
            "frame_ids_sha256": _frame_ids_digest(self.frame_ids),
            "fitting_frame_ids_sha256": _frame_ids_digest(self.fitting_frame_ids),
            "holdout_frame_ids_sha256": _frame_ids_digest(self.holdout_frame_ids),
        }
        for name, digest in expected.items():
            declared = getattr(self, name)
            if declared is not None and declared != digest:
                raise ValueError(f"{name} does not match the declared frame IDs")
        return self

    def with_digests(self) -> KoideSplitManifest:
        """Return the split with canonical frame inventory digests populated."""

        return self.model_copy(
            update={
                "frame_ids_sha256": _frame_ids_digest(self.frame_ids),
                "fitting_frame_ids_sha256": _frame_ids_digest(self.fitting_frame_ids),
                "holdout_frame_ids_sha256": _frame_ids_digest(self.holdout_frame_ids),
            }
        )


class KoideKnownBadInjection(StrictModel):
    """One signed perturbation that must be detected independently."""

    control_id: str
    dof: Literal["roll", "pitch", "yaw", "x", "y", "z"]
    sign: Literal["+", "-"]
    amount: float = Field(gt=0.0)
    unit: Literal["deg", "m"]
    injection: Literal["compose_after_candidate_no_refit"] = "compose_after_candidate_no_refit"
    refit_allowed: Literal[False] = False
    required_metric_delta: float = Field(default=0.01, gt=0.0)


def _default_known_bad_controls() -> list[KoideKnownBadInjection]:
    controls: list[KoideKnownBadInjection] = []
    rotational_dofs: tuple[Literal["roll", "pitch", "yaw"], ...] = (
        "roll",
        "pitch",
        "yaw",
    )
    translational_dofs: tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")
    signs: tuple[Literal["+", "-"], ...] = ("+", "-")
    for rotational_dof in rotational_dofs:
        for sign in signs:
            controls.append(
                KoideKnownBadInjection(
                    control_id=f"{rotational_dof}_{'plus' if sign == '+' else 'minus'}_5deg",
                    dof=rotational_dof,
                    sign=sign,
                    amount=5.0,
                    unit="deg",
                )
            )
    for translational_dof in translational_dofs:
        for sign in signs:
            controls.append(
                KoideKnownBadInjection(
                    control_id=f"{translational_dof}_{'plus' if sign == '+' else 'minus'}_0p25m",
                    dof=translational_dof,
                    sign=sign,
                    amount=0.25,
                    unit="m",
                )
            )
    return controls


class KoideExecutionStageRequest(StrictModel):
    """Argv-only command copied from the pinned upstream workflow."""

    name: Literal["preprocess", "initial_guess", "calibrate"]
    argv: list[str] = Field(min_length=1)
    official_documentation_url: str
    environment: dict[str, str] = Field(default_factory=dict)
    working_directory: str = "/tmp"
    requires_manual_gui: bool = False

    @model_validator(mode="after")
    def reject_shell_and_superglue(self) -> KoideExecutionStageRequest:
        if any("\x00" in item for item in self.argv):
            raise ValueError("Koide stage argv must not contain NUL bytes")
        if any("superglue" in item.lower() for item in self.argv):
            raise ValueError("commercial Koide request must exclude SuperGlue")
        if any(item in {"sh", "bash", "zsh", "cmd", "powershell", "pwsh"} for item in self.argv):
            raise ValueError("Koide stage argv must invoke the tool directly, not a shell")
        return self


class KoideExecutionRequest(StrictModel):
    """Pinned container, source, and multistage execution contract."""

    lock_schema_version: Literal["slac.koide_execution_lock/v0.1"] = (
        KOIDE_EXECUTION_LOCK_SCHEMA_VERSION
    )
    lock_path: str = KOIDE_LOCK_RELATIVE_PATH
    lock_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_repository: str = KOIDE_LOCK_SOURCE_REPOSITORY
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_license_spdx: Literal["MIT"] = "MIT"
    image_digest: str = Field(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
    base_image: str = "koide3/gtsam_docker:humble"
    base_image_digest: str | None = Field(default=None, pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
    base_image_digest_required_for_rebuild: Literal[True] = True
    initial_guess_mode: Literal["manual"] = "manual"
    superglue_policy: Literal["excluded"] = "excluded"
    engine: Literal["docker", "podman"] = "docker"
    network_mode: Literal["none"] = "none"
    input_mount_readonly: Literal[True] = True
    output_mount_writable: Literal[True] = True
    rootfs_readonly: Literal[True] = True
    input_representation: Literal["ros2_bag"] = "ros2_bag"
    host_requirements: list[str] = Field(
        default_factory=lambda: [
            "Docker or Podman with immutable image-pull support",
            "ROS 2 Humble container runtime and a display for manual initialization",
            "user-owned ROS 2 bags containing Image, PointCloud2, and CameraInfo",
        ],
        min_length=1,
    )
    environment: dict[str, str] = Field(
        default_factory=lambda: {
            "LC_ALL": "C.UTF-8",
            "ROS_DOMAIN_ID": "0",
            "DISPLAY": "<operator-display-required-for-manual-stage>",
        }
    )
    environment_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    command_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    expected_native_output_relative_path: str = "calib.json"
    expected_native_output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    expected_native_output_size_bytes: int | None = Field(default=None, ge=0)
    stages: list[KoideExecutionStageRequest] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_stages(self) -> KoideExecutionRequest:
        if [stage.name for stage in self.stages] != ["preprocess", "initial_guess", "calibrate"]:
            raise ValueError("Koide execution stages must be preprocess, initial_guess, calibrate")
        if "initial_guess_manual" not in " ".join(self.stages[1].argv):
            raise ValueError("commercial Koide request must use manual initial guess")
        if not self.image_digest.startswith("koide3/direct_visual_lidar_calibration@"):
            raise ValueError("Koide request image must use the pinned official repository")
        if (
            self.expected_native_output_sha256 is not None
            and self.expected_native_output_size_bytes is None
        ):
            raise ValueError("expected native output digest requires an expected byte size")
        return self


class KoideNativeOutputContract(StrictModel):
    """Expected native output and explicit frame convention."""

    relative_path: str = "calib.json"
    native_format: Literal["direct_visual_lidar_calibration.calib.json"] = (
        "direct_visual_lidar_calibration.calib.json"
    )
    native_transform: Literal["results.T_lidar_camera"] = "results.T_lidar_camera"
    vector_order: Literal["x,y,z,qx,qy,qz,qw"] = "x,y,z,qx,qy,qz,qw"
    convention: str = (
        "T_lidar_camera maps camera-frame points into the LiDAR frame: "
        "p_lidar = R_lidar_camera p_camera + t_lidar_camera"
    )
    lidar_frame: str = Field(min_length=1)
    camera_frame: str = Field(min_length=1)
    expected_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    expected_size_bytes: int | None = Field(default=None, ge=0)
    digest_status: Literal["not_yet_observed", "observed"] = "not_yet_observed"

    @model_validator(mode="after")
    def bind_output_digest_status(self) -> KoideNativeOutputContract:
        observed = self.expected_sha256 is not None and self.expected_size_bytes is not None
        if self.digest_status == "observed" and not observed:
            raise ValueError("observed native output digest requires SHA-256 and size")
        if self.digest_status == "not_yet_observed" and observed:
            raise ValueError("native output digest fields require digest_status='observed'")
        if self.lidar_frame == self.camera_frame:
            raise ValueError("lidar_frame and camera_frame must be different")
        return self


class KoideRealAcceptancePolicy(StrictModel):
    """Frozen criteria delegated to the existing independent pilot evaluator."""

    thresholds: KoidePilotThresholds = Field(default_factory=KoidePilotThresholds)
    require_successful_external_run: Literal[True] = True
    require_digest_bound_native_output: Literal[True] = True
    require_training_isolation: Literal[True] = True
    require_independent_holdout: Literal[True] = True
    require_all_signed_known_bad_controls: Literal[True] = True
    require_readiness_ready: Literal[True] = True
    require_autoware_export_only_after_pass_adopt: Literal[True] = True


class KoideRealPilotProvenance(StrictModel):
    """Provenance for the request or verifier artifact."""

    producer: Literal["calibrex"] = "calibrex"
    calibrex_version: str = __version__
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    request_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class KoideRealPilotRequest(StrictModel):
    """Machine-readable external request; never an official result claim."""

    schema_version: Literal["slac.koide_real_pilot_request/v0.1"] = (
        KOIDE_REAL_PILOT_REQUEST_SCHEMA_VERSION
    )
    request_id: str
    status: Literal["BLOCKED"] = "BLOCKED"
    evidence_boundary: Literal["external_execution_request_only"] = (
        "external_execution_request_only"
    )
    synthetic_data_included: Literal[False] = False
    real_data_observed: Literal[False] = False
    official_result_claim: Literal[False] = False
    dataset: KoideRealDatasetPin
    execution: KoideExecutionRequest
    split: KoideSplitManifest
    known_bad_controls: list[KoideKnownBadInjection] = Field(
        default_factory=_default_known_bad_controls,
        min_length=12,
        max_length=12,
    )
    output_contract: KoideNativeOutputContract
    acceptance: KoideRealAcceptancePolicy = Field(default_factory=KoideRealAcceptancePolicy)
    artifacts: list[KoideRealArtifactRef] = Field(default_factory=list)
    external_commands: list[list[str]] = Field(default_factory=list, min_length=3)
    operator_checklist: list[str] = Field(default_factory=list, min_length=1)
    provenance: KoideRealPilotProvenance
    request_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_request_boundary(self) -> KoideRealPilotRequest:
        if len(self.known_bad_controls) != 12:
            raise ValueError("exactly twelve signed known-bad controls are required")
        if self.execution.superglue_policy != "excluded":
            raise ValueError("commercial request must exclude SuperGlue")
        if any(
            "superglue" in argument.lower()
            for command in self.external_commands
            for argument in command
        ):
            raise ValueError("commercial request external commands must exclude SuperGlue")
        expected_output = self.execution.expected_native_output_sha256
        expected_size = self.execution.expected_native_output_size_bytes
        if expected_output is not None and (
            self.output_contract.expected_sha256 != expected_output
            or self.output_contract.expected_size_bytes != expected_size
        ):
            raise ValueError("execution and native output contract expected digests must agree")
        if self.provenance.request_sha256 is not None and (
            self.provenance.request_sha256 != self.request_sha256
        ):
            raise ValueError("request provenance digest must match request_sha256")
        return self

    def with_request_digest(self) -> KoideRealPilotRequest:
        """Return a request with its canonical self-digest populated."""

        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("request_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("request_sha256", None)
        provenance.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "request_sha256": digest,
                "provenance": self.provenance.model_copy(update={"request_sha256": digest}),
            }
        )

    def verify_request_digest(self) -> None:
        expected = self.with_request_digest().request_sha256
        if self.request_sha256 != expected:
            raise ValueError(
                "Koide real pilot request self-digest mismatch: "
                f"declared={self.request_sha256}, expected={expected}"
            )

    def save(self, path: str | Path) -> None:
        write_mapping(
            Path(path),
            self.with_request_digest().model_dump(mode="json", exclude_none=False),
        )


class KoideRealPilotVerification(StrictModel):
    """Digest/provenance verification; official evidence is opt-in and gated."""

    schema_version: Literal["slac.koide_real_pilot_verification/v0.1"] = (
        KOIDE_REAL_PILOT_VERIFICATION_SCHEMA_VERSION
    )
    request_id: str
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: Literal["BLOCKED", "READY_FOR_PILOT"]
    evidence_label: Literal["official", "test"] = "official"
    official_evidence: bool = False
    real_data_observed: bool = False
    dataset_verified: bool = False
    dataset_archive_verified: bool = False
    split_manifest_verified: bool = False
    config_verified: bool = False
    native_output_verified: bool = False
    external_run_verified: bool = False
    readiness_verified: bool = False
    execution_verified: bool = False
    execution_evidence_verified: bool = False
    execution_log_verified: bool = False
    environment_verified: bool = False
    known_bad_protocol_verified: bool = False
    kpi_protocol_verified: bool = False
    native_output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    native_output_size_bytes: int | None = Field(default=None, ge=0)
    external_run_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    external_run_size_bytes: int | None = Field(default=None, ge=0)
    input_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    readiness_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    execution_log_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    environment_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    execution_command_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    source_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    container_digest: str | None = Field(default=None, pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")
    reasons: list[str] = Field(default_factory=list, min_length=1)
    provenance: KoideRealPilotProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def enforce_claim_gate(self) -> KoideRealPilotVerification:
        complete = all(
            (
                self.dataset_verified,
                self.dataset_archive_verified,
                self.split_manifest_verified,
                self.config_verified,
                self.native_output_verified,
                self.external_run_verified,
                self.readiness_verified,
                self.execution_verified,
                self.execution_evidence_verified,
                self.execution_log_verified,
                self.environment_verified,
                self.known_bad_protocol_verified,
                self.kpi_protocol_verified,
            )
        )
        if self.official_evidence and (not complete or self.status != "READY_FOR_PILOT"):
            raise ValueError("official evidence requires every real-input and execution gate")
        if (
            self.status == "READY_FOR_PILOT"
            and self.evidence_label == "official"
            and not self.official_evidence
        ):
            raise ValueError("official READY_FOR_PILOT verification must set official_evidence")
        if self.official_evidence and self.evidence_label != "official":
            raise ValueError("test evidence cannot be declared official")
        if self.provenance.artifact_sha256 is not None and (
            self.provenance.artifact_sha256 != self.artifact_sha256
        ):
            raise ValueError("verification provenance artifact digest must match artifact_sha256")
        return self

    def with_artifact_digest(self) -> KoideRealPilotVerification:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("artifact_sha256", None)
        provenance.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "provenance": self.provenance.model_copy(update={"artifact_sha256": digest}),
            }
        )

    def verify_artifact_digest(self) -> None:
        expected = self.with_artifact_digest().artifact_sha256
        if self.artifact_sha256 != expected:
            raise ValueError(
                "Koide real pilot verification self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )

    def save(self, path: str | Path) -> None:
        write_mapping(
            Path(path),
            self.with_artifact_digest().model_dump(mode="json", exclude_none=False),
        )


class KoideRealPilotFinalization(StrictModel):
    """Terminal, fail-closed decision after the independent Koide pilot."""

    schema_version: Literal["slac.koide_real_pilot_finalization/v0.1"] = (
        KOIDE_REAL_PILOT_FINALIZATION_SCHEMA_VERSION
    )
    request_id: str
    request_sha256: str = Field(pattern=_SHA256_PATTERN)
    verification_sha256: str = Field(pattern=_SHA256_PATTERN)
    status: Literal["BLOCKED", "PASS", "FAIL", "INCONCLUSIVE", "TEST_ONLY"]
    execution_state: Literal["NOT_EXECUTED", "EXECUTED"] = "NOT_EXECUTED"
    evidence_label: Literal["official", "test"] = "official"
    official_result_claim: Literal[False, True] = False
    pilot_status: str | None = None
    adoption_decision: str | None = None
    pilot_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    native_output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    external_run_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    holdout_kpis_verified: bool = False
    known_bad_controls_verified: bool = False
    reasons: list[str] = Field(default_factory=list, min_length=1)
    provenance: KoideRealPilotProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def enforce_finalization_gate(self) -> KoideRealPilotFinalization:
        if self.official_result_claim and not (
            self.status == "PASS"
            and self.execution_state == "EXECUTED"
            and self.evidence_label == "official"
            and self.pilot_status == "PASS"
            and self.adoption_decision == "ADOPT"
            and self.holdout_kpis_verified
            and self.known_bad_controls_verified
        ):
            raise ValueError(
                "official result claim requires an executed PASS/ADOPT pilot and all gates"
            )
        if self.evidence_label == "test" and self.official_result_claim:
            raise ValueError("test evidence cannot carry an official result claim")
        if (
            self.status == "PASS"
            and not self.official_result_claim
            and self.evidence_label == "official"
        ):
            raise ValueError("official PASS finalization must carry an official result claim")
        if self.provenance.artifact_sha256 is not None and (
            self.provenance.artifact_sha256 != self.artifact_sha256
        ):
            raise ValueError("finalization provenance artifact digest must match artifact_sha256")
        return self

    def with_artifact_digest(self) -> KoideRealPilotFinalization:
        payload = self.model_dump(mode="json", exclude_none=False)
        payload.pop("artifact_sha256", None)
        provenance = cast(dict[str, Any], payload["provenance"])
        provenance.pop("artifact_sha256", None)
        provenance.pop("generated_at", None)
        digest = _canonical_sha256(payload)
        return self.model_copy(
            update={
                "artifact_sha256": digest,
                "provenance": self.provenance.model_copy(update={"artifact_sha256": digest}),
            }
        )

    def verify_artifact_digest(self) -> None:
        expected = self.with_artifact_digest().artifact_sha256
        if self.artifact_sha256 != expected:
            raise ValueError(
                "Koide real pilot finalization self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )

    def save(self, path: str | Path) -> None:
        write_mapping(
            Path(path),
            self.with_artifact_digest().model_dump(mode="json", exclude_none=False),
        )


def koide_real_pilot_request_json_schema() -> dict[str, Any]:
    """Return the generated request schema."""

    return KoideRealPilotRequest.model_json_schema()


def koide_real_pilot_verification_json_schema() -> dict[str, Any]:
    """Return the generated verification schema."""

    return KoideRealPilotVerification.model_json_schema()


def koide_real_pilot_finalization_json_schema() -> dict[str, Any]:
    """Return the generated terminal-decision schema."""

    return KoideRealPilotFinalization.model_json_schema()


def build_koide_real_pilot_request(
    *,
    camera_frame: str = "camera0",
    lidar_frame: str = "lidar0",
    command: Sequence[str] = (),
) -> KoideRealPilotRequest:
    """Build the honest, blocked handoff for one official KITTI raw sequence."""

    stages = [
        KoideExecutionStageRequest(
            name="preprocess",
            argv=[
                "ros2",
                "run",
                "direct_visual_lidar_calibration",
                "preprocess",
                "-a",
                "/tmp/input_bags",
                "/tmp/preprocessed",
            ],
            official_documentation_url=KOIDE_OFFICIAL_EXAMPLE_URL,
            environment={"LC_ALL": "C.UTF-8", "ROS_DOMAIN_ID": "0"},
            working_directory="/tmp",
        ),
        KoideExecutionStageRequest(
            name="initial_guess",
            argv=[
                "ros2",
                "run",
                "direct_visual_lidar_calibration",
                "initial_guess_manual",
                "/tmp/preprocessed",
            ],
            official_documentation_url=KOIDE_OFFICIAL_EXAMPLE_URL,
            environment={
                "LC_ALL": "C.UTF-8",
                "ROS_DOMAIN_ID": "0",
                "DISPLAY": "<operator-display-required>",
            },
            working_directory="/tmp",
            requires_manual_gui=True,
        ),
        KoideExecutionStageRequest(
            name="calibrate",
            argv=[
                "ros2",
                "run",
                "direct_visual_lidar_calibration",
                "calibrate",
                "/tmp/preprocessed",
            ],
            official_documentation_url=KOIDE_OFFICIAL_EXAMPLE_URL,
            environment={"LC_ALL": "C.UTF-8", "ROS_DOMAIN_ID": "0"},
            working_directory="/tmp",
        ),
    ]
    execution = KoideExecutionRequest(
        lock_sha256=_koide_lock_sha256(),
        source_commit=KOIDE_LOCK_SOURCE_COMMIT,
        image_digest=KOIDE_LOCK_HUMBLE_IMAGE,
        stages=stages,
    )
    split = KoideSplitManifest().with_digests()
    artifact_roles: tuple[KoideRequestArtifactRole, ...] = (
        "dataset_archive",
        "dataset_sequence",
        "input_manifest",
        "config",
        "readiness",
        "native_calib_json",
        "external_run",
        "execution_evidence",
        "execution_log",
        "environment",
        "pilot",
    )
    request = KoideRealPilotRequest(
        request_id="koide-real-pilot-kitti-0005-v0-1",
        dataset=KoideRealDatasetPin(),
        execution=execution,
        split=split,
        output_contract=KoideNativeOutputContract(
            lidar_frame=lidar_frame,
            camera_frame=camera_frame,
        ),
        artifacts=[KoideRealArtifactRef(role=role) for role in artifact_roles],
        external_commands=[
            ["docker", "pull", KOIDE_LOCK_HUMBLE_IMAGE],
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--mount",
                "type=bind,src=<input-dir>,dst=/tmp/input_bags,readonly",
                "--mount",
                "type=bind,src=<output-dir>,dst=/tmp/preprocessed,rw",
                KOIDE_LOCK_HUMBLE_IMAGE,
                *stages[0].argv,
            ],
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--mount",
                "type=bind,src=<output-dir>,dst=/tmp/preprocessed,rw",
                KOIDE_LOCK_HUMBLE_IMAGE,
                *stages[1].argv,
            ],
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--mount",
                "type=bind,src=<output-dir>,dst=/tmp/preprocessed,rw",
                KOIDE_LOCK_HUMBLE_IMAGE,
                *stages[2].argv,
            ],
        ],
        operator_checklist=[
            (
                "Register and download only the requested official KITTI raw sequence; "
                "do not redistribute it."
            ),
            (
                "Record archive byte size and SHA-256 after download; populate the dataset "
                "pin with archive_metadata_status=operator_observed and keep the archive "
                "immutable. The official record does not publish these values."
            ),
            (
                "Extract the sequence and run `calibrex kitti lock-benchmark-input` for "
                "the fixed input inventory."
            ),
            (
                "Convert only the predeclared fitting frames to a user-owned ROS 2 bag; "
                "record the converter command, environment, and digest. Never give "
                "holdout frames to Koide."
            ),
            (
                "Run `calibrex doctor --workflow koide` and require readiness=ready "
                "before the external run."
            ),
            (
                "Run the three pinned container stages with network=none, read-only "
                "inputs, a writable output directory, and DISPLAY only for the manual "
                "initial-guess stage."
            ),
            (
                "Record stage stdout/stderr and environment snapshots as digest-bound "
                "execution evidence; import the native calib.json with explicit "
                "camera/lidar frames and source commit/image digests."
            ),
            (
                "Run the frozen Koide pilot; require independent holdout, all twelve "
                "signed controls, and PASS/ADOPT before Autoware export."
            ),
        ],
        provenance=KoideRealPilotProvenance(
            git_commit=git_commit(),
            command=list(command),
        ),
        request_sha256="0" * 64,
    )
    return request.with_request_digest()


def load_koide_real_pilot_request(path: str | Path) -> KoideRealPilotRequest:
    """Load and verify one immutable handoff request."""

    request = KoideRealPilotRequest.model_validate(read_mapping(Path(path)))
    request.verify_request_digest()
    return request


def load_koide_real_pilot_verification(path: str | Path) -> KoideRealPilotVerification:
    """Load and verify one immutable verifier report."""

    verification = KoideRealPilotVerification.model_validate(read_mapping(Path(path)))
    verification.verify_artifact_digest()
    return verification


def load_koide_real_pilot_finalization(path: str | Path) -> KoideRealPilotFinalization:
    """Load and verify one terminal pilot decision."""

    finalization = KoideRealPilotFinalization.model_validate(read_mapping(Path(path)))
    finalization.verify_artifact_digest()
    return finalization


def verify_koide_real_pilot_request(
    request_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    dataset_archive_path: str | Path | None = None,
    input_manifest_path: str | Path | None = None,
    readiness_path: str | Path | None = None,
    config_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
    external_run_path: str | Path | None = None,
    execution_log_path: str | Path | None = None,
    environment_path: str | Path | None = None,
    evidence_label: Literal["official", "test"] = "official",
    command: Sequence[str] = (),
) -> KoideRealPilotVerification:
    """Verify supplied real artifacts without executing Koide.

    Missing or incomplete inputs always produce ``BLOCKED``.  In particular,
    a native JSON file by itself can never become official evidence: it must
    be bound to a successful external-run record, the pinned source/image,
    digest-locked KITTI inputs, readiness, and the independent split.
    """

    request_file = Path(request_path).resolve()
    request = load_koide_real_pilot_request(request_file)
    reasons: list[str] = [
        "request is an external-execution handoff only; no result is claimed until finalization"
    ]
    dataset_verified = False
    archive_verified = False
    split_verified = False
    config_verified = False
    native_verified = False
    external_verified = False
    readiness_verified = False
    execution_verified = False
    execution_evidence_verified = False
    execution_log_verified = False
    environment_verified = False
    known_bad_protocol_verified = len(request.known_bad_controls) == 12
    kpi_protocol_verified = request.acceptance.thresholds is not None
    native_sha: str | None = None
    native_size: int | None = None
    external_sha: str | None = None
    external_size: int | None = None
    manifest_sha: str | None = None
    config_sha: str | None = None
    readiness_sha: str | None = None
    execution_log_sha: str | None = None
    environment_sha: str | None = None
    command_sha: str | None = None
    source_commit: str | None = None
    container_digest: str | None = None
    manifest: Any | None = None
    external_run: ExternalCalibrationRunArtifact | None = None

    if evidence_label == "test":
        reasons.append("evidence_label=test: this verification can never support an official claim")

    lock_path = _resolve_reference(request_file, request.execution.lock_path)
    execution_lock_ok = False
    try:
        lock = load_koide_execution_lock(lock_path)
        if lock.lock_sha256 != request.execution.lock_sha256:
            reasons.append("execution lock SHA-256 does not match the request")
        elif lock.source.commit != request.execution.source_commit:
            reasons.append("execution lock source commit does not match the request")
        elif lock.container.source_image != request.execution.image_digest:
            reasons.append("execution lock image digest does not match the request")
        elif lock.source.repository != request.execution.source_repository:
            reasons.append("execution lock source repository does not match the request")
        else:
            execution_lock_ok = True
    except (OSError, ValueError) as exc:
        reasons.append(f"pinned execution lock is unavailable or invalid: {exc}")

    actual_dataset = _resolve_optional_reference(request_file, dataset_path)
    if actual_dataset is None or not actual_dataset.is_dir():
        reasons.append(
            "real KITTI sequence directory is missing; synthetic/KITTI plumbing cannot "
            "satisfy this gate"
        )
    else:
        dataset_verified = actual_dataset.name == request.dataset.sequence_id
        if evidence_label == "official" and any(
            token in actual_dataset.as_posix().lower() for token in ("synthetic", "fixture", "fake")
        ):
            dataset_verified = False
            reasons.append("synthetic or test fixture path cannot be official KITTI evidence")
        if not dataset_verified:
            reasons.append(
                f"dataset sequence name must be {request.dataset.sequence_id!r}; "
                f"got {actual_dataset.name!r}"
            )
        elif request.dataset.local_sequence_path is not None:
            expected = _resolve_reference(request_file, request.dataset.local_sequence_path)
            if expected != actual_dataset:
                dataset_verified = False
                reasons.append("supplied dataset path does not match the request's bound path")

    archive = _resolve_optional_reference(request_file, dataset_archive_path)
    if archive is not None and archive.is_file():
        observed_size = archive.stat().st_size
        observed_sha = sha256_path(archive)
        expected_size = request.dataset.archive_size_bytes
        expected_sha = request.dataset.archive_sha256
        if expected_size is not None and observed_size != expected_size:
            reasons.append("official dataset archive byte size does not match the request")
        elif expected_sha is not None and observed_sha != expected_sha:
            reasons.append("official dataset archive SHA-256 does not match the request")
        elif request.dataset.archive_metadata_status == "not_published_by_official_record":
            if evidence_label == "official":
                reasons.append(
                    "dataset archive size/SHA-256 was observed but the request does not "
                    "bind it; regenerate the plan with operator_observed metadata"
                )
            else:
                archive_verified = observed_sha is not None
        else:
            archive_verified = observed_sha is not None and expected_size == observed_size
    elif evidence_label == "test":
        archive_verified = True
    else:
        reasons.append(
            "official dataset archive is missing; the official record publishes no "
            "authoritative per-file size/checksum"
        )

    actual_manifest = _resolve_optional_reference(request_file, input_manifest_path)
    if actual_manifest is None or not actual_manifest.is_file():
        reasons.append("digest-locked KITTI input manifest is missing")
    elif actual_dataset is not None and dataset_verified:
        try:
            manifest = load_kitti_benchmark_input(actual_manifest)
            verify_kitti_benchmark_input(manifest, sequence_path=actual_dataset)
            expected_ids = set(request.split.frame_ids)
            if set(manifest.frame_ids) != expected_ids:
                raise ValueError("input manifest frame IDs differ from the predeclared split")
            if manifest.dataset_id != request.dataset.dataset_id:
                raise ValueError("input manifest dataset ID differs from the request")
            if manifest.sequence_id != request.dataset.sequence_id:
                raise ValueError("input manifest sequence ID differs from the request")
            if request.split.frame_ids_sha256 != _frame_ids_digest(request.split.frame_ids):
                raise ValueError("request frame inventory digest is stale")
            if request.split.fitting_frame_ids_sha256 != _frame_ids_digest(
                request.split.fitting_frame_ids
            ):
                raise ValueError("request fitting frame inventory digest is stale")
            if request.split.holdout_frame_ids_sha256 != _frame_ids_digest(
                request.split.holdout_frame_ids
            ):
                raise ValueError("request holdout frame inventory digest is stale")
            split_verified = True
            manifest_sha = sha256_path(actual_manifest)
        except (OSError, ValueError) as exc:
            reasons.append(f"KITTI input manifest verification failed: {exc}")
    if not split_verified:
        reasons.append("independent fitting/holdout split is not digest-verified")

    actual_config = _resolve_optional_reference(request_file, config_path)
    if actual_config is not None and actual_config.is_file():
        config_sha = sha256_path(actual_config)
        config_ref = _artifact_ref(request, "config")
        if config_ref is not None and config_ref.sha256 is not None:
            config_verified = config_sha == config_ref.sha256 and (
                config_ref.size_bytes is None
                or actual_config.stat().st_size == config_ref.size_bytes
            )
            if not config_verified:
                reasons.append(
                    "frozen Calibrex config digest differs from the request artifact reference"
                )
        else:
            config_verified = evidence_label == "test"
            if evidence_label == "official":
                reasons.append("frozen Calibrex config has no digest-bound request reference")
    else:
        reasons.append("frozen Calibrex config is missing")

    actual_readiness = _resolve_optional_reference(request_file, readiness_path)
    if actual_readiness is None or not actual_readiness.is_file():
        reasons.append("Koide readiness artifact is missing")
    else:
        readiness_sha = sha256_path(actual_readiness)
        try:
            readiness = load_koide_readiness(actual_readiness)
            readiness_verified = readiness.status == "ready"
            if not readiness_verified:
                reasons.append(f"Koide readiness status is {readiness.status}, not ready")
            readiness_ref = _artifact_ref(request, "readiness")
            if (
                readiness_ref is not None
                and readiness_ref.sha256 is not None
                and readiness_sha != readiness_ref.sha256
            ):
                readiness_verified = False
                reasons.append("readiness digest differs from the request artifact reference")
        except (OSError, ValueError) as exc:
            reasons.append(f"Koide readiness artifact is invalid: {exc}")

    actual_candidate = _resolve_optional_reference(request_file, candidate_path)
    if actual_candidate is None or not actual_candidate.is_file():
        reasons.append("native Koide calib.json is missing; no official output can be claimed")
    else:
        native_sha = sha256_path(actual_candidate)
        native_size = actual_candidate.stat().st_size
        expected_native_sha = (
            request.output_contract.expected_sha256
            or request.execution.expected_native_output_sha256
        )
        expected_native_size = (
            request.output_contract.expected_size_bytes
            or request.execution.expected_native_output_size_bytes
        )
        if expected_native_sha is not None and native_sha != expected_native_sha:
            reasons.append("native calib.json SHA-256 does not match the expected output digest")
        elif expected_native_size is not None and native_size != expected_native_size:
            reasons.append("native calib.json byte size does not match the expected output size")
        try:
            parse_koide_calib_json(
                actual_candidate,
                lidar_frame=request.output_contract.lidar_frame,
                camera_frame=request.output_contract.camera_frame,
            )
            native_verified = native_sha is not None
        except ValueError as exc:
            reasons.append(f"native Koide calib.json is invalid: {exc}")

    actual_external = _resolve_optional_reference(request_file, external_run_path)
    if actual_external is None or not actual_external.is_file():
        reasons.append("successful external-run artifact is missing")
    else:
        external_sha = sha256_path(actual_external)
        external_size = actual_external.stat().st_size
        try:
            external_run = load_external_run(actual_external)
            external_reasons = _verify_external_run(
                external_run,
                external_path=actual_external,
                request=request,
                candidate_path=actual_candidate,
                native_sha=native_sha,
                manifest=manifest,
                evidence_label=evidence_label,
            )
            reasons.extend(external_reasons)
            external_verified = not external_reasons
            if external_run.execution.command:
                command_sha = _canonical_sha256(external_run.execution.command)
            source_commit = external_run.tool.source_commit
            container_digest = external_run.execution.container_digest
            execution_evidence_verified = external_verified and external_sha is not None
            execution_verified = external_verified and execution_lock_ok
            if (
                external_run.execution.stdout_sha256 is None
                or external_run.execution.stderr_sha256 is None
            ):
                reasons.append("external-run execution is missing stdout/stderr log digests")
                execution_log_verified = False
            else:
                execution_log_verified = True
        except (OSError, ValueError) as exc:
            reasons.append(f"external-run artifact is invalid: {exc}")

    execution_log = _resolve_optional_reference(request_file, execution_log_path)
    if execution_log is not None and execution_log.is_file():
        execution_log_sha = sha256_path(execution_log)
        execution_log_verified = execution_log_sha is not None and execution_log_verified
    elif evidence_label == "official":
        reasons.append("digest-bound execution log is missing")
        execution_log_verified = False

    environment = _resolve_optional_reference(request_file, environment_path)
    if environment is not None and environment.is_file():
        environment_sha = sha256_path(environment)
        environment_verified = environment_sha is not None
        if (
            request.execution.environment_sha256 is not None
            and environment_sha != request.execution.environment_sha256
        ):
            environment_verified = False
            reasons.append("execution environment digest does not match the request")
    elif evidence_label == "official":
        reasons.append("digest-bound execution environment snapshot is missing")
        environment_verified = False
    else:
        environment_verified = external_verified

    if (
        request.execution.command_sha256 is not None
        and command_sha != request.execution.command_sha256
    ):
        reasons.append("external execution command digest does not match the request")
        execution_verified = False

    # A test fixture may exercise every machine gate without pretending to be
    # official evidence.  Official evidence still requires every durable
    # archive/config/log/environment binding above.
    complete = all(
        (
            dataset_verified,
            archive_verified,
            split_verified,
            config_verified,
            native_verified,
            external_verified,
            readiness_verified,
            execution_verified,
            execution_evidence_verified,
            execution_log_verified,
            environment_verified,
            known_bad_protocol_verified,
            kpi_protocol_verified,
        )
    )
    if complete:
        reasons = [
            (
                "test-labelled external evidence satisfies the machine gates; it is not "
                "an official result"
                if evidence_label == "test"
                else (
                    "real KITTI inputs, pinned external execution, native output, execution "
                    "logs/environment, readiness, and split provenance verified; finalize only "
                    "after the frozen pilot"
                )
            )
        ]
    verification = KoideRealPilotVerification(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
        status="READY_FOR_PILOT" if complete else "BLOCKED",
        evidence_label=evidence_label,
        official_evidence=complete and evidence_label == "official",
        real_data_observed=dataset_verified,
        dataset_verified=dataset_verified,
        dataset_archive_verified=archive_verified,
        split_manifest_verified=split_verified,
        config_verified=config_verified,
        native_output_verified=native_verified,
        external_run_verified=external_verified,
        readiness_verified=readiness_verified,
        execution_verified=execution_verified,
        execution_evidence_verified=execution_evidence_verified,
        execution_log_verified=execution_log_verified,
        environment_verified=environment_verified,
        known_bad_protocol_verified=known_bad_protocol_verified,
        kpi_protocol_verified=kpi_protocol_verified,
        native_output_sha256=native_sha,
        native_output_size_bytes=native_size,
        external_run_sha256=external_sha,
        external_run_size_bytes=external_size,
        input_manifest_sha256=manifest_sha,
        config_sha256=config_sha,
        readiness_sha256=readiness_sha,
        execution_log_sha256=execution_log_sha,
        environment_sha256=environment_sha,
        execution_command_sha256=command_sha,
        source_commit=source_commit,
        container_digest=container_digest,
        reasons=reasons,
        provenance=KoideRealPilotProvenance(
            git_commit=git_commit(),
            command=list(command),
        ),
        artifact_sha256="0" * 64,
    )
    return verification.with_artifact_digest()


def _artifact_ref(
    request: KoideRealPilotRequest, role: KoideRequestArtifactRole
) -> KoideRealArtifactRef | None:
    """Return the unique request reference for one evidence role."""

    for ref in request.artifacts:
        if ref.role == role:
            return ref
    return None


def _verify_external_run(
    external_run: ExternalCalibrationRunArtifact,
    *,
    external_path: Path,
    request: KoideRealPilotRequest,
    candidate_path: Path | None,
    native_sha: str | None,
    manifest: Any | None,
    evidence_label: Literal["official", "test"],
) -> list[str]:
    """Return all concrete reasons an external-run record is not admissible.

    This function deliberately checks the generic artifact again at the real
    pilot boundary.  A schema-valid imported ``calib.json`` is not execution
    evidence, and a claimed train/holdout digest is not enough when its input
    inventory contains holdout files.
    """

    reasons: list[str] = []
    execution = external_run.execution
    command = list(execution.command)
    command_values = command + [
        argument for stage in execution.stages for argument in stage.command
    ]
    if external_run.status != "success":
        reasons.append(f"external-run status is {external_run.status}, not success")
    if not execution.attempted:
        reasons.append("external-run artifact does not prove an attempted execution")
    if evidence_label == "official" and execution.mode != "container":
        reasons.append("official Koide evidence must record execution.mode=container")
    if external_run.tool.source_repository != request.execution.source_repository:
        reasons.append("external-run source repository does not match the pinned request")
    if external_run.tool.source_commit != request.execution.source_commit:
        reasons.append("external-run source commit does not match the pinned request")
    if external_run.tool.license_spdx != request.execution.source_license_spdx:
        reasons.append("external-run license does not match the pinned MIT source lock")
    if external_run.tool.license_boundary not in {"container", "subprocess", "imported"}:
        reasons.append("external-run license boundary is not an allowed external boundary")
    if execution.container_digest != request.execution.image_digest:
        reasons.append("external-run container digest does not match the pinned request")
    if execution.network_mode != request.execution.network_mode:
        reasons.append("external-run network mode does not match the locked network=none policy")
    if request.execution.engine not in command:
        reasons.append("external-run command does not record the locked container engine")
    if evidence_label == "official" and "--read-only" not in command:
        reasons.append("official external command does not record a read-only root filesystem")
    if evidence_label == "official" and not any(
        ":ro" in mount or "readonly" in mount for mount in execution.input_mounts
    ):
        reasons.append("official external run does not record a read-only input mount")
    if evidence_label == "official" and (
        execution.output_mount is None or ":rw" not in execution.output_mount
    ):
        reasons.append("official external run does not record a writable output mount")
    if _contains_superglue(command_values) or _contains_superglue(external_run.warnings):
        reasons.append("commercial Koide evidence contains a SuperGlue reference")
    if not external_run.train_data_isolation.declared:
        reasons.append("external-run training/holdout isolation is not declared")
    if external_run.train_data_isolation.training_data_ids_sha256 != _frame_ids_digest(
        request.split.fitting_frame_ids
    ):
        reasons.append("external-run fitting frame-ID digest does not match the request")
    if external_run.train_data_isolation.holdout_data_ids_sha256 != _frame_ids_digest(
        request.split.holdout_frame_ids
    ):
        reasons.append("external-run holdout frame-ID digest does not match the request")
    if not external_run.train_data_isolation.evidence:
        reasons.append("external-run training/holdout isolation has no evidence statement")

    source_path = external_run.provenance.source_artifact
    resolved_source = (
        _resolve_evidence_path(external_path.parent, source_path)
        if source_path is not None
        else None
    )
    if candidate_path is None or resolved_source is None:
        reasons.append("external-run does not bind a native output path")
    elif resolved_source != candidate_path.resolve():
        reasons.append("external-run native output path differs from supplied candidate")
    else:
        source_digest = sha256_path(resolved_source)
        if source_digest is None or source_digest != native_sha:
            reasons.append("external-run native output digest does not match supplied candidate")
        if external_run.provenance.source_artifact_sha256 != native_sha:
            reasons.append("external-run source artifact digest does not match native output")

    output_items = [item for item in external_run.artifacts if item.role == "output"]
    if not output_items:
        reasons.append("external-run has no digest-bound output artifact")
    elif candidate_path is not None and not any(
        _resolve_evidence_path(external_path.parent, item.path) == candidate_path.resolve()
        and item.sha256 == native_sha
        for item in output_items
    ):
        reasons.append("external-run output inventory does not bind the supplied calib.json")

    if manifest is None:
        reasons.append("external input inventory cannot be checked without the KITTI manifest")
    else:
        reasons.extend(
            _verify_external_input_inventory(
                external_run,
                manifest,
                fitting_frame_ids=request.split.fitting_frame_ids,
                holdout_frame_ids=request.split.holdout_frame_ids,
            )
        )
    return reasons


def _verify_external_input_inventory(
    external_run: ExternalCalibrationRunArtifact,
    manifest: Any,
    *,
    fitting_frame_ids: Sequence[str],
    holdout_frame_ids: Sequence[str],
) -> list[str]:
    """Check every external input digest and reject fitting/holdout overlap."""

    reasons: list[str] = []
    input_items = [item for item in external_run.artifacts if item.role == "input"]
    if not input_items:
        return ["external-run has no digest-bound fitting input artifacts"]
    common_root = Path(manifest.source_sequence_path).resolve().parent
    holdout_ids = set(holdout_frame_ids)
    fitting_ids = set(fitting_frame_ids)
    holdout_paths = {
        (common_root / item.path).resolve()
        for item in manifest.files
        if item.frame_id is not None and item.frame_id in holdout_ids
    }
    fitting_paths = {
        (common_root / item.path).resolve()
        for item in manifest.files
        if item.frame_id is not None and item.frame_id in fitting_ids
    }
    observed_input_paths: list[Path] = []
    for item in input_items:
        path = _resolve_evidence_path(common_root, item.path)
        observed_input_paths.append(path)
        observed_sha = sha256_path(path)
        observed_size = _path_size(path)
        if observed_sha is None:
            reasons.append(f"external input artifact is missing: {path}")
        elif observed_sha != item.sha256:
            reasons.append(f"external input digest mismatch: {path}")
        elif observed_size != item.size_bytes:
            reasons.append(f"external input size mismatch: {path}")
        if path.is_dir() and any(_path_is_within(holdout, path) for holdout in holdout_paths):
            reasons.append("external fitting input directory contains holdout files")
        elif path in holdout_paths:
            reasons.append(f"external fitting input overlaps holdout file: {path}")
    if external_run.digests.input_sha256 is None:
        reasons.append("external-run has no combined fitting-input digest")
    else:
        selected = sorted((str(item.path), item.sha256) for item in input_items)
        observed = hashlib.sha256(
            "\n".join(f"{path}\0{digest}" for path, digest in selected).encode("utf-8")
        ).hexdigest()
        if external_run.digests.input_sha256 != observed:
            reasons.append("external fitting-input combined digest does not match its inventory")
    for expected in sorted(fitting_paths):
        if not any(
            path == expected or _path_is_within(expected, path) for path in observed_input_paths
        ):
            reasons.append(f"external fitting input inventory does not bind train file: {expected}")
    return reasons


def finalize_koide_real_pilot(
    request_path: str | Path,
    verification_path: str | Path,
    pilot_path: str | Path | None,
    *,
    output_path: str | Path | None = None,
    evidence_label: Literal["official", "test"] | None = None,
    command: Sequence[str] = (),
) -> KoideRealPilotFinalization:
    """Finalize a verified external run against the frozen independent pilot.

    Finalization never executes Koide.  It consumes the existing independent
    ``koide-pilot`` artifact and checks that its candidate, split, readiness,
    execution identity, KPI values, and all twelve signed controls are bound to
    the earlier verification report.  Missing or failed evidence remains a
    machine-readable non-adoption state.
    """

    request_file = Path(request_path).resolve()
    verification_file = Path(verification_path).resolve()
    request = load_koide_real_pilot_request(request_file)
    verification = load_koide_real_pilot_verification(verification_file)
    label = evidence_label or verification.evidence_label
    reasons: list[str] = []
    pilot: KoidePilotArtifact | None = None
    pilot_sha: str | None = None
    holdout_kpis_verified = False
    known_bad_verified = False
    execution_state: Literal["NOT_EXECUTED", "EXECUTED"] = (
        "EXECUTED" if verification.external_run_verified else "NOT_EXECUTED"
    )

    if verification.request_id != request.request_id:
        reasons.append("verification request_id does not match the request")
    if verification.request_sha256 != request.request_sha256:
        reasons.append("verification request digest does not match the request")
    if label == "official" and not verification.official_evidence:
        reasons.append("official finalization requires verification.official_evidence=true")
    if verification.status != "READY_FOR_PILOT":
        reasons.append("verification is BLOCKED; no pilot result can be finalized")

    if pilot_path is None:
        reasons.append("independent Koide pilot artifact is missing")
    else:
        pilot_file = Path(pilot_path).resolve()
        if not pilot_file.is_file():
            reasons.append("independent Koide pilot artifact path is not a file")
        else:
            pilot_sha = sha256_path(pilot_file)
            try:
                pilot = load_koide_pilot(pilot_file)
            except (OSError, ValueError) as exc:
                reasons.append(f"independent Koide pilot artifact is invalid: {exc}")

    if pilot is not None:
        if pilot.readiness_status != "ready":
            reasons.append("independent Koide pilot readiness is not ready")
        if pilot.external_run_status != "success":
            reasons.append("independent Koide pilot external-run status is not success")
        if not pilot.execution.attempted:
            reasons.append("independent Koide pilot does not prove an attempted execution")
        if pilot.execution.source_commit != request.execution.source_commit:
            reasons.append("independent Koide pilot source commit differs from request")
        if pilot.execution.container_digest != request.execution.image_digest:
            reasons.append("independent Koide pilot container digest differs from request")
        if pilot.provenance.candidate_output_sha256 != verification.native_output_sha256:
            reasons.append("independent Koide pilot native output digest differs from verification")
        if pilot.provenance.input_manifest_sha256 != verification.input_manifest_sha256:
            reasons.append(
                "independent Koide pilot input manifest digest differs from verification"
            )
        if set(pilot.train_frame_ids) != set(request.split.fitting_frame_ids):
            reasons.append("independent Koide pilot fitting frame IDs differ from request")
        if set(pilot.holdout_frame_ids) != set(request.split.holdout_frame_ids):
            reasons.append("independent Koide pilot holdout frame IDs differ from request")

        metrics = pilot.candidate_metrics
        thresholds = request.acceptance.thresholds
        if metrics is None:
            reasons.append("independent Koide pilot has no candidate holdout metrics")
        else:
            metric_failures: list[str] = []
            if len(metrics.holdout_frame_ids) < thresholds.min_holdout_frames:
                metric_failures.append("minimum independent holdout frames")
            if metrics.holdout_projection_ratio is None or (
                metrics.holdout_projection_ratio < thresholds.min_holdout_projection_ratio
            ):
                metric_failures.append("holdout projection ratio")
            if metrics.holdout_edge_alignment is None or (
                metrics.holdout_edge_alignment < thresholds.min_holdout_edge_alignment
            ):
                metric_failures.append("holdout edge alignment")
            if metrics.holdout_depth_edge_alignment is None or (
                metrics.holdout_depth_edge_alignment < thresholds.min_holdout_depth_edge_alignment
            ):
                metric_failures.append("holdout depth-edge alignment")
            delta = pilot.candidate_to_reference
            if delta.translation_delta_m is not None and (
                delta.translation_delta_m > thresholds.max_candidate_reference_translation_m
            ):
                metric_failures.append("candidate/reference translation delta")
            if delta.rotation_delta_deg is not None and (
                delta.rotation_delta_deg > thresholds.max_candidate_reference_rotation_deg
            ):
                metric_failures.append("candidate/reference rotation delta")
            holdout_kpis_verified = not metric_failures
            if metric_failures:
                reasons.append("Koide pilot KPI failure: " + ", ".join(metric_failures))

        expected_controls = {item.control_id for item in request.known_bad_controls}
        observed_controls = {item.control_id for item in pilot.known_bad_controls}
        known_bad_verified = (
            len(pilot.known_bad_controls) == 12
            and observed_controls == expected_controls
            and all(item.detected and item.status == "pass" for item in pilot.known_bad_controls)
            and (pilot.known_bad_detectable_fraction or 0.0)
            >= thresholds.min_known_bad_detectable_fraction
        )
        if not known_bad_verified:
            reasons.append("known-bad/KPI failure: all twelve signed controls were not detected")
        if pilot.status != "PASS" or pilot.adoption_decision != "ADOPT":
            reasons.append(
                f"independent Koide pilot is not PASS/ADOPT (status={pilot.status}, "
                f"adoption={pilot.adoption_decision})"
            )

    success = (
        not reasons
        and pilot is not None
        and verification.status == "READY_FOR_PILOT"
        and verification.official_evidence == (label == "official")
        and holdout_kpis_verified
        and known_bad_verified
        and pilot.status == "PASS"
        and pilot.adoption_decision == "ADOPT"
    )
    if success and label == "official":
        status: Literal["BLOCKED", "PASS", "FAIL", "INCONCLUSIVE", "TEST_ONLY"] = "PASS"
        official_claim = True
        final_reason = (
            "official Koide output and independent pilot PASS/ADOPT satisfy every frozen gate"
        )
    elif success:
        status = "TEST_ONLY"
        official_claim = False
        final_reason = (
            "test-labelled evidence passed the machine gates; no official result is claimed"
        )
    else:
        official_claim = False
        if verification.external_run_verified:
            status = "FAIL" if pilot is not None and pilot.status == "FAIL" else "BLOCKED"
        else:
            status = "BLOCKED"
        final_reason = (
            "; ".join(reasons) if reasons else "pilot finalization did not satisfy every gate"
        )
    finalization = KoideRealPilotFinalization(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
        verification_sha256=verification.artifact_sha256,
        status=status,
        execution_state=execution_state,
        evidence_label=label,
        official_result_claim=official_claim,
        pilot_status=pilot.status if pilot is not None else None,
        adoption_decision=pilot.adoption_decision if pilot is not None else None,
        pilot_sha256=pilot_sha,
        native_output_sha256=verification.native_output_sha256,
        external_run_sha256=verification.external_run_sha256,
        holdout_kpis_verified=holdout_kpis_verified,
        known_bad_controls_verified=known_bad_verified,
        reasons=[final_reason],
        provenance=KoideRealPilotProvenance(
            git_commit=git_commit(),
            command=list(command),
        ),
        artifact_sha256="0" * 64,
    ).with_artifact_digest()
    if output_path is not None:
        finalization.save(output_path)
    return finalization


def _contains_superglue(value: object) -> bool:
    """Recursively detect a SuperGlue reference in commands or metadata."""

    if isinstance(value, Mapping):
        return any(_contains_superglue(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_superglue(item) for item in value)
    return isinstance(value, str) and "superglue" in value.lower()


def _resolve_evidence_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _path_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return None


def _path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _koide_lock_sha256() -> str:
    """Read the checked-in lock digest without hard-coding a second value."""

    lock_path = (
        Path(__file__).resolve().parents[3] / "examples" / "official" / "koide_execution_lock.yaml"
    )
    try:
        return load_koide_execution_lock(lock_path).lock_sha256
    except (OSError, ValueError):
        # The checked-in example is required by the repository tests.  Keep a
        # deterministic fallback for installed wheels where examples are not
        # packaged, while still requiring the actual lock at verification time.
        return "acb4add1590992e060fb43e09f998644e6a8d65410406c1df09bba267c675f28"


def _resolve_optional_reference(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return _resolve_reference(base, value)


def _resolve_reference(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base.parent / path
    return path.resolve()


def _frame_ids_digest(frame_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(set(frame_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


__all__ = [
    "KOIDE_REAL_PILOT_FINALIZATION_SCHEMA_VERSION",
    "KOIDE_REAL_PILOT_REQUEST_SCHEMA_VERSION",
    "KOIDE_REAL_PILOT_VERIFICATION_SCHEMA_VERSION",
    "KoideExecutionRequest",
    "KoideExecutionStageRequest",
    "KoideKnownBadInjection",
    "KoideNativeOutputContract",
    "KoideRealAcceptancePolicy",
    "KoideRealArtifactRef",
    "KoideRealDatasetPin",
    "KoideRealPilotFinalization",
    "KoideRealPilotProvenance",
    "KoideRealPilotRequest",
    "KoideRealPilotVerification",
    "KoideSplitManifest",
    "build_koide_real_pilot_request",
    "finalize_koide_real_pilot",
    "koide_real_pilot_finalization_json_schema",
    "koide_real_pilot_request_json_schema",
    "koide_real_pilot_verification_json_schema",
    "load_koide_real_pilot_finalization",
    "load_koide_real_pilot_request",
    "load_koide_real_pilot_verification",
    "verify_koide_real_pilot_request",
]
