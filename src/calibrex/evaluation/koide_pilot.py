"""Frozen acceptance evidence for an external Koide camera--LiDAR candidate.

The Koide implementation is intentionally outside this module.  A pilot run
consumes either a schema-valid ``external-run`` artifact or the native Koide
``calib.json`` through the existing importer, then evaluates the frozen pose
with the common KITTI camera--LiDAR projection machinery.  No Calibrex solver
is invoked and no holdout value is used to alter the candidate.

This boundary is deliberately stricter than the generic external adapter:
missing provenance, missing readiness, stale inputs, an unavailable official
execution environment, and incomplete holdout evidence are represented as
machine-readable non-adoption decisions rather than being promoted to PASS.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from calibrex import __version__
from calibrex.core.exceptions import CalibrexError
from calibrex.core.external_run import ExternalCalibrationRunArtifact, load_external_run
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.koide_readiness import load_koide_readiness
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import (
    StrictModel,
    TransformEstimateProvenance,
    TransformQuality,
    TransformResult,
)
from calibrex.data.kitti import find_camera_lidar_pairs, read_velodyne_to_camera_transform
from calibrex.data.kitti_benchmark import (
    load_kitti_benchmark_input,
    verify_kitti_benchmark_input,
)
from calibrex.evaluation.lidar_camera_comparison import (
    LidarCameraCandidateScore,
    evaluate_lidar_camera_candidates_on_kitti,
)
from calibrex.importers.koide import import_koide_result

KOIDE_PILOT_SCHEMA_VERSION: Literal["slac.koide_pilot/v0.1"] = "slac.koide_pilot/v0.1"
KOIDE_PILOT_PROTOCOL_ID = "koide_external_candidate_kitti_holdout/v0.1"
KOIDE_SOURCE_REPOSITORY = "https://github.com/koide3/direct_visual_lidar_calibration"

PilotStatus = Literal["PASS", "WARN", "FAIL", "INCONCLUSIVE", "BLOCKED"]
PilotAdoptionDecision = Literal[
    "ADOPT",
    "ADOPT_WITH_WARNING",
    "DO_NOT_ADOPT",
    "INCONCLUSIVE",
    "BLOCKED",
]
PilotExecutionStatus = Literal["not_attempted", "attempted", "blocked", "unavailable"]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class KoidePilotThresholds(StrictModel):
    """Frozen evaluation and adoption gates.

    The defaults are intentionally conservative for a small pilot.  A
    published run should serialize this object and never rely on defaults at
    review time.
    """

    max_frames: int = Field(default=50, ge=2)
    max_points: int = Field(default=4000, ge=1)
    holdout_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    split_seed: int = 20260823
    min_holdout_frames: int = Field(default=1, ge=1)
    min_holdout_projection_ratio: float = Field(default=0.50, ge=0.0, le=1.0)
    min_holdout_edge_alignment: float = Field(default=0.20, ge=0.0, le=1.0)
    min_holdout_depth_edge_alignment: float = Field(default=0.20, ge=0.0, le=1.0)
    max_candidate_reference_translation_m: float = Field(default=0.10, gt=0.0)
    max_candidate_reference_rotation_deg: float = Field(default=2.0, gt=0.0)
    known_bad_rotation_deg: float = Field(default=5.0, gt=0.0)
    known_bad_translation_m: float = Field(default=0.25, gt=0.0)
    known_bad_min_metric_delta: float = Field(default=0.01, gt=0.0)
    min_known_bad_detectable_fraction: float = Field(default=0.80, ge=0.0, le=1.0)
    require_readiness_ready: bool = True
    require_training_isolation: bool = True
    require_candidate_reference_comparison: bool = True


class KoidePilotInputDigest(StrictModel):
    """One digest-bound source used by the pilot."""

    role: str = Field(min_length=1)
    path: str
    frame_id: str | None = None
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


class KoidePilotExecutionManifest(StrictModel):
    """Exact execution intent and environment evidence for a pilot."""

    command: list[str] = Field(default_factory=list)
    status: PilotExecutionStatus = "not_attempted"
    attempted: bool = False
    reason: str = Field(min_length=1)
    docker_available: bool = False
    podman_available: bool = False
    engine: Literal["docker", "podman", "none"] = "none"
    source_repository: str = KOIDE_SOURCE_REPOSITORY
    source_commit: str | None = None
    container_digest: str | None = Field(
        default=None, pattern=r"^(?:[^@\s]+@)?sha256:[0-9a-f]{64}$"
    )


class KoidePilotTransform(StrictModel):
    """A frame-bound transform used in candidate/reference scoring."""

    convention: Literal["T_parent_child"] = "T_parent_child"
    parent: str
    child: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)
    source_path: str | None = None
    source_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    def as_se3(self) -> SE3:
        """Return the validated transform as an SE(3) value."""

        return SE3.from_lists(self.translation_m, self.rotation_quat_xyzw)

    def as_transform_result(self, *, source_run: str) -> TransformResult:
        """Build a result transform suitable for the Autoware adapter."""

        return TransformResult(
            parent=self.parent,
            child=self.child,
            translation_m=list(self.translation_m),
            rotation_quat_xyzw=list(self.rotation_quat_xyzw),
            quality=TransformQuality(grade="pass"),
            estimate_id="koide_pilot_candidate",
            provenance=TransformEstimateProvenance(
                producer="external_tool",
                execution_mode="imported",
                role_in_comparison="output",
                evidence_level="algorithmically_refined",
                source="Koide pilot admissible candidate",
                source_path=self.source_path,
                tool_name="direct_visual_lidar_calibration",
                adapter_version="calibrex.koide_pilot/v0.1",
                notes=[
                    f"source_sha256={self.source_sha256}",
                    f"pilot_run_id={source_run}",
                ],
            ),
        )


class KoidePilotMetricSet(StrictModel):
    """Train/holdout projection evidence for one frozen pose."""

    candidate_id: str
    train_frame_ids: list[str] = Field(default_factory=list)
    holdout_frame_ids: list[str] = Field(default_factory=list)
    train_frame_count: int = Field(ge=0)
    holdout_frame_count: int = Field(ge=0)
    scored_frame_count: int = Field(ge=0)
    train_projection_ratio: float | None = None
    holdout_projection_ratio: float | None = None
    train_edge_alignment: float | None = None
    holdout_edge_alignment: float | None = None
    train_depth_edge_alignment: float | None = None
    holdout_depth_edge_alignment: float | None = None
    training_isolation_declared: bool = False
    evaluation_policy: str = (
        "frozen candidate; deterministic timestamp-ordered frame split; "
        "holdout is never used for fitting"
    )


class KoidePilotReferenceDelta(StrictModel):
    """SE(3) candidate-to-reference comparison."""

    declared: bool
    reference_candidate_id: str = "kitti_dataset_reference"
    candidate_id: str = "koide_external_candidate"
    translation_delta_m: float | None = None
    rotation_delta_deg: float | None = None
    convention: str = "reference.inverse().compose(candidate); T_camera_lidar"
    reason: str = Field(min_length=1)


class KoidePilotKnownBadControl(StrictModel):
    """One mandatory signed six-DoF known-bad control."""

    control_id: str
    dof: Literal["roll", "pitch", "yaw", "x", "y", "z"]
    sign: Literal["+", "-"]
    amount: float = Field(gt=0.0)
    unit: Literal["deg", "m"]
    candidate_holdout_projection_ratio: float | None = None
    control_holdout_projection_ratio: float | None = None
    candidate_holdout_edge_alignment: float | None = None
    control_holdout_edge_alignment: float | None = None
    candidate_holdout_depth_edge_alignment: float | None = None
    control_holdout_depth_edge_alignment: float | None = None
    metric_delta_projection_ratio: float | None = None
    metric_delta_edge_alignment: float | None = None
    metric_delta_depth_edge_alignment: float | None = None
    detected: bool = False
    status: Literal["pass", "fail", "inconclusive", "blocked"]
    reason: str = Field(min_length=1)


class KoidePilotProvenance(StrictModel):
    """Complete lineage for the pilot decision."""

    producer: Literal["calibrex"] = "calibrex"
    calibrex_version: str = __version__
    tool_name: str = "calibrex.koide-pilot"
    tool_version: str = __version__
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    protocol_id: str = KOIDE_PILOT_PROTOCOL_ID
    dataset_path: str
    dataset_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    config_path: str | None = None
    config_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    readiness_path: str | None = None
    readiness_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    candidate_path: str | None = None
    candidate_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    candidate_output_path: str | None = None
    candidate_output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    input_manifest_path: str | None = None
    input_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    input_digests: dict[str, str] = Field(default_factory=dict)
    fitting_input_digests: dict[str, str] = Field(default_factory=dict)
    holdout_input_digests: dict[str, str] = Field(default_factory=dict)
    fitting_frame_ids_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    holdout_frame_ids_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifact_digest_scope: str = (
        "canonical artifact excluding artifact_sha256 and provenance.generated_at"
    )

    @model_validator(mode="after")
    def normalize_input_digests(self) -> KoidePilotProvenance:
        for name in ("input_digests", "fitting_input_digests", "holdout_input_digests"):
            values = getattr(self, name)
            for key, digest in values.items():
                if len(digest) != 64:
                    raise ValueError(f"input digest {key!r} must be SHA-256")
                int(digest, 16)
            setattr(self, name, {key: digest.lower() for key, digest in values.items()})
        return self


class KoidePilotArtifact(StrictModel):
    """Schema-valid frozen candidate acceptance and adoption decision."""

    schema_version: Literal["slac.koide_pilot/v0.1"] = KOIDE_PILOT_SCHEMA_VERSION
    pilot_id: str
    protocol_id: str = KOIDE_PILOT_PROTOCOL_ID
    status: PilotStatus
    adoption_decision: PilotAdoptionDecision
    admissible: bool = False
    reason: str = Field(min_length=1)
    dataset_path: str
    camera_frame: str
    lidar_frame: str
    camera_stream: str = "image_02"
    lidar_stream: str = "velodyne_points"
    selected_frame_ids: list[str] = Field(default_factory=list)
    train_frame_ids: list[str] = Field(default_factory=list)
    holdout_frame_ids: list[str] = Field(default_factory=list)
    split_policy: str = (
        "timestamp-ordered, seeded, frame-disjoint split; candidate frozen before scoring"
    )
    thresholds: KoidePilotThresholds
    readiness_status: Literal["ready", "warn", "blocked", "missing"]
    external_run_status: str = "missing"
    candidate_transform_camera_lidar: KoidePilotTransform | None = None
    reference_transform_camera_lidar: KoidePilotTransform | None = None
    candidate_metrics: KoidePilotMetricSet | None = None
    reference_metrics: KoidePilotMetricSet | None = None
    candidate_to_reference: KoidePilotReferenceDelta
    known_bad_controls: list[KoidePilotKnownBadControl] = Field(default_factory=list)
    known_bad_required_count: int = 12
    known_bad_detected_count: int = 0
    known_bad_detectable_fraction: float | None = None
    input_files: list[KoidePilotInputDigest] = Field(default_factory=list)
    fitting_input_files: list[KoidePilotInputDigest] = Field(default_factory=list)
    holdout_input_files: list[KoidePilotInputDigest] = Field(default_factory=list)
    execution: KoidePilotExecutionManifest
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: KoidePilotProvenance
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def enforce_admissibility(self) -> KoidePilotArtifact:
        if self.admissible and (self.status != "PASS" or self.adoption_decision != "ADOPT"):
            raise ValueError("only a PASS/ADOPT pilot may be admissible")
        if self.status == "PASS" and not self.admissible:
            raise ValueError("PASS pilot must be admissible")
        return self

    def with_artifact_digest(self) -> KoidePilotArtifact:
        """Return a copy with a deterministic canonical artifact digest."""

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
        """Raise when either copy of the pilot self-digest is stale."""

        expected = self.with_artifact_digest().artifact_sha256
        if self.artifact_sha256 != expected:
            raise ValueError(
                "Koide pilot artifact self-digest mismatch: "
                f"declared={self.artifact_sha256}, expected={expected}"
            )
        if self.provenance.artifact_sha256 != self.artifact_sha256:
            raise ValueError(
                "Koide pilot provenance artifact_sha256 does not match artifact_sha256"
            )

    def save(self, path: str | Path) -> None:
        """Persist the digest-complete pilot artifact."""

        write_mapping(
            Path(path), self.with_artifact_digest().model_dump(mode="json", exclude_none=False)
        )


def koide_pilot_json_schema() -> dict[str, Any]:
    """Return the generated JSON Schema for the pilot artifact."""

    return KoidePilotArtifact.model_json_schema()


def load_koide_pilot(path: str | Path) -> KoidePilotArtifact:
    """Load a schema-valid pilot artifact."""

    artifact = KoidePilotArtifact.model_validate(read_mapping(Path(path)))
    artifact.verify_artifact_digest()
    return artifact


def run_koide_pilot(
    dataset_path: str | Path,
    candidate_path: str | Path | None,
    *,
    output_directory: str | Path,
    config_path: str | Path | None = None,
    readiness_path: str | Path | None = None,
    input_manifest_path: str | Path | None = None,
    camera_frame: str = "camera0",
    lidar_frame: str = "lidar0",
    camera_stream: str = "image_02",
    lidar_stream: str = "velodyne_points",
    thresholds: KoidePilotThresholds | None = None,
    command: Sequence[str] = (),
    official_command: Sequence[str] = (),
    tool_source_commit: str | None = None,
    container_digest: str | None = None,
) -> KoidePilotArtifact:
    """Evaluate one frozen Koide candidate on a KITTI holdout.

    ``candidate_path`` may be an ``external-run`` artifact or an official
    native ``calib.json``.  If it is absent, or if a required input is
    unavailable, a blocked artifact is still materialized in ``output`` with
    the exact command and manifest so an official run cannot be accidentally
    represented as a successful benchmark.
    """

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path(dataset_path).resolve()
    policy = thresholds or KoidePilotThresholds()
    config = Path(config_path).resolve() if config_path is not None else None
    readiness = Path(readiness_path).resolve() if readiness_path is not None else None
    input_manifest = (
        Path(input_manifest_path).resolve() if input_manifest_path is not None else None
    )
    candidate = Path(candidate_path).resolve() if candidate_path is not None else None
    provenance = KoidePilotProvenance(
        command=list(command),
        git_commit=git_commit(),
        dataset_path=str(dataset),
        dataset_sha256=_digest_path_or_tree(dataset),
        config_path=str(config) if config is not None else None,
        config_sha256=_digest_file(config),
        readiness_path=str(readiness) if readiness is not None else None,
        readiness_sha256=_digest_file(readiness),
        candidate_path=str(candidate) if candidate is not None else None,
        candidate_sha256=_digest_file(candidate),
        input_manifest_path=str(input_manifest) if input_manifest is not None else None,
        input_manifest_sha256=_digest_file(input_manifest),
    )
    execution = _execution_manifest(
        official_command,
        source_commit=tool_source_commit,
        container_digest=container_digest,
    )
    blockers: list[str] = []
    warnings: list[str] = []
    # A2D2 is useful for the real-data readiness plumbing, but it is not a
    # KITTI raw sequence and therefore cannot be scored by this pilot's
    # projection/reference protocol.  Keep the distinction in the artifact
    # itself so a blocked diagnostic cannot be mistaken for an official run.
    if "a2d2" in dataset.as_posix().lower():
        blockers.append(
            "A2D2 is unsupported by the Koide KITTI pilot; this artifact is a "
            "readiness diagnostic only and makes no official Koide run or score claim"
        )
    input_files: list[KoidePilotInputDigest] = []
    fitting_input_files: list[KoidePilotInputDigest] = []
    holdout_input_files: list[KoidePilotInputDigest] = []
    selected_frame_ids: list[str] = []
    train_frame_ids: list[str] = []
    holdout_frame_ids: list[str] = []
    readiness_status: Literal["ready", "warn", "blocked", "missing"] = "missing"
    external_run: ExternalCalibrationRunArtifact | None = None
    candidate_transform: KoidePilotTransform | None = None
    reference_transform: KoidePilotTransform | None = None
    candidate_metrics: KoidePilotMetricSet | None = None
    reference_metrics: KoidePilotMetricSet | None = None
    controls: list[KoidePilotKnownBadControl] = []
    delta = KoidePilotReferenceDelta(
        declared=False,
        reason="candidate/reference comparison not available",
    )
    try:
        (
            input_files,
            selected_frame_ids,
            train_frame_ids,
            holdout_frame_ids,
            fitting_input_files,
            holdout_input_files,
        ) = _prepare_inputs(
            dataset,
            policy,
            input_manifest,
        )
        provenance.input_digests = {item.path: item.sha256 for item in input_files}
        provenance.fitting_input_digests = {
            item.path: item.sha256 for item in fitting_input_files
        }
        provenance.holdout_input_digests = {
            item.path: item.sha256 for item in holdout_input_files
        }
        provenance.fitting_frame_ids_sha256 = _frame_ids_digest(train_frame_ids)
        provenance.holdout_frame_ids_sha256 = _frame_ids_digest(holdout_frame_ids)
    except (OSError, ValueError, CalibrexError) as exc:
        blockers.append(f"KITTI input manifest is unavailable: {exc}")

    if config is None or not config.is_file():
        blockers.append("frozen Calibrex config is missing; provide --config")
    if readiness is None or not readiness.is_file():
        blockers.append("Koide readiness artifact is missing; provide --readiness")
    else:
        try:
            readiness_artifact = load_koide_readiness(readiness)
            readiness_status = cast(
                Literal["ready", "warn", "blocked", "missing"],
                readiness_artifact.status,
            )
            if readiness_artifact.status == "blocked":
                blockers.append("Koide readiness status is BLOCKED")
            elif readiness_artifact.status == "warn":
                warnings.append("Koide readiness status is WARN; adoption is not admissible")
            declared_digest = readiness_artifact.provenance.artifact_sha256
            recomputed_digest = readiness_artifact.with_artifact_digest().provenance.artifact_sha256
            if declared_digest is None or recomputed_digest != declared_digest:
                blockers.append(
                    "Koide readiness artifact self-digest does not match its canonical content"
                )
            if Path(readiness_artifact.dataset_path).resolve() != dataset:
                blockers.append("Koide readiness dataset path does not match the pilot dataset")
            if (
                readiness_artifact.provenance.dataset_sha256 is not None
                and readiness_artifact.provenance.dataset_sha256 != provenance.dataset_sha256
            ):
                blockers.append("Koide readiness dataset digest does not match the pilot dataset")
            if (
                config is not None
                and readiness_artifact.provenance.config_sha256 is not None
                and readiness_artifact.provenance.config_sha256 != provenance.config_sha256
            ):
                blockers.append("Koide readiness config digest does not match the pilot config")
        except (OSError, ValueError) as exc:
            blockers.append(f"Koide readiness artifact is invalid: {exc}")

    if candidate is None or not candidate.is_file():
        blockers.append(
            "official Koide candidate is unavailable; execute the recorded external command "
            "and provide its native calib.json or external-run artifact"
        )
    else:
        try:
            external_run = _load_candidate(
                candidate,
                camera_frame=camera_frame,
                lidar_frame=lidar_frame,
                fitting_input_files=fitting_input_files,
            )
            execution = _execution_manifest(
                official_command or tuple(external_run.execution.command),
                source_commit=tool_source_commit or external_run.tool.source_commit,
                container_digest=container_digest or external_run.execution.container_digest,
                candidate_execution_attempted=external_run.execution.attempted,
            )
            provenance.candidate_output_path = external_run.provenance.source_artifact
            provenance.candidate_output_sha256 = external_run.provenance.source_artifact_sha256
            if external_run.status != "success":
                blockers.append(f"external candidate status is {external_run.status}")
            candidate_transform = _candidate_transform(
                external_run,
                camera_frame=camera_frame,
                lidar_frame=lidar_frame,
            )
            if candidate_transform is None:
                blockers.append(
                    f"external candidate does not contain T_{camera_frame}_{lidar_frame}"
                )
            _verify_external_inputs(
                external_run,
                fitting_input_files=fitting_input_files,
                holdout_input_files=holdout_input_files,
                train_frame_ids=train_frame_ids,
                holdout_frame_ids=holdout_frame_ids,
            )
            if not external_run.train_data_isolation.declared:
                warnings.append("external training-data isolation is not declared")
            if external_run.train_data_isolation.declared and (
                not external_run.train_data_isolation.evidence
            ):
                warnings.append("external training-data isolation has no evidence string")
        except (OSError, ValueError) as exc:
            blockers.append(f"external Koide candidate is invalid: {exc}")

    # Reference construction and scoring are deliberately after all source
    # checks.  This keeps a malformed or stale candidate from being scored as
    # evidence merely because the local KITTI reference exists.
    if not blockers and candidate_transform is not None:
        try:
            reference_se3 = read_velodyne_to_camera_transform(dataset)
            if reference_se3 is None:
                raise ValueError("KITTI calibration has no camera<-LiDAR transform")
            reference_transform = _pilot_transform(
                reference_se3,
                parent=camera_frame,
                child=lidar_frame,
                source_path=_find_kitti_calibration(dataset),
            )
            candidates: dict[str, tuple[SE3, bool]] = {
                "kitti_dataset_reference": (reference_se3, True),
                "koide_external_candidate": (
                    candidate_transform.as_se3(),
                    bool(external_run and external_run.train_data_isolation.declared),
                ),
            }
            controls_se3 = _known_bad_transforms(candidate_transform.as_se3(), policy)
            candidates.update({name: (pose, False) for name, pose in controls_se3.items()})
            scores = evaluate_lidar_camera_candidates_on_kitti(
                dataset,
                candidates,
                max_frames=policy.max_frames,
                max_points=policy.max_points,
                holdout_ratio=policy.holdout_ratio,
                split_seed=policy.split_seed,
                frame_ids=selected_frame_ids,
                train_frame_ids=train_frame_ids,
                holdout_frame_ids=holdout_frame_ids,
            )
            candidate_score = scores.get("koide_external_candidate")
            reference_score = scores.get("kitti_dataset_reference")
            if candidate_score is None or reference_score is None:
                raise ValueError("candidate/reference KITTI scores are unavailable")
            selected_frame_ids = list(
                candidate_score.train_frame_ids + candidate_score.holdout_frame_ids
            )
            train_frame_ids = list(candidate_score.train_frame_ids)
            holdout_frame_ids = list(candidate_score.holdout_frame_ids)
            candidate_metrics = _metric_set(candidate_score)
            reference_metrics = _metric_set(reference_score)
            delta = _reference_delta(candidate_transform, reference_transform)
            controls = _controls_from_scores(
                scores,
                candidate_score,
                policy,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            blockers.append(f"KITTI frozen evaluation could not be scored: {exc}")

    status, adoption, reason = _decision(
        blockers=blockers,
        warnings=warnings,
        readiness_status=readiness_status,
        external_run=external_run,
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        delta=delta,
        controls=controls,
        thresholds=policy,
    )
    admissible = status == "PASS" and adoption == "ADOPT"
    if status == "BLOCKED" and not execution.reason:
        execution = execution.model_copy(update={"reason": "required input unavailable"})
    pilot_id = _pilot_id(provenance, policy)
    artifact = KoidePilotArtifact(
        pilot_id=pilot_id,
        status=status,
        adoption_decision=adoption,
        admissible=admissible,
        reason=reason,
        dataset_path=str(dataset),
        camera_frame=camera_frame,
        lidar_frame=lidar_frame,
        camera_stream=camera_stream,
        lidar_stream=lidar_stream,
        selected_frame_ids=selected_frame_ids,
        train_frame_ids=train_frame_ids,
        holdout_frame_ids=holdout_frame_ids,
        thresholds=policy,
        readiness_status=readiness_status,
        external_run_status=external_run.status if external_run is not None else "missing",
        candidate_transform_camera_lidar=candidate_transform,
        reference_transform_camera_lidar=reference_transform,
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        candidate_to_reference=delta,
        known_bad_controls=controls,
        known_bad_detected_count=sum(1 for item in controls if item.detected),
        known_bad_detectable_fraction=(
            sum(1 for item in controls if item.detected) / len(controls) if controls else None
        ),
        input_files=input_files,
        fitting_input_files=fitting_input_files,
        holdout_input_files=holdout_input_files,
        execution=execution,
        blockers=blockers,
        warnings=warnings,
        provenance=provenance,
        artifact_sha256="0" * 64,
    ).with_artifact_digest()
    artifact.save(output / "pilot.json")
    if (
        external_run is not None
        and candidate is not None
        and candidate.suffix.lower()
        not in {
            ".json",
            ".yaml",
            ".yml",
        }
    ):
        external_run.save(output / "candidate-external-run.yaml")
    write_mapping(
        output / "execution-manifest.yaml",
        execution.model_dump(mode="json", exclude_none=False),
    )
    return artifact


def export_koide_pilot_autoware(
    pilot_path: str | Path,
    output: str | Path,
    *,
    base_frame: str,
    sensor_frame: str | None = None,
    static_tf_output: str | Path | None = None,
    manifest_output: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export only an admissible pilot candidate to Autoware.

    The generic Autoware exporter remains available for ordinary Calibrex
    results.  This dedicated entry point is the only supported export path for
    the Koide pilot and rejects WARN, FAIL, INCONCLUSIVE, and BLOCKED records.
    """

    from calibrex.export.autoware import (
        AutowareExportConfig,
        AutowareExportError,
        build_autoware_export,
        write_autoware_export,
    )

    pilot = load_koide_pilot(pilot_path)
    if not pilot.admissible or pilot.adoption_decision != "ADOPT":
        raise AutowareExportError(
            "Koide pilot is not admissible for Autoware export: "
            f"status={pilot.status}, adoption_decision={pilot.adoption_decision}"
        )
    candidate = pilot.candidate_transform_camera_lidar
    if candidate is None:
        raise AutowareExportError("admissible Koide pilot has no candidate transform")
    transform = candidate.as_transform_result(source_run=pilot.pilot_id)
    config = AutowareExportConfig(
        base_frame=base_frame,
        sensor_frames=[sensor_frame] if sensor_frame else [],
        transform_names=[f"T_{base_frame}_{candidate.child}"],
    )
    # The pilot transform is camera<-LiDAR.  Autoware needs base<-LiDAR; the
    # caller must therefore provide the base-to-camera edge separately in a
    # full rig result.  For this dedicated export, base_frame is required to
    # equal the candidate parent, preventing a silent frame re-root.
    if candidate.parent != base_frame:
        raise AutowareExportError(
            "Koide pilot candidate parent is "
            f"{candidate.parent!r}; pass that exact frame as --base-frame"
        )
    artifact = build_autoware_export(
        {f"T_{base_frame}_{candidate.child}": transform},
        config=config,
        source_path=pilot_path,
        source_run=pilot.pilot_id,
        quality_grade="pass",
        input_paths={
            "pilot": pilot_path,
            **(
                {"candidate_output": candidate.source_path}
                if candidate.source_path is not None
                else {}
            ),
        },
        command=["calibrex", "camera-lidar", "export-koide-pilot", str(pilot_path)],
    )
    return write_autoware_export(
        artifact,
        output,
        static_tf_output=static_tf_output,
        manifest_output=manifest_output,
        overwrite=overwrite,
    )


def _load_candidate(
    path: Path,
    *,
    camera_frame: str,
    lidar_frame: str,
    fitting_input_files: Sequence[KoidePilotInputDigest],
) -> ExternalCalibrationRunArtifact:
    payload = read_mapping(path)
    schema_version = payload.get("schema_version")
    if schema_version == "slac.external_calibration_run/v0.1":
        artifact = load_external_run(path)
        if artifact.provenance.source_artifact is None:
            raise ValueError("external-run candidate has no source artifact path")
        return artifact
    artifact = import_koide_result(
        path,
        lidar_frame=lidar_frame,
        camera_frame=camera_frame,
        input_artifacts=tuple(item.path for item in fitting_input_files),
        source_commit=None,
        tool_version=None,
        training_isolation_declared=False,
    )
    return artifact


def _verify_external_inputs(
    artifact: ExternalCalibrationRunArtifact,
    *,
    fitting_input_files: Sequence[KoidePilotInputDigest],
    holdout_input_files: Sequence[KoidePilotInputDigest],
    train_frame_ids: Sequence[str],
    holdout_frame_ids: Sequence[str],
) -> None:
    """Verify candidate inventory and prove no holdout file was fitted."""

    if not any(item.role == "input" for item in artifact.artifacts):
        raise ValueError("external candidate has no digest-bound fitting input artifact")
    if not any(item.role == "output" for item in artifact.artifacts):
        raise ValueError("external candidate has no digest-bound native output artifact")
    for item in artifact.artifacts:
        path = Path(item.path)
        digest = _digest_path(path)
        if digest is None:
            raise ValueError(f"external artifact is missing: {path}")
        if digest != item.sha256:
            raise ValueError(f"external artifact digest mismatch: {path}")
    output_digest = artifact.digests.output_sha256 or artifact.provenance.source_artifact_sha256
    output_items = [item for item in artifact.artifacts if item.role == "output"]
    if output_items and output_digest != output_items[-1].sha256:
        raise ValueError("external output digest does not match its artifact inventory")
    if (
        artifact.provenance.source_artifact is None
        or artifact.provenance.source_artifact_sha256 is None
    ):
        raise ValueError("external candidate has no digest-bound native output provenance")
    source_path = Path(artifact.provenance.source_artifact)
    source_digest = _digest_path(source_path)
    if source_digest != artifact.provenance.source_artifact_sha256:
        raise ValueError("external native output provenance digest mismatch")
    if output_items and not any(
        Path(item.path).resolve() == source_path.resolve() for item in output_items
    ):
        raise ValueError("external native output is absent from its artifact inventory")

    input_items = [item for item in artifact.artifacts if item.role == "input"]
    input_paths = [Path(item.path).resolve() for item in input_items]
    holdout_paths = {Path(item.path).resolve() for item in holdout_input_files}
    fitting_frame_paths = {
        Path(item.path).resolve()
        for item in fitting_input_files
        if item.frame_id is not None
    }
    for path in input_paths:
        if path.is_dir():
            if any(_is_relative_to(holdout, path) for holdout in holdout_paths):
                raise ValueError(
                    "external fitting input directory contains independent holdout files"
                )
        elif path in holdout_paths:
            raise ValueError(f"external fitting inputs overlap holdout file: {path}")
    missing_fit = [
        path for path in sorted(fitting_frame_paths) if not _path_is_bound(path, input_paths)
    ]
    if missing_fit:
        raise ValueError(
            "external fitting input inventory does not bind every train frame: "
            + ", ".join(str(item) for item in missing_fit[:5])
        )
    if artifact.digests.input_sha256 is None:
        raise ValueError("external candidate has no combined fitting-input digest")
    observed_input_digest = _combined_external_digest(input_items)
    if artifact.digests.input_sha256 != observed_input_digest:
        raise ValueError("external fitting-input digest does not match its inventory")
    if artifact.train_data_isolation.declared:
        expected_train_ids = _frame_ids_digest(train_frame_ids)
        expected_holdout_ids = _frame_ids_digest(holdout_frame_ids)
        if artifact.train_data_isolation.training_data_ids_sha256 != expected_train_ids:
            raise ValueError("external training-data IDs do not match the pilot train split")
        if artifact.train_data_isolation.holdout_data_ids_sha256 != expected_holdout_ids:
            raise ValueError("external holdout-data IDs do not match the pilot holdout split")
        if not artifact.train_data_isolation.evidence:
            raise ValueError("external training-data isolation has no evidence string")


def _candidate_transform(
    artifact: ExternalCalibrationRunArtifact,
    *,
    camera_frame: str,
    lidar_frame: str,
) -> KoidePilotTransform | None:
    for item in artifact.parsed_outputs.transforms.values():
        if item.parent == camera_frame and item.child == lidar_frame:
            source_path = artifact.provenance.source_artifact
            return KoidePilotTransform(
                parent=item.parent,
                child=item.child,
                translation_m=list(item.translation_m),
                rotation_quat_xyzw=list(item.rotation_quat_xyzw),
                source_path=source_path,
                source_sha256=artifact.provenance.source_artifact_sha256,
            )
    return None


def _prepare_inputs(
    dataset: Path,
    thresholds: KoidePilotThresholds,
    input_manifest: Path | None,
) -> tuple[
    list[KoidePilotInputDigest],
    list[str],
    list[str],
    list[str],
    list[KoidePilotInputDigest],
    list[KoidePilotInputDigest],
]:
    if not dataset.is_dir():
        raise ValueError(f"KITTI sequence directory does not exist: {dataset}")
    if input_manifest is not None:
        manifest = load_kitti_benchmark_input(input_manifest)
        verify_kitti_benchmark_input(manifest, sequence_path=dataset)
        files = [
            KoidePilotInputDigest(
                role=item.role,
                path=str((dataset.parent / item.path).resolve()),
                frame_id=(
                    _canonical_frame_id(item.frame_id)
                    if item.frame_id is not None
                    else None
                ),
                sha256=item.sha256,
                size_bytes=item.size_bytes,
            )
            for item in manifest.files
        ]
    else:
        pairs = find_camera_lidar_pairs(dataset, max_pairs=thresholds.max_frames)
        if len(pairs) < 2:
            raise ValueError("pilot requires at least two camera-LiDAR frame pairs")
        files = []
        for pair in pairs:
            files.extend(
                [
                    _input_digest(Path(pair.camera_path), "camera"),
                    _input_digest(Path(pair.lidar_path), "lidar"),
                ]
            )
        for calibration in _calibration_paths(dataset):
            files.append(_input_digest(calibration, "calibration"))
    if len(files) < 2:
        raise ValueError("pilot input manifest contains fewer than two source files")
    frame_ids = _frame_ids_from_files(files)
    if len(frame_ids) < 2:
        pairs = find_camera_lidar_pairs(dataset, max_pairs=thresholds.max_frames)
        frame_ids = [f"camera:{pair.camera_index}:lidar:{pair.lidar_index}" for pair in pairs]
    train, holdout = _split_frame_ids(frame_ids, thresholds.holdout_ratio, thresholds.split_seed)
    fitting = [
        item
        for item in files
        if item.frame_id is None or item.frame_id in set(train)
    ]
    holdout_files = [item for item in files if item.frame_id in set(holdout)]
    if not fitting or not holdout_files:
        raise ValueError("pilot could not construct disjoint fitting and holdout file inventories")
    if {item.path for item in fitting} & {item.path for item in holdout_files}:
        raise ValueError("pilot fitting and holdout file inventories overlap")
    return files, frame_ids, train, holdout, fitting, holdout_files


def _metric_set(score: LidarCameraCandidateScore) -> KoidePilotMetricSet:
    return KoidePilotMetricSet(
        candidate_id=score.candidate_id,
        train_frame_ids=list(score.train_frame_ids),
        holdout_frame_ids=list(score.holdout_frame_ids),
        train_frame_count=len(score.train_frame_ids),
        holdout_frame_count=len(score.holdout_frame_ids),
        scored_frame_count=score.scored_frame_count,
        train_projection_ratio=score.train_projection_ratio,
        holdout_projection_ratio=score.holdout_projection_ratio,
        train_edge_alignment=score.train_edge_alignment,
        holdout_edge_alignment=score.holdout_edge_alignment,
        train_depth_edge_alignment=score.train_depth_edge_alignment,
        holdout_depth_edge_alignment=score.holdout_depth_edge_alignment,
        training_isolation_declared=score.training_isolation_declared,
    )


def _reference_delta(
    candidate: KoidePilotTransform, reference: KoidePilotTransform
) -> KoidePilotReferenceDelta:
    relative = reference.as_se3().inverse().compose(candidate.as_se3())
    return KoidePilotReferenceDelta(
        declared=True,
        translation_delta_m=math.sqrt(sum(value * value for value in relative.translation_m)),
        rotation_delta_deg=_rotation_angle_deg(relative),
        reason="dataset calibration is declared as the candidate/reference baseline",
    )


def _controls_from_scores(
    scores: Mapping[str, LidarCameraCandidateScore],
    candidate: LidarCameraCandidateScore,
    thresholds: KoidePilotThresholds,
) -> list[KoidePilotKnownBadControl]:
    controls: list[KoidePilotKnownBadControl] = []
    for name, dof, sign, amount, unit in _control_specs(thresholds):
        score = scores.get(name)
        if score is None:
            controls.append(
                KoidePilotKnownBadControl(
                    control_id=name,
                    dof=dof,
                    sign=sign,
                    amount=amount,
                    unit=unit,
                    status="blocked",
                    reason="control was not scored on the frozen frame split",
                )
            )
            continue
        projection = _delta(candidate.holdout_projection_ratio, score.holdout_projection_ratio)
        edge = _delta(candidate.holdout_edge_alignment, score.holdout_edge_alignment)
        depth = _delta(candidate.holdout_depth_edge_alignment, score.holdout_depth_edge_alignment)
        available = [value for value in (projection, edge, depth) if value is not None]
        detected = bool(available) and max(available) >= thresholds.known_bad_min_metric_delta
        controls.append(
            KoidePilotKnownBadControl(
                control_id=name,
                dof=dof,
                sign=sign,
                amount=amount,
                unit=unit,
                candidate_holdout_projection_ratio=candidate.holdout_projection_ratio,
                control_holdout_projection_ratio=score.holdout_projection_ratio,
                candidate_holdout_edge_alignment=candidate.holdout_edge_alignment,
                control_holdout_edge_alignment=score.holdout_edge_alignment,
                candidate_holdout_depth_edge_alignment=candidate.holdout_depth_edge_alignment,
                control_holdout_depth_edge_alignment=score.holdout_depth_edge_alignment,
                metric_delta_projection_ratio=projection,
                metric_delta_edge_alignment=edge,
                metric_delta_depth_edge_alignment=depth,
                detected=detected,
                status="pass" if detected else "fail",
                reason=(
                    "candidate holdout score degraded under the declared signed control"
                    if detected
                    else "control did not measurably degrade any holdout score"
                ),
            )
        )
    return controls


def _decision(
    *,
    blockers: list[str],
    warnings: list[str],
    readiness_status: str,
    external_run: ExternalCalibrationRunArtifact | None,
    candidate_metrics: KoidePilotMetricSet | None,
    reference_metrics: KoidePilotMetricSet | None,
    delta: KoidePilotReferenceDelta,
    controls: list[KoidePilotKnownBadControl],
    thresholds: KoidePilotThresholds,
) -> tuple[PilotStatus, PilotAdoptionDecision, str]:
    if blockers:
        return "BLOCKED", "BLOCKED", "; ".join(blockers)
    if readiness_status != "ready" and thresholds.require_readiness_ready:
        return (
            "WARN",
            "DO_NOT_ADOPT",
            "readiness is not READY; external evidence is retained but not admissible",
        )
    if external_run is None or candidate_metrics is None or reference_metrics is None:
        return "INCONCLUSIVE", "INCONCLUSIVE", "candidate/reference holdout metrics are unavailable"
    missing = []
    if len(candidate_metrics.holdout_frame_ids) < thresholds.min_holdout_frames:
        missing.append("minimum independent holdout frames")
    for name, value in (
        ("holdout projection ratio", candidate_metrics.holdout_projection_ratio),
        ("holdout edge alignment", candidate_metrics.holdout_edge_alignment),
        ("holdout depth-edge alignment", candidate_metrics.holdout_depth_edge_alignment),
    ):
        if value is None:
            missing.append(name)
    if missing:
        return "INCONCLUSIVE", "INCONCLUSIVE", "missing holdout evidence: " + ", ".join(missing)
    metric_failures: list[str] = []
    projection_ratio = candidate_metrics.holdout_projection_ratio
    edge_alignment = candidate_metrics.holdout_edge_alignment
    depth_edge_alignment = candidate_metrics.holdout_depth_edge_alignment
    if projection_ratio is not None and projection_ratio < thresholds.min_holdout_projection_ratio:
        metric_failures.append("holdout projection ratio below threshold")
    if edge_alignment is not None and edge_alignment < thresholds.min_holdout_edge_alignment:
        metric_failures.append("holdout edge alignment below threshold")
    if (
        depth_edge_alignment is not None
        and depth_edge_alignment < thresholds.min_holdout_depth_edge_alignment
    ):
        metric_failures.append("holdout depth-edge alignment below threshold")
    if (
        delta.declared
        and delta.translation_delta_m is not None
        and delta.translation_delta_m > thresholds.max_candidate_reference_translation_m
    ):
        metric_failures.append("candidate/reference translation delta exceeds threshold")
    if (
        delta.declared
        and delta.rotation_delta_deg is not None
        and delta.rotation_delta_deg > thresholds.max_candidate_reference_rotation_deg
    ):
        metric_failures.append("candidate/reference rotation delta exceeds threshold")
    if thresholds.require_candidate_reference_comparison and not delta.declared:
        return (
            "INCONCLUSIVE",
            "INCONCLUSIVE",
            "candidate/reference SE(3) comparison was not declared",
        )
    if len(controls) != 12:
        return (
            "INCONCLUSIVE",
            "INCONCLUSIVE",
            f"mandatory 6-DoF controls incomplete: {len(controls)}/12",
        )
    incomplete_controls = [
        item.control_id for item in controls if item.status in {"blocked", "inconclusive"}
    ]
    if incomplete_controls:
        return (
            "INCONCLUSIVE",
            "INCONCLUSIVE",
            "known-bad controls lack measurable holdout evidence: "
            + ", ".join(incomplete_controls),
        )
    detected_fraction = sum(1 for item in controls if item.detected) / 12.0
    if detected_fraction < thresholds.min_known_bad_detectable_fraction:
        metric_failures.append(
            "mandatory 6-DoF known-bad detection fraction below threshold "
            f"({detected_fraction:.3f} < {thresholds.min_known_bad_detectable_fraction:.3f})"
        )
    if metric_failures:
        return "FAIL", "DO_NOT_ADOPT", "; ".join(metric_failures)
    provenance_missing: list[str] = []
    if not external_run.provenance.source_artifact:
        provenance_missing.append("source artifact path")
    if not external_run.provenance.source_artifact_sha256:
        provenance_missing.append("source artifact digest")
    if not external_run.tool.source_commit:
        provenance_missing.append("tool source commit")
    if not external_run.tool.version:
        provenance_missing.append("tool version")
    if not external_run.execution.attempted:
        provenance_missing.append("external execution evidence")
    if provenance_missing:
        return (
            "WARN",
            "DO_NOT_ADOPT",
            "candidate provenance is incomplete; adoption is forbidden: "
            + ", ".join(provenance_missing),
        )
    if thresholds.require_training_isolation and not external_run.train_data_isolation.declared:
        return (
            "WARN",
            "DO_NOT_ADOPT",
            "holdout is scored independently but external training-data isolation is undeclared",
        )
    if warnings:
        return "WARN", "DO_NOT_ADOPT", "; ".join(warnings)
    return "PASS", "ADOPT", "frozen candidate passed holdout, SE(3), and mandatory known-bad gates"


def _known_bad_transforms(candidate: SE3, thresholds: KoidePilotThresholds) -> dict[str, SE3]:
    controls: dict[str, SE3] = {}
    for dof, amount, unit in (
        ("roll", thresholds.known_bad_rotation_deg, "deg"),
        ("pitch", thresholds.known_bad_rotation_deg, "deg"),
        ("yaw", thresholds.known_bad_rotation_deg, "deg"),
        ("x", thresholds.known_bad_translation_m, "m"),
        ("y", thresholds.known_bad_translation_m, "m"),
        ("z", thresholds.known_bad_translation_m, "m"),
    ):
        for sign in ("+", "-"):
            signed = amount if sign == "+" else -amount
            if unit == "deg":
                axis = {"roll": (1.0, 0.0, 0.0), "pitch": (0.0, 1.0, 0.0), "yaw": (0.0, 0.0, 1.0)}[
                    dof
                ]
                half = math.radians(signed) / 2.0
                delta = SE3(
                    (0.0, 0.0, 0.0),
                    (
                        axis[0] * math.sin(half),
                        axis[1] * math.sin(half),
                        axis[2] * math.sin(half),
                        math.cos(half),
                    ),
                )
            else:
                translation = {
                    "x": (signed, 0.0, 0.0),
                    "y": (0.0, signed, 0.0),
                    "z": (0.0, 0.0, signed),
                }[dof]
                delta = SE3(translation, (0.0, 0.0, 0.0, 1.0))
            controls[f"known_bad_{dof}{sign}"] = delta.compose(candidate)
    return controls


def _control_specs(
    thresholds: KoidePilotThresholds,
) -> list[
    tuple[
        str,
        Literal["roll", "pitch", "yaw", "x", "y", "z"],
        Literal["+", "-"],
        float,
        Literal["deg", "m"],
    ]
]:
    output: list[
        tuple[
            str,
            Literal["roll", "pitch", "yaw", "x", "y", "z"],
            Literal["+", "-"],
            float,
            Literal["deg", "m"],
        ]
    ] = []
    for dof in ("roll", "pitch", "yaw"):
        for sign in ("+", "-"):
            output.append(
                (
                    f"known_bad_{dof}{sign}",
                    cast(Any, dof),
                    cast(Any, sign),
                    thresholds.known_bad_rotation_deg,
                    "deg",
                )
            )
    for dof in ("x", "y", "z"):
        for sign in ("+", "-"):
            output.append(
                (
                    f"known_bad_{dof}{sign}",
                    cast(Any, dof),
                    cast(Any, sign),
                    thresholds.known_bad_translation_m,
                    "m",
                )
            )
    return output


def _split_frame_ids(frame_ids: list[str], ratio: float, seed: int) -> tuple[list[str], list[str]]:
    # Keep the split policy independent of metric values and stable across
    # Python versions: frame IDs are timestamp ordered before split_indices.
    from calibrex.evaluation.holdout import split_indices

    ordered = sorted(frame_ids, key=_frame_sort_key)
    train_indices, holdout_indices = split_indices(len(ordered), ratio, seed)
    if len(ordered) > 1 and not holdout_indices:
        train_indices, holdout_indices = split_indices(len(ordered), 1.0 / len(ordered), seed)
    return [ordered[index] for index in train_indices], [
        ordered[index] for index in holdout_indices
    ]


def _frame_ids_from_files(files: Sequence[KoidePilotInputDigest]) -> list[str]:
    ids: list[str] = []
    for item in files:
        if item.frame_id is not None:
            ids.append(item.frame_id)
            continue
        path = Path(item.path)
        if (
            path.parent.name == "data"
            and path.stem.isdigit()
            and path.suffix.lower() in {".png", ".bin"}
        ):
            ids.append(_canonical_frame_id(path.stem))
    return sorted(set(ids), key=_frame_sort_key)


def _canonical_frame_id(value: str | None) -> str:
    """Normalize a KITTI frame token to the evaluator's camera/lidar ID."""

    if value is None:
        raise ValueError("frame ID cannot be null")
    text = str(value)
    if text.startswith("camera:") and ":lidar:" in text:
        return text
    if not text.isdigit():
        raise ValueError(f"invalid KITTI frame ID: {value!r}")
    index = int(text)
    return f"camera:{index}:lidar:{index}"


def _frame_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value.rsplit(":", 1)[-1]), value
    except ValueError:
        return 0, value


def _pilot_transform(
    transform: SE3, *, parent: str, child: str, source_path: Path | None
) -> KoidePilotTransform:
    return KoidePilotTransform(
        parent=parent,
        child=child,
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        source_path=str(source_path) if source_path is not None else None,
        source_sha256=_digest_file(source_path),
    )


def _input_digest(path: Path, role: str) -> KoidePilotInputDigest:
    digest = _digest_file(path)
    if digest is None:
        raise ValueError(f"required KITTI input is missing: {path}")
    return KoidePilotInputDigest(
        role=role,
        path=str(path.resolve()),
        frame_id=_canonical_frame_id(path.stem) if path.parent.name == "data" else None,
        sha256=digest,
        size_bytes=path.stat().st_size,
    )


def _calibration_paths(dataset: Path) -> list[Path]:
    paths: list[Path] = []
    for parent in (dataset, dataset.parent, dataset.parent.parent):
        for name in ("calib_velo_to_cam.txt", "calib_cam_to_cam.txt"):
            path = parent / name
            if path.is_file() and path not in paths:
                paths.append(path)
    if len(paths) < 2:
        raise ValueError("KITTI calibration files are missing")
    return paths


def _find_kitti_calibration(dataset: Path) -> Path | None:
    paths = _calibration_paths(dataset)
    return paths[0] if paths else None


def _execution_manifest(
    command: Sequence[str],
    *,
    source_commit: str | None,
    container_digest: str | None,
    candidate_execution_attempted: bool = False,
) -> KoidePilotExecutionManifest:
    docker = shutil.which("docker") is not None
    podman = shutil.which("podman") is not None
    engine: Literal["docker", "podman", "none"] = (
        "docker" if docker else "podman" if podman else "none"
    )
    status: PilotExecutionStatus
    if candidate_execution_attempted:
        status = "attempted"
        reason = "external-run artifact records an attempted Koide execution"
    elif command and engine == "none":
        status = "unavailable"
        reason = (
            "official Koide command was recorded but not attempted: Docker and Podman "
            "are unavailable"
        )
    elif command:
        status = "not_attempted"
        reason = (
            "official Koide command was recorded for provenance; this pilot does not "
            "invoke external processes"
        )
    elif engine == "none":
        status = "unavailable"
        reason = "Docker and Podman are unavailable; official Koide execution was not attempted"
    else:
        status = "not_attempted"
        reason = "official Koide command was not supplied"
    return KoidePilotExecutionManifest(
        command=list(command),
        status=status,
        attempted=candidate_execution_attempted,
        reason=reason,
        docker_available=docker,
        podman_available=podman,
        engine=engine,
        source_commit=source_commit,
        container_digest=container_digest,
    )


def _pilot_id(provenance: KoidePilotProvenance, thresholds: KoidePilotThresholds) -> str:
    digest = _canonical_sha256(
        {
            "dataset": provenance.dataset_sha256,
            "config": provenance.config_sha256,
            "readiness": provenance.readiness_sha256,
            "candidate": provenance.candidate_sha256,
            "thresholds": thresholds.model_dump(mode="json"),
        }
    )
    return f"koide-pilot-{digest[:16]}"


def _digest_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return sha256_path(path)


def _digest_path_or_tree(path: Path) -> str | None:
    return sha256_path(path)


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _digest_path(path: Path | None) -> str | None:
    return _digest_file(path)


def _frame_ids_digest(frame_ids: Sequence[str]) -> str:
    """Digest a canonical ordered-independent frame-ID inventory."""

    payload = "\n".join(sorted(set(frame_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _combined_external_digest(items: Sequence[Any]) -> str:
    selected = sorted(
        (str(item.path), str(item.sha256))
        for item in items
        if getattr(item, "role", None) == "input"
    )
    payload = "\n".join(f"{path}\0{digest}" for path, digest in selected).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _path_is_bound(expected: Path, inputs: Sequence[Path]) -> bool:
    return any(
        path == expected or (path.is_dir() and _is_relative_to(expected, path))
        for path in inputs
    )


def _delta(candidate: float | None, control: float | None) -> float | None:
    return candidate - control if candidate is not None and control is not None else None


def _rotation_angle_deg(transform: SE3) -> float:
    scalar = min(1.0, max(-1.0, abs(transform.rotation_quat_xyzw[3])))
    return math.degrees(2.0 * math.acos(scalar))


__all__ = [
    "KOIDE_PILOT_PROTOCOL_ID",
    "KOIDE_PILOT_SCHEMA_VERSION",
    "KoidePilotArtifact",
    "KoidePilotExecutionManifest",
    "KoidePilotInputDigest",
    "KoidePilotKnownBadControl",
    "KoidePilotMetricSet",
    "KoidePilotReferenceDelta",
    "KoidePilotThresholds",
    "KoidePilotTransform",
    "export_koide_pilot_autoware",
    "koide_pilot_json_schema",
    "load_koide_pilot",
    "run_koide_pilot",
]
