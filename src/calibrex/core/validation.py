"""Validation helpers for schema-versioned Calibrex artifacts.

Artifact kinds, their models, accepted schema versions, retired versions, and
migrations are declared once in :mod:`calibrex.core.schema_registry`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from calibrex.core.capture_manifest import (
    CaptureManifest,
    CaptureManifestVerification,
    verify_capture_manifest_inputs,
)
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping
from calibrex.core.result import CalibrationResult, StrictModel, result_provenance_issues
from calibrex.core.schema_registry import (
    SchemaRegistryError,
    kind_for_schema_version,
    schema_entry,
    upgrade_payload,
    validatable_kinds,
)

ValidationKind = Literal[
    "auto",
    "config",
    "result",
    "comparison",
    "report-comparison",
    "dynamic-window-consistency",
    "assessment",
    "benchmark",
    "benchmark-definition",
    "policy",
    "protocol",
    "transforms",
    "dataset-manifest",
    "remote-archive-selection",
    "kitti360-lidar-window-integration",
    "kitti-raw-lidar-window-integration",
    "doctor",
    "environment-readiness",
    "calibration-ci",
    "calibration-lifecycle",
    "lifecycle-registry",
    "lifecycle-event",
    "lifecycle-evaluation",
    "lifecycle-registry-state",
    "lifecycle-head",
    "lifecycle-verification",
    "lifecycle-status",
    "external-run",
    "kitti-falsification",
    "kitti-benchmark-input",
    "koide-pilot",
    "koide-execution-lock",
    "koide-real-pilot-request",
    "koide-real-pilot-verification",
    "koide-real-pilot-finalization",
    "depth-provider",
    "continuous-time-camera-lidar-problem",
    "continuous-time-camera-lidar-result",
    "continuous-time-trajectory",
    "continuous-time-trajectory-measurements",
    "continuous-time-trajectory-fit",
    "probabilistic-correspondence",
    "probabilistic-pnp-result",
    "probabilistic-refinement-result",
    "empirical-se3-uncertainty",
    "camera-lidar-problem",
    "camera-lidar-correspondence-export",
    "camera-lidar-confidence-calibration",
    "camera-lidar-provider-support-comparison",
    "camera-lidar-pose-initializer-protocol",
    "camera-lidar-pose-initializer-failure-analysis",
    "camera-lidar-initializer-calibration",
    "camera-lidar-correspondence-quality",
    "camera-lidar-failure-analysis",
    "camera-lidar-sota-audit-protocol",
    "camera-lidar-sota-audit-result",
    "gnss-lidar-lever-arm",
    "imu-lidar-rotation",
    "imu-lidar-translation",
    "ins-lidar-hand-eye",
    "lidar-lidar-extrinsic",
    "vehicle-frame-rotation",
    "gnss-imu-lever-arm",
    "lidar-wheel-odometry",
    "camera-focal-scale",
    "camera-lidar-edge",
    "calibration-check",
    "bag-estimate",
    "calibration-drift",
    "check-frames",
    "sota-audit-protocol",
    "sota-audit-result",
    "sota-leaderboard",
    "camera-lidar-benchmark-protocol",
    "calibration-candidate-trace",
    "bullseye-plot",
    "report-summary",
    "report-metrics",
    "report-observability",
    "report-degeneracy",
    "report-evidence",
    "evidence-bundle",
    "evidence-bundle-verification",
    "online-timeline",
    "trajectory",
    "trajectory-window-drift",
    "capture-readiness",
    "capture-manifest",
    "capture-manifest-verification",
    "mcap-integrity",
    "koide-readiness",
    "continuous-time-lidar-pair",
    "continuous-time-lidar-point-to-plane",
    "continuous-time-imu-preintegration",
    "continuous-time-imu-lever-arm",
    "continuous-time-imu-clock-offset",
    "continuous-time-imu-accel-bias",
    "continuous-time-imu-intrinsics",
    "continuous-time-sliding-window",
    "continuous-time-lidar-train-diagnostics",
    "continuous-time-lidar-ablation",
    "solid-state-cross-dataset-benchmark-config",
    "solid-state-cross-dataset-benchmark",
    "solid-state-failure-analysis",
    "solid-state-synthetic-benchmark",
    "solid-state-metrology-evaluation",
    "solid-state-context",
    "livox-time-ablation",
    "autoware-export",
    "autoware-promotion",
    "autoware-smoke",
    "raw-replay-definition",
    "raw-replay-plan",
    "raw-replay-stage",
    "raw-replay-result",
    "raw-replay-comparison",
    "field-replacement-pilot",
    "multi-lidar-service-plan",
    "multi-lidar-service-evaluation",
    "camera-imu-service-plan",
    "camera-imu-service-evaluation",
    "radar-service-plan",
    "radar-service-evaluation",
    "camera-imu-motion-recording",
    "external-camera-imu-evidence",
]


class ValidationReport(StrictModel):
    """Machine-readable validation result."""

    path: str
    kind: str
    schema_version: str
    valid: bool = True
    migrated_from: list[str] = Field(default_factory=list, exclude=True)
    input_verification: CaptureManifestVerification | None = Field(default=None, exclude=True)
    production_valid: bool = Field(default=True, exclude=True)
    admissibility: Literal["admissible", "blocked"] = Field(
        default="admissible", exclude=True
    )
    provenance_issues: list[str] = Field(default_factory=list, exclude=True)


def validation_kinds() -> tuple[str, ...]:
    """Return supported explicit validation kinds."""

    return validatable_kinds()


def validation_kind_choices() -> tuple[str, ...]:
    """Return all CLI validation kind choices, including auto-detection."""

    return ("auto", *validation_kinds())


def validate_file(
    path: str | Path,
    kind: ValidationKind = "auto",
    *,
    verify_inputs: bool = False,
) -> ValidationReport:
    """Validate a schema-versioned Calibrex artifact.

    Payloads whose ``schema_version`` is older than the kind's model accepts
    are upgraded through registered lossless migrations first; the upgraded
    source versions are reported in ``migrated_from``.
    """

    artifact_path = Path(path)
    payload = read_mapping(artifact_path)
    detected_kind = _detect_kind(payload) if kind == "auto" else kind
    model = _model_for_kind(detected_kind)
    try:
        payload, migrated_from = upgrade_payload(detected_kind, payload)
    except SchemaRegistryError as exc:
        raise CalibrexError(str(exc)) from exc
    validated = model.model_validate(payload)
    digest_method = schema_entry(detected_kind).digest_method
    if digest_method is not None:
        getattr(validated, digest_method)()
    if isinstance(validated, CaptureManifest) and verify_inputs:
        verification = verify_capture_manifest_inputs(artifact_path)
        return ValidationReport(
            path=artifact_path.as_posix(),
            kind=detected_kind,
            schema_version=_schema_version(validated),
            valid=verification.valid,
            input_verification=verification,
        )
    if verify_inputs:
        raise CalibrexError("--verify-inputs is only supported for capture-manifest artifacts")
    schema_version = _schema_version(validated)
    if isinstance(validated, CalibrationResult):
        provenance_issues = result_provenance_issues(validated.run.provenance)
        return ValidationReport(
            path=artifact_path.as_posix(),
            kind=detected_kind,
            schema_version=schema_version,
            valid=not provenance_issues,
            migrated_from=list(migrated_from),
            production_valid=not provenance_issues,
            admissibility="admissible" if not provenance_issues else "blocked",
            provenance_issues=provenance_issues,
        )
    return ValidationReport(
        path=artifact_path.as_posix(),
        kind=detected_kind,
        schema_version=schema_version,
        migrated_from=list(migrated_from),
    )


def _detect_kind(payload: dict[str, object]) -> str:
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str):
        msg = "cannot auto-detect artifact kind without a string schema_version"
        raise CalibrexError(msg)
    try:
        return kind_for_schema_version(schema_version)
    except SchemaRegistryError as exc:
        raise CalibrexError(str(exc)) from exc


def _model_for_kind(kind: str) -> type[BaseModel]:
    try:
        model = schema_entry(kind).model
    except SchemaRegistryError:
        model = None
    if model is None:
        supported = ", ".join(validation_kind_choices())
        msg = f"unsupported validation kind {kind!r}; expected one of {supported}"
        raise CalibrexError(msg)
    return model


def _schema_version(model: BaseModel) -> str:
    value = getattr(model, "schema_version", None)
    if not isinstance(value, str):
        msg = "validated artifact did not expose a string schema_version"
        raise CalibrexError(msg)
    return value
