"""Validation helpers for schema-versioned Calibrex artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel

from calibrex.calibration_ci import (
    CALIBRATION_CI_SCHEMA_VERSION,
    CalibrationCIArtifact,
)
from calibrex.core.assessment import ASSESSMENT_SCHEMA_VERSION, AssessmentArtifact
from calibrex.core.benchmark import (
    BENCHMARK_DEFINITION_SCHEMA_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkArtifact,
    BenchmarkDefinition,
)
from calibrex.core.camera_lidar_artifacts import (
    BULLSEYE_PLOT_SCHEMA_VERSION,
    CALIBRATION_CANDIDATE_TRACE_SCHEMA_VERSION,
    CAMERA_LIDAR_BENCHMARK_PROTOCOL_SCHEMA_VERSION,
    CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION,
    BullseyePlotArtifact,
    CalibrationCandidateTrace,
    CameraLidarBenchmarkProtocol,
    CameraLidarCalibrationProblem,
)
from calibrex.core.camera_lidar_sota_audit import (
    CAMERA_LIDAR_SOTA_AUDIT_PROTOCOL_SCHEMA_VERSION,
    CAMERA_LIDAR_SOTA_AUDIT_RESULT_SCHEMA_VERSION,
    CameraLidarSotaAuditProtocol,
    CameraLidarSotaAuditResult,
)
from calibrex.core.capture_readiness import (
    CAPTURE_READINESS_SCHEMA_VERSION,
    CaptureReadinessArtifact,
)
from calibrex.core.config import CONFIG_SCHEMA_VERSION, CalibrationConfig
from calibrex.core.continuous_time_camera_lidar_artifacts import (
    CONTINUOUS_TIME_CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION,
    CONTINUOUS_TIME_CAMERA_LIDAR_RESULT_SCHEMA_VERSION,
    ContinuousTimeCameraLidarProblemArtifact,
    ContinuousTimeCameraLidarResultArtifact,
)
from calibrex.core.continuous_time_lidar_ablation import (
    CONTINUOUS_TIME_LIDAR_ABLATION_SCHEMA_VERSION,
    ContinuousTimeLidarAblationManifest,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    CONTINUOUS_TIME_LIDAR_PAIR_SCHEMA_VERSION,
    ContinuousTimeLidarPairArtifact,
)
from calibrex.core.dynamic_window import (
    DYNAMIC_WINDOW_CONSISTENCY_SCHEMA_VERSION,
    DynamicWindowConsistencyArtifact,
)
from calibrex.core.evidence_bundle import (
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EVIDENCE_BUNDLE_VERIFICATION_SCHEMA_VERSION,
    EvidenceBundleManifest,
    EvidenceBundleVerification,
)
from calibrex.core.evidence_contract import (
    POLICY_SCHEMA_VERSION,
    PROTOCOL_SCHEMA_VERSION,
    PolicyArtifact,
    ProtocolArtifact,
)
from calibrex.core.exceptions import CalibrexError
from calibrex.core.external_run import (
    EXTERNAL_RUN_SCHEMA_VERSION,
    ExternalCalibrationRunArtifact,
)
from calibrex.core.io import read_mapping
from calibrex.core.livox_time_ablation import (
    LIVOX_TIME_ABLATION_SCHEMA_VERSION,
    LivoxTimeAblationManifest,
)
from calibrex.core.online_timeline import (
    ONLINE_TIMELINE_SCHEMA_VERSION,
    OnlineCalibrationTimelineArtifact,
)
from calibrex.core.probabilistic_correspondence import (
    PROBABILISTIC_CORRESPONDENCE_SCHEMA_VERSION,
    PROBABILISTIC_PNP_RESULT_SCHEMA_VERSION,
    PROBABILISTIC_REFINEMENT_RESULT_SCHEMA_VERSION,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticPnpResultArtifact,
    ProbabilisticRefinementResultArtifact,
)
from calibrex.core.report_artifacts import (
    REPORT_DEGENERACY_SCHEMA_VERSION,
    REPORT_EVIDENCE_SCHEMA_VERSION,
    REPORT_METRICS_SCHEMA_VERSION,
    REPORT_OBSERVABILITY_SCHEMA_VERSION,
    REPORT_SUMMARY_SCHEMA_VERSION,
    ReportDegeneracyArtifact,
    ReportEvidenceArtifact,
    ReportMetricsArtifact,
    ReportObservabilityArtifact,
    ReportSummaryArtifact,
)
from calibrex.core.result import RESULT_SCHEMA_VERSION, CalibrationResult, StrictModel
from calibrex.core.solid_state import (
    SOLID_STATE_CONTEXT_SCHEMA_VERSION,
    SolidStateLidarCalibrationContext,
)
from calibrex.core.solid_state_cross_dataset_benchmark import (
    SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION,
    SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION_V0_1,
    SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION,
    SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION_V0_1,
    SolidStateCrossDatasetBenchmarkManifest,
    SolidStateCrossDatasetBenchmarkSpec,
)
from calibrex.core.solid_state_failure_analysis import (
    SOLID_STATE_FAILURE_ANALYSIS_SCHEMA_VERSION,
    SolidStateFailureAnalysisManifest,
)
from calibrex.core.solid_state_synthetic_benchmark import (
    SOLID_STATE_SYNTHETIC_BENCHMARK_SCHEMA_VERSION,
    SolidStateSyntheticBenchmarkArtifact,
)
from calibrex.core.trajectory import TRAJECTORY_SCHEMA_VERSION, TrajectoryArtifact
from calibrex.core.trajectory_window_drift import (
    TRAJECTORY_WINDOW_DRIFT_SCHEMA_VERSION,
    TrajectoryWindowDriftArtifact,
)
from calibrex.core.transform_artifacts import (
    TRANSFORMS_SCHEMA_VERSION,
    TransformArtifact,
)
from calibrex.data.depth import DEPTH_PROVIDER_SCHEMA_VERSION, DepthProviderArtifact
from calibrex.data.kitti_benchmark import (
    KITTI_BENCHMARK_INPUT_SCHEMA_VERSION,
    KITTIBenchmarkInputManifest,
)
from calibrex.data.manifest import DATASET_MANIFEST_SCHEMA_VERSION, DatasetManifest
from calibrex.diagnostics import DOCTOR_SCHEMA_VERSION, DoctorArtifact
from calibrex.evaluation.compare import COMPARISON_SCHEMA_VERSION, ResultComparison
from calibrex.evaluation.kitti_falsification_benchmark import (
    KITTI_FALSIFICATION_SCHEMA_VERSION,
    KITTIFalsificationBenchmarkArtifact,
)
from calibrex.evaluation.report_compare import (
    REPORT_COMPARISON_SCHEMA_VERSION,
    ReportComparison,
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
    "doctor",
    "calibration-ci",
    "external-run",
    "kitti-falsification",
    "kitti-benchmark-input",
    "depth-provider",
    "continuous-time-camera-lidar-problem",
    "continuous-time-camera-lidar-result",
    "probabilistic-correspondence",
    "probabilistic-pnp-result",
    "probabilistic-refinement-result",
    "camera-lidar-problem",
    "camera-lidar-sota-audit-protocol",
    "camera-lidar-sota-audit-result",
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
    "continuous-time-lidar-pair",
    "continuous-time-lidar-ablation",
    "solid-state-cross-dataset-benchmark-config",
    "solid-state-cross-dataset-benchmark",
    "solid-state-failure-analysis",
    "solid-state-synthetic-benchmark",
    "solid-state-context",
    "livox-time-ablation",
]

_MODEL_BY_KIND: Final[dict[str, type[BaseModel]]] = {
    "config": CalibrationConfig,
    "result": CalibrationResult,
    "comparison": ResultComparison,
    "report-comparison": ReportComparison,
    "dynamic-window-consistency": DynamicWindowConsistencyArtifact,
    "assessment": AssessmentArtifact,
    "benchmark": BenchmarkArtifact,
    "benchmark-definition": BenchmarkDefinition,
    "policy": PolicyArtifact,
    "protocol": ProtocolArtifact,
    "transforms": TransformArtifact,
    "dataset-manifest": DatasetManifest,
    "doctor": DoctorArtifact,
    "calibration-ci": CalibrationCIArtifact,
    "external-run": ExternalCalibrationRunArtifact,
    "kitti-falsification": KITTIFalsificationBenchmarkArtifact,
    "kitti-benchmark-input": KITTIBenchmarkInputManifest,
    "depth-provider": DepthProviderArtifact,
    "continuous-time-camera-lidar-problem": (
        ContinuousTimeCameraLidarProblemArtifact
    ),
    "continuous-time-camera-lidar-result": (
        ContinuousTimeCameraLidarResultArtifact
    ),
    "probabilistic-correspondence": ProbabilisticCorrespondenceArtifact,
    "probabilistic-pnp-result": ProbabilisticPnpResultArtifact,
    "probabilistic-refinement-result": ProbabilisticRefinementResultArtifact,
    "camera-lidar-problem": CameraLidarCalibrationProblem,
    "camera-lidar-sota-audit-protocol": CameraLidarSotaAuditProtocol,
    "camera-lidar-sota-audit-result": CameraLidarSotaAuditResult,
    "camera-lidar-benchmark-protocol": CameraLidarBenchmarkProtocol,
    "calibration-candidate-trace": CalibrationCandidateTrace,
    "bullseye-plot": BullseyePlotArtifact,
    "report-summary": ReportSummaryArtifact,
    "report-metrics": ReportMetricsArtifact,
    "report-observability": ReportObservabilityArtifact,
    "report-degeneracy": ReportDegeneracyArtifact,
    "report-evidence": ReportEvidenceArtifact,
    "evidence-bundle": EvidenceBundleManifest,
    "evidence-bundle-verification": EvidenceBundleVerification,
    "online-timeline": OnlineCalibrationTimelineArtifact,
    "trajectory": TrajectoryArtifact,
    "trajectory-window-drift": TrajectoryWindowDriftArtifact,
    "capture-readiness": CaptureReadinessArtifact,
    "continuous-time-lidar-pair": ContinuousTimeLidarPairArtifact,
    "continuous-time-lidar-ablation": ContinuousTimeLidarAblationManifest,
    "solid-state-cross-dataset-benchmark-config": SolidStateCrossDatasetBenchmarkSpec,
    "solid-state-cross-dataset-benchmark": SolidStateCrossDatasetBenchmarkManifest,
    "solid-state-failure-analysis": SolidStateFailureAnalysisManifest,
    "solid-state-synthetic-benchmark": SolidStateSyntheticBenchmarkArtifact,
    "solid-state-context": SolidStateLidarCalibrationContext,
    "livox-time-ablation": LivoxTimeAblationManifest,
}

_KIND_BY_SCHEMA_VERSION: Final[dict[str, str]] = {
    CONFIG_SCHEMA_VERSION: "config",
    RESULT_SCHEMA_VERSION: "result",
    COMPARISON_SCHEMA_VERSION: "comparison",
    REPORT_COMPARISON_SCHEMA_VERSION: "report-comparison",
    DYNAMIC_WINDOW_CONSISTENCY_SCHEMA_VERSION: "dynamic-window-consistency",
    ASSESSMENT_SCHEMA_VERSION: "assessment",
    BENCHMARK_SCHEMA_VERSION: "benchmark",
    BENCHMARK_DEFINITION_SCHEMA_VERSION: "benchmark-definition",
    POLICY_SCHEMA_VERSION: "policy",
    PROTOCOL_SCHEMA_VERSION: "protocol",
    TRANSFORMS_SCHEMA_VERSION: "transforms",
    DATASET_MANIFEST_SCHEMA_VERSION: "dataset-manifest",
    DOCTOR_SCHEMA_VERSION: "doctor",
    CALIBRATION_CI_SCHEMA_VERSION: "calibration-ci",
    EXTERNAL_RUN_SCHEMA_VERSION: "external-run",
    KITTI_FALSIFICATION_SCHEMA_VERSION: "kitti-falsification",
    KITTI_BENCHMARK_INPUT_SCHEMA_VERSION: "kitti-benchmark-input",
    DEPTH_PROVIDER_SCHEMA_VERSION: "depth-provider",
    CONTINUOUS_TIME_CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION: (
        "continuous-time-camera-lidar-problem"
    ),
    CONTINUOUS_TIME_CAMERA_LIDAR_RESULT_SCHEMA_VERSION: (
        "continuous-time-camera-lidar-result"
    ),
    PROBABILISTIC_CORRESPONDENCE_SCHEMA_VERSION: "probabilistic-correspondence",
    PROBABILISTIC_PNP_RESULT_SCHEMA_VERSION: "probabilistic-pnp-result",
    PROBABILISTIC_REFINEMENT_RESULT_SCHEMA_VERSION: (
        "probabilistic-refinement-result"
    ),
    CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION: "camera-lidar-problem",
    CAMERA_LIDAR_SOTA_AUDIT_PROTOCOL_SCHEMA_VERSION: (
        "camera-lidar-sota-audit-protocol"
    ),
    CAMERA_LIDAR_SOTA_AUDIT_RESULT_SCHEMA_VERSION: (
        "camera-lidar-sota-audit-result"
    ),
    CAMERA_LIDAR_BENCHMARK_PROTOCOL_SCHEMA_VERSION: (
        "camera-lidar-benchmark-protocol"
    ),
    CALIBRATION_CANDIDATE_TRACE_SCHEMA_VERSION: "calibration-candidate-trace",
    BULLSEYE_PLOT_SCHEMA_VERSION: "bullseye-plot",
    REPORT_SUMMARY_SCHEMA_VERSION: "report-summary",
    REPORT_METRICS_SCHEMA_VERSION: "report-metrics",
    REPORT_OBSERVABILITY_SCHEMA_VERSION: "report-observability",
    REPORT_DEGENERACY_SCHEMA_VERSION: "report-degeneracy",
    REPORT_EVIDENCE_SCHEMA_VERSION: "report-evidence",
    EVIDENCE_BUNDLE_SCHEMA_VERSION: "evidence-bundle",
    EVIDENCE_BUNDLE_VERIFICATION_SCHEMA_VERSION: "evidence-bundle-verification",
    ONLINE_TIMELINE_SCHEMA_VERSION: "online-timeline",
    TRAJECTORY_SCHEMA_VERSION: "trajectory",
    TRAJECTORY_WINDOW_DRIFT_SCHEMA_VERSION: "trajectory-window-drift",
    CAPTURE_READINESS_SCHEMA_VERSION: "capture-readiness",
    CONTINUOUS_TIME_LIDAR_PAIR_SCHEMA_VERSION: "continuous-time-lidar-pair",
    CONTINUOUS_TIME_LIDAR_ABLATION_SCHEMA_VERSION: "continuous-time-lidar-ablation",
    SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION: (
        "solid-state-cross-dataset-benchmark-config"
    ),
    SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION_V0_1: (
        "solid-state-cross-dataset-benchmark-config"
    ),
    SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION: (
        "solid-state-cross-dataset-benchmark"
    ),
    SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION_V0_1: (
        "solid-state-cross-dataset-benchmark"
    ),
    SOLID_STATE_FAILURE_ANALYSIS_SCHEMA_VERSION: "solid-state-failure-analysis",
    SOLID_STATE_SYNTHETIC_BENCHMARK_SCHEMA_VERSION: "solid-state-synthetic-benchmark",
    SOLID_STATE_CONTEXT_SCHEMA_VERSION: "solid-state-context",
    LIVOX_TIME_ABLATION_SCHEMA_VERSION: "livox-time-ablation",
}


class ValidationReport(StrictModel):
    """Machine-readable validation result."""

    path: str
    kind: str
    schema_version: str
    valid: bool = True


def validation_kinds() -> tuple[str, ...]:
    """Return supported explicit validation kinds."""

    return tuple(_MODEL_BY_KIND)


def validation_kind_choices() -> tuple[str, ...]:
    """Return all CLI validation kind choices, including auto-detection."""

    return ("auto", *validation_kinds())


def validate_file(path: str | Path, kind: ValidationKind = "auto") -> ValidationReport:
    """Validate a schema-versioned Calibrex artifact."""

    artifact_path = Path(path)
    payload = read_mapping(artifact_path)
    detected_kind = _detect_kind(payload) if kind == "auto" else kind
    model = _model_for_kind(detected_kind)
    validated = model.model_validate(payload)
    schema_version = _schema_version(validated)
    return ValidationReport(
        path=artifact_path.as_posix(),
        kind=detected_kind,
        schema_version=schema_version,
    )


def _detect_kind(payload: dict[str, object]) -> str:
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str):
        msg = "cannot auto-detect artifact kind without a string schema_version"
        raise CalibrexError(msg)
    try:
        return _KIND_BY_SCHEMA_VERSION[schema_version]
    except KeyError as exc:
        supported = ", ".join(sorted(_KIND_BY_SCHEMA_VERSION))
        msg = f"unsupported schema_version {schema_version!r}; expected one of {supported}"
        raise CalibrexError(msg) from exc


def _model_for_kind(kind: str) -> type[BaseModel]:
    try:
        return _MODEL_BY_KIND[kind]
    except KeyError as exc:
        supported = ", ".join(validation_kind_choices())
        msg = f"unsupported validation kind {kind!r}; expected one of {supported}"
        raise CalibrexError(msg) from exc


def _schema_version(model: BaseModel) -> str:
    value = getattr(model, "schema_version", None)
    if not isinstance(value, str):
        msg = "validated artifact did not expose a string schema_version"
        raise CalibrexError(msg)
    return value
