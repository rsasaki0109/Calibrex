"""Validation helpers for schema-versioned Calibrex artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, Field

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
from calibrex.core.calibration_lifecycle import (
    CALIBRATION_LIFECYCLE_SCHEMA_VERSION,
    CalibrationLifecycleArtifact,
)
from calibrex.core.camera_imu_service import (
    CAMERA_IMU_SERVICE_EVALUATION_SCHEMA_VERSION,
    CAMERA_IMU_SERVICE_PLAN_SCHEMA_VERSION,
    CameraImuServiceEvaluation,
    CameraImuServicePlan,
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
from calibrex.core.camera_lidar_confidence_calibration import (
    CAMERA_LIDAR_CONFIDENCE_CALIBRATION_SCHEMA_VERSION,
    CameraLidarConfidenceCalibrationArtifact,
)
from calibrex.core.camera_lidar_correspondence_export import (
    CAMERA_LIDAR_CORRESPONDENCE_EXPORT_SCHEMA_VERSION,
    CameraLidarCorrespondenceExportManifest,
)
from calibrex.core.camera_lidar_correspondence_quality import (
    CAMERA_LIDAR_CORRESPONDENCE_QUALITY_SCHEMA_VERSION,
    CameraLidarCorrespondenceQualityArtifact,
)
from calibrex.core.camera_lidar_failure_analysis import (
    CAMERA_LIDAR_FAILURE_ANALYSIS_SCHEMA_VERSION,
    CameraLidarFailureAnalysisArtifact,
)
from calibrex.core.camera_lidar_initializer_calibration import (
    CAMERA_LIDAR_INITIALIZER_CALIBRATION_SCHEMA_VERSION,
    CameraLidarInitializerCalibrationArtifact,
)
from calibrex.core.camera_lidar_pose_initializer_benchmark import (
    CAMERA_LIDAR_POSE_INITIALIZER_PROTOCOL_SCHEMA_VERSION,
    CameraLidarPoseInitializerBenchmarkProtocol,
)
from calibrex.core.camera_lidar_pose_initializer_failure_analysis import (
    CAMERA_LIDAR_POSE_INITIALIZER_FAILURE_ANALYSIS_SCHEMA_VERSION,
    CameraLidarPoseInitializerFailureAnalysis,
)
from calibrex.core.camera_lidar_provider_support_comparison import (
    CAMERA_LIDAR_PROVIDER_SUPPORT_COMPARISON_SCHEMA_VERSION,
    CameraLidarProviderSupportComparisonArtifact,
)
from calibrex.core.camera_lidar_sota_audit import (
    CAMERA_LIDAR_SOTA_AUDIT_PROTOCOL_SCHEMA_VERSION,
    CAMERA_LIDAR_SOTA_AUDIT_RESULT_SCHEMA_VERSION,
    CameraLidarSotaAuditProtocol,
    CameraLidarSotaAuditResult,
)
from calibrex.core.capture_manifest import (
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    CAPTURE_MANIFEST_VERIFICATION_SCHEMA_VERSION,
    CaptureManifest,
    CaptureManifestVerification,
    verify_capture_manifest_inputs,
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
from calibrex.core.continuous_time_contract import (
    CONTINUOUS_TIME_TRAJECTORY_SCHEMA_VERSION,
    ContinuousTimeTrajectoryContract,
)
from calibrex.core.continuous_time_fit_artifacts import (
    CONTINUOUS_TIME_FIT_SCHEMA_VERSION,
    CONTINUOUS_TIME_MEASUREMENTS_SCHEMA_VERSION,
    ContinuousTimeTrajectoryFitResultArtifact,
    ContinuousTimeTrajectoryMeasurements,
)
from calibrex.core.continuous_time_imu_accel_bias import (
    CONTINUOUS_TIME_IMU_ACCEL_BIAS_SCHEMA_VERSION,
    ContinuousTimeImuAccelBiasArtifact,
)
from calibrex.core.continuous_time_imu_clock_offset import (
    CONTINUOUS_TIME_IMU_CLOCK_OFFSET_SCHEMA_VERSION,
    ContinuousTimeImuClockOffsetArtifact,
)
from calibrex.core.continuous_time_imu_intrinsics import (
    CONTINUOUS_TIME_IMU_INTRINSICS_SCHEMA_VERSION,
    ContinuousTimeImuIntrinsicsArtifact,
)
from calibrex.core.continuous_time_imu_lever_arm import (
    CONTINUOUS_TIME_IMU_LEVER_ARM_SCHEMA_VERSION,
    ContinuousTimeImuLeverArmArtifact,
)
from calibrex.core.continuous_time_imu_preintegration import (
    CONTINUOUS_TIME_IMU_PREINTEGRATION_SCHEMA_VERSION,
    ContinuousTimeImuPreintegrationArtifact,
)
from calibrex.core.continuous_time_lidar_ablation import (
    CONTINUOUS_TIME_LIDAR_ABLATION_SCHEMA_VERSION,
    ContinuousTimeLidarAblationManifest,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    CONTINUOUS_TIME_LIDAR_PAIR_SCHEMA_VERSION,
    ContinuousTimeLidarPairArtifact,
)
from calibrex.core.continuous_time_lidar_point_to_plane import (
    CONTINUOUS_TIME_LIDAR_POINT_TO_PLANE_SCHEMA_VERSION,
    ContinuousTimeLidarPointToPlaneArtifact,
)
from calibrex.core.continuous_time_lidar_train_diagnostics import (
    CONTINUOUS_TIME_LIDAR_TRAIN_DIAGNOSTICS_SCHEMA_VERSION,
    ContinuousTimeLidarTrainDiagnostics,
)
from calibrex.core.continuous_time_sliding_window import (
    CONTINUOUS_TIME_SLIDING_WINDOW_SCHEMA_VERSION,
    ContinuousTimeSlidingWindowArtifact,
)
from calibrex.core.dynamic_window import (
    DYNAMIC_WINDOW_CONSISTENCY_SCHEMA_VERSION,
    DynamicWindowConsistencyArtifact,
)
from calibrex.core.empirical_uncertainty import (
    EMPIRICAL_SE3_UNCERTAINTY_SCHEMA_VERSION,
    EmpiricalSe3UncertaintyArtifact,
)
from calibrex.core.environment_readiness import (
    ENVIRONMENT_READINESS_SCHEMA_VERSION,
    EnvironmentReadinessArtifact,
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
from calibrex.core.koide_handoff import (
    KOIDE_EXECUTION_LOCK_SCHEMA_VERSION,
    KoideExecutionLock,
)
from calibrex.core.koide_readiness import (
    KOIDE_READINESS_SCHEMA_VERSION,
    KoideReadinessArtifact,
)
from calibrex.core.lifecycle_registry import (
    LIFECYCLE_EVALUATION_SCHEMA_VERSION,
    LIFECYCLE_EVENT_SCHEMA_VERSION,
    LIFECYCLE_REGISTRY_SCHEMA_VERSION,
    LifecycleEvaluationArtifact,
    LifecycleEvent,
    LifecycleRegistryState,
    LifecycleRegistryStatus,
    RegistryHead,
    RegistryManifest,
    RegistryVerificationArtifact,
)
from calibrex.core.livox_time_ablation import (
    LIVOX_TIME_ABLATION_SCHEMA_VERSION,
    LivoxTimeAblationManifest,
)
from calibrex.core.mcap_integrity import (
    MCAP_INTEGRITY_SCHEMA_VERSION,
    McapIntegrityEvidence,
)
from calibrex.core.multi_lidar_service import (
    MULTI_LIDAR_SERVICE_EVALUATION_SCHEMA_VERSION,
    MULTI_LIDAR_SERVICE_PLAN_SCHEMA_VERSION,
    MultiLidarServiceEvaluation,
    MultiLidarServicePlan,
)
from calibrex.core.online_timeline import (
    ONLINE_TIMELINE_SCHEMA_VERSION,
    OnlineCalibrationTimelineArtifact,
)
from calibrex.core.probabilistic_correspondence import (
    PROBABILISTIC_CORRESPONDENCE_SCHEMA_VERSION,
    PROBABILISTIC_PNP_RESULT_SCHEMA_VERSION,
    PROBABILISTIC_PNP_RESULT_SCHEMA_VERSION_V0_1,
    PROBABILISTIC_REFINEMENT_RESULT_SCHEMA_VERSION,
    PROBABILISTIC_REFINEMENT_RESULT_SCHEMA_VERSION_V0_1,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticPnpResultArtifact,
    ProbabilisticRefinementResultArtifact,
)
from calibrex.core.radar_service import (
    RADAR_SERVICE_EVALUATION_SCHEMA_VERSION,
    RADAR_SERVICE_PLAN_SCHEMA_VERSION,
    RadarServiceEvaluation,
    RadarServicePlan,
)
from calibrex.core.raw_replay import (
    FIELD_REPLACEMENT_PILOT_SCHEMA_VERSION,
    RAW_REPLAY_COMPARISON_SCHEMA_VERSION,
    RAW_REPLAY_DEFINITION_SCHEMA_VERSION,
    RAW_REPLAY_PLAN_SCHEMA_VERSION,
    RAW_REPLAY_RESULT_SCHEMA_VERSION,
    RAW_REPLAY_STAGE_SCHEMA_VERSION,
    FieldReplacementPilot,
    ReplayComparison,
    ReplayDefinition,
    ReplayPlan,
    ReplayResult,
    ReplayStageArtifact,
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
from calibrex.core.solid_state_metrology_evaluation import (
    SOLID_STATE_METROLOGY_EVALUATION_SCHEMA_VERSION,
    SolidStateMetrologyEvaluationArtifact,
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
from calibrex.data.kitti360_lidar_window_integration import (
    KITTI360_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION,
    KITTI360_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION_V0_1,
    Kitti360LidarWindowIntegrationArtifact,
)
from calibrex.data.kitti_benchmark import (
    KITTI_BENCHMARK_INPUT_SCHEMA_VERSION,
    KITTIBenchmarkInputManifest,
)
from calibrex.data.kitti_raw_lidar_window_integration import (
    KITTI_RAW_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION,
    KittiRawLidarWindowIntegrationArtifact,
)
from calibrex.data.manifest import DATASET_MANIFEST_SCHEMA_VERSION, DatasetManifest
from calibrex.data.remote_archive_selection import (
    REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION,
    REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3,
    RemoteArchiveSelectionArtifact,
)
from calibrex.diagnostics import DOCTOR_SCHEMA_VERSION, DoctorArtifact
from calibrex.evaluation.compare import COMPARISON_SCHEMA_VERSION, ResultComparison
from calibrex.evaluation.kitti_falsification_benchmark import (
    KITTI_FALSIFICATION_SCHEMA_VERSION,
    KITTIFalsificationBenchmarkArtifact,
)
from calibrex.evaluation.koide_pilot import (
    KOIDE_PILOT_SCHEMA_VERSION,
    KoidePilotArtifact,
)
from calibrex.evaluation.report_compare import (
    REPORT_COMPARISON_SCHEMA_VERSION,
    ReportComparison,
)
from calibrex.export.autoware import (
    AUTOWARE_EXPORT_SCHEMA_VERSION,
    AutowareExportArtifact,
)
from calibrex.export.autoware_promotion import (
    AUTOWARE_PROMOTION_SCHEMA_VERSION,
    AutowarePromotionArtifact,
)
from calibrex.export.autoware_smoke import (
    AUTOWARE_SMOKE_SCHEMA_VERSION,
    AutowareSmokeArtifact,
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
    "remote-archive-selection": RemoteArchiveSelectionArtifact,
    "kitti360-lidar-window-integration": Kitti360LidarWindowIntegrationArtifact,
    "kitti-raw-lidar-window-integration": KittiRawLidarWindowIntegrationArtifact,
    "doctor": DoctorArtifact,
    "environment-readiness": EnvironmentReadinessArtifact,
    "calibration-ci": CalibrationCIArtifact,
    "calibration-lifecycle": CalibrationLifecycleArtifact,
    "lifecycle-registry": RegistryManifest,
    "lifecycle-event": LifecycleEvent,
    "lifecycle-evaluation": LifecycleEvaluationArtifact,
    "lifecycle-registry-state": LifecycleRegistryState,
    "lifecycle-head": RegistryHead,
    "lifecycle-verification": RegistryVerificationArtifact,
    "lifecycle-status": LifecycleRegistryStatus,
    "external-run": ExternalCalibrationRunArtifact,
    "kitti-falsification": KITTIFalsificationBenchmarkArtifact,
    "kitti-benchmark-input": KITTIBenchmarkInputManifest,
    "koide-pilot": KoidePilotArtifact,
    "koide-execution-lock": KoideExecutionLock,
    "depth-provider": DepthProviderArtifact,
    "continuous-time-camera-lidar-problem": (
        ContinuousTimeCameraLidarProblemArtifact
    ),
    "continuous-time-camera-lidar-result": (
        ContinuousTimeCameraLidarResultArtifact
    ),
    "continuous-time-trajectory": ContinuousTimeTrajectoryContract,
    "continuous-time-trajectory-measurements": (
        ContinuousTimeTrajectoryMeasurements
    ),
    "continuous-time-trajectory-fit": (
        ContinuousTimeTrajectoryFitResultArtifact
    ),
    "probabilistic-correspondence": ProbabilisticCorrespondenceArtifact,
    "probabilistic-pnp-result": ProbabilisticPnpResultArtifact,
    "probabilistic-refinement-result": ProbabilisticRefinementResultArtifact,
    "empirical-se3-uncertainty": EmpiricalSe3UncertaintyArtifact,
    "camera-lidar-problem": CameraLidarCalibrationProblem,
    "camera-lidar-correspondence-export": CameraLidarCorrespondenceExportManifest,
    "camera-lidar-confidence-calibration": CameraLidarConfidenceCalibrationArtifact,
    "camera-lidar-provider-support-comparison": (CameraLidarProviderSupportComparisonArtifact),
    "camera-lidar-pose-initializer-protocol": (CameraLidarPoseInitializerBenchmarkProtocol),
    "camera-lidar-pose-initializer-failure-analysis": (
        CameraLidarPoseInitializerFailureAnalysis
    ),
    "camera-lidar-initializer-calibration": (CameraLidarInitializerCalibrationArtifact),
    "camera-lidar-correspondence-quality": CameraLidarCorrespondenceQualityArtifact,
    "camera-lidar-failure-analysis": CameraLidarFailureAnalysisArtifact,
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
    "capture-manifest": CaptureManifest,
    "capture-manifest-verification": CaptureManifestVerification,
    "mcap-integrity": McapIntegrityEvidence,
    "koide-readiness": KoideReadinessArtifact,
    "continuous-time-lidar-pair": ContinuousTimeLidarPairArtifact,
    "continuous-time-lidar-point-to-plane": ContinuousTimeLidarPointToPlaneArtifact,
    "continuous-time-imu-preintegration": ContinuousTimeImuPreintegrationArtifact,
    "continuous-time-imu-lever-arm": ContinuousTimeImuLeverArmArtifact,
    "continuous-time-imu-clock-offset": ContinuousTimeImuClockOffsetArtifact,
    "continuous-time-imu-accel-bias": ContinuousTimeImuAccelBiasArtifact,
    "continuous-time-imu-intrinsics": ContinuousTimeImuIntrinsicsArtifact,
    "continuous-time-sliding-window": ContinuousTimeSlidingWindowArtifact,
    "continuous-time-lidar-train-diagnostics": ContinuousTimeLidarTrainDiagnostics,
    "continuous-time-lidar-ablation": ContinuousTimeLidarAblationManifest,
    "solid-state-cross-dataset-benchmark-config": SolidStateCrossDatasetBenchmarkSpec,
    "solid-state-cross-dataset-benchmark": SolidStateCrossDatasetBenchmarkManifest,
    "solid-state-failure-analysis": SolidStateFailureAnalysisManifest,
    "solid-state-synthetic-benchmark": SolidStateSyntheticBenchmarkArtifact,
    "solid-state-metrology-evaluation": SolidStateMetrologyEvaluationArtifact,
    "solid-state-context": SolidStateLidarCalibrationContext,
    "livox-time-ablation": LivoxTimeAblationManifest,
    "autoware-export": AutowareExportArtifact,
    "autoware-promotion": AutowarePromotionArtifact,
    "autoware-smoke": AutowareSmokeArtifact,
    "raw-replay-definition": ReplayDefinition,
    "raw-replay-plan": ReplayPlan,
    "raw-replay-stage": ReplayStageArtifact,
    "raw-replay-result": ReplayResult,
    "raw-replay-comparison": ReplayComparison,
    "field-replacement-pilot": FieldReplacementPilot,
    "multi-lidar-service-plan": MultiLidarServicePlan,
    "multi-lidar-service-evaluation": MultiLidarServiceEvaluation,
    "camera-imu-service-plan": CameraImuServicePlan,
    "camera-imu-service-evaluation": CameraImuServiceEvaluation,
    "radar-service-plan": RadarServicePlan,
    "radar-service-evaluation": RadarServiceEvaluation,
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
    REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION: "remote-archive-selection",
    REMOTE_ARCHIVE_SELECTION_SCHEMA_VERSION_V0_3: "remote-archive-selection",
    KITTI360_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION: ("kitti360-lidar-window-integration"),
    KITTI360_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION_V0_1: ("kitti360-lidar-window-integration"),
    KITTI_RAW_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION: ("kitti-raw-lidar-window-integration"),
    DOCTOR_SCHEMA_VERSION: "doctor",
    ENVIRONMENT_READINESS_SCHEMA_VERSION: "environment-readiness",
    CALIBRATION_CI_SCHEMA_VERSION: "calibration-ci",
    CALIBRATION_LIFECYCLE_SCHEMA_VERSION: "calibration-lifecycle",
    LIFECYCLE_REGISTRY_SCHEMA_VERSION: "lifecycle-registry",
    LIFECYCLE_EVENT_SCHEMA_VERSION: "lifecycle-event",
    LIFECYCLE_EVALUATION_SCHEMA_VERSION: "lifecycle-evaluation",
    "slac.calibration_lifecycle_registry_state/v0.2": "lifecycle-registry-state",
    "slac.calibration_lifecycle_head/v0.2": "lifecycle-head",
    "slac.calibration_lifecycle_registry_verification/v0.2": "lifecycle-verification",
    "slac.calibration_lifecycle_status/v0.2": "lifecycle-status",
    EXTERNAL_RUN_SCHEMA_VERSION: "external-run",
    KITTI_FALSIFICATION_SCHEMA_VERSION: "kitti-falsification",
    KITTI_BENCHMARK_INPUT_SCHEMA_VERSION: "kitti-benchmark-input",
    KOIDE_PILOT_SCHEMA_VERSION: "koide-pilot",
    KOIDE_EXECUTION_LOCK_SCHEMA_VERSION: "koide-execution-lock",
    DEPTH_PROVIDER_SCHEMA_VERSION: "depth-provider",
    CONTINUOUS_TIME_CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION: (
        "continuous-time-camera-lidar-problem"
    ),
    CONTINUOUS_TIME_CAMERA_LIDAR_RESULT_SCHEMA_VERSION: (
        "continuous-time-camera-lidar-result"
    ),
    CONTINUOUS_TIME_TRAJECTORY_SCHEMA_VERSION: "continuous-time-trajectory",
    CONTINUOUS_TIME_MEASUREMENTS_SCHEMA_VERSION: (
        "continuous-time-trajectory-measurements"
    ),
    CONTINUOUS_TIME_FIT_SCHEMA_VERSION: "continuous-time-trajectory-fit",
    PROBABILISTIC_CORRESPONDENCE_SCHEMA_VERSION: "probabilistic-correspondence",
    PROBABILISTIC_PNP_RESULT_SCHEMA_VERSION: "probabilistic-pnp-result",
    PROBABILISTIC_PNP_RESULT_SCHEMA_VERSION_V0_1: "probabilistic-pnp-result",
    PROBABILISTIC_REFINEMENT_RESULT_SCHEMA_VERSION: (
        "probabilistic-refinement-result"
    ),
    PROBABILISTIC_REFINEMENT_RESULT_SCHEMA_VERSION_V0_1: (
        "probabilistic-refinement-result"
    ),
    EMPIRICAL_SE3_UNCERTAINTY_SCHEMA_VERSION: "empirical-se3-uncertainty",
    CAMERA_LIDAR_PROBLEM_SCHEMA_VERSION: "camera-lidar-problem",
    CAMERA_LIDAR_CORRESPONDENCE_EXPORT_SCHEMA_VERSION: ("camera-lidar-correspondence-export"),
    CAMERA_LIDAR_CONFIDENCE_CALIBRATION_SCHEMA_VERSION: ("camera-lidar-confidence-calibration"),
    CAMERA_LIDAR_PROVIDER_SUPPORT_COMPARISON_SCHEMA_VERSION: (
        "camera-lidar-provider-support-comparison"
    ),
    CAMERA_LIDAR_POSE_INITIALIZER_PROTOCOL_SCHEMA_VERSION: (
        "camera-lidar-pose-initializer-protocol"
    ),
    CAMERA_LIDAR_POSE_INITIALIZER_FAILURE_ANALYSIS_SCHEMA_VERSION: (
        "camera-lidar-pose-initializer-failure-analysis"
    ),
    CAMERA_LIDAR_INITIALIZER_CALIBRATION_SCHEMA_VERSION: ("camera-lidar-initializer-calibration"),
    CAMERA_LIDAR_CORRESPONDENCE_QUALITY_SCHEMA_VERSION: ("camera-lidar-correspondence-quality"),
    CAMERA_LIDAR_FAILURE_ANALYSIS_SCHEMA_VERSION: "camera-lidar-failure-analysis",
    CAMERA_LIDAR_SOTA_AUDIT_PROTOCOL_SCHEMA_VERSION: ("camera-lidar-sota-audit-protocol"),
    CAMERA_LIDAR_SOTA_AUDIT_RESULT_SCHEMA_VERSION: ("camera-lidar-sota-audit-result"),
    CAMERA_LIDAR_BENCHMARK_PROTOCOL_SCHEMA_VERSION: ("camera-lidar-benchmark-protocol"),
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
    CAPTURE_MANIFEST_SCHEMA_VERSION: "capture-manifest",
    CAPTURE_MANIFEST_VERIFICATION_SCHEMA_VERSION: "capture-manifest-verification",
    MCAP_INTEGRITY_SCHEMA_VERSION: "mcap-integrity",
    KOIDE_READINESS_SCHEMA_VERSION: "koide-readiness",
    CONTINUOUS_TIME_LIDAR_PAIR_SCHEMA_VERSION: "continuous-time-lidar-pair",
    CONTINUOUS_TIME_LIDAR_POINT_TO_PLANE_SCHEMA_VERSION: (
        "continuous-time-lidar-point-to-plane"
    ),
    CONTINUOUS_TIME_IMU_PREINTEGRATION_SCHEMA_VERSION: (
        "continuous-time-imu-preintegration"
    ),
    CONTINUOUS_TIME_IMU_LEVER_ARM_SCHEMA_VERSION: (
        "continuous-time-imu-lever-arm"
    ),
    CONTINUOUS_TIME_IMU_CLOCK_OFFSET_SCHEMA_VERSION: (
        "continuous-time-imu-clock-offset"
    ),
    CONTINUOUS_TIME_IMU_ACCEL_BIAS_SCHEMA_VERSION: (
        "continuous-time-imu-accel-bias"
    ),
    CONTINUOUS_TIME_IMU_INTRINSICS_SCHEMA_VERSION: (
        "continuous-time-imu-intrinsics"
    ),
    CONTINUOUS_TIME_SLIDING_WINDOW_SCHEMA_VERSION: (
        "continuous-time-sliding-window"
    ),
    CONTINUOUS_TIME_LIDAR_TRAIN_DIAGNOSTICS_SCHEMA_VERSION: (
        "continuous-time-lidar-train-diagnostics"
    ),
    CONTINUOUS_TIME_LIDAR_ABLATION_SCHEMA_VERSION: "continuous-time-lidar-ablation",
    SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION: (
        "solid-state-cross-dataset-benchmark-config"
    ),
    SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION_V0_1: (
        "solid-state-cross-dataset-benchmark-config"
    ),
    SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION: ("solid-state-cross-dataset-benchmark"),
    SOLID_STATE_CROSS_DATASET_BENCHMARK_SCHEMA_VERSION_V0_1: (
        "solid-state-cross-dataset-benchmark"
    ),
    SOLID_STATE_FAILURE_ANALYSIS_SCHEMA_VERSION: "solid-state-failure-analysis",
    SOLID_STATE_SYNTHETIC_BENCHMARK_SCHEMA_VERSION: "solid-state-synthetic-benchmark",
    SOLID_STATE_METROLOGY_EVALUATION_SCHEMA_VERSION: "solid-state-metrology-evaluation",
    SOLID_STATE_CONTEXT_SCHEMA_VERSION: "solid-state-context",
    LIVOX_TIME_ABLATION_SCHEMA_VERSION: "livox-time-ablation",
    AUTOWARE_EXPORT_SCHEMA_VERSION: "autoware-export",
    AUTOWARE_PROMOTION_SCHEMA_VERSION: "autoware-promotion",
    AUTOWARE_SMOKE_SCHEMA_VERSION: "autoware-smoke",
    RAW_REPLAY_DEFINITION_SCHEMA_VERSION: "raw-replay-definition",
    RAW_REPLAY_PLAN_SCHEMA_VERSION: "raw-replay-plan",
    RAW_REPLAY_STAGE_SCHEMA_VERSION: "raw-replay-stage",
    RAW_REPLAY_RESULT_SCHEMA_VERSION: "raw-replay-result",
    RAW_REPLAY_COMPARISON_SCHEMA_VERSION: "raw-replay-comparison",
    FIELD_REPLACEMENT_PILOT_SCHEMA_VERSION: "field-replacement-pilot",
    MULTI_LIDAR_SERVICE_PLAN_SCHEMA_VERSION: "multi-lidar-service-plan",
    MULTI_LIDAR_SERVICE_EVALUATION_SCHEMA_VERSION: "multi-lidar-service-evaluation",
    CAMERA_IMU_SERVICE_PLAN_SCHEMA_VERSION: "camera-imu-service-plan",
    CAMERA_IMU_SERVICE_EVALUATION_SCHEMA_VERSION: "camera-imu-service-evaluation",
    RADAR_SERVICE_PLAN_SCHEMA_VERSION: "radar-service-plan",
    RADAR_SERVICE_EVALUATION_SCHEMA_VERSION: "radar-service-evaluation",
}


class ValidationReport(StrictModel):
    """Machine-readable validation result."""

    path: str
    kind: str
    schema_version: str
    valid: bool = True
    input_verification: CaptureManifestVerification | None = Field(default=None, exclude=True)


def validation_kinds() -> tuple[str, ...]:
    """Return supported explicit validation kinds."""

    return tuple(_MODEL_BY_KIND)


def validation_kind_choices() -> tuple[str, ...]:
    """Return all CLI validation kind choices, including auto-detection."""

    return ("auto", *validation_kinds())


def validate_file(
    path: str | Path,
    kind: ValidationKind = "auto",
    *,
    verify_inputs: bool = False,
) -> ValidationReport:
    """Validate a schema-versioned Calibrex artifact."""

    artifact_path = Path(path)
    payload = read_mapping(artifact_path)
    detected_kind = _detect_kind(payload) if kind == "auto" else kind
    model = _model_for_kind(detected_kind)
    validated = model.model_validate(payload)
    if isinstance(
        validated,
        (
            KoidePilotArtifact,
            KoideReadinessArtifact,
            ReplayDefinition,
            ReplayPlan,
            ReplayStageArtifact,
            ReplayResult,
            ReplayComparison,
            FieldReplacementPilot,
            MultiLidarServicePlan,
            MultiLidarServiceEvaluation,
            CameraImuServicePlan,
            CameraImuServiceEvaluation,
            RadarServicePlan,
            RadarServiceEvaluation,
        ),
    ):
        validated.verify_artifact_digest()
    elif isinstance(validated, KoideExecutionLock):
        validated.verify_lock_digest()
    elif isinstance(validated, CaptureManifest):
        validated.verify_artifact_digest()
        if verify_inputs:
            verification = verify_capture_manifest_inputs(artifact_path)
            return ValidationReport(
                path=artifact_path.as_posix(),
                kind=detected_kind,
                schema_version=_schema_version(validated),
                valid=verification.valid,
                input_verification=verification,
            )
    elif verify_inputs:
        raise CalibrexError("--verify-inputs is only supported for capture-manifest artifacts")
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
