"""Single source of truth for schema-versioned Calibrex artifact kinds.

Every artifact kind that ``calibrex validate`` can read and every JSON Schema
that ``calibrex schema all`` writes is declared exactly once in
``SCHEMA_REGISTRY``.  Accepted schema versions are derived from each model's
``schema_version`` ``Literal`` annotation rather than from a hand-maintained
table, so a model that learns to read an older version is detected
automatically.

Released schema versions must never silently disappear.  A version that the
current models no longer read is either upgraded by a registered
``SchemaMigration`` (only when the conversion is lossless and invents no
values) or listed in ``RETIRED_SCHEMA_VERSIONS`` with the release that retired
it and the remedy users should apply.  ``schema_ledger()`` summarizes all of
this as a machine-readable ledger committed next to the generated schemas.
"""

from __future__ import annotations

import hashlib
import json
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, Literal

from pydantic import BaseModel

from calibrex.calibration_ci import (
    CalibrationCIArtifact,
    calibration_ci_json_schema,
)
from calibrex.core.assessment import (
    AssessmentArtifact,
    assessment_json_schema,
)
from calibrex.core.benchmark import (
    BenchmarkArtifact,
    BenchmarkDefinition,
    benchmark_definition_json_schema,
    benchmark_json_schema,
)
from calibrex.core.calibration_lifecycle import (
    CalibrationLifecycleArtifact,
    calibration_lifecycle_json_schema,
)
from calibrex.core.camera_imu_service import (
    CameraImuServiceEvaluation,
    CameraImuServicePlan,
    camera_imu_service_evaluation_json_schema,
    camera_imu_service_plan_json_schema,
)
from calibrex.core.camera_lidar_artifacts import (
    BullseyePlotArtifact,
    CalibrationCandidateTrace,
    CameraLidarBenchmarkProtocol,
    CameraLidarCalibrationProblem,
    bullseye_plot_json_schema,
    calibration_candidate_trace_json_schema,
    camera_lidar_benchmark_protocol_json_schema,
    camera_lidar_problem_json_schema,
)
from calibrex.core.camera_lidar_confidence_calibration import (
    CameraLidarConfidenceCalibrationArtifact,
    camera_lidar_confidence_calibration_json_schema,
)
from calibrex.core.camera_lidar_correspondence_export import (
    CameraLidarCorrespondenceExportManifest,
    camera_lidar_correspondence_export_json_schema,
)
from calibrex.core.camera_lidar_correspondence_quality import (
    CameraLidarCorrespondenceQualityArtifact,
    camera_lidar_correspondence_quality_json_schema,
)
from calibrex.core.camera_lidar_failure_analysis import (
    CameraLidarFailureAnalysisArtifact,
    camera_lidar_failure_analysis_json_schema,
)
from calibrex.core.camera_lidar_initializer_calibration import (
    CameraLidarInitializerCalibrationArtifact,
    camera_lidar_initializer_calibration_json_schema,
)
from calibrex.core.camera_lidar_pose_initializer_benchmark import (
    CameraLidarPoseInitializerBenchmarkProtocol,
    camera_lidar_pose_initializer_protocol_json_schema,
)
from calibrex.core.camera_lidar_pose_initializer_failure_analysis import (
    CameraLidarPoseInitializerFailureAnalysis,
    camera_lidar_pose_initializer_failure_analysis_json_schema,
)
from calibrex.core.camera_lidar_provider_support_comparison import (
    CameraLidarProviderSupportComparisonArtifact,
    camera_lidar_provider_support_comparison_json_schema,
)
from calibrex.core.camera_lidar_sota_audit import (
    CameraLidarSotaAuditProtocol,
    CameraLidarSotaAuditResult,
    camera_lidar_sota_audit_protocol_json_schema,
    camera_lidar_sota_audit_result_json_schema,
)
from calibrex.core.capture_manifest import (
    CaptureManifest,
    CaptureManifestVerification,
    capture_manifest_json_schema,
    capture_manifest_verification_json_schema,
)
from calibrex.core.capture_readiness import (
    CaptureReadinessArtifact,
    capture_readiness_json_schema,
)
from calibrex.core.config import (
    CalibrationConfig,
    config_json_schema,
)
from calibrex.core.continuous_time_camera_lidar_artifacts import (
    ContinuousTimeCameraLidarProblemArtifact,
    ContinuousTimeCameraLidarResultArtifact,
    continuous_time_camera_lidar_problem_json_schema,
    continuous_time_camera_lidar_result_json_schema,
)
from calibrex.core.continuous_time_contract import (
    ContinuousTimeTrajectoryContract,
    continuous_time_trajectory_json_schema,
)
from calibrex.core.continuous_time_fit_artifacts import (
    ContinuousTimeTrajectoryFitResultArtifact,
    ContinuousTimeTrajectoryMeasurements,
    continuous_time_fit_json_schema,
    continuous_time_measurements_json_schema,
)
from calibrex.core.continuous_time_imu_accel_bias import (
    ContinuousTimeImuAccelBiasArtifact,
    continuous_time_imu_accel_bias_json_schema,
)
from calibrex.core.continuous_time_imu_clock_offset import (
    ContinuousTimeImuClockOffsetArtifact,
    continuous_time_imu_clock_offset_json_schema,
)
from calibrex.core.continuous_time_imu_intrinsics import (
    ContinuousTimeImuIntrinsicsArtifact,
    continuous_time_imu_intrinsics_json_schema,
)
from calibrex.core.continuous_time_imu_lever_arm import (
    ContinuousTimeImuLeverArmArtifact,
    continuous_time_imu_lever_arm_json_schema,
)
from calibrex.core.continuous_time_imu_preintegration import (
    ContinuousTimeImuPreintegrationArtifact,
    continuous_time_imu_preintegration_json_schema,
)
from calibrex.core.continuous_time_lidar_ablation import (
    ContinuousTimeLidarAblationManifest,
    continuous_time_lidar_ablation_json_schema,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    ContinuousTimeLidarPairArtifact,
    continuous_time_lidar_pair_json_schema,
)
from calibrex.core.continuous_time_lidar_point_to_plane import (
    ContinuousTimeLidarPointToPlaneArtifact,
    continuous_time_lidar_point_to_plane_json_schema,
)
from calibrex.core.continuous_time_lidar_train_diagnostics import (
    ContinuousTimeLidarTrainDiagnostics,
    continuous_time_lidar_train_diagnostics_json_schema,
)
from calibrex.core.continuous_time_sliding_window import (
    ContinuousTimeSlidingWindowArtifact,
    continuous_time_sliding_window_json_schema,
)
from calibrex.core.dynamic_window import (
    DynamicWindowConsistencyArtifact,
    dynamic_window_consistency_json_schema,
)
from calibrex.core.empirical_uncertainty import (
    EmpiricalSe3UncertaintyArtifact,
    empirical_se3_uncertainty_json_schema,
)
from calibrex.core.environment_readiness import (
    EnvironmentReadinessArtifact,
    environment_readiness_json_schema,
)
from calibrex.core.evidence_bundle import (
    EvidenceBundleManifest,
    EvidenceBundleVerification,
    evidence_bundle_json_schema,
    evidence_bundle_verification_json_schema,
)
from calibrex.core.evidence_contract import (
    PolicyArtifact,
    ProtocolArtifact,
    policy_json_schema,
    protocol_json_schema,
)
from calibrex.core.external_camera_imu_evidence import (
    CameraImuMotionRecording,
    ExternalCameraImuEvidenceArtifact,
    camera_imu_motion_recording_json_schema,
    external_camera_imu_evidence_json_schema,
)
from calibrex.core.external_run import (
    ExternalCalibrationRunArtifact,
    external_run_json_schema,
)
from calibrex.core.gnss_imu_lever_arm import (
    GnssImuLeverArmArtifact,
    gnss_imu_lever_arm_json_schema,
)
from calibrex.core.gnss_lidar_lever_arm import (
    GnssLidarLeverArmArtifact,
    gnss_lidar_lever_arm_json_schema,
)
from calibrex.core.imu_lidar_rotation import (
    ImuLidarRotationArtifact,
    imu_lidar_rotation_json_schema,
)
from calibrex.core.imu_lidar_translation import (
    ImuLidarTranslationArtifact,
    imu_lidar_translation_json_schema,
)
from calibrex.core.ins_lidar_hand_eye import (
    InsLidarHandEyeArtifact,
    ins_lidar_hand_eye_json_schema,
)
from calibrex.core.koide_handoff import (
    KoideExecutionLock,
    koide_execution_lock_json_schema,
)
from calibrex.core.koide_readiness import (
    KoideReadinessArtifact,
    koide_readiness_json_schema,
)
from calibrex.core.koide_real_pilot import (
    KoideRealPilotFinalization,
    KoideRealPilotRequest,
    KoideRealPilotVerification,
    koide_real_pilot_finalization_json_schema,
    koide_real_pilot_request_json_schema,
    koide_real_pilot_verification_json_schema,
)
from calibrex.core.koide_runner import (
    koide_runner_json_schema,
)
from calibrex.core.lidar_lidar_extrinsic import (
    LidarLidarExtrinsicArtifact,
    lidar_lidar_extrinsic_json_schema,
)
from calibrex.core.lidar_wheel_odometry import (
    LidarWheelOdometryArtifact,
    lidar_wheel_odometry_json_schema,
)
from calibrex.core.lifecycle_registry import (
    LifecycleEvaluationArtifact,
    LifecycleEvent,
    LifecycleRegistryState,
    LifecycleRegistryStatus,
    RegistryHead,
    RegistryManifest,
    RegistryVerificationArtifact,
    lifecycle_evaluation_json_schema,
    lifecycle_event_json_schema,
    lifecycle_head_json_schema,
    lifecycle_registry_json_schema,
    lifecycle_registry_state_json_schema,
    lifecycle_registry_verification_json_schema,
    lifecycle_status_json_schema,
)
from calibrex.core.livox_time_ablation import (
    LivoxTimeAblationManifest,
    livox_time_ablation_json_schema,
)
from calibrex.core.mcap_integrity import (
    McapIntegrityEvidence,
    mcap_integrity_json_schema,
)
from calibrex.core.multi_lidar_service import (
    MultiLidarServiceEvaluation,
    MultiLidarServicePlan,
    multi_lidar_service_evaluation_json_schema,
    multi_lidar_service_plan_json_schema,
)
from calibrex.core.online_timeline import (
    OnlineCalibrationTimelineArtifact,
    online_timeline_json_schema,
)
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticPnpResultArtifact,
    ProbabilisticRefinementResultArtifact,
    probabilistic_correspondence_json_schema,
    probabilistic_pnp_result_json_schema,
    probabilistic_refinement_result_json_schema,
)
from calibrex.core.radar_service import (
    RadarServiceEvaluation,
    RadarServicePlan,
    radar_service_evaluation_json_schema,
    radar_service_plan_json_schema,
)
from calibrex.core.raw_replay import (
    FieldReplacementPilot,
    ReplayComparison,
    ReplayDefinition,
    ReplayPlan,
    ReplayResult,
    ReplayStageArtifact,
    field_replacement_pilot_json_schema,
    replay_comparison_json_schema,
    replay_definition_json_schema,
    replay_plan_json_schema,
    replay_result_json_schema,
    replay_stage_json_schema,
)
from calibrex.core.report_artifacts import (
    ReportDegeneracyArtifact,
    ReportEvidenceArtifact,
    ReportMetricsArtifact,
    ReportObservabilityArtifact,
    ReportSummaryArtifact,
    report_artifact_json_schema,
)
from calibrex.core.result import (
    CalibrationResult,
    result_json_schema,
)
from calibrex.core.solid_state import (
    SolidStateLidarCalibrationContext,
    solid_state_context_json_schema,
)
from calibrex.core.solid_state_cross_dataset_benchmark import (
    SolidStateCrossDatasetBenchmarkManifest,
    SolidStateCrossDatasetBenchmarkSpec,
    solid_state_cross_dataset_benchmark_config_json_schema,
    solid_state_cross_dataset_benchmark_json_schema,
)
from calibrex.core.solid_state_failure_analysis import (
    SolidStateFailureAnalysisManifest,
    solid_state_failure_analysis_json_schema,
)
from calibrex.core.solid_state_metrology_evaluation import (
    SolidStateMetrologyEvaluationArtifact,
    solid_state_metrology_evaluation_json_schema,
)
from calibrex.core.solid_state_synthetic_benchmark import (
    SolidStateSyntheticBenchmarkArtifact,
    solid_state_synthetic_benchmark_json_schema,
)
from calibrex.core.sota_audit import (
    SotaAuditProtocol,
    SotaAuditResult,
    sota_audit_protocol_json_schema,
    sota_audit_result_json_schema,
)
from calibrex.core.sota_leaderboard import SotaLeaderboard, sota_leaderboard_json_schema
from calibrex.core.trajectory import (
    TrajectoryArtifact,
    trajectory_json_schema,
)
from calibrex.core.trajectory_window_drift import (
    TrajectoryWindowDriftArtifact,
    trajectory_window_drift_json_schema,
)
from calibrex.core.transform_artifacts import (
    TransformArtifact,
    transform_artifact_json_schema,
)
from calibrex.core.vehicle_frame_rotation import (
    VehicleFrameRotationArtifact,
    vehicle_frame_rotation_json_schema,
)
from calibrex.data.depth import (
    DepthProviderArtifact,
    depth_provider_json_schema,
)
from calibrex.data.kitti360_lidar_window_integration import (
    Kitti360LidarWindowIntegrationArtifact,
    kitti360_lidar_window_integration_json_schema,
)
from calibrex.data.kitti_benchmark import (
    KITTIBenchmarkInputManifest,
    kitti_benchmark_input_json_schema,
)
from calibrex.data.kitti_raw_lidar_window_integration import (
    KittiRawLidarWindowIntegrationArtifact,
    kitti_raw_lidar_window_integration_json_schema,
)
from calibrex.data.manifest import (
    DatasetManifest,
    manifest_json_schema,
)
from calibrex.data.remote_archive_selection import (
    RemoteArchiveSelectionArtifact,
    remote_archive_selection_json_schema,
)
from calibrex.diagnostics import (
    doctor_json_schema,
)
from calibrex.evaluation.compare import (
    ResultComparison,
    comparison_json_schema,
)
from calibrex.evaluation.kitti_falsification_benchmark import (
    KITTIFalsificationBenchmarkArtifact,
    kitti_falsification_json_schema,
)
from calibrex.evaluation.koide_pilot import (
    KoidePilotArtifact,
    koide_pilot_json_schema,
)
from calibrex.evaluation.report_compare import (
    ReportComparison,
    report_comparison_json_schema,
)
from calibrex.export.autoware import (
    AutowareExportArtifact,
    autoware_export_json_schema,
)
from calibrex.export.autoware_promotion import (
    AutowarePromotionArtifact,
    autoware_promotion_json_schema,
)
from calibrex.export.autoware_smoke import (
    AutowareSmokeArtifact,
    autoware_smoke_json_schema,
)

SCHEMA_LEDGER_VERSION: Final = "slac.schema_ledger/v0.1"
SCHEMA_LEDGER_FILENAME: Final = "schema_ledger.json"

SchemaStatus = Literal["active", "alias", "schema-only"]
DigestMethod = Literal["verify_artifact_digest", "verify_lock_digest", "verify_request_digest"]
SchemaGenerator = Callable[[], Mapping[str, Any]]
SchemaUpgrade = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class SchemaEntry:
    """One artifact kind, its model, and its JSON Schema generator."""

    kind: str
    model: type[BaseModel] | None
    json_schema: SchemaGenerator
    filename: str
    alias_of: str | None = None
    digest_method: DigestMethod | None = None

    @property
    def status(self) -> SchemaStatus:
        """Return whether this kind is canonical, an alias, or schema-only."""

        if self.alias_of is not None:
            return "alias"
        if self.model is None:
            return "schema-only"
        return "active"

    @property
    def accepted_versions(self) -> tuple[str, ...]:
        """Return every ``schema_version`` the model reads, in annotation order."""

        if self.model is None:
            return ()
        field = self.model.model_fields.get("schema_version")
        if field is None:
            return ()
        return tuple(str(value) for value in typing.get_args(field.annotation))

    @property
    def current_version(self) -> str | None:
        """Return the ``schema_version`` that newly written artifacts carry."""

        if self.model is None:
            return None
        field = self.model.model_fields.get("schema_version")
        default = None if field is None else field.default
        return default if isinstance(default, str) else None


@dataclass(frozen=True)
class RetiredSchemaVersion:
    """A released schema version that current Calibrex deliberately cannot read."""

    schema_version: str
    kind: str
    retired_in: str
    reason: str
    remedy: str


@dataclass(frozen=True)
class SchemaMigration:
    """A lossless upgrade of one payload from ``from_version`` to ``to_version``.

    ``upgrade`` must be pure and deterministic, must not invent values that the
    source artifact did not record, and must return a payload whose
    ``schema_version`` equals ``to_version``.  Digest-bound kinds cannot be
    migrated because their self-digest covers the original bytes.
    """

    kind: str
    from_version: str
    to_version: str
    upgrade: SchemaUpgrade
    description: str


def _entry(
    kind: str,
    model: type[BaseModel] | None,
    json_schema: SchemaGenerator,
    *,
    filename: str | None = None,
    alias_of: str | None = None,
    digest_method: DigestMethod | None = None,
) -> SchemaEntry:
    return SchemaEntry(
        kind=kind,
        model=model,
        json_schema=json_schema,
        filename=filename or f"{kind.replace('-', '_')}.schema.json",
        alias_of=alias_of,
        digest_method=digest_method,
    )


SCHEMA_REGISTRY: Final[tuple[SchemaEntry, ...]] = (
    _entry("config", CalibrationConfig, config_json_schema),
    _entry("result", CalibrationResult, result_json_schema),
    _entry("comparison", ResultComparison, comparison_json_schema),
    _entry("report-comparison", ReportComparison, report_comparison_json_schema),
    _entry(
        "dynamic-window-consistency",
        DynamicWindowConsistencyArtifact,
        dynamic_window_consistency_json_schema,
    ),
    _entry(
        "trajectory-window-drift",
        TrajectoryWindowDriftArtifact,
        trajectory_window_drift_json_schema,
    ),
    _entry("capture-readiness", CaptureReadinessArtifact, capture_readiness_json_schema),
    _entry(
        "capture-manifest",
        CaptureManifest,
        capture_manifest_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "capture-manifest-verification",
        CaptureManifestVerification,
        capture_manifest_verification_json_schema,
    ),
    _entry("mcap-integrity", McapIntegrityEvidence, mcap_integrity_json_schema),
    _entry(
        "koide-readiness",
        KoideReadinessArtifact,
        koide_readiness_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "koide-execution-lock",
        KoideExecutionLock,
        koide_execution_lock_json_schema,
        digest_method="verify_lock_digest",
    ),
    _entry(
        "continuous-time-lidar-pair",
        ContinuousTimeLidarPairArtifact,
        continuous_time_lidar_pair_json_schema,
        filename="continuous_time_lidar_pair_result.schema.json",
    ),
    _entry(
        "continuous-time-lidar-point-to-plane",
        ContinuousTimeLidarPointToPlaneArtifact,
        continuous_time_lidar_point_to_plane_json_schema,
    ),
    _entry(
        "continuous-time-imu-preintegration",
        ContinuousTimeImuPreintegrationArtifact,
        continuous_time_imu_preintegration_json_schema,
    ),
    _entry(
        "continuous-time-imu-lever-arm",
        ContinuousTimeImuLeverArmArtifact,
        continuous_time_imu_lever_arm_json_schema,
    ),
    _entry(
        "continuous-time-imu-clock-offset",
        ContinuousTimeImuClockOffsetArtifact,
        continuous_time_imu_clock_offset_json_schema,
    ),
    _entry(
        "continuous-time-imu-accel-bias",
        ContinuousTimeImuAccelBiasArtifact,
        continuous_time_imu_accel_bias_json_schema,
    ),
    _entry(
        "continuous-time-imu-intrinsics",
        ContinuousTimeImuIntrinsicsArtifact,
        continuous_time_imu_intrinsics_json_schema,
    ),
    _entry(
        "calibration-lifecycle", CalibrationLifecycleArtifact, calibration_lifecycle_json_schema
    ),
    _entry("lifecycle-registry", RegistryManifest, lifecycle_registry_json_schema),
    _entry("lifecycle-event", LifecycleEvent, lifecycle_event_json_schema),
    _entry("lifecycle-evaluation", LifecycleEvaluationArtifact, lifecycle_evaluation_json_schema),
    _entry(
        "lifecycle-registry-state", LifecycleRegistryState, lifecycle_registry_state_json_schema
    ),
    _entry("lifecycle-head", RegistryHead, lifecycle_head_json_schema),
    _entry(
        "lifecycle-verification",
        RegistryVerificationArtifact,
        lifecycle_registry_verification_json_schema,
    ),
    _entry("lifecycle-status", LifecycleRegistryStatus, lifecycle_status_json_schema),
    _entry(
        "continuous-time-sliding-window",
        ContinuousTimeSlidingWindowArtifact,
        continuous_time_sliding_window_json_schema,
    ),
    _entry(
        "continuous-time-lidar-train-diagnostics",
        ContinuousTimeLidarTrainDiagnostics,
        continuous_time_lidar_train_diagnostics_json_schema,
    ),
    _entry(
        "continuous-time-lidar-ablation",
        ContinuousTimeLidarAblationManifest,
        continuous_time_lidar_ablation_json_schema,
    ),
    _entry(
        "solid-state-cross-dataset-benchmark-config",
        SolidStateCrossDatasetBenchmarkSpec,
        solid_state_cross_dataset_benchmark_config_json_schema,
    ),
    _entry(
        "solid-state-cross-dataset-benchmark",
        SolidStateCrossDatasetBenchmarkManifest,
        solid_state_cross_dataset_benchmark_json_schema,
    ),
    _entry(
        "solid-state-failure-analysis",
        SolidStateFailureAnalysisManifest,
        solid_state_failure_analysis_json_schema,
    ),
    _entry(
        "solid-state-synthetic-benchmark",
        SolidStateSyntheticBenchmarkArtifact,
        solid_state_synthetic_benchmark_json_schema,
    ),
    _entry(
        "solid-state-metrology-evaluation",
        SolidStateMetrologyEvaluationArtifact,
        solid_state_metrology_evaluation_json_schema,
    ),
    _entry("assessment", AssessmentArtifact, assessment_json_schema),
    _entry("benchmark", BenchmarkArtifact, benchmark_json_schema),
    _entry("benchmark-definition", BenchmarkDefinition, benchmark_definition_json_schema),
    _entry("policy", PolicyArtifact, policy_json_schema),
    _entry("protocol", ProtocolArtifact, protocol_json_schema),
    _entry("transforms", TransformArtifact, transform_artifact_json_schema),
    _entry("dataset-manifest", DatasetManifest, manifest_json_schema),
    _entry(
        "remote-archive-selection",
        RemoteArchiveSelectionArtifact,
        remote_archive_selection_json_schema,
    ),
    _entry(
        "kitti360-lidar-window-integration",
        Kitti360LidarWindowIntegrationArtifact,
        kitti360_lidar_window_integration_json_schema,
    ),
    _entry(
        "kitti-raw-lidar-window-integration",
        KittiRawLidarWindowIntegrationArtifact,
        kitti_raw_lidar_window_integration_json_schema,
    ),
    _entry(
        "doctor", EnvironmentReadinessArtifact, doctor_json_schema, alias_of="environment-readiness"
    ),
    _entry(
        "environment-readiness", EnvironmentReadinessArtifact, environment_readiness_json_schema
    ),
    _entry("calibration-ci", CalibrationCIArtifact, calibration_ci_json_schema),
    _entry("external-run", ExternalCalibrationRunArtifact, external_run_json_schema),
    _entry(
        "camera-imu-motion-recording",
        CameraImuMotionRecording,
        camera_imu_motion_recording_json_schema,
    ),
    _entry(
        "external-camera-imu-evidence",
        ExternalCameraImuEvidenceArtifact,
        external_camera_imu_evidence_json_schema,
    ),
    _entry(
        "kitti-falsification", KITTIFalsificationBenchmarkArtifact, kitti_falsification_json_schema
    ),
    _entry("kitti-benchmark-input", KITTIBenchmarkInputManifest, kitti_benchmark_input_json_schema),
    _entry(
        "koide-pilot",
        KoidePilotArtifact,
        koide_pilot_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "koide-runner", None, koide_runner_json_schema, filename="koide_runner_config.schema.json"
    ),
    _entry(
        "koide-real-pilot-request",
        KoideRealPilotRequest,
        koide_real_pilot_request_json_schema,
        digest_method="verify_request_digest",
    ),
    _entry(
        "koide-real-pilot-verification",
        KoideRealPilotVerification,
        koide_real_pilot_verification_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "koide-real-pilot-finalization",
        KoideRealPilotFinalization,
        koide_real_pilot_finalization_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry("depth-provider", DepthProviderArtifact, depth_provider_json_schema),
    _entry(
        "continuous-time-camera-lidar-problem",
        ContinuousTimeCameraLidarProblemArtifact,
        continuous_time_camera_lidar_problem_json_schema,
    ),
    _entry(
        "continuous-time-camera-lidar-result",
        ContinuousTimeCameraLidarResultArtifact,
        continuous_time_camera_lidar_result_json_schema,
    ),
    _entry(
        "continuous-time-trajectory",
        ContinuousTimeTrajectoryContract,
        continuous_time_trajectory_json_schema,
    ),
    _entry(
        "continuous-time-trajectory-measurements",
        ContinuousTimeTrajectoryMeasurements,
        continuous_time_measurements_json_schema,
    ),
    _entry(
        "continuous-time-trajectory-fit",
        ContinuousTimeTrajectoryFitResultArtifact,
        continuous_time_fit_json_schema,
    ),
    _entry(
        "probabilistic-correspondence",
        ProbabilisticCorrespondenceArtifact,
        probabilistic_correspondence_json_schema,
    ),
    _entry(
        "probabilistic-pnp-result",
        ProbabilisticPnpResultArtifact,
        probabilistic_pnp_result_json_schema,
    ),
    _entry(
        "probabilistic-refinement-result",
        ProbabilisticRefinementResultArtifact,
        probabilistic_refinement_result_json_schema,
    ),
    _entry(
        "empirical-se3-uncertainty",
        EmpiricalSe3UncertaintyArtifact,
        empirical_se3_uncertainty_json_schema,
    ),
    _entry("camera-lidar-problem", CameraLidarCalibrationProblem, camera_lidar_problem_json_schema),
    _entry(
        "camera-lidar-correspondence-export",
        CameraLidarCorrespondenceExportManifest,
        camera_lidar_correspondence_export_json_schema,
    ),
    _entry(
        "camera-lidar-confidence-calibration",
        CameraLidarConfidenceCalibrationArtifact,
        camera_lidar_confidence_calibration_json_schema,
    ),
    _entry(
        "camera-lidar-provider-support-comparison",
        CameraLidarProviderSupportComparisonArtifact,
        camera_lidar_provider_support_comparison_json_schema,
    ),
    _entry(
        "camera-lidar-pose-initializer-protocol",
        CameraLidarPoseInitializerBenchmarkProtocol,
        camera_lidar_pose_initializer_protocol_json_schema,
    ),
    _entry(
        "camera-lidar-pose-initializer-failure-analysis",
        CameraLidarPoseInitializerFailureAnalysis,
        camera_lidar_pose_initializer_failure_analysis_json_schema,
    ),
    _entry(
        "camera-lidar-initializer-calibration",
        CameraLidarInitializerCalibrationArtifact,
        camera_lidar_initializer_calibration_json_schema,
    ),
    _entry(
        "camera-lidar-correspondence-quality",
        CameraLidarCorrespondenceQualityArtifact,
        camera_lidar_correspondence_quality_json_schema,
    ),
    _entry(
        "camera-lidar-failure-analysis",
        CameraLidarFailureAnalysisArtifact,
        camera_lidar_failure_analysis_json_schema,
    ),
    _entry(
        "camera-lidar-sota-audit-protocol",
        CameraLidarSotaAuditProtocol,
        camera_lidar_sota_audit_protocol_json_schema,
    ),
    _entry(
        "camera-lidar-sota-audit-result",
        CameraLidarSotaAuditResult,
        camera_lidar_sota_audit_result_json_schema,
    ),
    _entry("gnss-lidar-lever-arm", GnssLidarLeverArmArtifact, gnss_lidar_lever_arm_json_schema),
    _entry("imu-lidar-rotation", ImuLidarRotationArtifact, imu_lidar_rotation_json_schema),
    _entry("imu-lidar-translation", ImuLidarTranslationArtifact, imu_lidar_translation_json_schema),
    _entry("ins-lidar-hand-eye", InsLidarHandEyeArtifact, ins_lidar_hand_eye_json_schema),
    _entry("lidar-lidar-extrinsic", LidarLidarExtrinsicArtifact, lidar_lidar_extrinsic_json_schema),
    _entry("gnss-imu-lever-arm", GnssImuLeverArmArtifact, gnss_imu_lever_arm_json_schema),
    _entry(
        "lidar-wheel-odometry", LidarWheelOdometryArtifact, lidar_wheel_odometry_json_schema
    ),
    _entry(
        "vehicle-frame-rotation", VehicleFrameRotationArtifact, vehicle_frame_rotation_json_schema
    ),
    _entry("sota-audit-protocol", SotaAuditProtocol, sota_audit_protocol_json_schema),
    _entry("sota-audit-result", SotaAuditResult, sota_audit_result_json_schema),
    _entry("sota-leaderboard", SotaLeaderboard, sota_leaderboard_json_schema),
    _entry(
        "camera-lidar-benchmark-protocol",
        CameraLidarBenchmarkProtocol,
        camera_lidar_benchmark_protocol_json_schema,
    ),
    _entry(
        "calibration-candidate-trace",
        CalibrationCandidateTrace,
        calibration_candidate_trace_json_schema,
    ),
    _entry("bullseye-plot", BullseyePlotArtifact, bullseye_plot_json_schema),
    _entry("evidence-bundle", EvidenceBundleManifest, evidence_bundle_json_schema),
    _entry(
        "evidence-bundle-verification",
        EvidenceBundleVerification,
        evidence_bundle_verification_json_schema,
    ),
    _entry("online-timeline", OnlineCalibrationTimelineArtifact, online_timeline_json_schema),
    _entry(
        "solid-state-context", SolidStateLidarCalibrationContext, solid_state_context_json_schema
    ),
    _entry("livox-time-ablation", LivoxTimeAblationManifest, livox_time_ablation_json_schema),
    _entry("autoware-export", AutowareExportArtifact, autoware_export_json_schema),
    _entry("autoware-promotion", AutowarePromotionArtifact, autoware_promotion_json_schema),
    _entry("autoware-smoke", AutowareSmokeArtifact, autoware_smoke_json_schema),
    _entry(
        "raw-replay-definition",
        ReplayDefinition,
        replay_definition_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "raw-replay-plan",
        ReplayPlan,
        replay_plan_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "raw-replay-stage",
        ReplayStageArtifact,
        replay_stage_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "raw-replay-result",
        ReplayResult,
        replay_result_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "raw-replay-comparison",
        ReplayComparison,
        replay_comparison_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "field-replacement-pilot",
        FieldReplacementPilot,
        field_replacement_pilot_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "multi-lidar-service-plan",
        MultiLidarServicePlan,
        multi_lidar_service_plan_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "multi-lidar-service-evaluation",
        MultiLidarServiceEvaluation,
        multi_lidar_service_evaluation_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "camera-imu-service-plan",
        CameraImuServicePlan,
        camera_imu_service_plan_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "camera-imu-service-evaluation",
        CameraImuServiceEvaluation,
        camera_imu_service_evaluation_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "radar-service-plan",
        RadarServicePlan,
        radar_service_plan_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry(
        "radar-service-evaluation",
        RadarServiceEvaluation,
        radar_service_evaluation_json_schema,
        digest_method="verify_artifact_digest",
    ),
    _entry("trajectory", TrajectoryArtifact, trajectory_json_schema),
    _entry(
        "report-summary",
        ReportSummaryArtifact,
        partial(report_artifact_json_schema, "report-summary"),
    ),
    _entry(
        "report-metrics",
        ReportMetricsArtifact,
        partial(report_artifact_json_schema, "report-metrics"),
    ),
    _entry(
        "report-observability",
        ReportObservabilityArtifact,
        partial(report_artifact_json_schema, "report-observability"),
    ),
    _entry(
        "report-degeneracy",
        ReportDegeneracyArtifact,
        partial(report_artifact_json_schema, "report-degeneracy"),
    ),
    _entry(
        "report-evidence",
        ReportEvidenceArtifact,
        partial(report_artifact_json_schema, "report-evidence"),
    ),
)

RETIRED_SCHEMA_VERSIONS: Final[tuple[RetiredSchemaVersion, ...]] = (
    RetiredSchemaVersion(
        schema_version="slac.doctor/v0.1",
        kind="environment-readiness",
        retired_in="0.5.0",
        reason=(
            "doctor artifacts became slac.environment_readiness/v0.1, whose workflow "
            "suggestions require a template_path that v0.1 doctor output never "
            "recorded; an upgrade would have to invent it"
        ),
        remedy="re-run `calibrex doctor --output <path>` to regenerate the artifact",
    ),
)

SCHEMA_MIGRATIONS: Final[tuple[SchemaMigration, ...]] = ()

_ENTRY_BY_KIND: Final[dict[str, SchemaEntry]] = {entry.kind: entry for entry in SCHEMA_REGISTRY}


class SchemaRegistryError(ValueError):
    """Raised when an artifact kind or schema version cannot be resolved."""


def schema_entries() -> tuple[SchemaEntry, ...]:
    """Return every registered schema entry in ``schema all`` order."""

    return SCHEMA_REGISTRY


def schema_entry(kind: str) -> SchemaEntry:
    """Return the entry for ``kind`` or raise ``SchemaRegistryError``."""

    try:
        return _ENTRY_BY_KIND[kind]
    except KeyError as exc:
        supported = ", ".join(sorted(_ENTRY_BY_KIND))
        msg = f"unsupported artifact kind {kind!r}; expected one of {supported}"
        raise SchemaRegistryError(msg) from exc


def validatable_kinds() -> tuple[str, ...]:
    """Return kinds that ``calibrex validate`` can read, in registry order."""

    return tuple(entry.kind for entry in SCHEMA_REGISTRY if entry.model is not None)


def retired_schema_version(schema_version: str) -> RetiredSchemaVersion | None:
    """Return retirement details when ``schema_version`` was deliberately retired."""

    for retired in RETIRED_SCHEMA_VERSIONS:
        if retired.schema_version == schema_version:
            return retired
    return None


def kind_for_schema_version(
    schema_version: str,
    migrations: Sequence[SchemaMigration] = SCHEMA_MIGRATIONS,
) -> str:
    """Resolve the canonical kind that reads or upgrades ``schema_version``."""

    for entry in SCHEMA_REGISTRY:
        if entry.status == "active" and schema_version in entry.accepted_versions:
            return entry.kind
    for migration in migrations:
        if migration.from_version == schema_version:
            return migration.kind
    retired = retired_schema_version(schema_version)
    if retired is not None:
        msg = (
            f"schema_version {schema_version!r} was retired in Calibrex {retired.retired_in}: "
            f"{retired.reason}; {retired.remedy}"
        )
        raise SchemaRegistryError(msg)
    msg = (
        f"unsupported schema_version {schema_version!r}; run `calibrex schema ledger` "
        "to list every readable version"
    )
    raise SchemaRegistryError(msg)


def upgrade_payload(
    kind: str,
    payload: dict[str, Any],
    migrations: Sequence[SchemaMigration] = SCHEMA_MIGRATIONS,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Upgrade ``payload`` until its ``schema_version`` is read by ``kind``'s model.

    Returns the (possibly unchanged) payload and the chain of source versions
    that were upgraded.  The input mapping is never mutated.
    """

    entry = schema_entry(kind)
    accepted = entry.accepted_versions
    upgraded: tuple[str, ...] = ()
    current = payload
    for _ in range(len(migrations) + 1):
        version = current.get("schema_version")
        if not isinstance(version, str) or version in accepted:
            return current, upgraded
        step = next(
            (
                migration
                for migration in migrations
                if migration.kind == kind and migration.from_version == version
            ),
            None,
        )
        if step is None:
            return current, upgraded
        if entry.digest_method is not None:
            msg = f"digest-bound kind {kind!r} cannot be migrated from {version!r}"
            raise SchemaRegistryError(msg)
        current = step.upgrade(dict(current))
        if current.get("schema_version") != step.to_version:
            msg = (
                f"migration {step.from_version} -> {step.to_version} for {kind!r} "
                f"produced schema_version {current.get('schema_version')!r}"
            )
            raise SchemaRegistryError(msg)
        upgraded = (*upgraded, version)
    msg = f"schema migrations for {kind!r} do not terminate"
    raise SchemaRegistryError(msg)


def schema_digest(schema: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a schema's canonical JSON serialization."""

    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def schema_ledger(
    migrations: Sequence[SchemaMigration] = SCHEMA_MIGRATIONS,
) -> dict[str, Any]:
    """Return the machine-readable ledger of kinds, versions, and schema digests."""

    kinds: dict[str, dict[str, Any]] = {}
    for entry in SCHEMA_REGISTRY:
        record: dict[str, Any] = {
            "status": entry.status,
            "schema_file": entry.filename,
            "schema_sha256": schema_digest(entry.json_schema()),
            "current_version": entry.current_version,
            "accepted_versions": list(entry.accepted_versions),
            "digest_bound": entry.digest_method is not None,
        }
        if entry.alias_of is not None:
            record["alias_of"] = entry.alias_of
        kinds[entry.kind] = record
    return {
        "schema_version": SCHEMA_LEDGER_VERSION,
        "kinds": kinds,
        "retired_versions": [
            {
                "schema_version": retired.schema_version,
                "kind": retired.kind,
                "retired_in": retired.retired_in,
                "reason": retired.reason,
                "remedy": retired.remedy,
            }
            for retired in RETIRED_SCHEMA_VERSIONS
        ],
        "migrations": [
            {
                "kind": migration.kind,
                "from_version": migration.from_version,
                "to_version": migration.to_version,
                "description": migration.description,
            }
            for migration in migrations
        ],
    }
