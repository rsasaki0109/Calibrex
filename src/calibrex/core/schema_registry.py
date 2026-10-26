"""Single source of truth for schema-versioned Calibrex artifact kinds.

Every artifact kind that ``calibrex validate`` can read and every JSON Schema
that ``calibrex schema all`` writes has one lightweight declaration. Models
and generators load on demand; ``SCHEMA_REGISTRY`` still exposes the complete
tuple of entries. Accepted schema versions are derived from each model's
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
from importlib import import_module
from threading import RLock
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel

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


if TYPE_CHECKING:
    SCHEMA_REGISTRY: Final[tuple[SchemaEntry, ...]] = ()


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


@dataclass(frozen=True)
class _EntrySpec:
    kind: str
    model_ref: str | None
    schema_ref: str
    filename: str | None = None
    alias_of: str | None = None
    digest_method: DigestMethod | None = None
    schema_args: tuple[str, ...] = ()

    def load(self) -> SchemaEntry:
        model = None if self.model_ref is None else _load_symbol(self.model_ref)
        generator = _load_symbol(self.schema_ref)
        if self.schema_args:
            generator = partial(generator, *self.schema_args)
        return SchemaEntry(
            kind=self.kind,
            model=model,
            json_schema=generator,
            filename=self.filename or f"{self.kind.replace('-', '_')}.schema.json",
            alias_of=self.alias_of,
            digest_method=self.digest_method,
        )


def _load_symbol(reference: str) -> Any:
    module, name = reference.split(":")
    return getattr(import_module(module), name)


_ENTRY_SPECS: Final = (
    _EntrySpec(
        "config",
        "calibrex.core.config:CalibrationConfig",
        "calibrex.core.config:config_json_schema",
    ),
    _EntrySpec(
        "result",
        "calibrex.core.result:CalibrationResult",
        "calibrex.core.result:result_json_schema",
    ),
    _EntrySpec(
        "comparison",
        "calibrex.evaluation.compare:ResultComparison",
        "calibrex.evaluation.compare:comparison_json_schema",
    ),
    _EntrySpec(
        "report-comparison",
        "calibrex.evaluation.report_compare:ReportComparison",
        "calibrex.evaluation.report_compare:report_comparison_json_schema",
    ),
    _EntrySpec(
        "dynamic-window-consistency",
        "calibrex.core.dynamic_window:DynamicWindowConsistencyArtifact",
        "calibrex.core.dynamic_window:dynamic_window_consistency_json_schema",
    ),
    _EntrySpec(
        "trajectory-window-drift",
        "calibrex.core.trajectory_window_drift:TrajectoryWindowDriftArtifact",
        "calibrex.core.trajectory_window_drift:trajectory_window_drift_json_schema",
    ),
    _EntrySpec(
        "capture-readiness",
        "calibrex.core.capture_readiness:CaptureReadinessArtifact",
        "calibrex.core.capture_readiness:capture_readiness_json_schema",
    ),
    _EntrySpec(
        "capture-manifest",
        "calibrex.core.capture_manifest:CaptureManifest",
        "calibrex.core.capture_manifest:capture_manifest_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "capture-manifest-verification",
        "calibrex.core.capture_manifest:CaptureManifestVerification",
        "calibrex.core.capture_manifest:capture_manifest_verification_json_schema",
    ),
    _EntrySpec(
        "mcap-integrity",
        "calibrex.core.mcap_integrity:McapIntegrityEvidence",
        "calibrex.core.mcap_integrity:mcap_integrity_json_schema",
    ),
    _EntrySpec(
        "koide-readiness",
        "calibrex.core.koide_readiness:KoideReadinessArtifact",
        "calibrex.core.koide_readiness:koide_readiness_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "koide-execution-lock",
        "calibrex.core.koide_handoff:KoideExecutionLock",
        "calibrex.core.koide_handoff:koide_execution_lock_json_schema",
        digest_method="verify_lock_digest",
    ),
    _EntrySpec(
        "continuous-time-lidar-pair",
        "calibrex.core.continuous_time_lidar_artifacts:ContinuousTimeLidarPairArtifact",
        "calibrex.core.continuous_time_lidar_artifacts:continuous_time_lidar_pair_json_schema",
        filename="continuous_time_lidar_pair_result.schema.json",
    ),
    _EntrySpec(
        "continuous-time-lidar-point-to-plane",
        "calibrex.core.continuous_time_lidar_point_to_plane:ContinuousTimeLidarPointToPlaneArtifact",
        "calibrex.core.continuous_time_lidar_point_to_plane:continuous_time_lidar_point_to_plane_json_schema",
    ),
    _EntrySpec(
        "continuous-time-imu-preintegration",
        "calibrex.core.continuous_time_imu_preintegration:ContinuousTimeImuPreintegrationArtifact",
        "calibrex.core.continuous_time_imu_preintegration:continuous_time_imu_preintegration_json_schema",
    ),
    _EntrySpec(
        "continuous-time-imu-lever-arm",
        "calibrex.core.continuous_time_imu_lever_arm:ContinuousTimeImuLeverArmArtifact",
        "calibrex.core.continuous_time_imu_lever_arm:continuous_time_imu_lever_arm_json_schema",
    ),
    _EntrySpec(
        "continuous-time-imu-clock-offset",
        "calibrex.core.continuous_time_imu_clock_offset:ContinuousTimeImuClockOffsetArtifact",
        "calibrex.core.continuous_time_imu_clock_offset:continuous_time_imu_clock_offset_json_schema",
    ),
    _EntrySpec(
        "continuous-time-imu-accel-bias",
        "calibrex.core.continuous_time_imu_accel_bias:ContinuousTimeImuAccelBiasArtifact",
        "calibrex.core.continuous_time_imu_accel_bias:continuous_time_imu_accel_bias_json_schema",
    ),
    _EntrySpec(
        "continuous-time-imu-intrinsics",
        "calibrex.core.continuous_time_imu_intrinsics:ContinuousTimeImuIntrinsicsArtifact",
        "calibrex.core.continuous_time_imu_intrinsics:continuous_time_imu_intrinsics_json_schema",
    ),
    _EntrySpec(
        "continuous-time-lidar-imu-joint",
        "calibrex.core.continuous_time_lidar_imu_joint:ContinuousTimeLidarImuJointArtifact",
        "calibrex.core.continuous_time_lidar_imu_joint:continuous_time_lidar_imu_joint_json_schema",
    ),
    _EntrySpec(
        "uncertainty-coverage",
        "calibrex.evaluation.uncertainty_coverage:UncertaintyCoverageReport",
        "calibrex.evaluation.uncertainty_coverage:uncertainty_coverage_json_schema",
    ),
    _EntrySpec(
        "lidar-imu-bag-calibration",
        "calibrex.core.lidar_imu_bag_calibration:LidarImuBagCalibrationArtifact",
        "calibrex.core.lidar_imu_bag_calibration:lidar_imu_bag_calibration_json_schema",
    ),
    _EntrySpec(
        "calibration-lifecycle",
        "calibrex.core.calibration_lifecycle:CalibrationLifecycleArtifact",
        "calibrex.core.calibration_lifecycle:calibration_lifecycle_json_schema",
    ),
    _EntrySpec(
        "lifecycle-registry",
        "calibrex.core.lifecycle_registry:RegistryManifest",
        "calibrex.core.lifecycle_registry:lifecycle_registry_json_schema",
    ),
    _EntrySpec(
        "lifecycle-event",
        "calibrex.core.lifecycle_registry:LifecycleEvent",
        "calibrex.core.lifecycle_registry:lifecycle_event_json_schema",
    ),
    _EntrySpec(
        "lifecycle-evaluation",
        "calibrex.core.lifecycle_registry:LifecycleEvaluationArtifact",
        "calibrex.core.lifecycle_registry:lifecycle_evaluation_json_schema",
    ),
    _EntrySpec(
        "lifecycle-registry-state",
        "calibrex.core.lifecycle_registry:LifecycleRegistryState",
        "calibrex.core.lifecycle_registry:lifecycle_registry_state_json_schema",
    ),
    _EntrySpec(
        "lifecycle-head",
        "calibrex.core.lifecycle_registry:RegistryHead",
        "calibrex.core.lifecycle_registry:lifecycle_head_json_schema",
    ),
    _EntrySpec(
        "lifecycle-verification",
        "calibrex.core.lifecycle_registry:RegistryVerificationArtifact",
        "calibrex.core.lifecycle_registry:lifecycle_registry_verification_json_schema",
    ),
    _EntrySpec(
        "lifecycle-status",
        "calibrex.core.lifecycle_registry:LifecycleRegistryStatus",
        "calibrex.core.lifecycle_registry:lifecycle_status_json_schema",
    ),
    _EntrySpec(
        "continuous-time-sliding-window",
        "calibrex.core.continuous_time_sliding_window:ContinuousTimeSlidingWindowArtifact",
        "calibrex.core.continuous_time_sliding_window:continuous_time_sliding_window_json_schema",
    ),
    _EntrySpec(
        "continuous-time-lidar-train-diagnostics",
        "calibrex.core.continuous_time_lidar_train_diagnostics:ContinuousTimeLidarTrainDiagnostics",
        "calibrex.core.continuous_time_lidar_train_diagnostics:continuous_time_lidar_train_diagnostics_json_schema",
    ),
    _EntrySpec(
        "continuous-time-lidar-ablation",
        "calibrex.core.continuous_time_lidar_ablation:ContinuousTimeLidarAblationManifest",
        "calibrex.core.continuous_time_lidar_ablation:continuous_time_lidar_ablation_json_schema",
    ),
    _EntrySpec(
        "solid-state-cross-dataset-benchmark-config",
        "calibrex.core.solid_state_cross_dataset_benchmark:SolidStateCrossDatasetBenchmarkSpec",
        "calibrex.core.solid_state_cross_dataset_benchmark:solid_state_cross_dataset_benchmark_config_json_schema",
    ),
    _EntrySpec(
        "solid-state-cross-dataset-benchmark",
        "calibrex.core.solid_state_cross_dataset_benchmark:SolidStateCrossDatasetBenchmarkManifest",
        "calibrex.core.solid_state_cross_dataset_benchmark:solid_state_cross_dataset_benchmark_json_schema",
    ),
    _EntrySpec(
        "solid-state-failure-analysis",
        "calibrex.core.solid_state_failure_analysis:SolidStateFailureAnalysisManifest",
        "calibrex.core.solid_state_failure_analysis:solid_state_failure_analysis_json_schema",
    ),
    _EntrySpec(
        "solid-state-synthetic-benchmark",
        "calibrex.core.solid_state_synthetic_benchmark:SolidStateSyntheticBenchmarkArtifact",
        "calibrex.core.solid_state_synthetic_benchmark:solid_state_synthetic_benchmark_json_schema",
    ),
    _EntrySpec(
        "solid-state-metrology-evaluation",
        "calibrex.core.solid_state_metrology_evaluation:SolidStateMetrologyEvaluationArtifact",
        "calibrex.core.solid_state_metrology_evaluation:solid_state_metrology_evaluation_json_schema",
    ),
    _EntrySpec(
        "assessment",
        "calibrex.core.assessment:AssessmentArtifact",
        "calibrex.core.assessment:assessment_json_schema",
    ),
    _EntrySpec(
        "benchmark",
        "calibrex.core.benchmark:BenchmarkArtifact",
        "calibrex.core.benchmark:benchmark_json_schema",
    ),
    _EntrySpec(
        "benchmark-definition",
        "calibrex.core.benchmark:BenchmarkDefinition",
        "calibrex.core.benchmark:benchmark_definition_json_schema",
    ),
    _EntrySpec(
        "policy",
        "calibrex.core.evidence_contract:PolicyArtifact",
        "calibrex.core.evidence_contract:policy_json_schema",
    ),
    _EntrySpec(
        "protocol",
        "calibrex.core.evidence_contract:ProtocolArtifact",
        "calibrex.core.evidence_contract:protocol_json_schema",
    ),
    _EntrySpec(
        "transforms",
        "calibrex.core.transform_artifacts:TransformArtifact",
        "calibrex.core.transform_artifacts:transform_artifact_json_schema",
    ),
    _EntrySpec(
        "dataset-manifest",
        "calibrex.data.manifest:DatasetManifest",
        "calibrex.data.manifest:manifest_json_schema",
    ),
    _EntrySpec(
        "remote-archive-selection",
        "calibrex.data.remote_archive_selection:RemoteArchiveSelectionArtifact",
        "calibrex.data.remote_archive_selection:remote_archive_selection_json_schema",
    ),
    _EntrySpec(
        "kitti360-lidar-window-integration",
        "calibrex.data.kitti360_lidar_window_integration:Kitti360LidarWindowIntegrationArtifact",
        "calibrex.data.kitti360_lidar_window_integration:kitti360_lidar_window_integration_json_schema",
    ),
    _EntrySpec(
        "kitti-raw-lidar-window-integration",
        "calibrex.data.kitti_raw_lidar_window_integration:KittiRawLidarWindowIntegrationArtifact",
        "calibrex.data.kitti_raw_lidar_window_integration:kitti_raw_lidar_window_integration_json_schema",
    ),
    _EntrySpec(
        "doctor",
        "calibrex.core.environment_readiness:EnvironmentReadinessArtifact",
        "calibrex.diagnostics:doctor_json_schema",
        alias_of="environment-readiness",
    ),
    _EntrySpec(
        "environment-readiness",
        "calibrex.core.environment_readiness:EnvironmentReadinessArtifact",
        "calibrex.core.environment_readiness:environment_readiness_json_schema",
    ),
    _EntrySpec(
        "calibration-ci",
        "calibrex.calibration_ci:CalibrationCIArtifact",
        "calibrex.calibration_ci:calibration_ci_json_schema",
    ),
    _EntrySpec(
        "calibration-ci-lifecycle",
        "calibrex.calibration_ci_lifecycle:CalibrationCILifecycleArtifact",
        "calibrex.calibration_ci_lifecycle:calibration_ci_lifecycle_json_schema",
    ),
    _EntrySpec(
        "external-run",
        "calibrex.core.external_run:ExternalCalibrationRunArtifact",
        "calibrex.core.external_run:external_run_json_schema",
    ),
    _EntrySpec(
        "camera-imu-motion-recording",
        "calibrex.core.external_camera_imu_evidence:CameraImuMotionRecording",
        "calibrex.core.external_camera_imu_evidence:camera_imu_motion_recording_json_schema",
    ),
    _EntrySpec(
        "external-camera-imu-evidence",
        "calibrex.core.external_camera_imu_evidence:ExternalCameraImuEvidenceArtifact",
        "calibrex.core.external_camera_imu_evidence:external_camera_imu_evidence_json_schema",
    ),
    _EntrySpec(
        "kitti-falsification",
        "calibrex.evaluation.kitti_falsification_benchmark:KITTIFalsificationBenchmarkArtifact",
        "calibrex.evaluation.kitti_falsification_benchmark:kitti_falsification_json_schema",
    ),
    _EntrySpec(
        "kitti-benchmark-input",
        "calibrex.data.kitti_benchmark:KITTIBenchmarkInputManifest",
        "calibrex.data.kitti_benchmark:kitti_benchmark_input_json_schema",
    ),
    _EntrySpec(
        "koide-pilot",
        "calibrex.evaluation.koide_pilot:KoidePilotArtifact",
        "calibrex.evaluation.koide_pilot:koide_pilot_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "koide-runner",
        None,
        "calibrex.core.koide_runner:koide_runner_json_schema",
        filename="koide_runner_config.schema.json",
    ),
    _EntrySpec(
        "koide-real-pilot-request",
        "calibrex.core.koide_real_pilot:KoideRealPilotRequest",
        "calibrex.core.koide_real_pilot:koide_real_pilot_request_json_schema",
        digest_method="verify_request_digest",
    ),
    _EntrySpec(
        "koide-real-pilot-verification",
        "calibrex.core.koide_real_pilot:KoideRealPilotVerification",
        "calibrex.core.koide_real_pilot:koide_real_pilot_verification_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "koide-real-pilot-finalization",
        "calibrex.core.koide_real_pilot:KoideRealPilotFinalization",
        "calibrex.core.koide_real_pilot:koide_real_pilot_finalization_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "depth-provider",
        "calibrex.data.depth:DepthProviderArtifact",
        "calibrex.data.depth:depth_provider_json_schema",
    ),
    _EntrySpec(
        "continuous-time-camera-lidar-problem",
        "calibrex.core.continuous_time_camera_lidar_artifacts:ContinuousTimeCameraLidarProblemArtifact",
        "calibrex.core.continuous_time_camera_lidar_artifacts:continuous_time_camera_lidar_problem_json_schema",
    ),
    _EntrySpec(
        "continuous-time-camera-lidar-result",
        "calibrex.core.continuous_time_camera_lidar_artifacts:ContinuousTimeCameraLidarResultArtifact",
        "calibrex.core.continuous_time_camera_lidar_artifacts:continuous_time_camera_lidar_result_json_schema",
    ),
    _EntrySpec(
        "continuous-time-trajectory",
        "calibrex.core.continuous_time_contract:ContinuousTimeTrajectoryContract",
        "calibrex.core.continuous_time_contract:continuous_time_trajectory_json_schema",
    ),
    _EntrySpec(
        "continuous-time-trajectory-measurements",
        "calibrex.core.continuous_time_fit_artifacts:ContinuousTimeTrajectoryMeasurements",
        "calibrex.core.continuous_time_fit_artifacts:continuous_time_measurements_json_schema",
    ),
    _EntrySpec(
        "continuous-time-trajectory-fit",
        "calibrex.core.continuous_time_fit_artifacts:ContinuousTimeTrajectoryFitResultArtifact",
        "calibrex.core.continuous_time_fit_artifacts:continuous_time_fit_json_schema",
    ),
    _EntrySpec(
        "probabilistic-correspondence",
        "calibrex.core.probabilistic_correspondence:ProbabilisticCorrespondenceArtifact",
        "calibrex.core.probabilistic_correspondence:probabilistic_correspondence_json_schema",
    ),
    _EntrySpec(
        "probabilistic-pnp-result",
        "calibrex.core.probabilistic_correspondence:ProbabilisticPnpResultArtifact",
        "calibrex.core.probabilistic_correspondence:probabilistic_pnp_result_json_schema",
    ),
    _EntrySpec(
        "probabilistic-refinement-result",
        "calibrex.core.probabilistic_correspondence:ProbabilisticRefinementResultArtifact",
        "calibrex.core.probabilistic_correspondence:probabilistic_refinement_result_json_schema",
    ),
    _EntrySpec(
        "empirical-se3-uncertainty",
        "calibrex.core.empirical_uncertainty:EmpiricalSe3UncertaintyArtifact",
        "calibrex.core.empirical_uncertainty:empirical_se3_uncertainty_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-problem",
        "calibrex.core.camera_lidar_artifacts:CameraLidarCalibrationProblem",
        "calibrex.core.camera_lidar_artifacts:camera_lidar_problem_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-correspondence-export",
        "calibrex.core.camera_lidar_correspondence_export:CameraLidarCorrespondenceExportManifest",
        "calibrex.core.camera_lidar_correspondence_export:camera_lidar_correspondence_export_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-confidence-calibration",
        "calibrex.core.camera_lidar_confidence_calibration:CameraLidarConfidenceCalibrationArtifact",
        "calibrex.core.camera_lidar_confidence_calibration:camera_lidar_confidence_calibration_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-provider-support-comparison",
        "calibrex.core.camera_lidar_provider_support_comparison:CameraLidarProviderSupportComparisonArtifact",
        "calibrex.core.camera_lidar_provider_support_comparison:camera_lidar_provider_support_comparison_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-pose-initializer-protocol",
        "calibrex.core.camera_lidar_pose_initializer_benchmark:CameraLidarPoseInitializerBenchmarkProtocol",
        "calibrex.core.camera_lidar_pose_initializer_benchmark:camera_lidar_pose_initializer_protocol_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-pose-initializer-failure-analysis",
        "calibrex.core.camera_lidar_pose_initializer_failure_analysis:CameraLidarPoseInitializerFailureAnalysis",
        "calibrex.core.camera_lidar_pose_initializer_failure_analysis:camera_lidar_pose_initializer_failure_analysis_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-initializer-calibration",
        "calibrex.core.camera_lidar_initializer_calibration:CameraLidarInitializerCalibrationArtifact",
        "calibrex.core.camera_lidar_initializer_calibration:camera_lidar_initializer_calibration_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-correspondence-quality",
        "calibrex.core.camera_lidar_correspondence_quality:CameraLidarCorrespondenceQualityArtifact",
        "calibrex.core.camera_lidar_correspondence_quality:camera_lidar_correspondence_quality_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-failure-analysis",
        "calibrex.core.camera_lidar_failure_analysis:CameraLidarFailureAnalysisArtifact",
        "calibrex.core.camera_lidar_failure_analysis:camera_lidar_failure_analysis_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-sota-audit-protocol",
        "calibrex.core.camera_lidar_sota_audit:CameraLidarSotaAuditProtocol",
        "calibrex.core.camera_lidar_sota_audit:camera_lidar_sota_audit_protocol_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-sota-audit-result",
        "calibrex.core.camera_lidar_sota_audit:CameraLidarSotaAuditResult",
        "calibrex.core.camera_lidar_sota_audit:camera_lidar_sota_audit_result_json_schema",
    ),
    _EntrySpec(
        "gnss-lidar-lever-arm",
        "calibrex.core.gnss_lidar_lever_arm:GnssLidarLeverArmArtifact",
        "calibrex.core.gnss_lidar_lever_arm:gnss_lidar_lever_arm_json_schema",
    ),
    _EntrySpec(
        "imu-lidar-rotation",
        "calibrex.core.imu_lidar_rotation:ImuLidarRotationArtifact",
        "calibrex.core.imu_lidar_rotation:imu_lidar_rotation_json_schema",
    ),
    _EntrySpec(
        "imu-lidar-translation",
        "calibrex.core.imu_lidar_translation:ImuLidarTranslationArtifact",
        "calibrex.core.imu_lidar_translation:imu_lidar_translation_json_schema",
    ),
    _EntrySpec(
        "ins-lidar-hand-eye",
        "calibrex.core.ins_lidar_hand_eye:InsLidarHandEyeArtifact",
        "calibrex.core.ins_lidar_hand_eye:ins_lidar_hand_eye_json_schema",
    ),
    _EntrySpec(
        "lidar-lidar-extrinsic",
        "calibrex.core.lidar_lidar_extrinsic:LidarLidarExtrinsicArtifact",
        "calibrex.core.lidar_lidar_extrinsic:lidar_lidar_extrinsic_json_schema",
    ),
    _EntrySpec(
        "gnss-imu-lever-arm",
        "calibrex.core.gnss_imu_lever_arm:GnssImuLeverArmArtifact",
        "calibrex.core.gnss_imu_lever_arm:gnss_imu_lever_arm_json_schema",
    ),
    _EntrySpec(
        "camera-focal-scale",
        "calibrex.core.camera_focal_scale:CameraFocalScaleArtifact",
        "calibrex.core.camera_focal_scale:camera_focal_scale_json_schema",
    ),
    _EntrySpec(
        "stereo-rotation",
        "calibrex.core.stereo_rotation:StereoRotationArtifact",
        "calibrex.core.stereo_rotation:stereo_rotation_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-edge",
        "calibrex.core.camera_lidar_edge:CameraLidarEdgeArtifact",
        "calibrex.core.camera_lidar_edge:camera_lidar_edge_json_schema",
    ),
    _EntrySpec(
        "calibration-check",
        "calibrex.core.calibration_check:CalibrationCheckArtifact",
        "calibrex.core.calibration_check:calibration_check_json_schema",
    ),
    _EntrySpec(
        "check-frames",
        "calibrex.core.calibration_check:CheckFramesFile",
        "calibrex.core.calibration_check:check_frames_json_schema",
    ),
    _EntrySpec(
        "bag-estimate",
        "calibrex.core.bag_estimate:BagEstimateArtifact",
        "calibrex.core.bag_estimate:bag_estimate_json_schema",
    ),
    _EntrySpec(
        "calibration-drift",
        "calibrex.core.calibration_drift:CalibrationDriftArtifact",
        "calibrex.core.calibration_drift:calibration_drift_json_schema",
    ),
    _EntrySpec(
        "lidar-wheel-odometry",
        "calibrex.core.lidar_wheel_odometry:LidarWheelOdometryArtifact",
        "calibrex.core.lidar_wheel_odometry:lidar_wheel_odometry_json_schema",
    ),
    _EntrySpec(
        "vehicle-frame-rotation",
        "calibrex.core.vehicle_frame_rotation:VehicleFrameRotationArtifact",
        "calibrex.core.vehicle_frame_rotation:vehicle_frame_rotation_json_schema",
    ),
    _EntrySpec(
        "sota-audit-protocol",
        "calibrex.core.sota_audit:SotaAuditProtocol",
        "calibrex.core.sota_audit:sota_audit_protocol_json_schema",
    ),
    _EntrySpec(
        "sota-audit-result",
        "calibrex.core.sota_audit:SotaAuditResult",
        "calibrex.core.sota_audit:sota_audit_result_json_schema",
    ),
    _EntrySpec(
        "sota-leaderboard",
        "calibrex.core.sota_leaderboard:SotaLeaderboard",
        "calibrex.core.sota_leaderboard:sota_leaderboard_json_schema",
    ),
    _EntrySpec(
        "camera-lidar-benchmark-protocol",
        "calibrex.core.camera_lidar_artifacts:CameraLidarBenchmarkProtocol",
        "calibrex.core.camera_lidar_artifacts:camera_lidar_benchmark_protocol_json_schema",
    ),
    _EntrySpec(
        "calibration-candidate-trace",
        "calibrex.core.camera_lidar_artifacts:CalibrationCandidateTrace",
        "calibrex.core.camera_lidar_artifacts:calibration_candidate_trace_json_schema",
    ),
    _EntrySpec(
        "bullseye-plot",
        "calibrex.core.camera_lidar_artifacts:BullseyePlotArtifact",
        "calibrex.core.camera_lidar_artifacts:bullseye_plot_json_schema",
    ),
    _EntrySpec(
        "evidence-bundle",
        "calibrex.core.evidence_bundle:EvidenceBundleManifest",
        "calibrex.core.evidence_bundle:evidence_bundle_json_schema",
    ),
    _EntrySpec(
        "evidence-bundle-verification",
        "calibrex.core.evidence_bundle:EvidenceBundleVerification",
        "calibrex.core.evidence_bundle:evidence_bundle_verification_json_schema",
    ),
    _EntrySpec(
        "online-timeline",
        "calibrex.core.online_timeline:OnlineCalibrationTimelineArtifact",
        "calibrex.core.online_timeline:online_timeline_json_schema",
    ),
    _EntrySpec(
        "solid-state-context",
        "calibrex.core.solid_state:SolidStateLidarCalibrationContext",
        "calibrex.core.solid_state:solid_state_context_json_schema",
    ),
    _EntrySpec(
        "livox-time-ablation",
        "calibrex.core.livox_time_ablation:LivoxTimeAblationManifest",
        "calibrex.core.livox_time_ablation:livox_time_ablation_json_schema",
    ),
    _EntrySpec(
        "autoware-export",
        "calibrex.export.autoware:AutowareExportArtifact",
        "calibrex.export.autoware:autoware_export_json_schema",
    ),
    _EntrySpec(
        "autoware-promotion",
        "calibrex.export.autoware_promotion:AutowarePromotionArtifact",
        "calibrex.export.autoware_promotion:autoware_promotion_json_schema",
    ),
    _EntrySpec(
        "autoware-smoke",
        "calibrex.export.autoware_smoke:AutowareSmokeArtifact",
        "calibrex.export.autoware_smoke:autoware_smoke_json_schema",
    ),
    _EntrySpec(
        "raw-replay-definition",
        "calibrex.core.raw_replay:ReplayDefinition",
        "calibrex.core.raw_replay:replay_definition_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "raw-replay-plan",
        "calibrex.core.raw_replay:ReplayPlan",
        "calibrex.core.raw_replay:replay_plan_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "raw-replay-stage",
        "calibrex.core.raw_replay:ReplayStageArtifact",
        "calibrex.core.raw_replay:replay_stage_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "raw-replay-result",
        "calibrex.core.raw_replay:ReplayResult",
        "calibrex.core.raw_replay:replay_result_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "raw-replay-comparison",
        "calibrex.core.raw_replay:ReplayComparison",
        "calibrex.core.raw_replay:replay_comparison_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "field-replacement-pilot",
        "calibrex.core.raw_replay:FieldReplacementPilot",
        "calibrex.core.raw_replay:field_replacement_pilot_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "multi-lidar-service-plan",
        "calibrex.core.multi_lidar_service:MultiLidarServicePlan",
        "calibrex.core.multi_lidar_service:multi_lidar_service_plan_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "multi-lidar-service-evaluation",
        "calibrex.core.multi_lidar_service:MultiLidarServiceEvaluation",
        "calibrex.core.multi_lidar_service:multi_lidar_service_evaluation_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "camera-imu-service-plan",
        "calibrex.core.camera_imu_service:CameraImuServicePlan",
        "calibrex.core.camera_imu_service:camera_imu_service_plan_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "camera-imu-service-evaluation",
        "calibrex.core.camera_imu_service:CameraImuServiceEvaluation",
        "calibrex.core.camera_imu_service:camera_imu_service_evaluation_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "radar-service-plan",
        "calibrex.core.radar_service:RadarServicePlan",
        "calibrex.core.radar_service:radar_service_plan_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "radar-service-evaluation",
        "calibrex.core.radar_service:RadarServiceEvaluation",
        "calibrex.core.radar_service:radar_service_evaluation_json_schema",
        digest_method="verify_artifact_digest",
    ),
    _EntrySpec(
        "trajectory",
        "calibrex.core.trajectory:TrajectoryArtifact",
        "calibrex.core.trajectory:trajectory_json_schema",
    ),
    _EntrySpec(
        "report-summary",
        "calibrex.core.report_artifacts:ReportSummaryArtifact",
        "calibrex.core.report_artifacts:report_artifact_json_schema",
        schema_args=("report-summary",),
    ),
    _EntrySpec(
        "report-metrics",
        "calibrex.core.report_artifacts:ReportMetricsArtifact",
        "calibrex.core.report_artifacts:report_artifact_json_schema",
        schema_args=("report-metrics",),
    ),
    _EntrySpec(
        "report-observability",
        "calibrex.core.report_artifacts:ReportObservabilityArtifact",
        "calibrex.core.report_artifacts:report_artifact_json_schema",
        schema_args=("report-observability",),
    ),
    _EntrySpec(
        "report-degeneracy",
        "calibrex.core.report_artifacts:ReportDegeneracyArtifact",
        "calibrex.core.report_artifacts:report_artifact_json_schema",
        schema_args=("report-degeneracy",),
    ),
    _EntrySpec(
        "report-evidence",
        "calibrex.core.report_artifacts:ReportEvidenceArtifact",
        "calibrex.core.report_artifacts:report_artifact_json_schema",
        schema_args=("report-evidence",),
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

_SPEC_BY_KIND: Final = {spec.kind: spec for spec in _ENTRY_SPECS}
_ENTRY_BY_KIND: Final[dict[str, SchemaEntry]] = {}
_ENTRY_LOCK: Final = RLock()
_ALL_ENTRIES: tuple[SchemaEntry, ...] | None = None

# Namespace hints only prioritize candidates. The model's Literal annotation
# remains authoritative; old or unrelated namespaces use the complete scan.
_NAMESPACE_HINTS: Final = {
    spec.kind: {
        spec.kind.replace("-", "_"),
        (spec.filename or "").removesuffix(".schema.json"),
        "" if spec.model_ref is None else spec.model_ref.split(":")[0].rsplit(".", 1)[-1],
    }
    for spec in _ENTRY_SPECS
    if spec.model_ref is not None and spec.alias_of is None
}
_LEGACY_EXPORTS: Final = {
    reference.split(":")[1]: reference
    for spec in _ENTRY_SPECS
    for reference in (spec.model_ref, spec.schema_ref)
    if reference is not None
}


def __getattr__(name: str) -> Any:
    if name == "SCHEMA_REGISTRY":
        return schema_entries()
    if name in _LEGACY_EXPORTS:
        value = _load_symbol(_LEGACY_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LEGACY_EXPORTS) | {"SCHEMA_REGISTRY"})


class SchemaRegistryError(ValueError):
    """Raised when an artifact kind or schema version cannot be resolved."""


def schema_entries() -> tuple[SchemaEntry, ...]:
    """Return every registered schema entry in ``schema all`` order."""

    global _ALL_ENTRIES
    if _ALL_ENTRIES is not None:
        return _ALL_ENTRIES
    entries = tuple(schema_entry(spec.kind) for spec in _ENTRY_SPECS)
    with _ENTRY_LOCK:
        if _ALL_ENTRIES is None:
            _ALL_ENTRIES = entries
        return _ALL_ENTRIES


def schema_entry(kind: str) -> SchemaEntry:
    """Return the entry for ``kind`` or raise ``SchemaRegistryError``."""

    try:
        spec = _SPEC_BY_KIND[kind]
    except KeyError as exc:
        supported = ", ".join(sorted(_SPEC_BY_KIND))
        msg = f"unsupported artifact kind {kind!r}; expected one of {supported}"
        raise SchemaRegistryError(msg) from exc
    cached = _ENTRY_BY_KIND.get(kind)
    if cached is not None:
        return cached
    # Never hold our cache lock while acquiring Python's module import locks.
    loaded = spec.load()
    with _ENTRY_LOCK:
        return _ENTRY_BY_KIND.setdefault(kind, loaded)


def validatable_kinds() -> tuple[str, ...]:
    """Return kinds that ``calibrex validate`` can read, in registry order."""

    return tuple(spec.kind for spec in _ENTRY_SPECS if spec.model_ref is not None)


def schema_kinds() -> tuple[str, ...]:
    """Return all schema kinds in export order without loading their models."""

    return tuple(spec.kind for spec in _ENTRY_SPECS)


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

    namespace = schema_version.partition("/")[0].removeprefix("slac.")
    candidates = [kind for kind, hints in _NAMESPACE_HINTS.items() if namespace in hints]
    for kind in candidates:
        if schema_version in schema_entry(kind).accepted_versions:
            return kind
    for spec in _ENTRY_SPECS:
        if spec.model_ref is None or spec.alias_of is not None or spec.kind in candidates:
            continue
        if schema_version in schema_entry(spec.kind).accepted_versions:
            return spec.kind
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
    for entry in schema_entries():
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
