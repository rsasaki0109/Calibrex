"""Calibrex command-line entry point."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from calibrex import __version__
from calibrex.calibration_ci import (
    calibration_ci_json_schema,
    run_calibration_ci,
)
from calibrex.core.assessment import (
    AssessmentArtifact,
    assess_evidence_file,
    assessment_json_schema,
)
from calibrex.core.benchmark import (
    aggregate_benchmark_definition,
    benchmark_definition_json_schema,
    benchmark_json_schema,
    load_benchmark_definition,
    render_benchmark_markdown,
    update_benchmark_table_in_markdown,
)
from calibrex.core.calibration_lifecycle import calibration_lifecycle_json_schema
from calibrex.core.camera_imu_service import (
    build_camera_imu_service_plan,
    camera_imu_service_evaluation_json_schema,
    camera_imu_service_plan_json_schema,
    evaluate_camera_imu_service,
    verify_camera_imu_service,
)
from calibrex.core.camera_lidar_artifacts import (
    bullseye_plot_json_schema,
    calibration_candidate_trace_json_schema,
    camera_lidar_benchmark_protocol_json_schema,
    camera_lidar_problem_json_schema,
    load_camera_lidar_problem,
)
from calibrex.core.camera_lidar_confidence_calibration import (
    camera_lidar_confidence_calibration_json_schema,
    load_camera_lidar_confidence_calibration,
)
from calibrex.core.camera_lidar_correspondence_export import (
    build_probabilistic_correspondence_artifact,
    camera_lidar_correspondence_export_json_schema,
)
from calibrex.core.camera_lidar_correspondence_quality import (
    camera_lidar_correspondence_quality_json_schema,
)
from calibrex.core.camera_lidar_failure_analysis import (
    camera_lidar_failure_analysis_json_schema,
)
from calibrex.core.camera_lidar_initializer_calibration import (
    CameraLidarInitializerRecoveryGate,
    camera_lidar_initializer_calibration_json_schema,
    load_camera_lidar_initializer_calibration,
)
from calibrex.core.camera_lidar_pose_initializer_benchmark import (
    camera_lidar_pose_initializer_protocol_json_schema,
)
from calibrex.core.camera_lidar_pose_initializer_failure_analysis import (
    camera_lidar_pose_initializer_failure_analysis_json_schema,
)
from calibrex.core.camera_lidar_provider_support_comparison import (
    camera_lidar_provider_support_comparison_json_schema,
)
from calibrex.core.camera_lidar_sota_audit import (
    camera_lidar_sota_audit_protocol_json_schema,
    camera_lidar_sota_audit_result_json_schema,
)
from calibrex.core.capture_manifest import (
    SensorIdentity,
    SensorType,
    capture_manifest_json_schema,
    capture_manifest_verification_json_schema,
    inspect_capture,
    verify_capture_manifest_inputs,
)
from calibrex.core.capture_readiness import capture_readiness_json_schema
from calibrex.core.config import (
    CalibrationConfig,
    DatasetConfig,
    DatasetType,
    config_json_schema,
    load_config,
)
from calibrex.core.continuous_time_camera_lidar_artifacts import (
    continuous_time_camera_lidar_problem_json_schema,
    continuous_time_camera_lidar_result_json_schema,
)
from calibrex.core.continuous_time_contract import (
    continuous_time_trajectory_json_schema,
)
from calibrex.core.continuous_time_fit_artifacts import (
    continuous_time_fit_json_schema,
    continuous_time_measurements_json_schema,
)
from calibrex.core.continuous_time_imu_accel_bias import (
    continuous_time_imu_accel_bias_json_schema,
)
from calibrex.core.continuous_time_imu_clock_offset import (
    continuous_time_imu_clock_offset_json_schema,
)
from calibrex.core.continuous_time_imu_intrinsics import (
    continuous_time_imu_intrinsics_json_schema,
)
from calibrex.core.continuous_time_imu_lever_arm import (
    continuous_time_imu_lever_arm_json_schema,
)
from calibrex.core.continuous_time_imu_preintegration import (
    continuous_time_imu_preintegration_json_schema,
)
from calibrex.core.continuous_time_lidar_ablation import (
    continuous_time_lidar_ablation_json_schema,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    ContinuousTimeLidarPairArtifact,
    continuous_time_lidar_pair_json_schema,
)
from calibrex.core.continuous_time_lidar_point_to_plane import (
    continuous_time_lidar_point_to_plane_json_schema,
)
from calibrex.core.continuous_time_lidar_train_diagnostics import (
    continuous_time_lidar_train_diagnostics_json_schema,
)
from calibrex.core.continuous_time_sliding_window import (
    continuous_time_sliding_window_json_schema,
)
from calibrex.core.dynamic_window import (
    DynamicWindowConsistencyThresholds,
    dynamic_window_consistency_json_schema,
)
from calibrex.core.empirical_uncertainty import (
    empirical_se3_uncertainty_json_schema,
)
from calibrex.core.environment_readiness import environment_readiness_json_schema
from calibrex.core.evidence_bundle import (
    EvidenceBundleVerification,
    evidence_bundle_json_schema,
    evidence_bundle_verification_json_schema,
    verify_evidence_bundle,
    verify_evidence_bundle_verification,
    write_evidence_bundle_verification,
)
from calibrex.core.evidence_contract import (
    PolicyArtifact,
    policy_json_schema,
    protocol_json_schema,
)
from calibrex.core.exceptions import BenchmarkError, CalibrexError
from calibrex.core.external_run import external_run_json_schema
from calibrex.core.frames import FrameGraph
from calibrex.core.io import read_mapping, write_mapping, write_text
from calibrex.core.koide_handoff import (
    koide_execution_lock_json_schema,
    load_koide_execution_lock,
)
from calibrex.core.koide_readiness import (
    evaluate_koide_readiness_from_config,
    koide_readiness_json_schema,
)
from calibrex.core.koide_real_pilot import (
    build_koide_real_pilot_request,
    finalize_koide_real_pilot,
    koide_real_pilot_finalization_json_schema,
    koide_real_pilot_request_json_schema,
    koide_real_pilot_verification_json_schema,
    verify_koide_real_pilot_request,
)
from calibrex.core.koide_runner import koide_runner_json_schema
from calibrex.core.lifecycle_registry import (
    evaluate_lifecycle,
    init_registry,
    lifecycle_evaluation_json_schema,
    lifecycle_event_json_schema,
    lifecycle_head_json_schema,
    lifecycle_registry_json_schema,
    lifecycle_registry_state_json_schema,
    lifecycle_registry_verification_json_schema,
    lifecycle_status_json_schema,
    load_registry,
    promote_lifecycle,
    record_capture,
    register_calibration_edge,
    register_sensor,
    rollback_lifecycle,
    verify_registry,
)
from calibrex.core.livox_time_ablation import livox_time_ablation_json_schema
from calibrex.core.mcap_integrity import mcap_integrity_json_schema
from calibrex.core.multi_lidar_service import (
    build_multi_lidar_service_plan,
    evaluate_multi_lidar_service,
    multi_lidar_service_evaluation_json_schema,
    multi_lidar_service_plan_json_schema,
    verify_multi_lidar_service,
)
from calibrex.core.online_timeline import online_timeline_json_schema
from calibrex.core.probabilistic_correspondence import (
    load_probabilistic_correspondence,
    probabilistic_correspondence_json_schema,
    probabilistic_pnp_result_json_schema,
    probabilistic_refinement_result_json_schema,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.radar_service import (
    build_radar_service_plan,
    evaluate_radar_service,
    radar_service_evaluation_json_schema,
    radar_service_plan_json_schema,
    verify_radar_service,
)
from calibrex.core.raw_replay import (
    compare_raw_replays,
    field_replacement_pilot_json_schema,
    plan_raw_replay,
    replay_comparison_json_schema,
    replay_definition_json_schema,
    replay_plan_json_schema,
    replay_result_json_schema,
    replay_stage_json_schema,
    run_field_replacement_pilot,
    run_raw_replay,
    verify_raw_replay,
)
from calibrex.core.report_artifacts import (
    report_artifact_json_schema,
    report_artifact_schema_kinds,
)
from calibrex.core.result import CalibrationResult, load_result, result_json_schema
from calibrex.core.solid_state import solid_state_context_json_schema
from calibrex.core.solid_state_cross_dataset_benchmark import (
    solid_state_cross_dataset_benchmark_config_json_schema,
    solid_state_cross_dataset_benchmark_json_schema,
)
from calibrex.core.solid_state_failure_analysis import (
    solid_state_failure_analysis_json_schema,
)
from calibrex.core.solid_state_metrology_evaluation import (
    solid_state_metrology_evaluation_json_schema,
)
from calibrex.core.solid_state_synthetic_benchmark import (
    solid_state_synthetic_benchmark_json_schema,
)
from calibrex.core.trajectory import trajectory_json_schema
from calibrex.core.trajectory_window_drift import trajectory_window_drift_json_schema
from calibrex.core.transform_artifacts import transform_artifact_json_schema
from calibrex.core.validation import (
    ValidationKind,
    validate_file,
    validation_kind_choices,
)
from calibrex.data.a2d2_camera_lidar_problem import (
    build_a2d2_camera_lidar_problem,
)
from calibrex.data.depth import depth_provider_json_schema
from calibrex.data.downloads import (
    LIVOX_BASE_PCD_NAME,
    LIVOX_TARGET_PCD_NAME,
    download_livox_horizon_horizon_pcd_sample,
    livox_horizon_horizon_pcd_sample_path,
)
from calibrex.data.inspect import DatasetInspection, inspect_dataset
from calibrex.data.kitti import read_kitti_initial_transforms
from calibrex.data.kitti360_camera_lidar_problem import (
    build_kitti360_camera_lidar_problem,
)
from calibrex.data.kitti360_lidar_window_integration import (
    kitti360_lidar_window_integration_json_schema,
)
from calibrex.data.kitti_benchmark import (
    build_kitti_raw_0005_benchmark_input,
    kitti_benchmark_input_json_schema,
)
from calibrex.data.kitti_camera_lidar_problem import (
    build_kitti_raw_camera_lidar_problem,
)
from calibrex.data.kitti_raw_lidar_window_integration import (
    kitti_raw_lidar_window_integration_json_schema,
)
from calibrex.data.manifest import manifest_json_schema
from calibrex.data.public_datasets import load_public_dataset_catalog
from calibrex.data.remote_archive_selection import remote_archive_selection_json_schema
from calibrex.diagnostics import (
    build_doctor_artifact,
    doctor_json_schema,
)
from calibrex.evaluation.borer_rotation_benchmark import (
    build_borer_rotation_protocol,
    build_borer_six_dof_protocol,
    run_borer_rotation_benchmark,
)
from calibrex.evaluation.borer_six_dof_benchmark import (
    ProjectionBackend,
    run_borer_six_dof_benchmark,
)
from calibrex.evaluation.camera_lidar_confidence_calibration import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    calibrate_camera_lidar_confidence,
    refinement_options_from_camera_lidar_confidence_calibration,
)
from calibrex.evaluation.camera_lidar_correspondence_quality import (
    analyze_camera_lidar_correspondence_quality,
)
from calibrex.evaluation.camera_lidar_external_baseline_audit import (
    materialize_camera_lidar_external_baseline_audit,
)
from calibrex.evaluation.camera_lidar_initializer_calibration import (
    DEFAULT_INITIALIZER_CONFIDENCE_THRESHOLDS,
    DEFAULT_INITIALIZER_RANDOM_SEEDS,
    DEFAULT_INITIALIZER_RANSAC_THRESHOLDS_PX,
    calibrate_camera_lidar_initializer,
    initializer_options_from_camera_lidar_calibration,
)
from calibrex.evaluation.camera_lidar_provider_support_comparison import (
    compare_camera_lidar_provider_support,
)
from calibrex.evaluation.camera_lidar_sota_audit import (
    audit_camera_lidar_sota_claim,
)
from calibrex.evaluation.compare import (
    ComparisonSide,
    ResultComparison,
    compare_results,
    comparison_json_schema,
)
from calibrex.evaluation.continuous_time_camera_lidar_ablation import (
    run_continuous_time_camera_lidar_ablation,
)
from calibrex.evaluation.continuous_time_camera_lidar_run import (
    run_continuous_time_camera_lidar_problem,
)
from calibrex.evaluation.continuous_time_trajectory_adapter import (
    attach_recorded_body_trajectory,
)
from calibrex.evaluation.continuous_time_trajectory_fit import (
    run_continuous_time_trajectory_fit,
)
from calibrex.evaluation.degeneracy import degeneracy_from_inspection
from calibrex.evaluation.dynamic_window import evaluate_dynamic_window_consistency
from calibrex.evaluation.empirical_se3_uncertainty import (
    run_empirical_se3_uncertainty,
)
from calibrex.evaluation.evidence_summary import evidence_cases_from_result
from calibrex.evaluation.kitti_falsification_benchmark import (
    kitti_falsification_json_schema,
    run_kitti_falsification_benchmark,
)
from calibrex.evaluation.koide_pilot import (
    export_koide_pilot_autoware,
    koide_pilot_json_schema,
    run_koide_pilot,
)
from calibrex.evaluation.lidar import lidar_metrics_from_inspection
from calibrex.evaluation.metrics import evaluate_quality
from calibrex.evaluation.motion import motion_metrics_from_inspection
from calibrex.evaluation.pandey_recovery_benchmark import (
    load_kitti_i2i_benchmark_data,
    run_kitti_i2i_recovery_benchmark,
)
from calibrex.evaluation.probabilistic_camera_lidar_falsification import (
    run_probabilistic_camera_lidar_falsification,
)
from calibrex.evaluation.probabilistic_camera_lidar_run import (
    run_probabilistic_camera_lidar_refinement,
)
from calibrex.evaluation.probabilistic_refinement_ablation import (
    run_probabilistic_refinement_ablation,
)
from calibrex.evaluation.recommendations import build_inspection_recommendations
from calibrex.evaluation.registry import list_metric_definitions
from calibrex.evaluation.report_compare import (
    ReportComparison,
    compare_reports,
    report_comparison_json_schema,
)
from calibrex.evaluation.thresholds import ThresholdProfile, apply_metric_thresholds
from calibrex.evaluation.timing import timing_metrics_from_inspection
from calibrex.export.autoware import (
    AutowareExportConfig,
    autoware_export_json_schema,
    build_autoware_export,
    write_autoware_export,
)
from calibrex.export.autoware_promotion import (
    AutowarePromotionPolicy,
    AutowarePromotionRoots,
    apply_autoware_promotion,
    autoware_promotion_json_schema,
    build_autoware_promotion_plan,
    load_autoware_promotion,
    rollback_autoware_promotion,
    verify_autoware_promotion,
)
from calibrex.export.autoware_smoke import (
    AutowareSmokeCommand,
    AutowareSmokeConfig,
    AutowareSmokePolicy,
    autoware_smoke_json_schema,
    import_autoware_smoke,
    run_autoware_smoke,
    verify_autoware_smoke,
)
from calibrex.export.ros_tf import export_ros_tf_transforms, export_ros_tf_yaml
from calibrex.graph.problem import build_problem
from calibrex.importers.kalibr import import_kalibr_camchain
from calibrex.importers.koide import import_koide_result
from calibrex.init_templates import (
    list_sensor_template_names,
    template_summary,
    write_sensor_template,
)
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration
from calibrex.pipelines.online import (
    OnlineCalibrationRunOptions,
    evaluate_continuous_time_lidar_pair,
    evaluate_rosbag2_capture_readiness,
    evaluate_rosbag2_trajectory_window_drift,
    run_online_calibration,
)
from calibrex.solvers.koide_lidar_camera_solver import KoideLidarCameraSolver
from calibrex.solvers.opencv_probabilistic_pnp_adapter import (
    OpenCvProbabilisticPnpAdapter,
    OpenCvProbabilisticPnpOptions,
)
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
)
from calibrex.visualization.bullseye import write_bullseye_plot
from calibrex.visualization.comparison_table import write_comparison_table
from calibrex.visualization.evidence_card import write_evidence_card
from calibrex.visualization.overlays import write_camera_lidar_overlay_artifact
from calibrex.visualization.report import (
    evidence_artifact_from_result,
    report_artifact_paths,
    write_evidence_artifact,
    write_evidence_contract_artifacts,
    write_report_artifacts,
)
from calibrex.visualization.rig3d import write_rig_3d_artifact


def main(argv: list[str] | None = None) -> int:
    """Run the Calibrex CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    command = cast(Callable[[argparse.Namespace], int], args.func)
    try:
        return command(args)
    except CalibrexError as exc:
        print(f"calibrex: error: {exc}", file=sys.stderr)
        return 2


def _positive_int(value: str) -> int:
    """Parse an argparse value as a positive (>= 1) integer."""

    try:
        number = int(value)
    except ValueError as exc:
        msg = f"invalid positive integer: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if number < 1:
        msg = f"must be >= 1, got {number}"
        raise argparse.ArgumentTypeError(msg)
    return number


def _holdout_ratio(value: str) -> float:
    """Parse an argparse value as a holdout ratio in (0.0, 0.9].

    The upper bound matches `calibrex.evaluation.holdout.split_indices`;
    a ratio of 0.0 is rejected because it would leave every online batch
    without holdout evidence, so no batch could ever be accepted.
    """

    try:
        ratio = float(value)
    except ValueError as exc:
        msg = f"invalid holdout ratio: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not 0.0 < ratio <= 0.9:
        msg = f"must be > 0.0 and <= 0.9, got {ratio}"
        raise argparse.ArgumentTypeError(msg)
    return ratio


def _packaged_kitti_lidar_camera_demo_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "kitti_lidar_camera_evidence"
        / "config.yaml"
    )


def _coverage_ratio(value: str) -> float:
    """Parse an argparse value as a coverage target in (0, 1)."""

    try:
        ratio = float(value)
    except ValueError as exc:
        msg = f"invalid coverage ratio: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not 0.0 < ratio < 1.0:
        msg = f"must be > 0.0 and < 1.0, got {ratio}"
        raise argparse.ArgumentTypeError(msg)
    return ratio


def _fit_ratio(value: str) -> float:
    """Parse an argparse value as a block fit ratio in (0, 1)."""

    try:
        ratio = float(value)
    except ValueError as exc:
        msg = f"invalid fit block ratio: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not 0.0 < ratio < 1.0:
        msg = f"must be > 0.0 and < 1.0, got {ratio}"
        raise argparse.ArgumentTypeError(msg)
    return ratio


def _overconfidence_scale(value: str) -> float:
    """Parse an argparse value as an overconfidence control scale in (0, 1)."""

    try:
        scale = float(value)
    except ValueError as exc:
        msg = f"invalid overconfidence scale: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not 0.0 < scale < 1.0:
        msg = f"must be > 0.0 and < 1.0, got {scale}"
        raise argparse.ArgumentTypeError(msg)
    return scale


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibrex")
    parser.add_argument("--version", action="version", version=f"Calibrex {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser(
        "doctor",
        help="check the environment and optionally diagnose a dataset",
    )
    doctor.add_argument("path", type=Path, nargs="?")
    doctor.add_argument(
        "--workflow",
        choices=["environment", "koide"],
        default="environment",
        help="run the standard environment doctor or the Koide input preflight",
    )
    doctor.add_argument(
        "--config",
        type=Path,
        help="CalibrationConfig used by --workflow koide",
    )
    doctor.add_argument(
        "--type",
        choices=[
            "auto",
            "a2d2-lidar",
            "a2d2_lidar",
            "filesystem",
            "kitti-raw",
            "kitti_raw",
            "livox-pcd",
            "livox_pcd",
            "mcap",
            "nuscenes",
            "rosbag1",
            "rosbag2",
            "tum-rgbd",
            "tum_rgbd",
        ],
        default="auto",
        help="dataset type; defaults to path-based inference",
    )
    doctor.add_argument("--sample-limit", type=_positive_int)
    doctor.add_argument("--output", type=Path, help="write a doctor YAML/JSON artifact")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    doctor.set_defaults(func=_cmd_doctor)

    schema = subcommands.add_parser("schema", help="print JSON schema")
    schema.add_argument(
        "kind",
        choices=[
            "config",
            "result",
            "comparison",
            "report-comparison",
            "dynamic-window-consistency",
            "trajectory-window-drift",
            "capture-readiness",
            "capture-manifest",
            "capture-manifest-verification",
            "mcap-integrity",
            "koide-readiness",
            "koide-execution-lock",
            "koide-real-pilot-request",
            "koide-real-pilot-verification",
            "koide-real-pilot-finalization",
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
            "koide-runner",
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
            "evidence-bundle",
            "evidence-bundle-verification",
            "online-timeline",
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
            *report_artifact_schema_kinds(),
            "all",
        ],
    )
    schema.add_argument("--output", type=Path, help="write schema to a file")
    schema.add_argument("--output-dir", type=Path, help="write all schemas to a directory")
    schema.set_defaults(func=_cmd_schema)

    validate = subcommands.add_parser("validate", help="validate a Calibrex artifact")
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--kind",
        choices=validation_kind_choices(),
        default="auto",
        help="artifact kind; defaults to schema_version auto-detection",
    )
    validate.add_argument(
        "--verify-inputs",
        action="store_true",
        help=(
            "for capture-manifest, recompute source/config paths as well as the "
            "portable self-digest; relocated/missing inputs fail"
        ),
    )
    validate.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    validate.set_defaults(func=_cmd_validate)

    benchmark = subcommands.add_parser(
        "benchmark",
        help="aggregate a shared N-way trial matrix",
    )
    benchmark.add_argument("definition", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--markdown-output", type=Path)
    benchmark.add_argument(
        "--update-markdown",
        action="append",
        type=Path,
        default=[],
        help="replace the matching marker-delimited table; may be repeated",
    )
    benchmark.add_argument("--json", action="store_true", help="emit machine-readable summary")
    benchmark.set_defaults(func=_cmd_benchmark)

    calibration_ci = subcommands.add_parser(
        "ci",
        help="validate and assess a calibration result for continuous integration",
    )
    calibration_ci.add_argument("candidate", type=Path)
    calibration_ci.add_argument("--baseline", type=Path)
    calibration_ci.add_argument("--policy", type=Path)
    calibration_ci.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibrex-ci"),
    )
    calibration_ci.add_argument(
        "--allow-incompatible-protocol",
        action="store_true",
        help="report protocol incompatibility without failing the CI decision",
    )
    calibration_ci.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero unless the final CI status is pass",
    )
    calibration_ci.add_argument("--json", action="store_true")
    calibration_ci.set_defaults(func=_cmd_calibration_ci)

    replay = subcommands.add_parser(
        "replay",
        help="digest-bound raw-capture calibration replay and field-replacement pilot",
    )
    replay_subcommands = replay.add_subparsers(dest="replay_command", required=True)
    replay_plan = replay_subcommands.add_parser(
        "plan", help="verify inputs and write a read-only replay plan"
    )
    replay_plan.add_argument("definition", type=Path)
    replay_plan.add_argument("--output", type=Path, required=True)
    replay_plan.add_argument("--json", action="store_true")
    replay_plan.set_defaults(func=_cmd_replay_plan)
    replay_run = replay_subcommands.add_parser("run", help="execute or import a replay candidate")
    replay_run.add_argument("definition", type=Path)
    replay_run.add_argument("--output-dir", type=Path, required=True)
    replay_run.add_argument(
        "--field-replacement",
        action="store_true",
        help="also record vehicle/sensor lifecycle pilot",
    )
    replay_run.add_argument("--json", action="store_true")
    replay_run.set_defaults(func=_cmd_replay_run)
    replay_verify = replay_subcommands.add_parser(
        "verify", help="verify a replay result and every stage digest"
    )
    replay_verify.add_argument("result", type=Path)
    replay_verify.add_argument("--definition", type=Path)
    replay_verify.add_argument("--json", action="store_true")
    replay_verify.set_defaults(func=_cmd_replay_verify)
    replay_compare = replay_subcommands.add_parser("compare", help="compare two replay results")
    replay_compare.add_argument("left", type=Path)
    replay_compare.add_argument("right", type=Path)
    replay_compare.add_argument("--output", type=Path, required=True)
    replay_compare.add_argument("--json", action="store_true")
    replay_compare.set_defaults(func=_cmd_replay_compare)
    replay_pilot = replay_subcommands.add_parser(
        "field-replacement", help="run a sensor replacement lifecycle pilot"
    )
    replay_pilot.add_argument("definition", type=Path)
    replay_pilot.add_argument("--output-dir", type=Path, required=True)
    replay_pilot.add_argument("--json", action="store_true")
    replay_pilot.set_defaults(func=_cmd_replay_field_replacement)

    multi_lidar_service = subcommands.add_parser(
        "multi-lidar-service",
        help="digest-bound, read-only multi-LiDAR replacement planning and evaluation",
    )
    multi_lidar_service_subcommands = multi_lidar_service.add_subparsers(
        dest="multi_lidar_service_command", required=True
    )
    multi_lidar_plan = multi_lidar_service_subcommands.add_parser(
        "plan", help="verify inputs and write a read-only replacement plan"
    )
    multi_lidar_plan.add_argument("definition", type=Path)
    multi_lidar_plan.add_argument("--output", type=Path, required=True)
    multi_lidar_plan.add_argument("--json", action="store_true")
    multi_lidar_plan.set_defaults(func=_cmd_multi_lidar_service_plan)

    multi_lidar_evaluate = multi_lidar_service_subcommands.add_parser(
        "evaluate", help="evaluate replay gates, edge scope, and graph topology"
    )
    multi_lidar_evaluate.add_argument("plan", type=Path)
    multi_lidar_evaluate.add_argument(
        "--candidate-replay",
        type=Path,
        help="candidate raw-replay result (defaults to the plan reference)",
    )
    multi_lidar_evaluate.add_argument("--output", type=Path, required=True)
    multi_lidar_evaluate.add_argument("--evaluation-id")
    multi_lidar_evaluate.add_argument("--json", action="store_true")
    multi_lidar_evaluate.set_defaults(func=_cmd_multi_lidar_service_evaluate)

    multi_lidar_verify = multi_lidar_service_subcommands.add_parser(
        "verify", help="verify a plan or evaluation and all available source digests"
    )
    multi_lidar_verify.add_argument("artifact", type=Path)
    multi_lidar_verify.add_argument(
        "--plan", type=Path, help="plan to bind when verifying an evaluation"
    )
    multi_lidar_verify.add_argument("--json", action="store_true")
    multi_lidar_verify.set_defaults(func=_cmd_multi_lidar_service_verify)

    camera_imu_service = subcommands.add_parser(
        "camera-imu-service",
        help="digest-bound, read-only camera--IMU replacement planning and evaluation",
    )
    camera_imu_service_subcommands = camera_imu_service.add_subparsers(
        dest="camera_imu_service_command", required=True
    )
    camera_imu_plan = camera_imu_service_subcommands.add_parser(
        "plan", help="verify inputs and write a read-only replacement plan"
    )
    camera_imu_plan.add_argument("definition", type=Path)
    camera_imu_plan.add_argument("--output", type=Path, required=True)
    camera_imu_plan.add_argument("--json", action="store_true")
    camera_imu_plan.set_defaults(func=_cmd_camera_imu_service_plan)

    camera_imu_evaluate = camera_imu_service_subcommands.add_parser(
        "evaluate", help="evaluate replay, evidence, metric, and component gates"
    )
    camera_imu_evaluate.add_argument("plan", type=Path)
    camera_imu_evaluate.add_argument(
        "--candidate-replay",
        type=Path,
        help="candidate raw-replay result (defaults to the plan reference)",
    )
    camera_imu_evaluate.add_argument("--output", type=Path, required=True)
    camera_imu_evaluate.add_argument("--evaluation-id")
    camera_imu_evaluate.add_argument("--json", action="store_true")
    camera_imu_evaluate.set_defaults(func=_cmd_camera_imu_service_evaluate)

    camera_imu_verify = camera_imu_service_subcommands.add_parser(
        "verify", help="verify a plan or evaluation and all source digests"
    )
    camera_imu_verify.add_argument("artifact", type=Path)
    camera_imu_verify.add_argument(
        "--plan", type=Path, help="plan to bind when verifying an evaluation"
    )
    camera_imu_verify.add_argument("--json", action="store_true")
    camera_imu_verify.set_defaults(func=_cmd_camera_imu_service_verify)

    radar_service = subcommands.add_parser(
        "radar-service",
        help="digest-bound, read-only automotive radar replacement planning and evaluation",
    )
    radar_service_subcommands = radar_service.add_subparsers(
        dest="radar_service_command", required=True
    )
    radar_plan = radar_service_subcommands.add_parser(
        "plan", help="verify inputs and write a read-only radar replacement plan"
    )
    radar_plan.add_argument("definition", type=Path)
    radar_plan.add_argument("--output", type=Path, required=True)
    radar_plan.add_argument("--json", action="store_true")
    radar_plan.set_defaults(func=_cmd_radar_service_plan)

    radar_evaluate = radar_service_subcommands.add_parser(
        "evaluate", help="evaluate replay, evidence, metric, and component gates"
    )
    radar_evaluate.add_argument("plan", type=Path)
    radar_evaluate.add_argument(
        "--candidate-replay",
        type=Path,
        help="candidate raw-replay result (defaults to the plan reference)",
    )
    radar_evaluate.add_argument("--output", type=Path, required=True)
    radar_evaluate.add_argument("--evaluation-id")
    radar_evaluate.add_argument("--json", action="store_true")
    radar_evaluate.set_defaults(func=_cmd_radar_service_evaluate)

    radar_verify = radar_service_subcommands.add_parser(
        "verify", help="verify a radar plan or evaluation and all source digests"
    )
    radar_verify.add_argument("artifact", type=Path)
    radar_verify.add_argument("--plan", type=Path, help="plan to bind when verifying an evaluation")
    radar_verify.add_argument("--json", action="store_true")
    radar_verify.set_defaults(func=_cmd_radar_service_verify)

    verify = subcommands.add_parser(
        "verify",
        help="verify an evidence bundle manifest or saved verification artifact",
    )
    verify.add_argument("artifact", type=Path)
    verify.add_argument("--output", type=Path, help="write verification YAML/JSON")
    verify.add_argument(
        "--require-raw-recomputed",
        action="store_true",
        help=(
            "fail unless primary evidence is recomputed from verified raw inputs "
            "with SHA-backed input files"
        ),
    )
    verify.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    verify.set_defaults(func=_cmd_verify)

    assess = subcommands.add_parser("assess", help="apply a policy to evidence")
    assess.add_argument("evidence", type=Path, help="report evidence artifact")
    assess.add_argument(
        "--policy",
        type=Path,
        help="policy artifact to apply; defaults to the built-in falsification policy",
    )
    assess.add_argument("--output", type=Path, help="write assessment YAML/JSON")
    assess.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero when the assessment status is not pass",
    )
    assess.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    assess.set_defaults(func=_cmd_assess)

    evidence = subcommands.add_parser(
        "evidence",
        help="materialize evidence.json from an existing result without recomputing metrics",
    )
    evidence.add_argument("result", type=Path)
    evidence.add_argument("--output", type=Path, required=True)
    evidence.add_argument(
        "--sidecars-dir",
        type=Path,
        help=(
            "also write assessment/protocol/policy/transforms/bundle/verification "
            "sidecars into this directory"
        ),
    )
    evidence.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    evidence.set_defaults(func=_cmd_evidence)

    init = subcommands.add_parser("init", help="write a starter config")
    init.add_argument(
        "profile",
        nargs="?",
        choices=["camera-lidar-imu", "autonomous-driving-rig"],
        help="legacy synthetic starter profile (ignored when --template is set)",
    )
    init.add_argument("--output", type=Path, help="output config file or directory")
    init.add_argument(
        "--template",
        help=("copy a working sensor template config (for example velodyne_vlp16_pair_rosbag2)"),
    )
    init.add_argument(
        "--list-templates",
        action="store_true",
        help="list available sensor templates and exit",
    )
    init.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    init.set_defaults(func=_cmd_init)

    calibrate = subcommands.add_parser("calibrate", help="run calibration")
    calibrate.add_argument("config", type=Path)
    calibrate.add_argument(
        "--dry-run",
        action="store_true",
        help="validate without writing results",
    )
    calibrate.add_argument("--output-dir", type=Path)
    calibrate.add_argument(
        "--candidate-extrinsics",
        type=Path,
        action="append",
        default=[],
        help="load candidate extrinsics from a YAML/JSON file; may be repeated",
    )
    calibrate.add_argument("--strict", action="store_true", help="treat warnings as failures")
    calibrate.add_argument("--seed", type=int)
    calibrate.add_argument(
        "--online",
        action="store_true",
        help=(
            "run an online/streaming calibration session instead of one offline solve; "
            "replays the dataset's LiDAR frames as an ordered point stream, warm-starting "
            "the native point-to-plane solve per batch and gating each batch with a "
            "holdout check (see --batch-size, --rolling-window, --holdout-ratio)"
        ),
    )
    calibrate.add_argument(
        "--batch-size",
        type=_positive_int,
        default=500,
        help="online mode: number of target LiDAR points replayed per batch",
    )
    calibrate.add_argument(
        "--rolling-window",
        type=_positive_int,
        default=2000,
        help="online mode: number of recent holdout residuals kept in the rolling RMSE window",
    )
    calibrate.add_argument(
        "--holdout-ratio",
        type=_holdout_ratio,
        default=0.2,
        help=(
            "online mode: fraction of each batch held out for the per-batch gate; "
            "must be > 0.0 and <= 0.9"
        ),
    )
    calibrate.add_argument(
        "--accumulation-batches",
        type=_positive_int,
        default=1,
        help=(
            "online mode: number of accepted batches whose train points are retained "
            "for observability and solve (1 preserves per-batch behavior)"
        ),
    )
    calibrate.add_argument(
        "--max-accumulated-train-points",
        type=_positive_int,
        default=None,
        help=(
            "online mode: optional cap on retained train points when accumulating "
            "observations across batches"
        ),
    )
    calibrate.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    calibrate.set_defaults(func=_cmd_calibrate)

    compile_cmd = subcommands.add_parser("compile", help="compile config to a graph problem")
    compile_cmd.add_argument("config", type=Path)
    compile_cmd.add_argument("--output", type=Path, help="write compiled problem YAML/JSON")
    compile_cmd.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    compile_cmd.set_defaults(func=_cmd_compile)

    evaluate = subcommands.add_parser("evaluate", help="evaluate a result file")
    evaluate.add_argument("result", type=Path)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--strict", action="store_true")
    evaluate.add_argument(
        "--threshold-profile",
        choices=["default", "autonomous_driving"],
        help="override metric threshold profile",
    )
    evaluate.add_argument("--export-html", action="store_true", help="write report.html")
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=_cmd_evaluate)

    render = subcommands.add_parser(
        "render",
        help="render artifacts from an existing result without recomputing metrics",
    )
    render.add_argument("result", type=Path)
    render.add_argument(
        "--format",
        choices=["html", "evidence-card"],
        default="html",
        help="rendered artifact format",
    )
    render.add_argument("--output-dir", type=Path)
    render.add_argument("--html", type=Path, help="HTML report path or filename")
    render.add_argument("--output", type=Path, help="evidence-card SVG path or filename")
    render.add_argument("--json", action="store_true")
    render.set_defaults(func=_cmd_render)

    report = subcommands.add_parser(
        "report",
        help="deprecated alias for 'render --format html'",
    )
    report.add_argument("result", type=Path)
    report.add_argument("--output-dir", type=Path)
    report.add_argument("--html", type=Path, help="HTML report path or filename")
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=_cmd_report)

    compare = subcommands.add_parser("compare", help="compare two result files")
    compare.add_argument("left_result", type=Path)
    compare.add_argument("right_result", type=Path)
    compare.add_argument(
        "--format",
        choices=["data", "evidence-table"],
        default="data",
        help="machine-readable comparison data or a provenance-bound SVG table",
    )
    compare.add_argument("--output", type=Path, help="write comparison data or SVG")
    compare.add_argument("--left-label", default="baseline", help="left SVG column label")
    compare.add_argument("--right-label", default="candidate", help="right SVG column label")
    compare.add_argument(
        "--enforce-compatible",
        action="store_true",
        help="return non-zero unless evidence protocols are compatible",
    )
    compare.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    compare.set_defaults(func=_cmd_compare)

    report_compare = subcommands.add_parser(
        "report-compare",
        help="compare two or more labeled result files in one report artifact",
    )
    report_compare.add_argument(
        "entries",
        nargs="+",
        metavar="LABEL=RESULT",
        help="labeled result file, e.g. reference=dataset_result.yaml",
    )
    report_compare.add_argument(
        "--reference",
        metavar="LABEL",
        help="compare this entry against each other entry instead of all pairs",
    )
    report_compare.add_argument("--output", type=Path, help="write report comparison as YAML/JSON")
    report_compare.add_argument(
        "--enforce-compatible",
        action="store_true",
        help="return non-zero unless every compared pair has compatible evidence protocols",
    )
    report_compare.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    report_compare.set_defaults(func=_cmd_report_compare)

    window_consistency = subcommands.add_parser(
        "window-consistency",
        help="gate one transform across multiple labeled capture-window results",
    )
    window_consistency.add_argument(
        "entries",
        nargs="+",
        metavar="LABEL=RESULT",
        help="labeled result file, e.g. main=dynamic_result.yaml",
    )
    window_consistency.add_argument("--transform", required=True, help="transform id to compare")
    window_consistency.add_argument(
        "--max-translation-delta-m",
        type=float,
        required=True,
        help="maximum allowed pairwise translation delta in metres",
    )
    window_consistency.add_argument(
        "--max-rotation-delta-deg",
        type=float,
        required=True,
        help="maximum allowed pairwise rotation delta in degrees",
    )
    window_consistency.add_argument("--reference", metavar="LABEL")
    window_consistency.add_argument("--output", type=Path, help="write the consistency artifact")
    window_consistency.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero when the consistency grade is not PASS",
    )
    window_consistency.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    window_consistency.set_defaults(func=_cmd_window_consistency)

    trajectory_window_drift = subcommands.add_parser(
        "trajectory-window-drift",
        help="recompute odometry/map consistency for each configured capture window",
    )
    trajectory_window_drift.add_argument("config", type=Path)
    trajectory_window_drift.add_argument(
        "--reference-result",
        type=Path,
        help="optional completed result to bind into artifact provenance",
    )
    trajectory_window_drift.add_argument(
        "--deskew",
        choices=["config", "on", "off"],
        default="config",
        help="override use_point_time_offsets for this diagnostic",
    )
    trajectory_window_drift.add_argument(
        "--odometry-burst-policy",
        choices=["config", "preserve", "keep_first", "keep_last"],
        default="config",
        help="override odometry burst preprocessing for this diagnostic",
    )
    trajectory_window_drift.add_argument(
        "--odometry-burst-min-interval-s",
        type=float,
        help="override the minimum retained odometry sample interval",
    )
    trajectory_window_drift.add_argument(
        "--output", type=Path, required=True, help="write the drift artifact"
    )
    trajectory_window_drift.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    trajectory_window_drift.set_defaults(func=_cmd_trajectory_window_drift)

    capture_readiness = subcommands.add_parser(
        "calibration-readiness",
        help="check motion and geometry readiness before LiDAR calibration",
    )
    capture_readiness.add_argument("config", type=Path)
    capture_readiness.add_argument(
        "--output", type=Path, required=True, help="write the readiness artifact"
    )
    capture_readiness.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    capture_readiness.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero unless every window passes readiness",
    )
    capture_readiness.set_defaults(func=_cmd_capture_readiness)

    continuous_time_lidar = subcommands.add_parser(
        "continuous-time-lidar-pair",
        help="jointly profile LiDAR extrinsic and clock offset on rosbag1/rosbag2",
    )
    continuous_time_lidar.add_argument("config", type=Path)
    continuous_time_lidar.add_argument(
        "--output", type=Path, required=True, help="write the continuous-time artifact"
    )
    continuous_time_lidar.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    continuous_time_lidar.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero unless optimization, rank, and holdout evidence pass",
    )
    continuous_time_lidar.set_defaults(func=_cmd_continuous_time_lidar_pair)

    metrics = subcommands.add_parser("metrics", help="list registered metric definitions")
    metrics.add_argument("--json", action="store_true")
    metrics.set_defaults(func=_cmd_metrics)

    public = subcommands.add_parser("public-datasets", help="inspect public dataset catalog")
    public_subcommands = public.add_subparsers(dest="public_command", required=True)
    public_list = public_subcommands.add_parser("list", help="list public datasets")
    public_list.add_argument("--json", action="store_true")
    public_list.set_defaults(func=_cmd_public_datasets_list)
    public_show = public_subcommands.add_parser("show", help="show one public dataset")
    public_show.add_argument("dataset")
    public_show.add_argument("--json", action="store_true")
    public_show.set_defaults(func=_cmd_public_datasets_show)

    demo = subcommands.add_parser("demo", help="run reproducible public demos")
    demo_subcommands = demo.add_subparsers(dest="demo_command", required=True)
    livox_demo = demo_subcommands.add_parser(
        "livox-evidence",
        help="download public Livox PCD data and recompute evidence artifacts",
    )
    livox_demo.add_argument(
        "--config",
        type=Path,
        default=Path("examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml"),
        help="base Livox evidence config",
    )
    livox_demo.add_argument("--data-dir", type=Path, default=Path("data/public"))
    livox_demo.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/livox_horizon_horizon_pcd_sample"),
    )
    livox_demo.add_argument(
        "--no-download",
        action="store_true",
        help="use existing local PCD files instead of downloading",
    )
    livox_demo.add_argument(
        "--strict-assessment",
        action="store_true",
        help="return non-zero unless the falsification assessment is PASS",
    )
    livox_demo.add_argument("--seed", type=int)
    livox_demo.add_argument("--json", action="store_true")
    livox_demo.set_defaults(func=_cmd_demo_livox_evidence)

    kitti_demo = demo_subcommands.add_parser(
        "kitti-lidar-camera-evidence",
        help=(
            "recompute KITTI raw camera-LiDAR overlay evidence artifacts "
            "(diagnostic overlay on the LiDAR candidate, not a standalone "
            "camera calibration)"
        ),
    )
    kitti_demo.add_argument(
        "--config",
        type=Path,
        default=_packaged_kitti_lidar_camera_demo_config(),
        help="base KITTI camera-LiDAR evidence config (defaults to the packaged fixture)",
    )
    kitti_demo.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "KITTI raw sequence directory (for example one obtained through the "
            "official KITTI raw download flow); defaults to the small synthetic "
            "fixture bundled with the example config"
        ),
    )
    kitti_demo.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kitti_lidar_camera_evidence"),
    )
    kitti_demo.add_argument(
        "--strict-assessment",
        action="store_true",
        help="return non-zero unless the falsification assessment is PASS",
    )
    kitti_demo.add_argument("--seed", type=int)
    kitti_demo.add_argument("--json", action="store_true")
    kitti_demo.set_defaults(func=_cmd_demo_kitti_lidar_camera_evidence)
    kitti_benchmark = demo_subcommands.add_parser(
        "kitti-falsification-benchmark",
        help="run the pinned full-scale KITTI reference vs known-bad benchmark",
    )
    kitti_benchmark.add_argument("dataset_path", type=Path)
    kitti_benchmark.add_argument(
        "--config",
        type=Path,
        default=Path("examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml"),
    )
    kitti_benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kitti_falsification_benchmark"),
    )
    kitti_benchmark.add_argument("--max-frames", type=_positive_int, default=50)
    kitti_benchmark.add_argument("--projection-sample-points", type=_positive_int, default=4000)
    kitti_benchmark.add_argument("--seed", type=int, default=20260729)
    kitti_benchmark.add_argument(
        "--source-url",
        default="https://www.cvlibs.net/datasets/kitti/raw_data.php",
    )
    kitti_benchmark.add_argument(
        "--source-note",
        default="official KITTI raw download; user-supplied local copy",
    )
    kitti_benchmark.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero unless reference PASSes and known-bad FAILs",
    )
    kitti_benchmark.add_argument("--json", action="store_true")
    kitti_benchmark.set_defaults(func=_cmd_demo_kitti_falsification_benchmark)

    kitti = subcommands.add_parser("kitti", help="KITTI raw utilities")
    kitti_subcommands = kitti.add_subparsers(dest="kitti_command", required=True)
    kitti_import = kitti_subcommands.add_parser(
        "import-calib",
        help="import KITTI calibration files as Calibrex transforms",
    )
    kitti_import.add_argument("path", type=Path, help="directory containing calib_velo_to_cam.txt")
    kitti_import.add_argument("--output", type=Path)
    kitti_import.add_argument("--json", action="store_true")
    kitti_import.set_defaults(func=_cmd_kitti_import_calib)
    kitti_lock = kitti_subcommands.add_parser(
        "lock-benchmark-input",
        help="inspect and digest-lock the fixed KITTI raw 0005 benchmark inputs",
    )
    kitti_lock.add_argument(
        "path",
        type=Path,
        help="2011_09_26_drive_0005_sync sequence directory",
    )
    kitti_lock.add_argument("--output", type=Path, required=True)
    kitti_lock.add_argument("--json", action="store_true")
    kitti_lock.set_defaults(func=_cmd_kitti_lock_benchmark_input)
    kitti_i2i = kitti_subcommands.add_parser(
        "benchmark-i2i",
        help="benchmark baseline versus coarse native I2I recovery on locked KITTI inputs",
    )
    kitti_i2i.add_argument("input_manifest", type=Path)
    kitti_i2i.add_argument("--sequence-path", type=Path)
    kitti_i2i.add_argument("--definition-output", type=Path, required=True)
    kitti_i2i.add_argument("--output", type=Path, required=True)
    kitti_i2i.add_argument("--max-points-per-frame", type=_positive_int, default=2_000)
    kitti_i2i.add_argument("--max-iterations", type=int, default=20)
    kitti_i2i.add_argument("--json", action="store_true")
    kitti_i2i.set_defaults(func=_cmd_kitti_benchmark_i2i)

    camera_lidar = subcommands.add_parser(
        "camera-lidar",
        help="schema-valid camera-LiDAR research benchmark utilities",
    )
    camera_lidar_subcommands = camera_lidar.add_subparsers(
        dest="camera_lidar_command",
        required=True,
    )
    build_kitti_problem = camera_lidar_subcommands.add_parser(
        "build-kitti-problem",
        help="bind a frozen depth-provider artifact to KITTI raw Velodyne scans",
    )
    build_kitti_problem.add_argument(
        "sequence_path",
        type=Path,
        help="KITTI *_sync sequence directory",
    )
    build_kitti_problem.add_argument("depth_provider", type=Path)
    build_kitti_problem.add_argument("--output", type=Path, required=True)
    build_kitti_problem.add_argument("--camera-stream", default="image_02")
    build_kitti_problem.add_argument("--lidar-directory", type=Path)
    build_kitti_problem.add_argument("--lidar-manifest", type=Path)
    build_kitti_problem.add_argument("--dataset-id")
    build_kitti_problem.add_argument("--problem-id")
    build_kitti_problem.add_argument("--rotation-bound-deg", type=float, default=20.0)
    build_kitti_problem.add_argument(
        "--translation-bound-m",
        type=float,
        default=0.0,
    )
    build_kitti_problem.add_argument("--json", action="store_true")
    build_kitti_problem.set_defaults(func=_cmd_camera_lidar_build_kitti_problem)
    build_kitti360_problem = camera_lidar_subcommands.add_parser(
        "build-kitti360-problem",
        help="bind a frozen depth-provider artifact to KITTI-360 raw scans",
    )
    build_kitti360_problem.add_argument("sequence_path", type=Path)
    build_kitti360_problem.add_argument("depth_provider", type=Path)
    build_kitti360_problem.add_argument("--output", type=Path, required=True)
    build_kitti360_problem.add_argument("--calibration-root", type=Path)
    build_kitti360_problem.add_argument("--lidar-directory", type=Path)
    build_kitti360_problem.add_argument("--lidar-manifest", type=Path)
    build_kitti360_problem.add_argument("--camera-stream", default="image_03")
    build_kitti360_problem.add_argument("--split-id", default="evaluation")
    build_kitti360_problem.add_argument("--dataset-id")
    build_kitti360_problem.add_argument("--problem-id")
    build_kitti360_problem.add_argument("--rotation-bound-deg", type=float, default=20.0)
    build_kitti360_problem.add_argument(
        "--translation-bound-m",
        type=float,
        default=0.0,
    )
    build_kitti360_problem.add_argument("--json", action="store_true")
    build_kitti360_problem.set_defaults(func=_cmd_camera_lidar_build_kitti360_problem)
    build_probabilistic_correspondence = camera_lidar_subcommands.add_parser(
        "build-probabilistic-correspondence",
        help="validate provider NPZ exports and build a probabilistic correspondence artifact",
    )
    build_probabilistic_correspondence.add_argument("export_manifest", type=Path)
    build_probabilistic_correspondence.add_argument("--output", type=Path, required=True)
    build_probabilistic_correspondence.add_argument("--json", action="store_true")
    build_probabilistic_correspondence.set_defaults(
        func=_cmd_camera_lidar_build_probabilistic_correspondence
    )
    build_a2d2_problem = camera_lidar_subcommands.add_parser(
        "build-a2d2-problem",
        help="build the pre-registered A2D2 cross-family D2D smoke problem",
    )
    build_a2d2_problem.add_argument("data_directory", type=Path)
    build_a2d2_problem.add_argument("depth_provider", type=Path)
    build_a2d2_problem.add_argument("--lidar-output-directory", type=Path, required=True)
    build_a2d2_problem.add_argument("--lidar-manifest-output", type=Path, required=True)
    build_a2d2_problem.add_argument("--output", type=Path, required=True)
    build_a2d2_problem.add_argument("--problem-id")
    build_a2d2_problem.add_argument("--rotation-bound-deg", type=float, default=20.0)
    build_a2d2_problem.add_argument("--json", action="store_true")
    build_a2d2_problem.set_defaults(func=_cmd_camera_lidar_build_a2d2_problem)
    koide_pilot = camera_lidar_subcommands.add_parser(
        "benchmark-koide-pilot",
        help=(
            "accept a frozen external Koide candidate on a KITTI frame/temporal "
            "holdout with mandatory six-DoF controls"
        ),
    )
    koide_pilot.add_argument("dataset_path", type=Path)
    koide_pilot.add_argument(
        "--candidate",
        type=Path,
        help="official native calib.json or schema-valid external-run artifact",
    )
    koide_pilot.add_argument("--config", type=Path, required=False)
    koide_pilot.add_argument("--readiness", type=Path, required=False)
    koide_pilot.add_argument("--input-manifest", type=Path)
    koide_pilot.add_argument("--output-dir", type=Path, required=True)
    koide_pilot.add_argument("--camera-frame", default="camera0")
    koide_pilot.add_argument("--lidar-frame", default="lidar0")
    koide_pilot.add_argument("--camera-stream", default="image_02")
    koide_pilot.add_argument("--lidar-stream", default="velodyne_points")
    koide_pilot.add_argument("--max-frames", type=_positive_int, default=50)
    koide_pilot.add_argument("--max-points", type=_positive_int, default=4000)
    koide_pilot.add_argument("--holdout-ratio", type=float, default=0.2)
    koide_pilot.add_argument("--split-seed", type=int, default=20260823)
    koide_pilot.add_argument("--min-holdout-projection-ratio", type=float, default=0.50)
    koide_pilot.add_argument("--min-holdout-edge-alignment", type=float, default=0.20)
    koide_pilot.add_argument("--min-holdout-depth-edge-alignment", type=float, default=0.20)
    koide_pilot.add_argument("--known-bad-min-metric-delta", type=float, default=0.01)
    koide_pilot.add_argument(
        "--official-command",
        help="exact official command as one shell-like string (recorded, never shell-executed)",
    )
    koide_pilot.add_argument("--tool-source-commit")
    koide_pilot.add_argument("--container-digest")
    koide_pilot.add_argument("--json", action="store_true")
    koide_pilot.set_defaults(func=_cmd_camera_lidar_benchmark_koide_pilot)
    koide_handoff = camera_lidar_subcommands.add_parser(
        "koide-handoff",
        help=(
            "validate the pinned commercial Koide lock and print the exact next "
            "operator actions; never executes Docker or Koide"
        ),
    )
    koide_handoff.add_argument(
        "--lock",
        type=Path,
        default=Path("examples/official/koide_execution_lock.yaml"),
        help="schema-valid immutable Koide execution lock",
    )
    koide_handoff.add_argument("--json", action="store_true")
    koide_handoff.set_defaults(func=_cmd_camera_lidar_koide_handoff)
    koide_real_plan = camera_lidar_subcommands.add_parser(
        "koide-real-plan",
        aliases=["koide-real-pilot-plan"],
        help="write a blocked, digest-bound real Koide execution handoff; never executes Koide",
    )
    koide_real_plan.add_argument("--output", type=Path, required=True)
    koide_real_plan.add_argument("--camera-frame", default="camera0")
    koide_real_plan.add_argument("--lidar-frame", default="lidar0")
    koide_real_plan.add_argument("--json", action="store_true")
    koide_real_plan.set_defaults(func=_cmd_camera_lidar_koide_real_plan)
    koide_real_verify = camera_lidar_subcommands.add_parser(
        "koide-real-verify",
        aliases=["koide-real-pilot-verify"],
        help="verify external Koide evidence without executing Koide",
    )
    koide_real_verify.add_argument("request", type=Path)
    koide_real_verify.add_argument("--dataset", type=Path)
    koide_real_verify.add_argument("--archive", type=Path)
    koide_real_verify.add_argument("--input-manifest", type=Path)
    koide_real_verify.add_argument("--readiness", type=Path)
    koide_real_verify.add_argument("--config", type=Path)
    koide_real_verify.add_argument("--candidate", type=Path)
    koide_real_verify.add_argument("--external-run", type=Path)
    koide_real_verify.add_argument("--execution-log", type=Path)
    koide_real_verify.add_argument("--environment", type=Path)
    koide_real_verify.add_argument(
        "--evidence-label", choices=["official", "test"], default="official"
    )
    koide_real_verify.add_argument("--output", type=Path, required=True)
    koide_real_verify.add_argument("--json", action="store_true")
    koide_real_verify.set_defaults(func=_cmd_camera_lidar_koide_real_verify)
    koide_real_finalize = camera_lidar_subcommands.add_parser(
        "koide-real-finalize",
        aliases=["koide-real-pilot-finalize"],
        help="finalize a verified Koide output against the independent pilot",
    )
    koide_real_finalize.add_argument("request", type=Path)
    koide_real_finalize.add_argument("verification", type=Path)
    koide_real_finalize.add_argument("--pilot", type=Path)
    koide_real_finalize.add_argument("--evidence-label", choices=["official", "test"])
    koide_real_finalize.add_argument("--output", type=Path, required=True)
    koide_real_finalize.add_argument("--json", action="store_true")
    koide_real_finalize.set_defaults(func=_cmd_camera_lidar_koide_real_finalize)
    koide_export = camera_lidar_subcommands.add_parser(
        "export-koide-pilot",
        help="export only an admissible Koide pilot candidate to Autoware",
    )
    koide_export.add_argument("pilot", type=Path)
    koide_export.add_argument("--output", type=Path, required=True)
    koide_export.add_argument("--base-frame", required=True)
    koide_export.add_argument("--sensor-frame")
    koide_export.add_argument("--static-tf-output", type=Path)
    koide_export.add_argument("--manifest-output", type=Path)
    koide_export.add_argument("--force", action="store_true")
    koide_export.add_argument("--json", action="store_true")
    koide_export.set_defaults(func=_cmd_camera_lidar_export_koide_pilot)
    freeze_rotation = camera_lidar_subcommands.add_parser(
        "freeze-rotation-protocol",
        help="freeze the explicit Borer Fibonacci-sphere rotation protocol",
    )
    freeze_rotation.add_argument("problem", type=Path)
    freeze_rotation.add_argument("--output", type=Path, required=True)
    freeze_rotation.add_argument("--perturbation-count", type=_positive_int, default=200)
    freeze_rotation.add_argument("--rotation-deg", type=float, default=10.0)
    freeze_rotation.add_argument("--histogram-bins", type=_positive_int, default=32)
    freeze_rotation.add_argument("--min-visible-points", type=_positive_int, default=64)
    freeze_rotation.add_argument("--bound-deg", type=float, default=20.0)
    freeze_rotation.add_argument("--initial-step-deg", type=float, default=4.0)
    freeze_rotation.add_argument("--minimum-step-deg", type=float, default=0.05)
    freeze_rotation.add_argument("--max-evaluations", type=_positive_int, default=400)
    freeze_rotation.add_argument("--json", action="store_true")
    freeze_rotation.set_defaults(func=_cmd_camera_lidar_freeze_rotation_protocol)
    freeze_six_dof = camera_lidar_subcommands.add_parser(
        "freeze-six-dof-protocol",
        help="freeze paired rotation/translation Fibonacci perturbations",
    )
    freeze_six_dof.add_argument("problem", type=Path)
    freeze_six_dof.add_argument("--output", type=Path, required=True)
    freeze_six_dof.add_argument("--perturbation-count", type=_positive_int, default=200)
    freeze_six_dof.add_argument("--rotation-deg", type=float, default=0.5)
    freeze_six_dof.add_argument("--translation-m", type=float, default=0.5)
    freeze_six_dof.add_argument("--histogram-bins", type=_positive_int, default=32)
    freeze_six_dof.add_argument("--min-visible-points", type=_positive_int, default=64)
    freeze_six_dof.add_argument("--rotation-bound-deg", type=float, default=2.0)
    freeze_six_dof.add_argument("--translation-bound-m", type=float, default=1.0)
    freeze_six_dof.add_argument("--initial-rotation-step-deg", type=float, default=0.25)
    freeze_six_dof.add_argument("--initial-translation-step-m", type=float, default=0.10)
    freeze_six_dof.add_argument("--minimum-rotation-step-deg", type=float, default=0.01)
    freeze_six_dof.add_argument("--minimum-translation-step-m", type=float, default=0.005)
    freeze_six_dof.add_argument("--max-evaluations", type=_positive_int, default=800)
    freeze_six_dof.add_argument("--json", action="store_true")
    freeze_six_dof.set_defaults(func=_cmd_camera_lidar_freeze_six_dof_protocol)
    benchmark_rotation = camera_lidar_subcommands.add_parser(
        "benchmark-rotation",
        help="execute a frozen native D2D rotation recovery protocol",
    )
    benchmark_rotation.add_argument("problem", type=Path)
    benchmark_rotation.add_argument("protocol", type=Path)
    benchmark_rotation.add_argument("--trace-dir", type=Path, required=True)
    benchmark_rotation.add_argument("--definition-output", type=Path, required=True)
    benchmark_rotation.add_argument("--output", type=Path, required=True)
    benchmark_rotation.add_argument("--bullseye-output", type=Path)
    benchmark_rotation.add_argument("--bullseye-artifact-output", type=Path)
    benchmark_rotation.add_argument("--bootstrap-samples", type=_positive_int, default=2000)
    benchmark_rotation.add_argument("--workers", type=_positive_int, default=1)
    benchmark_rotation.add_argument(
        "--resume",
        action="store_true",
        help="reuse only digest-compatible completed traces in --trace-dir",
    )
    benchmark_rotation.add_argument("--required-hit-rate", type=float)
    benchmark_rotation.add_argument("--json", action="store_true")
    benchmark_rotation.set_defaults(func=_cmd_camera_lidar_benchmark_rotation)
    benchmark_six_dof = camera_lidar_subcommands.add_parser(
        "benchmark-six-dof",
        help="execute a frozen native D2D six-DoF recovery protocol",
    )
    benchmark_six_dof.add_argument("problem", type=Path)
    benchmark_six_dof.add_argument("protocol", type=Path)
    benchmark_six_dof.add_argument("--trace-dir", type=Path, required=True)
    benchmark_six_dof.add_argument("--definition-output", type=Path, required=True)
    benchmark_six_dof.add_argument("--output", type=Path, required=True)
    benchmark_six_dof.add_argument("--bootstrap-samples", type=_positive_int, default=2000)
    benchmark_six_dof.add_argument("--workers", type=_positive_int, default=1)
    benchmark_six_dof.add_argument(
        "--projection-backend",
        choices=("numpy", "numba_cpu"),
        default="numpy",
        help="exact D2D projection implementation (default: numpy)",
    )
    benchmark_six_dof.add_argument("--resume", action="store_true")
    benchmark_six_dof.add_argument("--required-hit-rate", type=float)
    benchmark_six_dof.add_argument("--json", action="store_true")
    benchmark_six_dof.set_defaults(func=_cmd_camera_lidar_benchmark_six_dof)
    probabilistic_pnp = camera_lidar_subcommands.add_parser(
        "refine-probabilistic-pnp",
        help="solve a schema-valid probabilistic 2D-3D correspondence frame",
    )
    probabilistic_pnp.add_argument("correspondence_artifact", type=Path)
    probabilistic_pnp.add_argument("frame_id")
    probabilistic_pnp.add_argument("--output", type=Path, required=True)
    probabilistic_pnp.add_argument("--result-id")
    probabilistic_pnp.add_argument(
        "--initial-problem",
        type=Path,
        help="camera-LiDAR problem whose D2D output/initial pose seeds PnP",
    )
    probabilistic_pnp.add_argument("--minimum-confidence", type=float, default=0.25)
    probabilistic_pnp.add_argument("--minimum-correspondences", type=_positive_int, default=6)
    probabilistic_pnp.add_argument("--ransac-reprojection-threshold-px", type=float, default=4.0)
    probabilistic_pnp.add_argument("--ransac-confidence", type=float, default=0.999)
    probabilistic_pnp.add_argument("--ransac-iterations", type=_positive_int, default=1000)
    probabilistic_pnp.add_argument("--mahalanobis-inlier-threshold", type=float, default=3.0)
    probabilistic_pnp.add_argument("--random-seed", type=int, default=0)
    probabilistic_pnp.add_argument("--json", action="store_true")
    probabilistic_pnp.set_defaults(func=_cmd_camera_lidar_refine_probabilistic_pnp)
    probabilistic_pnp_aggregate = camera_lidar_subcommands.add_parser(
        "refine-probabilistic-pnp-aggregate",
        help=(
            "solve one shared pose from reference-free confidence-supported probabilistic frames"
        ),
    )
    probabilistic_pnp_aggregate.add_argument("correspondence_artifact", type=Path)
    probabilistic_pnp_aggregate.add_argument("--output", type=Path, required=True)
    probabilistic_pnp_aggregate.add_argument("--result-id")
    probabilistic_pnp_aggregate.add_argument(
        "--initial-problem",
        type=Path,
        help="camera-LiDAR problem whose D2D output/initial pose seeds PnP",
    )
    probabilistic_pnp_aggregate.add_argument(
        "--initializer-calibration",
        type=Path,
        help="development lock supplying every aggregate PnP threshold",
    )
    probabilistic_pnp_aggregate.add_argument("--minimum-confidence", type=float, default=0.25)
    probabilistic_pnp_aggregate.add_argument(
        "--minimum-correspondences", type=_positive_int, default=6
    )
    probabilistic_pnp_aggregate.add_argument(
        "--minimum-frame-correspondences", type=_positive_int, default=4
    )
    probabilistic_pnp_aggregate.add_argument("--minimum-frames", type=_positive_int, default=2)
    probabilistic_pnp_aggregate.add_argument(
        "--ransac-reprojection-threshold-px", type=float, default=4.0
    )
    probabilistic_pnp_aggregate.add_argument("--ransac-confidence", type=float, default=0.999)
    probabilistic_pnp_aggregate.add_argument(
        "--ransac-iterations", type=_positive_int, default=1000
    )
    probabilistic_pnp_aggregate.add_argument(
        "--mahalanobis-inlier-threshold", type=float, default=3.0
    )
    probabilistic_pnp_aggregate.add_argument("--random-seed", type=int, default=0)
    probabilistic_pnp_aggregate.add_argument("--json", action="store_true")
    probabilistic_pnp_aggregate.set_defaults(
        func=_cmd_camera_lidar_refine_probabilistic_pnp_aggregate
    )
    probabilistic_multiframe = camera_lidar_subcommands.add_parser(
        "refine-probabilistic-multiframe",
        help="refine one shared D2D pose from all probabilistic frames",
    )
    probabilistic_multiframe.add_argument("correspondence_artifact", type=Path)
    probabilistic_multiframe.add_argument("initial_problem", type=Path)
    probabilistic_multiframe.add_argument(
        "--initial-trace",
        type=Path,
        help="schema-valid D2D candidate trace whose output pose initializes refinement",
    )
    probabilistic_multiframe.add_argument(
        "--confidence-calibration",
        type=Path,
        help="development lock that supplies every runtime refinement threshold",
    )
    probabilistic_multiframe.add_argument("--output", type=Path, required=True)
    probabilistic_multiframe.add_argument("--result-id")
    probabilistic_multiframe.add_argument("--minimum-confidence", type=float, default=0.25)
    probabilistic_multiframe.add_argument("--holdout-ratio", type=float, default=0.25)
    probabilistic_multiframe.add_argument("--split-seed", type=int, default=0)
    probabilistic_multiframe.add_argument(
        "--minimum-train-correspondences", type=_positive_int, default=24
    )
    probabilistic_multiframe.add_argument(
        "--minimum-holdout-correspondences", type=_positive_int, default=8
    )
    probabilistic_multiframe.add_argument("--max-evaluations", type=_positive_int, default=400)
    probabilistic_multiframe.add_argument(
        "--minimum-absolute-train-objective-improvement",
        type=float,
        default=1.0e-6,
    )
    probabilistic_multiframe.add_argument(
        "--minimum-relative-train-objective-improvement",
        type=float,
        default=1.0e-4,
    )
    probabilistic_multiframe.add_argument(
        "--maximum-train-correspondence-loss-fraction",
        type=float,
        default=0.05,
    )
    probabilistic_multiframe.add_argument(
        "--maximum-accepted-bound-fraction",
        type=float,
        default=0.95,
    )
    probabilistic_multiframe.add_argument("--without-covariance", action="store_true")
    probabilistic_multiframe.add_argument("--without-outlier-probability", action="store_true")
    probabilistic_multiframe.add_argument("--without-reliability", action="store_true")
    probabilistic_multiframe.add_argument("--json", action="store_true")
    probabilistic_multiframe.set_defaults(func=_cmd_camera_lidar_refine_probabilistic_multiframe)
    probabilistic_quality = camera_lidar_subcommands.add_parser(
        "diagnose-probabilistic-correspondence",
        help="write post-hoc frame support and six-DoF observability diagnostics",
    )
    probabilistic_quality.add_argument("correspondence_artifact", type=Path)
    probabilistic_quality.add_argument("refinement_result", type=Path)
    probabilistic_quality.add_argument("--output", type=Path, required=True)
    probabilistic_quality.add_argument("--report-id")
    probabilistic_quality.add_argument(
        "--pose-role",
        choices=["initializer", "candidate", "selected"],
        default="initializer",
    )
    probabilistic_quality.add_argument(
        "--minimum-frame-correspondences", type=_positive_int, default=4
    )
    probabilistic_quality.add_argument("--json", action="store_true")
    probabilistic_quality.set_defaults(func=_cmd_camera_lidar_diagnose_probabilistic_correspondence)
    probabilistic_confidence_calibration = camera_lidar_subcommands.add_parser(
        "calibrate-probabilistic-confidence",
        help="lock provider and refinement thresholds on a development split",
    )
    probabilistic_confidence_calibration.add_argument("correspondence_artifact", type=Path)
    probabilistic_confidence_calibration.add_argument("initial_problem", type=Path)
    probabilistic_confidence_calibration.add_argument("--output", type=Path, required=True)
    probabilistic_confidence_calibration.add_argument("--calibration-id")
    probabilistic_confidence_calibration.add_argument(
        "--thresholds",
        default=",".join(f"{value:g}" for value in DEFAULT_CONFIDENCE_THRESHOLDS),
        help="ascending comma-separated effective confidence thresholds",
    )
    probabilistic_confidence_calibration.add_argument(
        "--calibration-split-seeds", default="0,1,2,3,4,5,6,7,8,9"
    )
    probabilistic_confidence_calibration.add_argument(
        "--evaluation-split-seeds", default="0,1,2,3,4"
    )
    probabilistic_confidence_calibration.add_argument("--holdout-ratio", type=float, default=0.25)
    probabilistic_confidence_calibration.add_argument(
        "--minimum-train-correspondences", type=_positive_int, default=24
    )
    probabilistic_confidence_calibration.add_argument(
        "--minimum-holdout-correspondences", type=_positive_int, default=8
    )
    probabilistic_confidence_calibration.add_argument(
        "--minimum-frame-correspondences", type=_positive_int, default=4
    )
    probabilistic_confidence_calibration.add_argument(
        "--minimum-frame-support-rate", type=float, default=0.95
    )
    probabilistic_confidence_calibration.add_argument(
        "--minimum-full-rank-frame-rate", type=float, default=0.95
    )
    probabilistic_confidence_calibration.add_argument(
        "--development-reprojection-inlier-threshold-px",
        type=float,
        default=8.0,
    )
    probabilistic_confidence_calibration.add_argument(
        "--minimum-geometric-inlier-rate", type=float, default=0.25
    )
    probabilistic_confidence_calibration.add_argument(
        "--minimum-frame-geometric-inlier-count",
        type=_positive_int,
        default=4,
    )
    probabilistic_confidence_calibration.add_argument(
        "--minimum-frame-geometric-support-rate", type=float, default=0.95
    )
    probabilistic_confidence_calibration.add_argument(
        "--excluded-evaluation-dataset-ids",
        required=True,
        help="comma-separated evaluation dataset IDs that were not read",
    )
    probabilistic_confidence_calibration.add_argument("--json", action="store_true")
    probabilistic_confidence_calibration.set_defaults(
        func=_cmd_camera_lidar_calibrate_probabilistic_confidence
    )
    provider_support_comparison = camera_lidar_subcommands.add_parser(
        "compare-probabilistic-provider-support",
        help="compare like-for-like development provider support calibrations",
    )
    provider_support_comparison.add_argument("confidence_calibrations", type=Path, nargs="+")
    provider_support_comparison.add_argument("--output", type=Path, required=True)
    provider_support_comparison.add_argument("--comparison-id")
    provider_support_comparison.add_argument("--json", action="store_true")
    provider_support_comparison.set_defaults(
        func=_cmd_camera_lidar_compare_probabilistic_provider_support
    )
    probabilistic_initializer_calibration = camera_lidar_subcommands.add_parser(
        "calibrate-probabilistic-pnp-initializer",
        help="freeze aggregate PnP options on a development sequence",
    )
    probabilistic_initializer_calibration.add_argument("correspondence_artifact", type=Path)
    probabilistic_initializer_calibration.add_argument("initial_problem", type=Path)
    probabilistic_initializer_calibration.add_argument("--output", type=Path, required=True)
    probabilistic_initializer_calibration.add_argument("--calibration-id")
    probabilistic_initializer_calibration.add_argument(
        "--confidence-thresholds",
        default=",".join(f"{value:g}" for value in DEFAULT_INITIALIZER_CONFIDENCE_THRESHOLDS),
    )
    probabilistic_initializer_calibration.add_argument(
        "--ransac-reprojection-thresholds-px",
        default=",".join(f"{value:g}" for value in DEFAULT_INITIALIZER_RANSAC_THRESHOLDS_PX),
    )
    probabilistic_initializer_calibration.add_argument(
        "--random-seeds",
        default=",".join(str(value) for value in DEFAULT_INITIALIZER_RANDOM_SEEDS),
    )
    probabilistic_initializer_calibration.add_argument(
        "--minimum-correspondences", type=_positive_int, default=4
    )
    probabilistic_initializer_calibration.add_argument(
        "--minimum-frame-correspondences", type=_positive_int, default=4
    )
    probabilistic_initializer_calibration.add_argument(
        "--minimum-frames", type=_positive_int, default=4
    )
    probabilistic_initializer_calibration.add_argument(
        "--ransac-confidence", type=float, default=0.999
    )
    probabilistic_initializer_calibration.add_argument(
        "--ransac-iterations", type=_positive_int, default=10000
    )
    probabilistic_initializer_calibration.add_argument(
        "--mahalanobis-inlier-threshold", type=float, default=3.0
    )
    probabilistic_initializer_calibration.add_argument(
        "--rotation-error-max-deg", type=float, default=0.5
    )
    probabilistic_initializer_calibration.add_argument(
        "--translation-error-max-m", type=float, default=0.20
    )
    probabilistic_initializer_calibration.add_argument(
        "--gate-minimum-selected-frame-count", type=_positive_int, default=4
    )
    probabilistic_initializer_calibration.add_argument(
        "--gate-minimum-selected-correspondence-count",
        type=_positive_int,
        default=16,
    )
    probabilistic_initializer_calibration.add_argument(
        "--gate-minimum-ransac-inlier-count", type=_positive_int, default=12
    )
    probabilistic_initializer_calibration.add_argument(
        "--gate-maximum-ransac-inlier-rmse-px", type=float, default=8.0
    )
    probabilistic_initializer_calibration.add_argument(
        "--gate-maximum-seed-rotation-delta-deg", type=float, default=0.01
    )
    probabilistic_initializer_calibration.add_argument(
        "--gate-maximum-seed-translation-delta-m", type=float, default=0.005
    )
    probabilistic_initializer_calibration.add_argument(
        "--excluded-evaluation-dataset-ids",
        required=True,
        help="comma-separated evaluation dataset IDs that were not read",
    )
    probabilistic_initializer_calibration.add_argument("--json", action="store_true")
    probabilistic_initializer_calibration.set_defaults(
        func=_cmd_camera_lidar_calibrate_probabilistic_pnp_initializer
    )
    probabilistic_ablation = camera_lidar_subcommands.add_parser(
        "benchmark-probabilistic-ablation",
        help="run paired covariance/outlier/reliability ablations",
    )
    probabilistic_ablation.add_argument("correspondence_artifact", type=Path)
    probabilistic_ablation.add_argument("initial_problem", type=Path)
    probabilistic_ablation.add_argument(
        "--initial-trace",
        type=Path,
        help="use one digest-pinned D2D output for every paired ablation",
    )
    probabilistic_ablation.add_argument("--result-dir", type=Path, required=True)
    probabilistic_ablation.add_argument("--definition-output", type=Path, required=True)
    probabilistic_ablation.add_argument("--output", type=Path, required=True)
    probabilistic_ablation.add_argument(
        "--split-seeds",
        default="0,1,2,3,4",
        help="comma-separated deterministic frame-split seeds",
    )
    probabilistic_ablation.add_argument("--bootstrap-samples", type=_positive_int, default=2000)
    probabilistic_ablation.add_argument("--json", action="store_true")
    probabilistic_ablation.set_defaults(func=_cmd_camera_lidar_benchmark_probabilistic_ablation)
    probabilistic_falsification = camera_lidar_subcommands.add_parser(
        "benchmark-probabilistic-falsification",
        help="run the integrated D2D-seeded six-DoF refinement falsification pack",
    )
    probabilistic_falsification.add_argument("correspondence_artifact", type=Path)
    probabilistic_falsification.add_argument("initial_problem", type=Path)
    probabilistic_trace_source = probabilistic_falsification.add_mutually_exclusive_group(
        required=True
    )
    probabilistic_trace_source.add_argument("--initial-trace", type=Path)
    probabilistic_trace_source.add_argument(
        "--trace-dir",
        type=Path,
        help=(
            "complete D2D trace directory with one <trial_id>.trace.yaml per protocol perturbation"
        ),
    )
    probabilistic_falsification.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="schema-valid fixed six-DoF D2D protocol for the initialization trace",
    )
    probabilistic_falsification.add_argument("--output-dir", type=Path, required=True)
    probabilistic_falsification.add_argument(
        "--split-seeds",
        default="0,1,2,3,4",
        help="comma-separated deterministic frame-split seeds",
    )
    probabilistic_falsification.add_argument("--known-bad-rotation-deg", type=float, default=5.0)
    probabilistic_falsification.add_argument("--known-bad-translation-m", type=float, default=0.5)
    probabilistic_falsification.add_argument(
        "--known-bad-min-holdout-rmse-delta-px", type=float, default=0.5
    )
    probabilistic_falsification.add_argument("--max-final-holdout-rmse-px", type=float)
    probabilistic_falsification.add_argument(
        "--bootstrap-samples", type=_positive_int, default=2000
    )
    probabilistic_falsification.add_argument("--json", action="store_true")
    probabilistic_falsification.set_defaults(
        func=_cmd_camera_lidar_benchmark_probabilistic_falsification
    )
    continuous_time = camera_lidar_subcommands.add_parser(
        "refine-continuous-time",
        help="jointly refine extrinsic and clock offset from a frozen problem",
    )
    continuous_time.add_argument("problem", type=Path)
    continuous_time.add_argument("--output", type=Path, required=True)
    continuous_time.add_argument("--result-id")
    continuous_time.add_argument("--json", action="store_true")
    continuous_time.set_defaults(func=_cmd_camera_lidar_refine_continuous_time)
    attach_continuous_trajectory = camera_lidar_subcommands.add_parser(
        "attach-continuous-trajectory",
        help="attach a recorded T_world_body trajectory to a frozen problem",
    )
    attach_continuous_trajectory.add_argument("problem", type=Path)
    attach_continuous_trajectory.add_argument("trajectory", type=Path)
    attach_continuous_trajectory.add_argument("--output", type=Path, required=True)
    attach_continuous_trajectory.add_argument("--problem-id")
    attach_continuous_trajectory.add_argument("--json", action="store_true")
    attach_continuous_trajectory.set_defaults(func=_cmd_camera_lidar_attach_continuous_trajectory)
    continuous_time_ablation = camera_lidar_subcommands.add_parser(
        "benchmark-continuous-time-ablation",
        help="run paired clock/per-point-time/covariance ablations",
    )
    continuous_time_ablation.add_argument("problem", type=Path)
    continuous_time_ablation.add_argument("--result-dir", type=Path, required=True)
    continuous_time_ablation.add_argument("--definition-output", type=Path, required=True)
    continuous_time_ablation.add_argument("--output", type=Path, required=True)
    continuous_time_ablation.add_argument("--split-seeds", default="0,1,2,3,4")
    continuous_time_ablation.add_argument("--bootstrap-samples", type=_positive_int, default=2000)
    continuous_time_ablation.add_argument("--json", action="store_true")
    continuous_time_ablation.set_defaults(func=_cmd_camera_lidar_benchmark_continuous_time_ablation)
    sota_audit = camera_lidar_subcommands.add_parser(
        "audit-sota",
        help="evaluate a digest-frozen Camera-LiDAR SOTA claim protocol",
    )
    sota_audit.add_argument("protocol", type=Path)
    sota_audit.add_argument("--output", type=Path, required=True)
    sota_audit.add_argument("--audit-id")
    sota_audit.add_argument("--json", action="store_true")
    sota_audit.set_defaults(func=_cmd_camera_lidar_audit_sota)
    baseline_audit = camera_lidar_subcommands.add_parser(
        "audit-external-baselines",
        help="materialize Koide/UniCalib license and comparability readiness records",
    )
    baseline_audit.add_argument("problem", type=Path)
    baseline_audit.add_argument("protocol", type=Path)
    baseline_audit.add_argument("--output-dir", type=Path, required=True)
    baseline_audit.add_argument("--koide-source-commit", required=True)
    baseline_audit.add_argument("--unicalib-source-commit", required=True)
    baseline_audit.add_argument("--json", action="store_true")
    baseline_audit.set_defaults(func=_cmd_camera_lidar_audit_external_baselines)

    empirical_uncertainty = camera_lidar_subcommands.add_parser(
        "empirical-uncertainty",
        help="report empirical SE(3) coverage from block-resampled refits",
    )
    empirical_uncertainty.add_argument("correspondence", type=Path)
    empirical_uncertainty.add_argument("initial_problem", type=Path)
    empirical_uncertainty.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="directory holding one schema-valid refit artifact per resample",
    )
    empirical_uncertainty.add_argument("--output", type=Path, required=True)
    empirical_uncertainty.add_argument("--uncertainty-id")
    empirical_uncertainty.add_argument(
        "--block-length",
        type=_positive_int,
        default=5,
        help="frames per contiguous temporal block (default 5)",
    )
    empirical_uncertainty.add_argument(
        "--target-coverage",
        type=_coverage_ratio,
        default=0.9,
        help="target interval coverage in (0, 1) (default 0.9)",
    )
    empirical_uncertainty.add_argument(
        "--resample-count",
        type=_positive_int,
        default=20,
        help="number of deterministic block subsamples (default 20)",
    )
    empirical_uncertainty.add_argument(
        "--seed",
        type=int,
        default=0,
        help="deterministic block-subsample seed (default 0)",
    )
    empirical_uncertainty.add_argument(
        "--fit-block-ratio",
        type=_fit_ratio,
        default=0.6,
        help="fraction of blocks used for fitting (default 0.6)",
    )
    empirical_uncertainty.add_argument(
        "--overconfidence-scale",
        type=_overconfidence_scale,
        default=0.1,
        help="interval scale tested by the overconfidence control (default 0.1)",
    )
    empirical_uncertainty.add_argument(
        "--stability-only",
        action="store_true",
        help="report the empirical spread without a coverage claim",
    )
    empirical_uncertainty.add_argument("--json", action="store_true")
    empirical_uncertainty.set_defaults(func=_cmd_camera_lidar_empirical_uncertainty)

    external_run = subcommands.add_parser(
        "external-run",
        help="import or inspect external calibration run artifacts",
    )
    external_run_subcommands = external_run.add_subparsers(
        dest="external_run_command",
        required=True,
    )
    kalibr_import = external_run_subcommands.add_parser(
        "import-kalibr",
        help="import a Kalibr camchain YAML as a generic external-run artifact",
    )
    kalibr_import.add_argument("source", type=Path)
    kalibr_import.add_argument("--output", type=Path, required=True)
    kalibr_import.add_argument(
        "--input-artifact",
        type=Path,
        action="append",
        default=[],
        help="digest-bound Kalibr fitting input; may be repeated",
    )
    kalibr_import.add_argument("--expected-source-sha256")
    kalibr_import.add_argument("--tool-version")
    kalibr_import.add_argument("--source-commit")
    kalibr_import.add_argument(
        "--license-spdx",
        default="BSD-4-Clause",
        help="SPDX ID for the imported tool, or 'unknown'",
    )
    kalibr_import.add_argument("--training-isolation-declared", action="store_true")
    kalibr_import.add_argument("--training-isolation-evidence")
    kalibr_import.add_argument("--json", action="store_true")
    kalibr_import.set_defaults(func=_cmd_external_run_import_kalibr)

    koide_import = external_run_subcommands.add_parser(
        "import-koide",
        help="import a Koide direct_visual_lidar_calibration calib.json result",
    )
    koide_import.add_argument("source", type=Path)
    koide_import.add_argument("--output", type=Path, required=True)
    koide_import.add_argument("--lidar-frame", required=True)
    koide_import.add_argument("--camera-frame", required=True)
    koide_import.add_argument(
        "--input-artifact",
        type=Path,
        action="append",
        default=[],
        help="digest-bound Koide fitting input; may be repeated",
    )
    koide_import.add_argument("--expected-source-sha256")
    koide_import.add_argument("--tool-version")
    koide_import.add_argument("--source-commit")
    koide_import.add_argument(
        "--license-spdx",
        default="MIT",
        help="SPDX ID for the imported tool, or 'unknown'",
    )
    koide_import.add_argument("--training-isolation-declared", action="store_true")
    koide_import.add_argument("--training-isolation-evidence")
    koide_import.add_argument("--training-data-ids-sha256")
    koide_import.add_argument("--holdout-data-ids-sha256")
    koide_import.add_argument("--json", action="store_true")
    koide_import.set_defaults(func=_cmd_external_run_import_koide)

    koide_run = external_run_subcommands.add_parser(
        "run-koide",
        help="run the typed Koide preprocess/initial_guess/calibrate workflow from a config",
    )
    koide_run.add_argument("config", type=Path)
    koide_run.add_argument("--output", type=Path, required=True)
    koide_run.add_argument("--input-artifact", type=Path, action="append", default=[])
    koide_run.add_argument("--json", action="store_true")
    koide_run.set_defaults(func=_cmd_external_run_koide)

    trajectory = subcommands.add_parser(
        "trajectory",
        help="schema-valid continuous-time trajectory utilities",
    )
    trajectory_subcommands = trajectory.add_subparsers(
        dest="trajectory_command",
        required=True,
    )
    trajectory_build = trajectory_subcommands.add_parser(
        "build-contract",
        help="build a trajectory contract from a simple knots YAML",
    )
    trajectory_build.add_argument("knots_yaml", type=Path)
    trajectory_build.add_argument("--output", type=Path, required=True)
    trajectory_build.add_argument("--trajectory-id", required=True)
    trajectory_build.add_argument("--world-frame", default="map")
    trajectory_build.add_argument("--body-frame", default="base")
    trajectory_build.add_argument(
        "--interpolation",
        choices=["piecewise_linear_slerp/v0.1", "screw_linear/v0.1"],
        default="screw_linear/v0.1",
    )
    trajectory_build.add_argument("--json", action="store_true")
    trajectory_build.set_defaults(func=_cmd_trajectory_build_contract)
    trajectory_fit = trajectory_subcommands.add_parser(
        "fit",
        help="fit trajectory knots against point/pose measurements",
    )
    trajectory_fit.add_argument("trajectory", type=Path)
    trajectory_fit.add_argument("measurements", type=Path)
    trajectory_fit.add_argument("--output", type=Path, required=True)
    trajectory_fit.add_argument("--fit-id")
    trajectory_fit.add_argument("--max-iterations", type=_positive_int, default=30)
    trajectory_fit.add_argument("--initial-damping", type=float, default=1.0e-3)
    trajectory_fit.add_argument("--json", action="store_true")
    trajectory_fit.set_defaults(func=_cmd_trajectory_fit)
    trajectory_p2p = trajectory_subcommands.add_parser(
        "recover-point-to-plane",
        help="synthetic LiDAR point-to-plane recovery with holdout and known-bad",
    )
    trajectory_p2p.add_argument("--output", type=Path, required=True)
    trajectory_p2p.add_argument("--seed", type=int, default=20260820)
    trajectory_p2p.add_argument("--recovery-id", default="ct-lidar-p2p-synthetic")
    trajectory_p2p.add_argument("--json", action="store_true")
    trajectory_p2p.set_defaults(func=_cmd_trajectory_recover_point_to_plane)
    trajectory_imu = trajectory_subcommands.add_parser(
        "recover-imu-preintegration",
        help="synthetic IMU gyro pre-integration recovery with holdout and known-bad",
    )
    trajectory_imu.add_argument("--output", type=Path, required=True)
    trajectory_imu.add_argument("--seed", type=int, default=20260820)
    trajectory_imu.add_argument("--recovery-id", default="ct-imu-preintegration-synthetic")
    trajectory_imu.add_argument("--json", action="store_true")
    trajectory_imu.set_defaults(func=_cmd_trajectory_recover_imu_preintegration)
    trajectory_lever = trajectory_subcommands.add_parser(
        "recover-imu-lever-arm",
        help="synthetic IMU lever-arm recovery with holdout and known-bad",
    )
    trajectory_lever.add_argument("--output", type=Path, required=True)
    trajectory_lever.add_argument("--seed", type=int, default=20260820)
    trajectory_lever.add_argument("--recovery-id", default="ct-imu-lever-arm-synthetic")
    trajectory_lever.add_argument("--json", action="store_true")
    trajectory_lever.set_defaults(func=_cmd_trajectory_recover_imu_lever_arm)
    trajectory_clock = trajectory_subcommands.add_parser(
        "recover-imu-clock-offset",
        help="synthetic IMU clock-offset recovery with holdout and known-bad",
    )
    trajectory_clock.add_argument("--output", type=Path, required=True)
    trajectory_clock.add_argument("--seed", type=int, default=20260820)
    trajectory_clock.add_argument("--recovery-id", default="ct-imu-clock-offset-synthetic")
    trajectory_clock.add_argument("--json", action="store_true")
    trajectory_clock.set_defaults(func=_cmd_trajectory_recover_imu_clock_offset)
    trajectory_accel = trajectory_subcommands.add_parser(
        "recover-imu-accel-bias",
        help="synthetic IMU accelerometer-bias and gravity recovery",
    )
    trajectory_accel.add_argument("--output", type=Path, required=True)
    trajectory_accel.add_argument("--seed", type=int, default=20260820)
    trajectory_accel.add_argument("--recovery-id", default="ct-imu-accel-bias-synthetic")
    trajectory_accel.add_argument("--json", action="store_true")
    trajectory_accel.set_defaults(func=_cmd_trajectory_recover_imu_accel_bias)
    trajectory_intrinsics = trajectory_subcommands.add_parser(
        "recover-imu-intrinsics",
        help="synthetic IMU diagonal scale intrinsics recovery",
    )
    trajectory_intrinsics.add_argument("--output", type=Path, required=True)
    trajectory_intrinsics.add_argument("--seed", type=int, default=20260820)
    trajectory_intrinsics.add_argument("--recovery-id", default="ct-imu-intrinsics-synthetic")
    trajectory_intrinsics.add_argument("--json", action="store_true")
    trajectory_intrinsics.set_defaults(func=_cmd_trajectory_recover_imu_intrinsics)
    trajectory_window = trajectory_subcommands.add_parser(
        "recover-sliding-window",
        help="synthetic sliding-window marginalization recovery",
    )
    trajectory_window.add_argument("--output", type=Path, required=True)
    trajectory_window.add_argument("--seed", type=int, default=20260820)
    trajectory_window.add_argument("--recovery-id", default="ct-sliding-window-synthetic")
    trajectory_window.add_argument("--json", action="store_true")
    trajectory_window.set_defaults(func=_cmd_trajectory_recover_sliding_window)

    lifecycle = subcommands.add_parser(
        "lifecycle",
        help="calibration lifecycle adoption and rollback evidence",
    )
    lifecycle_subcommands = lifecycle.add_subparsers(dest="lifecycle_command")
    lifecycle_subcommands.required = True
    lifecycle_simulate = lifecycle_subcommands.add_parser(
        "simulate",
        help="synthetic lifecycle replay with weak-observability and rollback controls",
    )
    lifecycle_simulate.add_argument("--output", type=Path, required=True)
    lifecycle_simulate.add_argument("--seed", type=int, default=20260820)
    lifecycle_simulate.add_argument("--lifecycle-id", default="calibration-lifecycle-synthetic")
    lifecycle_simulate.add_argument("--json", action="store_true")
    lifecycle_simulate.set_defaults(func=_cmd_lifecycle_simulate)

    lifecycle_init = lifecycle_subcommands.add_parser(
        "init",
        help="create an empty append-only calibration lifecycle registry",
    )
    lifecycle_init.add_argument("--registry-root", type=Path, required=True)
    lifecycle_init.add_argument("--registry-id", default="calibrex-registry")
    lifecycle_init.add_argument("--operator", default="unknown")
    lifecycle_init.add_argument("--reason", default="initialize calibration lifecycle registry")
    lifecycle_init.add_argument("--timestamp")
    lifecycle_init.add_argument("--json", action="store_true")
    lifecycle_init.set_defaults(func=_cmd_lifecycle_init)

    lifecycle_register_sensor = lifecycle_subcommands.add_parser(
        "register-sensor",
        help="register a physical sensor and its stable vehicle/sensor-kit identity",
    )
    lifecycle_register_sensor.add_argument("--registry-root", type=Path, required=True)
    lifecycle_register_sensor.add_argument("--sensor-id", required=True)
    lifecycle_register_sensor.add_argument("--vehicle-id", required=True)
    lifecycle_register_sensor.add_argument("--sensor-kit-id", required=True)
    lifecycle_register_sensor.add_argument("--serial", required=True)
    lifecycle_register_sensor.add_argument("--model", required=True)
    lifecycle_register_sensor.add_argument("--firmware", required=True)
    lifecycle_register_sensor.add_argument("--mount", required=True)
    lifecycle_register_sensor.add_argument("--frame")
    lifecycle_register_sensor.add_argument("--install", action="store_true")
    lifecycle_register_sensor.add_argument("--operator", default="unknown")
    lifecycle_register_sensor.add_argument("--reason", default="register physical sensor")
    lifecycle_register_sensor.add_argument("--timestamp")
    lifecycle_register_sensor.add_argument("--json", action="store_true")
    lifecycle_register_sensor.set_defaults(func=_cmd_lifecycle_register_sensor)

    lifecycle_capture = lifecycle_subcommands.add_parser(
        "capture",
        help="append a digest-verified capture-manifest event",
    )
    lifecycle_capture.add_argument("--registry-root", type=Path, required=True)
    lifecycle_capture.add_argument("--capture-manifest", type=Path, required=True)
    lifecycle_capture.add_argument("--operator", default="unknown")
    lifecycle_capture.add_argument("--reason", default="record calibration capture")
    lifecycle_capture.add_argument("--timestamp")
    lifecycle_capture.add_argument("--json", action="store_true")
    lifecycle_capture.set_defaults(func=_cmd_lifecycle_capture)

    lifecycle_edge = lifecycle_subcommands.add_parser(
        "register-edge",
        help="register one calibration edge and optional incumbent artifacts",
    )
    lifecycle_edge.add_argument("--registry-root", type=Path, required=True)
    lifecycle_edge.add_argument("--edge-id", required=True)
    lifecycle_edge.add_argument("--vehicle-id", required=True)
    lifecycle_edge.add_argument("--sensor-kit-id", required=True)
    lifecycle_edge.add_argument("--parent-frame", required=True)
    lifecycle_edge.add_argument("--child-frame", required=True)
    lifecycle_edge.add_argument("--incumbent-transform", type=Path)
    lifecycle_edge.add_argument("--incumbent-result", type=Path)
    lifecycle_edge.add_argument("--operator", default="unknown")
    lifecycle_edge.add_argument("--reason", default="register calibration edge")
    lifecycle_edge.add_argument("--timestamp")
    lifecycle_edge.add_argument("--json", action="store_true")
    lifecycle_edge.set_defaults(func=_cmd_lifecycle_register_edge)

    lifecycle_evaluate = lifecycle_subcommands.add_parser(
        "evaluate",
        help="evaluate candidate/capture/evidence without changing incumbents",
    )
    lifecycle_evaluate.add_argument("--registry-root", type=Path, required=True)
    lifecycle_evaluate.add_argument("--edge-id", action="append", required=True)
    lifecycle_evaluate.add_argument("--capture-manifest", type=Path, required=True)
    lifecycle_evaluate.add_argument("--candidate-result", type=Path)
    lifecycle_evaluate.add_argument("--candidate-transform", type=Path)
    lifecycle_evaluate.add_argument("--candidate-pilot", type=Path)
    lifecycle_evaluate.add_argument("--evidence", type=Path)
    lifecycle_evaluate.add_argument("--assessment", type=Path)
    lifecycle_evaluate.add_argument("--promotion", type=Path)
    lifecycle_evaluate.add_argument("--smoke", type=Path)
    lifecycle_evaluate.add_argument("--policy", type=Path)
    lifecycle_evaluate.add_argument("--output", type=Path)
    lifecycle_evaluate.add_argument("--operator", default="unknown")
    lifecycle_evaluate.add_argument("--reason", default="evaluate calibration candidate")
    lifecycle_evaluate.add_argument("--timestamp")
    lifecycle_evaluate.add_argument("--json", action="store_true")
    lifecycle_evaluate.set_defaults(func=_cmd_lifecycle_evaluate)

    lifecycle_promote = lifecycle_subcommands.add_parser(
        "promote",
        help="promote a PASS/ADOPT evaluation for exactly the declared edges",
    )
    lifecycle_promote.add_argument("--registry-root", type=Path, required=True)
    lifecycle_promote.add_argument("--edge-id", action="append", required=True)
    lifecycle_promote.add_argument("--evaluation", type=Path)
    lifecycle_promote.add_argument("--promotion", type=Path)
    lifecycle_promote.add_argument("--smoke", type=Path)
    lifecycle_promote.add_argument("--operator", default="unknown")
    lifecycle_promote.add_argument("--reason", default="promote evaluated calibration candidate")
    lifecycle_promote.add_argument("--timestamp")
    lifecycle_promote.add_argument("--json", action="store_true")
    lifecycle_promote.set_defaults(func=_cmd_lifecycle_promote)

    lifecycle_rollback = lifecycle_subcommands.add_parser(
        "rollback",
        help="append a guarded rollback to a prior incumbent",
    )
    lifecycle_rollback.add_argument("--registry-root", type=Path, required=True)
    lifecycle_rollback.add_argument("--edge-id", required=True)
    lifecycle_rollback.add_argument("--target-sequence", type=int)
    lifecycle_rollback.add_argument("--target-event-sha256")
    lifecycle_rollback.add_argument("--promotion", type=Path)
    lifecycle_rollback.add_argument("--operator", default="unknown")
    lifecycle_rollback.add_argument(
        "--reason", default="rollback calibration edge to prior incumbent"
    )
    lifecycle_rollback.add_argument("--timestamp")
    lifecycle_rollback.add_argument("--json", action="store_true")
    lifecycle_rollback.set_defaults(func=_cmd_lifecycle_rollback)

    lifecycle_verify = lifecycle_subcommands.add_parser(
        "verify",
        help="verify event hash chain, head, projection, and source digests",
    )
    lifecycle_verify.add_argument("--registry-root", type=Path, required=True)
    lifecycle_verify.add_argument("--no-verify-sources", action="store_true")
    lifecycle_verify.add_argument("--json", action="store_true")
    lifecycle_verify.set_defaults(func=_cmd_lifecycle_verify)

    lifecycle_status = lifecycle_subcommands.add_parser(
        "status",
        help="show the verified lifecycle registry status projection",
    )
    lifecycle_status.add_argument("--registry-root", type=Path, required=True)
    lifecycle_status.add_argument("--json", action="store_true")
    lifecycle_status.set_defaults(func=_cmd_lifecycle_status)

    visualize = subcommands.add_parser("visualize", help="render result visualizations")
    visualize.add_argument("result", type=Path)
    visualize.add_argument(
        "--reference-result",
        type=Path,
        help="overlay another result's transforms as reference extrinsics",
    )
    visualize.add_argument("--output-dir", type=Path)
    visualize.add_argument("--export-html", action="store_true")
    visualize.add_argument("--json", action="store_true")
    visualize.set_defaults(func=_cmd_visualize)

    capture = subcommands.add_parser(
        "capture",
        help="capture/session inventory and readiness evidence",
    )
    capture_subcommands = capture.add_subparsers(dest="capture_command")
    capture_subcommands.required = True
    capture_inspect = capture_subcommands.add_parser(
        "inspect",
        help="inspect a bag or plain-file source into a capture manifest",
    )
    capture_inspect.add_argument("path", type=Path)
    capture_inspect.add_argument(
        "--type",
        choices=["auto", "files", "filesystem", "rosbag1", "rosbag2", "mcap"],
        default="auto",
        help=(
            "source format; defaults to path-based inference. MCAP inspection "
            "also emits bounded official framing/CRC/link evidence"
        ),
    )
    capture_inspect.add_argument("--capture-id")
    capture_inspect.add_argument("--session-id")
    capture_inspect.add_argument("--vehicle-id")
    capture_inspect.add_argument("--sensor-kit-id")
    capture_inspect.add_argument(
        "--sensor",
        action="append",
        default=[],
        metavar="ID:TYPE[:SERIAL[:MODEL[:FIRMWARE[:MOUNT[:FRAME]]]]]]",
        help="sensor identity; may be repeated",
    )
    capture_inspect.add_argument(
        "--readiness-profile",
        choices=["strict", "declared"],
        default="strict",
        help=(
            "strict treats every discovered stream as required; declared uses "
            "--required-stream/--required-kind and explicit optional critical streams"
        ),
    )
    capture_inspect.add_argument(
        "--required-stream",
        action="append",
        default=[],
        help="stream ID required for calibration intake; may be repeated",
    )
    capture_inspect.add_argument(
        "--optional-stream",
        action="append",
        default=[],
        help="stream ID explicitly treated as non-gating; may be repeated",
    )
    capture_inspect.add_argument(
        "--required-kind",
        action="append",
        choices=[
            "image",
            "camera_info",
            "depth_image",
            "rgbd",
            "pointcloud",
            "imu",
            "radar",
            "odometry",
            "metadata",
            "trajectory",
            "file",
            "other",
        ],
        default=[],
        help="stream kind required for calibration intake; may be repeated",
    )
    capture_inspect.add_argument("--config", type=Path)
    capture_inspect.add_argument("--sample-limit", type=_positive_int, default=4)
    capture_inspect.add_argument("--output", type=Path)
    capture_inspect.add_argument("--json", action="store_true")
    capture_inspect.set_defaults(func=_cmd_capture_inspect)
    capture_verify = capture_subcommands.add_parser(
        "verify",
        help="verify a saved capture manifest and recompute declared input digests",
    )
    capture_verify.add_argument("manifest", type=Path)
    capture_verify.add_argument("--json", action="store_true")
    capture_verify.set_defaults(func=_cmd_capture_verify)

    autoware = subcommands.add_parser(
        "autoware",
        help="ROS-independent Autoware package adapters",
    )
    autoware_subcommands = autoware.add_subparsers(dest="autoware_command", required=True)
    promotion = autoware_subcommands.add_parser(
        "promotion",
        help="plan, verify, apply, or roll back a sensor-kit promotion",
    )
    promotion_subcommands = promotion.add_subparsers(dest="promotion_command", required=True)
    promotion_plan = promotion_subcommands.add_parser(
        "plan",
        help="inspect a package and generate a deterministic dry-run plan",
    )
    promotion_plan.add_argument("candidate", type=Path)
    promotion_plan.add_argument("--workspace-root", type=Path, required=True)
    promotion_plan.add_argument("--package-root", type=Path, required=True)
    promotion_plan.add_argument("--individual-params-root", type=Path)
    promotion_plan.add_argument("--sensor-kit-description-root", type=Path)
    promotion_plan.add_argument("--vehicle-id", required=True)
    promotion_plan.add_argument("--sensor-kit-id", required=True)
    promotion_plan.add_argument("--baseline-manifest-sha256")
    promotion_plan.add_argument("--base-frame", default="base_link")
    promotion_plan.add_argument("--sensor-kit-base-frame", default="sensor_kit_base_link")
    promotion_plan.add_argument("--allowed-frame", action="append", default=[])
    promotion_plan.add_argument("--allowed-topic", action="append", default=[])
    promotion_plan.add_argument("--allowed-file", action="append", default=[])
    promotion_plan.add_argument("--required-file", action="append", default=[])
    promotion_plan.add_argument("--allow-warnings", action="store_true")
    promotion_plan.add_argument(
        "--profile", choices=["production", "developer"], default="production"
    )
    promotion_plan.add_argument("--require-smoke-for-apply", action="store_true")
    promotion_plan.add_argument("--output", type=Path, required=True)
    promotion_plan.add_argument("--json", action="store_true")
    promotion_plan.set_defaults(func=_cmd_autoware_promotion_plan)

    promotion_verify = promotion_subcommands.add_parser(
        "verify",
        help="re-read a plan's package and candidate without writing source files",
    )
    promotion_verify.add_argument("plan", type=Path)
    promotion_verify.add_argument("--output", type=Path)
    promotion_verify.add_argument("--json", action="store_true")
    promotion_verify.set_defaults(func=_cmd_autoware_promotion_verify)

    promotion_apply = promotion_subcommands.add_parser(
        "apply",
        help="atomically apply an admissible PASS/ADOPT plan",
    )
    promotion_apply.add_argument("plan", type=Path)
    promotion_apply.add_argument(
        "--smoke-artifact",
        type=Path,
        help="digest-bound PASS smoke artifact required for production plans",
    )
    promotion_apply.add_argument("--output", type=Path)
    promotion_apply.add_argument("--json", action="store_true")
    promotion_apply.set_defaults(func=_cmd_autoware_promotion_apply)

    promotion_rollback = promotion_subcommands.add_parser(
        "rollback",
        help="restore the exact files recorded by an applied promotion",
    )
    promotion_rollback.add_argument("plan", type=Path)
    promotion_rollback.add_argument("--output", type=Path)
    promotion_rollback.add_argument("--json", action="store_true")
    promotion_rollback.set_defaults(func=_cmd_autoware_promotion_rollback)

    smoke = autoware_subcommands.add_parser(
        "smoke",
        help="run, import, or verify downstream Autoware smoke evidence",
    )
    smoke_subcommands = smoke.add_subparsers(dest="smoke_command", required=True)
    smoke_run = smoke_subcommands.add_parser("run", help="run argv-only smoke stages")
    smoke_run.add_argument("plan", type=Path)
    smoke_run.add_argument(
        "--mode",
        choices=["subprocess", "container", "precomputed"],
        default="subprocess",
    )
    smoke_run.add_argument("--profile", choices=["production", "developer"], default="production")
    smoke_run.add_argument(
        "--stage-command",
        action="append",
        default=[],
        metavar="STAGE=JSON_ARGV",
        help='stage command, e.g. xacro_urdf=["python","-c","pass"] (repeatable)',
    )
    smoke_run.add_argument("--precomputed", type=Path)
    smoke_run.add_argument("--container-digest")
    smoke_run.add_argument("--environment-digest")
    smoke_run.add_argument("--tool-digest")
    smoke_run.add_argument("--timeout", type=float, default=120.0)
    smoke_run.add_argument("--output", type=Path, required=True)
    smoke_run.add_argument("--json", action="store_true")
    smoke_run.set_defaults(func=_cmd_autoware_smoke_run)

    smoke_import = smoke_subcommands.add_parser(
        "import", help="import precomputed smoke evidence and bind it to a plan"
    )
    smoke_import.add_argument("artifact", type=Path)
    smoke_import.add_argument("--plan", type=Path, required=True)
    smoke_import.add_argument("--output", type=Path, required=True)
    smoke_import.add_argument("--json", action="store_true")
    smoke_import.set_defaults(func=_cmd_autoware_smoke_import)

    smoke_verify = smoke_subcommands.add_parser(
        "verify", help="verify smoke self-digest and exact plan binding"
    )
    smoke_verify.add_argument("artifact", type=Path)
    smoke_verify.add_argument("--plan", type=Path, required=True)
    smoke_verify.add_argument("--json", action="store_true")
    smoke_verify.set_defaults(func=_cmd_autoware_smoke_verify)

    inspect = subcommands.add_parser("inspect", help="inspect a dataset")
    inspect.add_argument("path", type=Path)
    inspect.add_argument(
        "--type",
        choices=[
            "a2d2-lidar",
            "a2d2_lidar",
            "filesystem",
            "kitti-raw",
            "kitti_raw",
            "livox-pcd",
            "livox_pcd",
            "mcap",
            "nuscenes",
            "rosbag1",
            "rosbag2",
            "tum-rgbd",
            "tum_rgbd",
        ],
        default="filesystem",
    )
    inspect.add_argument(
        "--sample-limit",
        type=_positive_int,
        help=(
            "override the number of frames/files sampled for public-dataset diagnostics "
            "(a2d2_lidar, livox_pcd, kitti_raw); defaults preserve today's behavior"
        ),
    )
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(func=_cmd_inspect)

    export = subcommands.add_parser("export", help="export a result to another format")
    export.add_argument("result", type=Path)
    export.add_argument("--format", choices=["ros-tf", "autoware"], required=True)
    export.add_argument(
        "--kind",
        choices=["result", "continuous-time-lidar-pair"],
        default="result",
        help="input artifact kind; continuous-time exports its refined transform",
    )
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--base-frame",
        help="explicit Autoware parent frame; required when transform parents are ambiguous",
    )
    export.add_argument(
        "--sensor-frame",
        dest="sensor_frames",
        action="append",
        default=[],
        help="sensor child frame to export; may be repeated",
    )
    export.add_argument(
        "--transform",
        dest="transform_names",
        action="append",
        default=[],
        help="exact source transform key to export; may be repeated",
    )
    export.add_argument(
        "--invert",
        dest="invert_transforms",
        action="append",
        default=[],
        help="exact source transform key to invert before export; may be repeated",
    )
    export.add_argument(
        "--static-tf-output",
        type=Path,
        help="ROS 2 static_transform_publisher launch snippet (default: output sibling)",
    )
    export.add_argument(
        "--manifest-output",
        type=Path,
        help="schema-valid provenance manifest (default: output sibling)",
    )
    export.add_argument(
        "--force",
        action="store_true",
        help="allow replacement of existing export outputs",
    )
    export.add_argument("--json", action="store_true", help="emit machine-readable summary")
    export.set_defaults(func=_cmd_export)

    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    if args.workflow == "koide":
        if args.config is None:
            _die("doctor --workflow koide requires --config CONFIG")
        koide_artifact = evaluate_koide_readiness_from_config(
            args.config,
            command=_doctor_command(args),
        )
        payload = koide_artifact.model_dump(mode="json", exclude_none=False)
        if args.output is not None:
            koide_artifact.save(args.output)
            payload = koide_artifact.with_artifact_digest().model_dump(
                mode="json", exclude_none=False
            )
        if args.json:
            _emit(payload, as_json=True)
        else:
            print(
                f"Koide readiness: {koide_artifact.status.upper()} "
                f"({koide_artifact.hardware_profile}); dataset={koide_artifact.dataset_path}"
            )
            for check in koide_artifact.checks:
                print(f"  {check.name}: {check.status} - {check.reason}")
            for recommendation in koide_artifact.recommendations:
                print(f"  next: {recommendation}")
            if args.output is not None:
                print(f"  artifact: {args.output}")
        return 1 if koide_artifact.status == "blocked" else 0
    explicit_type = None
    if args.path is not None and args.type != "auto":
        explicit_type = cast(DatasetType, str(args.type).replace("-", "_"))
    artifact = build_doctor_artifact(
        calibrex_version=__version__,
        command=_doctor_command(args),
        path=args.path,
        dataset_type=explicit_type,
        sample_limit=args.sample_limit,
    )
    payload = artifact.model_dump(mode="json")
    if args.output is not None:
        write_mapping(args.output, payload)
    if args.json:
        _emit(payload, as_json=True)
    else:
        print(f"Calibrex {artifact.environment.calibrex_version}")
        print(f"Python {artifact.environment.python_version}")
        for name, dependency in artifact.environment.dependencies.items():
            state = "ok" if dependency.available else "missing"
            optional = " (optional)" if dependency.optional else ""
            version = (
                f" ({dependency.version})" if dependency.available and dependency.version else ""
            )
            print(f"{name}: {state}{optional}{version}")
        if artifact.dataset is not None:
            print(f"Dataset: {artifact.dataset.dataset_type} ({artifact.status.upper()})")
            print(f"  path: {artifact.dataset.path}")
            if artifact.quality is not None:
                print(f"  degeneracy: {artifact.quality.degeneracy.grade.upper()}")
                for recommendation in artifact.quality.recommendations:
                    print(f"  recommendation: {recommendation}")
            for suggestion in artifact.workflow_suggestions:
                print(f"  workflow: {suggestion.workflow_id} ({suggestion.status})")
                print(f"  template: {suggestion.template_path}")
                if suggestion.next_command is not None:
                    print(f"  next:     {suggestion.next_command}")
    return 1 if artifact.status == "fail" else 0


def _doctor_command(args: argparse.Namespace) -> list[str]:
    command = ["calibrex", "doctor"]
    if args.workflow != "environment":
        command.extend(["--workflow", str(args.workflow)])
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    if args.path is not None:
        command.append(str(args.path))
    if args.type != "auto":
        command.extend(["--type", str(args.type)])
    if args.sample_limit is not None:
        command.extend(["--sample-limit", str(args.sample_limit)])
    if args.output is not None:
        command.extend(["--output", str(args.output)])
    if args.json:
        command.append("--json")
    return command


def _cmd_schema(args: argparse.Namespace) -> int:
    generators = _schema_generators()
    if args.kind == "all":
        if args.output is not None:
            _die("schema all does not support --output; use --output-dir")
        if args.output_dir is None:
            _die("schema all requires --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for kind, all_schema_generator in generators.items():
            write_mapping(args.output_dir / _schema_filename(kind), all_schema_generator())
        return 0
    if args.output_dir is not None:
        _die("--output-dir is only valid with schema all")
    schema_generator = generators.get(args.kind)
    if schema_generator is None:
        _die(f"unsupported schema kind: {args.kind}")
    schema = schema_generator()
    if args.output:
        write_mapping(args.output, schema)
    else:
        print(json.dumps(schema, indent=2, sort_keys=True))
    return 0


def _schema_generators() -> dict[str, Callable[[], dict[str, Any]]]:
    generators: dict[str, Callable[[], dict[str, Any]]] = {
        "config": config_json_schema,
        "result": result_json_schema,
        "comparison": comparison_json_schema,
        "report-comparison": report_comparison_json_schema,
        "dynamic-window-consistency": dynamic_window_consistency_json_schema,
        "trajectory-window-drift": trajectory_window_drift_json_schema,
        "capture-readiness": capture_readiness_json_schema,
        "capture-manifest": capture_manifest_json_schema,
        "capture-manifest-verification": capture_manifest_verification_json_schema,
        "mcap-integrity": mcap_integrity_json_schema,
        "koide-readiness": koide_readiness_json_schema,
        "koide-execution-lock": koide_execution_lock_json_schema,
        "continuous-time-lidar-pair": continuous_time_lidar_pair_json_schema,
        "continuous-time-lidar-point-to-plane": (continuous_time_lidar_point_to_plane_json_schema),
        "continuous-time-imu-preintegration": (continuous_time_imu_preintegration_json_schema),
        "continuous-time-imu-lever-arm": continuous_time_imu_lever_arm_json_schema,
        "continuous-time-imu-clock-offset": (continuous_time_imu_clock_offset_json_schema),
        "continuous-time-imu-accel-bias": continuous_time_imu_accel_bias_json_schema,
        "continuous-time-imu-intrinsics": (continuous_time_imu_intrinsics_json_schema),
        "calibration-lifecycle": calibration_lifecycle_json_schema,
        "lifecycle-registry": lifecycle_registry_json_schema,
        "lifecycle-event": lifecycle_event_json_schema,
        "lifecycle-evaluation": lifecycle_evaluation_json_schema,
        "lifecycle-registry-state": lifecycle_registry_state_json_schema,
        "lifecycle-head": lifecycle_head_json_schema,
        "lifecycle-verification": lifecycle_registry_verification_json_schema,
        "lifecycle-status": lifecycle_status_json_schema,
        "continuous-time-sliding-window": continuous_time_sliding_window_json_schema,
        "continuous-time-lidar-train-diagnostics": (
            continuous_time_lidar_train_diagnostics_json_schema
        ),
        "continuous-time-lidar-ablation": continuous_time_lidar_ablation_json_schema,
        "solid-state-cross-dataset-benchmark-config": (
            solid_state_cross_dataset_benchmark_config_json_schema
        ),
        "solid-state-cross-dataset-benchmark": solid_state_cross_dataset_benchmark_json_schema,
        "solid-state-failure-analysis": solid_state_failure_analysis_json_schema,
        "solid-state-synthetic-benchmark": solid_state_synthetic_benchmark_json_schema,
        "solid-state-metrology-evaluation": solid_state_metrology_evaluation_json_schema,
        "assessment": assessment_json_schema,
        "benchmark": benchmark_json_schema,
        "benchmark-definition": benchmark_definition_json_schema,
        "policy": policy_json_schema,
        "protocol": protocol_json_schema,
        "transforms": transform_artifact_json_schema,
        "dataset-manifest": manifest_json_schema,
        "remote-archive-selection": remote_archive_selection_json_schema,
        "kitti360-lidar-window-integration": (kitti360_lidar_window_integration_json_schema),
        "kitti-raw-lidar-window-integration": (kitti_raw_lidar_window_integration_json_schema),
        "doctor": doctor_json_schema,
        "environment-readiness": environment_readiness_json_schema,
        "calibration-ci": calibration_ci_json_schema,
        "external-run": external_run_json_schema,
        "kitti-falsification": kitti_falsification_json_schema,
        "kitti-benchmark-input": kitti_benchmark_input_json_schema,
        "koide-pilot": koide_pilot_json_schema,
        "koide-runner": koide_runner_json_schema,
        "koide-real-pilot-request": koide_real_pilot_request_json_schema,
        "koide-real-pilot-verification": koide_real_pilot_verification_json_schema,
        "koide-real-pilot-finalization": koide_real_pilot_finalization_json_schema,
        "depth-provider": depth_provider_json_schema,
        "continuous-time-camera-lidar-problem": (continuous_time_camera_lidar_problem_json_schema),
        "continuous-time-camera-lidar-result": (continuous_time_camera_lidar_result_json_schema),
        "continuous-time-trajectory": continuous_time_trajectory_json_schema,
        "continuous-time-trajectory-measurements": (continuous_time_measurements_json_schema),
        "continuous-time-trajectory-fit": continuous_time_fit_json_schema,
        "probabilistic-correspondence": probabilistic_correspondence_json_schema,
        "probabilistic-pnp-result": probabilistic_pnp_result_json_schema,
        "probabilistic-refinement-result": (probabilistic_refinement_result_json_schema),
        "empirical-se3-uncertainty": empirical_se3_uncertainty_json_schema,
        "camera-lidar-problem": camera_lidar_problem_json_schema,
        "camera-lidar-correspondence-export": (camera_lidar_correspondence_export_json_schema),
        "camera-lidar-confidence-calibration": (camera_lidar_confidence_calibration_json_schema),
        "camera-lidar-provider-support-comparison": (
            camera_lidar_provider_support_comparison_json_schema
        ),
        "camera-lidar-pose-initializer-protocol": (
            camera_lidar_pose_initializer_protocol_json_schema
        ),
        "camera-lidar-pose-initializer-failure-analysis": (
            camera_lidar_pose_initializer_failure_analysis_json_schema
        ),
        "camera-lidar-initializer-calibration": (camera_lidar_initializer_calibration_json_schema),
        "camera-lidar-correspondence-quality": (camera_lidar_correspondence_quality_json_schema),
        "camera-lidar-failure-analysis": camera_lidar_failure_analysis_json_schema,
        "camera-lidar-sota-audit-protocol": (camera_lidar_sota_audit_protocol_json_schema),
        "camera-lidar-sota-audit-result": (camera_lidar_sota_audit_result_json_schema),
        "camera-lidar-benchmark-protocol": (camera_lidar_benchmark_protocol_json_schema),
        "calibration-candidate-trace": calibration_candidate_trace_json_schema,
        "bullseye-plot": bullseye_plot_json_schema,
        "evidence-bundle": evidence_bundle_json_schema,
        "evidence-bundle-verification": evidence_bundle_verification_json_schema,
        "online-timeline": online_timeline_json_schema,
        "solid-state-context": solid_state_context_json_schema,
        "livox-time-ablation": livox_time_ablation_json_schema,
        "autoware-export": autoware_export_json_schema,
        "autoware-promotion": autoware_promotion_json_schema,
        "autoware-smoke": autoware_smoke_json_schema,
        "raw-replay-definition": replay_definition_json_schema,
        "raw-replay-plan": replay_plan_json_schema,
        "raw-replay-stage": replay_stage_json_schema,
        "raw-replay-result": replay_result_json_schema,
        "raw-replay-comparison": replay_comparison_json_schema,
        "field-replacement-pilot": field_replacement_pilot_json_schema,
        "multi-lidar-service-plan": multi_lidar_service_plan_json_schema,
        "multi-lidar-service-evaluation": multi_lidar_service_evaluation_json_schema,
        "camera-imu-service-plan": camera_imu_service_plan_json_schema,
        "camera-imu-service-evaluation": camera_imu_service_evaluation_json_schema,
        "radar-service-plan": radar_service_plan_json_schema,
        "radar-service-evaluation": radar_service_evaluation_json_schema,
        "trajectory": trajectory_json_schema,
    }
    for kind in report_artifact_schema_kinds():
        generators[kind] = _report_artifact_schema_generator(kind)
    return generators


def _report_artifact_schema_generator(kind: str) -> Callable[[], dict[str, Any]]:
    def generate_schema() -> dict[str, Any]:
        return report_artifact_json_schema(kind)

    return generate_schema


def _schema_filename(kind: str) -> str:
    if kind == "continuous-time-lidar-pair":
        return "continuous_time_lidar_pair_result.schema.json"
    if kind == "koide-runner":
        return "koide_runner_config.schema.json"
    return f"{kind.replace('-', '_')}.schema.json"


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        report = validate_file(
            args.path,
            cast(ValidationKind, args.kind),
            verify_inputs=bool(args.verify_inputs),
        )
    except (OSError, ValueError) as exc:
        _die(str(exc))
    payload = report.model_dump(mode="json")
    if report.kind == "result":
        payload.update(
            {
                "production_valid": report.production_valid,
                "admissibility": report.admissibility,
                "provenance_issues": report.provenance_issues,
            }
        )
    if report.input_verification is not None:
        payload["input_verification"] = report.input_verification.model_dump(
            mode="json",
            exclude_none=False,
        )
    _emit(payload, args.json)
    return 0 if report.valid else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    definition = load_benchmark_definition(args.definition)
    try:
        benchmark = aggregate_benchmark_definition(definition)
    except ValueError as exc:
        raise BenchmarkError(str(exc)) from exc
    benchmark.save(args.output)
    payload: dict[str, Any] = {
        "benchmark": str(args.output),
        "benchmark_id": benchmark.benchmark_id,
        "method_count": len(benchmark.methods),
        "split_count": len(benchmark.protocol.splits),
        "trial_count": len(benchmark.trials),
    }
    if args.markdown_output is not None:
        write_text(args.markdown_output, render_benchmark_markdown(benchmark))
        payload["markdown"] = str(args.markdown_output)
    updated_markdown: list[str] = []
    for markdown_path in args.update_markdown:
        update_benchmark_table_in_markdown(markdown_path, benchmark)
        updated_markdown.append(str(markdown_path))
    if updated_markdown:
        payload["updated_markdown"] = updated_markdown
    _emit(payload, args.json)
    return 0


def _cmd_calibration_ci(args: argparse.Namespace) -> int:
    command = ["calibrex", "ci", str(args.candidate)]
    if args.baseline is not None:
        command.extend(["--baseline", str(args.baseline)])
    if args.policy is not None:
        command.extend(["--policy", str(args.policy)])
    command.extend(["--output-dir", str(args.output_dir)])
    if args.allow_incompatible_protocol:
        command.append("--allow-incompatible-protocol")
    if args.enforce:
        command.append("--enforce")
    if args.json:
        command.append("--json")
    artifact = run_calibration_ci(
        args.candidate,
        output_dir=args.output_dir,
        calibrex_version=__version__,
        command=command,
        baseline_path=args.baseline,
        policy_path=args.policy,
        enforce_protocol=not args.allow_incompatible_protocol,
    )
    payload = artifact.model_dump(mode="json", exclude_none=True)
    payload["ci_artifact"] = str(args.output_dir / "calibration-ci.json")
    _emit(payload, args.json)
    return 1 if args.enforce and artifact.status != "pass" else 0


def _cmd_replay_plan(args: argparse.Namespace) -> int:
    plan = plan_raw_replay(
        args.definition,
        output=args.output,
        command=["calibrex", "replay", "plan", str(args.definition), "--output", str(args.output)],
    )
    _emit({**plan.model_dump(mode="json", exclude_none=False), "plan": str(args.output)}, args.json)
    return 0 if plan.status == "PASS" else 1


def _cmd_replay_run(args: argparse.Namespace) -> int:
    result = run_raw_replay(
        args.definition,
        output_directory=args.output_dir,
        command=[
            "calibrex",
            "replay",
            "run",
            str(args.definition),
            "--output-dir",
            str(args.output_dir),
        ],
    )
    payload: dict[str, Any] = {
        **result.model_dump(mode="json", exclude_none=False),
        "replay_result": str(args.output_dir / "replay-result.json"),
    }
    if args.field_replacement:
        pilot = run_field_replacement_pilot(
            args.definition,
            output_directory=args.output_dir,
            command=["calibrex", "replay", "field-replacement", str(args.definition)],
        )
        payload["field_replacement_pilot"] = pilot.model_dump(mode="json", exclude_none=False)
    _emit(payload, args.json)
    return 0 if result.status == "PASS" else 1


def _cmd_replay_verify(args: argparse.Namespace) -> int:
    result = verify_raw_replay(args.result, definition=args.definition)
    _emit(
        {
            "valid": True,
            "replay_id": result.replay_id,
            "status": result.status,
            "decision": result.decision,
            "artifact_sha256": result.artifact_sha256,
        },
        args.json,
    )
    return 0


def _cmd_replay_compare(args: argparse.Namespace) -> int:
    comparison = compare_raw_replays(args.left, args.right, output=args.output)
    _emit(
        {
            **comparison.model_dump(mode="json", exclude_none=False),
            "comparison": str(args.output),
        },
        args.json,
    )
    return 0 if comparison.status == "PASS" else 1


def _cmd_replay_field_replacement(args: argparse.Namespace) -> int:
    pilot = run_field_replacement_pilot(
        args.definition,
        output_directory=args.output_dir,
        command=["calibrex", "replay", "field-replacement", str(args.definition)],
    )
    _emit(
        {
            **pilot.model_dump(mode="json", exclude_none=False),
            "pilot": str(args.output_dir / "field-replacement-pilot.json"),
        },
        args.json,
    )
    return 0 if pilot.status == "PASS" else 1


def _cmd_multi_lidar_service_plan(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "multi-lidar-service",
        "plan",
        str(args.definition),
        "--output",
        str(args.output),
    ]
    plan = build_multi_lidar_service_plan(args.definition, output=args.output, command=command)
    _emit(
        {
            **plan.model_dump(mode="json", exclude_none=False),
            "plan": str(args.output),
        },
        args.json,
    )
    return 0


def _cmd_multi_lidar_service_evaluate(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "multi-lidar-service",
        "evaluate",
        str(args.plan),
        "--output",
        str(args.output),
    ]
    if args.candidate_replay is not None:
        command.extend(["--candidate-replay", str(args.candidate_replay)])
    evaluation = evaluate_multi_lidar_service(
        args.plan,
        candidate_replay=args.candidate_replay,
        output=args.output,
        evaluation_id=args.evaluation_id,
        command=command,
    )
    _emit(
        {
            **evaluation.model_dump(mode="json", exclude_none=False),
            "evaluation": str(args.output),
        },
        args.json,
    )
    return 0 if evaluation.status == "READY" else 1


def _cmd_multi_lidar_service_verify(args: argparse.Namespace) -> int:
    verification = verify_multi_lidar_service(args.artifact, plan=args.plan)
    _emit(verification.model_dump(mode="json", exclude_none=False), args.json)
    return 0 if verification.valid else 1


def _cmd_camera_imu_service_plan(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "camera-imu-service",
        "plan",
        str(args.definition),
        "--output",
        str(args.output),
    ]
    plan = build_camera_imu_service_plan(args.definition, output=args.output, command=command)
    _emit(
        {
            **plan.model_dump(mode="json", exclude_none=False),
            "plan": str(args.output),
        },
        args.json,
    )
    return 0


def _cmd_camera_imu_service_evaluate(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "camera-imu-service",
        "evaluate",
        str(args.plan),
        "--output",
        str(args.output),
    ]
    if args.candidate_replay is not None:
        command.extend(["--candidate-replay", str(args.candidate_replay)])
    evaluation = evaluate_camera_imu_service(
        args.plan,
        candidate_replay=args.candidate_replay,
        output=args.output,
        evaluation_id=args.evaluation_id,
        command=command,
    )
    _emit(
        {
            **evaluation.model_dump(mode="json", exclude_none=False),
            "evaluation": str(args.output),
        },
        args.json,
    )
    return 0 if evaluation.status == "READY" else 1


def _cmd_camera_imu_service_verify(args: argparse.Namespace) -> int:
    verification = verify_camera_imu_service(args.artifact, plan=args.plan)
    _emit(verification.model_dump(mode="json", exclude_none=False), args.json)
    return 0 if verification.valid else 1


def _cmd_radar_service_plan(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "radar-service",
        "plan",
        str(args.definition),
        "--output",
        str(args.output),
    ]
    plan = build_radar_service_plan(args.definition, output=args.output, command=command)
    _emit(
        {
            **plan.model_dump(mode="json", exclude_none=False),
            "plan": str(args.output),
        },
        args.json,
    )
    return 0


def _cmd_radar_service_evaluate(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "radar-service",
        "evaluate",
        str(args.plan),
        "--output",
        str(args.output),
    ]
    if args.candidate_replay is not None:
        command.extend(["--candidate-replay", str(args.candidate_replay)])
    evaluation = evaluate_radar_service(
        args.plan,
        candidate_replay=args.candidate_replay,
        output=args.output,
        evaluation_id=args.evaluation_id,
        command=command,
    )
    _emit(
        {
            **evaluation.model_dump(mode="json", exclude_none=False),
            "evaluation": str(args.output),
        },
        args.json,
    )
    return 0 if evaluation.status == "READY" else 1


def _cmd_radar_service_verify(args: argparse.Namespace) -> int:
    verification = verify_radar_service(args.artifact, plan=args.plan)
    _emit(verification.model_dump(mode="json", exclude_none=False), args.json)
    return 0 if verification.valid else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    artifact = Path(args.artifact)
    artifact_payload = read_mapping(artifact)
    schema_version = artifact_payload.get("schema_version")
    source_bundle_path: Path | str = artifact
    if schema_version == "slac.evidence_bundle.verification/v0.1":
        source_bundle_ref = artifact_payload.get("source_bundle")
        if isinstance(source_bundle_ref, dict):
            source_bundle_path_value = source_bundle_ref.get("path")
            if isinstance(source_bundle_path_value, str):
                source_bundle_path = _resolve_related_artifact_path(
                    artifact.parent,
                    source_bundle_path_value,
                )
        report = verify_evidence_bundle_verification(
            artifact,
            require_raw_recomputed=args.require_raw_recomputed,
        )
    else:
        report = verify_evidence_bundle(
            artifact,
            require_raw_recomputed=args.require_raw_recomputed,
        )
    payload = report.model_dump(mode="json")
    if args.output:
        report = write_evidence_bundle_verification(
            args.output,
            report,
            source_bundle_path=source_bundle_path,
        )
        payload = report.model_dump(mode="json")
    if args.json:
        _emit(payload, True)
    else:
        _emit_verification(report)
    return 0 if report.valid else 1


def _resolve_related_artifact_path(base_dir: Path, path: str) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path
    return base_dir / artifact_path


def _cmd_assess(args: argparse.Namespace) -> int:
    policy = None
    if args.policy is not None:
        policy_artifact = PolicyArtifact.model_validate(read_mapping(args.policy))
        policy = policy_artifact.policy
    assessment = assess_evidence_file(args.evidence, policy=policy)
    payload = assessment.model_dump(mode="json", exclude_none=True)
    if args.output:
        write_mapping(args.output, payload)
    cli_payload = {
        **payload,
        "enforced": args.enforce,
        "would_fail_enforcement": assessment.status != "pass",
    }
    _emit(cli_payload, args.json)
    return 1 if args.enforce and assessment.status != "pass" else 0


def _cmd_evidence(args: argparse.Namespace) -> int:
    result = load_result(args.result)
    if args.sidecars_dir is None:
        evidence = write_evidence_artifact(result, args.output)
        contract_artifacts: dict[str, str] = {}
    else:
        contract_artifacts = write_evidence_contract_artifacts(
            result,
            evidence_path=args.output,
            output_dir=args.sidecars_dir,
        )
        evidence = evidence_artifact_from_result(result)
    payload: dict[str, Any] = {
        "status": "ok",
        "source_result": str(args.result),
        "evidence": str(args.output),
        "run_id": evidence.run.id,
        "metrics_origin": evidence.materialization.metrics_origin,
        "data_verified": evidence.materialization.data_verified,
        "protocol_count": len(evidence.protocols),
        "case_count": len(evidence.cases),
        "summary_count": len(evidence.summaries),
        "recomputed_metrics": False,
    }
    if args.sidecars_dir is not None:
        payload["sidecars_dir"] = str(args.sidecars_dir)
        payload["contract_artifacts"] = contract_artifacts
    _emit(payload, args.json)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    if args.list_templates:
        payload = {
            "templates": template_summary(),
            "canonical": list_sensor_template_names(),
        }
        if args.json:
            _emit(payload, True)
        else:
            for name in list_sensor_template_names():
                print(name)
        return 0
    if args.template:
        if args.output is None:
            _die("--output is required with --template")
        try:
            written = write_sensor_template(args.template, args.output)
        except CalibrexError as exc:
            _die(str(exc))
        payload = {
            "status": "ok",
            "template": args.template,
            "output": str(written),
        }
        if args.json:
            _emit(payload, True)
        else:
            print(f"wrote {written}")
        return 0
    if args.profile is None:
        _die("profile is required unless --template or --list-templates is used")
    if args.output is None:
        _die("--output is required")
    config = _starter_config(args.profile)
    write_mapping(args.output, config)
    print(f"wrote {args.output}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    if args.online and args.candidate_extrinsics:
        _die(
            "--candidate-extrinsics is not supported with --online; "
            "run the offline `calibrex calibrate` to compare candidate extrinsics"
        )
    config = load_config(args.config)
    if args.online:
        return _cmd_calibrate_online(args, config)
    result = run_calibration(
        args.config,
        CalibrationRunOptions(
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            strict=args.strict,
            seed=args.seed,
            candidate_extrinsics=tuple(args.candidate_extrinsics),
        ),
    )
    if args.dry_run:
        inspection = inspect_dataset(config.dataset)
        payload = {
            "status": "ok",
            "config": str(args.config),
            "dataset": inspection.as_dict(),
        }
        _emit(payload, args.json)
        return 0
    if result is None:
        _die("calibration did not produce a result")
    output_dir = args.output_dir or config.output_dir
    artifacts = report_artifact_paths(output_dir, html_filename=config.outputs.report)
    summary: dict[str, Any] = {
        "status": result.run.status,
        "grade": result.quality.grade,
        "run_id": result.run.id,
        "result": str(output_dir / config.outputs.result),
        "report": artifacts["html_report"],
        "report_artifacts": artifacts,
    }
    _emit(summary, args.json)
    return 1 if args.strict and result.quality.grade != "pass" else 0


def _cmd_calibrate_online(args: argparse.Namespace, config: CalibrationConfig) -> int:
    result = run_online_calibration(
        args.config,
        OnlineCalibrationRunOptions(
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            strict=args.strict,
            seed=args.seed if args.seed is not None else 0,
            batch_size=args.batch_size,
            rolling_window=args.rolling_window,
            holdout_ratio=args.holdout_ratio,
            accumulation_batches=args.accumulation_batches,
            max_accumulated_train_points=args.max_accumulated_train_points,
        ),
    )
    if args.dry_run:
        inspection = inspect_dataset(config.dataset)
        payload = {
            "status": "ok",
            "config": str(args.config),
            "dataset": inspection.as_dict(),
        }
        _emit(payload, args.json)
        return 0
    if result is None:
        _die("online calibration did not produce a result")
    output_dir = args.output_dir or config.output_dir
    artifacts = report_artifact_paths(output_dir, html_filename=config.outputs.report)
    summary: dict[str, Any] = {
        "status": result.run.status,
        "grade": result.quality.grade,
        "run_id": result.run.id,
        "result": str(output_dir / config.outputs.result),
        "report": artifacts["html_report"],
        "report_artifacts": artifacts,
        "timeline": result.run.provenance.get("online_timeline_path"),
        "trajectory": result.run.provenance.get("trajectory_path"),
        "batch_count": result.run.provenance.get("online_batch_count"),
        "accepted_batch_count": result.run.provenance.get("online_accepted_batch_count"),
        "rejected_batch_count": result.run.provenance.get("online_rejected_batch_count"),
        "final_gate_status": result.run.provenance.get("online_final_gate_status"),
    }
    _emit(summary, args.json)
    return 1 if args.strict and result.quality.grade != "pass" else 0


def _cmd_compile(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    inspection = inspect_dataset(config.dataset)
    problem = build_problem(config, FrameGraph.from_config(config), inspection)
    payload = problem.as_dict()
    if args.output:
        write_mapping(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.output:
        _emit(problem.summary(), as_json=False)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    profile = cast(ThresholdProfile | None, args.threshold_profile)
    result = evaluate_quality(
        load_result(args.result),
        strict=args.strict,
        threshold_profile=profile,
    )
    if args.export_html:
        output_dir = args.output_dir or args.result.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        write_camera_lidar_overlay_artifact(result, output_dir / "artifacts")
        write_rig_3d_artifact(result, output_dir / "artifacts")
        report_artifacts = write_report_artifacts(result, output_dir)
    else:
        report_artifacts = {}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / args.result.name
        result.save(output)
    payload = {
        "grade": result.quality.grade,
        "warnings": result.quality.warnings,
        "blocking_failures": result.quality.blocking_failures,
        "recommendation": result.quality.recommendation,
        "html_report": result.artifacts.html_report,
        "report_artifacts": report_artifacts,
        "metrics_origin": result.run.provenance.get("metrics_origin", "unknown"),
        "data_verified": result.run.provenance.get("data_verified"),
        "evidence_case_count": len(evidence_cases_from_result(result)),
    }
    warning = _report_materialization_warning(
        payload["metrics_origin"],
        payload["data_verified"],
        command="evaluate",
    )
    if warning is not None:
        payload["warning"] = warning
    _emit(payload, args.json)
    return 1 if result.quality.grade == "fail" else 0


def _cmd_render(args: argparse.Namespace) -> int:
    if args.format == "evidence-card":
        if args.html is not None:
            _die("--html cannot be used with --format evidence-card")
        result = load_result(args.result)
        output_path = _evidence_card_output_path(args.result, args.output_dir, args.output)
        write_evidence_card(result, output_path, source_path=args.result)
        _emit(
            {
                "status": "ok",
                "command": "render",
                "render_only": True,
                "recomputed_metrics": False,
                "source_result": str(args.result),
                "output_format": "evidence-card",
                "evidence_card": str(output_path),
                "source_sha256_bound": True,
            },
            args.json,
        )
        return 0
    if args.output is not None:
        _die("--output is only supported with --format evidence-card")
    return _render_result_report(args, command="render", deprecated_alias=None)


def _cmd_report(args: argparse.Namespace) -> int:
    return _render_result_report(args, command="report", deprecated_alias="report")


def _render_result_report(
    args: argparse.Namespace,
    *,
    command: str,
    deprecated_alias: str | None,
) -> int:
    result = load_result(args.result)
    output_dir = args.output_dir or _report_output_dir(args.result, args.html)
    html_filename = _report_html_filename(args.html)
    metrics_origin = result.run.provenance.get("metrics_origin", "unknown")
    data_verified = result.run.provenance.get("data_verified")
    report_artifacts = write_report_artifacts(
        result,
        output_dir,
        html_filename=html_filename,
    )
    payload = {
        "status": "ok",
        "command": command,
        "canonical_command": "render",
        "deprecated_alias": deprecated_alias,
        "render_only": True,
        "recomputed_metrics": False,
        "source_result": str(args.result),
        "output_format": "html",
        "html_report": report_artifacts["html_report"],
        "report_artifacts": report_artifacts,
        "metrics_origin": metrics_origin,
        "data_verified": data_verified,
        "evidence_case_count": len(evidence_cases_from_result(result)),
    }
    warning = _report_materialization_warning(metrics_origin, data_verified, command=command)
    if warning is not None:
        payload["warning"] = warning
    _emit(payload, args.json)
    return 0


def _report_materialization_warning(
    metrics_origin: object,
    data_verified: object,
    *,
    command: str,
) -> str | None:
    if metrics_origin == "cached" or data_verified is False:
        return f"cached evidence: raw data was not read or recomputed by this {command} command"
    return None


def _report_output_dir(result_path: Path, html_path: Path | None) -> Path:
    if html_path is not None and html_path.is_absolute():
        return html_path.parent
    if html_path is not None and html_path.parent != Path("."):
        return html_path.parent
    return result_path.parent


def _report_html_filename(html_path: Path | None) -> str | Path:
    if html_path is None:
        return "report.html"
    if html_path.is_absolute():
        return html_path
    return html_path.name


def _evidence_card_output_path(
    result_path: Path,
    output_dir: Path | None,
    output_path: Path | None,
) -> Path:
    if output_path is not None and (output_path.is_absolute() or output_path.parent != Path(".")):
        return output_path
    base = output_dir or result_path.parent
    if output_path is None:
        return base / "evidence-card.svg"
    return base / output_path


def _cmd_compare(args: argparse.Namespace) -> int:
    left_result = load_result(args.left_result)
    right_result = load_result(args.right_result)
    comparison = compare_results(
        left_result,
        right_result,
        left_path=args.left_result,
        right_path=args.right_result,
    )
    if args.format == "evidence-table":
        output_path = args.output or Path("comparison-table.svg")
        write_comparison_table(
            left_result,
            right_result,
            output_path,
            left_path=args.left_result,
            right_path=args.right_result,
            left_label=args.left_label,
            right_label=args.right_label,
        )
        _emit(
            {
                "status": "ok",
                "command": "compare",
                "output_format": "evidence-table",
                "comparison_table": str(output_path),
                "left_source": str(args.left_result),
                "right_source": str(args.right_result),
                "source_sha256_bound": True,
                "protocol_compatibility": comparison.protocol_compatibility.status,
            },
            args.json,
        )
        if args.enforce_compatible and comparison.protocol_compatibility.status != "compatible":
            return 1
        return 0
    payload = comparison.model_dump(mode="json", exclude_none=True)
    if args.output:
        write_mapping(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.output:
        _emit_comparison(
            comparison,
            enforce_compatible=args.enforce_compatible,
        )
    if args.enforce_compatible and comparison.protocol_compatibility.status != "compatible":
        return 1
    return 0


def _cmd_report_compare(args: argparse.Namespace) -> int:
    labeled_paths = _parse_labeled_results(args.entries)
    labeled_results = [(label, load_result(path)) for label, path in labeled_paths]
    paths = {label: cast(Path | str | None, path) for label, path in labeled_paths}
    try:
        report = compare_reports(
            labeled_results,
            paths=paths,
            reference_label=args.reference,
        )
    except ValueError as exc:
        _die(str(exc))
    payload = report.model_dump(mode="json", exclude_none=True)
    if args.output:
        write_mapping(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.output:
        _emit_report_comparison(report, enforce_compatible=args.enforce_compatible)
    if args.enforce_compatible and report.summary.protocol_compatibility_status != "compatible":
        return 1
    return 0


def _cmd_window_consistency(args: argparse.Namespace) -> int:
    labeled_paths = _parse_labeled_results(args.entries)
    labeled_results = [(label, load_result(path)) for label, path in labeled_paths]
    try:
        artifact = evaluate_dynamic_window_consistency(
            labeled_results,
            paths=dict(labeled_paths),
            transform_id=args.transform,
            thresholds=DynamicWindowConsistencyThresholds(
                max_translation_delta_m=args.max_translation_delta_m,
                max_rotation_delta_deg=args.max_rotation_delta_deg,
            ),
            reference_label=args.reference,
        )
    except ValueError as exc:
        _die(str(exc))
    payload = artifact.model_dump(mode="json", exclude_none=True)
    if args.output:
        write_mapping(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.output:
        print(
            f"window consistency: {artifact.grade} "
            f"({len(artifact.blocking_failures)} blocking failures, "
            f"{len(artifact.warnings)} warnings)"
        )
    if args.enforce and artifact.grade != "pass":
        return 1
    return 0


def _cmd_trajectory_window_drift(args: argparse.Namespace) -> int:
    artifact = evaluate_rosbag2_trajectory_window_drift(
        args.config,
        reference_result_path=args.reference_result,
        use_point_time_offsets=(None if args.deskew == "config" else args.deskew == "on"),
        odometry_burst_policy=(
            None if args.odometry_burst_policy == "config" else args.odometry_burst_policy
        ),
        odometry_burst_min_interval_s=args.odometry_burst_min_interval_s,
    )
    payload = artifact.model_dump(mode="json", exclude_none=True)
    if args.output:
        write_mapping(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.output:
        print(f"trajectory window drift: {artifact.grade} ({artifact.interpretation})")
    return 0


def _cmd_capture_readiness(args: argparse.Namespace) -> int:
    artifact = evaluate_rosbag2_capture_readiness(args.config)
    payload = artifact.model_dump(mode="json", exclude_none=True)
    if args.output:
        write_mapping(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"calibration readiness: {artifact.grade.upper()} "
            f"({artifact.decision}); {artifact.summary}"
        )
        for recommendation in artifact.recommendations:
            print(f"  next: {recommendation.message}")
        if args.output:
            print(f"  artifact: {args.output}")
    return 1 if args.enforce and artifact.grade != "pass" else 0


def _cmd_continuous_time_lidar_pair(args: argparse.Namespace) -> int:
    artifact = evaluate_continuous_time_lidar_pair(args.config)
    payload = artifact.model_dump(mode="json", exclude_none=True)
    if args.output:
        write_mapping(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"continuous-time LiDAR pair: {artifact.status} "
            f"offset={artifact.estimated_time_offset_sec:.6f}s "
            f"train_rmse={artifact.final_train_rmse_m}"
        )
        print(
            f"  holdout_rmse={artifact.final_holdout_rmse_m} "
            f"rank={artifact.observability.rank} "
            f"condition={artifact.observability.condition_number}"
        )
        print(
            f"  correspondences=train:{artifact.train_correspondence_count} "
            f"holdout:{artifact.holdout_correspondence_count}; "
            f"outliers_rejected={artifact.outlier_rejected_count}"
        )
        print(
            f"  method=voxel:{artifact.options.voxel_strategy} "
            f"outliers:{artifact.options.outlier_policy}"
        )
        print(f"  reason: {artifact.reason}")
        if args.output:
            print(f"  artifact: {args.output}")
    enforce_ok = (
        artifact.status == "converged"
        and artifact.observability.rank is not None
        and artifact.observability.rank >= 6
        and artifact.final_holdout_rmse_m is not None
        and artifact.holdout_correspondence_count >= artifact.options.min_correspondences
    )
    return 1 if args.enforce and not enforce_ok else 0


def _parse_labeled_results(entries: list[str]) -> list[tuple[str, Path]]:
    labeled: list[tuple[str, Path]] = []
    for entry in entries:
        label, separator, path = entry.partition("=")
        if not separator or not label or not path:
            _die(f"report-compare entries must use LABEL=RESULT syntax, got: {entry}")
        labeled.append((label, Path(path)))
    if len(labeled) < 2:
        _die("report-compare requires at least two LABEL=RESULT entries")
    labels = [label for label, _ in labeled]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        _die(f"duplicate report-compare labels: {', '.join(duplicates)}")
    return labeled


def _cmd_metrics(args: argparse.Namespace) -> int:
    definitions = [
        {
            "name": definition.name,
            "description": definition.description,
            "unit": definition.unit,
            "family": definition.family,
        }
        for definition in list_metric_definitions()
    ]
    if args.json:
        print(json.dumps({"metrics": definitions}, indent=2, sort_keys=True))
    else:
        for definition in definitions:
            unit = f" [{definition['unit']}]" if definition["unit"] else ""
            print(f"{definition['name']}{unit}: {definition['description']}")
    return 0


def _cmd_public_datasets_list(args: argparse.Namespace) -> int:
    catalog = load_public_dataset_catalog()
    payload = {
        "datasets": [
            {
                "id": dataset_id,
                "name": entry.name,
                "family": entry.family,
                "domain": entry.domain,
                "recommended_pipeline": entry.recommended_pipeline,
                "slac_config": entry.slac_config,
            }
            for dataset_id, entry in sorted(catalog.datasets.items())
        ]
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for entry in payload["datasets"]:
            print(
                f"{entry['id']}: {entry['name']} "
                f"({entry['domain']}, {entry['recommended_pipeline']})"
            )
    return 0


def _cmd_public_datasets_show(args: argparse.Namespace) -> int:
    catalog = load_public_dataset_catalog()
    try:
        entry = catalog.datasets[args.dataset]
    except KeyError as exc:
        _die(f"unknown public dataset: {args.dataset}")
        raise AssertionError from exc
    payload = entry.model_dump(mode="json")
    payload["id"] = args.dataset
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _cmd_demo_livox_evidence(args: argparse.Namespace) -> int:
    dataset_path = livox_horizon_horizon_pcd_sample_path(args.data_dir)
    if args.no_download:
        _require_livox_demo_files(dataset_path)
        downloaded = False
    else:
        downloaded_result = download_livox_horizon_horizon_pcd_sample(args.data_dir)
        dataset_path = downloaded_result.path
        downloaded = downloaded_result.downloaded

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_config = output_dir / "demo_config.yaml"
    _write_demo_config(args.config, demo_config, dataset_path, output_dir)

    result = run_calibration(
        demo_config,
        CalibrationRunOptions(output_dir=output_dir, seed=args.seed),
    )
    if result is None:
        _die("Livox evidence demo did not produce a result")

    evidence_path = output_dir / "evidence.json"
    assessment_path = output_dir / "assessment.json"
    bundle_path = output_dir / "bundle.json"
    assessment = AssessmentArtifact.model_validate(read_mapping(assessment_path))
    verification = verify_evidence_bundle(bundle_path, require_raw_recomputed=True)
    verification_path = output_dir / "verification.json"
    verification = write_evidence_bundle_verification(
        verification_path,
        verification,
        source_bundle_path=bundle_path,
    )
    payload = {
        "status": "ok" if verification.valid else "invalid_bundle",
        "dataset": str(dataset_path),
        "downloaded": downloaded,
        "config": str(demo_config),
        "result": str(output_dir / "result.yaml"),
        "evidence": str(evidence_path),
        "assessment": str(assessment_path),
        "assessment_status": assessment.status,
        "assessment_reason": assessment.reason,
        "bundle": str(bundle_path),
        "verification": str(verification_path),
        "bundle_valid": verification.valid,
        "bundle_issues": verification.issues,
        "raw_recomputed_required": verification.raw_recomputed_required,
        "html_report": str(output_dir / "report.html"),
    }
    _emit(payload, args.json)
    if not verification.valid:
        return 1
    if args.strict_assessment and assessment.status != "pass":
        return 1
    return 0


def _require_livox_demo_files(dataset_path: Path) -> None:
    missing = [
        name
        for name in (LIVOX_BASE_PCD_NAME, LIVOX_TARGET_PCD_NAME)
        if not (dataset_path / name).exists()
    ]
    if missing:
        _die(
            "Livox demo data is missing: "
            f"{', '.join(missing)} under {dataset_path}. "
            "Run without --no-download or use tools/download_public_dataset.py."
        )


def _write_demo_config(
    source_config: Path,
    output_config: Path,
    dataset_path: Path,
    output_dir: Path,
) -> None:
    config_payload = read_mapping(source_config)
    dataset = config_payload.get("dataset")
    if not isinstance(dataset, dict):
        _die(f"{source_config} does not contain a dataset mapping")
    dataset["path"] = str(dataset_path)
    project = config_payload.get("project")
    if isinstance(project, dict):
        project["output_dir"] = str(output_dir)
    write_mapping(output_config, config_payload)


def _cmd_demo_kitti_lidar_camera_evidence(args: argparse.Namespace) -> int:
    config_payload = read_mapping(args.config)
    dataset_section = config_payload.get("dataset")
    if not isinstance(dataset_section, dict) or "path" not in dataset_section:
        _die(f"{args.config} does not contain a dataset mapping")
    if args.dataset_path is not None:
        dataset_path = Path(args.dataset_path)
    else:
        configured_dataset_path = Path(str(dataset_section["path"]))
        config_relative_path = args.config.parent / configured_dataset_path
        dataset_path = (
            configured_dataset_path
            if configured_dataset_path.is_absolute() or configured_dataset_path.exists()
            else config_relative_path
        )
    _require_kitti_lidar_camera_demo_files(dataset_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_config = output_dir / "demo_config.yaml"
    _write_demo_config(args.config, demo_config, dataset_path, output_dir)

    result = run_calibration(
        demo_config,
        CalibrationRunOptions(output_dir=output_dir, seed=args.seed),
    )
    if result is None:
        _die("KITTI camera-LiDAR evidence demo did not produce a result")

    evidence_path = output_dir / "evidence.json"
    assessment_path = output_dir / "assessment.json"
    bundle_path = output_dir / "bundle.json"
    assessment = AssessmentArtifact.model_validate(read_mapping(assessment_path))
    verification = verify_evidence_bundle(bundle_path, require_raw_recomputed=True)
    verification_path = output_dir / "verification.json"
    verification = write_evidence_bundle_verification(
        verification_path,
        verification,
        source_bundle_path=bundle_path,
    )
    payload = {
        "status": "ok" if verification.valid else "invalid_bundle",
        "dataset": str(dataset_path),
        "config": str(demo_config),
        "result": str(output_dir / "result.yaml"),
        "evidence": str(evidence_path),
        "assessment": str(assessment_path),
        "assessment_status": assessment.status,
        "assessment_reason": assessment.reason,
        "bundle": str(bundle_path),
        "verification": str(verification_path),
        "bundle_valid": verification.valid,
        "bundle_issues": verification.issues,
        "raw_recomputed_required": verification.raw_recomputed_required,
        "html_report": str(output_dir / "report.html"),
        "note": (
            "camera-LiDAR metrics are diagnostic overlay evidence on the LiDAR "
            "candidate extrinsic, scored against the calib_velo_to_cam.txt "
            "dataset reference; this is not a standalone camera calibration. "
            "Selected camera, Velodyne, and calibration inputs are SHA-256 "
            "bound and raw-recomputation verification is enforced."
        ),
    }
    _emit(payload, args.json)
    if not verification.valid:
        return 1
    if args.strict_assessment and assessment.status != "pass":
        return 1
    return 0


def _require_kitti_lidar_camera_demo_files(dataset_path: Path) -> None:
    required_paths = {
        "calib_velo_to_cam.txt": dataset_path.parent / "calib_velo_to_cam.txt",
        "calib_cam_to_cam.txt": dataset_path.parent / "calib_cam_to_cam.txt",
        "image_02/data": dataset_path / "image_02" / "data",
        "velodyne_points/data": dataset_path / "velodyne_points" / "data",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        _die(
            "KITTI camera-LiDAR demo data is missing: "
            f"{', '.join(missing)} under {dataset_path} (and its parent). "
            "Point --dataset-path at a local KITTI raw sequence obtained through "
            "the official KITTI raw download flow, or use the small synthetic "
            "fixture bundled with the example config."
        )


def _cmd_kitti_import_calib(args: argparse.Namespace) -> int:
    transforms = {
        name: {
            "convention": "T_parent_child",
            **transform.as_dict(),
        }
        for name, transform in sorted(read_kitti_initial_transforms(args.path).items())
    }
    payload = {"transforms": transforms}
    if args.output:
        write_mapping(args.output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif not args.output:
        _emit(payload, as_json=False)
    return 0


def _cmd_demo_kitti_falsification_benchmark(args: argparse.Namespace) -> int:
    try:
        benchmark = run_kitti_falsification_benchmark(
            args.dataset_path,
            config_path=args.config,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            projection_sample_points=args.projection_sample_points,
            seed=args.seed,
            source_url=args.source_url,
            source_note=args.source_note,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _die(str(exc))
    payload = {
        "status": benchmark.status,
        "falsification_passed": benchmark.status == "pass",
        "reason": benchmark.reason,
        "benchmark": str(Path(args.output_dir) / "benchmark.json"),
        "dataset": benchmark.dataset_path,
        "selected_frame_count": len(benchmark.selected_frame_ids),
        "trials": {
            trial.candidate_id: {
                "assessment_status": trial.assessment_status,
                "bundle_valid": trial.bundle_valid,
            }
            for trial in benchmark.trials
        },
    }
    _emit(payload, args.json)
    return 1 if args.enforce and benchmark.status != "pass" else 0


def _cmd_external_run_import_kalibr(args: argparse.Namespace) -> int:
    if args.source.resolve() == args.output.resolve():
        _die("external-run output must not overwrite the Kalibr source artifact")
    license_spdx = (
        None
        if str(args.license_spdx).strip().lower() == "unknown"
        else str(args.license_spdx).strip()
    )
    artifact = import_kalibr_camchain(
        args.source,
        input_artifacts=tuple(args.input_artifact),
        expected_source_sha256=args.expected_source_sha256,
        tool_version=args.tool_version,
        source_commit=args.source_commit,
        license_spdx=license_spdx,
        training_isolation_declared=args.training_isolation_declared,
        training_isolation_evidence=args.training_isolation_evidence,
    )
    artifact.save(args.output)
    _emit(
        {
            "status": artifact.status,
            "external_run": str(args.output),
            "source": str(args.source),
            "source_sha256": artifact.provenance.source_artifact_sha256,
            "transform_count": len(artifact.parsed_outputs.transforms),
            "time_offset_count": len(artifact.parsed_outputs.time_offsets_seconds),
            "camera_count": len(artifact.parsed_outputs.intrinsics),
            "warnings": artifact.warnings,
        },
        args.json,
    )
    return 0 if artifact.status == "success" else 1


def _cmd_external_run_import_koide(args: argparse.Namespace) -> int:
    """Import one Koide native result through the typed external-run boundary."""

    if args.source.resolve() == args.output.resolve():
        _die("external-run output must not overwrite the Koide source artifact")
    license_spdx = (
        None
        if str(args.license_spdx).strip().lower() == "unknown"
        else str(args.license_spdx).strip()
    )
    artifact = import_koide_result(
        args.source,
        lidar_frame=args.lidar_frame,
        camera_frame=args.camera_frame,
        input_artifacts=tuple(args.input_artifact),
        expected_source_sha256=args.expected_source_sha256,
        tool_version=args.tool_version,
        source_commit=args.source_commit,
        license_spdx=license_spdx,
        training_isolation_declared=args.training_isolation_declared,
        training_isolation_evidence=args.training_isolation_evidence,
        training_data_ids_sha256=args.training_data_ids_sha256,
        holdout_data_ids_sha256=args.holdout_data_ids_sha256,
    )
    artifact.save(args.output)
    _emit(
        {
            "status": artifact.status,
            "external_run": str(args.output),
            "source": str(args.source),
            "source_sha256": artifact.provenance.source_artifact_sha256,
            "transform_count": len(artifact.parsed_outputs.transforms),
            "warnings": artifact.warnings,
        },
        args.json,
    )
    return 0 if artifact.status == "success" else 1


def _cmd_external_run_koide(args: argparse.Namespace) -> int:
    """Run the typed Koide workflow declared by a Calibrex config."""

    if args.config.resolve() == args.output.resolve():
        _die("Koide external-run output must not overwrite the config")
    config = load_config(args.config)
    # ``--input-artifact`` is a CLI-level provenance input.  Preserve the
    # factor's declared paths and append these explicit files before the
    # solver builds its typed runner config, so they are actually digest-bound
    # in the resulting external-run artifact.
    if args.input_artifact:
        for factor_name in (
            "koide_lidar_camera",
            "direct_visual_lidar_calibration",
            "lidar_camera_targetless_baseline",
        ):
            factor = config.pipeline.factors.get(factor_name)
            if factor is None or not factor.enabled:
                continue
            raw_paths = factor.options.get(
                "input_paths",
                factor.options.get("input_artifacts", []),
            )
            if isinstance(raw_paths, (str, Path)):
                paths = [str(raw_paths)]
            elif isinstance(raw_paths, (list, tuple)):
                paths = [str(path) for path in raw_paths]
            else:
                paths = []
            for path in args.input_artifact:
                text_path = str(path)
                if text_path not in paths:
                    paths.append(text_path)
            factor.options["input_paths"] = paths
            break
    inspection = inspect_dataset(config.dataset)
    result = KoideLidarCameraSolver().solve(
        config,
        FrameGraph.from_config(config),
        inspection,
    )
    if result.external_run is None:
        _die("Koide factor did not produce an external-run artifact")
    result.external_run.save(args.output)
    _emit(
        {
            "status": result.external_run.status,
            "external_run": str(args.output),
            "transform_count": len(result.external_run.parsed_outputs.transforms),
            "stage_count": len(result.external_run.execution.stages),
            "warnings": result.external_run.warnings,
        },
        args.json,
    )
    return 0 if result.external_run.status == "success" else 1


def _cmd_kitti_lock_benchmark_input(args: argparse.Namespace) -> int:
    command = f"calibrex kitti lock-benchmark-input {args.path} --output {args.output}"
    manifest = build_kitti_raw_0005_benchmark_input(args.path, command=command)
    manifest.save(args.output)
    payload = {
        "status": "ok",
        "dataset_id": manifest.dataset_id,
        "sequence_id": manifest.sequence_id,
        "frame_count": len(manifest.frame_ids),
        "file_count": len(manifest.files),
        "input_sha256": manifest.input_sha256,
        "output": str(args.output),
    }
    _emit(payload, args.json)
    return 0


def _cmd_kitti_benchmark_i2i(args: argparse.Namespace) -> int:
    if args.max_iterations < 0:
        _die("--max-iterations must be non-negative")
    command = (
        f"calibrex kitti benchmark-i2i {args.input_manifest} "
        f"--definition-output {args.definition_output} --output {args.output}"
    )
    data = load_kitti_i2i_benchmark_data(
        args.input_manifest,
        sequence_path=args.sequence_path,
        max_points_per_frame=args.max_points_per_frame,
    )
    definition, benchmark = run_kitti_i2i_recovery_benchmark(
        data,
        command=command,
        max_iterations=args.max_iterations,
    )
    definition.save(args.definition_output)
    benchmark.save(args.output)
    coarse = benchmark.method_summaries["pandey_i2i_vectorized_safe_coarse_bb_v03"]
    baseline = benchmark.method_summaries["pandey_i2i_scalar_bb_v01"]
    baseline_runtime = baseline.runtime_seconds.mean
    coarse_runtime = coarse.runtime_seconds.mean
    runtime_speedup = (
        baseline_runtime / coarse_runtime
        if baseline_runtime is not None and coarse_runtime is not None and coarse_runtime > 0.0
        else None
    )
    baseline_trials = {
        trial.split_id: trial
        for trial in benchmark.trials
        if trial.method_id == "pandey_i2i_scalar_bb_v01"
    }
    coarse_trials = {
        trial.split_id: trial
        for trial in benchmark.trials
        if trial.method_id == "pandey_i2i_vectorized_safe_coarse_bb_v03"
    }
    accuracy_non_degraded = all(
        split_id in coarse_trials
        and baseline_trial.status == coarse_trials[split_id].status == "success"
        and coarse_trials[split_id].metrics["translation_error_m"]
        <= baseline_trial.metrics["translation_error_m"] + 1.0e-12
        and coarse_trials[split_id].metrics["rotation_error_deg"]
        <= baseline_trial.metrics["rotation_error_deg"] + 1.0e-12
        and coarse_trials[split_id].metrics["holdout_normalized_mutual_information"] + 1.0e-12
        >= baseline_trial.metrics["holdout_normalized_mutual_information"]
        and coarse_trials[split_id].metrics["recovered"] + 1.0e-12
        >= baseline_trial.metrics["recovered"]
        for split_id, baseline_trial in baseline_trials.items()
    )
    runtime_improved = runtime_speedup is not None and runtime_speedup > 1.0
    payload = {
        "status": "ok",
        "definition": str(args.definition_output),
        "benchmark": str(args.output),
        "input_sha256": data.manifest.input_sha256,
        "trial_count": len(benchmark.trials),
        "baseline_failure_rate": baseline.failure_rate,
        "coarse_failure_rate": coarse.failure_rate,
        "baseline_runtime_mean_s": baseline_runtime,
        "coarse_runtime_mean_s": coarse_runtime,
        "runtime_speedup": runtime_speedup,
        "accuracy_non_degraded": accuracy_non_degraded,
        "runtime_improved": runtime_improved,
        "performance_gate_passed": accuracy_non_degraded and runtime_improved,
    }
    _emit(payload, args.json)
    return 0


def _cmd_camera_lidar_freeze_rotation_protocol(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "freeze-rotation-protocol",
        str(args.problem),
        "--output",
        str(args.output),
        "--perturbation-count",
        str(args.perturbation_count),
        "--rotation-deg",
        str(args.rotation_deg),
        "--histogram-bins",
        str(args.histogram_bins),
        "--min-visible-points",
        str(args.min_visible_points),
        "--bound-deg",
        str(args.bound_deg),
        "--initial-step-deg",
        str(args.initial_step_deg),
        "--minimum-step-deg",
        str(args.minimum_step_deg),
        "--max-evaluations",
        str(args.max_evaluations),
    ]
    try:
        protocol = build_borer_rotation_protocol(
            args.problem,
            perturbation_count=args.perturbation_count,
            rotation_magnitude_deg=args.rotation_deg,
            histogram_bins=args.histogram_bins,
            min_visible_points=args.min_visible_points,
            bound_deg=args.bound_deg,
            initial_step_deg=args.initial_step_deg,
            minimum_step_deg=args.minimum_step_deg,
            max_evaluations=args.max_evaluations,
            command=tuple(command),
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    protocol.save(args.output)
    _emit(
        {
            "status": "ok",
            "protocol_id": protocol.protocol_id,
            "problem_sha256": protocol.problem_sha256,
            "frame_count": len(protocol.frame_ids),
            "perturbation_count": protocol.perturbation_count,
            "rotation_magnitude_deg": protocol.rotation_magnitude_deg,
            "output": str(args.output),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_freeze_six_dof_protocol(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "freeze-six-dof-protocol",
        str(args.problem),
        "--output",
        str(args.output),
        "--perturbation-count",
        str(args.perturbation_count),
        "--rotation-deg",
        str(args.rotation_deg),
        "--translation-m",
        str(args.translation_m),
        "--histogram-bins",
        str(args.histogram_bins),
        "--min-visible-points",
        str(args.min_visible_points),
        "--rotation-bound-deg",
        str(args.rotation_bound_deg),
        "--translation-bound-m",
        str(args.translation_bound_m),
        "--initial-rotation-step-deg",
        str(args.initial_rotation_step_deg),
        "--initial-translation-step-m",
        str(args.initial_translation_step_m),
        "--minimum-rotation-step-deg",
        str(args.minimum_rotation_step_deg),
        "--minimum-translation-step-m",
        str(args.minimum_translation_step_m),
        "--max-evaluations",
        str(args.max_evaluations),
    ]
    try:
        protocol = build_borer_six_dof_protocol(
            args.problem,
            perturbation_count=args.perturbation_count,
            rotation_magnitude_deg=args.rotation_deg,
            translation_magnitude_m=args.translation_m,
            histogram_bins=args.histogram_bins,
            min_visible_points=args.min_visible_points,
            rotation_bound_deg=args.rotation_bound_deg,
            translation_bound_m=args.translation_bound_m,
            initial_rotation_step_deg=args.initial_rotation_step_deg,
            initial_translation_step_m=args.initial_translation_step_m,
            minimum_rotation_step_deg=args.minimum_rotation_step_deg,
            minimum_translation_step_m=args.minimum_translation_step_m,
            max_evaluations=args.max_evaluations,
            command=tuple(command),
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    protocol.save(args.output)
    _emit(
        {
            "status": "ok",
            "protocol_id": protocol.protocol_id,
            "problem_sha256": protocol.problem_sha256,
            "frame_count": len(protocol.frame_ids),
            "perturbation_count": protocol.perturbation_count,
            "rotation_magnitude_deg": protocol.rotation_magnitude_deg,
            "translation_magnitude_m": protocol.translation_magnitude_m,
            "output": str(args.output),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_benchmark_koide_pilot(args: argparse.Namespace) -> int:
    from calibrex.evaluation.koide_pilot import KoidePilotThresholds

    try:
        thresholds = KoidePilotThresholds(
            max_frames=args.max_frames,
            max_points=args.max_points,
            holdout_ratio=args.holdout_ratio,
            split_seed=args.split_seed,
            min_holdout_projection_ratio=args.min_holdout_projection_ratio,
            min_holdout_edge_alignment=args.min_holdout_edge_alignment,
            min_holdout_depth_edge_alignment=args.min_holdout_depth_edge_alignment,
            known_bad_min_metric_delta=args.known_bad_min_metric_delta,
        )
        artifact = run_koide_pilot(
            args.dataset_path,
            args.candidate,
            output_directory=args.output_dir,
            config_path=args.config,
            readiness_path=args.readiness,
            input_manifest_path=args.input_manifest,
            camera_frame=args.camera_frame,
            lidar_frame=args.lidar_frame,
            camera_stream=args.camera_stream,
            lidar_stream=args.lidar_stream,
            thresholds=thresholds,
            command=[
                "calibrex",
                "camera-lidar",
                "benchmark-koide-pilot",
                str(args.dataset_path),
            ],
            official_command=(shlex.split(args.official_command) if args.official_command else ()),
            tool_source_commit=args.tool_source_commit,
            container_digest=args.container_digest,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _die(str(exc))
    payload = {
        "status": artifact.status,
        "adoption_decision": artifact.adoption_decision,
        "admissible": artifact.admissible,
        "reason": artifact.reason,
        "pilot": str(args.output_dir / "pilot.json"),
        "execution_manifest": str(args.output_dir / "execution-manifest.yaml"),
        "known_bad_detected_count": artifact.known_bad_detected_count,
        "known_bad_required_count": artifact.known_bad_required_count,
    }
    _emit(payload, as_json=args.json)
    return 0 if artifact.status in {"PASS", "WARN", "INCONCLUSIVE"} else 1


def _cmd_camera_lidar_koide_handoff(args: argparse.Namespace) -> int:
    """Validate the immutable Koide lock and print operator next actions."""

    try:
        lock = load_koide_execution_lock(args.lock)
    except (OSError, ValueError) as exc:
        _die(str(exc))
    payload = {
        "status": "handoff_only",
        "executes_external_process": False,
        "lock": str(args.lock),
        "lock_id": lock.lock_id,
        "lock_sha256": lock.lock_sha256,
        "profile": lock.profile,
        "initial_guess_mode": lock.initial_guess_mode,
        "superglue_policy": lock.superglue_policy,
        "source_repository": lock.source.repository,
        "source_commit": lock.source.commit,
        "license_spdx": lock.source.license_spdx,
        "image_digest": lock.image.digest,
        "base_image": lock.image.base_image,
        "base_image_digest": lock.image.base_image_digest,
        "base_image_digest_required_for_rebuild": (
            lock.image.base_image_digest_required_for_rebuild
        ),
        "runtime": {
            "docker_available": shutil.which("docker") is not None,
            "podman_available": shutil.which("podman") is not None,
            "ros2_available": shutil.which("ros2") is not None,
            "colcon_available": shutil.which("colcon") is not None,
        },
        "next_actions": lock.operator_steps,
        "stages": [stage.model_dump(mode="json") for stage in lock.stages],
    }
    _emit(payload, args.json)
    return 0


def _cmd_camera_lidar_koide_real_plan(args: argparse.Namespace) -> int:
    """Materialize the non-executing real Koide plan."""

    try:
        request = build_koide_real_pilot_request(
            camera_frame=args.camera_frame,
            lidar_frame=args.lidar_frame,
            command=["calibrex", "camera-lidar", "koide-real-plan"],
        )
        request.save(args.output)
    except (OSError, ValueError) as exc:
        _die(str(exc))
    _emit(
        {
            "status": request.status,
            "executes_external_process": False,
            "request": str(args.output),
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "source_commit": request.execution.source_commit,
            "container_digest": request.execution.image_digest,
            "official_result_claim": request.official_result_claim,
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_koide_real_verify(args: argparse.Namespace) -> int:
    """Verify a real Koide handoff's supplied evidence without running Koide."""

    try:
        verification = verify_koide_real_pilot_request(
            args.request,
            dataset_path=args.dataset,
            dataset_archive_path=args.archive,
            input_manifest_path=args.input_manifest,
            readiness_path=args.readiness,
            config_path=args.config,
            candidate_path=args.candidate,
            external_run_path=args.external_run,
            execution_log_path=args.execution_log,
            environment_path=args.environment,
            evidence_label=args.evidence_label,
            command=["calibrex", "camera-lidar", "koide-real-verify", str(args.request)],
        )
        verification.save(args.output)
    except (OSError, ValueError) as exc:
        _die(str(exc))
    _emit(
        {
            "status": verification.status,
            "official_evidence": verification.official_evidence,
            "evidence_label": verification.evidence_label,
            "verification": str(args.output),
            "request_id": verification.request_id,
            "reasons": verification.reasons,
        },
        args.json,
    )
    return 0 if verification.status == "READY_FOR_PILOT" else 1


def _cmd_camera_lidar_koide_real_finalize(args: argparse.Namespace) -> int:
    """Finalize a verified Koide pilot; never execute the external provider."""

    try:
        finalization = finalize_koide_real_pilot(
            args.request,
            args.verification,
            args.pilot,
            output_path=args.output,
            evidence_label=args.evidence_label,
            command=["calibrex", "camera-lidar", "koide-real-finalize", str(args.request)],
        )
    except (OSError, ValueError) as exc:
        _die(str(exc))
    _emit(
        {
            "status": finalization.status,
            "execution_state": finalization.execution_state,
            "official_result_claim": finalization.official_result_claim,
            "finalization": str(args.output),
            "request_id": finalization.request_id,
            "reasons": finalization.reasons,
        },
        args.json,
    )
    return 0 if finalization.status in {"PASS", "TEST_ONLY"} else 1


def _cmd_camera_lidar_export_koide_pilot(args: argparse.Namespace) -> int:
    try:
        paths = export_koide_pilot_autoware(
            args.pilot,
            args.output,
            base_frame=args.base_frame,
            sensor_frame=args.sensor_frame,
            static_tf_output=args.static_tf_output,
            manifest_output=args.manifest_output,
            overwrite=args.force,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _die(str(exc))
    payload = {"status": "ok", **{name: str(path) for name, path in paths.items()}}
    _emit(payload, as_json=args.json)
    return 0


def _cmd_camera_lidar_build_kitti_problem(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "build-kitti-problem",
        str(args.sequence_path),
        str(args.depth_provider),
        "--output",
        str(args.output),
        "--camera-stream",
        args.camera_stream,
        "--rotation-bound-deg",
        str(args.rotation_bound_deg),
        "--translation-bound-m",
        str(args.translation_bound_m),
    ]
    if args.lidar_directory is not None:
        command.extend(["--lidar-directory", str(args.lidar_directory)])
    if args.lidar_manifest is not None:
        command.extend(["--lidar-manifest", str(args.lidar_manifest)])
    if args.dataset_id is not None:
        command.extend(["--dataset-id", args.dataset_id])
    if args.problem_id is not None:
        command.extend(["--problem-id", args.problem_id])
    try:
        problem = build_kitti_raw_camera_lidar_problem(
            args.sequence_path,
            args.depth_provider,
            camera_stream=args.camera_stream,
            lidar_directory=args.lidar_directory,
            lidar_manifest_path=args.lidar_manifest,
            dataset_id=args.dataset_id,
            problem_id=args.problem_id,
            rotation_bound_deg=args.rotation_bound_deg,
            translation_bound_m=args.translation_bound_m,
            command=tuple(command),
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    problem.save(args.output)
    _emit(
        {
            "status": "ok",
            "problem_id": problem.problem_id,
            "dataset_id": problem.dataset_id,
            "sequence_id": problem.sequence_id,
            "frame_count": len(problem.observations),
            "depth_provider_sha256": problem.depth_provider_sha256,
            "output": str(args.output),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_build_kitti360_problem(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "build-kitti360-problem",
        str(args.sequence_path),
        str(args.depth_provider),
        "--output",
        str(args.output),
        "--camera-stream",
        args.camera_stream,
        "--split-id",
        args.split_id,
        "--rotation-bound-deg",
        str(args.rotation_bound_deg),
        "--translation-bound-m",
        str(args.translation_bound_m),
    ]
    if args.calibration_root is not None:
        command.extend(["--calibration-root", str(args.calibration_root)])
    if args.lidar_directory is not None:
        command.extend(["--lidar-directory", str(args.lidar_directory)])
    if args.lidar_manifest is not None:
        command.extend(["--lidar-manifest", str(args.lidar_manifest)])
    if args.dataset_id is not None:
        command.extend(["--dataset-id", args.dataset_id])
    if args.problem_id is not None:
        command.extend(["--problem-id", args.problem_id])
    try:
        problem = build_kitti360_camera_lidar_problem(
            args.sequence_path,
            args.depth_provider,
            calibration_root=args.calibration_root,
            lidar_directory=args.lidar_directory,
            lidar_manifest_path=args.lidar_manifest,
            camera_stream=args.camera_stream,
            split_id=args.split_id,
            dataset_id=args.dataset_id,
            problem_id=args.problem_id,
            rotation_bound_deg=args.rotation_bound_deg,
            translation_bound_m=args.translation_bound_m,
            command=tuple(command),
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    problem.save(args.output)
    _emit(
        {
            "status": "ok",
            "problem_id": problem.problem_id,
            "dataset_id": problem.dataset_id,
            "sequence_id": problem.sequence_id,
            "frame_count": len(problem.observations),
            "depth_provider_sha256": problem.depth_provider_sha256,
            "output": str(args.output),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_build_probabilistic_correspondence(
    args: argparse.Namespace,
) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "build-probabilistic-correspondence",
        str(args.export_manifest),
        "--output",
        str(args.output),
    ]
    try:
        artifact = build_probabilistic_correspondence_artifact(
            args.export_manifest,
            command=command,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "artifact_id": artifact.artifact_id,
            "dataset_id": artifact.dataset_id,
            "frame_count": len(artifact.frames),
            "correspondence_count": sum(len(frame.correspondences) for frame in artifact.frames),
            "provider": artifact.provider.provider,
            "output": str(args.output),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_build_a2d2_problem(
    args: argparse.Namespace,
) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "build-a2d2-problem",
        str(args.data_directory),
        str(args.depth_provider),
        "--lidar-output-directory",
        str(args.lidar_output_directory),
        "--lidar-manifest-output",
        str(args.lidar_manifest_output),
        "--output",
        str(args.output),
    ]
    try:
        problem = build_a2d2_camera_lidar_problem(
            args.data_directory,
            args.depth_provider,
            output_lidar_directory=args.lidar_output_directory,
            generated_manifest_path=args.lidar_manifest_output,
            command=command,
            problem_id=(args.problem_id or "a2d2-frontleft-preregistered-d2d"),
            rotation_bound_deg=args.rotation_bound_deg,
        )
        problem.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "problem_id": problem.problem_id,
            "dataset_id": problem.dataset_id,
            "frame_count": len(problem.observations),
            "depth_provider_sha256": problem.depth_provider_sha256,
            "lidar_manifest": str(args.lidar_manifest_output),
            "output": str(args.output),
            "accuracy_limitation": ("A2D2 source points are pre-registered into the camera view"),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_benchmark_rotation(args: argparse.Namespace) -> int:
    if args.required_hit_rate is not None and not 0.0 <= args.required_hit_rate <= 1.0:
        _die("--required-hit-rate must be in [0, 1]")
    command = (
        f"calibrex camera-lidar benchmark-rotation {args.problem} {args.protocol} "
        f"--trace-dir {args.trace_dir} --definition-output "
        f"{args.definition_output} --output {args.output} --workers {args.workers}"
    )
    if args.resume:
        command += " --resume"
    try:
        definition, benchmark = run_borer_rotation_benchmark(
            args.problem,
            args.protocol,
            trace_directory=args.trace_dir,
            command=command,
            bootstrap_samples=args.bootstrap_samples,
            workers=args.workers,
            resume=args.resume,
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    definition.save(args.definition_output)
    benchmark.save(args.output)
    bullseye_output = args.bullseye_output or args.output.with_name(
        f"{args.output.stem}_bullseye.svg"
    )
    bullseye_artifact_output = args.bullseye_artifact_output or args.output.with_name(
        f"{args.output.stem}_bullseye.json"
    )
    try:
        write_bullseye_plot(
            args.problem,
            args.protocol,
            args.trace_dir,
            svg_path=bullseye_output,
            artifact_path=bullseye_artifact_output,
            command=tuple(command.split()),
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    summary = benchmark.method_summaries["native_borer_d2d_rotation"]
    hit_rate = summary.metrics["hit"].distribution.mean
    gate_passed = (
        hit_rate is not None
        and args.required_hit_rate is not None
        and hit_rate >= args.required_hit_rate
    )
    _emit(
        {
            "status": "ok",
            "definition": str(args.definition_output),
            "benchmark": str(args.output),
            "trace_directory": str(args.trace_dir),
            "bullseye": str(bullseye_output),
            "bullseye_artifact": str(bullseye_artifact_output),
            "trial_count": summary.trial_count,
            "success_count": summary.success_count,
            "failure_count": summary.failure_count,
            "failure_rate": summary.failure_rate,
            "hit_rate": hit_rate,
            "required_hit_rate": args.required_hit_rate,
            "gate_evaluated": args.required_hit_rate is not None,
            "gate_passed": gate_passed,
        },
        args.json,
    )
    if args.required_hit_rate is not None and not gate_passed:
        return 2
    return 0


def _cmd_camera_lidar_benchmark_six_dof(args: argparse.Namespace) -> int:
    if args.required_hit_rate is not None and not 0.0 <= args.required_hit_rate <= 1.0:
        _die("--required-hit-rate must be in [0, 1]")
    command = (
        f"calibrex camera-lidar benchmark-six-dof {args.problem} {args.protocol} "
        f"--trace-dir {args.trace_dir} --definition-output "
        f"{args.definition_output} --output {args.output} --workers {args.workers} "
        f"--projection-backend {args.projection_backend} "
        f"--bootstrap-samples {args.bootstrap_samples}"
    )
    if args.required_hit_rate is not None:
        command += f" --required-hit-rate {args.required_hit_rate}"
    if args.resume:
        command += " --resume"
    try:
        definition, benchmark = run_borer_six_dof_benchmark(
            args.problem,
            args.protocol,
            trace_directory=args.trace_dir,
            command=command,
            bootstrap_samples=args.bootstrap_samples,
            workers=args.workers,
            resume=args.resume,
            projection_backend=cast(
                ProjectionBackend,
                args.projection_backend,
            ),
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    definition.save(args.definition_output)
    benchmark.save(args.output)
    summary = benchmark.method_summaries["native_borer_d2d_six_dof"]
    hit_rate = summary.metrics["hit"].distribution.mean
    gate_passed = (
        hit_rate is not None
        and args.required_hit_rate is not None
        and hit_rate >= args.required_hit_rate
    )
    _emit(
        {
            "status": "ok",
            "definition": str(args.definition_output),
            "benchmark": str(args.output),
            "trace_directory": str(args.trace_dir),
            "projection_backend": args.projection_backend,
            "trial_count": summary.trial_count,
            "success_count": summary.success_count,
            "failure_count": summary.failure_count,
            "failure_rate": summary.failure_rate,
            "hit_rate": hit_rate,
            "required_hit_rate": args.required_hit_rate,
            "gate_evaluated": args.required_hit_rate is not None,
            "gate_passed": gate_passed,
        },
        args.json,
    )
    if args.required_hit_rate is not None and not gate_passed:
        return 2
    return 0


def _cmd_camera_lidar_refine_probabilistic_pnp(
    args: argparse.Namespace,
) -> int:
    options = OpenCvProbabilisticPnpOptions(
        minimum_confidence=args.minimum_confidence,
        minimum_correspondences=args.minimum_correspondences,
        ransac_reprojection_threshold_px=(args.ransac_reprojection_threshold_px),
        ransac_confidence=args.ransac_confidence,
        ransac_iterations=args.ransac_iterations,
        mahalanobis_inlier_threshold=args.mahalanobis_inlier_threshold,
        random_seed=args.random_seed,
    )
    command = [
        "calibrex",
        "camera-lidar",
        "refine-probabilistic-pnp",
        str(args.correspondence_artifact),
        args.frame_id,
        "--output",
        str(args.output),
    ]
    initial_transform = None
    initialization_digest = None
    if args.initial_problem is not None:
        try:
            problem = load_camera_lidar_problem(args.initial_problem)
        except (OSError, ValueError) as exc:
            raise CalibrexError(str(exc)) from exc
        initial_transform = problem.initial_transform_camera_lidar.as_se3()
        initialization_digest = sha256_path(args.initial_problem)
        if initialization_digest is None:
            raise CalibrexError(f"initialization problem is not readable: {args.initial_problem}")
        command.extend(["--initial-problem", str(args.initial_problem)])
    try:
        result = OpenCvProbabilisticPnpAdapter().solve_artifact(
            args.correspondence_artifact,
            args.frame_id,
            options,
            initial_transform_camera_lidar=initial_transform,
            initialization_artifact_sha256=initialization_digest,
        )
        artifact = result.to_artifact(
            result_id=(args.result_id or f"{result.artifact_id}-{args.frame_id}-opencv-pnp"),
            options=options,
            command=command,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": result.status,
            "result": str(args.output),
            "frame_id": args.frame_id,
            "selected_correspondence_count": (result.selected_correspondence_count),
            "ransac_inlier_count": result.ransac_inlier_count,
            "probabilistic_inlier_count": (result.probabilistic_inlier_count),
            "weighted_reprojection_rmse_px": (result.weighted_reprojection_rmse_px),
            "ransac_inlier_reprojection_rmse_px": (result.ransac_inlier_reprojection_rmse_px),
            "mean_mahalanobis_error": result.mean_mahalanobis_error,
        },
        args.json,
    )
    return 0 if result.status == "converged" else 2


def _cmd_camera_lidar_refine_probabilistic_pnp_aggregate(
    args: argparse.Namespace,
) -> int:
    initializer_calibration_id = None
    initializer_calibration_digest = None
    try:
        if args.initializer_calibration is not None:
            custom_option_flags = (
                args.minimum_confidence != 0.25
                or args.minimum_correspondences != 6
                or args.minimum_frame_correspondences != 4
                or args.minimum_frames != 2
                or args.ransac_reprojection_threshold_px != 4.0
                or args.ransac_confidence != 0.999
                or args.ransac_iterations != 1000
                or args.mahalanobis_inlier_threshold != 3.0
            )
            if custom_option_flags:
                raise ValueError(
                    "initializer calibration cannot be combined with solver option overrides"
                )
            if args.initial_problem is not None:
                raise ValueError(
                    "locked aggregate PnP forbids an initial problem because "
                    "development calibration used no pose initializer"
                )
            calibration = load_camera_lidar_initializer_calibration(args.initializer_calibration)
            correspondence = load_probabilistic_correspondence(args.correspondence_artifact)
            if correspondence.provider != calibration.provider:
                raise ValueError("correspondence provider identity differs from initializer lock")
            allowed_dataset_ids = {
                calibration.dataset_id,
                *calibration.evaluation_dataset_ids_excluded,
            }
            if correspondence.dataset_id not in allowed_dataset_ids:
                raise ValueError("correspondence dataset was not prespecified by initializer lock")
            options = initializer_options_from_camera_lidar_calibration(
                calibration,
                random_seed=args.random_seed,
            )
            initializer_calibration_id = calibration.calibration_id
            initializer_calibration_digest = sha256_path(args.initializer_calibration)
            if initializer_calibration_digest is None:
                raise ValueError(
                    "initializer calibration artifact is not readable: "
                    f"{args.initializer_calibration}"
                )
        else:
            options = OpenCvProbabilisticPnpOptions(
                minimum_confidence=args.minimum_confidence,
                minimum_correspondences=args.minimum_correspondences,
                minimum_frame_correspondences=(args.minimum_frame_correspondences),
                minimum_frames=args.minimum_frames,
                ransac_reprojection_threshold_px=(args.ransac_reprojection_threshold_px),
                ransac_confidence=args.ransac_confidence,
                ransac_iterations=args.ransac_iterations,
                mahalanobis_inlier_threshold=(args.mahalanobis_inlier_threshold),
                random_seed=args.random_seed,
            )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    command = [
        "calibrex",
        "camera-lidar",
        "refine-probabilistic-pnp-aggregate",
        str(args.correspondence_artifact),
        "--output",
        str(args.output),
    ]
    if args.initializer_calibration is not None:
        command.extend(
            [
                "--initializer-calibration",
                str(args.initializer_calibration),
                "--random-seed",
                str(options.random_seed),
            ]
        )
    else:
        command.extend(
            [
                "--minimum-confidence",
                str(options.minimum_confidence),
                "--minimum-correspondences",
                str(options.minimum_correspondences),
                "--minimum-frame-correspondences",
                str(options.minimum_frame_correspondences),
                "--minimum-frames",
                str(options.minimum_frames),
                "--ransac-reprojection-threshold-px",
                str(options.ransac_reprojection_threshold_px),
                "--ransac-confidence",
                str(options.ransac_confidence),
                "--ransac-iterations",
                str(options.ransac_iterations),
                "--mahalanobis-inlier-threshold",
                str(options.mahalanobis_inlier_threshold),
                "--random-seed",
                str(options.random_seed),
            ]
        )
    if args.result_id is not None:
        command.extend(["--result-id", args.result_id])
    initial_transform = None
    initialization_digest = None
    if args.initial_problem is not None:
        try:
            problem = load_camera_lidar_problem(args.initial_problem)
        except (OSError, ValueError) as exc:
            raise CalibrexError(str(exc)) from exc
        initial_transform = problem.initial_transform_camera_lidar.as_se3()
        initialization_digest = sha256_path(args.initial_problem)
        if initialization_digest is None:
            raise CalibrexError(f"initialization problem is not readable: {args.initial_problem}")
        command.extend(["--initial-problem", str(args.initial_problem)])
    try:
        result = OpenCvProbabilisticPnpAdapter().solve_artifact_aggregate(
            args.correspondence_artifact,
            options,
            initial_transform_camera_lidar=initial_transform,
            initialization_artifact_sha256=initialization_digest,
            initializer_calibration_id=initializer_calibration_id,
            initializer_calibration_sha256=initializer_calibration_digest,
        )
        artifact = result.to_artifact(
            result_id=(args.result_id or f"{result.artifact_id}-aggregate-opencv-pnp"),
            options=options,
            command=command,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": result.status,
            "result": str(args.output),
            "frame_id": result.frame_id,
            "source_frame_ids": list(result.source_frame_ids),
            "frame_selection_rule_id": result.frame_selection_rule_id,
            "initializer_calibration_id": initializer_calibration_id,
            "initializer_calibration_sha256": initializer_calibration_digest,
            "selected_correspondence_count": (result.selected_correspondence_count),
            "ransac_inlier_count": result.ransac_inlier_count,
            "probabilistic_inlier_count": (result.probabilistic_inlier_count),
            "weighted_reprojection_rmse_px": (result.weighted_reprojection_rmse_px),
            "ransac_inlier_reprojection_rmse_px": (result.ransac_inlier_reprojection_rmse_px),
            "mean_mahalanobis_error": result.mean_mahalanobis_error,
        },
        args.json,
    )
    return 0 if result.status == "converged" else 2


def _cmd_camera_lidar_refine_probabilistic_multiframe(
    args: argparse.Namespace,
) -> int:
    try:
        if args.confidence_calibration is not None:
            custom_option_flags = (
                args.minimum_confidence != 0.25
                or args.holdout_ratio != 0.25
                or args.minimum_train_correspondences != 24
                or args.minimum_holdout_correspondences != 8
                or args.max_evaluations != 400
                or args.minimum_absolute_train_objective_improvement != 1.0e-6
                or args.minimum_relative_train_objective_improvement != 1.0e-4
                or args.maximum_train_correspondence_loss_fraction != 0.05
                or args.maximum_accepted_bound_fraction != 0.95
                or args.without_covariance
                or args.without_outlier_probability
                or args.without_reliability
            )
            if custom_option_flags:
                raise ValueError(
                    "confidence calibration cannot be combined with solver option overrides"
                )
            calibration = load_camera_lidar_confidence_calibration(args.confidence_calibration)
            options = refinement_options_from_camera_lidar_confidence_calibration(
                calibration,
                split_seed=args.split_seed,
            )
        else:
            options = ProbabilisticCameraLidarRefinementOptions(
                holdout_ratio=args.holdout_ratio,
                split_seed=args.split_seed,
                minimum_confidence=args.minimum_confidence,
                minimum_train_correspondences=(args.minimum_train_correspondences),
                minimum_holdout_correspondences=(args.minimum_holdout_correspondences),
                max_evaluations=args.max_evaluations,
                minimum_absolute_train_objective_improvement=(
                    args.minimum_absolute_train_objective_improvement
                ),
                minimum_relative_train_objective_improvement=(
                    args.minimum_relative_train_objective_improvement
                ),
                maximum_train_correspondence_loss_fraction=(
                    args.maximum_train_correspondence_loss_fraction
                ),
                maximum_accepted_bound_fraction=(args.maximum_accepted_bound_fraction),
                use_covariance=not args.without_covariance,
                use_outlier_probability=not args.without_outlier_probability,
                use_reliability=not args.without_reliability,
            )
        command = [
            "calibrex",
            "camera-lidar",
            "refine-probabilistic-multiframe",
            str(args.correspondence_artifact),
            str(args.initial_problem),
            "--output",
            str(args.output),
            "--split-seed",
            str(args.split_seed),
        ]
        if args.initial_trace is not None:
            command.extend(["--initial-trace", str(args.initial_trace)])
        if args.confidence_calibration is not None:
            command.extend(["--confidence-calibration", str(args.confidence_calibration)])
        result = run_probabilistic_camera_lidar_refinement(
            args.correspondence_artifact,
            args.initial_problem,
            initialization_trace_path=args.initial_trace,
            confidence_calibration_path=args.confidence_calibration,
            result_id=args.result_id,
            options=options,
            command=command,
        )
        result.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    acceptance = result.acceptance
    if acceptance is None:
        raise CalibrexError(
            "new probabilistic refinement output is missing its acceptance decision"
        )
    _emit(
        {
            "status": result.status,
            "result": str(args.output),
            "initialization_source": result.initialization_source,
            "initialization_trace_id": result.initialization_trace_id,
            "initialization_trace_status": result.initialization_trace_status,
            "initialization_trace_hit": result.initialization_trace_hit,
            "confidence_calibration_id": (result.provenance.confidence_calibration_id),
            "candidate_accepted": acceptance.accepted,
            "selected_source": acceptance.selected_source,
            "acceptance_reasons": acceptance.reasons,
            "train_frame_count": len(result.train_frame_ids),
            "holdout_frame_count": len(result.holdout_frame_ids),
            "initial_holdout_rmse_px": (
                result.initial_holdout_evaluation.weighted_reprojection_rmse_px
            ),
            "final_holdout_rmse_px": (
                result.final_holdout_evaluation.weighted_reprojection_rmse_px
            ),
            "final_rotation_error_deg": result.final_rotation_error_deg,
            "final_translation_error_m": result.final_translation_error_m,
            "use_covariance": result.options["use_covariance"],
            "use_outlier_probability": (result.options["use_outlier_probability"]),
            "use_reliability": result.options["use_reliability"],
        },
        args.json,
    )
    return 0 if result.status == "converged" else 2


def _cmd_camera_lidar_diagnose_probabilistic_correspondence(
    args: argparse.Namespace,
) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "diagnose-probabilistic-correspondence",
        str(args.correspondence_artifact),
        str(args.refinement_result),
        "--pose-role",
        args.pose_role,
        "--minimum-frame-correspondences",
        str(args.minimum_frame_correspondences),
        "--output",
        str(args.output),
    ]
    try:
        report = analyze_camera_lidar_correspondence_quality(
            args.correspondence_artifact,
            args.refinement_result,
            pose_role=args.pose_role,
            report_id=args.report_id,
            minimum_frame_correspondences=args.minimum_frame_correspondences,
            command=command,
        )
        report.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "grade": report.summary.grade,
            "report": str(args.output),
            "report_id": report.report_id,
            "pose_role": report.pose_role,
            "frame_count": report.summary.frame_count,
            "accepted_correspondence_count": (report.summary.accepted_correspondence_count),
            "accepted_correspondence_rate": (report.summary.accepted_correspondence_rate),
            "aggregate_information_rank": (report.summary.aggregate_observability.rank),
            "gate_reasons": report.summary.gate_reasons,
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_calibrate_probabilistic_confidence(
    args: argparse.Namespace,
) -> int:
    try:
        thresholds = tuple(
            float(value.strip()) for value in args.thresholds.split(",") if value.strip()
        )
        calibration_seeds = tuple(
            int(value.strip()) for value in args.calibration_split_seeds.split(",") if value.strip()
        )
        evaluation_seeds = tuple(
            int(value.strip()) for value in args.evaluation_split_seeds.split(",") if value.strip()
        )
    except ValueError as exc:
        raise CalibrexError(f"invalid confidence calibration list: {exc}") from exc
    excluded_dataset_ids = [
        value.strip() for value in args.excluded_evaluation_dataset_ids.split(",") if value.strip()
    ]
    command = [
        "calibrex",
        "camera-lidar",
        "calibrate-probabilistic-confidence",
        str(args.correspondence_artifact),
        str(args.initial_problem),
        "--output",
        str(args.output),
        "--thresholds",
        args.thresholds,
        "--calibration-split-seeds",
        args.calibration_split_seeds,
        "--evaluation-split-seeds",
        args.evaluation_split_seeds,
        "--holdout-ratio",
        str(args.holdout_ratio),
        "--minimum-train-correspondences",
        str(args.minimum_train_correspondences),
        "--minimum-holdout-correspondences",
        str(args.minimum_holdout_correspondences),
        "--minimum-frame-correspondences",
        str(args.minimum_frame_correspondences),
        "--minimum-frame-support-rate",
        str(args.minimum_frame_support_rate),
        "--minimum-full-rank-frame-rate",
        str(args.minimum_full_rank_frame_rate),
        "--development-reprojection-inlier-threshold-px",
        str(args.development_reprojection_inlier_threshold_px),
        "--minimum-geometric-inlier-rate",
        str(args.minimum_geometric_inlier_rate),
        "--minimum-frame-geometric-inlier-count",
        str(args.minimum_frame_geometric_inlier_count),
        "--minimum-frame-geometric-support-rate",
        str(args.minimum_frame_geometric_support_rate),
        "--excluded-evaluation-dataset-ids",
        args.excluded_evaluation_dataset_ids,
    ]
    try:
        calibration = calibrate_camera_lidar_confidence(
            args.correspondence_artifact,
            args.initial_problem,
            thresholds=thresholds,
            calibration_split_seeds=calibration_seeds,
            evaluation_split_seeds=evaluation_seeds,
            holdout_ratio=args.holdout_ratio,
            minimum_train_correspondences=args.minimum_train_correspondences,
            minimum_holdout_correspondences=(args.minimum_holdout_correspondences),
            minimum_frame_correspondences=args.minimum_frame_correspondences,
            minimum_frame_support_rate=args.minimum_frame_support_rate,
            minimum_full_rank_frame_rate=args.minimum_full_rank_frame_rate,
            development_reprojection_inlier_threshold_px=(
                args.development_reprojection_inlier_threshold_px
            ),
            minimum_geometric_inlier_rate=args.minimum_geometric_inlier_rate,
            minimum_frame_geometric_inlier_count=(args.minimum_frame_geometric_inlier_count),
            minimum_frame_geometric_support_rate=(args.minimum_frame_geometric_support_rate),
            evaluation_dataset_ids_excluded=excluded_dataset_ids,
            calibration_id=args.calibration_id,
            command=command,
        )
        calibration.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": calibration.status,
            "calibration": str(args.output),
            "calibration_id": calibration.calibration_id,
            "development_dataset_id": calibration.dataset_id,
            "selected_minimum_confidence": (calibration.selected_minimum_confidence),
            "passing_thresholds": [
                item.minimum_confidence for item in calibration.candidates if item.gate_pass
            ],
            "evaluation_dataset_ids_excluded": (calibration.evaluation_dataset_ids_excluded),
            "release_sota_claim_allowed": calibration.release_sota_claim_allowed,
        },
        args.json,
    )
    return 0 if calibration.status == "locked" else 2


def _cmd_camera_lidar_compare_probabilistic_provider_support(
    args: argparse.Namespace,
) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "compare-probabilistic-provider-support",
        *(str(path) for path in args.confidence_calibrations),
        "--output",
        str(args.output),
    ]
    if args.comparison_id is not None:
        command.extend(["--comparison-id", args.comparison_id])
    try:
        comparison = compare_camera_lidar_provider_support(
            args.confidence_calibrations,
            comparison_id=args.comparison_id,
            command=command,
        )
        comparison.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    leader = next(
        item
        for item in comparison.candidates
        if item.candidate_id == comparison.diagnostic_leader_candidate_id
    )
    _emit(
        {
            "status": comparison.status,
            "comparison": str(args.output),
            "comparison_id": comparison.comparison_id,
            "selected_candidate_id": comparison.selected_candidate_id,
            "diagnostic_leader_candidate_id": (comparison.diagnostic_leader_candidate_id),
            "diagnostic_leader_provider_version": leader.provider.version,
            "diagnostic_leader_threshold": (leader.diagnostic.minimum_confidence),
            "diagnostic_leader_worst_split_geometric_inliers": (
                leader.worst_split_geometric_inlier_count
            ),
            "evaluation_dataset_ids_excluded": (
                comparison.protocol.evaluation_dataset_ids_excluded
            ),
            "release_sota_claim_allowed": comparison.release_sota_claim_allowed,
        },
        args.json,
    )
    return 0 if comparison.status == "locked" else 2


def _cmd_camera_lidar_calibrate_probabilistic_pnp_initializer(
    args: argparse.Namespace,
) -> int:
    try:
        confidence_thresholds = tuple(
            float(value.strip()) for value in args.confidence_thresholds.split(",") if value.strip()
        )
        reprojection_thresholds = tuple(
            float(value.strip())
            for value in args.ransac_reprojection_thresholds_px.split(",")
            if value.strip()
        )
        random_seeds = tuple(
            int(value.strip()) for value in args.random_seeds.split(",") if value.strip()
        )
        gate = CameraLidarInitializerRecoveryGate(
            rotation_error_max_deg=args.rotation_error_max_deg,
            translation_error_max_m=args.translation_error_max_m,
            minimum_selected_frame_count=(args.gate_minimum_selected_frame_count),
            minimum_selected_correspondence_count=(args.gate_minimum_selected_correspondence_count),
            minimum_ransac_inlier_count=(args.gate_minimum_ransac_inlier_count),
            maximum_ransac_inlier_reprojection_rmse_px=(args.gate_maximum_ransac_inlier_rmse_px),
            maximum_seed_rotation_delta_deg=(args.gate_maximum_seed_rotation_delta_deg),
            maximum_seed_translation_delta_m=(args.gate_maximum_seed_translation_delta_m),
        )
    except ValueError as exc:
        raise CalibrexError(f"invalid initializer calibration option: {exc}") from exc
    excluded_dataset_ids = [
        value.strip() for value in args.excluded_evaluation_dataset_ids.split(",") if value.strip()
    ]
    command = [
        "calibrex",
        "camera-lidar",
        "calibrate-probabilistic-pnp-initializer",
        str(args.correspondence_artifact),
        str(args.initial_problem),
        "--output",
        str(args.output),
        "--confidence-thresholds",
        args.confidence_thresholds,
        "--ransac-reprojection-thresholds-px",
        args.ransac_reprojection_thresholds_px,
        "--random-seeds",
        args.random_seeds,
        "--minimum-correspondences",
        str(args.minimum_correspondences),
        "--minimum-frame-correspondences",
        str(args.minimum_frame_correspondences),
        "--minimum-frames",
        str(args.minimum_frames),
        "--ransac-confidence",
        str(args.ransac_confidence),
        "--ransac-iterations",
        str(args.ransac_iterations),
        "--mahalanobis-inlier-threshold",
        str(args.mahalanobis_inlier_threshold),
        "--rotation-error-max-deg",
        str(args.rotation_error_max_deg),
        "--translation-error-max-m",
        str(args.translation_error_max_m),
        "--gate-minimum-selected-frame-count",
        str(args.gate_minimum_selected_frame_count),
        "--gate-minimum-selected-correspondence-count",
        str(args.gate_minimum_selected_correspondence_count),
        "--gate-minimum-ransac-inlier-count",
        str(args.gate_minimum_ransac_inlier_count),
        "--gate-maximum-ransac-inlier-rmse-px",
        str(args.gate_maximum_ransac_inlier_rmse_px),
        "--gate-maximum-seed-rotation-delta-deg",
        str(args.gate_maximum_seed_rotation_delta_deg),
        "--gate-maximum-seed-translation-delta-m",
        str(args.gate_maximum_seed_translation_delta_m),
        "--excluded-evaluation-dataset-ids",
        args.excluded_evaluation_dataset_ids,
    ]
    if args.calibration_id is not None:
        command.extend(["--calibration-id", args.calibration_id])
    try:
        calibration = calibrate_camera_lidar_initializer(
            args.correspondence_artifact,
            args.initial_problem,
            confidence_thresholds=confidence_thresholds,
            ransac_reprojection_thresholds_px=reprojection_thresholds,
            random_seeds=random_seeds,
            minimum_correspondences=args.minimum_correspondences,
            minimum_frame_correspondences=(args.minimum_frame_correspondences),
            minimum_frames=args.minimum_frames,
            ransac_confidence=args.ransac_confidence,
            ransac_iterations=args.ransac_iterations,
            mahalanobis_inlier_threshold=(args.mahalanobis_inlier_threshold),
            recovery_gate=gate,
            evaluation_dataset_ids_excluded=excluded_dataset_ids,
            calibration_id=args.calibration_id,
            command=command,
        )
        calibration.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    locked = calibration.locked_initializer_options
    _emit(
        {
            "status": calibration.status,
            "calibration": str(args.output),
            "calibration_id": calibration.calibration_id,
            "development_dataset_id": calibration.dataset_id,
            "candidate_count": len(calibration.candidates),
            "passing_candidate_count": sum(item.gate_pass for item in calibration.candidates),
            "selected_candidate_id": calibration.selected_candidate_id,
            "selected_minimum_confidence": (
                locked.minimum_confidence if locked is not None else None
            ),
            "selected_ransac_reprojection_threshold_px": (
                locked.ransac_reprojection_threshold_px if locked is not None else None
            ),
            "evaluation_dataset_ids_excluded": (calibration.evaluation_dataset_ids_excluded),
            "release_sota_claim_allowed": calibration.release_sota_claim_allowed,
        },
        args.json,
    )
    return 0 if calibration.status == "locked" else 2


def _cmd_camera_lidar_benchmark_probabilistic_ablation(
    args: argparse.Namespace,
) -> int:
    try:
        seeds = tuple(int(value.strip()) for value in args.split_seeds.split(",") if value.strip())
    except ValueError as exc:
        raise CalibrexError("--split-seeds must contain integers") from exc
    command = (
        "calibrex camera-lidar benchmark-probabilistic-ablation "
        f"{args.correspondence_artifact} {args.initial_problem} "
        f"--result-dir {args.result_dir} --definition-output "
        f"{args.definition_output} --output {args.output} "
        f"--split-seeds {args.split_seeds}"
    )
    if args.initial_trace is not None:
        command += f" --initial-trace {args.initial_trace}"
    try:
        definition, benchmark = run_probabilistic_refinement_ablation(
            args.correspondence_artifact,
            args.initial_problem,
            initialization_trace_path=args.initial_trace,
            result_directory=args.result_dir,
            command=command,
            split_seeds=seeds,
            bootstrap_samples=args.bootstrap_samples,
        )
        definition.save(args.definition_output)
        benchmark.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "definition": str(args.definition_output),
            "benchmark": str(args.output),
            "result_directory": str(args.result_dir),
            "split_count": len(definition.protocol.splits),
            "method_count": len(definition.methods),
            "trial_count": len(definition.trials),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_benchmark_probabilistic_falsification(
    args: argparse.Namespace,
) -> int:
    try:
        seeds = tuple(int(value.strip()) for value in args.split_seeds.split(",") if value.strip())
    except ValueError as exc:
        raise CalibrexError("--split-seeds must contain integers") from exc
    command = (
        "calibrex camera-lidar benchmark-probabilistic-falsification "
        f"{args.correspondence_artifact} {args.initial_problem} "
        f"--protocol {args.protocol} --output-dir {args.output_dir} "
        f"--split-seeds {args.split_seeds} "
        f"--known-bad-rotation-deg {args.known_bad_rotation_deg} "
        f"--known-bad-translation-m {args.known_bad_translation_m} "
        f"--known-bad-min-holdout-rmse-delta-px "
        f"{args.known_bad_min_holdout_rmse_delta_px} "
        f"--bootstrap-samples {args.bootstrap_samples}"
    )
    if args.initial_trace is not None:
        command += f" --initial-trace {args.initial_trace}"
    else:
        command += f" --trace-dir {args.trace_dir}"
    if args.max_final_holdout_rmse_px is not None:
        command += f" --max-final-holdout-rmse-px {args.max_final_holdout_rmse_px}"
    try:
        artifacts = run_probabilistic_camera_lidar_falsification(
            args.correspondence_artifact,
            args.initial_problem,
            initialization_trace_path=args.initial_trace,
            initialization_trace_directory=args.trace_dir,
            benchmark_protocol_path=args.protocol,
            output_directory=args.output_dir,
            command=command,
            split_seeds=seeds,
            known_bad_rotation_deg=args.known_bad_rotation_deg,
            known_bad_translation_m=args.known_bad_translation_m,
            known_bad_min_holdout_rmse_delta_px=(args.known_bad_min_holdout_rmse_delta_px),
            max_final_holdout_rmse_px=args.max_final_holdout_rmse_px,
            bootstrap_samples=args.bootstrap_samples,
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": artifacts.assessment.status,
            "assessment": str(artifacts.assessment_path),
            "benchmark": str(artifacts.benchmark_path),
            "definition": str(artifacts.definition_path),
            "evidence": str(artifacts.evidence_path),
            "failure_analysis": str(artifacts.failure_analysis_path),
            "bundle": str(artifacts.bundle_path),
            "verification": str(artifacts.verification_path),
            "verification_valid": artifacts.verification.valid,
            "result_count": len(artifacts.result_paths),
            "known_bad_case_count": len(artifacts.evidence.cases),
            "rotation_failure_count": (artifacts.failure_analysis.summary.rotation_failure_count),
            "translation_failure_count": (
                artifacts.failure_analysis.summary.translation_failure_count
            ),
        },
        args.json,
    )
    return 0 if artifacts.assessment.status == "pass" and artifacts.verification.valid else 2


def _cmd_camera_lidar_refine_continuous_time(
    args: argparse.Namespace,
) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "refine-continuous-time",
        str(args.problem),
        "--output",
        str(args.output),
    ]
    try:
        result = run_continuous_time_camera_lidar_problem(
            args.problem,
            result_id=args.result_id,
            command=command,
        )
        result.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": result.status,
            "result": str(args.output),
            "estimated_time_offset_sec": result.estimated_time_offset_sec,
            "final_rotation_error_deg": result.final_rotation_error_deg,
            "final_translation_error_m": result.final_translation_error_m,
            "final_time_offset_error_sec": (result.final_time_offset_error_sec),
            "time_observability_rank": result.time_observability_rank,
            "trajectory_model": result.trajectory_model,
            "train_correspondence_count": (
                result.final_train_evaluation.valid_correspondence_count
            ),
            "holdout_correspondence_count": (
                result.final_holdout_evaluation.valid_correspondence_count
            ),
            "initial_holdout_rmse_px": (
                result.initial_holdout_evaluation.weighted_reprojection_rmse_px
            ),
            "final_holdout_rmse_px": (
                result.final_holdout_evaluation.weighted_reprojection_rmse_px
            ),
        },
        args.json,
    )
    return 0 if result.status == "converged" else 2


def _cmd_camera_lidar_attach_continuous_trajectory(
    args: argparse.Namespace,
) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "attach-continuous-trajectory",
        str(args.problem),
        str(args.trajectory),
        "--output",
        str(args.output),
    ]
    try:
        problem = attach_recorded_body_trajectory(
            args.problem,
            args.trajectory,
            problem_id=args.problem_id,
            command=command,
        )
        problem.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "problem": str(args.output),
            "problem_id": problem.problem_id,
            "trajectory_pose_count": (
                len(problem.body_trajectory.poses) if problem.body_trajectory is not None else 0
            ),
            "trajectory_source_sha256": (
                problem.body_trajectory.source_sha256
                if problem.body_trajectory is not None
                else None
            ),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_benchmark_continuous_time_ablation(
    args: argparse.Namespace,
) -> int:
    try:
        seeds = tuple(int(value.strip()) for value in args.split_seeds.split(",") if value.strip())
    except ValueError as exc:
        raise CalibrexError("--split-seeds must contain integers") from exc
    command = (
        "calibrex camera-lidar benchmark-continuous-time-ablation "
        f"{args.problem} --result-dir {args.result_dir} "
        f"--definition-output {args.definition_output} --output "
        f"{args.output} --split-seeds {args.split_seeds}"
    )
    try:
        definition, benchmark = run_continuous_time_camera_lidar_ablation(
            args.problem,
            result_directory=args.result_dir,
            command=command,
            split_seeds=seeds,
            bootstrap_samples=args.bootstrap_samples,
        )
        definition.save(args.definition_output)
        benchmark.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "definition": str(args.definition_output),
            "benchmark": str(args.output),
            "result_directory": str(args.result_dir),
            "split_count": len(definition.protocol.splits),
            "method_count": len(definition.methods),
            "trial_count": len(definition.trials),
        },
        args.json,
    )
    return 0


def _cmd_camera_lidar_audit_sota(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "audit-sota",
        str(args.protocol),
        "--output",
        str(args.output),
    ]
    try:
        audit = audit_camera_lidar_sota_claim(
            args.protocol,
            audit_id=args.audit_id,
            command=command,
        )
        audit.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "audit": str(args.output),
            "verdict": audit.verdict,
            "achieved_dataset_families": (audit.achieved_dataset_families),
            "achieved_independent_rig_count": (audit.achieved_independent_rig_count),
            "requirement_status": {item.requirement_id: item.status for item in audit.requirements},
        },
        args.json,
    )
    return 0 if audit.verdict == "supported" else 2


def _cmd_camera_lidar_empirical_uncertainty(
    args: argparse.Namespace,
) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "empirical-uncertainty",
        str(args.correspondence),
        str(args.initial_problem),
        "--result-dir",
        str(args.result_dir),
        "--output",
        str(args.output),
    ]
    try:
        artifact = run_empirical_se3_uncertainty(
            args.correspondence,
            args.initial_problem,
            result_directory=args.result_dir,
            command=" ".join(command),
            block_length=args.block_length,
            target_coverage=args.target_coverage,
            resample_count=args.resample_count,
            seed=args.seed,
            fit_block_ratio=args.fit_block_ratio,
            overconfidence_scale=args.overconfidence_scale,
            assess_coverage=not args.stability_only,
            result_id=args.uncertainty_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "uncertainty": str(args.output),
            "policy_status": artifact.policy_status,
            "coverage_score": artifact.coverage_score,
            "observed_coverage_translation": (artifact.observed_coverage_translation),
            "observed_coverage_rotation": artifact.observed_coverage_rotation,
            "observed_coverage_joint": artifact.observed_coverage_joint,
            "interval_halfwidth_translation_m": (artifact.interval_halfwidth_translation_m),
            "interval_halfwidth_rotation_deg": (artifact.interval_halfwidth_rotation_deg),
            "resample_files": len(artifact.iterations),
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_trajectory_build_contract(args: argparse.Namespace) -> int:
    from calibrex.core.continuous_time_contract import (
        ContinuousTimeClockSemantics,
        ContinuousTimeKnotDomain,
        ContinuousTimeKnotPose,
        ContinuousTimeTrajectoryContract,
        ContinuousTimeTrajectoryProvenance,
    )
    from calibrex.core.provenance import git_commit, sha256_path
    from calibrex.core.result import TransformResult

    mapping = read_mapping(args.knots_yaml)
    raw_knots = mapping["knots"]
    if not isinstance(raw_knots, list) or len(raw_knots) < 2:
        raise CalibrexError("knots YAML must contain a non-empty list of knots")
    digest = sha256_path(args.knots_yaml)
    if digest is None:
        raise CalibrexError(f"knots YAML is not readable: {args.knots_yaml}")
    knots = []
    for item in raw_knots:
        timestamp = float(item["timestamp_sec"])
        translation = [float(v) for v in item["translation_m"]]
        quaternion = [float(v) for v in item["rotation_quat_xyzw"]]
        knots.append(
            ContinuousTimeKnotPose(
                timestamp_sec=timestamp,
                transform_world_body=TransformResult(
                    parent=args.world_frame,
                    child=args.body_frame,
                    translation_m=translation,
                    rotation_quat_xyzw=quaternion,
                ),
            )
        )
    contract = ContinuousTimeTrajectoryContract(
        trajectory_id=args.trajectory_id,
        world_frame=args.world_frame,
        body_frame=args.body_frame,
        interpolation=args.interpolation,
        knot_domain=ContinuousTimeKnotDomain(
            minimum_time_sec=knots[0].timestamp_sec,
            maximum_time_sec=knots[-1].timestamp_sec,
            span_sec=knots[-1].timestamp_sec - knots[0].timestamp_sec,
        ),
        clock=ContinuousTimeClockSemantics(),
        knots=knots,
        provenance=ContinuousTimeTrajectoryProvenance(
            generator=__name__,
            generator_version="0.1",
            git_commit=git_commit(),
            source_sha256=digest,
        ),
    )
    contract.save(args.output)
    _emit(
        {
            "status": "ok",
            "trajectory": str(args.output),
            "trajectory_id": contract.trajectory_id,
            "knot_count": contract.knot_count,
            "domain_sec": [
                contract.knot_domain.minimum_time_sec,
                contract.knot_domain.maximum_time_sec,
            ],
        },
        args.json,
    )
    return 0


def _cmd_trajectory_fit(args: argparse.Namespace) -> int:
    command = [
        "calibrex",
        "trajectory",
        "fit",
        str(args.trajectory),
        str(args.measurements),
        "--output",
        str(args.output),
    ]
    from calibrex.core.continuous_time_sparse import (
        ContinuousTimeTrajectoryFitOptions,
    )

    try:
        artifact = run_continuous_time_trajectory_fit(
            args.trajectory,
            args.measurements,
            result_id=args.fit_id,
            command=command,
            options=ContinuousTimeTrajectoryFitOptions(
                max_iterations=args.max_iterations,
                initial_damping=args.initial_damping,
            ),
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "fit": str(args.output),
            "fit_id": artifact.fit_id,
            "fit_status": artifact.status,
            "iterations": artifact.iterations,
            "final_objective": artifact.final_objective,
            "final_point_rmse": artifact.final_point_rmse,
            "final_point_to_plane_rmse": artifact.final_point_to_plane_rmse,
            "final_pose_rmse": artifact.final_pose_rmse,
            "final_imu_rotation_rmse_rad": artifact.final_imu_rotation_rmse_rad,
            "final_lever_arm_rmse_m_s2": artifact.final_lever_arm_rmse_m_s2,
            "gyro_bias_rad_s": artifact.gyro_bias_rad_s,
            "lever_arm_body_m": artifact.lever_arm_body_m,
            "imu_clock_offset_sec": artifact.imu_clock_offset_sec,
            "accel_bias_body_m_s2": artifact.accel_bias_body_m_s2,
            "gravity_world_m_s2": artifact.gravity_world_m_s2,
        },
        args.json,
    )
    return 0


def _cmd_trajectory_recover_point_to_plane(args: argparse.Namespace) -> int:
    from calibrex.evaluation.continuous_time_lidar_point_to_plane import (
        run_synthetic_lidar_point_to_plane_recovery,
    )

    command = [
        "calibrex",
        "trajectory",
        "recover-point-to-plane",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    try:
        artifact = run_synthetic_lidar_point_to_plane_recovery(
            seed=args.seed,
            command=command,
            recovery_id=args.recovery_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "recovery": str(args.output),
            "policy_status": artifact.policy_status,
            "fit_status": artifact.fit_status,
            "max_knot_error": artifact.max_knot_error,
            "holdout_point_to_plane_rmse_m": artifact.holdout_point_to_plane_rmse_m,
            "known_bad_rmse_delta_m": artifact.known_bad_rmse_delta_m,
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_trajectory_recover_imu_preintegration(args: argparse.Namespace) -> int:
    from calibrex.evaluation.continuous_time_imu_preintegration import (
        run_synthetic_imu_preintegration_recovery,
    )

    command = [
        "calibrex",
        "trajectory",
        "recover-imu-preintegration",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    try:
        artifact = run_synthetic_imu_preintegration_recovery(
            seed=args.seed,
            command=command,
            recovery_id=args.recovery_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "recovery": str(args.output),
            "policy_status": artifact.policy_status,
            "fit_status": artifact.fit_status,
            "max_knot_rotation_error_rad": artifact.max_knot_rotation_error_rad,
            "gyro_bias_error_rad_s": artifact.gyro_bias_error_rad_s,
            "holdout_imu_rotation_rmse_rad": artifact.holdout_imu_rotation_rmse_rad,
            "known_bad_rmse_delta_rad": artifact.known_bad_rmse_delta_rad,
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_trajectory_recover_imu_lever_arm(args: argparse.Namespace) -> int:
    from calibrex.evaluation.continuous_time_imu_lever_arm import (
        run_synthetic_imu_lever_arm_recovery,
    )

    command = [
        "calibrex",
        "trajectory",
        "recover-imu-lever-arm",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    try:
        artifact = run_synthetic_imu_lever_arm_recovery(
            seed=args.seed,
            command=command,
            recovery_id=args.recovery_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "recovery": str(args.output),
            "policy_status": artifact.policy_status,
            "fit_status": artifact.fit_status,
            "max_knot_translation_error_m": artifact.max_knot_translation_error_m,
            "lever_arm_error_m": artifact.lever_arm_error_m,
            "holdout_lever_arm_rmse_m_s2": artifact.holdout_lever_arm_rmse_m_s2,
            "known_bad_rmse_delta_m_s2": artifact.known_bad_rmse_delta_m_s2,
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_trajectory_recover_imu_clock_offset(args: argparse.Namespace) -> int:
    from calibrex.evaluation.continuous_time_imu_clock_offset import (
        run_synthetic_imu_clock_offset_recovery,
    )

    command = [
        "calibrex",
        "trajectory",
        "recover-imu-clock-offset",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    try:
        artifact = run_synthetic_imu_clock_offset_recovery(
            seed=args.seed,
            command=command,
            recovery_id=args.recovery_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "recovery": str(args.output),
            "policy_status": artifact.policy_status,
            "fit_status": artifact.fit_status,
            "max_knot_rotation_error_rad": artifact.max_knot_rotation_error_rad,
            "imu_clock_offset_error_sec": artifact.imu_clock_offset_error_sec,
            "holdout_imu_rotation_rmse_rad": artifact.holdout_imu_rotation_rmse_rad,
            "known_bad_rmse_delta_rad": artifact.known_bad_rmse_delta_rad,
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_trajectory_recover_imu_accel_bias(args: argparse.Namespace) -> int:
    from calibrex.evaluation.continuous_time_imu_accel_bias import (
        run_synthetic_imu_accel_bias_recovery,
    )

    command = [
        "calibrex",
        "trajectory",
        "recover-imu-accel-bias",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    try:
        artifact = run_synthetic_imu_accel_bias_recovery(
            seed=args.seed,
            command=command,
            recovery_id=args.recovery_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "recovery": str(args.output),
            "policy_status": artifact.policy_status,
            "fit_status": artifact.fit_status,
            "max_knot_translation_error_m": artifact.max_knot_translation_error_m,
            "accel_bias_error_m_s2": artifact.accel_bias_error_m_s2,
            "gravity_error_m_s2": artifact.gravity_error_m_s2,
            "holdout_lever_arm_rmse_m_s2": artifact.holdout_lever_arm_rmse_m_s2,
            "known_bad_rmse_delta_m_s2": artifact.known_bad_rmse_delta_m_s2,
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_trajectory_recover_imu_intrinsics(args: argparse.Namespace) -> int:
    from calibrex.evaluation.continuous_time_imu_intrinsics import (
        run_synthetic_imu_intrinsics_recovery,
    )

    command = [
        "calibrex",
        "trajectory",
        "recover-imu-intrinsics",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    try:
        artifact = run_synthetic_imu_intrinsics_recovery(
            seed=args.seed,
            command=command,
            recovery_id=args.recovery_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "recovery": str(args.output),
            "policy_status": artifact.policy_status,
            "fit_status": artifact.fit_status,
            "gyro_scale_error": artifact.gyro_scale_error,
            "accel_scale_error": artifact.accel_scale_error,
            "holdout_imu_rotation_rmse_rad": artifact.holdout_imu_rotation_rmse_rad,
            "holdout_lever_arm_rmse_m_s2": artifact.holdout_lever_arm_rmse_m_s2,
            "known_bad_imu_rmse_delta_rad": artifact.known_bad_imu_rmse_delta_rad,
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_lifecycle_simulate(args: argparse.Namespace) -> int:
    from calibrex.evaluation.calibration_lifecycle import (
        run_synthetic_calibration_lifecycle,
    )

    command = [
        "calibrex",
        "lifecycle",
        "simulate",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    try:
        artifact = run_synthetic_calibration_lifecycle(
            seed=args.seed,
            command=command,
            lifecycle_id=args.lifecycle_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "lifecycle": str(args.output),
            "policy_status": artifact.policy_status,
            "adoption_count": artifact.adoption_count,
            "rejection_count": artifact.rejection_count,
            "rollback_count": artifact.rollback_count,
            "weak_observability_installed_delta_m": (artifact.weak_observability_installed_delta_m),
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_lifecycle_init(args: argparse.Namespace) -> int:
    try:
        registry = init_registry(
            args.registry_root,
            registry_id=args.registry_id,
            operator=args.operator,
            reason=args.reason,
            timestamp=args.timestamp,
            command=["calibrex", "lifecycle", "init"],
        )
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "registry_root": str(registry.root),
            "registry_id": registry.manifest().registry_id,
            "schema_version": registry.manifest().schema_version,
        },
        args.json,
    )
    return 0


def _cmd_lifecycle_register_sensor(args: argparse.Namespace) -> int:
    try:
        event = register_sensor(
            args.registry_root,
            sensor_id=args.sensor_id,
            vehicle_id=args.vehicle_id,
            sensor_kit_id=args.sensor_kit_id,
            serial=args.serial,
            model=args.model,
            firmware=args.firmware,
            mount=args.mount,
            frame=args.frame,
            install=args.install,
            operator=args.operator,
            reason=args.reason,
            timestamp=args.timestamp,
            command=["calibrex", "lifecycle", "register-sensor"],
        )
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(_lifecycle_event_summary(event), args.json)
    return 0


def _cmd_lifecycle_capture(args: argparse.Namespace) -> int:
    try:
        event = record_capture(
            args.registry_root,
            capture_manifest=args.capture_manifest,
            operator=args.operator,
            reason=args.reason,
            timestamp=args.timestamp,
            command=["calibrex", "lifecycle", "capture"],
        )
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(_lifecycle_event_summary(event), args.json)
    return 0 if event.status == "CAPTURED" else 2


def _cmd_lifecycle_register_edge(args: argparse.Namespace) -> int:
    try:
        event = register_calibration_edge(
            args.registry_root,
            edge_id=args.edge_id,
            vehicle_id=args.vehicle_id,
            sensor_kit_id=args.sensor_kit_id,
            parent_frame=args.parent_frame,
            child_frame=args.child_frame,
            incumbent_transform=args.incumbent_transform,
            incumbent_result=args.incumbent_result,
            operator=args.operator,
            reason=args.reason,
            timestamp=args.timestamp,
            command=["calibrex", "lifecycle", "register-edge"],
        )
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(_lifecycle_event_summary(event), args.json)
    return 0


def _cmd_lifecycle_evaluate(args: argparse.Namespace) -> int:
    try:
        artifact = evaluate_lifecycle(
            args.registry_root,
            edge_ids=args.edge_id,
            capture_manifest=args.capture_manifest,
            candidate_result=args.candidate_result,
            candidate_transform=args.candidate_transform,
            candidate_pilot=args.candidate_pilot,
            evidence=args.evidence,
            assessment=args.assessment,
            promotion=args.promotion,
            smoke=args.smoke,
            policy=args.policy,
            output=args.output,
            operator=args.operator,
            reason=args.reason,
            timestamp=args.timestamp,
            command=["calibrex", "lifecycle", "evaluate"],
        )
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": artifact.status,
            "admission": artifact.admission,
            "evaluation_id": artifact.evaluation_id,
            "evaluation": str(args.output) if args.output else None,
            "reason": artifact.reason,
            "schema_version": artifact.schema_version,
        },
        args.json,
    )
    return 0 if artifact.status == "PASS" and artifact.admission == "ADOPT" else 2


def _cmd_lifecycle_promote(args: argparse.Namespace) -> int:
    try:
        event = promote_lifecycle(
            args.registry_root,
            edge_ids=args.edge_id,
            evaluation=args.evaluation,
            promotion=args.promotion,
            smoke=args.smoke,
            operator=args.operator,
            reason=args.reason,
            timestamp=args.timestamp,
            command=["calibrex", "lifecycle", "promote"],
        )
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(_lifecycle_event_summary(event), args.json)
    return 0


def _cmd_lifecycle_rollback(args: argparse.Namespace) -> int:
    try:
        event = rollback_lifecycle(
            args.registry_root,
            edge_id=args.edge_id,
            target_sequence=args.target_sequence,
            target_event_sha256=args.target_event_sha256,
            promotion=args.promotion,
            operator=args.operator,
            reason=args.reason,
            timestamp=args.timestamp,
            command=["calibrex", "lifecycle", "rollback"],
        )
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(_lifecycle_event_summary(event), args.json)
    return 0


def _cmd_lifecycle_verify(args: argparse.Namespace) -> int:
    try:
        report = verify_registry(
            args.registry_root,
            verify_sources=not args.no_verify_sources,
        )
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(report.model_dump(mode="json"), args.json)
    return 0 if report.valid else 1


def _cmd_lifecycle_status(args: argparse.Namespace) -> int:
    try:
        status = load_registry(args.registry_root).status()
    except (OSError, ValueError, CalibrexError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(status.model_dump(mode="json"), args.json)
    return 0


def _lifecycle_event_summary(event: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "event_type": event.event_type,
        "event_status": event.status,
        "admission": event.admission,
        "sequence": event.sequence,
        "event_sha256": event.event_sha256,
        "previous_event_sha256": event.previous_event_sha256,
        "registry_id": event.registry_id,
        "edge_ids": event.edge_ids,
    }


def _cmd_trajectory_recover_sliding_window(args: argparse.Namespace) -> int:
    from calibrex.evaluation.continuous_time_sliding_window import (
        run_synthetic_sliding_window_recovery,
    )

    command = [
        "calibrex",
        "trajectory",
        "recover-sliding-window",
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    try:
        artifact = run_synthetic_sliding_window_recovery(
            seed=args.seed,
            command=command,
            recovery_id=args.recovery_id,
        )
        artifact.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "ok",
            "recovery": str(args.output),
            "policy_status": artifact.policy_status,
            "batch_fit_status": artifact.batch_fit_status,
            "sliding_fit_status": artifact.sliding_fit_status,
            "max_overlap_translation_error_m": artifact.max_overlap_translation_error_m,
            "sliding_holdout_rmse_m": artifact.sliding_holdout_rmse_m,
            "known_bad_rmse_delta_m": artifact.known_bad_rmse_delta_m,
        },
        args.json,
    )
    return 0 if artifact.policy_status != "fail" else 2


def _cmd_camera_lidar_audit_external_baselines(
    args: argparse.Namespace,
) -> int:
    command = [
        "calibrex",
        "camera-lidar",
        "audit-external-baselines",
        str(args.problem),
        str(args.protocol),
        "--output-dir",
        str(args.output_dir),
        "--koide-source-commit",
        args.koide_source_commit,
        "--unicalib-source-commit",
        args.unicalib_source_commit,
    ]
    try:
        artifacts = materialize_camera_lidar_external_baseline_audit(
            args.problem,
            args.protocol,
            output_directory=args.output_dir,
            koide_source_commit=args.koide_source_commit,
            unicalib_source_commit=args.unicalib_source_commit,
            command=command,
        )
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": "incomplete",
            "comparable_external_baseline_count": 0,
            "koide_status": artifacts.koide.status,
            "koide_audit": str(artifacts.koide_path),
            "unicalib_status": artifacts.unicalib.status,
            "unicalib_audit": str(artifacts.unicalib_path),
            "release_gate": "blocked",
        },
        args.json,
    )
    return 2


def _cmd_visualize(args: argparse.Namespace) -> int:
    result = load_result(args.result)
    if args.reference_result:
        _merge_reference_result(result, load_result(args.reference_result))
    output_dir = args.output_dir or args.result.parent
    html_path = output_dir / "report.html"
    if args.export_html:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_camera_lidar_overlay_artifact(result, output_dir / "artifacts")
        write_rig_3d_artifact(result, output_dir / "artifacts")
        report_artifacts = write_report_artifacts(result, output_dir)
    else:
        report_artifacts = {}
    payload = {
        "status": "ok",
        "html_report": str(html_path) if args.export_html else result.artifacts.html_report,
        "rig_3d_viewer": result.artifacts.rig_3d_viewer,
        "camera_lidar_overlay": result.artifacts.camera_lidar_overlay,
        "report_artifacts": report_artifacts,
    }
    _emit(payload, args.json)
    return 0


def _cmd_capture_inspect(args: argparse.Namespace) -> int:
    sensors: list[SensorIdentity] = []
    for specification in args.sensor:
        try:
            sensors.append(_parse_capture_sensor(specification))
        except ValueError as exc:
            _die(str(exc))
    command = ["calibrex", "capture", "inspect", str(args.path)]
    if args.type != "auto":
        command.extend(["--type", str(args.type)])
    for option, value in (
        ("--capture-id", args.capture_id),
        ("--session-id", args.session_id),
        ("--vehicle-id", args.vehicle_id),
        ("--sensor-kit-id", args.sensor_kit_id),
    ):
        if value is not None:
            command.extend([option, str(value)])
    for specification in args.sensor:
        command.extend(["--sensor", specification])
    if args.readiness_profile != "strict":
        command.extend(["--readiness-profile", str(args.readiness_profile)])
    for stream_id in args.required_stream:
        command.extend(["--required-stream", stream_id])
    for stream_id in args.optional_stream:
        command.extend(["--optional-stream", stream_id])
    for stream_kind in args.required_kind:
        command.extend(["--required-kind", stream_kind])
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    if args.sample_limit != 4:
        command.extend(["--sample-limit", str(args.sample_limit)])
    if args.output is not None:
        command.extend(["--output", str(args.output)])
    if args.json:
        command.append("--json")
    manifest = inspect_capture(
        args.path,
        source_format=str(args.type),
        capture_id=args.capture_id,
        session_id=args.session_id,
        vehicle_id=args.vehicle_id,
        sensor_kit_id=args.sensor_kit_id,
        sensors=sensors,
        config_path=args.config,
        command=command,
        readiness_profile=cast(Literal["strict", "declared"], args.readiness_profile),
        required_streams=tuple(args.required_stream),
        optional_streams=tuple(args.optional_stream),
        required_stream_kinds=tuple(args.required_kind),
        sample_limit=args.sample_limit,
    )
    if args.output is not None:
        manifest.save(args.output)
    payload = manifest.model_dump(mode="json", exclude_none=False)
    if args.json:
        _emit(payload, as_json=True)
    else:
        print(f"Capture manifest: {manifest.status.upper()}")
        print(f"  source: {args.path}")
        print(f"  streams: {len(manifest.streams)}")
        print(f"  sensors: {len(manifest.sensors)}")
        if manifest.source is not None and manifest.source.mcap_integrity is not None:
            print(f"  mcap_integrity: {manifest.source.mcap_integrity.status}")
        print(f"  summary: {manifest.summary}")
        for check in manifest.checks:
            print(f"  {check.check_id}: {check.status} - {check.reason}")
        for action in manifest.actions:
            print(f"  next: {action.message}")
        print(f"  artifact_sha256: {manifest.artifact_sha256}")
        if args.output is not None:
            print(f"  artifact: {args.output}")
    return 0


def _cmd_capture_verify(args: argparse.Namespace) -> int:
    """Verify a capture manifest and its declared external inputs."""

    try:
        report = verify_capture_manifest_inputs(args.manifest)
    except (OSError, ValueError) as exc:
        _die(str(exc))
    payload = report.model_dump(mode="json", exclude_none=False)
    if args.json:
        _emit(payload, as_json=True)
    else:
        print(f"Capture manifest verification: {'PASS' if report.valid else 'FAIL'}")
        print(f"  manifest: {args.manifest}")
        print(f"  self_digest: {report.self_digest_status}")
        for item in report.inputs:
            print(f"  {item.role}:{item.identifier}: {item.status} - {item.reason}")
        print(f"  summary: {report.summary}")
    return 0 if report.valid else 1


def _parse_capture_sensor(specification: str) -> SensorIdentity:
    fields = specification.split(":")
    if len(fields) < 2 or not fields[0] or not fields[1]:
        raise ValueError("--sensor must be ID:TYPE[:SERIAL[:MODEL[:FIRMWARE[:MOUNT[:FRAME]]]]]")
    if len(fields) > 7:
        raise ValueError("--sensor accepts at most seven colon-separated fields")
    values = fields + [None] * (7 - len(fields))
    try:
        return SensorIdentity(
            sensor_id=values[0] or "",
            type=cast(SensorType, values[1]),
            serial=values[2],
            model=values[3],
            firmware=values[4],
            mount_id=values[5],
            frame_id=values[6],
        )
    except ValueError as exc:
        raise ValueError(f"invalid --sensor {specification!r}: {exc}") from exc


def _cmd_inspect(args: argparse.Namespace) -> int:
    dataset_type = cast(DatasetType, str(args.type).replace("-", "_"))
    inspection = inspect_dataset(
        DatasetConfig(
            type=dataset_type,
            path=str(args.path),
            sample_limit=args.sample_limit,
        )
    )
    if args.json:
        _emit(inspection.as_dict(), as_json=True)
    else:
        _emit_inspection(inspection)
    return 0


def _promotion_policy_from_args(args: argparse.Namespace) -> AutowarePromotionPolicy:
    return AutowarePromotionPolicy(
        base_frame=args.base_frame,
        sensor_kit_base_frame=args.sensor_kit_base_frame,
        allowed_frames=list(args.allowed_frame),
        allowed_topics=list(args.allowed_topic),
        allowed_files=list(args.allowed_file),
        required_files=list(args.required_file),
        allow_warnings=bool(args.allow_warnings),
        profile=args.profile,
        require_smoke_for_apply=bool(args.require_smoke_for_apply),
    )


def _cmd_autoware_promotion_plan(args: argparse.Namespace) -> int:
    roots = AutowarePromotionRoots(
        workspace_root=args.workspace_root,
        package_root=args.package_root,
        individual_params_root=args.individual_params_root,
        sensor_kit_description_root=args.sensor_kit_description_root,
    )
    artifact = build_autoware_promotion_plan(
        args.candidate,
        roots=roots,
        vehicle_id=args.vehicle_id,
        sensor_kit_id=args.sensor_kit_id,
        baseline_manifest_sha256=args.baseline_manifest_sha256,
        policy=_promotion_policy_from_args(args),
        command=["calibrex", "autoware", "promotion", "plan", str(args.candidate)],
    )
    artifact.save(args.output)
    payload = artifact.model_dump(mode="json", exclude_none=False)
    payload["plan"] = str(args.output)
    summary = {
        "plan": str(args.output),
        "status": artifact.status,
        "decision": artifact.decision,
        "reason": artifact.reason,
    }
    _emit(payload if args.json else summary, args.json)
    return 0 if artifact.status == "PASS" else 1


def _cmd_autoware_promotion_verify(args: argparse.Namespace) -> int:
    artifact = verify_autoware_promotion(
        args.plan, command=["calibrex", "autoware", "promotion", "verify", str(args.plan)]
    )
    # Verification is a read-only operation by default.  Persist a refreshed
    # artifact only when the caller explicitly supplies --output.
    destination = args.output
    if destination is not None:
        artifact.save(destination)
    payload = artifact.model_dump(mode="json", exclude_none=False)
    payload["plan"] = str(destination or args.plan)
    summary = {
        "plan": str(destination or args.plan),
        "status": artifact.status,
        "decision": artifact.decision,
        "reason": artifact.reason,
    }
    _emit(payload if args.json else summary, args.json)
    return 0 if artifact.status == "PASS" else 1


def _cmd_autoware_promotion_apply(args: argparse.Namespace) -> int:
    artifact = apply_autoware_promotion(
        args.plan,
        smoke_artifact=args.smoke_artifact,
        output_path=args.output,
    )
    destination = args.output or args.plan
    payload = artifact.model_dump(mode="json", exclude_none=False)
    payload["plan"] = str(destination)
    summary = {
        "plan": str(destination),
        "status": artifact.status,
        "decision": artifact.decision,
        "application": artifact.application.status,
    }
    _emit(payload if args.json else summary, args.json)
    return 0


def _cmd_autoware_promotion_rollback(args: argparse.Namespace) -> int:
    artifact = rollback_autoware_promotion(args.plan, output_path=args.output)
    destination = args.output or args.plan
    payload = artifact.model_dump(mode="json", exclude_none=False)
    payload["plan"] = str(destination)
    summary = {
        "plan": str(destination),
        "status": artifact.status,
        "decision": artifact.decision,
        "application": artifact.application.status,
    }
    _emit(payload if args.json else summary, args.json)
    return 0


def _cmd_autoware_smoke_run(args: argparse.Namespace) -> int:
    commands: list[AutowareSmokeCommand] = []
    for specification in args.stage_command:
        if "=" not in specification:
            _die("--stage-command must be STAGE=JSON_ARGV")
        stage, raw_argv = specification.split("=", 1)
        try:
            argv = json.loads(raw_argv)
        except json.JSONDecodeError as exc:
            _die(f"invalid --stage-command JSON: {exc}")
        if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
            _die("--stage-command JSON must be an array of strings")
        commands.append(AutowareSmokeCommand(stage=stage, argv=argv))
    try:
        artifact = run_autoware_smoke(
            args.plan,
            config=AutowareSmokeConfig(
                mode=args.mode,
                profile=args.profile,
                commands=commands,
                container_digest=args.container_digest,
                environment_digest=args.environment_digest,
                tool_digest=args.tool_digest,
                precomputed_path=args.precomputed,
                policy=AutowareSmokePolicy(
                    profile=args.profile,
                    stage_timeout_seconds=args.timeout,
                ),
                command=["calibrex", "autoware", "smoke", "run", str(args.plan)],
            ),
            output_path=args.output,
        )
    except (CalibrexError, ValueError, OSError) as exc:
        _die(f"Autoware smoke failed: {exc}")
    payload = artifact.model_dump(mode="json", exclude_none=False)
    payload["artifact"] = str(args.output)
    summary = {
        "artifact": str(args.output),
        "status": artifact.status,
        "admission_label": artifact.admission_label,
        "reason": artifact.reason,
    }
    _emit(payload if args.json else summary, args.json)
    return 0 if artifact.status == "PASS" else 1


def _cmd_autoware_smoke_import(args: argparse.Namespace) -> int:
    try:
        artifact = import_autoware_smoke(
            args.artifact,
            promotion=load_autoware_promotion(args.plan),
            output_path=args.output,
        )
    except (CalibrexError, ValueError, OSError) as exc:
        _die(f"Autoware smoke import failed: {exc}")
    payload = artifact.model_dump(mode="json", exclude_none=False)
    payload["artifact"] = str(args.output)
    summary = {
        "artifact": str(args.output),
        "status": artifact.status,
        "admission_label": artifact.admission_label,
        "reason": artifact.reason,
    }
    _emit(payload if args.json else summary, args.json)
    return 0 if artifact.status == "PASS" else 1


def _cmd_autoware_smoke_verify(args: argparse.Namespace) -> int:
    try:
        artifact = verify_autoware_smoke(
            args.artifact,
            promotion=load_autoware_promotion(args.plan),
        )
    except (CalibrexError, ValueError, OSError) as exc:
        _die(f"Autoware smoke verification failed: {exc}")
    payload = artifact.model_dump(mode="json", exclude_none=False)
    payload["artifact"] = str(args.artifact)
    summary = {
        "artifact": str(args.artifact),
        "status": artifact.status,
        "admission_label": artifact.admission_label,
        "reason": artifact.reason,
    }
    _emit(payload if args.json else summary, args.json)
    return 0 if artifact.status == "PASS" else 1


def _cmd_export(args: argparse.Namespace) -> int:
    if args.format == "autoware":
        try:
            config = AutowareExportConfig(
                base_frame=args.base_frame,
                sensor_frames=list(args.sensor_frames),
                transform_names=list(args.transform_names),
                invert_transforms=list(args.invert_transforms),
            )
        except ValueError as exc:
            _die(f"invalid Autoware export configuration: {exc}")
        static_tf_output = args.static_tf_output or args.output.with_name(
            f"{args.output.stem}.static_tf.launch.py"
        )
        manifest_output = args.manifest_output or args.output.with_name(
            f"{args.output.stem}.manifest.yaml"
        )
        if args.kind == "continuous-time-lidar-pair":
            artifact = ContinuousTimeLidarPairArtifact.model_validate(read_mapping(args.result))
            transforms = {artifact.variable: artifact.refined_transform}
            export_artifact = build_autoware_export(
                transforms,
                config=config,
                source_path=args.result,
                source_run=artifact.variable,
                quality_grade=artifact.refined_transform.quality.grade,
                command=["calibrex", "export", str(args.result), "--format", "autoware"],
            )
        else:
            result = load_result(args.result)
            export_artifact = build_autoware_export(
                result,
                config=config,
                source_path=args.result,
                command=["calibrex", "export", str(args.result), "--format", "autoware"],
            )
        paths = write_autoware_export(
            export_artifact,
            args.output,
            static_tf_output=static_tf_output,
            manifest_output=manifest_output,
            overwrite=args.force,
        )
        summary = {
            "status": "ok",
            "calibration": str(paths["calibration"]),
            "static_tf": str(paths["static_tf"]),
            "manifest": str(paths["manifest"]),
            "artifact_sha256": export_artifact.artifact_sha256,
            "source_result_sha256": export_artifact.provenance.source_result_sha256,
            "schema_version": export_artifact.schema_version,
        }
        if args.json:
            _emit(summary, as_json=True)
        else:
            print(f"wrote {paths['calibration']}")
            print(f"wrote {paths['static_tf']}")
            print(f"wrote {paths['manifest']}")
        return 0
    if args.kind == "continuous-time-lidar-pair":
        artifact = ContinuousTimeLidarPairArtifact.model_validate(read_mapping(args.result))
        transforms = {artifact.variable: artifact.refined_transform}
        if args.format == "ros-tf":
            export_ros_tf_transforms(transforms, args.output)
        else:
            _die(f"unsupported export format: {args.format}")
    else:
        result = load_result(args.result)
        if args.format == "ros-tf":
            export_ros_tf_yaml(result, args.output)
        else:
            _die(f"unsupported export format: {args.format}")
    print(f"wrote {args.output}")
    return 0


def _merge_reference_result(result: CalibrationResult, reference: CalibrationResult) -> None:
    result.reference_extrinsics.update(reference.reference_extrinsics)
    result.reference_extrinsics.update(reference.transforms)
    result.run.provenance["visualization_reference_run_id"] = reference.run.id


def _starter_config(profile: str) -> dict[str, Any]:
    domain = "autonomous_driving" if profile == "autonomous-driving-rig" else "robotics"
    sensors: dict[str, Any] = {
        "camera0": {
            "type": "camera",
            "model": "pinhole",
            "topic": "/camera/front/image",
            "camera_info_topic": "/camera/front/camera_info",
            "intrinsics": {
                "estimate": False,
                "fx": 720.0,
                "fy": 720.0,
                "cx": 640.0,
                "cy": 360.0,
                "distortion_model": "radtan",
                "distortion": [0.0, 0.0, 0.0, 0.0],
            },
        },
        "lidar0": {
            "type": "lidar",
            "model": "spinning",
            "topic": "/lidar/points",
            "fields": ["x", "y", "z", "intensity"],
        },
        "imu0": {
            "type": "imu",
            "model": "six_axis",
            "topic": "/imu/data",
            "noise": {
                "gyro_noise_density": 0.0002,
                "accel_noise_density": 0.002,
                "gyro_random_walk": 0.00001,
                "accel_random_walk": 0.0001,
            },
        },
    }
    if profile == "autonomous-driving-rig":
        sensors["radar0"] = {
            "type": "radar",
            "model": "automotive_4d",
            "topic": "/radar/front/detections",
        }

    frames: dict[str, Any] = {
        "base": {"root": True},
        "camera0": _child_frame("base", [0.3, 0.0, 0.2]),
        "lidar0": _child_frame("base", [0.0, 0.0, 0.4]),
        "imu0": _child_frame("base", [0.0, 0.0, 0.0]),
    }
    if profile == "autonomous-driving-rig":
        frames["radar0"] = _child_frame("base", [0.8, 0.0, 0.35])

    return {
        "schema_version": "slac.config/v0.1",
        "project": {
            "name": f"{profile}_calibration",
            "description": "Starter Calibrex calibration config",
            "output_dir": "outputs/example",
            "domain": domain,
        },
        "dataset": {
            "type": "filesystem",
            "path": "examples/synthetic_camera_lidar_imu",
            "time_base": "sensor_time_ns",
        },
        "sensors": sensors,
        "frames": frames,
        "time_offsets": {
            "camera0": {"estimate": True, "initial_sec": 0.0, "prior_sigma_sec": 0.01},
            "lidar0": {"estimate": False, "initial_sec": 0.0},
        },
        "pipeline": {
            "type": "multi_sensor_slac",
            "frontends": ["imu_preintegration", "lidar_surfel_map", "camera_features"],
            "factors": {
                "imu_preintegration": {"enabled": True},
                "lidar_point_to_surfel": {"enabled": True},
                "camera_reprojection": {"enabled": True},
            },
        },
        "solver": {
            "backend": "scipy",
            "max_iterations": 50,
            "robust_loss": "huber",
            "convergence_tolerance": 1.0e-6,
        },
        "evaluation": {
            "holdout_ratio": 0.2,
            "kitti": {
                "max_projection_pairs": 3,
                "projection_sample_points": 800,
                "perturbation_rotation_deg": [0.5, 1.0],
                "perturbation_translation_m": [0.05, 0.10],
            },
            "metrics": [
                "reprojection_rmse_px",
                "lidar_point_to_plane_rmse_m",
                "trajectory_consistency",
                "observability_rank",
            ],
        },
        "outputs": {
            "result": "result.yaml",
            "report": "report.html",
            "artifacts_dir": "artifacts",
        },
    }


def _child_frame(parent: str, translation: list[float]) -> dict[str, Any]:
    return {
        "parent": parent,
        "transform": {
            "estimate": True,
            "initial": {
                "translation": translation,
                "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "prior_sigma": {
                "translation_m": 0.2,
                "rotation_deg": 10.0,
            },
        },
    }


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _emit_verification(verification: EvidenceBundleVerification) -> None:
    summary = verification.verification_summary
    if verification.path == verification.source_bundle.path:
        print(f"bundle: {verification.path}")
    else:
        print(f"verification: {verification.path}")
        print(f"source_bundle: {verification.source_bundle.path}")
    print(f"valid: {_format_bool(verification.valid)}")
    print(f"source_bundle_sha256: {verification.source_bundle.sha256}")
    if verification.raw_recomputed_required:
        print("raw_recomputed_required: yes")
    if verification.primary_evidence_materialization is not None:
        materialization = verification.primary_evidence_materialization
        print(
            "primary_evidence: "
            f"metrics_origin={materialization.metrics_origin}, "
            f"data_verified={_format_optional_bool(materialization.data_verified)}"
        )
    print(f"artifacts: {len(verification.checked_artifacts)}/{verification.artifact_count}")
    print(f"input_files: {verification.checked_input_file_count}/{verification.input_file_count}")
    print(
        "claims: "
        f"total={summary.total}, "
        f"ok={summary.ok}, "
        f"failed={summary.failed}, "
        f"skipped={summary.skipped}"
    )
    if summary.by_scope:
        print("claim_scopes:")
        for scope, count in sorted(summary.by_scope.items()):
            print(f"  {scope}: {count}")
    if verification.issues:
        print("issues:")
        for issue in verification.issues:
            print(f"  - {issue}")


def _emit_inspection(inspection: DatasetInspection) -> None:
    print("Dataset")
    print(f"  type: {inspection.dataset_type}")
    print(f"  path: {inspection.path}")
    print(f"  exists: {_format_bool(inspection.exists)}")
    if inspection.manifest:
        print(f"  manifest: {inspection.manifest}")

    print("Streams")
    if not inspection.streams:
        print("  none")
    for stream in inspection.streams:
        count = "unknown" if stream.message_count is None else str(stream.message_count)
        topic = f", topic={stream.topic}" if stream.topic else ""
        sensor = f", sensor={stream.sensor}" if stream.sensor else ""
        print(f"  - {stream.name} ({stream.kind}): {count} messages{topic}{sensor}")

    print("Warnings")
    if not inspection.warnings:
        print("  none")
    for warning in inspection.warnings:
        print(f"  - {warning}")

    _emit_diagnostics(inspection)


def _emit_diagnostics(inspection: DatasetInspection) -> None:
    if not inspection.diagnostics:
        return
    print("Diagnostics")
    for name, diagnostic in sorted(inspection.diagnostics.items()):
        if not isinstance(diagnostic, dict):
            print(f"  {name}: {diagnostic}")
            continue
        print(f"  {name}:")
        for key in _ordered_diagnostic_keys(diagnostic):
            value = diagnostic.get(key)
            if value is not None and value != []:
                print(f"    {key}: {_format_value(value)}")

    lidar_diagnostic_keys = {"velodyne_points", "livox_pcd", "rosbag1"}
    if lidar_diagnostic_keys & set(inspection.diagnostics):
        _emit_lidar_quality_hint(inspection)


def _emit_lidar_quality_hint(inspection: DatasetInspection) -> None:
    domain = "autonomous_driving" if inspection.dataset_type == "kitti_raw" else "robotics"
    metrics = lidar_metrics_from_inspection(inspection)
    metrics.update(motion_metrics_from_inspection(inspection))
    metrics.update(timing_metrics_from_inspection(inspection))
    apply_metric_thresholds(
        metrics,
        "autonomous_driving" if domain == "autonomous_driving" else "default",
    )
    degeneracy = degeneracy_from_inspection(inspection)
    recommendations = build_inspection_recommendations(metrics, degeneracy, domain=domain)

    print("LiDAR Quality")
    print(f"  degeneracy: {degeneracy.grade.upper()}")
    if degeneracy.reason:
        print(f"  reason: {degeneracy.reason}")
    if metrics:
        print("  metrics:")
        for name, metric in sorted(metrics.items()):
            value = metric.holdout if metric.holdout is not None else metric.value
            if value is None:
                value = metric.train
            suffix = f" {metric.unit}" if metric.unit else ""
            print(f"    - {name}: {metric.grade.upper()} ({_format_value(value)}{suffix})")
    print("  recommendations:")
    for recommendation in recommendations:
        print(f"    - {recommendation}")


def _ordered_diagnostic_keys(diagnostic: dict[str, object]) -> list[str]:
    preferred = [
        "frame_count",
        "status",
        "sampled_frame_count",
        "train_frame_count",
        "holdout_frame_count",
        "sampled_point_count",
        "min_points_per_frame",
        "max_points_per_frame",
        "bounds_min_m",
        "bounds_max_m",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
        "voxel_size_m",
        "planarity_min_points_per_voxel",
        "planarity_voxel_count",
        "local_planarity_mean",
        "roughness_mean_m",
        "map_sharpness_score",
        "point_to_plane_rmse_train_m",
        "point_to_plane_rmse_holdout_m",
        "point_to_plane_median_holdout_m",
        "point_to_plane_p95_holdout_m",
        "train_residual_count",
        "holdout_residual_count",
        "pair_target_transform_applied",
        "pair_transform_convention",
        "pair_voxel_size_m",
        "pair_source_voxel_count",
        "pair_target_voxel_count",
        "pair_shared_voxel_count",
        "pair_unmatched_source_voxel_count",
        "pair_unmatched_target_voxel_count",
        "pair_source_voxel_recall_in_target",
        "pair_target_voxel_recall_in_source",
        "pair_shared_voxel_centroid_rmse_m",
        "packet_count",
        "timestamp_count",
        "duration_sec",
        "mean_speed_mps",
        "max_speed_mps",
        "speed_range_mps",
        "yaw_excitation_deg",
        "pitch_excitation_deg",
        "roll_excitation_deg",
        "mean_acceleration_norm_mps2",
        "camera_lidar_pair_count",
        "camera_lidar_mean_abs_dt_ms",
        "camera_lidar_max_abs_dt_ms",
        "lidar_oxts_pair_count",
        "lidar_oxts_mean_abs_dt_ms",
        "lidar_oxts_max_abs_dt_ms",
        "camera_timestamp_count",
        "lidar_timestamp_count",
        "oxts_timestamp_count",
        "malformed_files",
    ]
    known = [key for key in preferred if key in diagnostic]
    extra = sorted(key for key in diagnostic if key not in set(preferred))
    return known + extra


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _emit_comparison(
    comparison: ResultComparison,
    *,
    enforce_compatible: bool = False,
) -> None:
    summary = comparison.summary
    print(f"left: {comparison.left.run_id} ({comparison.left.grade})")
    print(f"left_materialization: {_format_comparison_materialization(comparison.left)}")
    print(f"right: {comparison.right.run_id} ({comparison.right.grade})")
    print(f"right_materialization: {_format_comparison_materialization(comparison.right)}")
    print(f"metrics: {summary.metric_comparison_count}")
    print(
        "metric_winners: "
        f"left={summary.left_better_metric_count}, "
        f"right={summary.right_better_metric_count}, "
        f"tie={summary.tied_metric_count}, "
        f"not_comparable={summary.not_comparable_metric_count}"
    )
    print(f"transforms: {summary.transform_comparison_count}")
    if summary.max_translation_delta_m is not None:
        print(f"max_translation_delta_m: {summary.max_translation_delta_m:.6g}")
    if summary.max_rotation_delta_deg is not None:
        print(f"max_rotation_delta_deg: {summary.max_rotation_delta_deg:.6g}")
    if comparison.observability.only_right_weak_directions:
        print(f"new_weak_directions: {comparison.observability.only_right_weak_directions}")
    protocol = comparison.protocol_compatibility
    print(f"protocol_compatibility: {protocol.status}")
    print(f"protocol_compatibility_enforced: {_format_bool(enforce_compatible)}")
    if protocol.reasons:
        print(f"protocol_notes: {protocol.reasons}")
    if comparison.evidence_comparisons:
        winners = [item.winner for item in comparison.evidence_comparisons]
        print(f"evidence_checks: {len(comparison.evidence_comparisons)}")
        print(
            "evidence_winners: "
            f"left={winners.count('left')}, "
            f"right={winners.count('right')}, "
            f"tie={winners.count('tie')}, "
            f"not_comparable={winners.count('not_comparable')}"
        )
    not_comparable_reasons = _comparison_not_comparable_reasons(comparison)
    if not_comparable_reasons:
        print("not_comparable_reasons:")
        for reason, count in not_comparable_reasons:
            print(f"  - {reason} ({count})")


def _emit_report_comparison(
    report: ReportComparison,
    *,
    enforce_compatible: bool = False,
) -> None:
    summary = report.summary
    print(f"entries: {summary.entry_count}")
    print(f"comparison_mode: {summary.comparison_mode}")
    if summary.reference_label is not None:
        print(f"reference: {summary.reference_label}")
    for label, entry in report.entries.items():
        provenance = entry.provenance
        role = provenance.dominant_role or "n/a"
        reference_marker = " [reference]" if entry.is_reference else ""
        print(
            f"  {label}{reference_marker}: run_id={entry.side.run_id}, "
            f"grade={entry.side.grade}, "
            f"producer={provenance.dominant_producer}, "
            f"role={role}, "
            f"evidence_level={provenance.dominant_evidence_level}"
        )
    print(f"pairwise_comparisons: {summary.pairwise_comparison_count}")
    print(f"protocol_compatibility: {summary.protocol_compatibility_status}")
    print(f"protocol_compatibility_enforced: {_format_bool(enforce_compatible)}")
    print(
        "pair_compatibility: "
        f"compatible={summary.compatible_pair_count}, "
        f"warning={summary.warning_pair_count}, "
        f"not_comparable={summary.not_comparable_pair_count}"
    )
    if summary.not_comparable_pairs:
        print("not_comparable_pairs:")
        for pair_key in summary.not_comparable_pairs:
            pair = report.pairwise[pair_key]
            reasons = pair.comparison.protocol_compatibility.reasons
            suffix = f": {'; '.join(reasons)}" if reasons else ""
            print(f"  - {pair.left_label} vs {pair.right_label}{suffix}")
    if report.metric_family_rankings:
        print("metric_family_rankings:")
        for family, ranking in report.metric_family_rankings.items():
            print(f"  {family} ({ranking.metric_count} metrics):")
            for entry_rank in ranking.entries:
                rank = "n/a" if entry_rank.rank is None else str(entry_rank.rank)
                print(
                    f"    {rank}. {entry_rank.label}: "
                    f"wins={entry_rank.win_count}, "
                    f"ties={entry_rank.tie_count}, "
                    f"losses={entry_rank.loss_count}, "
                    f"not_comparable={entry_rank.not_comparable_count}"
                )


def _format_comparison_materialization(side: ComparisonSide) -> str:
    parts = [
        f"metrics_origin={side.metrics_origin}",
        f"data_verified={_format_optional_bool(side.data_verified)}",
    ]
    if side.computed_at is not None:
        parts.append(f"computed_at={side.computed_at}")
    return ", ".join(parts)


def _comparison_not_comparable_reasons(
    comparison: ResultComparison,
) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for metric in comparison.metrics.values():
        if metric.winner == "not_comparable" and metric.not_comparable_reason:
            counts[metric.not_comparable_reason] = counts.get(metric.not_comparable_reason, 0) + 1
    for evidence in comparison.evidence_comparisons:
        if evidence.winner == "not_comparable" and evidence.not_comparable_reason:
            counts[evidence.not_comparable_reason] = (
                counts.get(evidence.not_comparable_reason, 0) + 1
            )
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return _format_bool(value)


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return str(value)


def _die(message: str) -> NoReturn:
    raise CalibrexError(message)


if __name__ == "__main__":
    raise SystemExit(main())
