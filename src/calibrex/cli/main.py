"""Calibrex command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast

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
from calibrex.core.camera_lidar_artifacts import (
    bullseye_plot_json_schema,
    calibration_candidate_trace_json_schema,
    camera_lidar_benchmark_protocol_json_schema,
    camera_lidar_problem_json_schema,
    load_camera_lidar_problem,
)
from calibrex.core.camera_lidar_sota_audit import (
    camera_lidar_sota_audit_protocol_json_schema,
    camera_lidar_sota_audit_result_json_schema,
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
from calibrex.core.continuous_time_lidar_ablation import (
    continuous_time_lidar_ablation_json_schema,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    ContinuousTimeLidarPairArtifact,
    continuous_time_lidar_pair_json_schema,
)
from calibrex.core.dynamic_window import (
    DynamicWindowConsistencyThresholds,
    dynamic_window_consistency_json_schema,
)
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
from calibrex.core.livox_time_ablation import livox_time_ablation_json_schema
from calibrex.core.online_timeline import online_timeline_json_schema
from calibrex.core.probabilistic_correspondence import (
    probabilistic_correspondence_json_schema,
    probabilistic_pnp_result_json_schema,
    probabilistic_refinement_result_json_schema,
)
from calibrex.core.provenance import sha256_path
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
from calibrex.data.kitti_benchmark import (
    build_kitti_raw_0005_benchmark_input,
    kitti_benchmark_input_json_schema,
)
from calibrex.data.kitti_camera_lidar_problem import (
    build_kitti_raw_camera_lidar_problem,
)
from calibrex.data.manifest import manifest_json_schema
from calibrex.data.public_datasets import load_public_dataset_catalog
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
    run_borer_six_dof_benchmark,
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
from calibrex.evaluation.degeneracy import degeneracy_from_inspection
from calibrex.evaluation.dynamic_window import evaluate_dynamic_window_consistency
from calibrex.evaluation.evidence_summary import evidence_cases_from_result
from calibrex.evaluation.kitti_falsification_benchmark import (
    kitti_falsification_json_schema,
    run_kitti_falsification_benchmark,
)
from calibrex.evaluation.lidar import lidar_metrics_from_inspection
from calibrex.evaluation.metrics import evaluate_quality
from calibrex.evaluation.motion import motion_metrics_from_inspection
from calibrex.evaluation.pandey_recovery_benchmark import (
    load_kitti_i2i_benchmark_data,
    run_kitti_i2i_recovery_benchmark,
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
from calibrex.export.autoware import export_autoware_transforms, export_autoware_yaml
from calibrex.export.ros_tf import export_ros_tf_transforms, export_ros_tf_yaml
from calibrex.graph.problem import build_problem
from calibrex.importers.kalibr import import_kalibr_camchain
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration
from calibrex.pipelines.online import (
    OnlineCalibrationRunOptions,
    evaluate_rosbag2_capture_readiness,
    evaluate_rosbag2_continuous_time_lidar_pair,
    evaluate_rosbag2_trajectory_window_drift,
    run_online_calibration,
)
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
            "continuous-time-lidar-pair",
            "continuous-time-lidar-ablation",
            "solid-state-cross-dataset-benchmark-config",
            "solid-state-cross-dataset-benchmark",
            "solid-state-failure-analysis",
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
            "evidence-bundle",
            "evidence-bundle-verification",
            "online-timeline",
            "solid-state-context",
            "livox-time-ablation",
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
    init.add_argument("profile", choices=["camera-lidar-imu", "autonomous-driving-rig"])
    init.add_argument("--output", type=Path, required=True)
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
    capture_readiness.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
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
        default=Path("examples/public_datasets/kitti_lidar_camera_evidence/config.yaml"),
        help="base KITTI camera-LiDAR evidence config",
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
    build_kitti360_problem.add_argument("--dataset-id")
    build_kitti360_problem.add_argument("--problem-id")
    build_kitti360_problem.add_argument(
        "--rotation-bound-deg", type=float, default=20.0
    )
    build_kitti360_problem.add_argument(
        "--translation-bound-m",
        type=float,
        default=0.0,
    )
    build_kitti360_problem.add_argument("--json", action="store_true")
    build_kitti360_problem.set_defaults(
        func=_cmd_camera_lidar_build_kitti360_problem
    )
    build_a2d2_problem = camera_lidar_subcommands.add_parser(
        "build-a2d2-problem",
        help="build the pre-registered A2D2 cross-family D2D smoke problem",
    )
    build_a2d2_problem.add_argument("data_directory", type=Path)
    build_a2d2_problem.add_argument("depth_provider", type=Path)
    build_a2d2_problem.add_argument(
        "--lidar-output-directory", type=Path, required=True
    )
    build_a2d2_problem.add_argument(
        "--lidar-manifest-output", type=Path, required=True
    )
    build_a2d2_problem.add_argument("--output", type=Path, required=True)
    build_a2d2_problem.add_argument("--problem-id")
    build_a2d2_problem.add_argument(
        "--rotation-bound-deg", type=float, default=20.0
    )
    build_a2d2_problem.add_argument("--json", action="store_true")
    build_a2d2_problem.set_defaults(
        func=_cmd_camera_lidar_build_a2d2_problem
    )
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
    freeze_six_dof.add_argument(
        "--perturbation-count", type=_positive_int, default=200
    )
    freeze_six_dof.add_argument("--rotation-deg", type=float, default=0.5)
    freeze_six_dof.add_argument("--translation-m", type=float, default=0.5)
    freeze_six_dof.add_argument("--histogram-bins", type=_positive_int, default=32)
    freeze_six_dof.add_argument(
        "--min-visible-points", type=_positive_int, default=64
    )
    freeze_six_dof.add_argument("--rotation-bound-deg", type=float, default=2.0)
    freeze_six_dof.add_argument("--translation-bound-m", type=float, default=1.0)
    freeze_six_dof.add_argument(
        "--initial-rotation-step-deg", type=float, default=0.25
    )
    freeze_six_dof.add_argument(
        "--initial-translation-step-m", type=float, default=0.10
    )
    freeze_six_dof.add_argument(
        "--minimum-rotation-step-deg", type=float, default=0.01
    )
    freeze_six_dof.add_argument(
        "--minimum-translation-step-m", type=float, default=0.005
    )
    freeze_six_dof.add_argument(
        "--max-evaluations", type=_positive_int, default=800
    )
    freeze_six_dof.add_argument("--json", action="store_true")
    freeze_six_dof.set_defaults(
        func=_cmd_camera_lidar_freeze_six_dof_protocol
    )
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
    benchmark_six_dof.add_argument(
        "--definition-output", type=Path, required=True
    )
    benchmark_six_dof.add_argument("--output", type=Path, required=True)
    benchmark_six_dof.add_argument(
        "--bootstrap-samples", type=_positive_int, default=2000
    )
    benchmark_six_dof.add_argument("--workers", type=_positive_int, default=1)
    benchmark_six_dof.add_argument("--resume", action="store_true")
    benchmark_six_dof.add_argument("--required-hit-rate", type=float)
    benchmark_six_dof.add_argument("--json", action="store_true")
    benchmark_six_dof.set_defaults(
        func=_cmd_camera_lidar_benchmark_six_dof
    )
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
    probabilistic_pnp.add_argument(
        "--minimum-confidence", type=float, default=0.25
    )
    probabilistic_pnp.add_argument(
        "--minimum-correspondences", type=_positive_int, default=6
    )
    probabilistic_pnp.add_argument(
        "--ransac-reprojection-threshold-px", type=float, default=4.0
    )
    probabilistic_pnp.add_argument(
        "--ransac-confidence", type=float, default=0.999
    )
    probabilistic_pnp.add_argument(
        "--ransac-iterations", type=_positive_int, default=1000
    )
    probabilistic_pnp.add_argument(
        "--mahalanobis-inlier-threshold", type=float, default=3.0
    )
    probabilistic_pnp.add_argument("--random-seed", type=int, default=0)
    probabilistic_pnp.add_argument("--json", action="store_true")
    probabilistic_pnp.set_defaults(
        func=_cmd_camera_lidar_refine_probabilistic_pnp
    )
    probabilistic_multiframe = camera_lidar_subcommands.add_parser(
        "refine-probabilistic-multiframe",
        help="refine one shared D2D pose from all probabilistic frames",
    )
    probabilistic_multiframe.add_argument(
        "correspondence_artifact", type=Path
    )
    probabilistic_multiframe.add_argument("initial_problem", type=Path)
    probabilistic_multiframe.add_argument(
        "--initial-trace",
        type=Path,
        help="schema-valid D2D candidate trace whose output pose initializes refinement",
    )
    probabilistic_multiframe.add_argument("--output", type=Path, required=True)
    probabilistic_multiframe.add_argument("--result-id")
    probabilistic_multiframe.add_argument(
        "--minimum-confidence", type=float, default=0.25
    )
    probabilistic_multiframe.add_argument(
        "--holdout-ratio", type=float, default=0.25
    )
    probabilistic_multiframe.add_argument("--split-seed", type=int, default=0)
    probabilistic_multiframe.add_argument(
        "--minimum-train-correspondences", type=_positive_int, default=24
    )
    probabilistic_multiframe.add_argument(
        "--minimum-holdout-correspondences", type=_positive_int, default=8
    )
    probabilistic_multiframe.add_argument(
        "--max-evaluations", type=_positive_int, default=400
    )
    probabilistic_multiframe.add_argument(
        "--without-covariance", action="store_true"
    )
    probabilistic_multiframe.add_argument(
        "--without-outlier-probability", action="store_true"
    )
    probabilistic_multiframe.add_argument(
        "--without-reliability", action="store_true"
    )
    probabilistic_multiframe.add_argument("--json", action="store_true")
    probabilistic_multiframe.set_defaults(
        func=_cmd_camera_lidar_refine_probabilistic_multiframe
    )
    probabilistic_ablation = camera_lidar_subcommands.add_parser(
        "benchmark-probabilistic-ablation",
        help="run paired covariance/outlier/reliability ablations",
    )
    probabilistic_ablation.add_argument(
        "correspondence_artifact", type=Path
    )
    probabilistic_ablation.add_argument("initial_problem", type=Path)
    probabilistic_ablation.add_argument(
        "--initial-trace",
        type=Path,
        help="use one digest-pinned D2D output for every paired ablation",
    )
    probabilistic_ablation.add_argument(
        "--result-dir", type=Path, required=True
    )
    probabilistic_ablation.add_argument(
        "--definition-output", type=Path, required=True
    )
    probabilistic_ablation.add_argument("--output", type=Path, required=True)
    probabilistic_ablation.add_argument(
        "--split-seeds",
        default="0,1,2,3,4",
        help="comma-separated deterministic frame-split seeds",
    )
    probabilistic_ablation.add_argument(
        "--bootstrap-samples", type=_positive_int, default=2000
    )
    probabilistic_ablation.add_argument("--json", action="store_true")
    probabilistic_ablation.set_defaults(
        func=_cmd_camera_lidar_benchmark_probabilistic_ablation
    )
    continuous_time = camera_lidar_subcommands.add_parser(
        "refine-continuous-time",
        help="jointly refine extrinsic and clock offset from a frozen problem",
    )
    continuous_time.add_argument("problem", type=Path)
    continuous_time.add_argument("--output", type=Path, required=True)
    continuous_time.add_argument("--result-id")
    continuous_time.add_argument("--json", action="store_true")
    continuous_time.set_defaults(
        func=_cmd_camera_lidar_refine_continuous_time
    )
    attach_continuous_trajectory = camera_lidar_subcommands.add_parser(
        "attach-continuous-trajectory",
        help="attach a recorded T_world_body trajectory to a frozen problem",
    )
    attach_continuous_trajectory.add_argument("problem", type=Path)
    attach_continuous_trajectory.add_argument("trajectory", type=Path)
    attach_continuous_trajectory.add_argument(
        "--output", type=Path, required=True
    )
    attach_continuous_trajectory.add_argument("--problem-id")
    attach_continuous_trajectory.add_argument("--json", action="store_true")
    attach_continuous_trajectory.set_defaults(
        func=_cmd_camera_lidar_attach_continuous_trajectory
    )
    continuous_time_ablation = camera_lidar_subcommands.add_parser(
        "benchmark-continuous-time-ablation",
        help="run paired clock/per-point-time/covariance ablations",
    )
    continuous_time_ablation.add_argument("problem", type=Path)
    continuous_time_ablation.add_argument(
        "--result-dir", type=Path, required=True
    )
    continuous_time_ablation.add_argument(
        "--definition-output", type=Path, required=True
    )
    continuous_time_ablation.add_argument(
        "--output", type=Path, required=True
    )
    continuous_time_ablation.add_argument(
        "--split-seeds", default="0,1,2,3,4"
    )
    continuous_time_ablation.add_argument(
        "--bootstrap-samples", type=_positive_int, default=2000
    )
    continuous_time_ablation.add_argument("--json", action="store_true")
    continuous_time_ablation.set_defaults(
        func=_cmd_camera_lidar_benchmark_continuous_time_ablation
    )
    sota_audit = camera_lidar_subcommands.add_parser(
        "audit-sota",
        help="evaluate a digest-frozen Camera-LiDAR SOTA claim protocol",
    )
    sota_audit.add_argument("protocol", type=Path)
    sota_audit.add_argument("--output", type=Path, required=True)
    sota_audit.add_argument("--audit-id")
    sota_audit.add_argument("--json", action="store_true")
    sota_audit.set_defaults(func=_cmd_camera_lidar_audit_sota)

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
    export.set_defaults(func=_cmd_export)

    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
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
        print(f"Python {artifact.environment.python}")
        for name, dependency in artifact.environment.dependencies.items():
            state = "ok" if dependency.available else "missing"
            optional = " (optional)" if dependency.optional else ""
            print(f"{name}: {state}{optional}")
        if artifact.dataset is not None:
            print(f"Dataset: {artifact.dataset.dataset_type} ({artifact.status.upper()})")
            print(f"  path: {artifact.dataset.path}")
            if artifact.quality is not None:
                print(f"  degeneracy: {artifact.quality.degeneracy.grade.upper()}")
                for recommendation in artifact.quality.recommendations:
                    print(f"  recommendation: {recommendation}")
            for workflow in artifact.workflows:
                print(f"  workflow: {workflow.workflow_id} ({workflow.status})")
    return 1 if artifact.status == "fail" else 0


def _doctor_command(args: argparse.Namespace) -> list[str]:
    command = ["calibrex", "doctor"]
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
        "continuous-time-lidar-pair": continuous_time_lidar_pair_json_schema,
        "continuous-time-lidar-ablation": continuous_time_lidar_ablation_json_schema,
        "solid-state-cross-dataset-benchmark-config": (
            solid_state_cross_dataset_benchmark_config_json_schema
        ),
        "solid-state-cross-dataset-benchmark": solid_state_cross_dataset_benchmark_json_schema,
        "solid-state-failure-analysis": solid_state_failure_analysis_json_schema,
        "assessment": assessment_json_schema,
        "benchmark": benchmark_json_schema,
        "benchmark-definition": benchmark_definition_json_schema,
        "policy": policy_json_schema,
        "protocol": protocol_json_schema,
        "transforms": transform_artifact_json_schema,
        "dataset-manifest": manifest_json_schema,
        "doctor": doctor_json_schema,
        "calibration-ci": calibration_ci_json_schema,
        "external-run": external_run_json_schema,
        "kitti-falsification": kitti_falsification_json_schema,
        "kitti-benchmark-input": kitti_benchmark_input_json_schema,
        "depth-provider": depth_provider_json_schema,
        "continuous-time-camera-lidar-problem": (
            continuous_time_camera_lidar_problem_json_schema
        ),
        "continuous-time-camera-lidar-result": (
            continuous_time_camera_lidar_result_json_schema
        ),
        "probabilistic-correspondence": probabilistic_correspondence_json_schema,
        "probabilistic-pnp-result": probabilistic_pnp_result_json_schema,
        "probabilistic-refinement-result": (
            probabilistic_refinement_result_json_schema
        ),
        "camera-lidar-problem": camera_lidar_problem_json_schema,
        "camera-lidar-sota-audit-protocol": (
            camera_lidar_sota_audit_protocol_json_schema
        ),
        "camera-lidar-sota-audit-result": (
            camera_lidar_sota_audit_result_json_schema
        ),
        "camera-lidar-benchmark-protocol": (
            camera_lidar_benchmark_protocol_json_schema
        ),
        "calibration-candidate-trace": calibration_candidate_trace_json_schema,
        "bullseye-plot": bullseye_plot_json_schema,
        "evidence-bundle": evidence_bundle_json_schema,
        "evidence-bundle-verification": evidence_bundle_verification_json_schema,
        "online-timeline": online_timeline_json_schema,
        "solid-state-context": solid_state_context_json_schema,
        "livox-time-ablation": livox_time_ablation_json_schema,
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
    return f"{kind.replace('-', '_')}.schema.json"


def _cmd_validate(args: argparse.Namespace) -> int:
    report = validate_file(args.path, cast(ValidationKind, args.kind))
    payload = report.model_dump(mode="json")
    _emit(payload, args.json)
    return 0


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
        use_point_time_offsets=(
            None if args.deskew == "config" else args.deskew == "on"
        ),
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
        print(
            f"trajectory window drift: {artifact.grade} "
            f"({artifact.interpretation})"
        )
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
    artifact = evaluate_rosbag2_continuous_time_lidar_pair(args.config)
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
    dataset_path = (
        Path(args.dataset_path)
        if args.dataset_path is not None
        else Path(str(dataset_section["path"]))
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
    command = (
        f"calibrex camera-lidar freeze-rotation-protocol {args.problem} "
        f"--output {args.output}"
    )
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
            command=tuple(command.split()),
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
    command = (
        f"calibrex camera-lidar freeze-six-dof-protocol {args.problem} "
        f"--output {args.output} --rotation-deg {args.rotation_deg} "
        f"--translation-m {args.translation_m}"
    )
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
            command=tuple(command.split()),
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


def _cmd_camera_lidar_build_kitti_problem(args: argparse.Namespace) -> int:
    command = (
        "calibrex camera-lidar build-kitti-problem "
        f"{args.sequence_path} {args.depth_provider} --output {args.output} "
        f"--camera-stream {args.camera_stream}"
    )
    try:
        problem = build_kitti_raw_camera_lidar_problem(
            args.sequence_path,
            args.depth_provider,
            camera_stream=args.camera_stream,
            dataset_id=args.dataset_id,
            problem_id=args.problem_id,
            rotation_bound_deg=args.rotation_bound_deg,
            translation_bound_m=args.translation_bound_m,
            command=tuple(command.split()),
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
    ]
    if args.calibration_root is not None:
        command.extend(["--calibration-root", str(args.calibration_root)])
    if args.lidar_directory is not None:
        command.extend(["--lidar-directory", str(args.lidar_directory)])
    if args.lidar_manifest is not None:
        command.extend(["--lidar-manifest", str(args.lidar_manifest)])
    try:
        problem = build_kitti360_camera_lidar_problem(
            args.sequence_path,
            args.depth_provider,
            calibration_root=args.calibration_root,
            lidar_directory=args.lidar_directory,
            lidar_manifest_path=args.lidar_manifest,
            camera_stream=args.camera_stream,
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
            problem_id=(
                args.problem_id or "a2d2-frontleft-preregistered-d2d"
            ),
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
            "accuracy_limitation": (
                "A2D2 source points are pre-registered into the camera view"
            ),
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
    bullseye_artifact_output = (
        args.bullseye_artifact_output
        or args.output.with_name(f"{args.output.stem}_bullseye.json")
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
        f"{args.definition_output} --output {args.output} --workers {args.workers}"
    )
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
        ransac_reprojection_threshold_px=(
            args.ransac_reprojection_threshold_px
        ),
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
            raise CalibrexError(
                f"initialization problem is not readable: {args.initial_problem}"
            )
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
            result_id=(
                args.result_id
                or f"{result.artifact_id}-{args.frame_id}-opencv-pnp"
            ),
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
            "selected_correspondence_count": (
                result.selected_correspondence_count
            ),
            "ransac_inlier_count": result.ransac_inlier_count,
            "probabilistic_inlier_count": (
                result.probabilistic_inlier_count
            ),
            "weighted_reprojection_rmse_px": (
                result.weighted_reprojection_rmse_px
            ),
            "mean_mahalanobis_error": result.mean_mahalanobis_error,
        },
        args.json,
    )
    return 0 if result.status == "converged" else 2


def _cmd_camera_lidar_refine_probabilistic_multiframe(
    args: argparse.Namespace,
) -> int:
    try:
        options = ProbabilisticCameraLidarRefinementOptions(
            holdout_ratio=args.holdout_ratio,
            split_seed=args.split_seed,
            minimum_confidence=args.minimum_confidence,
            minimum_train_correspondences=(
                args.minimum_train_correspondences
            ),
            minimum_holdout_correspondences=(
                args.minimum_holdout_correspondences
            ),
            max_evaluations=args.max_evaluations,
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
        ]
        if args.initial_trace is not None:
            command.extend(["--initial-trace", str(args.initial_trace)])
        result = run_probabilistic_camera_lidar_refinement(
            args.correspondence_artifact,
            args.initial_problem,
            initialization_trace_path=args.initial_trace,
            result_id=args.result_id,
            options=options,
            command=command,
        )
        result.save(args.output)
    except (OSError, ValueError) as exc:
        raise CalibrexError(str(exc)) from exc
    _emit(
        {
            "status": result.status,
            "result": str(args.output),
            "initialization_source": result.initialization_source,
            "initialization_trace_id": result.initialization_trace_id,
            "initialization_trace_status": result.initialization_trace_status,
            "initialization_trace_hit": result.initialization_trace_hit,
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
            "use_outlier_probability": (
                result.options["use_outlier_probability"]
            ),
            "use_reliability": result.options["use_reliability"],
        },
        args.json,
    )
    return 0 if result.status == "converged" else 2


def _cmd_camera_lidar_benchmark_probabilistic_ablation(
    args: argparse.Namespace,
) -> int:
    try:
        seeds = tuple(
            int(value.strip())
            for value in args.split_seeds.split(",")
            if value.strip()
        )
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
            "final_time_offset_error_sec": (
                result.final_time_offset_error_sec
            ),
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
                len(problem.body_trajectory.poses)
                if problem.body_trajectory is not None
                else 0
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
        seeds = tuple(
            int(value.strip())
            for value in args.split_seeds.split(",")
            if value.strip()
        )
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
            "achieved_dataset_families": (
                audit.achieved_dataset_families
            ),
            "achieved_independent_rig_count": (
                audit.achieved_independent_rig_count
            ),
            "requirement_status": {
                item.requirement_id: item.status
                for item in audit.requirements
            },
        },
        args.json,
    )
    return 0 if audit.verdict == "supported" else 2


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


def _cmd_export(args: argparse.Namespace) -> int:
    if args.kind == "continuous-time-lidar-pair":
        artifact = ContinuousTimeLidarPairArtifact.model_validate(
            read_mapping(args.result)
        )
        transforms = {artifact.variable: artifact.refined_transform}
        if args.format == "ros-tf":
            export_ros_tf_transforms(transforms, args.output)
        elif args.format == "autoware":
            export_autoware_transforms(
                transforms,
                args.output,
                source_run=artifact.provenance.tool_name,
                quality_grade=artifact.refined_transform.quality.grade,
            )
        else:
            _die(f"unsupported export format: {args.format}")
    else:
        result = load_result(args.result)
        if args.format == "ros-tf":
            export_ros_tf_yaml(result, args.output)
        elif args.format == "autoware":
            export_autoware_yaml(result, args.output)
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
