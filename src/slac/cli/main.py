"""slac command-line entry point."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast

from slac import __version__
from slac.core.assessment import (
    AssessmentArtifact,
    assess_evidence_file,
    assessment_json_schema,
)
from slac.core.config import (
    CalibrationConfig,
    DatasetConfig,
    DatasetType,
    config_json_schema,
    load_config,
)
from slac.core.evidence_bundle import (
    EvidenceBundleVerification,
    evidence_bundle_json_schema,
    evidence_bundle_verification_json_schema,
    verify_evidence_bundle,
    verify_evidence_bundle_verification,
    write_evidence_bundle_verification,
)
from slac.core.evidence_contract import (
    PolicyArtifact,
    policy_json_schema,
    protocol_json_schema,
)
from slac.core.exceptions import SlacError
from slac.core.frames import FrameGraph
from slac.core.io import read_mapping, write_mapping
from slac.core.online_timeline import online_timeline_json_schema
from slac.core.report_artifacts import (
    report_artifact_json_schema,
    report_artifact_schema_kinds,
)
from slac.core.result import CalibrationResult, load_result, result_json_schema
from slac.core.trajectory import trajectory_json_schema
from slac.core.transform_artifacts import transform_artifact_json_schema
from slac.core.validation import (
    ValidationKind,
    validate_file,
    validation_kind_choices,
)
from slac.data.downloads import (
    LIVOX_BASE_PCD_NAME,
    LIVOX_TARGET_PCD_NAME,
    download_livox_horizon_horizon_pcd_sample,
    livox_horizon_horizon_pcd_sample_path,
)
from slac.data.inspect import DatasetInspection, inspect_dataset
from slac.data.kitti import read_kitti_initial_transforms
from slac.data.manifest import manifest_json_schema
from slac.data.public_datasets import load_public_dataset_catalog
from slac.evaluation.compare import (
    ComparisonSide,
    ResultComparison,
    compare_results,
    comparison_json_schema,
)
from slac.evaluation.degeneracy import degeneracy_from_inspection
from slac.evaluation.evidence_summary import evidence_cases_from_result
from slac.evaluation.lidar import lidar_metrics_from_inspection
from slac.evaluation.metrics import evaluate_quality
from slac.evaluation.motion import motion_metrics_from_inspection
from slac.evaluation.recommendations import build_inspection_recommendations
from slac.evaluation.registry import list_metric_definitions
from slac.evaluation.report_compare import (
    ReportComparison,
    compare_reports,
    report_comparison_json_schema,
)
from slac.evaluation.thresholds import ThresholdProfile, apply_metric_thresholds
from slac.evaluation.timing import timing_metrics_from_inspection
from slac.export.autoware import export_autoware_yaml
from slac.export.ros_tf import export_ros_tf_yaml
from slac.graph.problem import build_problem
from slac.pipelines.calibrate import CalibrationRunOptions, run_calibration
from slac.pipelines.online import OnlineCalibrationRunOptions, run_online_calibration
from slac.visualization.overlays import write_camera_lidar_overlay_artifact
from slac.visualization.report import (
    evidence_artifact_from_result,
    report_artifact_paths,
    write_evidence_artifact,
    write_evidence_contract_artifacts,
    write_report_artifacts,
)
from slac.visualization.rig3d import write_rig_3d_artifact


def main(argv: list[str] | None = None) -> int:
    """Run the slac CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    command = cast(Callable[[argparse.Namespace], int], args.func)
    try:
        return command(args)
    except SlacError as exc:
        print(f"slac: error: {exc}", file=sys.stderr)
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

    The upper bound matches `slac.evaluation.holdout.split_indices`;
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
    parser = argparse.ArgumentParser(prog="slac")
    parser.add_argument("--version", action="version", version=f"slac {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="check local slac environment")
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
            "assessment",
            "policy",
            "protocol",
            "transforms",
            "dataset-manifest",
            "evidence-bundle",
            "evidence-bundle-verification",
            "online-timeline",
            *report_artifact_schema_kinds(),
            "all",
        ],
    )
    schema.add_argument("--output", type=Path, help="write schema to a file")
    schema.add_argument("--output-dir", type=Path, help="write all schemas to a directory")
    schema.set_defaults(func=_cmd_schema)

    validate = subcommands.add_parser("validate", help="validate a slac artifact")
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--kind",
        choices=validation_kind_choices(),
        default="auto",
        help="artifact kind; defaults to schema_version auto-detection",
    )
    validate.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    validate.set_defaults(func=_cmd_validate)

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
        choices=["html"],
        default="html",
        help="rendered artifact format; only html is supported in v0.1",
    )
    render.add_argument("--output-dir", type=Path)
    render.add_argument("--html", type=Path, help="HTML report path or filename")
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
    compare.add_argument("--output", type=Path, help="write comparison as YAML/JSON")
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

    kitti = subcommands.add_parser("kitti", help="KITTI raw utilities")
    kitti_subcommands = kitti.add_subparsers(dest="kitti_command", required=True)
    kitti_import = kitti_subcommands.add_parser(
        "import-calib",
        help="import KITTI calibration files as slac transforms",
    )
    kitti_import.add_argument("path", type=Path, help="directory containing calib_velo_to_cam.txt")
    kitti_import.add_argument("--output", type=Path)
    kitti_import.add_argument("--json", action="store_true")
    kitti_import.set_defaults(func=_cmd_kitti_import_calib)

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
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=_cmd_export)

    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    dependencies: dict[str, bool] = {
        "pydantic": _has_module("pydantic"),
        "yaml": _has_module("yaml"),
        "jsonschema": _has_module("jsonschema"),
        "mcap_optional": _has_module("mcap"),
        "open3d_optional": _has_module("open3d"),
    }
    checks: dict[str, Any] = {
        "slac_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
        "core_ros_independent": True,
    }
    if args.json:
        print(json.dumps(checks, indent=2, sort_keys=True))
    else:
        print(f"slac {__version__}")
        print(f"Python {checks['python']}")
        for name, available in dependencies.items():
            state = "ok" if available else "missing"
            optional = " (optional)" if name.endswith("_optional") else ""
            print(f"{name}: {state}{optional}")
    return 0


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
        "assessment": assessment_json_schema,
        "policy": policy_json_schema,
        "protocol": protocol_json_schema,
        "transforms": transform_artifact_json_schema,
        "dataset-manifest": manifest_json_schema,
        "evidence-bundle": evidence_bundle_json_schema,
        "evidence-bundle-verification": evidence_bundle_verification_json_schema,
        "online-timeline": online_timeline_json_schema,
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
    return f"{kind.replace('-', '_')}.schema.json"


def _cmd_validate(args: argparse.Namespace) -> int:
    report = validate_file(args.path, cast(ValidationKind, args.kind))
    payload = report.model_dump(mode="json")
    _emit(payload, args.json)
    return 0


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
            "run the offline `slac calibrate` to compare candidate extrinsics"
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
    if args.format != "html":
        _die(f"unsupported render format: {args.format}")
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


def _cmd_compare(args: argparse.Namespace) -> int:
    comparison = compare_results(
        load_result(args.left_result),
        load_result(args.right_result),
        left_path=args.left_result,
        right_path=args.right_result,
    )
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
    if (
        args.enforce_compatible
        and comparison.protocol_compatibility.status != "compatible"
    ):
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
    if (
        args.enforce_compatible
        and report.summary.protocol_compatibility_status != "compatible"
    ):
        return 1
    return 0


def _parse_labeled_results(entries: list[str]) -> list[tuple[str, Path]]:
    labeled: list[tuple[str, Path]] = []
    for entry in entries:
        label, separator, path = entry.partition("=")
        if not separator or not label or not path:
            _die(
                "report-compare entries must use LABEL=RESULT syntax, "
                f"got: {entry}"
            )
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
    # Unlike the Livox pair demo, the KITTI raw pipeline does not yet populate
    # per-frame raw input file hashes, so raw-recomputation cannot be gated on
    # here; verification still checks that bundle artifacts are internally
    # consistent and unmodified.
    verification = verify_evidence_bundle(bundle_path, require_raw_recomputed=False)
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
            "Raw-recomputation input hashing is not yet implemented for KITTI "
            "raw, so verification does not gate on it here."
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
            "description": "Starter slac calibration config",
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
    print(
        "input_files: "
        f"{verification.checked_input_file_count}/{verification.input_file_count}"
    )
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
            counts[metric.not_comparable_reason] = (
                counts.get(metric.not_comparable_reason, 0) + 1
            )
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


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _die(message: str) -> NoReturn:
    raise SlacError(message)


if __name__ == "__main__":
    raise SystemExit(main())
