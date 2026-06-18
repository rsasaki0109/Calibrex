"""Calibrex command-line entry point."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast

from calibrex import __version__
from calibrex.core.config import (
    DatasetConfig,
    DatasetType,
    config_json_schema,
    load_config,
)
from calibrex.core.exceptions import CalibrexError
from calibrex.core.frames import FrameGraph
from calibrex.core.io import write_mapping
from calibrex.core.report_artifacts import (
    is_report_artifact_schema_kind,
    report_artifact_json_schema,
    report_artifact_schema_kinds,
)
from calibrex.core.result import CalibrationResult, load_result, result_json_schema
from calibrex.core.validation import (
    ValidationKind,
    validate_file,
    validation_kind_choices,
)
from calibrex.data.inspect import DatasetInspection, inspect_dataset
from calibrex.data.kitti import read_kitti_initial_transforms
from calibrex.data.manifest import manifest_json_schema
from calibrex.data.public_datasets import load_public_dataset_catalog
from calibrex.evaluation.compare import ResultComparison, compare_results
from calibrex.evaluation.degeneracy import degeneracy_from_inspection
from calibrex.evaluation.lidar import lidar_metrics_from_inspection
from calibrex.evaluation.metrics import evaluate_quality
from calibrex.evaluation.motion import motion_metrics_from_inspection
from calibrex.evaluation.recommendations import build_inspection_recommendations
from calibrex.evaluation.registry import list_metric_definitions
from calibrex.evaluation.thresholds import ThresholdProfile, apply_metric_thresholds
from calibrex.evaluation.timing import timing_metrics_from_inspection
from calibrex.export.autoware import export_autoware_yaml
from calibrex.export.ros_tf import export_ros_tf_yaml
from calibrex.graph.problem import build_problem
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration
from calibrex.visualization.overlays import write_camera_lidar_overlay_artifact
from calibrex.visualization.report import report_artifact_paths, write_report_artifacts
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibrex")
    parser.add_argument("--version", action="version", version=f"calibrex {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="check local Calibrex environment")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    doctor.set_defaults(func=_cmd_doctor)

    schema = subcommands.add_parser("schema", help="print JSON schema")
    schema.add_argument(
        "kind",
        choices=[
            "config",
            "result",
            "dataset-manifest",
            *report_artifact_schema_kinds(),
        ],
    )
    schema.add_argument("--output", type=Path, help="write schema to a file")
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

    compare = subcommands.add_parser("compare", help="compare two result files")
    compare.add_argument("left_result", type=Path)
    compare.add_argument("right_result", type=Path)
    compare.add_argument("--output", type=Path, help="write comparison as YAML/JSON")
    compare.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    compare.set_defaults(func=_cmd_compare)

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
            "filesystem",
            "kitti-raw",
            "kitti_raw",
            "mcap",
            "nuscenes",
            "rosbag1",
            "rosbag2",
            "tum-rgbd",
            "tum_rgbd",
        ],
        default="filesystem",
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
        "calibrex_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
        "core_ros_independent": True,
    }
    if args.json:
        print(json.dumps(checks, indent=2, sort_keys=True))
    else:
        print(f"Calibrex {__version__}")
        print(f"Python {checks['python']}")
        for name, available in dependencies.items():
            state = "ok" if available else "missing"
            optional = " (optional)" if name.endswith("_optional") else ""
            print(f"{name}: {state}{optional}")
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    if args.kind == "config":
        schema = config_json_schema()
    elif args.kind == "result":
        schema = result_json_schema()
    elif args.kind == "dataset-manifest":
        schema = manifest_json_schema()
    elif is_report_artifact_schema_kind(args.kind):
        schema = report_artifact_json_schema(args.kind)
    else:
        _die(f"unsupported schema kind: {args.kind}")
    if args.output:
        write_mapping(args.output, schema)
    else:
        print(json.dumps(schema, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    report = validate_file(args.path, cast(ValidationKind, args.kind))
    payload = report.model_dump(mode="json")
    _emit(payload, args.json)
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    config = _starter_config(args.profile)
    write_mapping(args.output, config)
    print(f"wrote {args.output}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
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
    }
    _emit(payload, args.json)
    return 1 if result.quality.grade == "fail" else 0


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
        _emit_comparison(comparison)
    return 0


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
                "calibrex_config": entry.calibrex_config,
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
    inspection = inspect_dataset(DatasetConfig(type=dataset_type, path=str(args.path)))
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
        "schema_version": "calibrex.config/v0.1",
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

    if "velodyne_points" in inspection.diagnostics:
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


def _emit_comparison(comparison: ResultComparison) -> None:
    summary = comparison.summary
    print(f"left: {comparison.left.run_id} ({comparison.left.grade})")
    print(f"right: {comparison.right.run_id} ({comparison.right.grade})")
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
    raise CalibrexError(message)


if __name__ == "__main__":
    raise SystemExit(main())
