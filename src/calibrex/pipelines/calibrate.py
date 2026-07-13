"""Minimal calibration pipeline compiler.

The alpha implementation intentionally separates the stable problem/result
surface from production-grade solvers. It validates inputs, compiles the frame
graph, emits provenance, and marks provisional solver output as WARN.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from calibrex import __version__
from calibrex.core.config import CalibrationConfig, load_config
from calibrex.core.exceptions import ConfigError
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3, normalize_quaternion_xyzw
from calibrex.core.io import read_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import (
    CalibrationResult,
    MetricResult,
    ObservabilityResult,
    RunInfo,
    TimeOffsetQuality,
    TimeOffsetResult,
    TransformEstimateProvenance,
    TransformQuality,
    TransformResult,
)
from calibrex.data.downloads import (
    LIVOX_BASE_PCD_NAME,
    LIVOX_BASE_PCD_URL,
    LIVOX_TARGET_PCD_NAME,
    LIVOX_TARGET_PCD_URL,
)
from calibrex.data.inspect import DatasetInspection, inspect_dataset
from calibrex.data.kitti import read_kitti_initial_transforms
from calibrex.data.livox import find_livox_pcd_files
from calibrex.data.nuscenes import read_nuscenes_reference_extrinsics
from calibrex.evaluation.degeneracy import degeneracy_from_inspection
from calibrex.evaluation.imu import imu_metrics_from_result
from calibrex.evaluation.lidar import (
    lidar_metrics_from_inspection,
    livox_pair_evidence_from_dataset,
)
from calibrex.evaluation.lidar_camera import lidar_camera_metrics_from_result
from calibrex.evaluation.metrics import evaluate_quality
from calibrex.evaluation.motion import motion_metrics_from_inspection
from calibrex.evaluation.radar import radar_velocity_metrics_from_result
from calibrex.evaluation.timing import timing_metrics_from_inspection
from calibrex.graph.problem import build_problem
from calibrex.solvers.base import SolverAdapterResult
from calibrex.solvers.koide_lidar_camera_solver import (
    ADAPTER_FACTOR_NAMES as KOIDE_LIDAR_CAMERA_FACTOR_NAMES,
)
from calibrex.solvers.koide_lidar_camera_solver import KoideLidarCameraSolver
from calibrex.solvers.native_hand_eye_comparison_solver import (
    NATIVE_HAND_EYE_COMPARISON_BACKEND,
    NativeHandEyeComparisonSolver,
)
from calibrex.solvers.native_lidar_point_to_plane_solver import (
    NATIVE_LIDAR_POINT_TO_PLANE_BACKEND,
    NativeLidarPointToPlaneSolver,
)
from calibrex.solvers.native_planar_board_solver import (
    NATIVE_PLANAR_BOARD_BACKEND,
    NativePlanarBoardSolver,
)
from calibrex.solvers.native_registration_comparison_solver import (
    NATIVE_REGISTRATION_COMPARISON_BACKEND,
    NativeRegistrationComparisonSolver,
)
from calibrex.solvers.open3d_slac_solver import Open3DSLACSolver
from calibrex.visualization.overlays import write_camera_lidar_overlay_artifact
from calibrex.visualization.report import write_report_artifacts
from calibrex.visualization.rig3d import write_rig_3d_artifact


@dataclass(frozen=True)
class CalibrationRunOptions:
    """Runtime options shared by CLI and Python API."""

    dry_run: bool = False
    output_dir: Path | None = None
    strict: bool = False
    seed: int | None = None
    candidate_extrinsics: tuple[Path, ...] = ()


def run_calibration(
    config_path: str | Path,
    options: CalibrationRunOptions,
) -> CalibrationResult | None:
    """Run a calibration config through the current alpha pipeline."""

    config_file = Path(config_path)
    config = load_config(config_file)
    frame_graph = FrameGraph.from_config(config)
    inspection = inspect_dataset(config.dataset)
    problem = build_problem(config, frame_graph, inspection)

    if options.dry_run:
        for candidate_path in options.candidate_extrinsics:
            _load_candidate_extrinsics(candidate_path)
        _raise_on_dry_run_failures(inspection)
        return None

    output_dir = options.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / config.outputs.result
    report_path = output_dir / config.outputs.report

    result = _build_provisional_result(config, config_file, frame_graph, inspection, options)
    result.run.provenance["problem_summary"] = problem.summary()
    result.run.provenance["problem_observability"] = (
        problem.observability.as_dict() if problem.observability else None
    )
    result.run.provenance["dataset_inspection"] = inspection.as_dict()
    _apply_dataset_initialization(config, result)
    _apply_external_candidate_extrinsics(options.candidate_extrinsics, result)
    _apply_livox_pair_candidate_evidence(config, result)
    _apply_extrinsic_reference_comparisons(result)
    _apply_pipeline_adapter(config, frame_graph, inspection, result)
    result.metrics.update(lidar_camera_metrics_from_result(config, result, inspection))
    result.metrics.update(imu_metrics_from_result(config, result, inspection))
    result.metrics.update(radar_velocity_metrics_from_result(config, result, inspection))
    _apply_world_map_dof_diagnostics(result)
    result.artifacts.html_report = str(report_path)
    evaluate_quality(result, strict=options.strict)
    result.run.status = cast(
        Literal["success", "warning", "failed", "dry_run"],
        {"pass": "success", "warn": "warning", "fail": "failed"}[
            result.quality.grade
        ],
    )
    write_camera_lidar_overlay_artifact(result, output_dir / config.outputs.artifacts_dir)
    write_rig_3d_artifact(result, output_dir / config.outputs.artifacts_dir)
    result.save(result_path)
    write_report_artifacts(result, output_dir, html_filename=config.outputs.report)
    return result


def _apply_pipeline_adapter(
    config: CalibrationConfig,
    frame_graph: FrameGraph,
    inspection: DatasetInspection,
    result: CalibrationResult,
) -> None:
    adapter_result: SolverAdapterResult | None = None
    if config.pipeline.type == "rgbd_open3d_slac" or config.solver.backend == "open3d_slac":
        adapter_result = Open3DSLACSolver().solve(config, frame_graph, inspection)
    elif config.solver.backend == NATIVE_LIDAR_POINT_TO_PLANE_BACKEND:
        adapter_result = NativeLidarPointToPlaneSolver().solve(config, frame_graph, inspection)
    elif config.solver.backend == NATIVE_PLANAR_BOARD_BACKEND:
        adapter_result = NativePlanarBoardSolver().solve(config, frame_graph, inspection)
    elif config.solver.backend == NATIVE_HAND_EYE_COMPARISON_BACKEND:
        adapter_result = NativeHandEyeComparisonSolver().solve(
            config, frame_graph, inspection
        )
    elif config.solver.backend == NATIVE_REGISTRATION_COMPARISON_BACKEND:
        adapter_result = NativeRegistrationComparisonSolver().solve(
            config, frame_graph, inspection
        )
    elif _uses_koide_lidar_camera_adapter(config):
        adapter_result = KoideLidarCameraSolver().solve(config, frame_graph, inspection)
    if adapter_result is None:
        return
    result.metrics.update(adapter_result.metrics)
    result.run.provenance.update(adapter_result.provenance)
    result.run.provenance["solver_adapter"] = adapter_result.backend
    result.run.provenance["solver_adapter_status"] = adapter_result.status
    _apply_adapter_transforms(result, adapter_result)
    if (
        adapter_result.backend
        in {
            NATIVE_LIDAR_POINT_TO_PLANE_BACKEND,
            NATIVE_PLANAR_BOARD_BACKEND,
            NATIVE_HAND_EYE_COMPARISON_BACKEND,
            NATIVE_REGISTRATION_COMPARISON_BACKEND,
        }
        and adapter_result.transforms
    ):
        result.metrics["prototype_solver"] = MetricResult(
            value=1.0,
            grade="pass",
            reason=f"native solver {adapter_result.backend} produced an output transform",
        )
    if adapter_result.observability is not None:
        result.observability = adapter_result.observability
    if adapter_result.backend == NATIVE_PLANAR_BOARD_BACKEND and adapter_result.status == "pass":
        result.degeneracy.grade = "pass"
        result.degeneracy.reason = None
    result.degeneracy.reason = _append_reason(result.degeneracy.reason, adapter_result.warnings)


def _append_reason(existing: str | None, warnings: list[str]) -> str | None:
    additions = [warning for warning in warnings if warning]
    if not additions:
        return existing
    if existing:
        return "; ".join([existing, *additions])
    return "; ".join(additions)


def _apply_world_map_dof_diagnostics(result: CalibrationResult) -> None:
    weak_metric = result.metrics.get("lidar_world_map_weak_dof_count")
    if weak_metric is None or weak_metric.value is None or weak_metric.value <= 0.0:
        return

    reason = weak_metric.reason or "weak LiDAR world-map DoF from perturbation sensitivity"
    result.degeneracy.reason = _append_reason(result.degeneracy.reason, [reason])
    if result.degeneracy.grade == "pass":
        result.degeneracy.grade = "warn"

    weak_directions = _world_map_weak_directions(result.metrics)
    result.observability.weak_directions = _dedupe(
        [*result.observability.weak_directions, *weak_directions]
    )
    if result.observability.grade == "pass" and weak_directions:
        result.observability.grade = "warn"


def _world_map_weak_directions(metrics: dict[str, MetricResult]) -> list[str]:
    mapping = {
        "lidar_world_map_sensitivity_roll_m": "roll_lidar0",
        "lidar_world_map_sensitivity_pitch_m": "pitch_lidar0",
        "lidar_world_map_sensitivity_yaw_m": "yaw_lidar0",
        "lidar_world_map_sensitivity_x_m": "x_lidar0",
        "lidar_world_map_sensitivity_y_m": "y_lidar0",
        "lidar_world_map_sensitivity_z_m": "z_lidar0",
    }
    return [
        direction
        for metric_name, direction in mapping.items()
        if (metric := metrics.get(metric_name)) is not None and metric.grade == "warn"
    ]


def _apply_livox_pair_candidate_evidence(
    config: CalibrationConfig,
    result: CalibrationResult,
) -> None:
    if config.dataset.type != "livox_pcd":
        return
    if len(result.candidate_extrinsics) != 1:
        return
    transform_name, candidate = next(iter(result.candidate_extrinsics.items()))
    evidence = livox_pair_evidence_from_dataset(
        dataset_path=config.dataset.path,
        target_transform=candidate.as_se3(),
        config=config,
    )
    raw_input_files = _livox_raw_input_file_manifest(Path(config.dataset.path))
    result.metrics.update(evidence.metrics)
    result.run.provenance["metrics_origin"] = "recomputed"
    result.run.provenance["data_verified"] = _raw_input_files_verified(raw_input_files)
    result.run.provenance["raw_input_files"] = raw_input_files
    result.run.provenance["livox_pair_evidence"] = {
        "candidate_transform": transform_name,
        "target_transform_applied": True,
        "transform_convention": "T_source_target maps target PCD points into source PCD frame",
        "holdout_geometry": evidence.point_to_plane.as_dict(),
        "known_bad_perturbation": "left-multiplied source-frame SE(3) controls",
        "known_bad_case_count": len(evidence.cases),
        "known_bad_challenge": evidence.known_bad_challenge,
    }
    existing_cases = result.run.provenance.get("evidence_cases")
    cases = existing_cases if isinstance(existing_cases, list) else []
    result.run.provenance["evidence_cases"] = [
        *cases,
        *(case.model_dump(mode="json") for case in evidence.cases),
    ]


def _livox_raw_input_file_manifest(dataset_path: Path) -> list[dict[str, object]]:
    files = find_livox_pcd_files(dataset_path)
    roles = ("source", "target")
    return [
        _livox_raw_input_file_payload(
            path=file_path,
            role=roles[index] if index < len(roles) else f"sample_{index}",
        )
        for index, file_path in enumerate(files)
    ]


def _livox_raw_input_file_payload(path: Path, *, role: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": str(path),
        "role": role,
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    source_url = _livox_public_sample_source_url(path.name)
    if source_url is not None:
        payload["source_url"] = source_url
    return payload


def _raw_input_files_verified(files: list[dict[str, object]]) -> bool:
    if len(files) < 2:
        return False
    return all(isinstance(item.get("sha256"), str) for item in files)


def _livox_public_sample_source_url(filename: str) -> str | None:
    if filename == LIVOX_BASE_PCD_NAME:
        return LIVOX_BASE_PCD_URL
    if filename == LIVOX_TARGET_PCD_NAME:
        return LIVOX_TARGET_PCD_URL
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def _uses_koide_lidar_camera_adapter(config: CalibrationConfig) -> bool:
    if config.solver.backend == "koide_lidar_camera":
        return True
    return any(
        factor.enabled
        for name, factor in config.pipeline.factors.items()
        if name in KOIDE_LIDAR_CAMERA_FACTOR_NAMES
    )


def _apply_adapter_transforms(
    result: CalibrationResult,
    adapter_result: SolverAdapterResult,
) -> None:
    applied: list[str] = []
    for name, transform in sorted(adapter_result.transforms.items()):
        if name in result.transforms:
            result.transforms[name].translation_m = list(transform.translation_m)
            result.transforms[name].rotation_quat_xyzw = list(transform.rotation_quat_xyzw)
            result.transforms[name].estimate_id = name
            result.transforms[name].provenance = _adapter_output_provenance(
                adapter_result.backend,
                note="adapter output applied to Calibrex output estimate",
            )
            if (
                adapter_result.backend
                in {
                    NATIVE_PLANAR_BOARD_BACKEND,
                    NATIVE_HAND_EYE_COMPARISON_BACKEND,
                    NATIVE_REGISTRATION_COMPARISON_BACKEND,
                }
                and adapter_result.status == "pass"
            ):
                result.transforms[name].quality = TransformQuality(grade="pass")
            applied.append(name)
        elif name == "T_camera0_lidar0":
            applied_name = _apply_relative_lidar_camera_transform(result, transform)
            if applied_name is not None:
                result.transforms[applied_name].estimate_id = applied_name
                result.transforms[applied_name].provenance = _adapter_output_provenance(
                    adapter_result.backend,
                    note="relative adapter output composed into rig-frame estimate",
                )
                if (
                    adapter_result.backend
                    in {
                        NATIVE_PLANAR_BOARD_BACKEND,
                        NATIVE_HAND_EYE_COMPARISON_BACKEND,
                        NATIVE_REGISTRATION_COMPARISON_BACKEND,
                    }
                    and adapter_result.status == "pass"
                ):
                    result.transforms[applied_name].quality = TransformQuality(grade="pass")
                applied.append(applied_name)
    if applied:
        result.run.provenance["solver_adapter_applied_transforms"] = applied


def _adapter_output_provenance(backend: str, *, note: str) -> TransformEstimateProvenance:
    if backend in {
        NATIVE_LIDAR_POINT_TO_PLANE_BACKEND,
        NATIVE_PLANAR_BOARD_BACKEND,
        NATIVE_HAND_EYE_COMPARISON_BACKEND,
        NATIVE_REGISTRATION_COMPARISON_BACKEND,
    }:
        return TransformEstimateProvenance(
            producer="slac_native",
            execution_mode="offline_batch",
            role_in_comparison="output",
            evidence_level="algorithmically_refined",
            tool_name=backend,
            source={
                NATIVE_LIDAR_POINT_TO_PLANE_BACKEND: "native_lidar_point_to_plane_solver",
                NATIVE_PLANAR_BOARD_BACKEND: "native_planar_board_solver",
                NATIVE_HAND_EYE_COMPARISON_BACKEND: (
                    "native_hand_eye_comparison_solver"
                ),
                NATIVE_REGISTRATION_COMPARISON_BACKEND: (
                    "native_registration_comparison_solver"
                ),
            }[backend],
            notes=[note],
        )
    return TransformEstimateProvenance(
        producer="external_tool",
        execution_mode="imported",
        role_in_comparison="output",
        evidence_level="algorithmically_refined",
        tool_name=backend,
        source="solver_adapter",
        notes=[note],
    )


def _apply_dataset_initialization(
    config: CalibrationConfig,
    result: CalibrationResult,
) -> None:
    if config.dataset.type == "nuscenes":
        _apply_nuscenes_reference_extrinsics(config, result)
        return
    if config.dataset.type != "kitti_raw":
        return

    kitti_transforms = read_kitti_initial_transforms(config.dataset.path)
    if not kitti_transforms:
        result.run.provenance["dataset_initialization"] = {
            "source": "kitti_raw",
            "status": "missing_calibration",
        }
        return

    result.run.provenance["dataset_initialization"] = {
        "source": "kitti_raw",
        "status": "loaded",
        "transforms": {
            name: {"convention": "T_parent_child", **transform.as_dict()}
            for name, transform in sorted(kitti_transforms.items())
        },
    }
    _apply_kitti_lidar_initial_transform(result, kitti_transforms)


def _apply_kitti_lidar_initial_transform(
    result: CalibrationResult,
    kitti_transforms: dict[str, SE3],
) -> None:
    t_camera_lidar = kitti_transforms.get("T_camera0_lidar0")
    if t_camera_lidar is None:
        return
    applied_name = _apply_relative_lidar_camera_transform(result, t_camera_lidar)
    if applied_name is not None:
        result.run.provenance["dataset_initialization"]["applied_to"] = applied_name
        result.transforms[applied_name].estimate_id = applied_name
        result.transforms[applied_name].provenance = TransformEstimateProvenance(
            producer="dataset_provider",
            execution_mode="dataset_reference",
            role_in_comparison="output",
            evidence_level="dataset_provided",
            source="kitti_raw.calib_cam_to_cam_and_calib_velo_to_cam",
            notes=["KITTI calibration composed into rig-frame output estimate"],
        )


def _apply_nuscenes_reference_extrinsics(
    config: CalibrationConfig,
    result: CalibrationResult,
) -> None:
    references = read_nuscenes_reference_extrinsics(config.dataset.path)
    if not references:
        result.run.provenance["dataset_initialization"] = {
            "source": "nuscenes",
            "status": "missing_calibrated_sensor",
        }
        return

    typed_references: dict[str, TransformResult] = {}
    for name, raw in sorted(references.items()):
        try:
            typed_references[name] = TransformResult.model_validate(
                {
                    "convention": raw.get("convention"),
                    "parent": raw.get("parent"),
                    "child": raw.get("child"),
                    "translation_m": raw.get("translation_m"),
                    "rotation_quat_xyzw": raw.get("rotation_quat_xyzw"),
                    "quality": raw.get("quality"),
                    "estimate_id": name,
                    "provenance": {
                        "producer": "dataset_provider",
                        "execution_mode": "dataset_reference",
                        "role_in_comparison": "selected_reference",
                        "evidence_level": "dataset_provided",
                        "source": "nuscenes.calibrated_sensor",
                    },
                }
            )
        except ValueError:
            continue
    result.reference_extrinsics.update(typed_references)
    result.metrics["nuscenes_reference_extrinsic_count"] = MetricResult(
        value=float(len(typed_references)),
        unit="transforms",
        grade="pass" if typed_references else "warn",
        reason="nuScenes calibrated_sensor transforms imported as reference extrinsics",
    )
    result.run.provenance["dataset_initialization"] = {
        "source": "nuscenes",
        "status": "loaded" if typed_references else "invalid_calibrated_sensor",
        "reference_extrinsics": references,
    }


def _apply_extrinsic_reference_comparisons(result: CalibrationResult) -> None:
    comparisons: list[tuple[str, float, float]] = []
    references_by_edge = {
        (transform.parent, transform.child): (name, transform)
        for name, transform in sorted(result.reference_extrinsics.items())
    }
    for candidate_name, candidate in sorted(result.candidate_extrinsics.items()):
        reference_item = references_by_edge.get((candidate.parent, candidate.child))
        if reference_item is None:
            continue
        reference_name, reference = reference_item
        translation_delta_m = _translation_delta_m(candidate, reference)
        rotation_delta_deg = _rotation_delta_deg(candidate, reference)
        comparisons.append((candidate_name, translation_delta_m, rotation_delta_deg))
        metric_prefix = f"extrinsic_reference_delta_{candidate_name}"
        result.metrics[f"{metric_prefix}_translation_m"] = MetricResult(
            value=translation_delta_m,
            unit="m",
            grade="pass" if translation_delta_m <= 0.05 else "warn",
            reason=(
                f"{candidate_name} candidate translation delta against "
                f"{reference_name} reference"
            ),
        )
        result.metrics[f"{metric_prefix}_rotation_deg"] = MetricResult(
            value=rotation_delta_deg,
            unit="deg",
            grade="pass" if rotation_delta_deg <= 1.0 else "warn",
            reason=f"{candidate_name} candidate rotation delta against {reference_name} reference",
        )

    if not comparisons:
        if result.reference_extrinsics and result.candidate_extrinsics:
            result.metrics["extrinsic_reference_comparison_count"] = MetricResult(
                value=0.0,
                unit="pairs",
                grade="warn",
                reason="no candidate extrinsics matched reference extrinsic parent/child pairs",
            )
        return

    max_translation = max(item[1] for item in comparisons)
    max_rotation = max(item[2] for item in comparisons)
    result.metrics["extrinsic_reference_comparison_count"] = MetricResult(
        value=float(len(comparisons)),
        unit="pairs",
        grade="pass",
        reason="candidate extrinsics matched reference extrinsic parent/child pairs",
    )
    result.metrics["extrinsic_reference_translation_delta_max_m"] = MetricResult(
        value=max_translation,
        unit="m",
        grade="pass" if max_translation <= 0.05 else "warn",
        reason="maximum candidate-reference translation delta",
    )
    result.metrics["extrinsic_reference_rotation_delta_max_deg"] = MetricResult(
        value=max_rotation,
        unit="deg",
        grade="pass" if max_rotation <= 1.0 else "warn",
        reason="maximum candidate-reference rotation delta",
    )


def _apply_external_candidate_extrinsics(
    paths: tuple[Path, ...],
    result: CalibrationResult,
) -> None:
    if not paths:
        return

    source_summaries: list[dict[str, object]] = []
    imported_count = 0
    overwritten: list[str] = []
    for path in paths:
        candidates = _load_candidate_extrinsics(path)
        for name in candidates:
            if name in result.candidate_extrinsics:
                overwritten.append(name)
        result.candidate_extrinsics.update(candidates)
        imported_count += len(candidates)
        source_summaries.append(
            {
                "path": str(path),
                "count": len(candidates),
                "names": sorted(candidates),
            }
        )

    result.run.provenance["external_candidate_extrinsics"] = {
        "sources": source_summaries,
        "imported_count": imported_count,
        "overwritten": sorted(set(overwritten)),
    }
    result.metrics["candidate_extrinsic_import_count"] = MetricResult(
        value=float(imported_count),
        unit="transforms",
        grade="pass" if imported_count > 0 else "warn",
        reason="external candidate extrinsics imported",
    )


def _load_candidate_extrinsics(path: Path) -> dict[str, TransformResult]:
    try:
        payload = read_mapping(path)
    except Exception as exc:
        raise ConfigError(f"invalid candidate extrinsics file {path}: {exc}") from exc

    raw_transforms = _candidate_transform_payload(payload)
    candidates: dict[str, TransformResult] = {}
    for name, raw_transform in sorted(raw_transforms.items()):
        if not isinstance(raw_transform, dict):
            raise ConfigError(f"candidate extrinsic {name} in {path} must be a mapping")
        try:
            candidate = TransformResult.model_validate(raw_transform)
        except ValueError as exc:
            raise ConfigError(f"invalid candidate extrinsic {name} in {path}: {exc}") from exc
        _fill_imported_candidate_metadata(candidate, estimate_id=name, source_path=path)
        candidates[name] = candidate

    if not candidates:
        raise ConfigError(f"candidate extrinsics file {path} did not contain any transforms")
    return candidates


def _fill_imported_candidate_metadata(
    transform: TransformResult,
    *,
    estimate_id: str,
    source_path: Path,
) -> None:
    if transform.estimate_id is None:
        transform.estimate_id = estimate_id
    provenance = transform.provenance
    if provenance.producer == "unknown":
        provenance.producer = "unknown"
    if provenance.execution_mode == "unknown":
        provenance.execution_mode = "imported"
    if provenance.role_in_comparison is None:
        provenance.role_in_comparison = "candidate"
    if provenance.evidence_level == "unknown":
        provenance.evidence_level = "imported_without_documented_derivation"
    if provenance.source_path is None:
        provenance.source_path = str(source_path)
    if provenance.source is None:
        provenance.source = "candidate_extrinsics_file"


def _candidate_transform_payload(payload: dict[str, object]) -> dict[str, object]:
    if isinstance(payload.get("candidate_extrinsics"), dict):
        return cast(dict[str, object], payload["candidate_extrinsics"])
    if isinstance(payload.get("transforms"), dict):
        return cast(dict[str, object], payload["transforms"])
    transform_like: dict[str, object] = {
        name: value
        for name, value in payload.items()
        if isinstance(value, dict)
        and {"parent", "child", "translation_m", "rotation_quat_xyzw"}.issubset(value)
    }
    return transform_like


def _translation_delta_m(left: TransformResult, right: TransformResult) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.translation_m,
                right.translation_m,
                strict=True,
            )
        )
    )


def _rotation_delta_deg(left: TransformResult, right: TransformResult) -> float:
    left_quat = normalize_quaternion_xyzw(left.rotation_quat_xyzw)
    right_quat = normalize_quaternion_xyzw(right.rotation_quat_xyzw)
    dot = abs(
        sum(
            left_value * right_value
            for left_value, right_value in zip(
                left_quat,
                right_quat,
                strict=True,
            )
        )
    )
    clamped = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(clamped))


def _apply_relative_lidar_camera_transform(
    result: CalibrationResult,
    t_camera_lidar: SE3,
) -> str | None:
    for _camera_transform_name, camera_transform in sorted(result.transforms.items()):
        if camera_transform.child != "camera0":
            continue
        lidar_transform_name = f"T_{camera_transform.parent}_lidar0"
        lidar_transform = result.transforms.get(lidar_transform_name)
        if lidar_transform is None:
            continue
        t_parent_camera = camera_transform.as_se3()
        t_parent_lidar = t_parent_camera.compose(t_camera_lidar)
        lidar_transform.translation_m = list(t_parent_lidar.translation_m)
        lidar_transform.rotation_quat_xyzw = list(t_parent_lidar.rotation_quat_xyzw)
        return lidar_transform_name
    return None


def _raise_on_dry_run_failures(inspection: DatasetInspection) -> None:
    # Missing datasets are a dry-run warning rather than a hard failure at this
    # stage so users can validate schemas before collecting logs.
    _ = inspection


def _build_provisional_result(
    config: CalibrationConfig,
    config_path: Path,
    frame_graph: FrameGraph,
    inspection: DatasetInspection,
    options: CalibrationRunOptions,
) -> CalibrationResult:
    run_id = _run_id(config.project.name)
    transforms: dict[str, TransformResult] = {}
    candidate_extrinsics: dict[str, TransformResult] = {}
    for name, node in sorted(frame_graph.nodes.items()):
        if node.parent is None:
            continue
        transform_name = f"T_{node.parent}_{name}"
        transform = node.transform_to_parent
        transforms[transform_name] = _transform_result_from_initial(
            parent=node.parent,
            child=name,
            transform=transform,
            estimate=node.estimate,
            estimate_id=transform_name,
            provenance=TransformEstimateProvenance(
                producer="slac_native",
                execution_mode="offline_batch",
                role_in_comparison="output",
                evidence_level="unknown",
                source="config.frame_graph",
                notes=["alpha output initialized from frame graph before native refinement"],
            ),
        )
        candidate_extrinsics[transform_name] = _transform_result_from_initial(
            parent=node.parent,
            child=name,
            transform=transform,
            estimate=node.estimate,
            estimate_id=transform_name,
            provenance=TransformEstimateProvenance(
                producer="human",
                execution_mode="manual",
                role_in_comparison="candidate",
                evidence_level="imported_without_documented_derivation",
                source="config.frame_graph",
            ),
        )

    time_offsets: dict[str, TimeOffsetResult] = {}
    for sensor_name, offset in sorted(config.time_offsets.items()):
        time_offsets[f"dt_{sensor_name}"] = TimeOffsetResult(
            seconds=offset.initial_sec,
            std_seconds=0.0 if not offset.estimate else (offset.prior_sigma_sec or 0.01),
            quality=TimeOffsetQuality(grade="warn" if offset.estimate else "pass"),
        )

    metrics = _default_metrics(config, inspection)
    return CalibrationResult(
        run=RunInfo(
            id=run_id,
            slac_version=__version__,
            git_commit=git_commit(),
            config_sha256=sha256_path(config_path),
            dataset_sha256=sha256_path(Path(config.dataset.path)),
            status="warning",
            domain=config.project.domain,
            provenance={
                "pipeline": config.pipeline.type,
                "solver_backend": config.solver.backend,
                "max_iterations": config.solver.max_iterations,
                "seed": options.seed if options.seed is not None else config.solver.seed,
                "dataset_type": config.dataset.type,
                "dataset_path": config.dataset.path,
                "dry_run": False,
            },
        ),
        frame_graph=frame_graph.snapshot(),
        candidate_extrinsics=candidate_extrinsics,
        transforms=transforms,
        time_offsets=time_offsets,
        metrics=metrics,
        observability=ObservabilityResult(
            rank=None,
            condition_number=None,
            weak_directions=["uncomputed_alpha_backend"],
            grade="warn",
        ),
        degeneracy=degeneracy_from_inspection(inspection),
    )


def _transform_result_from_initial(
    *,
    parent: str,
    child: str,
    transform: SE3,
    estimate: bool,
    estimate_id: str | None = None,
    provenance: TransformEstimateProvenance | None = None,
) -> TransformResult:
    return TransformResult(
        parent=parent,
        child=child,
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        estimate_id=estimate_id,
        provenance=provenance or TransformEstimateProvenance(),
        quality=TransformQuality(
            grade="warn" if estimate else "pass",
            std_translation_m=(
                [0.0, 0.0, 0.0] if not estimate else [0.1, 0.1, 0.1]
            ),
            std_rotation_deg=(
                [0.0, 0.0, 0.0] if not estimate else [5.0, 5.0, 5.0]
            ),
        ),
    )


def _default_metrics(
    config: CalibrationConfig,
    inspection: DatasetInspection,
) -> dict[str, MetricResult]:
    metrics: dict[str, MetricResult] = {
        "schema_validation": MetricResult(
            value=1.0,
            grade="pass",
            reason="config parsed successfully",
        ),
        "dataset_exists": MetricResult(
            value=1.0 if inspection.exists else 0.0,
            grade="pass" if inspection.exists else "warn",
            reason="dataset path exists" if inspection.exists else "dataset path was not found",
        ),
        "prototype_solver": MetricResult(
            value=0.0,
            grade="warn",
            reason="alpha pipeline emitted initial values; production solver is not yet enabled",
        ),
    }
    if config.project.domain == "autonomous_driving":
        metrics["autonomous_driving_dynamic_holdout"] = MetricResult(
            value=None,
            grade="warn",
            reason=(
                "dynamic-object holdout and radar-camera-lidar timing checks are "
                "required before deployment"
            ),
        )
    metrics.update(lidar_metrics_from_inspection(inspection, config))
    metrics.update(motion_metrics_from_inspection(inspection))
    metrics.update(timing_metrics_from_inspection(inspection))
    for metric_name in config.evaluation.metrics:
        metrics.setdefault(
            metric_name,
            MetricResult(
                value=None,
                grade="warn",
                reason=f"{metric_name} is declared but not computed by the alpha backend",
            ),
        )
    return metrics


def _run_id(project_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", project_name).strip("_") or "calibrex"
    return f"{timestamp}_{slug}"
