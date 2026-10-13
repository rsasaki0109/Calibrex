"""Public-data adapter for native ICP and optional GICP/NDT comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.provenance import sha256_path
from calibrex.core.result import MetricResult, ObservabilityResult
from calibrex.data.downloads import (
    LIVOX_BASE_PCD_NAME,
    LIVOX_BASE_PCD_URL,
    LIVOX_TARGET_PCD_NAME,
    LIVOX_TARGET_PCD_URL,
)
from calibrex.data.inspect import DatasetInspection
from calibrex.data.livox import read_livox_binary_pcd
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.chen_medioni_point_to_plane_icp_solver import (
    ChenMedioniPointToPlaneIcpSolver,
    ChenMedioniPointToPlaneOptions,
    ChenMedioniPointToPlaneResult,
)
from calibrex.solvers.registration_adapters import (
    NdtSubprocessAdapter,
    NdtSubprocessOptions,
    Open3DGeneralizedIcpAdapter,
    Open3DGeneralizedIcpOptions,
    RegistrationAdapterResult,
)
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpCandidateEvaluation,
    IcpPoint,
    RobustPointToPointIcpOptions,
    RobustPointToPointIcpResult,
    RobustPointToPointIcpSolver,
)

NATIVE_REGISTRATION_COMPARISON_BACKEND = "native_registration_comparison"


class NativeRegistrationComparisonSolver(SolverAdapter):
    """Run registration backends with one frame convention and evidence policy."""

    backend = NATIVE_REGISTRATION_COMPARISON_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        del inspection
        factor_options = _factor_options(config)
        base_path = Path(config.dataset.path) / LIVOX_BASE_PCD_NAME
        target_path = Path(config.dataset.path) / LIVOX_TARGET_PCD_NAME
        missing = [path for path in (base_path, target_path) if not path.exists()]
        if missing:
            reason = "registration input is missing: " + ", ".join(map(str, missing))
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status="missing_dataset",
                metrics={
                    "registration_dataset_available": MetricResult(
                        value=0.0, unit="bool", grade="fail", reason=reason
                    )
                },
                warnings=[reason],
            )

        voxel_size = float(factor_options.get("voxel_size_m", 0.5))
        max_points = int(factor_options.get("max_points_per_cloud", 500))
        base_raw = read_livox_binary_pcd(base_path)
        target_raw = read_livox_binary_pcd(target_path)
        # The result edge is T_base_horizon_target_horizon, hence the target
        # sensor cloud is the ICP source and the base sensor cloud is its target.
        source = voxel_downsample_icp_points(
            [point[:3] for point in target_raw], "target_horizon", voxel_size, max_points
        )
        target = voxel_downsample_icp_points(
            [point[:3] for point in base_raw], "base_horizon", voxel_size, max_points
        )
        initial = frame_graph.nodes["target_horizon"].transform_to_parent
        common_options = _icp_options(config, factor_options)
        native = RobustPointToPointIcpSolver().solve(
            source, target, initial_transform=initial, options=common_options
        )
        point_to_plane = ChenMedioniPointToPlaneIcpSolver().solve(
            source,
            target,
            initial_transform=initial,
            common_options=common_options,
            options=ChenMedioniPointToPlaneOptions(
                normal_neighbor_count=int(factor_options.get("normal_neighbor_count", 12)),
                minimum_normal_eigengap=float(
                    factor_options.get("minimum_normal_eigengap", 0.02)
                ),
                max_condition_number=float(
                    factor_options.get("max_point_to_plane_condition_number", 1.0e8)
                ),
                known_bad_residual_margin_m=float(
                    factor_options.get("point_to_plane_probe_margin_m", 0.005)
                ),
            ),
        )
        gicp = Open3DGeneralizedIcpAdapter().solve(
            source,
            target,
            initial,
            common_options=common_options,
            options=Open3DGeneralizedIcpOptions(
                correspondence_distance_m=common_options.correspondence_distance_m,
                max_iterations=config.solver.max_iterations,
            ),
        )
        ndt = _run_ndt(source, target, common_options, factor_options)
        metrics = _metrics(native, point_to_plane, gicp, ndt, factor_options)
        warnings = _warnings(native, point_to_plane, gicp, ndt, metrics)
        transform = native.transform_target_source
        status = "pass" if transform is not None and not warnings else "inconclusive"
        weak = (
            list(native.rematching_diagnostics.weak_directions)
            if native.rematching_diagnostics
            else []
        )
        if native.symmetry_ambiguous:
            weak.append("multi_start_symmetry")
        if point_to_plane.tangent_rank < 6:
            weak.append("chen_medioni_tangent_geometry")
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status=status,
            metrics=metrics,
            transforms={"T_base_horizon_target_horizon": transform} if transform else {},
            observability=ObservabilityResult(
                rank=native.information_rank,
                condition_number=native.condition_number,
                weak_directions=weak,
                grade="pass" if native.information_rank == 6 and not weak else "warn",
            ),
            provenance={
                "metrics_origin": "recomputed",
                "raw_input_files": [
                    _input_manifest(base_path, "icp_target_base_horizon", LIVOX_BASE_PCD_URL),
                    _input_manifest(target_path, "icp_source_target_horizon", LIVOX_TARGET_PCD_URL),
                ],
                "native_registration_comparison": {
                    "frame_convention": (
                        "T_base_horizon_target_horizon maps target PCD into base PCD"
                    ),
                    "downsampling": {
                        "policy": "lexicographic voxel centroid then deterministic uniform cap",
                        "voxel_size_m": voxel_size,
                        "max_points_per_cloud": max_points,
                        "raw_source_count": len(target_raw),
                        "raw_target_count": len(base_raw),
                        "source_count": len(source),
                        "target_count": len(target),
                    },
                    "native_icp": native.as_dict(),
                    "native_chen_medioni_point_to_plane": point_to_plane.as_dict(),
                    "open3d_gicp": gicp.as_dict(),
                    "external_ndt": ndt.as_dict(),
                    "external_code_vendored": False,
                },
            },
            warnings=warnings,
        )


def voxel_downsample_icp_points(
    points: Sequence[Sequence[float]],
    id_prefix: str,
    voxel_size_m: float,
    max_points: int,
) -> list[IcpPoint]:
    """Return deterministic voxel centroids with a spatially uniform hard cap."""

    if voxel_size_m <= 0.0 or max_points < 1:
        raise ValueError("voxel_size_m and max_points must be positive")
    accumulators: dict[tuple[int, int, int], tuple[float, float, float, int]] = {}
    for values in points:
        x, y, z = (float(values[0]), float(values[1]), float(values[2]))
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        key = (
            math.floor(x / voxel_size_m),
            math.floor(y / voxel_size_m),
            math.floor(z / voxel_size_m),
        )
        sx, sy, sz, count = accumulators.get(key, (0.0, 0.0, 0.0, 0))
        accumulators[key] = (sx + x, sy + y, sz + z, count + 1)
    centroids = [
        (key, (sx / count, sy / count, sz / count))
        for key, (sx, sy, sz, count) in sorted(accumulators.items())
    ]
    if len(centroids) > max_points:
        indexes = [index * len(centroids) // max_points for index in range(max_points)]
        centroids = [centroids[index] for index in indexes]
    return [
        IcpPoint(f"{id_prefix}:{key[0]}:{key[1]}:{key[2]}", position) for key, position in centroids
    ]


def _icp_options(
    config: CalibrationConfig, factor_options: dict[str, Any]
) -> RobustPointToPointIcpOptions:
    return RobustPointToPointIcpOptions(
        max_iterations=config.solver.max_iterations,
        convergence_tolerance_m=config.solver.convergence_tolerance,
        correspondence_distance_m=float(factor_options.get("correspondence_distance_m", 1.5)),
        trim_fraction=float(factor_options.get("trim_fraction", 0.8)),
        mutual_correspondences=bool(factor_options.get("mutual_correspondences", True)),
        holdout_ratio=config.evaluation.holdout_ratio,
        holdout_voxel_size_m=float(factor_options.get("holdout_voxel_size_m", 1.0)),
        split_seed=config.solver.seed or 0,
        multi_start_translation_m=float(factor_options.get("multi_start_translation_m", 0.10)),
        multi_start_rotation_rad=math.radians(
            float(factor_options.get("multi_start_rotation_deg", 3.0))
        ),
    )


def _run_ndt(
    source: list[IcpPoint],
    target: list[IcpPoint],
    common_options: RobustPointToPointIcpOptions,
    factor_options: dict[str, Any],
) -> RegistrationAdapterResult:
    result_path = Path(str(factor_options.get("ndt_result_path", "")))
    if not str(factor_options.get("ndt_result_path", "")):
        result_path = Path("__calibrex_ndt_not_configured__.yaml")
    command = tuple(str(item) for item in factor_options.get("ndt_command", []))
    return NdtSubprocessAdapter().solve(
        source,
        target,
        NdtSubprocessOptions(
            result_path=result_path,
            command=command,
            execute=bool(factor_options.get("execute_ndt", False)),
            tool_name=str(factor_options.get("ndt_tool_name", "external_ndt")),
            tool_version=str(factor_options.get("ndt_tool_version", "unknown")),
            license_spdx=str(factor_options.get("ndt_license_spdx", "unknown")),
            training_isolation_declared=bool(
                factor_options.get("ndt_training_isolation_declared", False)
            ),
        ),
        common_options=common_options,
    )


def _metrics(
    native: RobustPointToPointIcpResult,
    point_to_plane: ChenMedioniPointToPlaneResult,
    gicp: RegistrationAdapterResult,
    ndt: RegistrationAdapterResult,
    options: dict[str, Any],
) -> dict[str, MetricResult]:
    max_rmse = float(options.get("max_holdout_rmse_m", 0.75))
    min_inlier = float(options.get("min_inlier_fraction", 0.25))
    min_jaccard = float(options.get("min_correspondence_jaccard", 0.5))
    max_point_to_plane_rmse = float(options.get("max_point_to_plane_holdout_rmse_m", 0.5))
    min_normal_fraction = float(options.get("min_valid_target_normal_fraction", 0.5))
    min_normal_gap = float(options.get("minimum_normal_eigengap", 0.02))
    max_point_to_plane_condition = float(
        options.get("max_point_to_plane_condition_number", 1.0e8)
    )
    min_detectable = float(options.get("min_point_to_plane_probe_detectable_fraction", 0.75))
    holdout_pass = native.holdout_rmse_m is not None and native.holdout_rmse_m <= max_rmse
    weak_count = (
        len(native.rematching_diagnostics.weak_directions) if native.rematching_diagnostics else 6
    )
    jaccard = (
        native.rematching_diagnostics.minimum_correspondence_jaccard
        if native.rematching_diagnostics
        else None
    )
    metrics = {
        "registration_dataset_available": MetricResult(value=1.0, unit="bool", grade="pass"),
        "registration_native_icp_train_rmse_m": MetricResult(
            value=native.train_rmse_m,
            unit="m",
            grade="pass" if native.train_rmse_m is not None else "fail",
        ),
        "registration_native_icp_holdout_rmse_m": MetricResult(
            value=native.holdout_rmse_m,
            unit="m",
            grade="pass" if holdout_pass else "fail",
            reason=f"public-data gate <= {max_rmse:g} m",
        ),
        "registration_native_icp_inlier_fraction": MetricResult(
            value=native.inlier_fraction,
            unit="fraction",
            grade="pass" if native.inlier_fraction >= min_inlier else "fail",
        ),
        "registration_native_icp_min_correspondence_jaccard": MetricResult(
            value=jaccard,
            unit="fraction",
            grade="pass" if jaccard is not None and jaccard >= min_jaccard else "warn",
            reason=f"rematched correspondence Jaccard >= {min_jaccard:g}",
        ),
        "registration_native_icp_weak_direction_count": MetricResult(
            value=float(weak_count), unit="dof", grade="pass" if weak_count == 0 else "warn"
        ),
        "registration_native_icp_symmetry_ambiguous": MetricResult(
            value=float(native.symmetry_ambiguous),
            unit="bool",
            grade="warn" if native.symmetry_ambiguous else "pass",
        ),
        "registration_chen_medioni_valid_target_normal_fraction": MetricResult(
            value=point_to_plane.target_normal_fraction,
            unit="fraction",
            grade=(
                "pass"
                if point_to_plane.target_normal_fraction >= min_normal_fraction
                else "fail"
            ),
            reason=f"valid PCA target-normal fraction >= {min_normal_fraction:g}",
        ),
        "registration_chen_medioni_target_normal_eigengap_minimum": MetricResult(
            value=point_to_plane.target_normal_eigengap_minimum,
            unit="ratio",
            grade=(
                "pass"
                if point_to_plane.target_normal_eigengap_minimum is not None
                and point_to_plane.target_normal_eigengap_minimum >= min_normal_gap
                else "fail"
            ),
            reason=f"every accepted target tangent has PCA eigengap >= {min_normal_gap:g}",
        ),
        "registration_chen_medioni_tangent_rank": MetricResult(
            value=float(point_to_plane.tangent_rank),
            unit="rank",
            grade="pass" if point_to_plane.tangent_rank == 6 else "fail",
            reason="point-to-plane tangent system must constrain all six directions",
        ),
        "registration_chen_medioni_tangent_condition_number": MetricResult(
            value=point_to_plane.tangent_condition_number,
            grade=(
                "pass"
                if point_to_plane.tangent_condition_number is not None
                and point_to_plane.tangent_condition_number <= max_point_to_plane_condition
                else "fail"
            ),
            reason=f"point-to-plane tangent condition number <= {max_point_to_plane_condition:g}",
        ),
        "registration_chen_medioni_train_point_to_plane_rmse_m": MetricResult(
            value=point_to_plane.train_point_to_plane_rmse_m,
            unit="m",
            grade=("pass" if point_to_plane.train_point_to_plane_rmse_m is not None else "fail"),
        ),
        "registration_chen_medioni_holdout_point_to_plane_rmse_m": MetricResult(
            value=point_to_plane.holdout_point_to_plane_rmse_m,
            unit="m",
            grade=(
                "pass"
                if point_to_plane.holdout_point_to_plane_rmse_m is not None
                and point_to_plane.holdout_point_to_plane_rmse_m <= max_point_to_plane_rmse
                else "fail"
            ),
            reason=f"point-to-plane spatial-holdout gate <= {max_point_to_plane_rmse:g} m",
        ),
        "registration_chen_medioni_known_bad_detectable_fraction": MetricResult(
            value=(
                sum(probe.detectable is True for probe in point_to_plane.probes)
                / len(point_to_plane.probes)
                if point_to_plane.probes
                else None
            ),
            unit="fraction",
            grade=(
                "pass"
                if point_to_plane.probes
                and sum(probe.detectable is True for probe in point_to_plane.probes)
                / len(point_to_plane.probes)
                >= min_detectable
                else "warn"
            ),
            reason=f"signed point-to-plane controls detectable fraction >= {min_detectable:g}",
        ),
    }
    point_to_plane_common = point_to_plane.common_evaluation
    metrics.update(
        _candidate_common_metrics(
            "registration_chen_medioni_common",
            point_to_plane_common,
            max_rmse=max_rmse,
            min_inlier=min_inlier,
            min_jaccard=min_jaccard,
        )
    )
    metrics.update(
        _adapter_common_metrics(
            "registration_open3d_gicp",
            gicp,
            max_rmse=max_rmse,
            min_inlier=min_inlier,
            min_jaccard=min_jaccard,
        )
    )
    metrics.update(
        _adapter_common_metrics(
            "registration_external_ndt",
            ndt,
            max_rmse=max_rmse,
            min_inlier=min_inlier,
            min_jaccard=min_jaccard,
        )
    )
    evaluated: list[IcpCandidateEvaluation] = []
    if point_to_plane_common is not None:
        evaluated.append(point_to_plane_common)
    evaluated.extend(
        result.common_evaluation
        for result in (gicp, ndt)
        if result.common_evaluation is not None
    )
    split_consistent = all(
        evaluation.train_source_ids == native.train_source_ids
        and evaluation.holdout_source_ids == native.holdout_source_ids
        for evaluation in evaluated
    )
    metrics["registration_backend_common_split_consistent"] = MetricResult(
        value=float(split_consistent) if evaluated else None,
        unit="bool",
        grade="pass" if evaluated and split_consistent else "warn",
        reason="every executed adapter must reuse native spatial train/holdout IDs",
    )
    for prefix, result in (
        ("registration_open3d_gicp_native", gicp),
        ("registration_external_ndt_native", ndt),
    ):
        rotation_delta, translation_delta = _transform_delta(
            result.transform_target_source, native.transform_target_source
        )
        metrics[f"{prefix}_rotation_delta_deg"] = MetricResult(
            value=rotation_delta,
            unit="deg",
            grade="warn",
            reason="backend-comparison diagnostic; public data have no transform ground truth",
        )
        metrics[f"{prefix}_translation_delta_m"] = MetricResult(
            value=translation_delta,
            unit="m",
            grade="warn",
            reason="backend-comparison diagnostic; public data have no transform ground truth",
        )
    return metrics


def _adapter_common_metrics(
    prefix: str,
    result: RegistrationAdapterResult,
    *,
    max_rmse: float,
    min_inlier: float,
    min_jaccard: float,
) -> dict[str, MetricResult]:
    """Expose one adapter through the same rematching-aware evaluation gates."""

    evaluation = result.common_evaluation
    rematching = evaluation.rematching_diagnostics if evaluation is not None else None
    jaccard = rematching.minimum_correspondence_jaccard if rematching is not None else None
    weak_count = len(rematching.weak_directions) if rematching is not None else None
    unavailable_grade: Literal["fail", "warn"] = "fail" if result.available else "warn"
    return {
        f"{prefix}_available": MetricResult(
            value=float(result.available),
            unit="bool",
            grade="pass" if result.available else "warn",
            reason=result.status,
        ),
        f"{prefix}_train_rmse_m": MetricResult(
            value=evaluation.train_rmse_m if evaluation is not None else None,
            unit="m",
            grade=(
                "pass"
                if evaluation is not None and evaluation.train_rmse_m is not None
                else unavailable_grade
            ),
            reason=result.status,
        ),
        f"{prefix}_holdout_rmse_m": MetricResult(
            value=evaluation.holdout_rmse_m if evaluation is not None else None,
            unit="m",
            grade=(
                "pass"
                if evaluation is not None
                and evaluation.holdout_rmse_m is not None
                and evaluation.holdout_rmse_m <= max_rmse
                else unavailable_grade
            ),
            reason=f"common spatial-holdout gate <= {max_rmse:g} m; status={result.status}",
        ),
        f"{prefix}_inlier_fraction": MetricResult(
            value=evaluation.inlier_fraction if evaluation is not None else None,
            unit="fraction",
            grade=(
                "pass"
                if evaluation is not None and evaluation.inlier_fraction >= min_inlier
                else unavailable_grade
            ),
            reason=f"common train inlier fraction >= {min_inlier:g}; status={result.status}",
        ),
        f"{prefix}_min_correspondence_jaccard": MetricResult(
            value=jaccard,
            unit="fraction",
            grade=(
                "pass"
                if jaccard is not None and jaccard >= min_jaccard
                else ("warn" if not result.available or evaluation is not None else "fail")
            ),
            reason=f"common rematched correspondence Jaccard >= {min_jaccard:g}",
        ),
        f"{prefix}_weak_direction_count": MetricResult(
            value=float(weak_count) if weak_count is not None else None,
            unit="dof",
            grade=(
                "pass"
                if weak_count == 0
                else ("warn" if not result.available or evaluation is not None else "fail")
            ),
            reason="common rematching curvature must constrain all six directions",
        ),
    }


def _candidate_common_metrics(
    prefix: str,
    evaluation: IcpCandidateEvaluation | None,
    *,
    max_rmse: float,
    min_inlier: float,
    min_jaccard: float,
) -> dict[str, MetricResult]:
    """Expose a native candidate through the same rematching-aware gates."""

    rematching = evaluation.rematching_diagnostics if evaluation is not None else None
    jaccard = rematching.minimum_correspondence_jaccard if rematching is not None else None
    weak_count = len(rematching.weak_directions) if rematching is not None else None
    return {
        f"{prefix}_train_rmse_m": MetricResult(
            value=evaluation.train_rmse_m if evaluation is not None else None,
            unit="m",
            grade=(
                "pass"
                if evaluation is not None and evaluation.train_rmse_m is not None
                else "fail"
            ),
        ),
        f"{prefix}_holdout_rmse_m": MetricResult(
            value=evaluation.holdout_rmse_m if evaluation is not None else None,
            unit="m",
            grade=(
                "pass"
                if evaluation is not None
                and evaluation.holdout_rmse_m is not None
                and evaluation.holdout_rmse_m <= max_rmse
                else "fail"
            ),
            reason=f"common spatial-holdout gate <= {max_rmse:g} m",
        ),
        f"{prefix}_inlier_fraction": MetricResult(
            value=evaluation.inlier_fraction if evaluation is not None else None,
            unit="fraction",
            grade=(
                "pass"
                if evaluation is not None and evaluation.inlier_fraction >= min_inlier
                else "fail"
            ),
            reason=f"common train inlier fraction >= {min_inlier:g}",
        ),
        f"{prefix}_min_correspondence_jaccard": MetricResult(
            value=jaccard,
            unit="fraction",
            grade="pass" if jaccard is not None and jaccard >= min_jaccard else "warn",
            reason=f"common rematched correspondence Jaccard >= {min_jaccard:g}",
        ),
        f"{prefix}_weak_direction_count": MetricResult(
            value=float(weak_count) if weak_count is not None else None,
            unit="dof",
            grade="pass" if weak_count == 0 else "warn",
            reason="common rematching curvature must constrain all six directions",
        ),
    }


def _transform_delta(left: SE3 | None, right: SE3 | None) -> tuple[float | None, float | None]:
    if left is None or right is None:
        return None, None
    delta = left.inverse().compose(right)
    quaternion_w = min(1.0, max(-1.0, abs(delta.rotation_quat_xyzw[3])))
    rotation_deg = math.degrees(2.0 * math.acos(quaternion_w))
    translation_m = math.sqrt(sum(value * value for value in delta.translation_m))
    return rotation_deg, translation_m


def _warnings(
    native: RobustPointToPointIcpResult,
    point_to_plane: ChenMedioniPointToPlaneResult,
    gicp: RegistrationAdapterResult,
    ndt: RegistrationAdapterResult,
    metrics: dict[str, MetricResult],
) -> list[str]:
    warnings: list[str] = []
    if native.status != "converged":
        warnings.append(f"native ICP status is {native.status}: {native.reason}")
    if point_to_plane.status != "converged":
        warnings.append(
            f"Chen-Medioni point-to-plane status is {point_to_plane.status}: "
            f"{point_to_plane.reason}"
        )
    if any(metric.grade == "fail" for metric in metrics.values()):
        warnings.append("registration public-data evidence did not satisfy all fixed gates")
    if not gicp.available:
        warnings.append(
            "Open3D GICP was unavailable; native result remains independently evaluated"
        )
    if not ndt.available:
        warnings.append("external NDT was not configured; no NDT result was fabricated")
    if native.symmetry_ambiguous:
        warnings.append("multi-start diagnostics found an equivalent distinct ICP solution")
    return warnings


def _input_manifest(path: Path, role: str, source_url: str) -> dict[str, object]:
    return {
        "path": str(path),
        "role": role,
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
        "source_url": source_url,
    }


def _factor_options(config: CalibrationConfig) -> dict[str, Any]:
    factor = config.pipeline.factors.get("registration_backend_comparison")
    return factor.options if factor is not None else {}
