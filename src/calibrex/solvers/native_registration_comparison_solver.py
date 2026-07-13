"""Public-data adapter for native ICP and optional GICP/NDT comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
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
from calibrex.solvers.registration_adapters import (
    NdtSubprocessAdapter,
    NdtSubprocessOptions,
    Open3DGeneralizedIcpAdapter,
    Open3DGeneralizedIcpOptions,
    RegistrationAdapterResult,
)
from calibrex.solvers.robust_point_to_point_icp_solver import (
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
        metrics = _metrics(native, gicp, ndt, factor_options)
        warnings = _warnings(native, gicp, ndt, metrics)
        transform = native.transform_target_source
        status = "pass" if transform is not None and not warnings else "inconclusive"
        weak = (
            list(native.rematching_diagnostics.weak_directions)
            if native.rematching_diagnostics
            else []
        )
        if native.symmetry_ambiguous:
            weak.append("multi_start_symmetry")
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
    gicp: RegistrationAdapterResult,
    ndt: RegistrationAdapterResult,
    options: dict[str, Any],
) -> dict[str, MetricResult]:
    max_rmse = float(options.get("max_holdout_rmse_m", 0.75))
    min_inlier = float(options.get("min_inlier_fraction", 0.25))
    holdout_pass = native.holdout_rmse_m is not None and native.holdout_rmse_m <= max_rmse
    weak_count = (
        len(native.rematching_diagnostics.weak_directions) if native.rematching_diagnostics else 6
    )
    jaccard = (
        native.rematching_diagnostics.minimum_correspondence_jaccard
        if native.rematching_diagnostics
        else None
    )
    return {
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
            grade="pass" if jaccard is not None and jaccard >= 0.5 else "warn",
        ),
        "registration_native_icp_weak_direction_count": MetricResult(
            value=float(weak_count), unit="dof", grade="pass" if weak_count == 0 else "warn"
        ),
        "registration_native_icp_symmetry_ambiguous": MetricResult(
            value=float(native.symmetry_ambiguous),
            unit="bool",
            grade="warn" if native.symmetry_ambiguous else "pass",
        ),
        "registration_open3d_gicp_available": MetricResult(
            value=float(gicp.available),
            unit="bool",
            grade="pass" if gicp.available else "warn",
            reason=gicp.status,
        ),
        "registration_open3d_gicp_holdout_rmse_m": MetricResult(
            value=gicp.common_evaluation.holdout_rmse_m if gicp.common_evaluation else None,
            unit="m",
            grade="pass" if gicp.common_evaluation else "warn",
            reason=gicp.status,
        ),
        "registration_external_ndt_available": MetricResult(
            value=float(ndt.available),
            unit="bool",
            grade="pass" if ndt.available else "warn",
            reason=ndt.status,
        ),
        "registration_external_ndt_holdout_rmse_m": MetricResult(
            value=ndt.common_evaluation.holdout_rmse_m if ndt.common_evaluation else None,
            unit="m",
            grade="pass" if ndt.common_evaluation else "warn",
            reason=ndt.status,
        ),
    }


def _warnings(
    native: RobustPointToPointIcpResult,
    gicp: RegistrationAdapterResult,
    ndt: RegistrationAdapterResult,
    metrics: dict[str, MetricResult],
) -> list[str]:
    warnings: list[str] = []
    if native.status != "converged":
        warnings.append(f"native ICP status is {native.status}: {native.reason}")
    if any(metric.grade == "fail" for metric in metrics.values()):
        warnings.append("native ICP public-data evidence did not satisfy all fixed gates")
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
