"""Native fixed-rig LiDAR point-to-plane solver adapter.

This adapter wires the pure-Python :class:`LidarRigPointToPlaneFactor` and
:class:`FixedTrajectorySe3ExtrinsicSolver` into the calibration pipeline. It is
selected explicitly with ``solver.backend: native_lidar_point_to_plane`` and
activates only when the dataset provides two LiDAR point-cloud frames (an A2D2
NPZ pair or a Livox PCD pair). The source frame is treated as the world frame of
a single fixed-trajectory pair, so ``T_world_ego`` is identity and the optimized
variable is the relative extrinsic ``T_source_target``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slac.core.config import CalibrationConfig
from slac.core.frames import FrameGraph
from slac.core.geometry import Vector3
from slac.core.result import MetricResult, ObservabilityResult
from slac.data.a2d2 import find_a2d2_lidar_npz_files, read_a2d2_lidar_points
from slac.data.inspect import DatasetInspection
from slac.data.livox import (
    LivoxPointRecord,
    find_livox_pcd_files,
    read_livox_binary_pcd_records,
)
from slac.evaluation.lidar import (
    build_rig_point_to_plane_observations,
    lidar_rig_point_to_plane_metrics_from_evaluation,
)
from slac.graph.lidar_point_to_plane import (
    LidarRigPointToPlaneEvaluation,
    LidarRigPointToPlaneFactor,
)
from slac.solvers.base import SolverAdapter, SolverAdapterResult
from slac.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
    FixedTrajectorySe3SolverResult,
    RobustLoss,
)

NATIVE_LIDAR_POINT_TO_PLANE_BACKEND = "native_lidar_point_to_plane"
_FACTOR_NAME = "lidar_rig_point_to_plane"
_SUPPORTED_DATASET_TYPES = ("a2d2_lidar", "livox_pcd")
_MIN_OBSERVATIONS = 6


@dataclass(frozen=True)
class _NativeSolveInputs:
    """Resolved geometry and tuning inputs for a native LiDAR pair solve."""

    voxel_size_m: float
    correspondence_gate_m: float
    max_source_points: int | None
    max_target_points: int | None


@dataclass(frozen=True)
class _LidarPairData:
    """Source plane records and target query points for one LiDAR pair."""

    source_records: list[LivoxPointRecord]
    target_points: list[Vector3]
    source_path: str
    target_path: str


class NativeLidarPointToPlaneSolver(SolverAdapter):
    """Run the native fixed-trajectory point-to-plane solver on a LiDAR pair."""

    backend = NATIVE_LIDAR_POINT_TO_PLANE_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        """Optimize a fixed-rig LiDAR extrinsic and report real observability."""

        del inspection
        lidars = sorted(
            name for name, sensor in config.sensors.items() if sensor.type == "lidar"
        )
        if len(lidars) < 2:
            return self._unavailable(
                "native LiDAR point-to-plane requires at least two LiDAR sensors"
            )
        if config.dataset.type not in _SUPPORTED_DATASET_TYPES:
            return self._unavailable(
                f"native LiDAR point-to-plane does not support dataset type "
                f"'{config.dataset.type}'"
            )
        files = _dataset_files(config)
        if len(files) < 2:
            return self._unavailable(
                "native LiDAR point-to-plane requires at least two LiDAR frames"
            )

        source_sensor, target_sensor = lidars[0], lidars[1]
        source_node = frame_graph.nodes.get(source_sensor)
        target_node = frame_graph.nodes.get(target_sensor)
        if source_node is None or target_node is None or target_node.parent is None:
            return self._unavailable(
                "source and target LiDAR frames must be present in the frame graph"
            )

        inputs = _solve_inputs(config)
        pair = _load_pair(config, files, inputs)
        t_base_source = source_node.transform_to_parent
        t_base_target = target_node.transform_to_parent
        t_source_target_init = t_base_source.inverse().compose(t_base_target)
        variable = f"T_{target_node.parent}_{target_sensor}"

        observations = build_rig_point_to_plane_observations(
            source_records=pair.source_records,
            target_points=pair.target_points,
            initial_t_source_target=t_source_target_init,
            voxel_size_m=inputs.voxel_size_m,
            correspondence_gate_m=inputs.correspondence_gate_m,
        )
        if len(observations) < _MIN_OBSERVATIONS:
            return self._unavailable(
                "native LiDAR point-to-plane found too few plane correspondences "
                f"({len(observations)} < {_MIN_OBSERVATIONS}); check voxel size, "
                "correspondence gate, and pair overlap"
            )

        factor = LidarRigPointToPlaneFactor(
            variable=variable,
            t_ego_lidar=t_source_target_init,
            observations=observations,
            sensor=target_sensor,
        )
        solver_options = FixedTrajectorySe3SolverOptions(
            max_iterations=max(1, config.solver.max_iterations),
            convergence_tolerance=config.solver.convergence_tolerance,
            robust_loss=_robust_loss(config.solver.robust_loss),
        )
        solver_result = FixedTrajectorySe3ExtrinsicSolver().solve(factor, solver_options)
        evaluation = factor.evaluate(list(solver_result.correction))

        metrics = lidar_rig_point_to_plane_metrics_from_evaluation(evaluation)
        metrics["native_lidar_point_to_plane_correspondence_count"] = MetricResult(
            value=float(len(observations)),
            unit="correspondences",
            grade="pass" if observations else "warn",
            reason="fixed point-to-plane correspondences built for the native solve",
        )
        metrics["prototype_solver"] = MetricResult(
            value=1.0 if solver_result.status == "converged" else 0.0,
            grade="pass" if solver_result.status == "converged" else "warn",
            reason=(
                "native fixed-trajectory point-to-plane solver produced optimized "
                f"extrinsics (status={solver_result.status})"
            ),
        )

        t_source_target_refined = solver_result.refined_transform
        t_base_target_refined = t_base_source.compose(t_source_target_refined)
        observability = ObservabilityResult(
            rank=evaluation.rank,
            condition_number=evaluation.normalized_condition_number_estimate,
            weak_directions=list(evaluation.weak_directions),
            grade=(
                "pass"
                if evaluation.rank >= 6 and not evaluation.weak_directions
                else "warn"
            ),
        )
        warnings = _warnings(solver_result, evaluation)
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status=solver_result.status,
            metrics=metrics,
            transforms={variable: t_base_target_refined},
            provenance=_provenance(
                variable=variable,
                source_sensor=source_sensor,
                target_sensor=target_sensor,
                pair=pair,
                inputs=inputs,
                observation_count=len(observations),
                solver_result=solver_result,
                evaluation_rank=evaluation.rank,
            ),
            warnings=warnings,
            observability=observability,
        )

    def _unavailable(self, reason: str) -> SolverAdapterResult:
        return SolverAdapterResult(
            backend=self.backend,
            available=False,
            status="unavailable",
            metrics={
                "native_lidar_point_to_plane_available": MetricResult(
                    value=0.0,
                    grade="warn",
                    reason=reason,
                )
            },
            provenance={
                "native_lidar_point_to_plane_status": "unavailable",
                "native_lidar_point_to_plane_reason": reason,
            },
            warnings=[reason],
            observability=None,
        )


def _dataset_files(config: CalibrationConfig) -> list[Path]:
    if config.dataset.type == "a2d2_lidar":
        return find_a2d2_lidar_npz_files(config.dataset.path)
    if config.dataset.type == "livox_pcd":
        return find_livox_pcd_files(config.dataset.path)
    return []


def _load_pair(
    config: CalibrationConfig,
    files: list[Path],
    inputs: _NativeSolveInputs,
) -> _LidarPairData:
    source_path, target_path = files[0], files[1]
    if config.dataset.type == "a2d2_lidar":
        source_points = read_a2d2_lidar_points(
            source_path, max_points=inputs.max_source_points
        )
        target_points = read_a2d2_lidar_points(
            target_path, max_points=inputs.max_target_points
        )
        source_records = [
            LivoxPointRecord(point=(x, y, z, 0.0), normal_xyz=None)
            for x, y, z in source_points
        ]
        return _LidarPairData(
            source_records=source_records,
            target_points=list(target_points),
            source_path=str(source_path),
            target_path=str(target_path),
        )
    source_records = _stride(
        read_livox_binary_pcd_records(source_path), inputs.max_source_points
    )
    target_records = _stride(
        read_livox_binary_pcd_records(target_path), inputs.max_target_points
    )
    return _LidarPairData(
        source_records=source_records,
        target_points=[
            (record.point[0], record.point[1], record.point[2])
            for record in target_records
        ],
        source_path=str(source_path),
        target_path=str(target_path),
    )


def _stride(records: list[LivoxPointRecord], max_records: int | None) -> list[LivoxPointRecord]:
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records
    step = (len(records) + max_records - 1) // max_records
    return records[::step]


def _solve_inputs(config: CalibrationConfig) -> _NativeSolveInputs:
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    return _NativeSolveInputs(
        voxel_size_m=_float_option(options, "voxel_size_m", default=1.0, minimum=0.05),
        correspondence_gate_m=_float_option(
            options, "correspondence_gate_m", default=1.5, minimum=0.05
        ),
        max_source_points=_int_option(options, "max_source_points", default=None),
        max_target_points=_int_option(options, "max_target_points", default=2000),
    )


def _float_option(
    options: dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
) -> float:
    try:
        value = float(options[key])
    except (KeyError, TypeError, ValueError):
        return default
    return max(value, minimum)


def _int_option(options: dict[str, Any], key: str, *, default: int | None) -> int | None:
    if key not in options:
        return default
    try:
        value = int(options[key])
    except (TypeError, ValueError):
        return default
    return value if value > 0 else None


def _robust_loss(name: str) -> RobustLoss:
    return "huber" if name == "huber" else "none"


def _warnings(
    solver_result: FixedTrajectorySe3SolverResult,
    evaluation: LidarRigPointToPlaneEvaluation,
) -> list[str]:
    warnings: list[str] = []
    if solver_result.status != "converged":
        warnings.append(
            "native LiDAR point-to-plane solver did not converge "
            f"({solver_result.status}: {solver_result.stop_reason})"
        )
    if evaluation.rank < 6:
        warnings.append(
            f"native LiDAR point-to-plane normal equations are rank-deficient "
            f"(rank {evaluation.rank} < 6)"
        )
    if evaluation.weak_directions:
        warnings.append(
            "native LiDAR point-to-plane weak DoF: "
            + ", ".join(evaluation.weak_directions)
        )
    return warnings


def _provenance(
    *,
    variable: str,
    source_sensor: str,
    target_sensor: str,
    pair: _LidarPairData,
    inputs: _NativeSolveInputs,
    observation_count: int,
    solver_result: FixedTrajectorySe3SolverResult,
    evaluation_rank: int,
) -> dict[str, Any]:
    return {
        "native_lidar_point_to_plane_status": solver_result.status,
        "native_lidar_point_to_plane_variable": variable,
        "native_lidar_point_to_plane_source_sensor": source_sensor,
        "native_lidar_point_to_plane_target_sensor": target_sensor,
        "native_lidar_point_to_plane_source_path": pair.source_path,
        "native_lidar_point_to_plane_target_path": pair.target_path,
        "native_lidar_point_to_plane_voxel_size_m": inputs.voxel_size_m,
        "native_lidar_point_to_plane_correspondence_gate_m": inputs.correspondence_gate_m,
        "native_lidar_point_to_plane_correspondence_count": observation_count,
        "native_lidar_point_to_plane_observability_rank": evaluation_rank,
        "native_lidar_point_to_plane_convention": (
            "world=source LiDAR frame; T_world_ego=identity; optimized variable is "
            "T_source_target with left-SE(3) correction [x,y,z,roll,pitch,yaw]"
        ),
        "native_lidar_point_to_plane_solver": solver_result.as_dict(),
    }
