"""Native joint pose-extrinsic SLAC adapter for public LiDAR cloud pairs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.provenance import sha256_path
from calibrex.core.result import Grade, MetricResult, ObservabilityResult
from calibrex.data.inspect import DatasetInspection
from calibrex.evaluation.lidar import build_rig_point_to_plane_observations
from calibrex.graph.joint_factors import (
    JointPointToPlaneMeasurement,
    make_joint_point_to_plane_factor,
    make_joint_prior_factor,
    se3_from_tangent,
)
from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointOptimizerOptions,
    JointOptimizerResult,
    JointParameterBlock,
    JointResidualBlock,
)
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.native_lidar_point_to_plane_solver import (
    LidarPairData,
    LidarPairSolveInputs,
    NativeLidarPointToPlaneSolver,
    lidar_pair_dataset_files,
    lidar_pair_solve_inputs,
    load_lidar_pair,
)

NATIVE_JOINT_SLAC_BACKEND = "native_joint_slac"
_FACTOR_NAME = "lidar_rig_point_to_plane"
_SUPPORTED_DATASET_TYPES = ("a2d2_lidar", "livox_pcd")
_POSE_BLOCK = "source_pose_correction"
_EXTRINSIC_BLOCK = "target_extrinsic_correction"
_MIN_OBSERVATIONS = 12


class NativeJointSlacSolver(SolverAdapter):
    """Jointly estimate a source-pose nuisance state and LiDAR extrinsic.

    A single synchronized cloud pair cannot distinguish those two transforms
    from geometry alone. The adapter therefore uses an explicit train-only
    source-pose prior and reports augmented, rather than data-only,
    observability. This makes the gauge choice inspectable in provenance.
    """

    backend = NATIVE_JOINT_SLAC_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        """Run the joint graph on a supported public or local LiDAR pair."""

        lidars = sorted(
            name for name, sensor in config.sensors.items() if sensor.type == "lidar"
        )
        if len(lidars) < 2:
            return _unavailable("native joint SLAC requires at least two LiDAR sensors")
        if config.dataset.type not in _SUPPORTED_DATASET_TYPES:
            return _unavailable(
                f"native joint SLAC does not support dataset type '{config.dataset.type}'"
            )
        files = lidar_pair_dataset_files(config)
        if len(files) < 2:
            return _unavailable("native joint SLAC requires at least two LiDAR frames")

        source_sensor, target_sensor = lidars[0], lidars[1]
        source_node = frame_graph.nodes.get(source_sensor)
        target_node = frame_graph.nodes.get(target_sensor)
        if source_node is None or target_node is None or target_node.parent is None:
            return _unavailable("source and target LiDAR frames must exist in the frame graph")

        frontend = lidar_pair_solve_inputs(config)
        pair = load_lidar_pair(config, files, frontend)
        t_base_source = source_node.transform_to_parent
        t_source_target_initial = t_base_source.inverse().compose(
            target_node.transform_to_parent
        )
        observations = build_rig_point_to_plane_observations(
            source_records=pair.source_records,
            target_points=pair.target_points,
            initial_t_source_target=t_source_target_initial,
            voxel_size_m=frontend.voxel_size_m,
            correspondence_gate_m=frontend.correspondence_gate_m,
        )
        if len(observations) < _MIN_OBSERVATIONS:
            return _unavailable(
                "native joint SLAC found too few plane correspondences "
                f"({len(observations)} < {_MIN_OBSERVATIONS})"
            )

        options = _joint_options(config)
        data_factors = [
            make_joint_point_to_plane_factor(
                JointPointToPlaneMeasurement(
                    measurement_id=f"a2d2-plane-{index:05d}",
                    observation_group=_spatial_group(
                        observation.plane_point_world_m,
                        options["holdout_voxel_m"],
                    ),
                    point_sensor_m=observation.point_lidar_m,
                    plane_point_world_m=observation.plane_point_world_m,
                    plane_normal_world=observation.plane_normal_world,
                    transform_world_body_initial=SE3.identity(),
                    transform_body_sensor_initial=t_source_target_initial,
                    weight=observation.weight,
                ),
                pose_block=_POSE_BLOCK,
                extrinsic_block=_EXTRINSIC_BLOCK,
            )
            for index, observation in enumerate(observations)
        ]
        pose_sigma = (
            options["pose_translation_sigma_m"],
            options["pose_translation_sigma_m"],
            options["pose_translation_sigma_m"],
            options["pose_rotation_sigma_rad"],
            options["pose_rotation_sigma_rad"],
            options["pose_rotation_sigma_rad"],
        )
        factors = [
            *data_factors,
            make_joint_prior_factor(
                factor_id="source-pose-gauge-prior",
                observation_group="source-pose-gauge-prior",
                block=_POSE_BLOCK,
                target=(0.0,) * 6,
                sigma=pose_sigma,
            ),
        ]
        known_bad = (
            options["known_bad_translation_m"],
            options["known_bad_translation_m"],
            options["known_bad_translation_m"],
            options["known_bad_rotation_rad"],
            options["known_bad_rotation_rad"],
            options["known_bad_rotation_rad"],
        )
        blocks = [
            JointParameterBlock(
                _POSE_BLOCK,
                (0.0,) * 6,
                finite_difference_steps=(1.0e-4,) * 3 + (1.0e-5,) * 3,
                known_bad_steps=known_bad,
            ),
            JointParameterBlock(
                _EXTRINSIC_BLOCK,
                (0.0,) * 6,
                finite_difference_steps=(1.0e-4,) * 3 + (1.0e-5,) * 3,
                known_bad_steps=known_bad,
            ),
        ]
        result = BackendNeutralJointOptimizer().solve(
            blocks,
            factors,
            JointOptimizerOptions(
                max_iterations=max(1, config.solver.max_iterations),
                convergence_tolerance=config.solver.convergence_tolerance,
                huber_delta=options["huber_delta_m"],
                holdout_ratio=options["holdout_ratio"],
                split_seed=config.solver.seed or 0,
                minimum_train_factors=6,
                max_condition_number=options["max_condition_number"],
                known_bad_margin=options["known_bad_margin_m"],
            ),
        )
        refined_relative = se3_from_tangent(
            result.optimized_values[_EXTRINSIC_BLOCK]
        ).compose(t_source_target_initial)
        variable = f"T_{target_node.parent}_{target_sensor}"
        refined = t_base_source.compose(refined_relative)
        baseline = NativeLidarPointToPlaneSolver().solve(config, frame_graph, inspection)
        metrics = _metrics(result, data_factors, baseline, variable, refined)
        warnings = _warnings(result)
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status=result.status,
            metrics=metrics,
            transforms={variable: refined},
            provenance=_provenance(
                result=result,
                pair=pair,
                frontend=frontend,
                options=options,
                source_sensor=source_sensor,
                target_sensor=target_sensor,
                variable=variable,
                observation_count=len(observations),
                baseline=baseline,
            ),
            warnings=warnings,
            observability=ObservabilityResult(
                rank=result.information_rank,
                condition_number=result.condition_number,
                weak_directions=list(result.weak_parameter_blocks),
                grade="warn",
            ),
        )


def _joint_options(config: CalibrationConfig) -> dict[str, float]:
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    values: dict[str, Any] = dict(factor.options) if factor is not None else {}
    return {
        "holdout_ratio": _bounded(values, "joint_holdout_ratio", 0.2, 0.05, 0.5),
        "holdout_voxel_m": _positive(values, "joint_holdout_voxel_m", 5.0),
        "pose_translation_sigma_m": _positive(
            values, "joint_pose_prior_translation_sigma_m", 0.02
        ),
        "pose_rotation_sigma_rad": math.radians(
            _positive(values, "joint_pose_prior_rotation_sigma_deg", 0.5)
        ),
        "known_bad_translation_m": _positive(
            values, "joint_known_bad_translation_m", 0.05
        ),
        "known_bad_rotation_rad": math.radians(
            _positive(values, "joint_known_bad_rotation_deg", 1.0)
        ),
        "known_bad_margin_m": _positive(values, "joint_known_bad_margin_m", 1.0e-4),
        "huber_delta_m": _positive(values, "joint_huber_delta_m", 0.1),
        "max_condition_number": _positive(
            values, "joint_max_condition_number", 1.0e12
        ),
    }


def _positive(values: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if value > 0.0 else default


def _bounded(
    values: dict[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    return min(max(_positive(values, key, default), minimum), maximum)


def _spatial_group(point: tuple[float, float, float], voxel_m: float) -> str:
    return "region:" + ":".join(str(math.floor(value / voxel_m)) for value in point)


def _metrics(
    result: JointOptimizerResult,
    data_factors: list[JointResidualBlock],
    baseline: SolverAdapterResult,
    variable: str,
    refined: SE3,
) -> dict[str, MetricResult]:
    train_ids = set(result.train_factor_ids)
    train_data = [factor for factor in data_factors if factor.factor_id in train_ids]
    train_rmse = _factor_rmse(train_data, result.optimized_values)
    detectable = [probe.detectable for probe in result.probes if probe.detectable is not None]
    detectable_fraction = (
        sum(value is True for value in detectable) / len(detectable) if detectable else None
    )
    rank_grade: Grade = "pass" if result.information_rank == 12 else "warn"
    family_fractions = [
        item.robust_information_fraction
        for item in result.train_factor_family_diagnostics
        if item.robust_information_fraction is not None
    ]
    metrics = {
        "native_joint_slac_available": MetricResult(
            value=1.0,
            grade="pass",
            reason="native backend-neutral joint optimizer executed",
        ),
        "joint_slac_point_to_plane_rmse_m": MetricResult(
            train=train_rmse,
            holdout=result.holdout_rmse,
            unit="m",
            grade="warn",
            reason=(
                "empirical point-to-plane residuals with pose prior excluded; "
                "reported without a universal accuracy threshold"
            ),
        ),
        "joint_slac_augmented_information_rank": MetricResult(
            value=float(result.information_rank),
            grade=rank_grade,
            reason="rank of data factors plus the declared source-pose gauge prior",
        ),
        "joint_slac_augmented_condition_number": MetricResult(
            value=result.condition_number,
            grade=(
                "pass"
                if result.condition_number is not None and result.status != "degenerate"
                else "warn"
            ),
            reason="unit-dependent augmented Jacobian diagnostic; not covariance",
        ),
        "joint_slac_known_bad_detectable_fraction": MetricResult(
            value=detectable_fraction,
            grade="pass" if detectable_fraction == 1.0 else "warn",
            reason="fraction of signed pose/extrinsic holdout perturbations that worsen RMSE",
        ),
        "joint_slac_train_factor_family_count": MetricResult(
            value=float(len(result.train_factor_family_diagnostics)),
            unit="families",
            grade="pass" if len(result.train_factor_family_diagnostics) >= 2 else "warn",
            reason="distinct data/prior factor families in the final train Jacobian",
        ),
        "joint_slac_train_max_family_information_fraction": MetricResult(
            value=max(family_fractions) if family_fractions else None,
            unit="fraction",
            grade="warn",
            reason=(
                "largest family share of Huber-weighted local Jacobian energy; "
                "unit-dependent diagnostic without a universal acceptance threshold"
            ),
        ),
    }
    baseline_transform = baseline.transforms.get(variable)
    if baseline.available and baseline_transform is not None:
        delta = baseline_transform.inverse().compose(refined)
        metrics["joint_slac_vs_fixed_baseline_translation_m"] = MetricResult(
            value=math.dist((0.0, 0.0, 0.0), delta.translation_m),
            unit="m",
            grade="pass",
            reason="SE(3) delta from independent fixed-trajectory native baseline",
        )
        metrics["joint_slac_vs_fixed_baseline_rotation_deg"] = MetricResult(
            value=_rotation_angle_deg(delta),
            unit="deg",
            grade="pass",
            reason="SE(3) delta from independent fixed-trajectory native baseline",
        )
    return metrics


def _factor_rmse(
    factors: list[JointResidualBlock], values: dict[str, tuple[float, ...]]
) -> float | None:
    residuals = [value for factor in factors for value in factor.residuals(values)]
    if not residuals:
        return None
    return math.sqrt(sum(value * value for value in residuals) / len(residuals))


def _rotation_angle_deg(transform: SE3) -> float:
    w = min(1.0, abs(transform.rotation_quat_xyzw[3]))
    return math.degrees(2.0 * math.acos(w))


def _warnings(result: JointOptimizerResult) -> list[str]:
    warnings = [
        "single-pair source pose and extrinsic are geometrically gauge-coupled; "
        "reported rank includes the explicit train-only source-pose prior"
    ]
    if result.status != "converged":
        warnings.append(f"native joint SLAC status is {result.status}: {result.reason}")
    if result.weak_parameter_blocks:
        warnings.append(
            "native joint SLAC weak blocks: " + ", ".join(result.weak_parameter_blocks)
        )
    return warnings


def _provenance(
    *,
    result: JointOptimizerResult,
    pair: LidarPairData,
    frontend: LidarPairSolveInputs,
    options: dict[str, float],
    source_sensor: str,
    target_sensor: str,
    variable: str,
    observation_count: int,
    baseline: SolverAdapterResult,
) -> dict[str, Any]:
    return {
        "native_joint_slac_method": "pose_extrinsic_point_to_plane/v0.1",
        "native_joint_slac_status": result.status,
        "native_joint_slac_variable": variable,
        "native_joint_slac_source_sensor": source_sensor,
        "native_joint_slac_target_sensor": target_sensor,
        "native_joint_slac_observation_count": observation_count,
        "native_joint_slac_source_path": pair.source_path,
        "native_joint_slac_source_sha256": sha256_path(Path(pair.source_path)),
        "native_joint_slac_target_path": pair.target_path,
        "native_joint_slac_target_sha256": sha256_path(Path(pair.target_path)),
        "native_joint_slac_frontend": {
            "voxel_size_m": frontend.voxel_size_m,
            "correspondence_gate_m": frontend.correspondence_gate_m,
            "max_source_points": frontend.max_source_points,
            "max_target_points": frontend.max_target_points,
        },
        "native_joint_slac_options": options,
        "native_joint_slac_gauge_policy": (
            "one source-pose correction and one target-extrinsic correction; "
            "source pose has a train-only diagonal prior; augmented rank includes prior"
        ),
        "native_joint_slac_train_holdout_policy": (
            "target correspondences split by source-plane spatial region; source plane map "
            "is shared; prior is train-only"
        ),
        "native_joint_slac_solver": result.as_dict(),
        "native_joint_slac_fixed_baseline_status": baseline.status,
        "native_joint_slac_fixed_baseline_backend": baseline.backend,
    }


def _unavailable(reason: str) -> SolverAdapterResult:
    return SolverAdapterResult(
        backend=NATIVE_JOINT_SLAC_BACKEND,
        available=False,
        status="unavailable",
        metrics={
            "native_joint_slac_available": MetricResult(
                value=0.0,
                grade="warn",
                reason=reason,
            )
        },
        provenance={
            "native_joint_slac_status": "unavailable",
            "native_joint_slac_reason": reason,
        },
        warnings=[reason],
    )
