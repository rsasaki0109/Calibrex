"""Native multi-capture joint pose/extrinsic adapter for public TUM RGB-D."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3, QuaternionXYZW, rotate_vector_xyzw
from calibrex.core.provenance import sha256_path
from calibrex.core.result import Grade, MetricResult, ObservabilityResult
from calibrex.data.inspect import DatasetInspection
from calibrex.data.livox import (
    LivoxPointRecord,
    VoxelPlaneMatch,
    build_voxel_plane_map,
    nearest_voxel_plane,
    nearest_voxel_plane_match,
)
from calibrex.data.tum_rgbd import (
    TUMDepthIntrinsics,
    TUMDepthPoseEntry,
    associate_depth_groundtruth,
    read_groundtruth,
    read_image_index,
    read_tum_depth_png,
    sample_tum_depth_points,
)
from calibrex.evaluation.correspondence import (
    CorrespondenceAssignment,
    CorrespondenceStabilityEvaluation,
    evaluate_correspondence_stability,
)
from calibrex.evaluation.multistart import (
    MultiStartEvaluation,
    PatternSearchOptions,
    PatternSearchResult,
    coordinate_pattern_search,
    evaluate_multistart_solutions,
)
from calibrex.evaluation.numerical_curvature import (
    NumericalCurvatureEvaluation,
    evaluate_numerical_curvature,
)
from calibrex.graph.joint_factors import (
    JointDepthPointToPlaneMeasurement,
    JointTrilinearDepthPointToPlaneMeasurement,
    JointTrilinearXYZPairMeasurement,
    JointTrilinearXYZPointToPlaneMeasurement,
    estimate_xyz_lattice_local_rotations,
    make_joint_centered_trilinear_depth_point_to_plane_factor,
    make_joint_depth_point_to_plane_factor,
    make_joint_lattice_smoothness_factor,
    make_joint_lattice_zero_mean_factor,
    make_joint_prior_factor,
    make_joint_trilinear_xyz_pair_factor,
    make_joint_trilinear_xyz_point_to_plane_factor,
    make_joint_xyz_lattice_rigid_gauge_factor,
    make_joint_xyz_lattice_shape_factor,
    regular_lattice_control_points,
    se3_from_tangent,
    trilinear_lattice_weights,
)
from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointObservabilityEvaluation,
    JointOptimizerOptions,
    JointOptimizerResult,
    JointParameterBlock,
    JointResidualBlock,
    ParameterValues,
    evaluate_joint_observability,
)
from calibrex.graph.joint_reassociation import (
    BackendNeutralJointReassociation,
    JointReassociationOptions,
    JointReassociationProbeEvaluation,
    JointReassociationProbeOptions,
    JointReassociationResult,
    JointReassociationState,
    evaluate_joint_reassociation_probes,
)
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult

NATIVE_TUM_JOINT_SLAC_BACKEND = "native_tum_joint_slac"
_FACTOR_NAME = "tum_rgbd_joint_point_to_plane"
_EXTRINSIC_BLOCK = "T_trajectory_camera_correction"
_DEPTH_BLOCK = "depth_log_scale_bias_m"
_SPATIAL_BIAS_BLOCK = "depth_spatial_constant_bias_m"
_SPATIAL_DEPTH_BLOCK = "depth_zero_mean_trilinear_ray_offsets_m"
_XYZ_LATTICE_BLOCK = "depth_full_xyz_lattice_offsets_m"
_SPATIAL_LATTICE_SHAPE = (2, 2, 2)
_SPATIAL_LATTICE_MINIMUM_M = (-3.1, -2.4, 0.2)
_SPATIAL_LATTICE_MAXIMUM_M = (3.1, 2.4, 5.0)
_MIN_QUERY_FRAMES = 4
_MIN_TOTAL_CORRESPONDENCES = 100


@dataclass(frozen=True)
class TUMJointSlacOptions:
    """Resolved public TUM frame, geometry, prior, and optimizer policy."""

    frame_start_index: int = 180
    frame_stride: int = 10
    map_frame_count: int = 3
    query_frame_count: int = 8
    map_points_per_frame: int = 5000
    query_points_per_frame: int = 1000
    max_correspondences_per_frame: int = 400
    voxel_size_m: float = 0.1
    correspondence_gate_m: float = 0.15
    max_pose_time_delta_sec: float = 0.02
    pose_prior_translation_sigma_m: float = 0.01
    pose_prior_rotation_sigma_rad: float = math.radians(0.5)
    holdout_ratio: float = 0.25
    huber_delta_m: float = 0.05
    known_bad_translation_m: float = 0.03
    known_bad_rotation_rad: float = math.radians(0.5)
    known_bad_margin_m: float = 1.0e-4
    reference_translation_gate_m: float = 0.05
    reference_rotation_gate_deg: float = 1.0
    known_bad_depth_log_scale: float = 0.02
    known_bad_depth_bias_m: float = 0.02
    reference_depth_scale_error_gate_percent: float = 2.0
    reference_depth_bias_gate_m: float = 0.03
    replication_start_indices: tuple[int, ...] = (60, 180, 300)
    cross_window_nondegradation_margin_m: float = 0.005
    spatial_lattice_smoothness_sigma_m: float = 0.05
    spatial_lattice_zero_mean_sigma_m: float = 1.0e-4
    reassociation_max_outer_iterations: int = 2
    reassociation_min_train_pair_jaccard: float = 0.99
    reassociation_min_train_retained_fraction: float = 0.95
    reassociation_unmatched_residual_penalty_m: float = 0.15
    xyz_lattice_shape_sigma_m: float = 0.05
    xyz_lattice_translation_gauge_sigma_m: float = 1.0e-4
    xyz_lattice_rotation_gauge_sigma_rad: float = 1.0e-4
    xyz_lattice_local_rotation_updates: int = 1
    xyz_pair_max_correspondences_per_pair: int = 120
    xyz_pair_stride: int = 1

    def as_dict(self) -> dict[str, float | int | list[int]]:
        values: dict[str, float | int | list[int]] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, tuple):
                values[name] = list(value)
            elif isinstance(value, (float, int)):
                values[name] = value
        return values


@dataclass(frozen=True)
class _TUMJointProblem:
    map_frames: tuple[TUMDepthPoseEntry, ...]
    query_frames: tuple[TUMDepthPoseEntry, ...]
    measurements: tuple[JointDepthPointToPlaneMeasurement, ...]
    data_factors: tuple[JointResidualBlock, ...]
    parameter_blocks: tuple[JointParameterBlock, ...]
    all_factors: tuple[JointResidualBlock, ...]
    map_record_count: int
    map_plane_count: int


@dataclass(frozen=True)
class _TUMWindowRun:
    start_index: int
    problem: _TUMJointProblem
    result: JointOptimizerResult


@dataclass(frozen=True)
class _TUMSpatialWindowRun:
    start_index: int
    source_problem: _TUMJointProblem
    data_factors: tuple[JointResidualBlock, ...]
    result: JointOptimizerResult


@dataclass(frozen=True)
class _TUMIterativeReassociationRun:
    start_index: int
    baseline: _TUMSpatialWindowRun
    result: JointReassociationResult
    probe_evaluation: JointReassociationProbeEvaluation


@dataclass(frozen=True)
class _TUMXYZWindowRun:
    start_index: int
    source_problem: _TUMJointProblem
    data_factors: tuple[JointResidualBlock, ...]
    local_rotation_history: tuple[tuple[QuaternionXYZW, ...], ...]
    results: tuple[JointOptimizerResult, ...]
    data_only_field_observability: JointObservabilityEvaluation
    data_only_shared_observability: JointObservabilityEvaluation

    @property
    def result(self) -> JointOptimizerResult:
        return self.results[-1]


@dataclass(frozen=True)
class _TUMXYZPairAssociation:
    """Reconstructable identity for one fixed query-query association."""

    measurement_id: str
    observation_group: str
    source_frame_index: int
    target_frame_index: int
    source_point_index: int
    target_id: str
    initial_centroid_distance_m: float

    def as_dict(self) -> dict[str, str | int | float]:
        """Return schema-friendly correspondence lineage."""

        return self.__dict__


@dataclass(frozen=True)
class _TUMXYZPairProblem:
    """Fixed query-query correspondences for Zhou--Koltun Eq. (2)."""

    query_frames: tuple[TUMDepthPoseEntry, ...]
    measurements: tuple[JointTrilinearXYZPairMeasurement, ...]
    data_factors: tuple[JointResidualBlock, ...]
    parameter_blocks: tuple[JointParameterBlock, ...]
    pair_counts: tuple[tuple[str, int], ...]
    associations: tuple[_TUMXYZPairAssociation, ...]


@dataclass(frozen=True)
class _TUMXYZPairWindowRun:
    """One public-window two-sided Eq. (2) optimization and diagnostics."""

    start_index: int
    problem: _TUMXYZPairProblem
    local_rotation_history: tuple[tuple[QuaternionXYZW, ...], ...]
    results: tuple[JointOptimizerResult, ...]
    data_only_field_observability: JointObservabilityEvaluation
    data_only_joint_observability: JointObservabilityEvaluation

    @property
    def result(self) -> JointOptimizerResult:
        return self.results[-1]


@dataclass(frozen=True)
class _TUMRematchEvaluation:
    start_index: int
    fixed_holdout_rmse_m: float | None
    rematched_holdout_rmse_m: float | None
    stability: CorrespondenceStabilityEvaluation
    fixed_curvature: NumericalCurvatureEvaluation
    rematched_curvature: NumericalCurvatureEvaluation
    relative_hessian_difference: float
    multistart_searches: tuple[PatternSearchResult, ...]
    multistart_evaluation: MultiStartEvaluation | None


class NativeTUMJointSlacSolver(SolverAdapter):
    """Jointly refine per-frame TUM poses and a shared camera mounting transform."""

    backend = NATIVE_TUM_JOINT_SLAC_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        """Run a disjoint-map/query joint solve on a public TUM RGB-D sequence."""

        del inspection
        if config.dataset.type != "tum_rgbd":
            return _unavailable("native TUM joint SLAC requires dataset.type=tum_rgbd")
        cameras = sorted(name for name, sensor in config.sensors.items() if sensor.type == "camera")
        if not cameras:
            return _unavailable("native TUM joint SLAC requires one camera sensor")
        camera = cameras[0]
        camera_node = frame_graph.nodes.get(camera)
        if camera_node is None or camera_node.parent is None:
            return _unavailable("TUM camera must have a parent mounting frame")
        root = Path(config.dataset.path)
        depth_index = root / "depth.txt"
        groundtruth_path = root / "groundtruth.txt"
        if not depth_index.exists() or not groundtruth_path.exists():
            return _unavailable("TUM depth.txt and groundtruth.txt are required")

        options = _options(config)
        intrinsics = _intrinsics(config, camera)
        paired = associate_depth_groundtruth(
            read_image_index(depth_index),
            read_groundtruth(groundtruth_path),
            max_difference_sec=options.max_pose_time_delta_sec,
        )
        selected = _select_frames(paired, options)
        if selected is None:
            return _unavailable("not enough time-aligned TUM frames for joint selection")
        try:
            problem = _build_problem(root, selected, intrinsics, options)
        except (OSError, ValueError) as exc:
            return _unavailable(f"TUM joint frontend failed: {exc}")
        if len(problem.query_frames) < _MIN_QUERY_FRAMES:
            return _unavailable("too few TUM query frames for grouped holdout")
        if len(problem.measurements) < _MIN_TOTAL_CORRESPONDENCES:
            return _unavailable(
                "too few TUM world-plane correspondences "
                f"({len(problem.measurements)} < {_MIN_TOTAL_CORRESPONDENCES})"
            )

        result = _optimize_problem(config, problem, options)
        primary_run = _TUMWindowRun(options.frame_start_index, problem, result)
        replication_runs = _replicate_windows(
            root, paired, intrinsics, config, options, primary_run
        )
        transfers = _cross_window_transfers(replication_runs)
        spatial_runs = _spatial_ablation_runs(config, replication_runs, options)
        spatial_transfers = _spatial_cross_window_transfers(spatial_runs)
        xyz_runs = _xyz_ablation_runs(config, replication_runs, options)
        xyz_transfers = _xyz_cross_window_transfers(xyz_runs)
        xyz_pair_runs = _xyz_pair_ablation_runs(root, intrinsics, config, replication_runs, options)
        xyz_pair_transfers = _xyz_pair_cross_window_transfers(xyz_pair_runs)
        reassociation_runs = _iterative_reassociation_runs(
            root, intrinsics, config, spatial_runs, options
        )
        rematching = _rematch_spatial_holdouts(root, intrinsics, spatial_runs, options)
        extrinsic = se3_from_tangent(result.optimized_values[_EXTRINSIC_BLOCK])
        variable = f"T_{camera_node.parent}_{camera}"
        extrinsic_observability = _data_only_shared_observability(
            problem, result, include_depth_calibration=False
        )
        shared_observability = _data_only_shared_observability(
            problem, result, include_depth_calibration=True
        )
        metrics = _metrics(
            problem,
            result,
            extrinsic,
            extrinsic_observability,
            shared_observability,
            options,
        )
        metrics.update(_replication_metrics(replication_runs, transfers, options))
        metrics.update(
            _spatial_ablation_metrics(replication_runs, spatial_runs, spatial_transfers, options)
        )
        metrics.update(_xyz_ablation_metrics(spatial_runs, xyz_runs, xyz_transfers, options))
        metrics.update(_xyz_pair_ablation_metrics(xyz_pair_runs, xyz_pair_transfers, options))
        metrics.update(_rematching_metrics(rematching))
        metrics.update(_iterative_reassociation_metrics(reassociation_runs, options))
        warnings = _warnings(
            result,
            extrinsic_observability.information_rank,
            shared_observability.information_rank,
        )
        xyz_shared_rank = min(
            run.data_only_shared_observability.information_rank for run in xyz_runs
        )
        if xyz_shared_rank < 30:
            warnings.append(
                "TUM full-XYZ data-only extrinsic/field rank is "
                f"{xyz_shared_rank} < 30; rigid field modes exchange with extrinsic"
            )
        if xyz_pair_runs:
            xyz_pair_rank = min(
                run.data_only_joint_observability.information_rank for run in xyz_pair_runs
            )
            xyz_pair_dimension = xyz_pair_runs[0].data_only_joint_observability.parameter_dimension
            if xyz_pair_rank < xyz_pair_dimension:
                warnings.append(
                    "TUM two-sided Eq. (2) data-only pose/field rank is "
                    f"{xyz_pair_rank} < {xyz_pair_dimension} after fixing the world gauge"
                )
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status=result.status,
            metrics=metrics,
            transforms={variable: extrinsic},
            provenance=_provenance(
                root=root,
                depth_index=depth_index,
                groundtruth_path=groundtruth_path,
                paired_count=len(paired),
                problem=problem,
                result=result,
                intrinsics=intrinsics,
                options=options,
                variable=variable,
                extrinsic_observability=extrinsic_observability,
                shared_observability=shared_observability,
                replication_runs=replication_runs,
                transfers=transfers,
                spatial_runs=spatial_runs,
                spatial_transfers=spatial_transfers,
                xyz_runs=xyz_runs,
                xyz_transfers=xyz_transfers,
                xyz_pair_runs=xyz_pair_runs,
                xyz_pair_transfers=xyz_pair_transfers,
                reassociation_runs=reassociation_runs,
                rematching=rematching,
            ),
            warnings=warnings,
            observability=ObservabilityResult(
                rank=result.information_rank,
                condition_number=result.condition_number,
                weak_directions=list(result.weak_parameter_blocks),
                grade="warn",
            ),
        )


def _select_frames(
    paired: list[TUMDepthPoseEntry], options: TUMJointSlacOptions
) -> tuple[TUMDepthPoseEntry, ...] | None:
    count = options.map_frame_count + options.query_frame_count
    indexes = [options.frame_start_index + index * options.frame_stride for index in range(count)]
    if not indexes or indexes[-1] >= len(paired):
        return None
    return tuple(paired[index] for index in indexes)


def _optimize_problem(
    config: CalibrationConfig,
    problem: _TUMJointProblem,
    options: TUMJointSlacOptions,
) -> JointOptimizerResult:
    return _optimize_joint(config, problem.parameter_blocks, problem.all_factors, options)


def _optimize_joint(
    config: CalibrationConfig,
    parameter_blocks: tuple[JointParameterBlock, ...],
    factors: tuple[JointResidualBlock, ...],
    options: TUMJointSlacOptions,
) -> JointOptimizerResult:
    return BackendNeutralJointOptimizer().solve(
        parameter_blocks,
        factors,
        _joint_optimizer_options(config, options),
    )


def _joint_optimizer_options(
    config: CalibrationConfig, options: TUMJointSlacOptions
) -> JointOptimizerOptions:
    return JointOptimizerOptions(
        max_iterations=max(1, config.solver.max_iterations),
        convergence_tolerance=config.solver.convergence_tolerance,
        huber_delta=options.huber_delta_m,
        holdout_ratio=options.holdout_ratio,
        split_seed=config.solver.seed or 0,
        minimum_train_factors=12,
        known_bad_margin=options.known_bad_margin_m,
        max_condition_number=1.0e12,
    )


def _replicate_windows(
    root: Path,
    paired: list[TUMDepthPoseEntry],
    intrinsics: TUMDepthIntrinsics,
    config: CalibrationConfig,
    options: TUMJointSlacOptions,
    primary: _TUMWindowRun,
) -> tuple[_TUMWindowRun, ...]:
    runs: list[_TUMWindowRun] = []
    for start_index in options.replication_start_indices:
        if start_index == primary.start_index:
            runs.append(primary)
            continue
        window_options = replace(options, frame_start_index=start_index)
        selected = _select_frames(paired, window_options)
        if selected is None:
            continue
        try:
            problem = _build_problem(root, selected, intrinsics, window_options)
        except (OSError, ValueError):
            continue
        if (
            len(problem.query_frames) < _MIN_QUERY_FRAMES
            or len(problem.measurements) < _MIN_TOTAL_CORRESPONDENCES
        ):
            continue
        runs.append(
            _TUMWindowRun(
                start_index,
                problem,
                _optimize_problem(config, problem, window_options),
            )
        )
    return tuple(runs)


def _cross_window_transfers(
    runs: tuple[_TUMWindowRun, ...],
) -> tuple[dict[str, float | int | None], ...]:
    transfers: list[dict[str, float | int | None]] = []
    for source in runs:
        for target in runs:
            if source.start_index == target.start_index:
                continue
            holdout_ids = set(target.result.holdout_factor_ids)
            factors = [
                factor for factor in target.problem.data_factors if factor.factor_id in holdout_ids
            ]
            values = dict(target.result.optimized_values)
            values[_EXTRINSIC_BLOCK] = source.result.optimized_values[_EXTRINSIC_BLOCK]
            values[_DEPTH_BLOCK] = source.result.optimized_values[_DEPTH_BLOCK]
            transferred_rmse = _factor_rmse(factors, values)
            baseline_rmse = target.result.holdout_rmse
            delta = (
                transferred_rmse - baseline_rmse
                if transferred_rmse is not None and baseline_rmse is not None
                else None
            )
            transfers.append(
                {
                    "source_start_index": source.start_index,
                    "target_start_index": target.start_index,
                    "target_baseline_holdout_rmse_m": baseline_rmse,
                    "transferred_holdout_rmse_m": transferred_rmse,
                    "delta_rmse_m": delta,
                }
            )
    return tuple(transfers)


def _spatial_ablation_runs(
    config: CalibrationConfig,
    baseline_runs: tuple[_TUMWindowRun, ...],
    options: TUMJointSlacOptions,
) -> tuple[_TUMSpatialWindowRun, ...]:
    runs: list[_TUMSpatialWindowRun] = []
    for baseline in baseline_runs:
        factors = _spatial_data_factors(baseline.problem)
        regularizers = _spatial_train_only_factors(baseline.problem, options)
        blocks = _spatial_parameter_blocks(baseline.problem, options)
        runs.append(
            _TUMSpatialWindowRun(
                baseline.start_index,
                baseline.problem,
                factors,
                _optimize_joint(
                    config,
                    blocks,
                    (*factors, *regularizers),
                    options,
                ),
            )
        )
    return tuple(runs)


def _xyz_ablation_runs(
    config: CalibrationConfig,
    baseline_runs: tuple[_TUMWindowRun, ...],
    options: TUMJointSlacOptions,
) -> tuple[_TUMXYZWindowRun, ...]:
    controls = regular_lattice_control_points(
        minimum=_SPATIAL_LATTICE_MINIMUM_M,
        maximum=_SPATIAL_LATTICE_MAXIMUM_M,
        shape=_SPATIAL_LATTICE_SHAPE,
    )
    identity_rotations: tuple[QuaternionXYZW, ...] = ((0.0, 0.0, 0.0, 1.0),) * len(controls)
    runs: list[_TUMXYZWindowRun] = []
    for baseline in baseline_runs:
        problem = baseline.problem
        data_factors = _xyz_data_factors(problem)
        blocks = _xyz_parameter_blocks(problem, options)
        rotations = identity_rotations
        rotation_history: list[tuple[QuaternionXYZW, ...]] = []
        results: list[JointOptimizerResult] = []
        current_blocks = blocks
        for _round in range(options.xyz_lattice_local_rotation_updates + 1):
            rotation_history.append(rotations)
            result = _optimize_joint(
                config,
                current_blocks,
                (
                    *data_factors,
                    *_xyz_train_only_factors(problem, controls, rotations, options),
                ),
                options,
            )
            results.append(result)
            if result.status != "converged":
                break
            rotations = estimate_xyz_lattice_local_rotations(
                control_points_m=controls,
                offsets_m=result.optimized_values[_XYZ_LATTICE_BLOCK],
                shape=_SPATIAL_LATTICE_SHAPE,
            )
            current_blocks = tuple(
                replace(block, initial_values=result.optimized_values[block.name])
                for block in blocks
            )
        final = results[-1]
        runs.append(
            _TUMXYZWindowRun(
                baseline.start_index,
                problem,
                data_factors,
                tuple(rotation_history),
                tuple(results),
                _xyz_data_only_observability(blocks, data_factors, final, include_extrinsic=False),
                _xyz_data_only_observability(blocks, data_factors, final, include_extrinsic=True),
            )
        )
    return tuple(runs)


def _xyz_pair_cross_window_transfers(
    runs: tuple[_TUMXYZPairWindowRun, ...],
) -> tuple[dict[str, float | int | None], ...]:
    transfers: list[dict[str, float | int | None]] = []
    for source in runs:
        for target in runs:
            if source.start_index == target.start_index:
                continue
            holdout_groups = set(target.result.holdout_observation_groups)
            factors = [
                factor
                for factor in target.problem.data_factors
                if factor.observation_group in holdout_groups
            ]
            values = dict(target.result.optimized_values)
            values[_XYZ_LATTICE_BLOCK] = source.result.optimized_values[_XYZ_LATTICE_BLOCK]
            transferred_rmse = _factor_rmse(factors, values)
            baseline_rmse = target.result.holdout_rmse
            transfers.append(
                {
                    "source_start_index": source.start_index,
                    "target_start_index": target.start_index,
                    "target_baseline_holdout_rmse_m": baseline_rmse,
                    "transferred_holdout_rmse_m": transferred_rmse,
                    "delta_rmse_m": (
                        transferred_rmse - baseline_rmse
                        if transferred_rmse is not None and baseline_rmse is not None
                        else None
                    ),
                }
            )
    return tuple(transfers)


def _xyz_pair_ablation_runs(
    root: Path,
    intrinsics: TUMDepthIntrinsics,
    config: CalibrationConfig,
    baseline_runs: tuple[_TUMWindowRun, ...],
    options: TUMJointSlacOptions,
) -> tuple[_TUMXYZPairWindowRun, ...]:
    """Run the paper's two-sided Eq. (2) on fixed query-query pairs."""

    controls = regular_lattice_control_points(
        minimum=_SPATIAL_LATTICE_MINIMUM_M,
        maximum=_SPATIAL_LATTICE_MAXIMUM_M,
        shape=_SPATIAL_LATTICE_SHAPE,
    )
    identity_rotations: tuple[QuaternionXYZW, ...] = ((0.0, 0.0, 0.0, 1.0),) * len(controls)
    runs: list[_TUMXYZPairWindowRun] = []
    for baseline in baseline_runs:
        pair_problem = _build_xyz_pair_problem(
            root, baseline.problem.query_frames, intrinsics, options
        )
        if len(pair_problem.data_factors) < _MIN_TOTAL_CORRESPONDENCES:
            continue
        rotations = identity_rotations
        current_blocks = pair_problem.parameter_blocks
        rotation_history: list[tuple[QuaternionXYZW, ...]] = []
        results: list[JointOptimizerResult] = []
        for _round in range(options.xyz_lattice_local_rotation_updates + 1):
            rotation_history.append(rotations)
            result = _optimize_joint(
                config,
                current_blocks,
                (
                    *pair_problem.data_factors,
                    *_xyz_pair_train_only_factors(pair_problem, controls, rotations, options),
                ),
                options,
            )
            results.append(result)
            if result.status != "converged":
                break
            rotations = estimate_xyz_lattice_local_rotations(
                control_points_m=controls,
                offsets_m=result.optimized_values[_XYZ_LATTICE_BLOCK],
                shape=_SPATIAL_LATTICE_SHAPE,
            )
            current_blocks = tuple(
                replace(block, initial_values=result.optimized_values[block.name])
                for block in pair_problem.parameter_blocks
            )
        final = results[-1]
        runs.append(
            _TUMXYZPairWindowRun(
                baseline.start_index,
                pair_problem,
                tuple(rotation_history),
                tuple(results),
                _xyz_pair_data_only_observability(pair_problem, final, include_poses=False),
                _xyz_pair_data_only_observability(pair_problem, final, include_poses=True),
            )
        )
    return tuple(runs)


def _build_xyz_pair_problem(
    root: Path,
    query_frames: tuple[TUMDepthPoseEntry, ...],
    intrinsics: TUMDepthIntrinsics,
    options: TUMJointSlacOptions,
) -> _TUMXYZPairProblem:
    sampled_points = tuple(
        tuple(
            sample_tum_depth_points(
                read_tum_depth_png(root / frame.depth.path),
                intrinsics,
                max_points=options.query_points_per_frame,
            )
        )
        for frame in query_frames
    )
    measurements: list[JointTrilinearXYZPairMeasurement] = []
    factors: list[JointResidualBlock] = []
    pair_counts: list[tuple[str, int]] = []
    associations: list[_TUMXYZPairAssociation] = []
    for source_index in range(len(query_frames) - options.xyz_pair_stride):
        target_index = source_index + options.xyz_pair_stride
        source_frame = query_frames[source_index]
        target_frame = query_frames[target_index]
        target_records = [
            LivoxPointRecord((*point, 0.0), None) for point in sampled_points[target_index]
        ]
        target_plane_map = build_voxel_plane_map(target_records, options.voxel_size_m)
        transform_target_source = target_frame.transform_world_camera.inverse().compose(
            source_frame.transform_world_camera
        )
        group = f"tum-xyz-pair-{source_index:03d}-{target_index:03d}"
        candidates: list[tuple[JointTrilinearXYZPairMeasurement, _TUMXYZPairAssociation]] = []
        for point_index, source_point in enumerate(sampled_points[source_index]):
            source_in_target = transform_target_source.transform_point(source_point)
            match = nearest_voxel_plane_match(
                source_in_target,
                target_plane_map,
                voxel_size_m=options.voxel_size_m,
                correspondence_gate_m=options.correspondence_gate_m,
            )
            if match is None:
                continue
            normal_world = rotate_vector_xyzw(
                target_frame.transform_world_camera.rotation_quat_xyzw,
                match.normal,
            )
            measurement_id = f"tum-xyz-pair-{source_index:03d}-{target_index:03d}-{point_index:05d}"
            candidates.append(
                (
                    JointTrilinearXYZPairMeasurement(
                        measurement_id=measurement_id,
                        observation_group=group,
                        source_point_sensor_m=source_point,
                        target_point_sensor_m=match.centroid,
                        source_lattice_weights=trilinear_lattice_weights(
                            source_point,
                            minimum=_SPATIAL_LATTICE_MINIMUM_M,
                            maximum=_SPATIAL_LATTICE_MAXIMUM_M,
                            shape=_SPATIAL_LATTICE_SHAPE,
                        ),
                        target_lattice_weights=trilinear_lattice_weights(
                            match.centroid,
                            minimum=_SPATIAL_LATTICE_MINIMUM_M,
                            maximum=_SPATIAL_LATTICE_MAXIMUM_M,
                            shape=_SPATIAL_LATTICE_SHAPE,
                        ),
                        plane_normal_world=normal_world,
                        transform_world_source_initial=source_frame.transform_world_camera,
                        transform_world_target_initial=target_frame.transform_world_camera,
                    ),
                    _TUMXYZPairAssociation(
                        measurement_id,
                        group,
                        source_index,
                        target_index,
                        point_index,
                        match.target_id,
                        match.centroid_distance_m,
                    ),
                )
            )
        retained_pairs = _bounded_pair_sample(
            candidates, options.xyz_pair_max_correspondences_per_pair
        )
        retained = [measurement for measurement, _association in retained_pairs]
        associations.extend(association for _measurement, association in retained_pairs)
        pair_counts.append((group, len(retained)))
        measurements.extend(retained)
        factors.extend(
            make_joint_trilinear_xyz_pair_factor(
                measurement,
                source_pose_block=_pose_block(source_index, source_frame),
                target_pose_block=_pose_block(target_index, target_frame),
                xyz_lattice_block=_XYZ_LATTICE_BLOCK,
            )
            for measurement in retained
        )
    pose_known_bad = (options.known_bad_translation_m,) * 3 + (options.known_bad_rotation_rad,) * 3
    blocks: list[JointParameterBlock] = [
        JointParameterBlock(
            _pose_block(index, frame),
            (0.0,) * 6,
            fixed=index == 0,
            finite_difference_steps=(1.0e-4,) * 3 + (1.0e-5,) * 3,
            known_bad_steps=pose_known_bad,
        )
        for index, frame in enumerate(query_frames)
    ]
    dimension = 3 * math.prod(_SPATIAL_LATTICE_SHAPE)
    blocks.append(
        JointParameterBlock(
            _XYZ_LATTICE_BLOCK,
            (0.0,) * dimension,
            finite_difference_steps=(1.0e-5,) * dimension,
            known_bad_steps=(options.known_bad_depth_bias_m,) * dimension,
        )
    )
    return _TUMXYZPairProblem(
        query_frames,
        tuple(measurements),
        tuple(factors),
        tuple(blocks),
        tuple(pair_counts),
        tuple(associations),
    )


def _bounded_pair_sample(
    measurements: list[tuple[JointTrilinearXYZPairMeasurement, _TUMXYZPairAssociation]],
    maximum: int,
) -> list[tuple[JointTrilinearXYZPairMeasurement, _TUMXYZPairAssociation]]:
    if len(measurements) <= maximum:
        return measurements
    step = math.ceil(len(measurements) / maximum)
    return measurements[::step][:maximum]


def _xyz_pair_train_only_factors(
    problem: _TUMXYZPairProblem,
    controls: tuple[tuple[float, float, float], ...],
    rotations: tuple[QuaternionXYZW, ...],
    options: TUMJointSlacOptions,
) -> tuple[JointResidualBlock, ...]:
    pose_sigma = (options.pose_prior_translation_sigma_m,) * 3 + (
        options.pose_prior_rotation_sigma_rad,
    ) * 3
    priors = tuple(
        make_joint_prior_factor(
            factor_id=f"xyz-pair-pose-prior-{index:03d}",
            observation_group="tum-xyz-pair-regularization",
            block=_pose_block(index, frame),
            target=(0.0,) * 6,
            sigma=pose_sigma,
        )
        for index, frame in enumerate(problem.query_frames)
        if index > 0
    )
    return (
        *priors,
        make_joint_xyz_lattice_shape_factor(
            factor_id="xyz-pair-lattice-shape",
            observation_group="tum-xyz-pair-regularization",
            block=_XYZ_LATTICE_BLOCK,
            control_points_m=controls,
            local_rotations_xyzw=rotations,
            shape=_SPATIAL_LATTICE_SHAPE,
            sigma_m=options.xyz_lattice_shape_sigma_m,
        ),
        make_joint_xyz_lattice_rigid_gauge_factor(
            factor_id="xyz-pair-lattice-rigid-gauge",
            observation_group="tum-xyz-pair-regularization",
            block=_XYZ_LATTICE_BLOCK,
            control_points_m=controls,
            translation_sigma_m=options.xyz_lattice_translation_gauge_sigma_m,
            rotation_sigma_rad=options.xyz_lattice_rotation_gauge_sigma_rad,
        ),
    )


def _xyz_pair_data_only_observability(
    problem: _TUMXYZPairProblem,
    result: JointOptimizerResult,
    *,
    include_poses: bool,
) -> JointObservabilityEvaluation:
    blocks = tuple(
        replace(
            block,
            initial_values=result.optimized_values[block.name],
            fixed=(block.name != _XYZ_LATTICE_BLOCK and (not include_poses or block.fixed)),
        )
        for block in problem.parameter_blocks
    )
    train_groups = set(result.train_observation_groups)
    train_factors = tuple(
        factor for factor in problem.data_factors if factor.observation_group in train_groups
    )
    return evaluate_joint_observability(blocks, train_factors, result.optimized_values)


def _xyz_data_factors(problem: _TUMJointProblem) -> tuple[JointResidualBlock, ...]:
    return tuple(_xyz_factor(problem, measurement) for measurement in problem.measurements)


def _xyz_factor(
    problem: _TUMJointProblem, measurement: JointDepthPointToPlaneMeasurement
) -> JointResidualBlock:
    frame_index = int(measurement.observation_group.rsplit("-", 1)[1])
    frame = problem.query_frames[frame_index]
    point = _scale_ray(measurement.normalized_ray_sensor, measurement.nominal_depth_m)
    return make_joint_trilinear_xyz_point_to_plane_factor(
        JointTrilinearXYZPointToPlaneMeasurement(
            measurement.measurement_id,
            measurement.observation_group,
            point,
            trilinear_lattice_weights(
                point,
                minimum=_SPATIAL_LATTICE_MINIMUM_M,
                maximum=_SPATIAL_LATTICE_MAXIMUM_M,
                shape=_SPATIAL_LATTICE_SHAPE,
            ),
            measurement.plane_point_world_m,
            measurement.plane_normal_world,
            measurement.transform_world_body_initial,
            measurement.transform_body_sensor_initial,
            measurement.weight,
        ),
        pose_block=_pose_block(frame_index, frame),
        extrinsic_block=_EXTRINSIC_BLOCK,
        xyz_lattice_block=_XYZ_LATTICE_BLOCK,
    )


def _xyz_parameter_blocks(
    problem: _TUMJointProblem, options: TUMJointSlacOptions
) -> tuple[JointParameterBlock, ...]:
    dimension = 3 * math.prod(_SPATIAL_LATTICE_SHAPE)
    return (
        *(block for block in problem.parameter_blocks if block.name != _DEPTH_BLOCK),
        JointParameterBlock(
            _XYZ_LATTICE_BLOCK,
            (0.0,) * dimension,
            finite_difference_steps=(1.0e-5,) * dimension,
            known_bad_steps=(options.known_bad_depth_bias_m,) * dimension,
        ),
    )


def _xyz_train_only_factors(
    problem: _TUMJointProblem,
    controls: tuple[tuple[float, float, float], ...],
    rotations: tuple[QuaternionXYZW, ...],
    options: TUMJointSlacOptions,
) -> tuple[JointResidualBlock, ...]:
    priors = tuple(factor for factor in problem.all_factors if factor.family == "diagonal_prior")
    shape = make_joint_xyz_lattice_shape_factor(
        factor_id="depth-xyz-lattice-shape",
        observation_group="depth-xyz-lattice-regularization",
        block=_XYZ_LATTICE_BLOCK,
        control_points_m=controls,
        local_rotations_xyzw=rotations,
        shape=_SPATIAL_LATTICE_SHAPE,
        sigma_m=options.xyz_lattice_shape_sigma_m,
    )
    gauge = make_joint_xyz_lattice_rigid_gauge_factor(
        factor_id="depth-xyz-lattice-rigid-gauge",
        observation_group="depth-xyz-lattice-regularization",
        block=_XYZ_LATTICE_BLOCK,
        control_points_m=controls,
        translation_sigma_m=options.xyz_lattice_translation_gauge_sigma_m,
        rotation_sigma_rad=options.xyz_lattice_rotation_gauge_sigma_rad,
    )
    return (*priors, shape, gauge)


def _xyz_data_only_observability(
    blocks: tuple[JointParameterBlock, ...],
    data_factors: tuple[JointResidualBlock, ...],
    result: JointOptimizerResult,
    *,
    include_extrinsic: bool,
) -> JointObservabilityEvaluation:
    free = {_XYZ_LATTICE_BLOCK}
    if include_extrinsic:
        free.add(_EXTRINSIC_BLOCK)
    diagnostic_blocks = tuple(
        replace(
            block,
            initial_values=result.optimized_values[block.name],
            fixed=block.name not in free,
        )
        for block in blocks
    )
    train_groups = set(result.train_observation_groups)
    train_data = tuple(
        factor for factor in data_factors if factor.observation_group in train_groups
    )
    return evaluate_joint_observability(
        diagnostic_blocks,
        train_data,
        result.optimized_values,
    )


def _spatial_data_factors(
    problem: _TUMJointProblem,
) -> tuple[JointResidualBlock, ...]:
    return tuple(_spatial_factor(problem, measurement) for measurement in problem.measurements)


def _spatial_factor(
    problem: _TUMJointProblem,
    measurement: JointDepthPointToPlaneMeasurement,
    match: VoxelPlaneMatch | None = None,
) -> JointResidualBlock:
    frame_index = int(measurement.observation_group.rsplit("-", 1)[1])
    frame = problem.query_frames[frame_index]
    return make_joint_centered_trilinear_depth_point_to_plane_factor(
        JointTrilinearDepthPointToPlaneMeasurement(
            measurement.measurement_id,
            measurement.observation_group,
            measurement.normalized_ray_sensor,
            measurement.nominal_depth_m,
            trilinear_lattice_weights(
                _scale_ray(
                    measurement.normalized_ray_sensor,
                    measurement.nominal_depth_m,
                ),
                minimum=_SPATIAL_LATTICE_MINIMUM_M,
                maximum=_SPATIAL_LATTICE_MAXIMUM_M,
                shape=_SPATIAL_LATTICE_SHAPE,
            ),
            match.centroid if match is not None else measurement.plane_point_world_m,
            match.normal if match is not None else measurement.plane_normal_world,
            measurement.transform_world_body_initial,
            measurement.transform_body_sensor_initial,
            measurement.weight,
        ),
        pose_block=_pose_block(frame_index, frame),
        extrinsic_block=_EXTRINSIC_BLOCK,
        depth_bias_block=_SPATIAL_BIAS_BLOCK,
        centered_lattice_block=_SPATIAL_DEPTH_BLOCK,
    )


def _spatial_parameter_blocks(
    problem: _TUMJointProblem, options: TUMJointSlacOptions
) -> tuple[JointParameterBlock, ...]:
    lattice_size = math.prod(_SPATIAL_LATTICE_SHAPE)
    return (
        *(block for block in problem.parameter_blocks if block.name != _DEPTH_BLOCK),
        JointParameterBlock(
            _SPATIAL_BIAS_BLOCK,
            (0.0,),
            finite_difference_steps=(1.0e-5,),
            known_bad_steps=(options.known_bad_depth_bias_m,),
        ),
        JointParameterBlock(
            _SPATIAL_DEPTH_BLOCK,
            (0.0,) * lattice_size,
            finite_difference_steps=(1.0e-5,) * lattice_size,
            known_bad_steps=(options.known_bad_depth_bias_m,) * lattice_size,
        ),
    )


def _spatial_train_only_factors(
    problem: _TUMJointProblem, options: TUMJointSlacOptions
) -> tuple[JointResidualBlock, ...]:
    priors = tuple(factor for factor in problem.all_factors if factor.family == "diagonal_prior")
    smoothness = make_joint_lattice_smoothness_factor(
        factor_id="depth-lattice-smoothness",
        observation_group="depth-lattice-regularization",
        block=_SPATIAL_DEPTH_BLOCK,
        shape=_SPATIAL_LATTICE_SHAPE,
        sigma_m=options.spatial_lattice_smoothness_sigma_m,
    )
    zero_mean = make_joint_lattice_zero_mean_factor(
        factor_id="depth-lattice-zero-mean",
        observation_group="depth-lattice-regularization",
        block=_SPATIAL_DEPTH_BLOCK,
        size=math.prod(_SPATIAL_LATTICE_SHAPE),
        sigma_m=options.spatial_lattice_zero_mean_sigma_m,
    )
    return (*priors, smoothness, zero_mean)


def _spatial_cross_window_transfers(
    runs: tuple[_TUMSpatialWindowRun, ...],
) -> tuple[dict[str, float | int | None], ...]:
    transfers: list[dict[str, float | int | None]] = []
    for source in runs:
        for target in runs:
            if source.start_index == target.start_index:
                continue
            holdout_ids = set(target.result.holdout_factor_ids)
            factors = [factor for factor in target.data_factors if factor.factor_id in holdout_ids]
            values = dict(target.result.optimized_values)
            values[_EXTRINSIC_BLOCK] = source.result.optimized_values[_EXTRINSIC_BLOCK]
            values[_SPATIAL_BIAS_BLOCK] = source.result.optimized_values[_SPATIAL_BIAS_BLOCK]
            values[_SPATIAL_DEPTH_BLOCK] = source.result.optimized_values[_SPATIAL_DEPTH_BLOCK]
            transferred_rmse = _factor_rmse(factors, values)
            baseline_rmse = target.result.holdout_rmse
            transfers.append(
                {
                    "source_start_index": source.start_index,
                    "target_start_index": target.start_index,
                    "target_spatial_holdout_rmse_m": baseline_rmse,
                    "transferred_spatial_holdout_rmse_m": transferred_rmse,
                    "delta_rmse_m": (
                        transferred_rmse - baseline_rmse
                        if transferred_rmse is not None and baseline_rmse is not None
                        else None
                    ),
                }
            )
    return tuple(transfers)


def _xyz_cross_window_transfers(
    runs: tuple[_TUMXYZWindowRun, ...],
) -> tuple[dict[str, float | int | None], ...]:
    transfers: list[dict[str, float | int | None]] = []
    for source in runs:
        for target in runs:
            if source.start_index == target.start_index:
                continue
            holdout_ids = set(target.result.holdout_factor_ids)
            factors = [factor for factor in target.data_factors if factor.factor_id in holdout_ids]
            values = dict(target.result.optimized_values)
            values[_EXTRINSIC_BLOCK] = source.result.optimized_values[_EXTRINSIC_BLOCK]
            values[_XYZ_LATTICE_BLOCK] = source.result.optimized_values[_XYZ_LATTICE_BLOCK]
            transferred_rmse = _factor_rmse(factors, values)
            baseline_rmse = target.result.holdout_rmse
            transfers.append(
                {
                    "source_start_index": source.start_index,
                    "target_start_index": target.start_index,
                    "target_xyz_holdout_rmse_m": baseline_rmse,
                    "transferred_xyz_holdout_rmse_m": transferred_rmse,
                    "delta_rmse_m": (
                        transferred_rmse - baseline_rmse
                        if transferred_rmse is not None and baseline_rmse is not None
                        else None
                    ),
                }
            )
    return tuple(transfers)


def _iterative_reassociation_runs(
    root: Path,
    intrinsics: TUMDepthIntrinsics,
    config: CalibrationConfig,
    runs: tuple[_TUMSpatialWindowRun, ...],
    options: TUMJointSlacOptions,
) -> tuple[_TUMIterativeReassociationRun, ...]:
    output: list[_TUMIterativeReassociationRun] = []
    for run in runs:
        plane_map = _build_map_planes(root, run.source_problem.map_frames, intrinsics, options)
        initial_state = _initial_spatial_reassociation_state(run, plane_map, options)
        blocks = _spatial_parameter_blocks(run.source_problem, options)
        reassociate = partial(_reassociated_spatial_state, run, plane_map, options)
        result = BackendNeutralJointReassociation().refine(
            blocks,
            initial_state,
            _spatial_train_only_factors(run.source_problem, options),
            run.result,
            reassociate,
            _joint_optimizer_options(config, options),
            JointReassociationOptions(
                max_outer_iterations=options.reassociation_max_outer_iterations,
                minimum_train_pair_jaccard=(options.reassociation_min_train_pair_jaccard),
                minimum_train_retained_fraction=(options.reassociation_min_train_retained_fraction),
            ),
        )
        probe_evaluation = evaluate_joint_reassociation_probes(
            blocks,
            initial_state,
            result,
            reassociate,
            JointReassociationProbeOptions(
                unmatched_residual_penalty=(options.reassociation_unmatched_residual_penalty_m),
                known_bad_margin=options.known_bad_margin_m,
                minimum_retained_fraction=(options.reassociation_min_train_retained_fraction),
            ),
        )
        output.append(
            _TUMIterativeReassociationRun(
                run.start_index,
                run,
                result,
                probe_evaluation,
            )
        )
    return tuple(output)


def _initial_spatial_reassociation_state(
    run: _TUMSpatialWindowRun,
    plane_map: dict[tuple[int, int, int], Any],
    options: TUMJointSlacOptions,
) -> JointReassociationState:
    assignments: list[CorrespondenceAssignment] = []
    for measurement in run.source_problem.measurements:
        match = nearest_voxel_plane_match(
            measurement.plane_point_world_m,
            plane_map,
            voxel_size_m=options.voxel_size_m,
            correspondence_gate_m=options.correspondence_gate_m,
        )
        if match is None:
            raise ValueError(
                f"initial TUM target identity is unavailable for {measurement.measurement_id}"
            )
        assignments.append(CorrespondenceAssignment(measurement.measurement_id, match.target_id))
    return JointReassociationState(run.data_factors, tuple(assignments))


def _reassociated_spatial_state(
    run: _TUMSpatialWindowRun,
    plane_map: dict[tuple[int, int, int], Any],
    options: TUMJointSlacOptions,
    values: ParameterValues,
) -> JointReassociationState:
    factors: list[JointResidualBlock] = []
    assignments: list[CorrespondenceAssignment] = []
    for measurement in run.source_problem.measurements:
        point_world = _spatial_measurement_world_point_from_values(
            run.source_problem, measurement, values
        )
        match = nearest_voxel_plane_match(
            point_world,
            plane_map,
            voxel_size_m=options.voxel_size_m,
            correspondence_gate_m=options.correspondence_gate_m,
        )
        if match is None:
            continue
        factors.append(_spatial_factor(run.source_problem, measurement, match))
        assignments.append(CorrespondenceAssignment(measurement.measurement_id, match.target_id))
    return JointReassociationState(tuple(factors), tuple(assignments))


def _scale_ray(ray: tuple[float, float, float], depth_m: float) -> tuple[float, float, float]:
    return (ray[0] * depth_m, ray[1] * depth_m, ray[2] * depth_m)


def _rematch_spatial_holdouts(
    root: Path,
    intrinsics: TUMDepthIntrinsics,
    runs: tuple[_TUMSpatialWindowRun, ...],
    options: TUMJointSlacOptions,
) -> tuple[_TUMRematchEvaluation, ...]:
    evaluations: list[_TUMRematchEvaluation] = []
    for run in runs:
        plane_map = _build_map_planes(root, run.source_problem.map_frames, intrinsics, options)
        holdout_ids = set(run.result.holdout_factor_ids)
        reference: list[CorrespondenceAssignment] = []
        rematched: list[CorrespondenceAssignment] = []
        fixed_residuals: list[float] = []
        rematched_residuals: list[float] = []
        values = run.result.optimized_values
        holdout_measurements: list[JointDepthPointToPlaneMeasurement] = []
        for measurement in run.source_problem.measurements:
            if measurement.measurement_id not in holdout_ids:
                continue
            holdout_measurements.append(measurement)
            original = nearest_voxel_plane_match(
                measurement.plane_point_world_m,
                plane_map,
                voxel_size_m=options.voxel_size_m,
                correspondence_gate_m=options.correspondence_gate_m,
            )
            if original is not None:
                reference.append(
                    CorrespondenceAssignment(measurement.measurement_id, original.target_id)
                )
            frame_index = int(measurement.observation_group.rsplit("-", 1)[1])
            frame = run.source_problem.query_frames[frame_index]
            weights = trilinear_lattice_weights(
                _scale_ray(
                    measurement.normalized_ray_sensor,
                    measurement.nominal_depth_m,
                ),
                minimum=_SPATIAL_LATTICE_MINIMUM_M,
                maximum=_SPATIAL_LATTICE_MAXIMUM_M,
                shape=_SPATIAL_LATTICE_SHAPE,
            )
            depth_m = (
                measurement.nominal_depth_m
                + values[_SPATIAL_BIAS_BLOCK][0]
                + sum(
                    weight * offset
                    for weight, offset in zip(weights, values[_SPATIAL_DEPTH_BLOCK], strict=True)
                )
            )
            point_sensor = _scale_ray(measurement.normalized_ray_sensor, depth_m)
            world_sensor = (
                se3_from_tangent(values[_pose_block(frame_index, frame)])
                .compose(frame.transform_world_camera)
                .compose(se3_from_tangent(values[_EXTRINSIC_BLOCK]))
            )
            point_world = world_sensor.transform_point(point_sensor)
            fixed_residuals.append(
                _point_plane_residual(
                    point_world,
                    measurement.plane_point_world_m,
                    measurement.plane_normal_world,
                )
            )
            match = nearest_voxel_plane_match(
                point_world,
                plane_map,
                voxel_size_m=options.voxel_size_m,
                correspondence_gate_m=options.correspondence_gate_m,
            )
            if match is None:
                continue
            rematched.append(CorrespondenceAssignment(measurement.measurement_id, match.target_id))
            rematched_residuals.append(
                _point_plane_residual(point_world, match.centroid, match.normal)
            )
        curvature_center = (
            *values[_EXTRINSIC_BLOCK],
            values[_SPATIAL_BIAS_BLOCK][0],
        )
        curvature_steps = (0.002,) * 3 + (math.radians(0.1),) * 3 + (0.002,)
        fixed_curvature = evaluate_numerical_curvature(
            partial(
                _shared_holdout_objective,
                run,
                holdout_measurements,
                plane_map,
                options,
                rematch=False,
            ),
            curvature_center,
            curvature_steps,
        )
        rematched_curvature = evaluate_numerical_curvature(
            partial(
                _shared_holdout_objective,
                run,
                holdout_measurements,
                plane_map,
                options,
                rematch=True,
            ),
            curvature_center,
            curvature_steps,
        )
        multistart_searches, multistart_evaluation = (
            _run_primary_multistart(
                run,
                holdout_measurements,
                plane_map,
                options,
                curvature_center,
            )
            if run.start_index == options.frame_start_index
            else ((), None)
        )
        evaluations.append(
            _TUMRematchEvaluation(
                run.start_index,
                _rmse_values(fixed_residuals),
                _rmse_values(rematched_residuals),
                evaluate_correspondence_stability(tuple(reference), tuple(rematched)),
                fixed_curvature,
                rematched_curvature,
                _relative_hessian_difference(fixed_curvature, rematched_curvature),
                multistart_searches,
                multistart_evaluation,
            )
        )
    return tuple(evaluations)


def _run_primary_multistart(
    run: _TUMSpatialWindowRun,
    measurements: list[JointDepthPointToPlaneMeasurement],
    plane_map: dict[tuple[int, int, int], Any],
    options: TUMJointSlacOptions,
    center: tuple[float, ...],
) -> tuple[tuple[PatternSearchResult, ...], MultiStartEvaluation]:
    objective = partial(
        _shared_holdout_objective,
        run,
        measurements,
        plane_map,
        options,
        rematch=True,
    )
    starts = [
        ("optimized-center", center),
        ("translation-z-positive", _shift_parameter(center, 2, 0.03)),
        ("translation-z-negative", _shift_parameter(center, 2, -0.03)),
        ("depth-bias-positive", _shift_parameter(center, 6, 0.02)),
        ("depth-bias-negative", _shift_parameter(center, 6, -0.02)),
    ]
    initial_steps = (0.01,) * 3 + (math.radians(0.1),) * 3 + (0.01,)
    searches = tuple(
        coordinate_pattern_search(
            objective,
            start_id=start_id,
            start=start,
            options=PatternSearchOptions(
                initial_steps=initial_steps,
                minimum_steps=tuple(step * 0.5 for step in initial_steps),
                max_sweeps=10,
                improvement_tolerance=1.0e-8,
            ),
        )
        for start_id, start in starts
    )
    evaluation = evaluate_multistart_solutions(
        tuple(search.solution for search in searches),
        parameter_scales=(0.03,) * 3 + (math.radians(0.5),) * 3 + (0.02,),
        cluster_radius=1.0,
        objective_absolute_tolerance=1.0e-6,
        objective_relative_tolerance=0.01,
    )
    return searches, evaluation


def _shift_parameter(values: tuple[float, ...], index: int, amount: float) -> tuple[float, ...]:
    shifted = list(values)
    shifted[index] += amount
    return tuple(shifted)


def _shared_holdout_objective(
    run: _TUMSpatialWindowRun,
    measurements: list[JointDepthPointToPlaneMeasurement],
    plane_map: dict[tuple[int, int, int], Any],
    options: TUMJointSlacOptions,
    candidate: tuple[float, ...],
    *,
    rematch: bool,
) -> float:
    extrinsic = candidate[:6]
    bias_m = candidate[6]
    squared_sum = 0.0
    for measurement in measurements:
        point_world = _spatial_measurement_world_point(run, measurement, extrinsic, bias_m)
        if rematch:
            match = nearest_voxel_plane_match(
                point_world,
                plane_map,
                voxel_size_m=options.voxel_size_m,
                correspondence_gate_m=options.correspondence_gate_m,
            )
            residual = (
                _point_plane_residual(point_world, match.centroid, match.normal)
                if match is not None
                else options.correspondence_gate_m
            )
        else:
            residual = _point_plane_residual(
                point_world,
                measurement.plane_point_world_m,
                measurement.plane_normal_world,
            )
        squared_sum += residual * residual
    return squared_sum / len(measurements) if measurements else 0.0


def _spatial_measurement_world_point(
    run: _TUMSpatialWindowRun,
    measurement: JointDepthPointToPlaneMeasurement,
    extrinsic: tuple[float, ...],
    bias_m: float,
) -> tuple[float, float, float]:
    values = dict(run.result.optimized_values)
    values[_EXTRINSIC_BLOCK] = extrinsic
    values[_SPATIAL_BIAS_BLOCK] = (bias_m,)
    return _spatial_measurement_world_point_from_values(run.source_problem, measurement, values)


def _spatial_measurement_world_point_from_values(
    problem: _TUMJointProblem,
    measurement: JointDepthPointToPlaneMeasurement,
    values: ParameterValues,
) -> tuple[float, float, float]:
    frame_index = int(measurement.observation_group.rsplit("-", 1)[1])
    frame = problem.query_frames[frame_index]
    weights = trilinear_lattice_weights(
        _scale_ray(measurement.normalized_ray_sensor, measurement.nominal_depth_m),
        minimum=_SPATIAL_LATTICE_MINIMUM_M,
        maximum=_SPATIAL_LATTICE_MAXIMUM_M,
        shape=_SPATIAL_LATTICE_SHAPE,
    )
    depth_m = (
        measurement.nominal_depth_m
        + values[_SPATIAL_BIAS_BLOCK][0]
        + sum(
            weight * offset
            for weight, offset in zip(weights, values[_SPATIAL_DEPTH_BLOCK], strict=True)
        )
    )
    world_sensor = (
        se3_from_tangent(values[_pose_block(frame_index, frame)])
        .compose(frame.transform_world_camera)
        .compose(se3_from_tangent(values[_EXTRINSIC_BLOCK]))
    )
    return world_sensor.transform_point(_scale_ray(measurement.normalized_ray_sensor, depth_m))


def _relative_hessian_difference(
    fixed: NumericalCurvatureEvaluation,
    rematched: NumericalCurvatureEvaluation,
) -> float:
    difference = math.sqrt(
        sum(
            (rematched.hessian[row][column] - fixed.hessian[row][column]) ** 2
            for row in range(fixed.parameter_dimension)
            for column in range(fixed.parameter_dimension)
        )
    )
    fixed_norm = math.sqrt(sum(value * value for row in fixed.hessian for value in row))
    return difference / fixed_norm if fixed_norm > 0.0 else math.inf


def _build_map_planes(
    root: Path,
    frames: tuple[TUMDepthPoseEntry, ...],
    intrinsics: TUMDepthIntrinsics,
    options: TUMJointSlacOptions,
) -> dict[tuple[int, int, int], Any]:
    records: list[LivoxPointRecord] = []
    for frame in frames:
        image = read_tum_depth_png(root / frame.depth.path)
        for point in sample_tum_depth_points(
            image, intrinsics, max_points=options.map_points_per_frame
        ):
            records.append(
                LivoxPointRecord((*frame.transform_world_camera.transform_point(point), 0.0))
            )
    return build_voxel_plane_map(records, options.voxel_size_m)


def _point_plane_residual(
    point: tuple[float, float, float],
    centroid: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> float:
    norm = math.sqrt(sum(value * value for value in normal))
    return sum(normal[index] * (point[index] - centroid[index]) / norm for index in range(3))


def _rmse_values(values: list[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def _build_problem(
    root: Path,
    selected: tuple[TUMDepthPoseEntry, ...],
    intrinsics: TUMDepthIntrinsics,
    options: TUMJointSlacOptions,
) -> _TUMJointProblem:
    map_frames = selected[: options.map_frame_count]
    query_frames = selected[options.map_frame_count :]
    map_records: list[LivoxPointRecord] = []
    for frame in map_frames:
        image = read_tum_depth_png(root / frame.depth.path)
        for point in sample_tum_depth_points(
            image,
            intrinsics,
            max_points=options.map_points_per_frame,
        ):
            world = frame.transform_world_camera.transform_point(point)
            map_records.append(LivoxPointRecord((*world, 0.0), None))
    plane_map = build_voxel_plane_map(map_records, options.voxel_size_m)
    measurements: list[JointDepthPointToPlaneMeasurement] = []
    factors: list[JointResidualBlock] = []
    blocks: list[JointParameterBlock] = []
    priors: list[JointResidualBlock] = []
    pose_sigma = (options.pose_prior_translation_sigma_m,) * 3 + (
        options.pose_prior_rotation_sigma_rad,
    ) * 3
    for frame_index, frame in enumerate(query_frames):
        pose_block = _pose_block(frame_index, frame)
        blocks.append(
            JointParameterBlock(
                pose_block,
                (0.0,) * 6,
                finite_difference_steps=(1.0e-4,) * 3 + (1.0e-5,) * 3,
            )
        )
        priors.append(
            make_joint_prior_factor(
                factor_id=f"pose-prior-{frame_index:03d}",
                observation_group=f"pose-prior-{frame_index:03d}",
                block=pose_block,
                target=(0.0,) * 6,
                sigma=pose_sigma,
            )
        )
        image = read_tum_depth_png(root / frame.depth.path)
        candidates: list[JointDepthPointToPlaneMeasurement] = []
        for point_index, point in enumerate(
            sample_tum_depth_points(
                image,
                intrinsics,
                max_points=options.query_points_per_frame,
            )
        ):
            world = frame.transform_world_camera.transform_point(point)
            plane = nearest_voxel_plane(
                world,
                plane_map,
                voxel_size_m=options.voxel_size_m,
                correspondence_gate_m=options.correspondence_gate_m,
            )
            if plane is None or plane.normal is None:
                continue
            candidates.append(
                JointDepthPointToPlaneMeasurement(
                    measurement_id=f"tum-{frame_index:03d}-{point_index:05d}",
                    observation_group=f"tum-frame-{frame_index:03d}",
                    normalized_ray_sensor=(
                        point[0] / point[2],
                        point[1] / point[2],
                        1.0,
                    ),
                    nominal_depth_m=point[2],
                    plane_point_world_m=plane.centroid,
                    plane_normal_world=plane.normal,
                    transform_world_body_initial=frame.transform_world_camera,
                    transform_body_sensor_initial=SE3.identity(),
                )
            )
        retained = _bounded_sample(candidates, options.max_correspondences_per_frame)
        measurements.extend(retained)
        factors.extend(
            make_joint_depth_point_to_plane_factor(
                measurement,
                pose_block=pose_block,
                extrinsic_block=_EXTRINSIC_BLOCK,
                depth_calibration_block=_DEPTH_BLOCK,
            )
            for measurement in retained
        )
    known_bad = (options.known_bad_translation_m,) * 3 + (options.known_bad_rotation_rad,) * 3
    blocks.append(
        JointParameterBlock(
            _EXTRINSIC_BLOCK,
            (0.0,) * 6,
            finite_difference_steps=(1.0e-4,) * 3 + (1.0e-5,) * 3,
            known_bad_steps=known_bad,
        )
    )
    blocks.append(
        JointParameterBlock(
            _DEPTH_BLOCK,
            (0.0, 0.0),
            finite_difference_steps=(1.0e-5, 1.0e-5),
            known_bad_steps=(
                options.known_bad_depth_log_scale,
                options.known_bad_depth_bias_m,
            ),
        )
    )
    return _TUMJointProblem(
        map_frames,
        query_frames,
        tuple(measurements),
        tuple(factors),
        tuple(blocks),
        (*factors, *priors),
        len(map_records),
        len(plane_map),
    )


def _bounded_sample(
    measurements: list[JointDepthPointToPlaneMeasurement], maximum: int
) -> list[JointDepthPointToPlaneMeasurement]:
    if len(measurements) <= maximum:
        return measurements
    step = math.ceil(len(measurements) / maximum)
    return measurements[::step][:maximum]


def _pose_block(index: int, frame: TUMDepthPoseEntry) -> str:
    return f"pose_{index:03d}_{frame.depth.timestamp_sec:.6f}"


def _data_only_shared_observability(
    problem: _TUMJointProblem,
    result: JointOptimizerResult,
    *,
    include_depth_calibration: bool,
) -> JointObservabilityEvaluation:
    train_ids = set(result.train_factor_ids)
    factors = [factor for factor in problem.data_factors if factor.factor_id in train_ids]
    blocks = [
        JointParameterBlock(
            _pose_block(index, frame),
            result.optimized_values[_pose_block(index, frame)],
            fixed=True,
        )
        for index, frame in enumerate(problem.query_frames)
    ]
    blocks.extend(
        [
            JointParameterBlock(
                _EXTRINSIC_BLOCK,
                result.optimized_values[_EXTRINSIC_BLOCK],
                finite_difference_steps=(1.0e-4,) * 3 + (1.0e-5,) * 3,
            ),
            JointParameterBlock(
                _DEPTH_BLOCK,
                result.optimized_values[_DEPTH_BLOCK],
                fixed=not include_depth_calibration,
                finite_difference_steps=(1.0e-5, 1.0e-5),
            ),
        ]
    )
    values = {block.name: block.initial_values for block in blocks}
    return evaluate_joint_observability(blocks, factors, values)


def _metrics(
    problem: _TUMJointProblem,
    result: JointOptimizerResult,
    extrinsic: SE3,
    extrinsic_observability: JointObservabilityEvaluation,
    shared_observability: JointObservabilityEvaluation,
    options: TUMJointSlacOptions,
) -> dict[str, MetricResult]:
    train_ids = set(result.train_factor_ids)
    train_factors = [factor for factor in problem.data_factors if factor.factor_id in train_ids]
    train_rmse = _factor_rmse(train_factors, result.optimized_values)
    probes = [probe for probe in result.probes if probe.detectable is not None]
    detectable = sum(probe.detectable is True for probe in probes) / len(probes) if probes else None
    translation_error = math.dist((0.0, 0.0, 0.0), extrinsic.translation_m)
    rotation_error = _rotation_angle_deg(extrinsic)
    reference_grade: Grade = (
        "pass"
        if translation_error <= options.reference_translation_gate_m
        and rotation_error <= options.reference_rotation_gate_deg
        else "fail"
    )
    depth_log_scale, depth_bias_m = result.optimized_values[_DEPTH_BLOCK]
    depth_scale = math.exp(depth_log_scale)
    depth_scale_error_percent = abs(depth_scale - 1.0) * 100.0
    depth_bias_error_m = abs(depth_bias_m)
    depth_scale_grade: Grade = (
        "pass"
        if depth_scale_error_percent <= options.reference_depth_scale_error_gate_percent
        else "fail"
    )
    depth_bias_grade: Grade = (
        "pass" if depth_bias_error_m <= options.reference_depth_bias_gate_m else "fail"
    )
    expected_rank = 6 * (len(problem.query_frames) + 1) + 2
    return {
        "native_tum_joint_slac_available": MetricResult(
            value=1.0, grade="pass", reason="native TUM multi-capture joint solver executed"
        ),
        "tum_joint_map_frame_count": MetricResult(
            value=float(len(problem.map_frames)), unit="frames", grade="pass"
        ),
        "tum_joint_query_frame_count": MetricResult(
            value=float(len(problem.query_frames)), unit="frames", grade="pass"
        ),
        "tum_joint_correspondence_count": MetricResult(
            value=float(len(problem.measurements)), unit="correspondences", grade="pass"
        ),
        "tum_joint_point_to_plane_rmse_m": MetricResult(
            train=train_rmse,
            holdout=result.holdout_rmse,
            unit="m",
            grade="warn",
            reason="query-frame point-to-plane RMSE; pose priors excluded",
        ),
        "tum_joint_augmented_information_rank": MetricResult(
            value=float(result.information_rank),
            grade="pass" if result.information_rank == expected_rank else "warn",
            reason=(
                f"joint rank for {len(problem.query_frames)} pose blocks plus shared "
                "extrinsic, including pose priors"
            ),
        ),
        "tum_joint_data_only_extrinsic_rank": MetricResult(
            value=float(extrinsic_observability.information_rank),
            grade="pass" if extrinsic_observability.information_rank == 6 else "warn",
            reason="shared-extrinsic rank from train geometry with optimized poses fixed",
        ),
        "tum_joint_data_only_extrinsic_condition_number": MetricResult(
            value=extrinsic_observability.condition_number,
            grade=(
                "pass"
                if extrinsic_observability.condition_number is not None
                and extrinsic_observability.information_rank == 6
                else "warn"
            ),
            reason="fixed-pose train Jacobian diagnostic; not covariance",
        ),
        "tum_joint_data_only_shared_rank": MetricResult(
            value=float(shared_observability.information_rank),
            grade="pass" if shared_observability.information_rank == 8 else "warn",
            reason="train rank of shared extrinsic plus depth log-scale/bias with poses fixed",
        ),
        "tum_joint_data_only_shared_condition_number": MetricResult(
            value=shared_observability.condition_number,
            grade=(
                "pass"
                if shared_observability.condition_number is not None
                and shared_observability.information_rank == 8
                else "warn"
            ),
            reason="unit-dependent shared 8D train Jacobian diagnostic; not covariance",
        ),
        "tum_joint_depth_scale": MetricResult(
            value=depth_scale,
            grade=depth_scale_grade,
            reason="multiplicative correction applied to nominal metric query depth",
        ),
        "tum_joint_depth_bias_m": MetricResult(
            value=depth_bias_m,
            unit="m",
            grade=depth_bias_grade,
            reason="additive correction applied after multiplicative query-depth scale",
        ),
        "tum_joint_depth_scale_reference_error_percent": MetricResult(
            value=depth_scale_error_percent,
            unit="percent",
            grade=depth_scale_grade,
            reason="TUM depth is officially pre-scaled; reference multiplier is one",
        ),
        "tum_joint_depth_bias_reference_error_m": MetricResult(
            value=depth_bias_error_m,
            unit="m",
            grade=depth_bias_grade,
            reason="TUM depth is officially pre-scaled; reference additive bias is zero",
        ),
        "tum_joint_known_bad_detectable_fraction": MetricResult(
            value=detectable,
            grade="pass" if detectable == 1.0 else "warn",
            reason=(
                "signed shared-extrinsic and depth-calibration perturbations scored "
                "on held-out query frames"
            ),
        ),
        "tum_joint_identity_reference_translation_error_m": MetricResult(
            value=translation_error,
            unit="m",
            grade=reference_grade,
            reason=(
                "TUM ground truth already describes the camera frame; mounting truth is identity"
            ),
        ),
        "tum_joint_identity_reference_rotation_error_deg": MetricResult(
            value=rotation_error,
            unit="deg",
            grade=reference_grade,
            reason=(
                "TUM ground truth already describes the camera frame; mounting truth is identity"
            ),
        ),
    }


def _factor_rmse(
    factors: list[JointResidualBlock], values: dict[str, tuple[float, ...]]
) -> float | None:
    residuals = [value for factor in factors for value in factor.residuals(values)]
    if not residuals:
        return None
    return math.sqrt(sum(value * value for value in residuals) / len(residuals))


def _replication_metrics(
    runs: tuple[_TUMWindowRun, ...],
    transfers: tuple[dict[str, float | int | None], ...],
    options: TUMJointSlacOptions,
) -> dict[str, MetricResult]:
    expected = len(options.replication_start_indices)
    converged = sum(run.result.status == "converged" for run in runs)
    scales = [math.exp(run.result.optimized_values[_DEPTH_BLOCK][0]) for run in runs]
    biases = [run.result.optimized_values[_DEPTH_BLOCK][1] for run in runs]
    translations = [
        math.dist(
            (0.0, 0.0, 0.0),
            se3_from_tangent(run.result.optimized_values[_EXTRINSIC_BLOCK]).translation_m,
        )
        for run in runs
    ]
    rotations = [
        _rotation_angle_deg(se3_from_tangent(run.result.optimized_values[_EXTRINSIC_BLOCK]))
        for run in runs
    ]
    reference_passes = sum(
        abs(scale - 1.0) * 100.0 <= options.reference_depth_scale_error_gate_percent
        and abs(bias) <= options.reference_depth_bias_gate_m
        and translation <= options.reference_translation_gate_m
        and rotation <= options.reference_rotation_gate_deg
        for scale, bias, translation, rotation in zip(
            scales, biases, translations, rotations, strict=True
        )
    )
    deltas = [
        float(transfer["delta_rmse_m"])
        for transfer in transfers
        if transfer["delta_rmse_m"] is not None
    ]
    nondegrading = sum(delta <= options.cross_window_nondegradation_margin_m for delta in deltas)
    complete = len(runs) == expected
    return {
        "tum_joint_replication_window_count": MetricResult(
            value=float(len(runs)),
            unit="windows",
            grade="pass" if complete else "fail",
            reason=f"resolved disjoint temporal windows; expected {expected}",
        ),
        "tum_joint_replication_converged_fraction": MetricResult(
            value=converged / expected if expected else None,
            grade="pass" if converged == expected else "fail",
        ),
        "tum_joint_replication_reference_pass_fraction": MetricResult(
            value=reference_passes / expected if expected else None,
            grade="pass" if reference_passes == expected else "fail",
            reason="fraction passing unchanged mounting and depth references",
        ),
        "tum_joint_replication_depth_scale_range_percent": MetricResult(
            value=(max(scales) - min(scales)) * 100.0 if scales else None,
            unit="percent",
            grade="warn",
            reason="temporal-window spread; diagnostic, not a relaxed reference gate",
        ),
        "tum_joint_replication_depth_bias_range_m": MetricResult(
            value=max(biases) - min(biases) if biases else None,
            unit="m",
            grade="warn",
            reason="temporal-window spread; diagnostic, not a relaxed reference gate",
        ),
        "tum_joint_cross_window_max_holdout_delta_rmse_m": MetricResult(
            value=max(deltas) if deltas else None,
            unit="m",
            grade=(
                "pass"
                if deltas and max(deltas) <= options.cross_window_nondegradation_margin_m
                else "fail"
            ),
            reason="source shared calibration transferred onto target-window optimized poses",
        ),
        "tum_joint_cross_window_nondegrading_fraction": MetricResult(
            value=nondegrading / len(deltas) if deltas else None,
            grade="pass" if deltas and nondegrading == len(deltas) else "fail",
            reason=(
                "ordered transfers within declared "
                f"{options.cross_window_nondegradation_margin_m:g} m RMSE margin"
            ),
        ),
    }


def _spatial_ablation_metrics(
    baseline_runs: tuple[_TUMWindowRun, ...],
    spatial_runs: tuple[_TUMSpatialWindowRun, ...],
    transfers: tuple[dict[str, float | int | None], ...],
    options: TUMJointSlacOptions,
) -> dict[str, MetricResult]:
    baselines = {run.start_index: run for run in baseline_runs}
    holdout_deltas: list[float] = []
    for run in spatial_runs:
        spatial_rmse = run.result.holdout_rmse
        scalar_rmse = baselines[run.start_index].result.holdout_rmse
        if spatial_rmse is not None and scalar_rmse is not None:
            holdout_deltas.append(spatial_rmse - scalar_rmse)
    improved = sum(delta < 0.0 for delta in holdout_deltas)
    transfer_deltas = [
        float(transfer["delta_rmse_m"])
        for transfer in transfers
        if transfer["delta_rmse_m"] is not None
    ]
    nondegrading = sum(
        delta <= options.cross_window_nondegradation_margin_m for delta in transfer_deltas
    )
    probe_fractions = []
    for run in spatial_runs:
        probes = [probe for probe in run.result.probes if probe.detectable is not None]
        if probes:
            probe_fractions.append(sum(probe.detectable is True for probe in probes) / len(probes))
    max_offset = max(
        (
            abs(offset)
            for run in spatial_runs
            for offset in run.result.optimized_values[_SPATIAL_DEPTH_BLOCK]
        ),
        default=None,
    )
    spatial_biases = [run.result.optimized_values[_SPATIAL_BIAS_BLOCK][0] for run in spatial_runs]
    return {
        "tum_joint_spatial_ablation_converged_fraction": MetricResult(
            value=(
                sum(run.result.status == "converged" for run in spatial_runs) / len(baseline_runs)
                if baseline_runs
                else None
            ),
            grade=(
                "pass"
                if baseline_runs
                and all(run.result.status == "converged" for run in spatial_runs)
                and len(spatial_runs) == len(baseline_runs)
                else "fail"
            ),
        ),
        "tum_joint_spatial_ablation_holdout_improved_fraction": MetricResult(
            value=improved / len(holdout_deltas) if holdout_deltas else None,
            grade="pass" if holdout_deltas and improved == len(holdout_deltas) else "fail",
            reason="same-window spatial-lattice holdout RMSE compared with scalar scale/bias",
        ),
        "tum_joint_spatial_ablation_worst_holdout_delta_rmse_m": MetricResult(
            value=max(holdout_deltas) if holdout_deltas else None,
            unit="m",
            grade=("pass" if holdout_deltas and max(holdout_deltas) <= 0.0 else "fail"),
        ),
        "tum_joint_spatial_lattice_max_abs_offset_m": MetricResult(
            value=max_offset,
            unit="m",
            grade="warn",
            reason="largest fitted zero-mean ray-depth contrast over all windows",
        ),
        "tum_joint_spatial_constant_bias_range_m": MetricResult(
            value=(max(spatial_biases) - min(spatial_biases) if spatial_biases else None),
            unit="m",
            grade="warn",
            reason="window spread of the constant mode separated from spatial contrasts",
        ),
        "tum_joint_spatial_known_bad_detectable_fraction_min": MetricResult(
            value=min(probe_fractions) if probe_fractions else None,
            grade=("pass" if probe_fractions and min(probe_fractions) == 1.0 else "warn"),
        ),
        "tum_joint_spatial_cross_window_max_holdout_delta_rmse_m": MetricResult(
            value=max(transfer_deltas) if transfer_deltas else None,
            unit="m",
            grade=(
                "pass"
                if transfer_deltas
                and max(transfer_deltas) <= options.cross_window_nondegradation_margin_m
                else "fail"
            ),
        ),
        "tum_joint_spatial_cross_window_nondegrading_fraction": MetricResult(
            value=nondegrading / len(transfer_deltas) if transfer_deltas else None,
            grade=("pass" if transfer_deltas and nondegrading == len(transfer_deltas) else "fail"),
        ),
    }


def _xyz_ablation_metrics(
    scalar_runs: tuple[_TUMSpatialWindowRun, ...],
    xyz_runs: tuple[_TUMXYZWindowRun, ...],
    transfers: tuple[dict[str, float | int | None], ...],
    options: TUMJointSlacOptions,
) -> dict[str, MetricResult]:
    baselines = {run.start_index: run for run in scalar_runs}
    holdout_deltas: list[float] = []
    for run in xyz_runs:
        baseline = baselines.get(run.start_index)
        xyz_rmse = run.result.holdout_rmse
        scalar_rmse = baseline.result.holdout_rmse if baseline is not None else None
        if xyz_rmse is not None and scalar_rmse is not None:
            holdout_deltas.append(xyz_rmse - scalar_rmse)
    improved = sum(delta < 0.0 for delta in holdout_deltas)
    field_probe_fractions: list[float] = []
    for run in xyz_runs:
        probes = [
            probe.detectable
            for probe in run.result.probes
            if probe.block == _XYZ_LATTICE_BLOCK and probe.detectable is not None
        ]
        if probes:
            field_probe_fractions.append(sum(value is True for value in probes) / len(probes))
    offsets = [
        value for run in xyz_runs for value in run.result.optimized_values[_XYZ_LATTICE_BLOCK]
    ]
    rotation_angles = [
        _quaternion_angle_deg(rotation)
        for run in xyz_runs
        for rotations in run.local_rotation_history
        for rotation in rotations
    ]
    field_ranks = [run.data_only_field_observability.information_rank for run in xyz_runs]
    shared_ranks = [run.data_only_shared_observability.information_rank for run in xyz_runs]
    transfer_deltas = [
        float(transfer["delta_rmse_m"])
        for transfer in transfers
        if transfer["delta_rmse_m"] is not None
    ]
    nondegrading = sum(
        delta <= options.cross_window_nondegradation_margin_m for delta in transfer_deltas
    )
    expected = len(scalar_runs)
    complete = len(xyz_runs) == expected
    return {
        "tum_joint_xyz_ablation_window_count": MetricResult(
            value=float(len(xyz_runs)),
            unit="windows",
            grade="pass" if complete else "fail",
            reason=f"full XYZ lattice windows; expected {expected}",
        ),
        "tum_joint_xyz_ablation_converged_fraction": MetricResult(
            value=(
                sum(run.result.status == "converged" for run in xyz_runs) / expected
                if expected
                else None
            ),
            grade=(
                "pass"
                if complete and all(run.result.status == "converged" for run in xyz_runs)
                else "fail"
            ),
            reason="final frozen-local-rotation optimizer status over temporal windows",
        ),
        "tum_joint_xyz_ablation_holdout_improved_fraction": MetricResult(
            value=improved / len(holdout_deltas) if holdout_deltas else None,
            grade="pass" if holdout_deltas and improved == len(holdout_deltas) else "fail",
            reason="full XYZ versus scalar ray-depth same-window holdout RMSE",
        ),
        "tum_joint_xyz_ablation_worst_holdout_delta_rmse_m": MetricResult(
            value=max(holdout_deltas) if holdout_deltas else None,
            unit="m",
            grade="pass" if holdout_deltas and max(holdout_deltas) <= 0.0 else "fail",
        ),
        "tum_joint_xyz_data_only_field_rank_min": MetricResult(
            value=float(min(field_ranks)) if field_ranks else None,
            grade="pass" if field_ranks and min(field_ranks) == 24 else "fail",
            reason="train data rank of 24 XYZ controls with poses/extrinsic fixed",
        ),
        "tum_joint_xyz_data_only_shared_rank_min": MetricResult(
            value=float(min(shared_ranks)) if shared_ranks else None,
            grade="warn" if shared_ranks and min(shared_ranks) == 24 else "fail",
            reason="train data rank of 30 extrinsic/field dimensions; six rigid exchange modes",
        ),
        "tum_joint_xyz_lattice_max_abs_offset_m": MetricResult(
            value=max((abs(value) for value in offsets), default=None),
            unit="m",
            grade="warn",
            reason="largest fitted full-XYZ control displacement component",
        ),
        "tum_joint_xyz_local_rotation_update_max_deg": MetricResult(
            value=max(rotation_angles) if rotation_angles else None,
            unit="deg",
            grade="warn",
            reason="largest frozen local Procrustes rotation used by an optimizer round",
        ),
        "tum_joint_xyz_known_bad_detectable_fraction_min": MetricResult(
            value=min(field_probe_fractions) if field_probe_fractions else None,
            grade=(
                "pass" if field_probe_fractions and min(field_probe_fractions) == 1.0 else "warn"
            ),
            reason="minimum final frozen-factor detection over 48 signed XYZ control probes",
        ),
        "tum_joint_xyz_cross_window_max_holdout_delta_rmse_m": MetricResult(
            value=max(transfer_deltas) if transfer_deltas else None,
            unit="m",
            grade=(
                "pass"
                if transfer_deltas
                and max(transfer_deltas) <= options.cross_window_nondegradation_margin_m
                else "fail"
            ),
            reason="worst ordered shared-extrinsic/full-XYZ field transfer",
        ),
        "tum_joint_xyz_cross_window_nondegrading_fraction": MetricResult(
            value=nondegrading / len(transfer_deltas) if transfer_deltas else None,
            grade=("pass" if transfer_deltas and nondegrading == len(transfer_deltas) else "fail"),
            reason=(
                "ordered full-XYZ transfers within declared "
                f"{options.cross_window_nondegradation_margin_m:g} m RMSE margin"
            ),
        ),
    }


def _xyz_pair_ablation_metrics(
    runs: tuple[_TUMXYZPairWindowRun, ...],
    transfers: tuple[dict[str, float | int | None], ...],
    options: TUMJointSlacOptions,
) -> dict[str, MetricResult]:
    expected = len(options.replication_start_indices)
    initial_deltas: list[float] = []
    probe_fractions: list[float] = []
    for run in runs:
        holdout_groups = set(run.result.holdout_observation_groups)
        holdout = [
            factor
            for factor in run.problem.data_factors
            if factor.observation_group in holdout_groups
        ]
        initial_values = {
            block.name: block.initial_values for block in run.problem.parameter_blocks
        }
        initial_rmse = _factor_rmse(holdout, initial_values)
        if initial_rmse is not None and run.result.holdout_rmse is not None:
            initial_deltas.append(run.result.holdout_rmse - initial_rmse)
        probes = [probe.detectable for probe in run.result.probes if probe.detectable is not None]
        if probes:
            probe_fractions.append(sum(detectable is True for detectable in probes) / len(probes))
    field_ranks = [run.data_only_field_observability.information_rank for run in runs]
    joint_ranks = [run.data_only_joint_observability.information_rank for run in runs]
    augmented_ranks = [run.result.information_rank for run in runs]
    correspondence_counts = [len(run.problem.measurements) for run in runs]
    transfer_deltas = [
        float(transfer["delta_rmse_m"])
        for transfer in transfers
        if transfer["delta_rmse_m"] is not None
    ]
    nondegrading = sum(
        delta <= options.cross_window_nondegradation_margin_m for delta in transfer_deltas
    )
    expected_augmented_rank = 6 * (len(runs[0].problem.query_frames) - 1) + 24 if runs else None
    complete = len(runs) == expected
    return {
        "tum_joint_xyz_pair_window_count": MetricResult(
            value=float(len(runs)),
            unit="windows",
            grade="pass" if complete else "fail",
            reason=f"two-sided Eq. (2) windows; expected {expected}",
        ),
        "tum_joint_xyz_pair_converged_fraction": MetricResult(
            value=(
                sum(run.result.status == "converged" for run in runs) / expected
                if expected
                else None
            ),
            grade=(
                "pass"
                if complete and all(run.result.status == "converged" for run in runs)
                else "fail"
            ),
        ),
        "tum_joint_xyz_pair_correspondence_count_min": MetricResult(
            value=float(min(correspondence_counts)) if correspondence_counts else None,
            unit="correspondences",
            grade=(
                "pass"
                if correspondence_counts
                and min(correspondence_counts) >= _MIN_TOTAL_CORRESPONDENCES
                else "fail"
            ),
        ),
        "tum_joint_xyz_pair_holdout_improved_fraction": MetricResult(
            value=(
                sum(delta < 0.0 for delta in initial_deltas) / len(initial_deltas)
                if initial_deltas
                else None
            ),
            grade=(
                "pass"
                if initial_deltas and all(delta <= 0.0 for delta in initial_deltas)
                else "fail"
            ),
            reason="final versus initial fixed-correspondence pair holdout RMSE",
        ),
        "tum_joint_xyz_pair_worst_holdout_delta_rmse_m": MetricResult(
            value=max(initial_deltas) if initial_deltas else None,
            unit="m",
            grade=("pass" if initial_deltas and max(initial_deltas) <= 0.0 else "fail"),
        ),
        "tum_joint_xyz_pair_data_only_field_rank_min": MetricResult(
            value=float(min(field_ranks)) if field_ranks else None,
            grade="pass" if field_ranks and min(field_ranks) == 24 else "fail",
            reason="two-sided train data rank of 24 XYZ controls with poses fixed",
        ),
        "tum_joint_xyz_pair_data_only_joint_rank_min": MetricResult(
            value=float(min(joint_ranks)) if joint_ranks else None,
            grade=(
                "pass"
                if joint_ranks
                and expected_augmented_rank is not None
                and min(joint_ranks) == expected_augmented_rank
                else "warn"
            ),
            reason="pair-data pose/field rank after fixing the first-pose world gauge",
        ),
        "tum_joint_xyz_pair_augmented_rank_min": MetricResult(
            value=float(min(augmented_ranks)) if augmented_ranks else None,
            grade=(
                "pass"
                if augmented_ranks
                and expected_augmented_rank is not None
                and min(augmented_ranks) == expected_augmented_rank
                else "fail"
            ),
            reason="pair data plus train-only pose, shape, and rigid-field priors",
        ),
        "tum_joint_xyz_pair_known_bad_detectable_fraction_min": MetricResult(
            value=min(probe_fractions) if probe_fractions else None,
            grade=("pass" if probe_fractions and min(probe_fractions) == 1.0 else "warn"),
            reason="signed non-gauge pose and XYZ-field perturbations on held-out pairs",
        ),
        "tum_joint_xyz_pair_cross_window_max_holdout_delta_rmse_m": MetricResult(
            value=max(transfer_deltas) if transfer_deltas else None,
            unit="m",
            grade=(
                "pass"
                if transfer_deltas
                and max(transfer_deltas) <= options.cross_window_nondegradation_margin_m
                else "fail"
            ),
        ),
        "tum_joint_xyz_pair_cross_window_nondegrading_fraction": MetricResult(
            value=nondegrading / len(transfer_deltas) if transfer_deltas else None,
            grade=("pass" if transfer_deltas and nondegrading == len(transfer_deltas) else "fail"),
        ),
    }


def _rematching_metrics(
    evaluations: tuple[_TUMRematchEvaluation, ...],
) -> dict[str, MetricResult]:
    pair_jaccards = [
        item.stability.pair_jaccard
        for item in evaluations
        if item.stability.pair_jaccard is not None
    ]
    retentions = [
        item.stability.retained_query_fraction
        for item in evaluations
        if item.stability.retained_query_fraction is not None
    ]
    same_targets = [
        item.stability.same_target_fraction
        for item in evaluations
        if item.stability.same_target_fraction is not None
    ]
    deltas = [
        item.rematched_holdout_rmse_m - item.fixed_holdout_rmse_m
        for item in evaluations
        if item.rematched_holdout_rmse_m is not None and item.fixed_holdout_rmse_m is not None
    ]
    multistart = next(
        (
            item.multistart_evaluation
            for item in evaluations
            if item.multistart_evaluation is not None
        ),
        None,
    )
    return {
        "tum_joint_rematch_pair_jaccard_min": MetricResult(
            value=min(pair_jaccards) if pair_jaccards else None,
            grade="warn",
            reason="minimum exact query/voxel-plane pair Jaccard over temporal windows",
        ),
        "tum_joint_rematch_retained_query_fraction_min": MetricResult(
            value=min(retentions) if retentions else None,
            grade="warn",
        ),
        "tum_joint_rematch_same_target_fraction_min": MetricResult(
            value=min(same_targets) if same_targets else None,
            grade="warn",
        ),
        "tum_joint_rematch_best_holdout_delta_rmse_m": MetricResult(
            value=min(deltas) if deltas else None,
            unit="m",
            grade="warn",
            reason="rematched minus fixed-assignment RMSE; selection-dependent diagnostic",
        ),
        "tum_joint_rematched_curvature_rank_min": MetricResult(
            value=(
                float(min(item.rematched_curvature.rank for item in evaluations))
                if evaluations
                else None
            ),
            grade="warn",
            reason="rank of shared extrinsic/bias rematched holdout objective Hessian",
        ),
        "tum_joint_rematched_negative_curvature_count_max": MetricResult(
            value=(
                float(
                    max(item.rematched_curvature.negative_eigenvalue_count for item in evaluations)
                )
                if evaluations
                else None
            ),
            grade="warn",
            reason="negative eigenvalues expose local non-convex rematching directions",
        ),
        "tum_joint_curvature_relative_hessian_difference_max": MetricResult(
            value=(
                max(item.relative_hessian_difference for item in evaluations)
                if evaluations
                else None
            ),
            grade="warn",
            reason="Frobenius difference between rematched and fixed Hessians / fixed norm",
        ),
        "tum_joint_multistart_converged_fraction": MetricResult(
            value=(
                multistart.converged_start_count / multistart.declared_start_count
                if multistart is not None and multistart.declared_start_count
                else None
            ),
            grade="warn",
        ),
        "tum_joint_multistart_basin_count": MetricResult(
            value=float(multistart.cluster_count) if multistart is not None else None,
            unit="basins",
            grade="warn",
        ),
        "tum_joint_multistart_competitive_basin_count": MetricResult(
            value=(float(multistart.competitive_cluster_count) if multistart is not None else None),
            unit="basins",
            grade=(
                "pass"
                if multistart is not None and multistart.competitive_cluster_count == 1
                else "fail"
            ),
        ),
        "tum_joint_multistart_ambiguity": MetricResult(
            value=(1.0 if multistart is not None and multistart.ambiguous else 0.0),
            grade=("fail" if multistart is not None and multistart.ambiguous else "pass"),
            reason="one means separated objective-competitive solution basins exist",
        ),
    }


def _iterative_reassociation_metrics(
    runs: tuple[_TUMIterativeReassociationRun, ...],
    options: TUMJointSlacOptions,
) -> dict[str, MetricResult]:
    expected = len(options.replication_start_indices)
    converged = sum(run.result.status == "converged" for run in runs)
    train_jaccards = [
        run.result.terminal_train_stability.pair_jaccard
        for run in runs
        if run.result.terminal_train_stability.pair_jaccard is not None
    ]
    train_retentions = [
        run.result.terminal_train_stability.retained_query_fraction
        for run in runs
        if run.result.terminal_train_stability.retained_query_fraction is not None
    ]
    holdout_jaccards = [
        run.result.terminal_holdout_stability.pair_jaccard
        for run in runs
        if run.result.terminal_holdout_stability.pair_jaccard is not None
    ]
    holdout_deltas = [
        run.result.final_result.holdout_rmse - run.baseline.result.holdout_rmse
        for run in runs
        if run.result.final_result.holdout_rmse is not None
        and run.baseline.result.holdout_rmse is not None
    ]
    detectable_fractions: list[float] = []
    aware_detectable_fractions: list[float] = []
    aware_support_collapse_fractions: list[float] = []
    aware_pair_jaccards: list[float] = []
    aware_valid_fractions: list[float] = []
    aware_baseline_retentions: list[float] = []
    for run in runs:
        probes = [
            probe.detectable
            for probe in run.result.final_result.probes
            if probe.detectable is not None
        ]
        if probes:
            detectable_fractions.append(sum(value is True for value in probes) / len(probes))
        aware = run.probe_evaluation
        aware_baseline_retentions.append(aware.baseline_retained_fraction)
        valid = [probe for probe in aware.probes if probe.detectable is not None]
        if aware.probes:
            aware_valid_fractions.append(len(valid) / len(aware.probes))
        if valid:
            aware_detectable_fractions.append(
                sum(probe.detectable is True for probe in valid) / len(valid)
            )
        support = [
            probe.support_collapse for probe in aware.probes if probe.support_collapse is not None
        ]
        if support:
            aware_support_collapse_fractions.append(
                sum(value is True for value in support) / len(support)
            )
        aware_pair_jaccards.extend(
            probe.stability.pair_jaccard
            for probe in aware.probes
            if probe.stability is not None and probe.stability.pair_jaccard is not None
        )
    train_jaccard_min = min(train_jaccards) if train_jaccards else None
    train_retention_min = min(train_retentions) if train_retentions else None
    complete = len(runs) == expected
    return {
        "tum_joint_reassociation_window_count": MetricResult(
            value=float(len(runs)),
            unit="windows",
            grade="pass" if complete else "fail",
            reason=f"iterative reassociation windows; expected {expected}",
        ),
        "tum_joint_reassociation_converged_fraction": MetricResult(
            value=converged / expected if expected else None,
            grade="pass" if complete and converged == expected else "fail",
            reason="outer-loop convergence uses train assignments only",
        ),
        "tum_joint_reassociation_train_pair_jaccard_min": MetricResult(
            value=train_jaccard_min,
            grade=(
                "pass"
                if train_jaccard_min is not None
                and train_jaccard_min >= options.reassociation_min_train_pair_jaccard
                else "fail"
            ),
            reason="terminal train query/target pair Jaccard over temporal windows",
        ),
        "tum_joint_reassociation_train_retained_query_fraction_min": MetricResult(
            value=train_retention_min,
            grade=(
                "pass"
                if train_retention_min is not None
                and train_retention_min >= options.reassociation_min_train_retained_fraction
                else "fail"
            ),
            reason="terminal train query retention over temporal windows",
        ),
        "tum_joint_reassociation_holdout_pair_jaccard_min": MetricResult(
            value=min(holdout_jaccards) if holdout_jaccards else None,
            grade="warn",
            reason="holdout assignment stability is diagnostic and never stops fitting",
        ),
        "tum_joint_reassociation_holdout_rmse_delta_max_m": MetricResult(
            value=max(holdout_deltas) if holdout_deltas else None,
            unit="m",
            grade="warn",
            reason="final reassociated minus fixed-assignment spatial holdout RMSE",
        ),
        "tum_joint_reassociation_outer_iterations_max": MetricResult(
            value=(float(max(len(run.result.iterations) for run in runs)) if runs else None),
            unit="iterations",
            grade="warn",
            reason="maximum executed reassociation rounds over temporal windows",
        ),
        "tum_joint_reassociation_known_bad_detectable_fraction_min": MetricResult(
            value=min(detectable_fractions) if detectable_fractions else None,
            grade=("pass" if detectable_fractions and min(detectable_fractions) == 1.0 else "warn"),
            reason=(
                "signed probes on final frozen correspondences; reported separately "
                "from reassociation-aware probes"
            ),
        ),
        "tum_joint_reassociation_aware_known_bad_detectable_fraction_min": MetricResult(
            value=(min(aware_detectable_fractions) if aware_detectable_fractions else None),
            grade=(
                "pass"
                if aware_detectable_fractions and min(aware_detectable_fractions) == 1.0
                else "warn"
            ),
            reason="fixed-population holdout probes rebuild correspondences",
        ),
        "tum_joint_reassociation_aware_probe_valid_fraction_min": MetricResult(
            value=min(aware_valid_fractions) if aware_valid_fractions else None,
            grade=(
                "pass" if aware_valid_fractions and min(aware_valid_fractions) == 1.0 else "fail"
            ),
            reason="fraction of declared probes with a valid reassociation evaluation",
        ),
        "tum_joint_reassociation_aware_support_collapse_fraction_max": MetricResult(
            value=(
                max(aware_support_collapse_fractions) if aware_support_collapse_fractions else None
            ),
            grade=(
                "pass"
                if aware_support_collapse_fractions and max(aware_support_collapse_fractions) == 0.0
                else "warn"
            ),
            reason="probe retention below the declared fixed-population support gate",
        ),
        "tum_joint_reassociation_aware_pair_jaccard_min": MetricResult(
            value=min(aware_pair_jaccards) if aware_pair_jaccards else None,
            grade="warn",
            reason="minimum baseline/perturbed holdout query-target pair Jaccard",
        ),
        "tum_joint_reassociation_aware_baseline_retained_fraction_min": MetricResult(
            value=(min(aware_baseline_retentions) if aware_baseline_retentions else None),
            grade=(
                "pass"
                if aware_baseline_retentions
                and min(aware_baseline_retentions)
                >= options.reassociation_min_train_retained_fraction
                else "fail"
            ),
            reason="terminal rematched holdout support relative to the initial population",
        ),
    }


def _rotation_angle_deg(transform: SE3) -> float:
    return _quaternion_angle_deg(transform.rotation_quat_xyzw)


def _quaternion_angle_deg(quaternion: QuaternionXYZW) -> float:
    return math.degrees(2.0 * math.acos(min(1.0, abs(quaternion[3]))))


def _warnings(result: JointOptimizerResult, extrinsic_rank: int, shared_rank: int) -> list[str]:
    warnings = [
        "joint rank includes independently measured TUM pose priors; use the separate "
        "data-only shared calibration ranks for geometric observability"
    ]
    if result.status != "converged":
        warnings.append(f"native TUM joint SLAC status is {result.status}: {result.reason}")
    if extrinsic_rank < 6:
        warnings.append(f"TUM train geometry shared-extrinsic rank is {extrinsic_rank} < 6")
    if shared_rank < 8:
        warnings.append(f"TUM train geometry extrinsic/depth shared rank is {shared_rank} < 8")
    return warnings


def _provenance(
    *,
    root: Path,
    depth_index: Path,
    groundtruth_path: Path,
    paired_count: int,
    problem: _TUMJointProblem,
    result: JointOptimizerResult,
    intrinsics: TUMDepthIntrinsics,
    options: TUMJointSlacOptions,
    variable: str,
    extrinsic_observability: JointObservabilityEvaluation,
    shared_observability: JointObservabilityEvaluation,
    replication_runs: tuple[_TUMWindowRun, ...],
    transfers: tuple[dict[str, float | int | None], ...],
    spatial_runs: tuple[_TUMSpatialWindowRun, ...],
    spatial_transfers: tuple[dict[str, float | int | None], ...],
    xyz_runs: tuple[_TUMXYZWindowRun, ...],
    xyz_transfers: tuple[dict[str, float | int | None], ...],
    xyz_pair_runs: tuple[_TUMXYZPairWindowRun, ...],
    xyz_pair_transfers: tuple[dict[str, float | int | None], ...],
    reassociation_runs: tuple[_TUMIterativeReassociationRun, ...],
    rematching: tuple[_TUMRematchEvaluation, ...],
) -> dict[str, Any]:
    selected = [*problem.map_frames, *problem.query_frames]
    return {
        "native_tum_joint_slac_method": "tum_multicapture_pose_extrinsic_depth/v1.3",
        "native_tum_joint_slac_status": result.status,
        "native_tum_joint_slac_variable": variable,
        "native_tum_joint_slac_depth_index_path": str(depth_index),
        "native_tum_joint_slac_depth_index_sha256": sha256_path(depth_index),
        "native_tum_joint_slac_groundtruth_path": str(groundtruth_path),
        "native_tum_joint_slac_groundtruth_sha256": sha256_path(groundtruth_path),
        "native_tum_joint_slac_paired_frame_count": paired_count,
        "native_tum_joint_slac_map_frame_ids": [
            f"{frame.depth.timestamp_sec:.6f}" for frame in problem.map_frames
        ],
        "native_tum_joint_slac_query_frame_ids": [
            f"{frame.depth.timestamp_sec:.6f}" for frame in problem.query_frames
        ],
        "native_tum_joint_slac_selected_depth_files": [
            {
                "path": str(root / frame.depth.path),
                "sha256": sha256_path(root / frame.depth.path),
                "pose_time_delta_sec": frame.absolute_time_delta_sec,
            }
            for frame in selected
        ],
        "native_tum_joint_slac_intrinsics": intrinsics.__dict__,
        "native_tum_joint_slac_options": options.as_dict(),
        "native_tum_joint_slac_map_record_count": problem.map_record_count,
        "native_tum_joint_slac_map_plane_count": problem.map_plane_count,
        "native_tum_joint_slac_correspondence_count": len(problem.measurements),
        "native_tum_joint_slac_split_policy": (
            "map frames are disjoint from query frames; query frame IDs are grouped "
            "before train/holdout; ground-truth pose priors are train-only"
        ),
        "native_tum_joint_slac_reference_policy": (
            "TUM ground truth is T_world_camera, so T_trajectory_camera identity is "
            "the independently measured mounting reference; official TUM depth is "
            "pre-scaled, so depth multiplier/bias reference is one/zero"
        ),
        "native_tum_joint_slac_depth_correction_convention": (
            "query z_corrected_m = exp(log_scale) * z_nominal_m + bias_m; "
            "disjoint map depths remain the fixed metric reference"
        ),
        "native_tum_joint_slac_data_only_extrinsic_observability": (
            extrinsic_observability.as_dict()
        ),
        "native_tum_joint_slac_data_only_shared_observability": (shared_observability.as_dict()),
        "native_tum_joint_slac_solver": result.as_dict(),
        "native_tum_joint_slac_replication_policy": (
            "each configured start index rebuilds an independent fixed map and query "
            "problem; the primary window is reused without recomputation"
        ),
        "native_tum_joint_slac_replication_windows": [
            _window_provenance(root, run) for run in replication_runs
        ],
        "native_tum_joint_slac_cross_window_policy": (
            "source-window shared extrinsic and depth correction are evaluated on "
            "target holdout factors while retaining target-window optimized poses; "
            "this tests shared-parameter transfer, not independent pose transfer"
        ),
        "native_tum_joint_slac_cross_window_transfers": list(transfers),
        "native_tum_joint_slac_spatial_ablation_model": {
            "kind": "constant_plus_zero_mean_trilinear_ray_depth_displacement",
            "paper_full_xyz_displacement_field": False,
            "lattice_shape": list(_SPATIAL_LATTICE_SHAPE),
            "minimum_sensor_m": list(_SPATIAL_LATTICE_MINIMUM_M),
            "maximum_sensor_m": list(_SPATIAL_LATTICE_MAXIMUM_M),
            "smoothness": "first_order_scalar_neighbor_differences",
            "smoothness_sigma_m": options.spatial_lattice_smoothness_sigma_m,
            "zero_mean": "soft mean-offset equality residual",
            "zero_mean_sigma_m": options.spatial_lattice_zero_mean_sigma_m,
        },
        "native_tum_joint_slac_spatial_ablation_windows": [
            _spatial_window_provenance(run) for run in spatial_runs
        ],
        "native_tum_joint_slac_spatial_cross_window_policy": (
            "source spatial lattice and extrinsic are evaluated with target optimized "
            "poses and target spatial holdout factors"
        ),
        "native_tum_joint_slac_spatial_cross_window_transfers": list(spatial_transfers),
        "native_tum_joint_slac_xyz_lattice_model": {
            "kind": "shared_full_xyz_trilinear_displacement_field",
            "primary_source": "Zhou and Koltun, CVPR 2014, equations 2-4",
            "lattice_shape": list(_SPATIAL_LATTICE_SHAPE),
            "minimum_sensor_m": list(_SPATIAL_LATTICE_MINIMUM_M),
            "maximum_sensor_m": list(_SPATIAL_LATTICE_MAXIMUM_M),
            "shape_regularizer": (
                "directed neighbor residuals with local rotations frozen within "
                "each optimizer round"
            ),
            "shape_sigma_m": options.xyz_lattice_shape_sigma_m,
            "local_rotation_updates": options.xyz_lattice_local_rotation_updates,
            "local_rotation_estimator": "per-node proper-rotation Procrustes fit",
            "rigid_gauge": ("train-only mean translation plus infinitesimal rotation moment"),
            "translation_gauge_sigma_m": (options.xyz_lattice_translation_gauge_sigma_m),
            "rotation_gauge_sigma_rad": options.xyz_lattice_rotation_gauge_sigma_rad,
            "target_policy": (
                "the fixed-map ablation calibrates only the query side; the separate "
                "xyz_pair model below calibrates both correspondence endpoints"
            ),
        },
        "native_tum_joint_slac_xyz_lattice_windows": [
            _xyz_window_provenance(run) for run in xyz_runs
        ],
        "native_tum_joint_slac_xyz_cross_window_policy": (
            "source shared extrinsic and 24-dimensional full-XYZ field are evaluated "
            "on target final-round holdout factors while retaining target optimized "
            "poses; local-rotation and gauge residuals are excluded"
        ),
        "native_tum_joint_slac_xyz_cross_window_transfers": list(xyz_transfers),
        "native_tum_joint_slac_xyz_pair_model": {
            "kind": "two_sided_query_query_full_xyz_trilinear_field",
            "primary_source": "Zhou and Koltun, CVPR 2014, equation 2",
            "residual": "(T_i C(p) - T_j C(q)) dot n_world",
            "pair_stride": options.xyz_pair_stride,
            "maximum_correspondences_per_pair": (options.xyz_pair_max_correspondences_per_pair),
            "correspondence_policy": (
                "source points transformed by initial T_target_source and associated "
                "to target-local voxel-plane centroids; assignments remain frozen"
            ),
            "normal_policy": "target-local plane normal rotated to world by initial T_world_target",
            "split_policy": (
                "observation groups are complete directed frame-pair edges; held-out "
                "edges are factor-disjoint but may share endpoint frames with train edges"
            ),
            "world_gauge": "first query pose correction fixed to zero",
            "regularization": (
                "train-only ground-truth pose priors, Eq. (4) frozen local rotations, "
                "and full-XYZ rigid-field gauge"
            ),
        },
        "native_tum_joint_slac_xyz_pair_windows": [
            _xyz_pair_window_provenance(run) for run in xyz_pair_runs
        ],
        "native_tum_joint_slac_xyz_pair_cross_window_policy": (
            "source 24-dimensional calibration field is evaluated on target held-out "
            "pair edges while retaining target optimized pose corrections"
        ),
        "native_tum_joint_slac_xyz_pair_cross_window_transfers": list(xyz_pair_transfers),
        "native_tum_joint_slac_iterative_reassociation_policy": (
            "each spatial window alternates nearest voxel-plane reassociation and "
            "warm-start backend-neutral optimization; only train pair Jaccard and "
            "train retention stop fitting; holdout stability is diagnostic"
        ),
        "native_tum_joint_slac_iterative_reassociation": [
            {
                "start_index": run.start_index,
                "baseline_holdout_rmse_m": run.baseline.result.holdout_rmse,
                "result": run.result.as_dict(),
                "reassociation_aware_probe_evaluation": (run.probe_evaluation.as_dict()),
            }
            for run in reassociation_runs
        ],
        "native_tum_joint_slac_rematching_policy": (
            "held-out optimized points are reassociated to the nearest voxel plane "
            "under the unchanged gate; target identity is the integer voxel key"
        ),
        "native_tum_joint_slac_rematching": [
            {
                "start_index": item.start_index,
                "fixed_holdout_rmse_m": item.fixed_holdout_rmse_m,
                "rematched_holdout_rmse_m": item.rematched_holdout_rmse_m,
                "stability": item.stability.as_dict(),
                "curvature_parameter_order": [
                    "extrinsic_tx_m",
                    "extrinsic_ty_m",
                    "extrinsic_tz_m",
                    "extrinsic_rx_rad",
                    "extrinsic_ry_rad",
                    "extrinsic_rz_rad",
                    "constant_depth_bias_m",
                ],
                "fixed_curvature": item.fixed_curvature.as_dict(),
                "rematched_curvature": item.rematched_curvature.as_dict(),
                "relative_hessian_difference": item.relative_hessian_difference,
                "multistart_searches": [search.as_dict() for search in item.multistart_searches],
                "multistart_evaluation": (
                    item.multistart_evaluation.as_dict()
                    if item.multistart_evaluation is not None
                    else None
                ),
            }
            for item in rematching
        ],
    }


def _window_provenance(root: Path, run: _TUMWindowRun) -> dict[str, Any]:
    selected = [*run.problem.map_frames, *run.problem.query_frames]
    extrinsic = se3_from_tangent(run.result.optimized_values[_EXTRINSIC_BLOCK])
    depth_log_scale, depth_bias_m = run.result.optimized_values[_DEPTH_BLOCK]
    return {
        "start_index": run.start_index,
        "status": run.result.status,
        "map_frame_ids": [f"{frame.depth.timestamp_sec:.6f}" for frame in run.problem.map_frames],
        "query_frame_ids": [
            f"{frame.depth.timestamp_sec:.6f}" for frame in run.problem.query_frames
        ],
        "selected_depth_files": [
            {
                "path": str(root / frame.depth.path),
                "sha256": sha256_path(root / frame.depth.path),
                "pose_time_delta_sec": frame.absolute_time_delta_sec,
            }
            for frame in selected
        ],
        "map_record_count": run.problem.map_record_count,
        "map_plane_count": run.problem.map_plane_count,
        "correspondence_count": len(run.problem.measurements),
        "depth_scale": math.exp(depth_log_scale),
        "depth_bias_m": depth_bias_m,
        "identity_reference_translation_error_m": math.dist(
            (0.0, 0.0, 0.0), extrinsic.translation_m
        ),
        "identity_reference_rotation_error_deg": _rotation_angle_deg(extrinsic),
        "solver": run.result.as_dict(),
    }


def _spatial_window_provenance(run: _TUMSpatialWindowRun) -> dict[str, Any]:
    offsets = run.result.optimized_values[_SPATIAL_DEPTH_BLOCK]
    return {
        "start_index": run.start_index,
        "status": run.result.status,
        "constant_bias_m": run.result.optimized_values[_SPATIAL_BIAS_BLOCK][0],
        "zero_mean_lattice_offsets_m": list(offsets),
        "lattice_offset_mean_m": sum(offsets) / len(offsets),
        "lattice_max_abs_contrast_m": max(abs(offset) for offset in offsets),
        "solver": run.result.as_dict(),
    }


def _xyz_window_provenance(run: _TUMXYZWindowRun) -> dict[str, Any]:
    offsets = run.result.optimized_values[_XYZ_LATTICE_BLOCK]
    vectors = [list(offsets[index : index + 3]) for index in range(0, len(offsets), 3)]
    return {
        "start_index": run.start_index,
        "status": run.result.status,
        "xyz_control_offsets_m": vectors,
        "max_abs_offset_component_m": max(abs(value) for value in offsets),
        "local_rotation_history_xyzw": [
            [list(rotation) for rotation in rotations] for rotations in run.local_rotation_history
        ],
        "data_only_field_observability": (run.data_only_field_observability.as_dict()),
        "data_only_shared_observability": (run.data_only_shared_observability.as_dict()),
        "optimizer_rounds": [result.as_dict() for result in run.results],
    }


def _xyz_pair_window_provenance(run: _TUMXYZPairWindowRun) -> dict[str, Any]:
    offsets = run.result.optimized_values[_XYZ_LATTICE_BLOCK]
    holdout_groups = set(run.result.holdout_observation_groups)
    return {
        "start_index": run.start_index,
        "status": run.result.status,
        "query_frame_ids": [
            f"{frame.depth.timestamp_sec:.6f}" for frame in run.problem.query_frames
        ],
        "pair_correspondence_counts": dict(run.problem.pair_counts),
        "fixed_associations": [association.as_dict() for association in run.problem.associations],
        "train_pair_groups": list(run.result.train_observation_groups),
        "holdout_pair_groups": sorted(holdout_groups),
        "train_correspondence_count": sum(
            count for group, count in run.problem.pair_counts if group not in holdout_groups
        ),
        "holdout_correspondence_count": sum(
            count for group, count in run.problem.pair_counts if group in holdout_groups
        ),
        "xyz_control_offsets_m": [
            list(offsets[index : index + 3]) for index in range(0, len(offsets), 3)
        ],
        "max_abs_offset_component_m": max(abs(value) for value in offsets),
        "local_rotation_history_xyzw": [
            [list(rotation) for rotation in rotations] for rotations in run.local_rotation_history
        ],
        "data_only_field_observability": (run.data_only_field_observability.as_dict()),
        "data_only_joint_observability": (run.data_only_joint_observability.as_dict()),
        "optimizer_rounds": [result.as_dict() for result in run.results],
    }


def _intrinsics(config: CalibrationConfig, camera: str) -> TUMDepthIntrinsics:
    sensor = config.sensors[camera]
    values = sensor.intrinsics
    if values is None:
        return TUMDepthIntrinsics()
    return TUMDepthIntrinsics(
        fx=values.fx if values.fx is not None else 525.0,
        fy=values.fy if values.fy is not None else 525.0,
        cx=values.cx if values.cx is not None else 319.5,
        cy=values.cy if values.cy is not None else 239.5,
        depth_scale=5000.0,
    )


def _options(config: CalibrationConfig) -> TUMJointSlacOptions:
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    values: dict[str, Any] = dict(factor.options) if factor is not None else {}
    defaults = TUMJointSlacOptions()
    return TUMJointSlacOptions(
        frame_start_index=_nonnegative_integer(
            values, "frame_start_index", defaults.frame_start_index
        ),
        frame_stride=_integer(values, "frame_stride", defaults.frame_stride),
        map_frame_count=_integer(values, "map_frame_count", defaults.map_frame_count),
        query_frame_count=_integer(values, "query_frame_count", defaults.query_frame_count),
        map_points_per_frame=_integer(
            values, "map_points_per_frame", defaults.map_points_per_frame
        ),
        query_points_per_frame=_integer(
            values, "query_points_per_frame", defaults.query_points_per_frame
        ),
        max_correspondences_per_frame=_integer(
            values,
            "max_correspondences_per_frame",
            defaults.max_correspondences_per_frame,
        ),
        voxel_size_m=_positive(values, "voxel_size_m", defaults.voxel_size_m),
        correspondence_gate_m=_positive(
            values, "correspondence_gate_m", defaults.correspondence_gate_m
        ),
        max_pose_time_delta_sec=_positive(
            values, "max_pose_time_delta_sec", defaults.max_pose_time_delta_sec
        ),
        pose_prior_translation_sigma_m=_positive(
            values,
            "pose_prior_translation_sigma_m",
            defaults.pose_prior_translation_sigma_m,
        ),
        pose_prior_rotation_sigma_rad=math.radians(
            _positive(
                values,
                "pose_prior_rotation_sigma_deg",
                math.degrees(defaults.pose_prior_rotation_sigma_rad),
            )
        ),
        holdout_ratio=min(0.5, _positive(values, "holdout_ratio", defaults.holdout_ratio)),
        huber_delta_m=_positive(values, "huber_delta_m", defaults.huber_delta_m),
        known_bad_translation_m=_positive(
            values, "known_bad_translation_m", defaults.known_bad_translation_m
        ),
        known_bad_rotation_rad=math.radians(
            _positive(
                values,
                "known_bad_rotation_deg",
                math.degrees(defaults.known_bad_rotation_rad),
            )
        ),
        known_bad_margin_m=_positive(values, "known_bad_margin_m", defaults.known_bad_margin_m),
        reference_translation_gate_m=_positive(
            values,
            "reference_translation_gate_m",
            defaults.reference_translation_gate_m,
        ),
        reference_rotation_gate_deg=_positive(
            values, "reference_rotation_gate_deg", defaults.reference_rotation_gate_deg
        ),
        known_bad_depth_log_scale=_positive(
            values, "known_bad_depth_log_scale", defaults.known_bad_depth_log_scale
        ),
        known_bad_depth_bias_m=_positive(
            values, "known_bad_depth_bias_m", defaults.known_bad_depth_bias_m
        ),
        reference_depth_scale_error_gate_percent=_positive(
            values,
            "reference_depth_scale_error_gate_percent",
            defaults.reference_depth_scale_error_gate_percent,
        ),
        reference_depth_bias_gate_m=_positive(
            values, "reference_depth_bias_gate_m", defaults.reference_depth_bias_gate_m
        ),
        replication_start_indices=_nonnegative_integer_tuple(
            values,
            "replication_start_indices",
            defaults.replication_start_indices,
        ),
        cross_window_nondegradation_margin_m=_positive(
            values,
            "cross_window_nondegradation_margin_m",
            defaults.cross_window_nondegradation_margin_m,
        ),
        spatial_lattice_smoothness_sigma_m=_positive(
            values,
            "spatial_lattice_smoothness_sigma_m",
            defaults.spatial_lattice_smoothness_sigma_m,
        ),
        spatial_lattice_zero_mean_sigma_m=_positive(
            values,
            "spatial_lattice_zero_mean_sigma_m",
            defaults.spatial_lattice_zero_mean_sigma_m,
        ),
        reassociation_max_outer_iterations=_integer(
            values,
            "reassociation_max_outer_iterations",
            defaults.reassociation_max_outer_iterations,
        ),
        reassociation_min_train_pair_jaccard=_fraction(
            values,
            "reassociation_min_train_pair_jaccard",
            defaults.reassociation_min_train_pair_jaccard,
        ),
        reassociation_min_train_retained_fraction=_fraction(
            values,
            "reassociation_min_train_retained_fraction",
            defaults.reassociation_min_train_retained_fraction,
        ),
        reassociation_unmatched_residual_penalty_m=_positive(
            values,
            "reassociation_unmatched_residual_penalty_m",
            defaults.reassociation_unmatched_residual_penalty_m,
        ),
        xyz_lattice_shape_sigma_m=_positive(
            values,
            "xyz_lattice_shape_sigma_m",
            defaults.xyz_lattice_shape_sigma_m,
        ),
        xyz_lattice_translation_gauge_sigma_m=_positive(
            values,
            "xyz_lattice_translation_gauge_sigma_m",
            defaults.xyz_lattice_translation_gauge_sigma_m,
        ),
        xyz_lattice_rotation_gauge_sigma_rad=_positive(
            values,
            "xyz_lattice_rotation_gauge_sigma_rad",
            defaults.xyz_lattice_rotation_gauge_sigma_rad,
        ),
        xyz_lattice_local_rotation_updates=_nonnegative_integer(
            values,
            "xyz_lattice_local_rotation_updates",
            defaults.xyz_lattice_local_rotation_updates,
        ),
        xyz_pair_max_correspondences_per_pair=_integer(
            values,
            "xyz_pair_max_correspondences_per_pair",
            defaults.xyz_pair_max_correspondences_per_pair,
        ),
        xyz_pair_stride=_integer(
            values,
            "xyz_pair_stride",
            defaults.xyz_pair_stride,
        ),
    )


def _integer(values: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(values[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if value > 0 else default


def _nonnegative_integer(values: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(values[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _nonnegative_integer_tuple(
    values: dict[str, Any], key: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    raw = values.get(key)
    if not isinstance(raw, (list, tuple)):
        return default
    parsed: list[int] = []
    for item in raw:
        try:
            value = int(item)
        except (TypeError, ValueError):
            return default
        if value < 0:
            return default
        if value not in parsed:
            parsed.append(value)
    return tuple(parsed) if parsed else default


def _positive(values: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if value > 0.0 else default


def _fraction(values: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else default


def _unavailable(reason: str) -> SolverAdapterResult:
    return SolverAdapterResult(
        backend=NATIVE_TUM_JOINT_SLAC_BACKEND,
        available=False,
        status="unavailable",
        metrics={
            "native_tum_joint_slac_available": MetricResult(value=0.0, grade="warn", reason=reason)
        },
        provenance={
            "native_tum_joint_slac_status": "unavailable",
            "native_tum_joint_slac_reason": reason,
        },
        warnings=[reason],
    )
