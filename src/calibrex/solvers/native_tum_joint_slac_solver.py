"""Native multi-capture joint pose/extrinsic adapter for public TUM RGB-D."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.provenance import sha256_path
from calibrex.core.result import Grade, MetricResult, ObservabilityResult
from calibrex.data.inspect import DatasetInspection
from calibrex.data.livox import (
    LivoxPointRecord,
    build_voxel_plane_map,
    nearest_voxel_plane,
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
from calibrex.graph.joint_factors import (
    JointDepthPointToPlaneMeasurement,
    JointTrilinearDepthPointToPlaneMeasurement,
    make_joint_centered_trilinear_depth_point_to_plane_factor,
    make_joint_depth_point_to_plane_factor,
    make_joint_lattice_smoothness_factor,
    make_joint_lattice_zero_mean_factor,
    make_joint_prior_factor,
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
    evaluate_joint_observability,
)
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult

NATIVE_TUM_JOINT_SLAC_BACKEND = "native_tum_joint_slac"
_FACTOR_NAME = "tum_rgbd_joint_point_to_plane"
_EXTRINSIC_BLOCK = "T_trajectory_camera_correction"
_DEPTH_BLOCK = "depth_log_scale_bias_m"
_SPATIAL_BIAS_BLOCK = "depth_spatial_constant_bias_m"
_SPATIAL_DEPTH_BLOCK = "depth_zero_mean_trilinear_ray_offsets_m"
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
        warnings = _warnings(
            result,
            extrinsic_observability.information_rank,
            shared_observability.information_rank,
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
        JointOptimizerOptions(
            max_iterations=max(1, config.solver.max_iterations),
            convergence_tolerance=config.solver.convergence_tolerance,
            huber_delta=options.huber_delta_m,
            holdout_ratio=options.holdout_ratio,
            split_seed=config.solver.seed or 0,
            minimum_train_factors=12,
            known_bad_margin=options.known_bad_margin_m,
            max_condition_number=1.0e12,
        ),
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
    lattice_size = math.prod(_SPATIAL_LATTICE_SHAPE)
    for baseline in baseline_runs:
        factors = tuple(
            make_joint_centered_trilinear_depth_point_to_plane_factor(
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
                    measurement.plane_point_world_m,
                    measurement.plane_normal_world,
                    measurement.transform_world_body_initial,
                    measurement.transform_body_sensor_initial,
                    measurement.weight,
                ),
                pose_block=_pose_block(index, frame),
                extrinsic_block=_EXTRINSIC_BLOCK,
                depth_bias_block=_SPATIAL_BIAS_BLOCK,
                centered_lattice_block=_SPATIAL_DEPTH_BLOCK,
            )
            for index, frame in enumerate(baseline.problem.query_frames)
            for measurement in baseline.problem.measurements
            if measurement.observation_group == f"tum-frame-{index:03d}"
        )
        priors = tuple(
            factor for factor in baseline.problem.all_factors if factor.family == "diagonal_prior"
        )
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
            size=lattice_size,
            sigma_m=options.spatial_lattice_zero_mean_sigma_m,
        )
        blocks = (
            *(block for block in baseline.problem.parameter_blocks if block.name != _DEPTH_BLOCK),
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
        runs.append(
            _TUMSpatialWindowRun(
                baseline.start_index,
                baseline.problem,
                factors,
                _optimize_joint(
                    config,
                    blocks,
                    (*factors, *priors, smoothness, zero_mean),
                    options,
                ),
            )
        )
    return tuple(runs)


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


def _scale_ray(ray: tuple[float, float, float], depth_m: float) -> tuple[float, float, float]:
    return (ray[0] * depth_m, ray[1] * depth_m, ray[2] * depth_m)


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


def _rotation_angle_deg(transform: SE3) -> float:
    return math.degrees(2.0 * math.acos(min(1.0, abs(transform.rotation_quat_xyzw[3]))))


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
) -> dict[str, Any]:
    selected = [*problem.map_frames, *problem.query_frames]
    return {
        "native_tum_joint_slac_method": "tum_multicapture_pose_extrinsic_depth/v0.5",
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
