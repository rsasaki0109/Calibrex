"""Native multi-capture joint pose/extrinsic adapter for public TUM RGB-D."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneEvaluation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult

NATIVE_TUM_JOINT_SLAC_BACKEND = "native_tum_joint_slac"
_FACTOR_NAME = "tum_rgbd_joint_point_to_plane"
_EXTRINSIC_BLOCK = "T_trajectory_camera_correction"
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

    def as_dict(self) -> dict[str, float | int]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if isinstance(value, (float, int))
        }


@dataclass(frozen=True)
class _TUMJointProblem:
    map_frames: tuple[TUMDepthPoseEntry, ...]
    query_frames: tuple[TUMDepthPoseEntry, ...]
    measurements: tuple[JointPointToPlaneMeasurement, ...]
    data_factors: tuple[JointResidualBlock, ...]
    parameter_blocks: tuple[JointParameterBlock, ...]
    all_factors: tuple[JointResidualBlock, ...]
    map_record_count: int
    map_plane_count: int


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
        cameras = sorted(
            name for name, sensor in config.sensors.items() if sensor.type == "camera"
        )
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

        result = BackendNeutralJointOptimizer().solve(
            problem.parameter_blocks,
            problem.all_factors,
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
        extrinsic = se3_from_tangent(result.optimized_values[_EXTRINSIC_BLOCK])
        variable = f"T_{camera_node.parent}_{camera}"
        data_observability = _data_only_extrinsic_observability(
            problem, result, extrinsic, variable, camera
        )
        metrics = _metrics(problem, result, extrinsic, data_observability, options)
        warnings = _warnings(result, data_observability.rank)
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
    indexes = [
        options.frame_start_index + index * options.frame_stride for index in range(count)
    ]
    if not indexes or indexes[-1] >= len(paired):
        return None
    return tuple(paired[index] for index in indexes)


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
    measurements: list[JointPointToPlaneMeasurement] = []
    factors: list[JointResidualBlock] = []
    blocks: list[JointParameterBlock] = []
    priors: list[JointResidualBlock] = []
    pose_sigma = (
        (options.pose_prior_translation_sigma_m,) * 3
        + (options.pose_prior_rotation_sigma_rad,) * 3
    )
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
        candidates: list[JointPointToPlaneMeasurement] = []
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
                JointPointToPlaneMeasurement(
                    measurement_id=f"tum-{frame_index:03d}-{point_index:05d}",
                    observation_group=f"tum-frame-{frame_index:03d}",
                    point_sensor_m=point,
                    plane_point_world_m=plane.centroid,
                    plane_normal_world=plane.normal,
                    transform_world_body_initial=frame.transform_world_camera,
                    transform_body_sensor_initial=SE3.identity(),
                )
            )
        retained = _bounded_sample(candidates, options.max_correspondences_per_frame)
        measurements.extend(retained)
        factors.extend(
            make_joint_point_to_plane_factor(
                measurement,
                pose_block=pose_block,
                extrinsic_block=_EXTRINSIC_BLOCK,
            )
            for measurement in retained
        )
    known_bad = (
        (options.known_bad_translation_m,) * 3
        + (options.known_bad_rotation_rad,) * 3
    )
    blocks.append(
        JointParameterBlock(
            _EXTRINSIC_BLOCK,
            (0.0,) * 6,
            finite_difference_steps=(1.0e-4,) * 3 + (1.0e-5,) * 3,
            known_bad_steps=known_bad,
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
    measurements: list[JointPointToPlaneMeasurement], maximum: int
) -> list[JointPointToPlaneMeasurement]:
    if len(measurements) <= maximum:
        return measurements
    step = math.ceil(len(measurements) / maximum)
    return measurements[::step][:maximum]


def _pose_block(index: int, frame: TUMDepthPoseEntry) -> str:
    return f"pose_{index:03d}_{frame.depth.timestamp_sec:.6f}"


def _data_only_extrinsic_observability(
    problem: _TUMJointProblem,
    result: JointOptimizerResult,
    extrinsic: SE3,
    variable: str,
    camera: str,
) -> LidarRigPointToPlaneEvaluation:
    train_ids = set(result.train_factor_ids)
    observations: list[LidarPointToPlaneObservation] = []
    for measurement in problem.measurements:
        if measurement.measurement_id not in train_ids:
            continue
        frame_index = int(measurement.observation_group.rsplit("-", 1)[1])
        frame = problem.query_frames[frame_index]
        pose = se3_from_tangent(
            result.optimized_values[_pose_block(frame_index, frame)]
        ).compose(frame.transform_world_camera)
        observations.append(
            LidarPointToPlaneObservation(
                point_lidar_m=measurement.point_sensor_m,
                plane_point_world_m=measurement.plane_point_world_m,
                plane_normal_world=measurement.plane_normal_world,
                t_world_ego=pose,
                weight=measurement.weight,
            )
        )
    return LidarRigPointToPlaneFactor(
        variable=variable,
        t_ego_lidar=extrinsic,
        observations=observations,
        sensor=camera,
    ).evaluate()


def _metrics(
    problem: _TUMJointProblem,
    result: JointOptimizerResult,
    extrinsic: SE3,
    data_observability: LidarRigPointToPlaneEvaluation,
    options: TUMJointSlacOptions,
) -> dict[str, MetricResult]:
    train_ids = set(result.train_factor_ids)
    train_factors = [
        factor for factor in problem.data_factors if factor.factor_id in train_ids
    ]
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
    expected_rank = 6 * (len(problem.query_frames) + 1)
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
            value=float(data_observability.rank),
            grade="pass" if data_observability.rank == 6 else "warn",
            reason="shared-extrinsic rank from train geometry with optimized poses fixed",
        ),
        "tum_joint_data_only_extrinsic_condition_number": MetricResult(
            value=data_observability.normalized_condition_number_estimate,
            grade=(
                "pass"
                if data_observability.normalized_condition_number_estimate is not None
                and data_observability.rank == 6
                else "warn"
            ),
            reason="normalized fixed-pose train curvature diagnostic; not covariance",
        ),
        "tum_joint_known_bad_detectable_fraction": MetricResult(
            value=detectable,
            grade="pass" if detectable == 1.0 else "warn",
            reason="signed shared-extrinsic perturbations scored on held-out query frames",
        ),
        "tum_joint_identity_reference_translation_error_m": MetricResult(
            value=translation_error,
            unit="m",
            grade=reference_grade,
            reason=(
                "TUM ground truth already describes the camera frame; "
                "mounting truth is identity"
            ),
        ),
        "tum_joint_identity_reference_rotation_error_deg": MetricResult(
            value=rotation_error,
            unit="deg",
            grade=reference_grade,
            reason=(
                "TUM ground truth already describes the camera frame; "
                "mounting truth is identity"
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


def _rotation_angle_deg(transform: SE3) -> float:
    return math.degrees(
        2.0 * math.acos(min(1.0, abs(transform.rotation_quat_xyzw[3])))
    )


def _warnings(result: JointOptimizerResult, data_rank: int) -> list[str]:
    warnings = [
        "joint rank includes independently measured TUM pose priors; use the separate "
        "data-only shared-extrinsic rank for geometric observability"
    ]
    if result.status != "converged":
        warnings.append(f"native TUM joint SLAC status is {result.status}: {result.reason}")
    if data_rank < 6:
        warnings.append(f"TUM train geometry shared-extrinsic rank is {data_rank} < 6")
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
) -> dict[str, Any]:
    selected = [*problem.map_frames, *problem.query_frames]
    return {
        "native_tum_joint_slac_method": "tum_multicapture_pose_extrinsic/v0.1",
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
            "the independently measured mounting reference"
        ),
        "native_tum_joint_slac_solver": result.as_dict(),
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
        holdout_ratio=min(
            0.5, _positive(values, "holdout_ratio", defaults.holdout_ratio)
        ),
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
        known_bad_margin_m=_positive(
            values, "known_bad_margin_m", defaults.known_bad_margin_m
        ),
        reference_translation_gate_m=_positive(
            values,
            "reference_translation_gate_m",
            defaults.reference_translation_gate_m,
        ),
        reference_rotation_gate_deg=_positive(
            values, "reference_rotation_gate_deg", defaults.reference_rotation_gate_deg
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
            "native_tum_joint_slac_available": MetricResult(
                value=0.0, grade="warn", reason=reason
            )
        },
        provenance={
            "native_tum_joint_slac_status": "unavailable",
            "native_tum_joint_slac_reason": reason,
        },
        warnings=[reason],
    )
