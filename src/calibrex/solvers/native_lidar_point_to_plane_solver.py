"""Native fixed-rig LiDAR point-to-plane solver adapter.

This adapter wires the pure-Python :class:`LidarRigPointToPlaneFactor` and
:class:`FixedTrajectorySe3ExtrinsicSolver` into the calibration pipeline. It is
selected explicitly with ``solver.backend: native_lidar_point_to_plane`` and
activates on A2D2/PCD pairs and on bounded ROS1 LiDAR topics. The source frame
is treated as the world frame of a fixed-trajectory pair; the ROS1 adapter adds
a temporal source-map boundary and target-window holdout, while the optimized
variable remains the relative extrinsic ``T_source_target``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.core.exceptions import DatasetError
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3, Vector3
from calibrex.core.result import Grade, MetricResult, ObservabilityResult
from calibrex.data.a2d2 import find_a2d2_lidar_npz_files, read_a2d2_lidar_points
from calibrex.data.inspect import DatasetInspection
from calibrex.data.livox import (
    LivoxPointRecord,
    find_livox_pcd_files,
    read_livox_binary_pcd_records,
)
from calibrex.data.ros_messages import LivoxCustomMessage, PointCloud2Message
from calibrex.data.rosbag1 import (
    LIDAR_MESSAGE_TYPES,
    POINTCLOUD2_TYPE,
    decode_bag_lidar_message,
    decode_pointcloud2,
    iter_messages,
)
from calibrex.data.rosbag2 import (
    LIDAR_MESSAGE_TYPES as ROSBAG2_LIDAR_MESSAGE_TYPES,
    decode_lidar_message as decode_rosbag2_lidar_message,
    iter_messages as iter_rosbag2_messages,
)
from calibrex.evaluation.lidar import (
    build_rig_point_to_plane_observations,
    lidar_rig_point_to_plane_metrics_from_evaluation,
)
from calibrex.graph.lidar_point_to_plane import (
    LidarRigPointToPlaneEvaluation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
    FixedTrajectorySe3SolverResult,
    RobustLoss,
)

NATIVE_LIDAR_POINT_TO_PLANE_BACKEND = "native_lidar_point_to_plane"
_FACTOR_NAME = "lidar_rig_point_to_plane"
_SUPPORTED_DATASET_TYPES = ("a2d2_lidar", "livox_pcd", "rosbag1", "rosbag2")
_MIN_OBSERVATIONS = 6
_MIN_HOLDOUT_OBSERVATIONS = 3


@dataclass(frozen=True)
class LidarPairSolveInputs:
    """Resolved geometry and tuning inputs for a native LiDAR pair solve."""

    voxel_size_m: float
    correspondence_gate_m: float
    max_source_points: int | None
    max_target_points: int | None
    max_source_messages: int | None
    max_target_messages: int | None
    max_replay_duration_s: float | None


@dataclass(frozen=True)
class LidarPairData:
    """Source plane records and target query points for one LiDAR pair."""

    source_records: list[LivoxPointRecord]
    target_points: list[Vector3]
    source_path: str
    target_path: str
    target_holdout_points: list[Vector3] | None = None
    source_window_count: int | None = None
    target_train_window_count: int | None = None
    target_holdout_window_count: int | None = None
    holdout_start_timestamp_ns: int | None = None
    temporal_holdout_independent: bool = False
    point_time_observed_by_sensor: dict[str, bool] | None = None
    point_time_reference_by_sensor: dict[str, str] | None = None
    point_time_clock_mapping_by_sensor: dict[str, dict[str, Any]] | None = None
    point_time_deskew_applied: bool = False


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
        files = lidar_pair_dataset_files(config)
        if config.dataset.type not in ("rosbag1", "rosbag2") and len(files) < 2:
            return self._unavailable(
                "native LiDAR point-to-plane requires at least two LiDAR frames"
            )

        source_sensor, target_sensor = _resolve_lidar_pair(frame_graph, lidars)
        source_node = frame_graph.nodes.get(source_sensor)
        target_node = frame_graph.nodes.get(target_sensor)
        if source_node is None or target_node is None or target_node.parent is None:
            return self._unavailable(
                "source and target LiDAR frames must be present in the frame graph"
            )

        inputs = lidar_pair_solve_inputs(config)
        pair = load_lidar_pair(
            config,
            files,
            inputs,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
        )
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

        holdout_metrics, holdout_provenance = _holdout_evidence(
            pair=pair,
            refined_transform=solver_result.refined_transform,
            variable=variable,
            sensor=target_sensor,
            voxel_size_m=inputs.voxel_size_m,
            correspondence_gate_m=inputs.correspondence_gate_m,
            min_detection_fraction=config.evaluation.solid_state.min_known_bad_detectable_fraction,
        )
        metrics.update(holdout_metrics)

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
            )
            | holdout_provenance,
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


def lidar_pair_dataset_files(config: CalibrationConfig) -> list[Path]:
    """Return deterministic LiDAR-pair files supported by native adapters."""

    if config.dataset.type == "a2d2_lidar":
        return find_a2d2_lidar_npz_files(config.dataset.path)
    if config.dataset.type == "livox_pcd":
        return find_livox_pcd_files(config.dataset.path)
    return []


def _resolve_lidar_pair(
    frame_graph: FrameGraph,
    lidars: list[str],
) -> tuple[str, str]:
    """Pick the root-mounted source and its estimated LiDAR child when explicit."""

    root_lidars = [
        name
        for name in lidars
        if frame_graph.nodes.get(name) is not None and frame_graph.nodes[name].root
    ]
    if len(root_lidars) == 1:
        root = root_lidars[0]
        children = [
            name
            for name in lidars
            if name != root
            and frame_graph.nodes.get(name) is not None
            and frame_graph.nodes[name].parent == root
            and frame_graph.nodes[name].estimate
        ]
        if len(children) == 1:
            return root, children[0]
    # Dual-base-link layouts retain the historical deterministic order.
    return lidars[0], lidars[1]


def load_lidar_pair(
    config: CalibrationConfig,
    files: list[Path],
    inputs: LidarPairSolveInputs,
    *,
    source_sensor: str | None = None,
    target_sensor: str | None = None,
) -> LidarPairData:
    """Load the first supported source/target cloud pair without ROS dependencies."""

    if config.dataset.type in ("rosbag1", "rosbag2"):
        if source_sensor is None or target_sensor is None:
            lidars = sorted(
                name for name, sensor in config.sensors.items() if sensor.type == "lidar"
            )
            source_sensor, target_sensor = lidars[0], lidars[1]
        if config.dataset.type == "rosbag2":
            return _load_rosbag2_lidar_pair(
                config,
                inputs,
                source_sensor=source_sensor,
                target_sensor=target_sensor,
            )
        return _load_rosbag1_lidar_pair(
            config,
            inputs,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
        )
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
        return LidarPairData(
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
    return LidarPairData(
        source_records=source_records,
        target_points=[
            (record.point[0], record.point[1], record.point[2])
            for record in target_records
        ],
        source_path=str(source_path),
        target_path=str(target_path),
    )


def _load_rosbag1_lidar_pair(
    config: CalibrationConfig,
    inputs: LidarPairSolveInputs,
    *,
    source_sensor: str,
    target_sensor: str,
) -> LidarPairData:
    """Load a bounded ROS1 pair and reserve later target windows for holdout."""

    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        return LidarPairData(
            source_records=[],
            target_points=[],
            source_path=f"{bag_path}::{source_sensor}",
            target_path=f"{bag_path}::{target_sensor}",
        )
    source_topic = _sensor_topic(config, source_sensor)
    target_topic = _sensor_topic(config, target_sensor)
    source_windows: list[tuple[int, list[Vector3]]] = []
    target_windows: list[tuple[int, list[Vector3]]] = []
    point_time_observed = {source_sensor: False, target_sensor: False}
    point_time_reference: dict[str, str] = {}
    point_time_clock_mapping: dict[str, dict[str, Any]] = {}
    replay_start_ns: int | None = None

    for connection, timestamp_ns, data in iter_messages(
        bag_path, topics={source_topic, target_topic}
    ):
        if connection.message_type not in LIDAR_MESSAGE_TYPES:
            continue
        if connection.topic == source_topic:
            if (
                inputs.max_source_messages is not None
                and len(source_windows) >= inputs.max_source_messages
            ):
                continue
            message = _decode_rosbag1_offline_message(
                config,
                source_sensor,
                connection.message_type,
                timestamp_ns,
                data,
            )
            points = _message_points(message, inputs.max_source_points)
            if points:
                source_windows.append((timestamp_ns, points))
            observed, reference = _message_point_time_semantics(message)
            point_time_observed[source_sensor] |= observed
            if reference is not None:
                point_time_reference[source_sensor] = reference
            _record_livox_clock_mapping(
                point_time_clock_mapping,
                source_sensor,
                message,
                ros_timestamp_ns=timestamp_ns,
            )
            continue
        if connection.topic != target_topic:
            continue
        if replay_start_ns is None:
            replay_start_ns = timestamp_ns
        if (
            inputs.max_replay_duration_s is not None
            and timestamp_ns - replay_start_ns
            > int(inputs.max_replay_duration_s * 1_000_000_000)
        ):
            break
        if (
            inputs.max_target_messages is not None
            and len(target_windows) >= inputs.max_target_messages
        ):
            break
        message = _decode_rosbag1_offline_message(
            config,
            target_sensor,
            connection.message_type,
            timestamp_ns,
            data,
        )
        points = _message_points(message, inputs.max_target_points)
        if points:
            target_windows.append((timestamp_ns, points))
        observed, reference = _message_point_time_semantics(message)
        point_time_observed[target_sensor] |= observed
        if reference is not None:
            point_time_reference[target_sensor] = reference
        _record_livox_clock_mapping(
            point_time_clock_mapping,
            target_sensor,
            message,
            ros_timestamp_ns=timestamp_ns,
        )

    if len(target_windows) >= 2:
        train_window_count = max(
            1,
            min(
                len(target_windows) - 1,
                int(len(target_windows) * (1.0 - config.evaluation.holdout_ratio)),
            ),
        )
        holdout_windows = target_windows[train_window_count:]
        train_windows = target_windows[:train_window_count]
    else:
        train_window_count = len(target_windows)
        train_windows = target_windows
        holdout_windows = []
    holdout_start_ns = holdout_windows[0][0] if holdout_windows else None

    source_train_windows = (
        [
            window
            for window in source_windows
            if holdout_start_ns is None or window[0] < holdout_start_ns
        ]
        if holdout_start_ns is not None
        else source_windows
    )
    temporal_holdout_independent = bool(
        holdout_windows and source_train_windows and holdout_start_ns is not None
    )
    if not source_train_windows and source_windows:
        # Keep the solver diagnosable on recordings whose topics are not
        # interleaved, but do not claim temporal independence in provenance.
        source_train_windows = source_windows
        temporal_holdout_independent = False

    source_records = [
        record
        for _timestamp, points in source_train_windows
        for record in _points_to_records(points)
    ]
    if inputs.max_source_points is not None:
        source_records = _stride(source_records, inputs.max_source_points)
    target_points = [point for _timestamp, points in train_windows for point in points]
    target_holdout_points = [
        point for _timestamp, points in holdout_windows for point in points
    ]
    return LidarPairData(
        source_records=source_records,
        target_points=target_points,
        target_holdout_points=target_holdout_points,
        source_path=f"{bag_path}::{source_topic}",
        target_path=f"{bag_path}::{target_topic}",
        source_window_count=len(source_train_windows),
        target_train_window_count=len(train_windows),
        target_holdout_window_count=len(holdout_windows),
        holdout_start_timestamp_ns=holdout_start_ns,
        temporal_holdout_independent=temporal_holdout_independent,
        point_time_observed_by_sensor=point_time_observed,
        point_time_reference_by_sensor=point_time_reference,
        point_time_clock_mapping_by_sensor=_finalize_livox_clock_mappings(
            point_time_clock_mapping
        ),
        point_time_deskew_applied=False,
    )


def _load_rosbag2_lidar_pair(
    config: CalibrationConfig,
    inputs: LidarPairSolveInputs,
    *,
    source_sensor: str,
    target_sensor: str,
) -> LidarPairData:
    """Load a bounded ROS 2 bag pair and reserve later target windows for holdout.

    Mirrors ``_load_rosbag1_lidar_pair`` but reads rosbag2 sqlite3/mcap containers
    using the pure-Python rosbag2 reader.  No ``rclpy`` or ROS installation is
    required.
    """
    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        return LidarPairData(
            source_records=[],
            target_points=[],
            source_path=f"{bag_path}::{source_sensor}",
            target_path=f"{bag_path}::{target_sensor}",
        )
    source_topic = _sensor_topic_rosbag2(config, source_sensor)
    target_topic = _sensor_topic_rosbag2(config, target_sensor)
    source_sensor_cfg = config.sensors.get(source_sensor)
    target_sensor_cfg = config.sensors.get(target_sensor)
    source_point_time_field = (
        source_sensor_cfg.point_time_field if source_sensor_cfg is not None else None
    )
    target_point_time_field = (
        target_sensor_cfg.point_time_field if target_sensor_cfg is not None else None
    )
    source_windows: list[tuple[int, list[Vector3]]] = []
    target_windows: list[tuple[int, list[Vector3]]] = []
    point_time_observed = {source_sensor: False, target_sensor: False}
    point_time_reference: dict[str, str] = {}
    point_time_clock_mapping: dict[str, dict[str, Any]] = {}
    replay_start_ns: int | None = None

    for connection, timestamp_ns, data in iter_rosbag2_messages(
        bag_path, topics={source_topic, target_topic}
    ):
        if connection.message_type not in ROSBAG2_LIDAR_MESSAGE_TYPES:
            continue
        if connection.topic == source_topic:
            if (
                inputs.max_source_messages is not None
                and len(source_windows) >= inputs.max_source_messages
            ):
                continue
            message = decode_rosbag2_lidar_message(
                source_topic,
                connection.message_type,
                timestamp_ns,
                data,
                point_time_field=source_point_time_field,
            )
            points = _message_points(message, inputs.max_source_points)
            if points:
                source_windows.append((timestamp_ns, points))
            observed, reference = _message_point_time_semantics(message)
            point_time_observed[source_sensor] |= observed
            if reference is not None:
                point_time_reference[source_sensor] = reference
            _record_livox_clock_mapping(
                point_time_clock_mapping,
                source_sensor,
                message,
                ros_timestamp_ns=timestamp_ns,
            )
            continue
        if connection.topic != target_topic:
            continue
        if replay_start_ns is None:
            replay_start_ns = timestamp_ns
        if (
            inputs.max_replay_duration_s is not None
            and timestamp_ns - replay_start_ns
            > int(inputs.max_replay_duration_s * 1_000_000_000)
        ):
            break
        if (
            inputs.max_target_messages is not None
            and len(target_windows) >= inputs.max_target_messages
        ):
            break
        message = decode_rosbag2_lidar_message(
            target_topic,
            connection.message_type,
            timestamp_ns,
            data,
            point_time_field=target_point_time_field,
        )
        points = _message_points(message, inputs.max_target_points)
        if points:
            target_windows.append((timestamp_ns, points))
        observed, reference = _message_point_time_semantics(message)
        point_time_observed[target_sensor] |= observed
        if reference is not None:
            point_time_reference[target_sensor] = reference
        _record_livox_clock_mapping(
            point_time_clock_mapping,
            target_sensor,
            message,
            ros_timestamp_ns=timestamp_ns,
        )

    if len(target_windows) >= 2:
        train_window_count = max(
            1,
            min(
                len(target_windows) - 1,
                int(len(target_windows) * (1.0 - config.evaluation.holdout_ratio)),
            ),
        )
        holdout_windows = target_windows[train_window_count:]
        train_windows = target_windows[:train_window_count]
    else:
        train_window_count = len(target_windows)
        train_windows = target_windows
        holdout_windows = []
    holdout_start_ns = holdout_windows[0][0] if holdout_windows else None

    source_train_windows = (
        [
            window
            for window in source_windows
            if holdout_start_ns is None or window[0] < holdout_start_ns
        ]
        if holdout_start_ns is not None
        else source_windows
    )
    temporal_holdout_independent = bool(
        holdout_windows and source_train_windows and holdout_start_ns is not None
    )
    if not source_train_windows and source_windows:
        source_train_windows = source_windows
        temporal_holdout_independent = False

    source_records = [
        record
        for _timestamp, points in source_train_windows
        for record in _points_to_records(points)
    ]
    if inputs.max_source_points is not None:
        source_records = _stride(source_records, inputs.max_source_points)
    target_points = [point for _timestamp, points in train_windows for point in points]
    target_holdout_points = [
        point for _timestamp, points in holdout_windows for point in points
    ]
    return LidarPairData(
        source_records=source_records,
        target_points=target_points,
        target_holdout_points=target_holdout_points,
        source_path=f"{bag_path}::{source_topic}",
        target_path=f"{bag_path}::{target_topic}",
        source_window_count=len(source_train_windows),
        target_train_window_count=len(train_windows),
        target_holdout_window_count=len(holdout_windows),
        holdout_start_timestamp_ns=holdout_start_ns,
        temporal_holdout_independent=temporal_holdout_independent,
        point_time_observed_by_sensor=point_time_observed,
        point_time_reference_by_sensor=point_time_reference,
        point_time_clock_mapping_by_sensor=_finalize_livox_clock_mappings(
            point_time_clock_mapping
        ),
        point_time_deskew_applied=False,
    )


def _sensor_topic(config: CalibrationConfig, sensor_name: str) -> str:
    sensor = config.sensors.get(sensor_name)
    if sensor is None or not sensor.topic:
        raise DatasetError(f"ROS1 LiDAR sensor {sensor_name!r} has no topic")
    return sensor.topic


def _sensor_topic_rosbag2(config: CalibrationConfig, sensor_name: str) -> str:
    sensor = config.sensors.get(sensor_name)
    if sensor is None or not sensor.topic:
        raise DatasetError(f"ROS2 LiDAR sensor {sensor_name!r} has no topic")
    return sensor.topic


def _decode_rosbag1_offline_message(
    config: CalibrationConfig,
    sensor_name: str,
    message_type: str,
    timestamp_ns: int,
    data: bytes,
) -> LivoxCustomMessage | PointCloud2Message:
    sensor = config.sensors.get(sensor_name)
    point_time_field = sensor.point_time_field if sensor is not None else None
    if message_type == POINTCLOUD2_TYPE:
        return decode_pointcloud2(
            _sensor_topic(config, sensor_name),
            timestamp_ns,
            data,
            point_time_field=point_time_field,
        )
    return decode_bag_lidar_message(
        _sensor_topic(config, sensor_name), message_type, timestamp_ns, data
    )


def _message_points(
    message: LivoxCustomMessage | PointCloud2Message,
    max_points: int | None,
) -> list[Vector3]:
    xyz = message.xyz
    count = int(xyz.shape[0])
    if max_points is None or max_points <= 0 or count <= max_points:
        indices = range(count)
    else:
        step = (count + max_points - 1) // max_points
        indices = range(0, count, step)
    return [
        (float(xyz[index, 0]), float(xyz[index, 1]), float(xyz[index, 2]))
        for index in indices
    ]


def _points_to_records(points: list[Vector3]) -> list[LivoxPointRecord]:
    return [LivoxPointRecord(point=(x, y, z, 0.0), normal_xyz=None) for x, y, z in points]


def _message_point_time_semantics(
    message: LivoxCustomMessage | PointCloud2Message,
) -> tuple[bool, str | None]:
    if isinstance(message, LivoxCustomMessage):
        return (
            message.offset_time_ns is not None,
            "timebase_mapped_to_ros_bag_timestamp",
        )
    return (
        message.point_time_offsets_s is not None,
        "message_stamp_mapped_to_ros_bag_timestamp",
    )


def _record_livox_clock_mapping(
    mappings: dict[str, dict[str, Any]],
    sensor_name: str,
    message: LivoxCustomMessage | PointCloud2Message,
    *,
    ros_timestamp_ns: int,
) -> None:
    """Record the raw Livox clock to ROS bag timestamp relationship."""

    if not isinstance(message, LivoxCustomMessage) or message.timebase_ns <= 0:
        return
    offset_ns = int(ros_timestamp_ns - message.timebase_ns)
    entry = mappings.setdefault(
        sensor_name,
        {
            "source": "rosbag_record_timestamp_minus_livox_timebase",
            "raw_reference": "Livox CustomMsg timebase",
            "mapped_reference": "ROS bag record timestamp",
            "mapping_policy": (
                "capture_ns = bag_record_timestamp_ns + offset_time_ns; "
                "timebase is retained for clock-consistency diagnostics"
            ),
            "sample_count": 0,
            "offset_min_ns": offset_ns,
            "offset_max_ns": offset_ns,
            "nominal_offset_ns": offset_ns,
            "residual_max_abs_ns": 0,
        },
    )
    entry["sample_count"] = int(entry["sample_count"]) + 1
    entry["offset_min_ns"] = min(int(entry["offset_min_ns"]), offset_ns)
    entry["offset_max_ns"] = max(int(entry["offset_max_ns"]), offset_ns)
    residual = abs(offset_ns - int(entry["nominal_offset_ns"]))
    entry["residual_max_abs_ns"] = max(
        int(entry["residual_max_abs_ns"]), residual
    )


def _finalize_livox_clock_mappings(
    mappings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return clock mappings with a conservative stability status."""

    finalized: dict[str, dict[str, Any]] = {}
    for sensor_name, raw in mappings.items():
        item = dict(raw)
        residual_ns = int(item["residual_max_abs_ns"])
        item["residual_max_abs_s"] = residual_ns * 1.0e-9
        item["status"] = "stable" if residual_ns <= 20_000_000 else "unstable"
        finalized[sensor_name] = item
    return finalized


def _stride(records: list[LivoxPointRecord], max_records: int | None) -> list[LivoxPointRecord]:
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records
    step = (len(records) + max_records - 1) // max_records
    return records[::step]


def lidar_pair_solve_inputs(config: CalibrationConfig) -> LidarPairSolveInputs:
    """Resolve shared native LiDAR-pair frontend options from the config."""

    factor = config.pipeline.factors.get(_FACTOR_NAME)
    options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    return LidarPairSolveInputs(
        voxel_size_m=_float_option(options, "voxel_size_m", default=1.0, minimum=0.05),
        correspondence_gate_m=_float_option(
            options, "correspondence_gate_m", default=1.5, minimum=0.05
        ),
        max_source_points=_int_option(options, "max_source_points", default=None),
        max_target_points=_int_option(options, "max_target_points", default=2000),
        max_source_messages=_int_option(options, "max_source_messages", default=None),
        max_target_messages=_int_option(options, "max_target_messages", default=None),
        max_replay_duration_s=_optional_float_option(
            options, "max_replay_duration_s", minimum=0.0
        ),
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


def _optional_float_option(
    options: dict[str, Any],
    key: str,
    *,
    minimum: float,
) -> float | None:
    if key not in options or options[key] is None:
        return None
    try:
        value = float(options[key])
    except (TypeError, ValueError):
        return None
    return value if value >= minimum else None


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


def _holdout_evidence(
    *,
    pair: LidarPairData,
    refined_transform: SE3,
    variable: str,
    sensor: str,
    voxel_size_m: float,
    correspondence_gate_m: float,
    min_detection_fraction: float,
) -> tuple[dict[str, MetricResult], dict[str, Any]]:
    """Score held-out ROS1 target windows and fixed known-bad probes."""

    is_windowed_pair = pair.target_holdout_window_count is not None
    if not is_windowed_pair:
        return {}, {}
    holdout_points = pair.target_holdout_points or []
    evidence: dict[str, Any] = {
        "solid_state_evaluation": {
            "independent_holdout": bool(pair.temporal_holdout_independent),
            "temporal_holdout_independent": bool(pair.temporal_holdout_independent),
            "split_policy": (
                "ROS1 source-map messages before target holdout boundary; "
                "later target capture windows reserved for evaluation"
            ),
            "source_map_window_count": pair.source_window_count,
            "train_window_count": pair.target_train_window_count,
            "holdout_window_count": pair.target_holdout_window_count,
            "holdout_start_timestamp_ns": pair.holdout_start_timestamp_ns,
            "known_bad_case_count": 0,
            "known_bad_evaluation_status": "not_yet_run",
        }
    }
    if len(holdout_points) == 0:
        reason = "ROS1 target stream has no reserved holdout points"
        evidence["solid_state_evaluation"].update(
            {"known_bad_evaluation_status": "unavailable", "reason": reason}
        )
        return (
            {
                "lidar_pair_holdout_point_to_plane_rmse_m": MetricResult(
                    value=None, unit="m", grade="warn", reason=reason
                ),
                "lidar_pair_known_bad_case_count": MetricResult(
                    value=None, unit="cases", grade="warn", reason=reason
                ),
                "lidar_pair_known_bad_detectable_fraction": MetricResult(
                    value=None, grade="warn", reason=reason
                ),
            },
            evidence,
        )

    observations = build_rig_point_to_plane_observations(
        source_records=pair.source_records,
        target_points=holdout_points,
        initial_t_source_target=refined_transform,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
    )
    if len(observations) < _MIN_HOLDOUT_OBSERVATIONS:
        reason = (
            "ROS1 target holdout has too few point-to-plane correspondences "
            f"({len(observations)} < {_MIN_HOLDOUT_OBSERVATIONS})"
        )
        evidence["solid_state_evaluation"].update(
            {
                "known_bad_evaluation_status": "inconclusive",
                "holdout_correspondence_count": len(observations),
                "reason": reason,
            }
        )
        return (
            {
                "lidar_pair_holdout_point_to_plane_rmse_m": MetricResult(
                    value=None, unit="m", grade="warn", reason=reason
                ),
                "lidar_pair_known_bad_case_count": MetricResult(
                    value=None, unit="cases", grade="warn", reason=reason
                ),
                "lidar_pair_known_bad_detectable_fraction": MetricResult(
                    value=None, grade="warn", reason=reason
                ),
            },
            evidence,
        )

    baseline_factor = LidarRigPointToPlaneFactor(
        variable=variable,
        t_ego_lidar=refined_transform,
        observations=observations,
        sensor=sensor,
    )
    baseline_residuals = baseline_factor.residuals()
    baseline_rmse = _rmse(baseline_residuals)
    baseline_p90 = _p90(baseline_residuals)
    cases: list[dict[str, Any]] = []
    detected = 0
    for dof, candidate in _known_bad_candidates(refined_transform):
        candidate_factor = LidarRigPointToPlaneFactor(
            variable=variable,
            t_ego_lidar=candidate,
            observations=observations,
            sensor=sensor,
        )
        candidate_residuals = candidate_factor.residuals()
        candidate_rmse = _rmse(candidate_residuals)
        candidate_p90 = _p90(candidate_residuals)
        worsened = (
            baseline_rmse is not None
            and candidate_rmse is not None
            and candidate_rmse > baseline_rmse + 1.0e-3
        ) or (
            baseline_p90 is not None
            and candidate_p90 is not None
            and candidate_p90 > baseline_p90 + 1.0e-3
        )
        detected += int(worsened)
        cases.append(
            {
                "dof": dof,
                "baseline_rmse_m": baseline_rmse,
                "candidate_rmse_m": candidate_rmse,
                "baseline_p90_m": baseline_p90,
                "candidate_p90_m": candidate_p90,
                "detected": worsened,
            }
        )
    detectable_fraction = detected / len(cases) if cases else None
    grade: Grade = (
        "pass"
        if detectable_fraction is not None
        and detectable_fraction >= min_detection_fraction
        else "warn"
    )
    evidence["solid_state_evaluation"].update(
        {
            "known_bad_evaluation_status": "scored",
            "holdout_correspondence_count": len(observations),
            "known_bad_case_count": len(cases),
            "known_bad_detectable_fraction": detectable_fraction,
        }
    )
    evidence["native_lidar_point_to_plane_holdout"] = {
        "holdout_correspondence_count": len(observations),
        "baseline_rmse_m": baseline_rmse,
        "baseline_p90_m": baseline_p90,
        "known_bad_cases": cases,
    }
    return (
        {
            "lidar_pair_holdout_point_to_plane_rmse_m": MetricResult(
                value=baseline_rmse,
                unit="m",
                grade="pass" if baseline_rmse is not None else "warn",
                reason=(
                    "final offline extrinsic scored on target capture windows held "
                    "out from the source-map time boundary"
                ),
            ),
            "lidar_pair_known_bad_case_count": MetricResult(
                value=float(len(cases)),
                unit="cases",
                grade=grade,
                reason="six fixed roll/pitch/yaw/x/y/z perturbations scored on ROS1 holdout",
            ),
            "lidar_pair_known_bad_detectable_fraction": MetricResult(
                value=detectable_fraction,
                grade=grade,
                reason=(
                    "fraction of six-DoF known-bad perturbations whose ROS1 holdout "
                    f"RMSE or P90 increased; target >= {min_detection_fraction:g}"
                ),
            ),
        },
        evidence,
    )


def _known_bad_candidates(base: SE3) -> list[tuple[str, SE3]]:
    candidates: list[tuple[str, SE3]] = []
    for index, dof in enumerate(("x", "y", "z")):
        translation = [0.0, 0.0, 0.0]
        translation[index] = 0.1
        translation_tuple = (translation[0], translation[1], translation[2])
        candidates.append(
            (dof, SE3(translation_tuple, (0.0, 0.0, 0.0, 1.0)).compose(base))
        )
    for index, dof in enumerate(("roll", "pitch", "yaw")):
        rotation = [0.0, 0.0, 0.0]
        rotation[index] = math.radians(5.0)
        angle = rotation[index]
        half = angle * 0.5
        quaternion = [0.0, 0.0, 0.0, math.cos(half)]
        quaternion[index] = math.sin(half)
        quaternion_tuple = (
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
        )
        candidates.append(
            (dof, SE3((0.0, 0.0, 0.0), quaternion_tuple).compose(base))
        )
    return candidates


def _rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(abs(value) for value in values)
    index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ordered[index]


def _provenance(
    *,
    variable: str,
    source_sensor: str,
    target_sensor: str,
    pair: LidarPairData,
    inputs: LidarPairSolveInputs,
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
        "native_lidar_point_to_plane_source_window_count": pair.source_window_count,
        "native_lidar_point_to_plane_target_train_window_count": (
            pair.target_train_window_count
        ),
        "native_lidar_point_to_plane_target_holdout_window_count": (
            pair.target_holdout_window_count
        ),
        "native_lidar_point_to_plane_holdout_start_timestamp_ns": (
            pair.holdout_start_timestamp_ns
        ),
        "point_time_observed_by_sensor": pair.point_time_observed_by_sensor,
        "point_time_reference_by_sensor": pair.point_time_reference_by_sensor,
        "point_time_clock_mapping_by_sensor": pair.point_time_clock_mapping_by_sensor,
        "point_time_deskew_applied": pair.point_time_deskew_applied,
        "native_lidar_point_to_plane_convention": (
            "world=source LiDAR frame; T_world_ego=identity; optimized variable is "
            "T_source_target with left-SE(3) correction [x,y,z,roll,pitch,yaw]"
        ),
        "native_lidar_point_to_plane_solver": solver_result.as_dict(),
    }
