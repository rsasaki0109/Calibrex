"""Online/streaming LiDAR-pair extrinsic calibration.

This module drives the same native fixed-trajectory point-to-plane primitives
used by the offline path (`calibrex.solvers.native_lidar_point_to_plane_solver`)
from a stream of incremental point batches instead of one offline solve.

Motion compensation (optional ``dataset.odometry_topic`` on rosbag replays)
uses per-message rig poses by default. When a participating LiDAR sensor sets
``point_time_field`` and odometry is configured, per-point deskew applies capture
times ``t_i = message_timestamp + offset_i`` (offsets in seconds relative to
the header stamp) for rosbag2 replay.

Formulation when odometry is configured:

* Let ``T_world_base(t)`` be the interpolated odometry pose, ``T_base_source`` the
  configured mount of the source (map-building) sensor, and ``T_source_target``
  the extrinsic being estimated.
* Source map: each source message contributes points transformed into the world
  frame. Without deskew, ``p_world = T_world_base(t_msg) * T_base_source * p_sensor``.
  With deskew, each point uses its capture time:
  ``p_world = T_world_base(t_i) * T_base_source * p_sensor`` where
  ``t_i = message_timestamp + offset_i``.
* Target batches: each target point keeps its sensor-frame coordinates. Without
  deskew the per-message pose ``T_world_source(t_msg)`` is recorded; with deskew
  each point carries ``T_world_source(t_j_point)`` at its capture time.
* Correspondence applies ``T_world_source(t_j) * T_hat_source_target`` to map a
  target point against world-frame planes while the solver still estimates a
  single constant ``T_source_target``. When no odometry topic is configured the
  static-rig path is unchanged (identity frame poses).

Per batch, `OnlineCalibrationSession` warm-starts the native point-to-plane
solve from the previous *accepted* estimate, splits the batch into a
deterministic train/holdout split (reusing `calibrex.evaluation.holdout`),
and evaluates a holdout gate mirroring the pass/fail/inconclusive semantics of
`calibrex.core.assessment`. Only a batch whose gate PASSES is adopted (the
tentative estimate becomes the running estimate and its holdout residuals feed
the rolling window). A batch that FAILS the gate (an absolute holdout RMSE
spike, a sharp regression against the rolling baseline, or rank deficiency when
``accumulation_batches == 1``) is rejected and never enters the retention
buffer. A batch whose evidence is INCONCLUSIVE because of too few train or
holdout correspondences is likewise not adopted and not retained. When
``accumulation_batches > 1``, a batch whose *only* deficiency is accumulated
observability (rank below ``min_rank`` or weak DoF) but whose holdout RMSE is
acceptable on the observable directions becomes inconclusive with retention: its
train points enter the buffer so complementary later views can bootstrap full
rank, while the running estimate and rolling window stay unchanged until a
later batch passes.

`run_online_calibration` is the CLI/Python entry point: it loads the same
`CalibrationConfig` YAML used by `solver.backend: native_lidar_point_to_plane`,
replays the dataset's LiDAR frames as an ordered point stream, and produces a
standard `CalibrationResult` (`producer=calibrex_native`,
`execution_mode=online_stream`) plus a machine-readable per-batch timeline
artifact (`calibrex.core.online_timeline`).
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calibrex import __version__
from calibrex.core.config import CalibrationConfig, load_config
from calibrex.core.exceptions import ConfigError, DatasetError
from calibrex.core.frames import FrameGraph, FrameNode
from calibrex.core.geometry import SE3, Vector3
from calibrex.core.io import write_mapping
from calibrex.core.online_timeline import (
    ONLINE_TIMELINE_SCHEMA_VERSION,
    OnlineBatchSnapshot,
    OnlineCalibrationTimelineArtifact,
    OnlineGateStatus,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.report_artifacts import ReportRunInfo
from calibrex.core.result import (
    CalibrationResult,
    DegeneracyResult,
    Grade,
    MetricResult,
    ObservabilityResult,
    QualitySummary,
    RunInfo,
    TransformEstimateProvenance,
    TransformQuality,
    TransformResult,
)
from calibrex.data.a2d2 import find_a2d2_lidar_npz_files, read_a2d2_lidar_points
from calibrex.data.livox import (
    LivoxPointRecord,
    find_livox_pcd_files,
    read_livox_binary_pcd_records,
)
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.data.rosbag1 import (
    LIDAR_MESSAGE_TYPES,
    decode_bag_lidar_message,
    iter_messages,
)
from calibrex.data.rosbag2 import (
    ODOMETRY_TYPE,
    POINTCLOUD2_TYPE,
    decode_odometry,
    decode_pointcloud2,
)
from calibrex.data.rosbag2 import (
    iter_messages as iter_rosbag2_messages,
)
from calibrex.evaluation.holdout import split_indices
from calibrex.evaluation.lidar import build_rig_point_to_plane_observations
from calibrex.evaluation.metrics import evaluate_quality
from calibrex.graph.lidar_point_to_plane import (
    LidarRigPointToPlaneEvaluation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
    RobustLoss,
)
from calibrex.visualization.report import write_report_artifacts

ONLINE_LIDAR_POINT_TO_PLANE_BACKEND = "online_lidar_point_to_plane"
_SUPPORTED_DATASET_TYPES = ("a2d2_lidar", "livox_pcd", "rosbag1", "rosbag2")
_FACTOR_NAME = "lidar_rig_point_to_plane"
_MIN_TRAIN_OBSERVATIONS = 6
_MIN_HOLDOUT_OBSERVATIONS = 3
_TIMELINE_FILENAME = "timeline.json"
OnlineTargetEntry = tuple[int, Vector3, SE3, float]


@dataclass(frozen=True)
class OnlineGateThresholds:
    """Thresholds controlling the per-batch holdout gate."""

    min_rank: int = 6
    max_holdout_rmse_m: float = 0.05
    max_rolling_regression_m: float = 0.02
    max_odometry_extrapolation_s: float = 0.25


class OnlineCalibrationSession:
    """Incremental online/streaming point-to-plane calibration session.

    Consumes ordered LiDAR point batches against a fixed source-frame plane
    map (built once from ``source_records``), warm-starting the native
    fixed-trajectory solve from the previously *accepted* estimate. Each
    batch is scored with a deterministic holdout split; only a batch whose
    holdout gate passes updates ``current_estimate`` and the rolling residual
    window. Batches that fail the gate (bad holdout evidence, or rank
    deficiency when ``accumulation_batches == 1``) leave both untouched and
    are excluded from the retention buffer. Batches that are inconclusive
    because of too little train/holdout evidence to score are likewise
    non-destructive and not retained. When ``accumulation_batches > 1``,
    observability-only inconclusive batches retain their train points for later
    complementary views while leaving the estimate and rolling window
    unchanged. All batches are recorded in ``history`` so operators can see
    what was proposed and why it was or was not adopted.
    """

    def __init__(
        self,
        *,
        variable: str,
        parent: str,
        sensor: str,
        source_records: list[LivoxPointRecord],
        initial_transform: SE3,
        voxel_size_m: float = 1.0,
        correspondence_gate_m: float = 1.5,
        holdout_ratio: float = 0.2,
        rolling_window: int = 2000,
        seed: int = 0,
        solver_options: FixedTrajectorySe3SolverOptions | None = None,
        gate_thresholds: OnlineGateThresholds | None = None,
        accumulation_batches: int = 1,
        max_accumulated_train_points: int | None = None,
    ) -> None:
        self.variable = variable
        self.parent = parent
        self.sensor = sensor
        self.source_records = source_records
        self.voxel_size_m = voxel_size_m
        self.correspondence_gate_m = correspondence_gate_m
        self.holdout_ratio = holdout_ratio
        self.rolling_window = max(1, rolling_window)
        self.seed = seed
        self.solver_options = solver_options or FixedTrajectorySe3SolverOptions()
        self.gate_thresholds = gate_thresholds or OnlineGateThresholds()
        self.accumulation_batches = max(1, accumulation_batches)
        self.max_accumulated_train_points = (
            max_accumulated_train_points
            if max_accumulated_train_points is None or max_accumulated_train_points > 0
            else None
        )

        self.current_estimate = initial_transform
        self.batch_index = 0
        self.history: list[OnlineBatchSnapshot] = []
        self._rolling_residuals: deque[float] = deque(maxlen=self.rolling_window)
        self._retained_train_batches: deque[list[Vector3]] = deque(
            maxlen=max(0, self.accumulation_batches - 1)
        )
        self._retained_frame_pose_batches: deque[list[SE3]] = deque(
            maxlen=max(0, self.accumulation_batches - 1)
        )

    def process_batch(
        self,
        points: list[Vector3],
        *,
        frame_count: int = 1,
        target_t_world_source: list[SE3] | None = None,
        batch_max_odometry_extrapolation_s: float | None = None,
    ) -> OnlineBatchSnapshot:
        """Consume one batch of target LiDAR points and return its snapshot."""

        if target_t_world_source is not None and len(target_t_world_source) != len(points):
            msg = "target_t_world_source length must match points"
            raise ValueError(msg)

        extrapolation_gate_reason = self._odometry_extrapolation_gate_reason(
            batch_max_odometry_extrapolation_s
        )
        if extrapolation_gate_reason is not None:
            snapshot = self._extrapolation_fail_snapshot(
                points=points,
                frame_count=frame_count,
                gate_reason=extrapolation_gate_reason,
                batch_max_odometry_extrapolation_s=batch_max_odometry_extrapolation_s,
            )
            self.history.append(snapshot)
            self.batch_index += 1
            return snapshot

        batch_seed = self.seed + self.batch_index
        train_indices, holdout_indices = split_indices(
            len(points), self.holdout_ratio, seed=batch_seed
        )
        train_points = [points[index] for index in train_indices]
        holdout_points = [points[index] for index in holdout_indices]

        prior_rolling_rmse = self._rolling_rmse()
        accumulated_train_points, accumulated_frame_poses = self._accumulated_train_batch(
            train_points,
            _slice_frame_poses(target_t_world_source, train_indices),
        )
        tentative_transform, train_observation_count = self._solve_batch(
            accumulated_train_points,
            target_t_world_source=accumulated_frame_poses,
        )
        holdout_observation_count, holdout_rmse, holdout_residuals, holdout_evaluation = (
            self._evaluate_holdout(
                holdout_points,
                tentative_transform,
                target_t_world_source=_slice_frame_poses(
                    target_t_world_source, holdout_indices
                ),
            )
        )
        accumulated_evaluation = self._evaluate_train_observability(
            accumulated_train_points,
            tentative_transform,
            target_t_world_source=accumulated_frame_poses,
        )
        batch_evaluation = (
            self._evaluate_train_observability(
                train_points,
                tentative_transform,
                target_t_world_source=_slice_frame_poses(
                    target_t_world_source, train_indices
                ),
            )
            if self.accumulation_batches > 1
            else None
        )
        rank_evaluation = (
            accumulated_evaluation
            if self.accumulation_batches > 1
            else holdout_evaluation
        )

        gate_status, gate_reason, retained_for_accumulation = self._evaluate_gate(
            train_observation_count=train_observation_count,
            holdout_observation_count=holdout_observation_count,
            holdout_rmse=holdout_rmse,
            evaluation=rank_evaluation,
            prior_rolling_rmse=prior_rolling_rmse,
        )
        # Only a PASS gate adopts the batch: fail and inconclusive are both
        # non-destructive for the running estimate and rolling window.
        # Observability-only inconclusive batches (when accumulating) still
        # retain train points so complementary views can bootstrap full rank.
        accepted = gate_status == "pass"
        if accepted:
            self.current_estimate = tentative_transform
            self._rolling_residuals.extend(abs(value) for value in holdout_residuals)
        if accepted or retained_for_accumulation:
            self._retained_train_batches.append(list(train_points))
            if target_t_world_source is not None:
                train_poses = _slice_frame_poses(target_t_world_source, train_indices)
                assert train_poses is not None
                self._retained_frame_pose_batches.append(train_poses)

        gate_observability = (
            _observability_from_evaluation(accumulated_evaluation)
            if self.accumulation_batches > 1
            else _observability_from_evaluation(holdout_evaluation)
        )
        snapshot = OnlineBatchSnapshot(
            batch_index=self.batch_index,
            frame_count=frame_count,
            point_count=len(points),
            train_point_count=len(train_points),
            holdout_point_count=len(holdout_points),
            correspondence_count=train_observation_count,
            holdout_correspondence_count=holdout_observation_count,
            estimate=_transform_result(
                variable=self.variable,
                parent=self.parent,
                child=self.sensor,
                transform=tentative_transform,
                accepted=accepted,
                retained_for_accumulation=retained_for_accumulation,
            ),
            estimate_accepted=accepted,
            batch_holdout_rmse_m=holdout_rmse,
            rolling_rmse_m=self._rolling_rmse(),
            rolling_window_residual_count=len(self._rolling_residuals),
            observability=gate_observability,
            batch_observability=(
                _observability_from_evaluation(batch_evaluation)
                if batch_evaluation is not None
                else None
            ),
            gate_status=gate_status,
            gate_reason=gate_reason,
            retained_for_accumulation=retained_for_accumulation,
            provenance={
                "batch_seed": batch_seed,
                "warm_start_from": (
                    "previous_accepted_estimate"
                    if self.batch_index > 0
                    else "config_initial_transform"
                ),
                **(
                    {
                        "accumulation_batches": self.accumulation_batches,
                        "accumulated_train_point_count": len(accumulated_train_points),
                        "retained_batch_count": len(self._retained_train_batches),
                    }
                    if self.accumulation_batches > 1
                    else {}
                ),
            },
        )
        self.history.append(snapshot)
        self.batch_index += 1
        return snapshot

    def _solve_batch(
        self,
        train_points: list[Vector3],
        *,
        target_t_world_source: list[SE3] | None = None,
    ) -> tuple[SE3, int]:
        observations = build_rig_point_to_plane_observations(
            source_records=self.source_records,
            target_points=train_points,
            initial_t_source_target=self.current_estimate,
            target_t_world_source=target_t_world_source,
            voxel_size_m=self.voxel_size_m,
            correspondence_gate_m=self.correspondence_gate_m,
        )
        if len(observations) < _MIN_TRAIN_OBSERVATIONS:
            return self.current_estimate, len(observations)
        factor = LidarRigPointToPlaneFactor(
            variable=self.variable,
            t_ego_lidar=self.current_estimate,
            observations=observations,
            sensor=self.sensor,
        )
        solver_result = FixedTrajectorySe3ExtrinsicSolver().solve(factor, self.solver_options)
        return solver_result.refined_transform, len(observations)

    def _accumulated_train_batch(
        self,
        current_train_points: list[Vector3],
        current_frame_poses: list[SE3] | None,
    ) -> tuple[list[Vector3], list[SE3] | None]:
        """Combine retained-batch train points (and poses) with the current batch."""

        if self.accumulation_batches <= 1:
            return current_train_points, current_frame_poses
        retained_points = [point for batch in self._retained_train_batches for point in batch]
        combined_points = [*retained_points, *current_train_points]
        downsampled_points = _downsample_points(combined_points, self.max_accumulated_train_points)
        if current_frame_poses is None:
            return downsampled_points, None
        retained_poses = [pose for batch in self._retained_frame_pose_batches for pose in batch]
        combined_poses = [*retained_poses, *current_frame_poses]
        if len(combined_poses) != len(combined_points):
            return downsampled_points, None
        downsampled_poses = _downsample_poses(
            combined_poses,
            self.max_accumulated_train_points,
            len(combined_points),
        )
        return downsampled_points, downsampled_poses

    def _accumulated_train_points(self, current_train_points: list[Vector3]) -> list[Vector3]:
        """Combine retained-batch train points with the current batch."""

        points, _poses = self._accumulated_train_batch(current_train_points, None)
        return points

    def _evaluate_train_observability(
        self,
        train_points: list[Vector3],
        tentative_transform: SE3,
        *,
        target_t_world_source: list[SE3] | None = None,
    ) -> LidarRigPointToPlaneEvaluation | None:
        observations = build_rig_point_to_plane_observations(
            source_records=self.source_records,
            target_points=train_points,
            initial_t_source_target=tentative_transform,
            target_t_world_source=target_t_world_source,
            voxel_size_m=self.voxel_size_m,
            correspondence_gate_m=self.correspondence_gate_m,
        )
        if not observations:
            return None
        factor = LidarRigPointToPlaneFactor(
            variable=self.variable,
            t_ego_lidar=tentative_transform,
            observations=observations,
            sensor=self.sensor,
        )
        return factor.evaluate()

    def _evaluate_holdout(
        self,
        holdout_points: list[Vector3],
        tentative_transform: SE3,
        *,
        target_t_world_source: list[SE3] | None = None,
    ) -> tuple[int, float | None, list[float], LidarRigPointToPlaneEvaluation | None]:
        observations = build_rig_point_to_plane_observations(
            source_records=self.source_records,
            target_points=holdout_points,
            initial_t_source_target=tentative_transform,
            target_t_world_source=target_t_world_source,
            voxel_size_m=self.voxel_size_m,
            correspondence_gate_m=self.correspondence_gate_m,
        )
        if not observations:
            return 0, None, [], None
        factor = LidarRigPointToPlaneFactor(
            variable=self.variable,
            t_ego_lidar=tentative_transform,
            observations=observations,
            sensor=self.sensor,
        )
        residuals = factor.residuals()
        evaluation = factor.evaluate()
        return len(observations), evaluation.rmse_m, residuals, evaluation

    def _evaluate_gate(
        self,
        *,
        train_observation_count: int,
        holdout_observation_count: int,
        holdout_rmse: float | None,
        evaluation: LidarRigPointToPlaneEvaluation | None,
        prior_rolling_rmse: float | None,
    ) -> tuple[OnlineGateStatus, str, bool]:
        thresholds = self.gate_thresholds
        if train_observation_count < _MIN_TRAIN_OBSERVATIONS:
            return (
                "inconclusive",
                "too few train point-to-plane correspondences to fit this batch "
                f"({train_observation_count} < {_MIN_TRAIN_OBSERVATIONS})",
                False,
            )
        if holdout_observation_count < _MIN_HOLDOUT_OBSERVATIONS or evaluation is None:
            return (
                "inconclusive",
                "too few holdout point-to-plane correspondences to score this batch "
                f"({holdout_observation_count} < {_MIN_HOLDOUT_OBSERVATIONS})",
                False,
            )
        if holdout_rmse is not None and holdout_rmse > thresholds.max_holdout_rmse_m:
            return (
                "fail",
                f"holdout RMSE {holdout_rmse:.4f} m exceeds the "
                f"{thresholds.max_holdout_rmse_m:.4f} m gate; rejecting this batch's update",
                False,
            )
        if (
            holdout_rmse is not None
            and prior_rolling_rmse is not None
            and (holdout_rmse - prior_rolling_rmse) > thresholds.max_rolling_regression_m
        ):
            return (
                "fail",
                f"holdout RMSE regressed by {holdout_rmse - prior_rolling_rmse:.4f} m "
                f"against the rolling baseline ({prior_rolling_rmse:.4f} m); "
                "rejecting this batch's update",
                False,
            )
        if evaluation.rank < thresholds.min_rank or evaluation.weak_directions:
            rank_scope = (
                "accumulated batch geometry"
                if self.accumulation_batches > 1
                else "batch geometry"
            )
            observability_reason = (
                f"{rank_scope} is rank-deficient or has weak DoF "
                f"(rank={evaluation.rank}, weak={list(evaluation.weak_directions)})"
            )
            if self.accumulation_batches > 1:
                return (
                    "inconclusive",
                    f"insufficient observability; retained for accumulation: "
                    f"{observability_reason}",
                    True,
                )
            return (
                "fail",
                f"{observability_reason}; rejecting this batch's update",
                False,
            )
        return "pass", "holdout evidence supports this batch's update", False

    def _odometry_extrapolation_gate_reason(
        self,
        batch_max_odometry_extrapolation_s: float | None,
    ) -> str | None:
        if batch_max_odometry_extrapolation_s is None:
            return None
        threshold = self.gate_thresholds.max_odometry_extrapolation_s
        if batch_max_odometry_extrapolation_s <= threshold:
            return None
        return (
            f"odometry_extrapolation: max extrapolation "
            f"{batch_max_odometry_extrapolation_s:.4f} s exceeds the "
            f"{threshold:.4f} s gate; rejecting this batch's update"
        )

    def _extrapolation_fail_snapshot(
        self,
        *,
        points: list[Vector3],
        frame_count: int,
        gate_reason: str,
        batch_max_odometry_extrapolation_s: float | None,
    ) -> OnlineBatchSnapshot:
        batch_seed = self.seed + self.batch_index
        train_indices, holdout_indices = split_indices(
            len(points), self.holdout_ratio, seed=batch_seed
        )
        return OnlineBatchSnapshot(
            batch_index=self.batch_index,
            frame_count=frame_count,
            point_count=len(points),
            train_point_count=len(train_indices),
            holdout_point_count=len(holdout_indices),
            correspondence_count=0,
            holdout_correspondence_count=0,
            estimate=_transform_result(
                variable=self.variable,
                parent=self.parent,
                child=self.sensor,
                transform=self.current_estimate,
                accepted=False,
            ),
            estimate_accepted=False,
            batch_holdout_rmse_m=None,
            rolling_rmse_m=self._rolling_rmse(),
            rolling_window_residual_count=len(self._rolling_residuals),
            observability=ObservabilityResult(),
            gate_status="fail",
            gate_reason=gate_reason,
            provenance={
                "batch_seed": batch_seed,
                "warm_start_from": (
                    "previous_accepted_estimate"
                    if self.batch_index > 0
                    else "config_initial_transform"
                ),
                "batch_max_odometry_extrapolation_s": batch_max_odometry_extrapolation_s,
            },
        )

    def _rolling_rmse(self) -> float | None:
        if not self._rolling_residuals:
            return None
        mean_square = sum(value * value for value in self._rolling_residuals) / len(
            self._rolling_residuals
        )
        return float(mean_square**0.5)


@dataclass(frozen=True)
class OnlineCalibrationRunOptions:
    """Runtime options for the online/streaming calibration pipeline."""

    dry_run: bool = False
    output_dir: Path | None = None
    strict: bool = False
    seed: int = 0
    batch_size: int = 500
    rolling_window: int = 2000
    holdout_ratio: float = 0.2
    gate_thresholds: OnlineGateThresholds | None = None
    accumulation_batches: int = 1
    max_accumulated_train_points: int | None = None


def run_online_calibration(
    config_path: str | Path,
    options: OnlineCalibrationRunOptions,
) -> CalibrationResult | None:
    """Replay a LiDAR pair as an online/streaming calibration session."""

    if not 0.0 < options.holdout_ratio <= 0.9:
        raise ConfigError(
            "online calibration holdout_ratio must be > 0.0 and <= 0.9 "
            f"(got {options.holdout_ratio}); 0.0 would leave every batch "
            "without holdout evidence, so no batch could ever be accepted"
        )
    if options.accumulation_batches < 1:
        raise ConfigError(
            "online calibration accumulation_batches must be >= 1 "
            f"(got {options.accumulation_batches})"
        )
    if (
        options.max_accumulated_train_points is not None
        and options.max_accumulated_train_points < 1
    ):
        raise ConfigError(
            "online calibration max_accumulated_train_points must be >= 1 when set "
            f"(got {options.max_accumulated_train_points})"
        )

    config_file = Path(config_path)
    config = load_config(config_file)
    frame_graph = FrameGraph.from_config(config)

    lidars = sorted(name for name, sensor in config.sensors.items() if sensor.type == "lidar")
    if len(lidars) < 2:
        raise ConfigError("online calibration requires at least two LiDAR sensors")
    if config.dataset.type not in _SUPPORTED_DATASET_TYPES:
        raise ConfigError(
            f"online calibration does not support dataset type '{config.dataset.type}'; "
            f"expected one of {_SUPPORTED_DATASET_TYPES}"
        )

    source_sensor, target_sensor = _resolve_online_lidar_pair(config, frame_graph, lidars)
    source_node = frame_graph.nodes.get(source_sensor)
    target_node = frame_graph.nodes.get(target_sensor)
    if source_node is None or target_node is None or target_node.parent is None:
        raise ConfigError(
            "source and target LiDAR frames must be present in the frame graph"
        )

    if options.dry_run:
        return None

    parent = target_node.parent
    variable = f"T_{parent}_{target_sensor}"
    inputs = _solve_inputs(config)
    replay_provenance: dict[str, Any]
    motion_compensated = False
    if config.dataset.type == "rosbag1":
        source_records, target_stream, replay_provenance = _load_rosbag1_online_pair(
            config,
            inputs,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
        )
    elif config.dataset.type == "rosbag2":
        source_records, target_stream, replay_provenance, motion_compensated = (
            _load_rosbag2_online_pair(
                config,
                inputs,
                source_sensor=source_sensor,
                target_sensor=target_sensor,
                t_base_source=source_node.transform_to_parent,
            )
        )
    else:
        files = _dataset_files(config)
        if len(files) < 2:
            return _unavailable_result(
                config=config,
                config_file=config_file,
                frame_graph=frame_graph,
                options=options,
                reason=(
                    "online calibration requires at least two LiDAR frames "
                    f"(found {len(files)})"
                ),
            )
        source_records = _load_source_records(config, files[0], inputs)
        target_stream = _load_target_stream(config, files[1:], inputs)
        replay_provenance = {}
    if not source_records:
        return _unavailable_result(
            config=config,
            config_file=config_file,
            frame_graph=frame_graph,
            options=options,
            reason="online calibration found no source LiDAR points for the voxel map",
        )
    if not target_stream:
        return _unavailable_result(
            config=config,
            config_file=config_file,
            frame_graph=frame_graph,
            options=options,
            reason="online calibration found no target LiDAR points to replay",
        )

    t_base_source = source_node.transform_to_parent
    t_base_target = target_node.transform_to_parent
    t_source_target_init = t_base_source.inverse().compose(t_base_target)

    gate_thresholds = options.gate_thresholds or _gate_thresholds_from_config(config)
    session = OnlineCalibrationSession(
        variable=variable,
        parent=parent,
        sensor=target_sensor,
        source_records=source_records,
        initial_transform=t_source_target_init,
        voxel_size_m=inputs.voxel_size_m,
        correspondence_gate_m=inputs.correspondence_gate_m,
        holdout_ratio=options.holdout_ratio,
        rolling_window=options.rolling_window,
        seed=options.seed,
        solver_options=FixedTrajectorySe3SolverOptions(
            max_iterations=max(1, config.solver.max_iterations),
            convergence_tolerance=config.solver.convergence_tolerance,
            robust_loss=_robust_loss(config.solver.robust_loss),
        ),
        gate_thresholds=gate_thresholds,
        accumulation_batches=options.accumulation_batches,
        max_accumulated_train_points=options.max_accumulated_train_points,
    )

    batch_size = max(1, options.batch_size)
    for start in range(0, len(target_stream), batch_size):
        chunk = target_stream[start : start + batch_size]
        batch_points = [point for _frame_index, point, _pose, _extrap in chunk]
        batch_poses = [pose for _frame_index, _point, pose, _extrap in chunk]
        frame_extrapolations = [extrap for _frame_index, _point, _pose, extrap in chunk]
        frame_count = len({frame_index for frame_index, _point, _pose, _extrap in chunk})
        batch_max_extrapolation_s: float | None = None
        if motion_compensated:
            batch_max_extrapolation_s = max(frame_extrapolations) if frame_extrapolations else 0.0
        session.process_batch(
            batch_points,
            frame_count=frame_count,
            target_t_world_source=(
                batch_poses if motion_compensated else None
            ),
            batch_max_odometry_extrapolation_s=batch_max_extrapolation_s,
        )

    return _build_result(
        config=config,
        config_file=config_file,
        frame_graph=frame_graph,
        options=options,
        session=session,
        variable=variable,
        parent=parent,
        source_sensor=source_sensor,
        target_sensor=target_sensor,
        t_base_source=t_base_source,
        batch_size=batch_size,
        replay_provenance=replay_provenance,
        motion_compensated=motion_compensated,
    )


@dataclass(frozen=True)
class _OnlineSolveInputs:
    voxel_size_m: float
    correspondence_gate_m: float
    max_source_points: int | None
    max_target_points: int | None
    max_source_messages: int | None
    max_target_messages: int | None
    max_replay_duration_s: float | None


def _solve_inputs(config: CalibrationConfig) -> _OnlineSolveInputs:
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    return _OnlineSolveInputs(
        voxel_size_m=_float_option(options, "voxel_size_m", default=1.0, minimum=0.05),
        correspondence_gate_m=_float_option(
            options, "correspondence_gate_m", default=1.5, minimum=0.05
        ),
        max_source_points=_int_option(options, "max_source_points", default=None),
        max_target_points=_int_option(options, "max_target_points", default=2000),
        max_source_messages=_int_option(options, "max_source_messages", default=None),
        max_target_messages=_int_option(options, "max_target_messages", default=None),
        max_replay_duration_s=_optional_float_option(
            options, "max_replay_duration_s", default=None, minimum=0.1
        ),
    )


def _float_option(
    options: dict[str, Any], key: str, *, default: float, minimum: float
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


def _optional_float_option(
    options: dict[str, Any], key: str, *, default: float | None, minimum: float
) -> float | None:
    if key not in options:
        return default
    try:
        value = float(options[key])
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return None
    return max(value, minimum)


def _gate_thresholds_from_config(config: CalibrationConfig) -> OnlineGateThresholds:
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    return OnlineGateThresholds(
        min_rank=_int_option(options, "online_gate_min_rank", default=6) or 6,
        max_holdout_rmse_m=_float_option(
            options, "online_gate_max_holdout_rmse_m", default=0.05, minimum=0.001
        ),
        max_rolling_regression_m=_float_option(
            options, "online_gate_max_rolling_regression_m", default=0.02, minimum=0.001
        ),
        max_odometry_extrapolation_s=_float_option(
            options, "online_gate_max_odometry_extrapolation_s", default=0.25, minimum=0.0
        ),
    )


def _robust_loss(name: str) -> RobustLoss:
    return "huber" if name == "huber" else "none"


def _dataset_files(config: CalibrationConfig) -> list[Path]:
    if config.dataset.type == "a2d2_lidar":
        return find_a2d2_lidar_npz_files(config.dataset.path)
    if config.dataset.type == "livox_pcd":
        return find_livox_pcd_files(config.dataset.path)
    return []


def _load_source_records(
    config: CalibrationConfig,
    source_path: Path,
    inputs: _OnlineSolveInputs,
) -> list[LivoxPointRecord]:
    if config.dataset.type == "a2d2_lidar":
        points = read_a2d2_lidar_points(source_path, max_points=inputs.max_source_points)
        return [LivoxPointRecord(point=(x, y, z, 0.0), normal_xyz=None) for x, y, z in points]
    records = read_livox_binary_pcd_records(source_path)
    return _stride(records, inputs.max_source_points)


def _load_target_stream(
    config: CalibrationConfig,
    target_files: list[Path],
    inputs: _OnlineSolveInputs,
) -> list[OnlineTargetEntry]:
    """Return an ordered `(frame_index, point, frame_pose)` stream across frames."""

    stream: list[OnlineTargetEntry] = []
    for frame_index, path in enumerate(target_files):
        if config.dataset.type == "a2d2_lidar":
            points = read_a2d2_lidar_points(path, max_points=inputs.max_target_points)
        else:
            records = _stride(read_livox_binary_pcd_records(path), inputs.max_target_points)
            points = [(record.point[0], record.point[1], record.point[2]) for record in records]
        stream.extend((frame_index, point, SE3.identity(), 0.0) for point in points)
    return stream


def _stride(records: list[LivoxPointRecord], max_records: int | None) -> list[LivoxPointRecord]:
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records
    step = (len(records) + max_records - 1) // max_records
    return records[::step]


def _resolve_online_lidar_pair(
    config: CalibrationConfig,
    frame_graph: FrameGraph,
    lidars: list[str],
) -> tuple[str, str]:
    """Pick the fixed source map sensor and the streaming target sensor."""

    root_lidars = [
        name
        for name in lidars
        if frame_graph.nodes.get(name) is not None and frame_graph.nodes[name].root
    ]
    if len(root_lidars) == 1:
        root = root_lidars[0]
        estimating_children = [
            name
            for name in lidars
            if name != root
            and frame_graph.nodes.get(name) is not None
            and frame_graph.nodes[name].parent == root
            and frame_graph.nodes[name].estimate
        ]
        if len(estimating_children) == 1:
            return root, estimating_children[0]

    # Dual-base-link layouts (A2D2, Livox PCD pair) keep the historical sorted order.
    return lidars[0], lidars[1]


def _sensor_point_time_field(config: CalibrationConfig, sensor_name: str) -> str | None:
    sensor = config.sensors.get(sensor_name)
    if sensor is None:
        return None
    return sensor.point_time_field


def _deskew_time_field_map(
    config: CalibrationConfig,
    source_sensor: str,
    target_sensor: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sensor_name in (source_sensor, target_sensor):
        field_name = _sensor_point_time_field(config, sensor_name)
        if field_name:
            mapping[sensor_name] = field_name
    return mapping


def _deskew_replay_provenance(
    *,
    deskew_time_field: dict[str, str],
    motion_compensated: bool,
    deskew_applied: bool,
    deskew_span_s: float | None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {"deskew_applied": deskew_applied}
    if deskew_time_field:
        provenance["deskew_time_field"] = deskew_time_field
        if not motion_compensated:
            provenance["deskew_ignored_no_odometry"] = True
    if deskew_span_s is not None:
        provenance["deskew_span_s"] = deskew_span_s
    return provenance


def _subsample_indices(count: int, max_points: int | None) -> range:
    if max_points is not None and max_points > 0 and count > max_points:
        step = (count + max_points - 1) // max_points
        return range(0, count, step)
    return range(count)


def _capture_timestamp_ns(message_timestamp_ns: int, offset_s: float) -> int:
    return message_timestamp_ns + round(offset_s * 1_000_000_000)


def _update_deskew_span(
    offsets: Any,
    *,
    span_min: float | None,
    span_max: float | None,
) -> tuple[float | None, float | None]:
    if offsets is None or int(offsets.shape[0]) == 0:
        return span_min, span_max
    lo = float(offsets.min())
    hi = float(offsets.max())
    span_min = lo if span_min is None else min(span_min, lo)
    span_max = hi if span_max is None else max(span_max, hi)
    return span_min, span_max


def _deskew_pointcloud_to_world_records(
    xyz: Any,
    offsets_s: Any,
    message_timestamp_ns: int,
    t_base_source: SE3,
    odometry_track: OdometryTrack,
    max_points: int | None,
    extrapolation_tolerance_s: float | None,
) -> list[LivoxPointRecord]:
    count = int(xyz.shape[0])
    records: list[LivoxPointRecord] = []
    for index in _subsample_indices(count, max_points):
        capture_ns = _capture_timestamp_ns(message_timestamp_ns, float(offsets_s[index]))
        t_world_base, _clamped, extrapolation_s = odometry_track.interpolate(capture_ns)
        if (
            extrapolation_tolerance_s is not None
            and extrapolation_s > extrapolation_tolerance_s
        ):
            msg = (
                "source LiDAR point capture time lies "
                f"{extrapolation_s:.4f} s outside the odometry track "
                f"(tolerance {extrapolation_tolerance_s:.4f} s); "
                "motion-compensated map premise is broken"
            )
            raise DatasetError(msg)
        t_world_source = t_world_base.compose(t_base_source)
        point_sensor = (
            float(xyz[index, 0]),
            float(xyz[index, 1]),
            float(xyz[index, 2]),
        )
        point_world = t_world_source.transform_point(point_sensor)
        records.append(
            LivoxPointRecord(
                point=(point_world[0], point_world[1], point_world[2], 0.0),
                normal_xyz=None,
            )
        )
    return records


def _deskew_target_entries(
    frame_index: int,
    xyz: Any,
    offsets_s: Any,
    message_timestamp_ns: int,
    t_base_source: SE3,
    odometry_track: OdometryTrack,
    max_points: int | None,
) -> list[OnlineTargetEntry]:
    count = int(xyz.shape[0])
    entries: list[OnlineTargetEntry] = []
    for index in _subsample_indices(count, max_points):
        capture_ns = _capture_timestamp_ns(message_timestamp_ns, float(offsets_s[index]))
        t_world_base, _clamped, extrapolation_s = odometry_track.interpolate(capture_ns)
        t_world_source = t_world_base.compose(t_base_source)
        point = (
            float(xyz[index, 0]),
            float(xyz[index, 1]),
            float(xyz[index, 2]),
        )
        entries.append((frame_index, point, t_world_source, extrapolation_s))
    return entries


def _lidar_topic(config: CalibrationConfig, sensor_name: str) -> str:
    sensor = config.sensors.get(sensor_name)
    if sensor is None or not sensor.topic:
        raise ConfigError(
            f"rosbag1 online calibration requires sensors.{sensor_name}.topic"
        )
    return sensor.topic


def _load_rosbag1_online_pair(
    config: CalibrationConfig,
    inputs: _OnlineSolveInputs,
    *,
    source_sensor: str,
    target_sensor: str,
) -> tuple[list[LivoxPointRecord], list[OnlineTargetEntry], dict[str, Any]]:
    """Stream one ROS bag once, building a bounded source map and target stream."""

    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        return [], [], {"rosbag1_path": str(bag_path), "rosbag1_exists": False}

    source_topic = _lidar_topic(config, source_sensor)
    target_topic = _lidar_topic(config, target_sensor)
    topics = {source_topic, target_topic}

    source_records: list[LivoxPointRecord] = []
    target_stream: list[OnlineTargetEntry] = []
    source_messages = 0
    target_messages = 0
    frame_index = 0
    replay_start_ns: int | None = None
    replay_end_ns: int | None = None
    first_source_ns: int | None = None
    first_target_ns: int | None = None

    for connection, timestamp_ns, data in iter_messages(bag_path, topics=topics):
        if connection.message_type not in LIDAR_MESSAGE_TYPES:
            continue
        topic = connection.topic
        if topic == source_topic:
            if (
                inputs.max_source_messages is not None
                and source_messages >= inputs.max_source_messages
            ):
                continue
            if (
                inputs.max_source_points is not None
                and len(source_records) >= inputs.max_source_points
            ):
                continue
            message = decode_bag_lidar_message(
                topic,
                connection.message_type,
                timestamp_ns,
                data,
            )
            first_source_ns = first_source_ns or message.timestamp_ns
            remaining = (
                None
                if inputs.max_source_points is None
                else max(0, inputs.max_source_points - len(source_records))
            )
            source_records.extend(_pointcloud_xyz_to_records(message.xyz, remaining))
            source_messages += 1
            continue

        if topic != target_topic:
            continue

        if replay_start_ns is None:
            replay_start_ns = timestamp_ns
            first_target_ns = timestamp_ns
        replay_end_ns = timestamp_ns
        if inputs.max_replay_duration_s is not None:
            elapsed_ns = timestamp_ns - replay_start_ns
            if elapsed_ns > int(inputs.max_replay_duration_s * 1_000_000_000):
                break
        if inputs.max_target_messages is not None and target_messages >= inputs.max_target_messages:
            break

        message = decode_bag_lidar_message(
            topic,
            connection.message_type,
            timestamp_ns,
            data,
        )
        points = _pointcloud_xyz_to_vectors(message.xyz, inputs.max_target_points)
        target_stream.extend(
            (frame_index, point, SE3.identity(), 0.0) for point in points
        )
        frame_index += 1
        target_messages += 1

    replay_duration_s: float | None = None
    if replay_start_ns is not None and replay_end_ns is not None:
        replay_duration_s = (replay_end_ns - replay_start_ns) / 1_000_000_000

    provenance = {
        "rosbag1_path": str(bag_path),
        "rosbag1_source_topic": source_topic,
        "rosbag1_target_topic": target_topic,
        "rosbag1_source_message_count": source_messages,
        "rosbag1_target_message_count": target_messages,
        "rosbag1_source_point_count": len(source_records),
        "rosbag1_target_point_count": len(target_stream),
        "rosbag1_replay_duration_s": replay_duration_s,
        "rosbag1_first_source_timestamp_ns": first_source_ns,
        "rosbag1_first_target_timestamp_ns": first_target_ns,
        "rosbag1_replay_start_timestamp_ns": replay_start_ns,
        "rosbag1_replay_end_timestamp_ns": replay_end_ns,
        "rosbag1_max_source_messages": inputs.max_source_messages,
        "rosbag1_max_target_messages": inputs.max_target_messages,
        "rosbag1_max_replay_duration_s": inputs.max_replay_duration_s,
    }
    provenance.update(
        _deskew_replay_provenance(
            deskew_time_field=_deskew_time_field_map(config, source_sensor, target_sensor),
            motion_compensated=False,
            deskew_applied=False,
            deskew_span_s=None,
        )
    )
    return source_records, target_stream, provenance


def _load_rosbag2_online_pair(
    config: CalibrationConfig,
    inputs: _OnlineSolveInputs,
    *,
    source_sensor: str,
    target_sensor: str,
    t_base_source: SE3,
) -> tuple[list[LivoxPointRecord], list[OnlineTargetEntry], dict[str, Any], bool]:
    """Stream one rosbag2 recording once, building a bounded source map and target stream."""

    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        return [], [], {"rosbag2_path": str(bag_path), "rosbag2_exists": False}, False

    source_topic = _lidar_topic(config, source_sensor)
    target_topic = _lidar_topic(config, target_sensor)
    odometry_topic = config.dataset.odometry_topic
    topics = {source_topic, target_topic}
    if odometry_topic:
        topics.add(odometry_topic)

    odometry_track = _load_rosbag2_odometry_track(bag_path, odometry_topic)
    motion_compensated = odometry_track is not None
    extrapolation_tolerance_s = _odometry_extrapolation_tolerance_s(config, motion_compensated)

    deskew_time_field = _deskew_time_field_map(config, source_sensor, target_sensor)
    source_time_field = deskew_time_field.get(source_sensor)
    target_time_field = deskew_time_field.get(target_sensor)
    source_deskew = motion_compensated and source_time_field is not None
    target_deskew = motion_compensated and target_time_field is not None
    deskew_applied = source_deskew or target_deskew
    deskew_span_min: float | None = None
    deskew_span_max: float | None = None

    source_records: list[LivoxPointRecord] = []
    target_stream: list[OnlineTargetEntry] = []
    source_messages = 0
    target_messages = 0
    frame_index = 0
    replay_start_ns: int | None = None
    replay_end_ns: int | None = None
    first_source_ns: int | None = None
    first_target_ns: int | None = None

    for connection, timestamp_ns, data in iter_rosbag2_messages(bag_path, topics=topics):
        if odometry_topic and connection.topic == odometry_topic:
            continue
        if connection.message_type != POINTCLOUD2_TYPE:
            continue
        topic = connection.topic
        if topic == source_topic:
            if (
                inputs.max_source_messages is not None
                and source_messages >= inputs.max_source_messages
            ):
                continue
            if (
                inputs.max_source_points is not None
                and len(source_records) >= inputs.max_source_points
            ):
                continue
            message = decode_pointcloud2(
                topic,
                timestamp_ns,
                data,
                point_time_field=source_time_field if source_deskew else None,
            )
            first_source_ns = first_source_ns or message.timestamp_ns
            remaining = (
                None
                if inputs.max_source_points is None
                else max(0, inputs.max_source_points - len(source_records))
            )
            if source_deskew and odometry_track is not None:
                offsets_s = message.point_time_offsets_s
                if offsets_s is None:
                    msg = (
                        f"deskew requested for source sensor {source_sensor!r} but "
                        "per-point time offsets were not decoded"
                    )
                    raise DatasetError(msg)
                deskew_span_min, deskew_span_max = _update_deskew_span(
                    offsets_s,
                    span_min=deskew_span_min,
                    span_max=deskew_span_max,
                )
                source_records.extend(
                    _deskew_pointcloud_to_world_records(
                        message.xyz,
                        offsets_s,
                        message.timestamp_ns,
                        t_base_source,
                        odometry_track,
                        remaining,
                        extrapolation_tolerance_s,
                    )
                )
            elif motion_compensated and odometry_track is not None:
                t_world_base, _clamped, extrapolation_s = odometry_track.interpolate(
                    message.timestamp_ns
                )
                if (
                    extrapolation_tolerance_s is not None
                    and extrapolation_s > extrapolation_tolerance_s
                ):
                    msg = (
                        "source LiDAR message timestamp lies "
                        f"{extrapolation_s:.4f} s outside the odometry track "
                        f"(tolerance {extrapolation_tolerance_s:.4f} s); "
                        "motion-compensated map premise is broken"
                    )
                    raise DatasetError(msg)
                t_world_source = t_world_base.compose(t_base_source)
                source_records.extend(
                    _transform_pointcloud_to_world_records(
                        message.xyz,
                        t_world_source,
                        remaining,
                    )
                )
            else:
                source_records.extend(_pointcloud_xyz_to_records(message.xyz, remaining))
            source_messages += 1
            continue

        if topic != target_topic:
            continue

        if replay_start_ns is None:
            replay_start_ns = timestamp_ns
            first_target_ns = timestamp_ns
        replay_end_ns = timestamp_ns
        if inputs.max_replay_duration_s is not None:
            elapsed_ns = timestamp_ns - replay_start_ns
            if elapsed_ns > int(inputs.max_replay_duration_s * 1_000_000_000):
                break
        if inputs.max_target_messages is not None and target_messages >= inputs.max_target_messages:
            break

        message = decode_pointcloud2(
            topic,
            timestamp_ns,
            data,
            point_time_field=target_time_field if target_deskew else None,
        )
        if target_deskew and odometry_track is not None:
            offsets_s = message.point_time_offsets_s
            if offsets_s is None:
                msg = (
                    f"deskew requested for target sensor {target_sensor!r} but "
                    "per-point time offsets were not decoded"
                )
                raise DatasetError(msg)
            deskew_span_min, deskew_span_max = _update_deskew_span(
                offsets_s,
                span_min=deskew_span_min,
                span_max=deskew_span_max,
            )
            target_stream.extend(
                _deskew_target_entries(
                    frame_index,
                    message.xyz,
                    offsets_s,
                    message.timestamp_ns,
                    t_base_source,
                    odometry_track,
                    inputs.max_target_points,
                )
            )
        elif motion_compensated and odometry_track is not None:
            points = _pointcloud_xyz_to_vectors(message.xyz, inputs.max_target_points)
            target_extrapolation_s = 0.0
            t_world_base, _clamped, target_extrapolation_s = odometry_track.interpolate(
                message.timestamp_ns
            )
            t_world_source = t_world_base.compose(t_base_source)
            target_stream.extend(
                (frame_index, point, t_world_source, target_extrapolation_s) for point in points
            )
        else:
            points = _pointcloud_xyz_to_vectors(message.xyz, inputs.max_target_points)
            target_stream.extend(
                (frame_index, point, SE3.identity(), 0.0) for point in points
            )
        frame_index += 1
        target_messages += 1

    replay_duration_s: float | None = None
    if replay_start_ns is not None and replay_end_ns is not None:
        replay_duration_s = (replay_end_ns - replay_start_ns) / 1_000_000_000

    provenance: dict[str, Any] = {
        "rosbag2_path": str(bag_path),
        "rosbag2_source_topic": source_topic,
        "rosbag2_target_topic": target_topic,
        "rosbag2_source_message_count": source_messages,
        "rosbag2_target_message_count": target_messages,
        "rosbag2_source_point_count": len(source_records),
        "rosbag2_target_point_count": len(target_stream),
        "rosbag2_replay_duration_s": replay_duration_s,
        "rosbag2_first_source_timestamp_ns": first_source_ns,
        "rosbag2_first_target_timestamp_ns": first_target_ns,
        "rosbag2_replay_start_timestamp_ns": replay_start_ns,
        "rosbag2_replay_end_timestamp_ns": replay_end_ns,
        "rosbag2_max_source_messages": inputs.max_source_messages,
        "rosbag2_max_target_messages": inputs.max_target_messages,
        "rosbag2_max_replay_duration_s": inputs.max_replay_duration_s,
        "motion_compensated": motion_compensated,
    }
    if odometry_topic:
        provenance.update(
            _odometry_replay_provenance(odometry_topic, odometry_track, motion_compensated)
        )
    deskew_span_s: float | None = None
    if deskew_span_min is not None and deskew_span_max is not None:
        deskew_span_s = deskew_span_max - deskew_span_min
    provenance.update(
        _deskew_replay_provenance(
            deskew_time_field=deskew_time_field,
            motion_compensated=motion_compensated,
            deskew_applied=deskew_applied,
            deskew_span_s=deskew_span_s,
        )
    )
    return source_records, target_stream, provenance, motion_compensated


def _load_rosbag2_odometry_track(
    bag_path: Path,
    odometry_topic: str | None,
) -> OdometryTrack | None:
    if not odometry_topic:
        return None
    samples: list[OdometryPoseSample] = []
    for connection, timestamp_ns, data in iter_rosbag2_messages(
        bag_path,
        topics={odometry_topic},
    ):
        if connection.message_type != ODOMETRY_TYPE:
            continue
        message = decode_odometry(connection.topic, timestamp_ns, data)
        samples.append(
            OdometryPoseSample(
                timestamp_ns=message.timestamp_ns,
                pose=SE3.from_lists(list(message.position), list(message.orientation_xyzw)),
            )
        )
    return OdometryTrack(samples) if samples else None


def _odometry_replay_provenance(
    odometry_topic: str,
    odometry_track: OdometryTrack | None,
    motion_compensated: bool,
) -> dict[str, Any]:
    return {
        "odometry_topic": odometry_topic,
        "odometry_message_count": odometry_track.message_count if odometry_track else 0,
        "odometry_first_timestamp_ns": (
            odometry_track.first_timestamp_ns if odometry_track else None
        ),
        "odometry_last_timestamp_ns": (
            odometry_track.last_timestamp_ns if odometry_track else None
        ),
        "odometry_interpolation_method": "linear_translation_slerp_rotation",
        "odometry_interpolation_clamp_count": (
            odometry_track.clamp_count if odometry_track else 0
        ),
        "odometry_interpolation_count": (
            odometry_track.interpolation_count if odometry_track else 0
        ),
        "odometry_interpolation_max_extrapolation_s": (
            odometry_track.max_extrapolation_s if odometry_track else 0.0
        ),
        "motion_compensated": motion_compensated,
    }


def _odometry_extrapolation_tolerance_s(
    config: CalibrationConfig,
    motion_compensated: bool,
) -> float | None:
    if not motion_compensated:
        return None
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    return _float_option(
        options, "online_gate_max_odometry_extrapolation_s", default=0.25, minimum=0.0
    )


def _transform_pointcloud_to_world_records(
    xyz: Any,
    t_world_source: SE3,
    max_points: int | None,
) -> list[LivoxPointRecord]:
    count = int(xyz.shape[0])
    records: list[LivoxPointRecord] = []
    for index in _subsample_indices(count, max_points):
        point_sensor = (
            float(xyz[index, 0]),
            float(xyz[index, 1]),
            float(xyz[index, 2]),
        )
        point_world = t_world_source.transform_point(point_sensor)
        records.append(
            LivoxPointRecord(
                point=(point_world[0], point_world[1], point_world[2], 0.0),
                normal_xyz=None,
            )
        )
    return records


def _slice_frame_poses(
    poses: list[SE3] | None,
    indices: list[int],
) -> list[SE3] | None:
    if poses is None:
        return None
    return [poses[index] for index in indices]


def _downsample_poses(
    poses: list[SE3],
    max_points: int | None,
    original_count: int,
) -> list[SE3]:
    if max_points is None or max_points <= 0 or original_count <= max_points:
        return poses
    step = (original_count + max_points - 1) // max_points
    return poses[::step]


def _pointcloud_xyz_to_records(
    xyz: Any,
    max_points: int | None,
) -> list[LivoxPointRecord]:
    count = int(xyz.shape[0])
    return [
        LivoxPointRecord(
            point=(float(xyz[index, 0]), float(xyz[index, 1]), float(xyz[index, 2]), 0.0),
            normal_xyz=None,
        )
        for index in _subsample_indices(count, max_points)
    ]


def _pointcloud_xyz_to_vectors(xyz: Any, max_points: int | None) -> list[Vector3]:
    count = int(xyz.shape[0])
    return [
        (float(xyz[index, 0]), float(xyz[index, 1]), float(xyz[index, 2]))
        for index in _subsample_indices(count, max_points)
    ]


def _downsample_points(points: list[Vector3], max_points: int | None) -> list[Vector3]:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    step = (len(points) + max_points - 1) // max_points
    return points[::step]


def _transform_result(
    *,
    variable: str,
    parent: str,
    child: str,
    transform: SE3,
    accepted: bool,
    retained_for_accumulation: bool = False,
) -> TransformResult:
    return TransformResult(
        parent=parent,
        child=child,
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        estimate_id=variable,
        quality=TransformQuality(grade="pass" if accepted else "warn"),
        provenance=TransformEstimateProvenance(
            producer="calibrex_native",
            execution_mode="online_stream",
            role_in_comparison="output" if accepted else "candidate",
            evidence_level="algorithmically_refined",
            tool_name=ONLINE_LIDAR_POINT_TO_PLANE_BACKEND,
            source="online_lidar_point_to_plane_session",
            notes=(
                []
                if accepted
                else [
                    (
                        "batch was not adopted by the online holdout gate "
                        "(observability-only inconclusive); train points were "
                        "retained for accumulation while the running estimate "
                        "is unchanged"
                    )
                    if retained_for_accumulation
                    else (
                        "batch was not adopted by the online holdout gate "
                        "(fail or inconclusive); the running estimate is unchanged"
                    )
                ]
            ),
        ),
    )


def _observability_from_evaluation(
    evaluation: LidarRigPointToPlaneEvaluation | None,
) -> ObservabilityResult:
    if evaluation is None:
        return ObservabilityResult()
    return ObservabilityResult(
        rank=evaluation.rank,
        condition_number=evaluation.normalized_condition_number_estimate,
        weak_directions=list(evaluation.weak_directions),
        grade="pass" if evaluation.rank >= 6 and not evaluation.weak_directions else "warn",
    )


def _unavailable_result(
    *,
    config: CalibrationConfig,
    config_file: Path,
    frame_graph: FrameGraph,
    options: OnlineCalibrationRunOptions,
    reason: str,
) -> CalibrationResult:
    output_dir = options.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = CalibrationResult(
        run=RunInfo(
            id=_run_id(config.project.name),
            calibrex_version=__version__,
            git_commit=git_commit(),
            config_sha256=sha256_path(config_file),
            dataset_sha256=sha256_path(Path(config.dataset.path)),
            status="warning",
            domain=config.project.domain,
            provenance={
                "pipeline": "online_calibration",
                "solver_adapter": ONLINE_LIDAR_POINT_TO_PLANE_BACKEND,
                "solver_adapter_status": "unavailable",
                "dataset_type": config.dataset.type,
                "dataset_path": config.dataset.path,
                "online_batch_size": options.batch_size,
                "dry_run": False,
            },
        ),
        frame_graph=frame_graph.snapshot(),
        metrics={
            "online_lidar_point_to_plane_available": MetricResult(
                value=0.0, grade="warn", reason=reason
            )
        },
        quality=QualitySummary(grade="warn", warnings=[reason]),
    )
    result.save(output_dir / config.outputs.result)
    write_report_artifacts(result, output_dir, html_filename=config.outputs.report)
    return result


def _build_result(
    *,
    config: CalibrationConfig,
    config_file: Path,
    frame_graph: FrameGraph,
    options: OnlineCalibrationRunOptions,
    session: OnlineCalibrationSession,
    variable: str,
    parent: str,
    source_sensor: str,
    target_sensor: str,
    t_base_source: SE3,
    batch_size: int,
    replay_provenance: dict[str, Any] | None = None,
    motion_compensated: bool = False,
) -> CalibrationResult:
    history = session.history
    final_snapshot = history[-1]
    accepted_count = sum(1 for snapshot in history if snapshot.gate_status == "pass")
    rejected_count = sum(1 for snapshot in history if snapshot.gate_status == "fail")
    inconclusive_count = sum(1 for snapshot in history if snapshot.gate_status == "inconclusive")

    t_base_target_refined = t_base_source.compose(session.current_estimate)
    final_grade = _grade_from_gate(final_snapshot.gate_status)
    output_transform = TransformResult(
        parent=parent,
        child=target_sensor,
        translation_m=list(t_base_target_refined.translation_m),
        rotation_quat_xyzw=list(t_base_target_refined.rotation_quat_xyzw),
        estimate_id=variable,
        quality=TransformQuality(grade=final_grade),
        provenance=TransformEstimateProvenance(
            producer="calibrex_native",
            execution_mode="online_stream",
            role_in_comparison="output",
            evidence_level="algorithmically_refined",
            tool_name=ONLINE_LIDAR_POINT_TO_PLANE_BACKEND,
            source="online_lidar_point_to_plane_session",
            notes=[
                f"warm-started online session over {len(history)} batches "
                f"({accepted_count} accepted, {rejected_count} rejected, "
                f"{inconclusive_count} inconclusive)"
            ],
        ),
    )
    transforms: dict[str, TransformResult] = {
        f"T_{node.parent}_{name}": _initial_transform_result(name, node.parent, node)
        for name, node in sorted(frame_graph.nodes.items())
        if node.parent is not None and name != target_sensor
    }
    transforms[variable] = output_transform

    metrics = _summary_metrics(
        history,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        inconclusive_count=inconclusive_count,
        gate_thresholds=session.gate_thresholds,
    )

    result = CalibrationResult(
        run=RunInfo(
            id=_run_id(config.project.name),
            calibrex_version=__version__,
            git_commit=git_commit(),
            config_sha256=sha256_path(config_file),
            dataset_sha256=sha256_path(Path(config.dataset.path)),
            status="warning",
            domain=config.project.domain,
            provenance={
                "pipeline": "online_calibration",
                "solver_adapter": ONLINE_LIDAR_POINT_TO_PLANE_BACKEND,
                "solver_adapter_status": final_snapshot.gate_status,
                "dataset_type": config.dataset.type,
                "dataset_path": config.dataset.path,
                "online_variable": variable,
                "online_source_sensor": source_sensor,
                "online_target_sensor": target_sensor,
                "online_batch_size": batch_size,
                "online_rolling_window": session.rolling_window,
                "online_holdout_ratio": session.holdout_ratio,
                "online_seed": session.seed,
                "online_accumulation_batches": session.accumulation_batches,
                "online_max_accumulated_train_points": session.max_accumulated_train_points,
                "online_batch_count": len(history),
                "online_accepted_batch_count": accepted_count,
                "online_rejected_batch_count": rejected_count,
                "online_inconclusive_batch_count": inconclusive_count,
                "online_final_rolling_rmse_m": final_snapshot.rolling_rmse_m,
                "online_final_gate_status": final_snapshot.gate_status,
                "online_gate_min_rank": session.gate_thresholds.min_rank,
                "online_gate_max_holdout_rmse_m": session.gate_thresholds.max_holdout_rmse_m,
                "online_gate_max_rolling_regression_m": (
                    session.gate_thresholds.max_rolling_regression_m
                ),
                "online_gate_max_odometry_extrapolation_s": (
                    session.gate_thresholds.max_odometry_extrapolation_s
                ),
                "dry_run": False,
                **(replay_provenance or {}),
            },
        ),
        frame_graph=frame_graph.snapshot(),
        transforms=transforms,
        metrics=metrics,
        observability=final_snapshot.observability,
        degeneracy=DegeneracyResult(
            grade="pass" if rejected_count == 0 else "warn",
            reason=(
                None
                if rejected_count == 0
                else f"{rejected_count} of {len(history)} online batches were "
                "rejected by the holdout gate"
            ),
        ),
    )

    output_dir = options.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result.artifacts.html_report = str(output_dir / config.outputs.report)
    evaluate_quality(result, strict=options.strict)
    result.save(output_dir / config.outputs.result)

    timeline_path = output_dir / _TIMELINE_FILENAME
    timeline_artifact = build_online_timeline_artifact(
        result=result,
        session=session,
        variable=variable,
        source_sensor=source_sensor,
        target_sensor=target_sensor,
        batch_size=batch_size,
        motion_compensated=motion_compensated,
    )
    write_report_artifacts(
        result,
        output_dir,
        html_filename=config.outputs.report,
        timeline=timeline_artifact,
    )
    write_mapping(
        timeline_path,
        timeline_artifact.model_dump(mode="json", exclude_none=True),
    )
    result.run.provenance["online_timeline_path"] = str(timeline_path)
    result.save(output_dir / config.outputs.result)
    return result


def _grade_from_gate(status: OnlineGateStatus) -> Grade:
    if status == "pass":
        return "pass"
    if status == "inconclusive":
        return "warn"
    return "fail"


def _initial_transform_result(name: str, parent: str, node: FrameNode) -> TransformResult:
    transform = node.transform_to_parent
    return TransformResult(
        parent=parent,
        child=name,
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        estimate_id=f"T_{parent}_{name}",
        quality=TransformQuality(grade="warn" if node.estimate else "pass"),
        provenance=TransformEstimateProvenance(
            producer="human" if node.estimate else "unknown",
            execution_mode="manual",
            role_in_comparison="candidate" if node.estimate else "output",
            evidence_level="imported_without_documented_derivation",
            source="config.frame_graph",
        ),
    )


def _summary_metrics(
    history: list[OnlineBatchSnapshot],
    *,
    accepted_count: int,
    rejected_count: int,
    inconclusive_count: int,
    gate_thresholds: OnlineGateThresholds,
) -> dict[str, MetricResult]:
    final_snapshot = history[-1]
    pass_fraction = accepted_count / len(history) if history else None
    return {
        "online_calibration_batch_count": MetricResult(
            value=float(len(history)),
            unit="batches",
            grade="pass" if history else "warn",
            reason="number of streamed batches processed by the online session",
        ),
        "online_calibration_accepted_batch_count": MetricResult(
            value=float(accepted_count),
            unit="batches",
            grade="pass" if accepted_count > 0 else "warn",
            reason="batches whose holdout gate passed and updated the running estimate",
        ),
        "online_calibration_rejected_batch_count": MetricResult(
            value=float(rejected_count),
            unit="batches",
            grade="pass" if rejected_count == 0 else "warn",
            reason="batches rejected by the holdout gate (estimate left unchanged)",
        ),
        "online_calibration_inconclusive_batch_count": MetricResult(
            value=float(inconclusive_count),
            unit="batches",
            grade="pass" if inconclusive_count == 0 else "warn",
            reason=(
                "batches with too little train/holdout evidence to accept or "
                "reject, or observability-only inconclusive batches retained "
                "for accumulation (estimate left unchanged)"
            ),
        ),
        "online_calibration_gate_pass_fraction": MetricResult(
            value=pass_fraction,
            grade="pass" if pass_fraction == 1.0 else "warn",
            reason="fraction of streamed batches accepted by the holdout gate",
        ),
        "online_calibration_final_rolling_rmse_m": MetricResult(
            value=final_snapshot.rolling_rmse_m,
            unit="m",
            grade=(
                "pass"
                if final_snapshot.rolling_rmse_m is not None
                and final_snapshot.rolling_rmse_m <= gate_thresholds.max_holdout_rmse_m
                else "warn"
            ),
            reason="rolling holdout RMSE over the most recent accepted residual window",
        ),
        "prototype_solver": MetricResult(
            value=1.0 if final_snapshot.gate_status == "pass" else 0.0,
            grade="pass" if final_snapshot.gate_status == "pass" else "warn",
            reason=(
                "native online fixed-trajectory point-to-plane solver produced "
                f"optimized extrinsics (final gate={final_snapshot.gate_status})"
            ),
        ),
    }


def _run_id(project_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", project_name).strip("_") or "calibrex"
    return f"{timestamp}_{slug}_online"


def build_online_timeline_artifact(
    *,
    result: CalibrationResult,
    session: OnlineCalibrationSession,
    variable: str,
    source_sensor: str,
    target_sensor: str,
    batch_size: int,
    motion_compensated: bool | None = None,
) -> OnlineCalibrationTimelineArtifact:
    """Build the machine-readable per-batch timeline artifact."""

    history = session.history
    final_status: OnlineGateStatus = history[-1].gate_status if history else "inconclusive"
    return OnlineCalibrationTimelineArtifact(
        schema_version=ONLINE_TIMELINE_SCHEMA_VERSION,
        run=_report_run_info(result),
        variable=variable,
        source_sensor=source_sensor,
        target_sensor=target_sensor,
        batch_size=batch_size,
        rolling_window=session.rolling_window,
        holdout_ratio=session.holdout_ratio,
        seed=session.seed,
        accumulation_batches=(
            session.accumulation_batches if session.accumulation_batches > 1 else None
        ),
        max_accumulated_train_points=session.max_accumulated_train_points,
        batches=history,
        final_gate_status=final_status,
        accepted_batch_count=sum(1 for s in history if s.gate_status == "pass"),
        rejected_batch_count=sum(1 for s in history if s.gate_status == "fail"),
        inconclusive_batch_count=sum(1 for s in history if s.gate_status == "inconclusive"),
        motion_compensated=motion_compensated,
    )


def write_online_timeline_artifact(
    *,
    result: CalibrationResult,
    session: OnlineCalibrationSession,
    variable: str,
    source_sensor: str,
    target_sensor: str,
    batch_size: int,
    output_path: str | Path,
) -> OnlineCalibrationTimelineArtifact:
    """Build and write the machine-readable per-batch timeline artifact."""

    artifact = build_online_timeline_artifact(
        result=result,
        session=session,
        variable=variable,
        source_sensor=source_sensor,
        target_sensor=target_sensor,
        batch_size=batch_size,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mapping(path, artifact.model_dump(mode="json", exclude_none=True))
    return artifact


def _report_run_info(result: CalibrationResult) -> ReportRunInfo:
    provenance = result.run.provenance
    dataset_type = provenance.get("dataset_type")
    dataset_path = provenance.get("dataset_path")
    return ReportRunInfo(
        id=result.run.id,
        status=result.run.status,
        domain=result.run.domain,
        calibrex_version=result.run.calibrex_version,
        git_commit=result.run.git_commit,
        created_at=result.run.created_at,
        dataset_type=dataset_type if isinstance(dataset_type, str) else None,
        dataset_path=dataset_path if isinstance(dataset_path, str) else None,
    )
