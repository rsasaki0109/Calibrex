"""Online/streaming LiDAR-pair extrinsic calibration.

This module drives the same native fixed-trajectory point-to-plane primitives
used by the offline path (`calibrex.solvers.native_lidar_point_to_plane_solver`)
from a stream of incremental point batches instead of one offline solve.

Motion compensation (optional ``dataset.odometry_topic`` on rosbag replays)
uses per-message rig poses by default. When a participating LiDAR sensor sets
``point_time_field`` and odometry is configured, per-point deskew applies the
declared capture-time convention. PointCloud2 offsets are relative to the
message stamp; Livox ``CustomMsg`` offsets are nanoseconds relative to its
``timebase``.

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
standard `CalibrationResult` (`producer=slac_native`,
`execution_mode=online_stream`) plus a machine-readable per-batch timeline
artifact (`calibrex.core.online_timeline`).
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from calibrex import __version__
from calibrex.core.capture_readiness import (
    CaptureReadinessArtifact,
    CaptureReadinessGeometry,
    CaptureReadinessParameters,
    CaptureReadinessProvenance,
    CaptureReadinessThresholds,
    CaptureReadinessWindow,
    build_capture_readiness_recommendations,
    classify_capture_readiness_window,
    summarize_capture_window_motion,
)
from calibrex.core.config import (
    CalibrationConfig,
    OdometryBurstPolicy,
    OdometryPreprocessingConfig,
    load_config,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    ContinuousTimeLidarPairArtifact,
    ContinuousTimeLidarPairIterationArtifact,
    ContinuousTimeLidarPairObservability,
    ContinuousTimeLidarPairOptionsArtifact,
    ContinuousTimeLidarPairProvenance,
)
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
from calibrex.core.solid_state import (
    SolidStateCaptureWindow,
    SolidStateLidarCalibrationContext,
    build_solid_state_context,
)
from calibrex.core.time import apply_time_offset_ns
from calibrex.core.trajectory_window_drift import (
    TrajectoryWindowCrossSegmentEvidence,
    TrajectoryWindowDriftArtifact,
    TrajectoryWindowDriftParameters,
    TrajectoryWindowDriftProvenance,
    TrajectoryWindowDriftThresholds,
    TrajectoryWindowDriftWindow,
    TrajectoryWindowOdometryStats,
    interpret_trajectory_window_drift,
)
from calibrex.data.a2d2 import find_a2d2_lidar_npz_files, read_a2d2_lidar_points
from calibrex.data.livox import (
    LivoxPointRecord,
    build_voxel_plane_map,
    find_livox_pcd_files,
    nearest_voxel_plane,
    read_livox_binary_pcd_records,
)
from calibrex.data.manifest import find_manifest, load_manifest
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.data.ros_messages import LivoxCustomMessage, PointCloud2Message
from calibrex.data.rosbag1 import (
    LIDAR_MESSAGE_TYPES,
    POSE_STAMPED_TYPE,
    decode_bag_lidar_message,
    decode_pose_stamped,
    iter_messages,
)
from calibrex.data.rosbag1 import (
    POINTCLOUD2_TYPE as ROS1_POINTCLOUD2_TYPE,
)
from calibrex.data.rosbag1 import (
    decode_pointcloud2 as decode_ros1_pointcloud2,
)
from calibrex.data.rosbag2 import (
    LIDAR_MESSAGE_TYPES as ROS2_LIDAR_MESSAGE_TYPES,
)
from calibrex.data.rosbag2 import (
    ODOMETRY_TYPE,
    decode_odometry,
)
from calibrex.data.rosbag2 import (
    decode_lidar_message as decode_rosbag2_lidar_message,
)
from calibrex.data.rosbag2 import (
    iter_messages as iter_rosbag2_messages,
)
from calibrex.evaluation.holdout import split_indices
from calibrex.evaluation.lidar import build_rig_point_to_plane_observations
from calibrex.evaluation.metrics import evaluate_quality
from calibrex.evaluation.solid_state import solid_state_metrics_from_inspection
from calibrex.evaluation.temporal import (
    DualTemporalEvidenceResult,
    TemporalSeparabilityVerdict,
    dual_temporal_evidence_to_provenance,
    evaluate_dual_temporal_evidence,
    temporal_evidence_options_from_factor,
    temporal_evidence_requested,
)
from calibrex.evaluation.trajectory import (
    CrossSegmentScanCollection,
    OdometrySourceInfo,
    OnlineSourceScan,
    TrajectoryEvidenceOptions,
    TrajectoryEvidenceResult,
    _select_evenly_spaced_timestamps,
    build_trajectory_artifact,
    evaluate_trajectory_evidence,
    trajectory_evidence_metrics,
    trajectory_evidence_options_from_factor,
    trajectory_evidence_to_provenance,
)
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneEvaluation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.continuous_time_lidar_pair_solver import (
    ContinuousTimeLidarPairOptions,
    ContinuousTimeLidarPairProblem,
    ContinuousTimeLidarPairSolver,
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
_TRAJECTORY_FILENAME = "trajectory.json"
OnlineTargetEntry = tuple[int, Vector3, SE3, float, int]


def _dataset_manifest_provenance(config_file: Path) -> dict[str, Any]:
    """Bind a nearby schema-validated dataset manifest to an online run."""

    manifest_path = find_manifest(config_file.parent)
    if manifest_path is None:
        return {}
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError):
        return {
            "dataset_manifest_path": str(manifest_path),
            "dataset_manifest_sha256": sha256_path(manifest_path),
            "dataset_manifest_status": "invalid",
        }
    return {
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": sha256_path(manifest_path),
        "dataset_manifest_status": "validated",
        "dataset_manifest_name": manifest.name,
        "dataset_manifest_provenance": dict(manifest.provenance),
    }


def _solid_state_context_for_config(
    config: CalibrationConfig,
    config_file: Path,
) -> SolidStateLidarCalibrationContext | None:
    """Materialize configured solid-state semantics in an online result."""

    return build_solid_state_context(
        {
            sensor_name: sensor.solid_state
            for sensor_name, sensor in config.sensors.items()
            if sensor.solid_state is not None
        },
        config_sha256=sha256_path(config_file),
        dataset_sha256=sha256_path(Path(config.dataset.path)),
        source_paths=[str(config_file), config.dataset.path],
        tool_version=__version__,
        git_commit=git_commit(),
        notes=[
            "profile copied from the schema-validated calibration config",
            "online adapters may add capture-window timing and temperature telemetry",
        ],
    )


def _attach_online_capture_window(
    result: CalibrationResult,
    config: CalibrationConfig,
    target_sensor: str,
    target_stream: list[OnlineTargetEntry] | None,
    replay_provenance: Mapping[str, Any] | None,
) -> None:
    """Attach observed online capture timing without inventing PCD timestamps."""

    if result.solid_state is None or not target_stream:
        return
    profile = result.solid_state.sensors.get(target_sensor)
    sensor = config.sensors.get(target_sensor)
    if profile is None or sensor is None:
        return

    capture_timestamps = [entry[4] for entry in target_stream]
    timestamp_min = min(capture_timestamps)
    timestamp_max = max(capture_timestamps)
    provenance = replay_provenance or {}
    provenance_prefix = "rosbag2" if "rosbag2_path" in provenance else "rosbag1"
    declared_windows = _capture_windows_from_provenance(
        provenance,
        prefix=provenance_prefix,
    )
    declared_start = _int_or_none(
        provenance.get("rosbag2_first_target_timestamp_ns")
        or provenance.get("rosbag1_first_target_timestamp_ns")
    )
    declared_end = _int_or_none(
        provenance.get("rosbag2_replay_end_timestamp_ns")
        or provenance.get("rosbag1_replay_end_timestamp_ns")
    )
    has_declared_timestamps = declared_start is not None or declared_end is not None
    start_timestamp_ns = declared_start
    end_timestamp_ns = declared_end
    if declared_windows:
        bounded_starts = [start for start, _end in declared_windows if start is not None]
        bounded_ends = [end for _start, end in declared_windows if end is not None]
        if bounded_starts:
            start_timestamp_ns = min(bounded_starts)
        if bounded_ends:
            end_timestamp_ns = max(bounded_ends)
    if start_timestamp_ns is None and not (timestamp_min == timestamp_max == 0):
        start_timestamp_ns = timestamp_min
    if end_timestamp_ns is None and not (timestamp_min == timestamp_max == 0):
        end_timestamp_ns = timestamp_max
    capture_span_s = (
        (end_timestamp_ns - start_timestamp_ns) / 1_000_000_000
        if start_timestamp_ns is not None
        and end_timestamp_ns is not None
        and end_timestamp_ns >= start_timestamp_ns
        else None
    )
    deskew_applied = provenance.get("deskew_applied") is True
    point_time_observed = any(
        value is True
        for value in (
            provenance.get("point_time_observed"),
            (provenance.get("point_time_observed_by_sensor") or {}).get(target_sensor),
        )
    )
    point_time_available = (
        profile.point_time_available
        and point_time_observed
        and sensor.point_time_field is not None
    )
    point_time_range = provenance.get("point_time_range_by_sensor")
    target_point_time_range: Mapping[str, object] = {}
    if isinstance(point_time_range, Mapping):
        candidate_range = point_time_range.get(target_sensor)
        if isinstance(candidate_range, Mapping):
            target_point_time_range = candidate_range
    result.solid_state.capture_windows[target_sensor] = SolidStateCaptureWindow(
        start_timestamp_ns=start_timestamp_ns,
        end_timestamp_ns=end_timestamp_ns,
        reference_timestamp_ns=start_timestamp_ns,
        point_time_field=sensor.point_time_field,
        point_time_offsets_available=point_time_available,
        point_time_deskew_applied=deskew_applied,
        point_time_min_s=_float_or_none(target_point_time_range.get("min_s")),
        point_time_max_s=_float_or_none(target_point_time_range.get("max_s")),
        integration_window_s=profile.integration_window_s,
        capture_span_s=capture_span_s,
        temperature=profile.temperature,
        point_count=len(target_stream),
    )
    result.run.provenance["solid_state_capture_window"] = {
        "sensor": target_sensor,
        "capture_window_count": len(declared_windows),
        "capture_windows": [
            {
                "start_timestamp_ns": start_timestamp_ns,
                "end_timestamp_ns": end_timestamp_ns,
            }
            for start_timestamp_ns, end_timestamp_ns in declared_windows
        ],
        "timestamp_declaration_available": has_declared_timestamps,
        "point_time_offsets_available": point_time_available,
        "point_time_deskew_applied": deskew_applied,
        "capture_span_s": capture_span_s,
    }


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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

    inputs = _solve_inputs(config)
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    factor_options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    temporal_options = temporal_evidence_options_from_factor(factor_options)
    if abs(temporal_options.inject_time_offset_s) > 1.0e-12 and not config.dataset.odometry_topic:
        raise ConfigError(
            "inject_time_offset_s requires dataset.odometry_topic for motion-compensated replay"
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
    replay_provenance: dict[str, Any]
    motion_compensated = False
    odometry_track: OdometryTrack | None = None
    odometry_source_info: OdometrySourceInfo | None = None
    if config.dataset.type == "rosbag1":
        (
            source_records,
            target_stream,
            replay_provenance,
            motion_compensated,
            odometry_track,
            odometry_source_info,
        ) = _load_rosbag1_online_pair(
            config,
            inputs,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
            t_base_source=source_node.transform_to_parent,
        )
    elif config.dataset.type == "rosbag2":
        (
            source_records,
            target_stream,
            replay_provenance,
            motion_compensated,
            odometry_track,
            odometry_source_info,
        ) = _load_rosbag2_online_pair(
            config,
            inputs,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
            t_base_source=source_node.transform_to_parent,
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
        batch_points = [point for _frame_index, point, _pose, _extrap, _capture_ns in chunk]
        batch_poses = [pose for _frame_index, _point, pose, _extrap, _capture_ns in chunk]
        frame_extrapolations = [
            extrap for _frame_index, _point, _pose, extrap, _capture_ns in chunk
        ]
        frame_count = len(
            {frame_index for frame_index, _point, _pose, _extrap, _capture_ns in chunk}
        )
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
        t_source_target_init=t_source_target_init,
        batch_size=batch_size,
        replay_provenance=replay_provenance,
        motion_compensated=motion_compensated,
        target_stream=target_stream,
        odometry_track=odometry_track,
        odometry_source_info=odometry_source_info,
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
    capture_window_start_timestamp_ns: int | None
    capture_window_end_timestamp_ns: int | None
    capture_windows: tuple[tuple[int | None, int | None], ...] = ()
    capture_window_sampling_policy: str = "first"
    capture_window_record_prefilter_margin_s: float = 0.0
    inject_time_offset_s: float = 0.0


def _solve_inputs(config: CalibrationConfig) -> _OnlineSolveInputs:
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    capture_window_start_timestamp_ns = _optional_int_option(
        options, "capture_window_start_timestamp_ns", default=None
    )
    capture_window_end_timestamp_ns = _optional_int_option(
        options, "capture_window_end_timestamp_ns", default=None
    )
    capture_windows = _capture_windows_option(
        options,
        single_start_timestamp_ns=capture_window_start_timestamp_ns,
        single_end_timestamp_ns=capture_window_end_timestamp_ns,
    )
    capture_window_sampling_policy = _capture_window_sampling_policy_option(options)
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
        capture_window_start_timestamp_ns=capture_window_start_timestamp_ns,
        capture_window_end_timestamp_ns=capture_window_end_timestamp_ns,
        capture_windows=capture_windows,
        capture_window_sampling_policy=capture_window_sampling_policy,
        capture_window_record_prefilter_margin_s=(
            _optional_float_option(
                options,
                "capture_window_record_prefilter_margin_s",
                default=None,
                minimum=0.0,
            )
            or 0.0
        ),
        inject_time_offset_s=_signed_float_option(
            options, "inject_time_offset_s", default=0.0
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


def _signed_float_option(options: dict[str, Any], key: str, *, default: float) -> float:
    """Read a finite validation offset while preserving its sign."""

    try:
        value = float(options[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _int_option(options: dict[str, Any], key: str, *, default: int | None) -> int | None:
    if key not in options:
        return default
    try:
        value = int(options[key])
    except (TypeError, ValueError):
        return default
    return value if value > 0 else None


def _seed_option(options: dict[str, Any], key: str, *, default: int) -> int:
    """Read an arbitrary deterministic integer seed from factor options."""

    if key not in options or isinstance(options[key], bool):
        return default
    try:
        return int(options[key])
    except (TypeError, ValueError):
        return default


def _bool_option(options: dict[str, Any], key: str, *, default: bool) -> bool:
    """Read a boolean factor option without treating non-empty strings as true."""

    value = options.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _unit_interval_option(options: dict[str, Any], key: str, *, default: float) -> float:
    """Read a strict fraction used to define the temporal holdout boundary."""

    if key not in options:
        return default
    try:
        value = float(options[key])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a finite fraction in (0, 1)") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ConfigError(f"{key} must be a finite fraction in (0, 1)")
    return value


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


def _optional_int_option(
    options: dict[str, Any], key: str, *, default: int | None
) -> int | None:
    if key not in options:
        return default
    try:
        value = int(options[key])
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _capture_windows_option(
    options: Mapping[str, Any],
    *,
    single_start_timestamp_ns: int | None,
    single_end_timestamp_ns: int | None,
) -> tuple[tuple[int | None, int | None], ...]:
    """Parse the optional multi-window replay selector.

    An empty tuple means that replay is unrestricted. The legacy pair of
    ``capture_window_*`` options is normalized to one window. Multi-window
    entries intentionally retain their input order so overlapping windows have
    deterministic, first-match provenance accounting.
    """

    raw_windows = options.get("capture_windows")
    single_window_requested = (
        single_start_timestamp_ns is not None or single_end_timestamp_ns is not None
    )
    if raw_windows is None:
        if (
            single_start_timestamp_ns is not None
            and single_end_timestamp_ns is not None
            and single_end_timestamp_ns < single_start_timestamp_ns
        ):
            raise ConfigError(
                "capture_window_end_timestamp_ns must be greater than or equal to "
                "capture_window_start_timestamp_ns"
            )
        return (
            ((single_start_timestamp_ns, single_end_timestamp_ns),)
            if single_window_requested
            else ()
        )
    if single_window_requested:
        raise ConfigError(
            "capture_windows cannot be combined with the legacy "
            "capture_window_start_timestamp_ns/capture_window_end_timestamp_ns options"
        )
    if isinstance(raw_windows, (str, bytes)) or not isinstance(raw_windows, (list, tuple)):
        raise ConfigError("factor option capture_windows must be a non-empty list")
    if not raw_windows:
        raise ConfigError("factor option capture_windows must not be empty")

    parsed: list[tuple[int | None, int | None]] = []
    for index, raw_window in enumerate(raw_windows):
        if not isinstance(raw_window, Mapping):
            raise ConfigError(
                f"capture_windows[{index}] must be a mapping with start/end timestamps"
            )
        start_timestamp_ns = _capture_window_bound(
            raw_window.get("start_timestamp_ns"),
            option_name=f"capture_windows[{index}].start_timestamp_ns",
        )
        end_timestamp_ns = _capture_window_bound(
            raw_window.get("end_timestamp_ns"),
            option_name=f"capture_windows[{index}].end_timestamp_ns",
        )
        if start_timestamp_ns is None and end_timestamp_ns is None:
            raise ConfigError(
                f"capture_windows[{index}] must declare start_timestamp_ns or "
                "end_timestamp_ns"
            )
        if (
            start_timestamp_ns is not None
            and end_timestamp_ns is not None
            and end_timestamp_ns < start_timestamp_ns
        ):
            raise ConfigError(
                f"capture_windows[{index}].end_timestamp_ns must be greater than or "
                "equal to start_timestamp_ns"
            )
        parsed.append((start_timestamp_ns, end_timestamp_ns))
    return tuple(parsed)


def _capture_window_sampling_policy_option(options: Mapping[str, Any]) -> str:
    value = options.get("capture_window_sampling_policy", "first")
    if value is None:
        return "first"
    if not isinstance(value, str) or value not in {"first", "evenly_spaced"}:
        raise ConfigError(
            "capture_window_sampling_policy must be 'first' or 'evenly_spaced'"
        )
    return value


def _capture_window_bound(value: object, *, option_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"{option_name} must be a non-negative integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value, 10)
        except ValueError as exc:
            raise ConfigError(f"{option_name} must be a non-negative integer") from exc
    else:
        raise ConfigError(f"{option_name} must be a non-negative integer")
    if parsed < 0:
        raise ConfigError(f"{option_name} must be a non-negative integer")
    return parsed


def _capture_timestamp_in_window(
    timestamp_ns: int,
    *,
    start_timestamp_ns: int | None,
    end_timestamp_ns: int | None,
) -> bool:
    """Return whether a capture timestamp belongs to an inclusive replay window."""

    if start_timestamp_ns is not None and timestamp_ns < start_timestamp_ns:
        return False
    return end_timestamp_ns is None or timestamp_ns <= end_timestamp_ns


def _capture_timestamp_in_windows(
    timestamp_ns: int,
    windows: tuple[tuple[int | None, int | None], ...],
) -> bool:
    """Return whether a timestamp belongs to the configured window union."""

    return not windows or any(
        _capture_timestamp_in_window(
            timestamp_ns,
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
        )
        for start_timestamp_ns, end_timestamp_ns in windows
    )


def _record_timestamp_may_be_in_capture_window(
    timestamp_ns: int,
    *,
    start_timestamp_ns: int | None,
    end_timestamp_ns: int | None,
    margin_s: float,
) -> bool:
    """Keep records near a capture window for exact post-decode timestamp checks."""

    margin_ns = int(max(0.0, margin_s) * 1_000_000_000)
    if start_timestamp_ns is not None and timestamp_ns < start_timestamp_ns - margin_ns:
        return False
    return end_timestamp_ns is None or timestamp_ns <= end_timestamp_ns + margin_ns


def _record_timestamp_may_be_in_capture_windows(
    timestamp_ns: int,
    windows: tuple[tuple[int | None, int | None], ...],
    *,
    margin_s: float,
) -> bool:
    """Keep records near at least one capture window for exact filtering."""

    return not windows or any(
        _record_timestamp_may_be_in_capture_window(
            timestamp_ns,
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
            margin_s=margin_s,
        )
        for start_timestamp_ns, end_timestamp_ns in windows
    )


def _capture_window_index(
    timestamp_ns: int,
    windows: tuple[tuple[int | None, int | None], ...],
) -> int | None:
    """Return the first matching window index for deterministic accounting."""

    for index, (start_timestamp_ns, end_timestamp_ns) in enumerate(windows):
        if _capture_timestamp_in_window(
            timestamp_ns,
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
        ):
            return index
    return None


def _select_record_timestamps_per_capture_window(
    candidates: list[tuple[int, int]],
    windows: tuple[tuple[int | None, int | None], ...],
    *,
    max_samples_per_window: int | None,
) -> set[int]:
    selected: set[int] = set()
    for start_timestamp_ns, end_timestamp_ns in windows:
        window_candidates = sorted(
            (
                (record_timestamp_ns, capture_timestamp_ns)
                for record_timestamp_ns, capture_timestamp_ns in candidates
                if _capture_timestamp_in_window(
                    capture_timestamp_ns,
                    start_timestamp_ns=start_timestamp_ns,
                    end_timestamp_ns=end_timestamp_ns,
                )
            ),
            key=lambda item: item[1],
        )
        if not window_candidates:
            continue
        max_samples = max_samples_per_window or len(window_candidates)
        if len(window_candidates) <= max_samples:
            selected.update(record for record, _capture in window_candidates)
            continue
        capture_timestamps = [capture for _record, capture in window_candidates]
        sampling_start_ns = (
            start_timestamp_ns
            if start_timestamp_ns is not None
            else capture_timestamps[0]
        )
        sampling_end_ns = (
            end_timestamp_ns
            if end_timestamp_ns is not None
            else capture_timestamps[-1]
        )
        selected_captures = set(
            _select_evenly_spaced_timestamps(
                capture_timestamps,
                window_start_ns=sampling_start_ns,
                window_end_ns=sampling_end_ns,
                max_samples=max_samples,
            )
        )
        selected_count = 0
        for record_timestamp_ns, capture_timestamp_ns in window_candidates:
            if capture_timestamp_ns in selected_captures:
                selected.add(record_timestamp_ns)
                selected_count += 1
                if selected_count >= max_samples:
                    break
    return selected


def _select_rosbag2_capture_record_timestamps(
    bag_path: Path,
    *,
    source_topic: str,
    target_topic: str,
    source_point_time_field: str | None,
    target_point_time_field: str | None,
    windows: tuple[tuple[int | None, int | None], ...],
    max_source_messages: int | None,
    max_target_messages: int | None,
) -> tuple[set[int], set[int]]:
    """Select bag records evenly in each window before full point decoding."""

    source_candidates: list[tuple[int, int]] = []
    target_candidates: list[tuple[int, int]] = []
    for connection, timestamp_ns, data in iter_rosbag2_messages(
        bag_path,
        topics={source_topic, target_topic},
    ):
        if connection.message_type not in ROS2_LIDAR_MESSAGE_TYPES:
            continue
        if connection.topic == source_topic:
            message = decode_rosbag2_lidar_message(
                connection.topic,
                connection.message_type,
                timestamp_ns,
                data,
                point_time_field=source_point_time_field,
            )
            source_candidates.append(
                (timestamp_ns, _rosbag2_message_timestamp_ns(message, timestamp_ns))
            )
        elif connection.topic == target_topic:
            message = decode_rosbag2_lidar_message(
                connection.topic,
                connection.message_type,
                timestamp_ns,
                data,
                point_time_field=target_point_time_field,
            )
            target_candidates.append(
                (timestamp_ns, _rosbag2_message_timestamp_ns(message, timestamp_ns))
            )
    return (
        _select_record_timestamps_per_capture_window(
            source_candidates,
            windows,
            max_samples_per_window=max_source_messages,
        ),
        _select_record_timestamps_per_capture_window(
            target_candidates,
            windows,
            max_samples_per_window=max_target_messages,
        ),
    )


def _new_capture_window_stats(
    windows: tuple[tuple[int | None, int | None], ...],
) -> list[dict[str, Any]]:
    return [
        {
            "window_index": index,
            "start_timestamp_ns": start_timestamp_ns,
            "end_timestamp_ns": end_timestamp_ns,
            "selected_source_message_count": 0,
            "selected_source_point_count": 0,
            "selected_target_message_count": 0,
            "selected_target_point_count": 0,
        }
        for index, (start_timestamp_ns, end_timestamp_ns) in enumerate(windows)
    ]


def _refresh_capture_window_source_stats(
    stats: list[dict[str, Any]],
    windows: tuple[tuple[int | None, int | None], ...],
    source_record_windows: list[tuple[int, list[LivoxPointRecord]]],
) -> None:
    for item in stats:
        item["selected_source_message_count"] = 0
        item["selected_source_point_count"] = 0
    for timestamp_ns, records in source_record_windows:
        index = _capture_window_index(timestamp_ns, windows)
        if index is None:
            continue
        stats[index]["selected_source_message_count"] += 1
        stats[index]["selected_source_point_count"] += len(records)


def _capture_windows_from_provenance(
    provenance: Mapping[str, Any] | None,
    *,
    prefix: str,
) -> tuple[tuple[int | None, int | None], ...]:
    """Recover normalized replay windows for downstream evidence adapters."""

    if provenance is None:
        return ()
    raw_windows = provenance.get(f"{prefix}_capture_windows")
    if isinstance(raw_windows, list):
        windows: list[tuple[int | None, int | None]] = []
        for raw_window in raw_windows:
            if not isinstance(raw_window, Mapping):
                continue
            start = _int_or_none(raw_window.get("start_timestamp_ns"))
            end = _int_or_none(raw_window.get("end_timestamp_ns"))
            if start is not None or end is not None:
                windows.append((start, end))
        if windows:
            return tuple(windows)
    start = _int_or_none(provenance.get(f"{prefix}_capture_window_start_timestamp_ns"))
    end = _int_or_none(provenance.get(f"{prefix}_capture_window_end_timestamp_ns"))
    return ((start, end),) if start is not None or end is not None else ()


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
        stream.extend((frame_index, point, SE3.identity(), 0.0, 0) for point in points)
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

    # A fixed source and an explicitly estimated target may be connected through
    # a non-LiDAR base frame (or the target may be a descendant of the source).
    # Prefer those declarations over lexical order; otherwise a frame graph such
    # as base_link -> velodyne -> ouster can accidentally estimate the wrong edge.
    estimating_lidars = [
        name
        for name in lidars
        if frame_graph.nodes.get(name) is not None and frame_graph.nodes[name].estimate
    ]
    fixed_lidars = [
        name
        for name in lidars
        if frame_graph.nodes.get(name) is not None and not frame_graph.nodes[name].estimate
    ]
    if len(estimating_lidars) == 1 and len(fixed_lidars) == 1:
        return fixed_lidars[0], estimating_lidars[0]

    # Dual-base-link layouts (A2D2, Livox PCD pair) keep the historical sorted order
    # when the config does not identify a unique source/target pair.
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


def _point_time_deskew_enabled(config: CalibrationConfig) -> bool:
    """Return whether a replay should consume declared per-point offsets."""

    factor = config.pipeline.factors.get(_FACTOR_NAME)
    options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    value = options.get("use_point_time_offsets", True)
    return value if isinstance(value, bool) else True


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


def _target_capture_ns(capture_ns: int, *, inject_time_offset_s: float) -> int:
    if abs(inject_time_offset_s) < 1.0e-12:
        return capture_ns
    return apply_time_offset_ns(capture_ns, inject_time_offset_s)


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


def _update_point_time_range(
    offsets: Any,
    *,
    point_time_min: float | None,
    point_time_max: float | None,
) -> tuple[float | None, float | None]:
    """Accumulate observed per-point offset bounds in seconds."""

    if offsets is None or int(offsets.shape[0]) == 0:
        return point_time_min, point_time_max
    lo = float(offsets.min())
    hi = float(offsets.max())
    point_time_min = lo if point_time_min is None else min(point_time_min, lo)
    point_time_max = hi if point_time_max is None else max(point_time_max, hi)
    return point_time_min, point_time_max


def _deskew_pointcloud_to_world_records(
    xyz: Any,
    offsets_s: Any,
    reference_timestamp_ns: int,
    t_base_source: SE3,
    odometry_track: OdometryTrack,
    max_points: int | None,
    extrapolation_tolerance_s: float | None,
) -> list[LivoxPointRecord]:
    count = int(xyz.shape[0])
    records: list[LivoxPointRecord] = []
    for index in _subsample_indices(count, max_points):
        capture_ns = _capture_timestamp_ns(reference_timestamp_ns, float(offsets_s[index]))
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
    reference_timestamp_ns: int,
    t_base_source: SE3,
    odometry_track: OdometryTrack,
    max_points: int | None,
    inject_time_offset_s: float = 0.0,
) -> list[OnlineTargetEntry]:
    count = int(xyz.shape[0])
    entries: list[OnlineTargetEntry] = []
    for index in _subsample_indices(count, max_points):
        capture_ns = _capture_timestamp_ns(reference_timestamp_ns, float(offsets_s[index]))
        capture_ns = _target_capture_ns(capture_ns, inject_time_offset_s=inject_time_offset_s)
        t_world_base, _clamped, extrapolation_s = odometry_track.interpolate(capture_ns)
        t_world_source = t_world_base.compose(t_base_source)
        point = (
            float(xyz[index, 0]),
            float(xyz[index, 1]),
            float(xyz[index, 2]),
        )
        entries.append((frame_index, point, t_world_source, extrapolation_s, capture_ns))
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
    t_base_source: SE3,
) -> tuple[
    list[LivoxPointRecord],
    list[OnlineTargetEntry],
    dict[str, Any],
    bool,
    OdometryTrack | None,
    OdometrySourceInfo | None,
]:
    """Stream one ROS bag once, building a bounded source map and target stream."""

    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        return (
            [],
            [],
            {"rosbag1_path": str(bag_path), "rosbag1_exists": False},
            False,
            None,
            None,
        )

    source_topic = _lidar_topic(config, source_sensor)
    target_topic = _lidar_topic(config, target_sensor)
    topics = {source_topic, target_topic}
    odometry_track, odometry_source_info = _load_rosbag1_odometry_track(
        bag_path,
        config.dataset.odometry_topic,
        preprocessing=config.dataset.odometry_preprocessing,
    )
    motion_compensated = odometry_track is not None
    extrapolation_tolerance_s = _odometry_extrapolation_tolerance_s(
        config, motion_compensated
    )
    if config.dataset.odometry_topic:
        topics.add(config.dataset.odometry_topic)
    deskew_time_field = _deskew_time_field_map(config, source_sensor, target_sensor)
    point_time_deskew_enabled = _point_time_deskew_enabled(config)
    source_time_field = deskew_time_field.get(source_sensor)
    target_time_field = deskew_time_field.get(target_sensor)
    source_deskew = (
        motion_compensated
        and point_time_deskew_enabled
        and source_time_field is not None
    )
    target_deskew = (
        motion_compensated
        and point_time_deskew_enabled
        and target_time_field is not None
    )
    deskew_applied = source_deskew or target_deskew
    deskew_span_min: float | None = None
    deskew_span_max: float | None = None
    point_time_observed_by_sensor = {source_sensor: False, target_sensor: False}
    point_time_reference_by_sensor: dict[str, str] = {}
    point_time_clock_mapping_by_sensor: dict[str, dict[str, Any]] = {}
    point_time_min_by_sensor: dict[str, float | None] = {
        source_sensor: None,
        target_sensor: None,
    }
    point_time_max_by_sensor: dict[str, float | None] = {
        source_sensor: None,
        target_sensor: None,
    }

    source_records: list[LivoxPointRecord] = []
    source_record_windows: list[tuple[int, list[LivoxPointRecord]]] = []
    target_stream: list[OnlineTargetEntry] = []
    source_messages = 0
    target_messages = 0
    capture_window_stats = _new_capture_window_stats(inputs.capture_windows)
    frame_index = 0
    replay_start_ns: int | None = None
    replay_end_ns: int | None = None
    first_source_ns: int | None = None
    first_target_ns: int | None = None

    for connection, timestamp_ns, data in iter_messages(bag_path, topics=topics):
        if config.dataset.odometry_topic and connection.topic == config.dataset.odometry_topic:
            continue
        if connection.message_type not in LIDAR_MESSAGE_TYPES:
            continue
        topic = connection.topic
        if topic == source_topic:
            if not _record_timestamp_may_be_in_capture_windows(
                timestamp_ns,
                inputs.capture_windows,
                margin_s=inputs.capture_window_record_prefilter_margin_s,
            ):
                continue
            message = _decode_rosbag1_online_lidar(
                topic,
                connection.message_type,
                timestamp_ns,
                data,
                point_time_field=source_time_field,
            )
            # Livox ROS1 CustomMsg headers in TIERS can remain in the sensor's
            # since-boot clock while odometry and bag replay use ROS epoch time.
            # _rosbag1_point_time_data already maps point offsets into the bag
            # record clock, so message-level temporal partitioning must use the
            # same authority rather than the raw CustomMsg header.
            source_capture_ns = int(timestamp_ns)
            source_window_index = _capture_window_index(
                source_capture_ns, inputs.capture_windows
            )
            if inputs.capture_windows and source_window_index is None:
                continue
            selected_source_messages = (
                source_messages
                if source_window_index is None
                else int(
                    capture_window_stats[source_window_index][
                        "selected_source_message_count"
                    ]
                )
            )
            selected_source_points = (
                len(source_records)
                if source_window_index is None
                else int(
                    capture_window_stats[source_window_index][
                        "selected_source_point_count"
                    ]
                )
            )
            if (
                inputs.max_source_messages is not None
                and selected_source_messages >= inputs.max_source_messages
            ):
                continue
            if (
                inputs.max_source_points is not None
                and selected_source_points >= inputs.max_source_points
            ):
                continue
            first_source_ns = first_source_ns or source_capture_ns
            remaining = (
                None
                if inputs.max_source_points is None
                else max(0, inputs.max_source_points - selected_source_points)
            )
            offsets_s, point_time_reference_ns, point_time_reference = (
                _rosbag1_point_time_data(
                    message, source_time_field, ros_timestamp_ns=timestamp_ns
                )
            )
            _record_rosbag1_point_time_mapping(
                point_time_clock_mapping_by_sensor,
                source_sensor,
                message,
                ros_timestamp_ns=timestamp_ns,
            )
            if offsets_s is not None:
                point_time_observed_by_sensor[source_sensor] = True
                (
                    point_time_min_by_sensor[source_sensor],
                    point_time_max_by_sensor[source_sensor],
                ) = _update_point_time_range(
                    offsets_s,
                    point_time_min=point_time_min_by_sensor[source_sensor],
                    point_time_max=point_time_max_by_sensor[source_sensor],
                )
                if point_time_reference is not None:
                    point_time_reference_by_sensor[source_sensor] = point_time_reference
            scan_records: list[LivoxPointRecord]
            if source_deskew and odometry_track is not None:
                if offsets_s is None or point_time_reference_ns is None:
                    raise DatasetError(
                        f"deskew requested for source sensor {source_sensor!r} but "
                        "per-point time offsets were not decoded"
                    )
                deskew_span_min, deskew_span_max = _update_deskew_span(
                    offsets_s,
                    span_min=deskew_span_min,
                    span_max=deskew_span_max,
                )
                scan_records = _deskew_pointcloud_to_world_records(
                    message.xyz,
                    offsets_s,
                    point_time_reference_ns,
                    t_base_source,
                    odometry_track,
                    remaining,
                    extrapolation_tolerance_s,
                )
            elif motion_compensated and odometry_track is not None:
                t_world_base, _clamped, extrapolation_s = odometry_track.interpolate(
                    timestamp_ns
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
                scan_records = _transform_pointcloud_to_world_records(
                    message.xyz,
                    t_world_source,
                    remaining,
                )
            else:
                scan_records = _pointcloud_xyz_to_records(message.xyz, remaining)
            source_records.extend(scan_records)
            source_record_windows.append((source_capture_ns, scan_records))
            source_messages += 1
            if source_window_index is not None:
                capture_window_stats[source_window_index][
                    "selected_source_message_count"
                ] += 1
                capture_window_stats[source_window_index][
                    "selected_source_point_count"
                ] += len(scan_records)
            continue

        if topic != target_topic:
            continue

        if not _record_timestamp_may_be_in_capture_windows(
            timestamp_ns,
            inputs.capture_windows,
            margin_s=inputs.capture_window_record_prefilter_margin_s,
        ):
            continue
        message = _decode_rosbag1_online_lidar(
            topic,
            connection.message_type,
            timestamp_ns,
            data,
            point_time_field=target_time_field,
        )
        # Keep message-level replay and temporal holdout in the ROS bag-record
        # clock domain. A Livox CustomMsg header may be a since-boot timestamp.
        target_capture_ns = int(timestamp_ns)
        target_window_index = _capture_window_index(
            target_capture_ns, inputs.capture_windows
        )
        if inputs.capture_windows and target_window_index is None:
            continue
        selected_target_messages = (
            target_messages
            if target_window_index is None
            else int(
                capture_window_stats[target_window_index][
                    "selected_target_message_count"
                ]
            )
        )
        if (
            inputs.max_target_messages is not None
            and selected_target_messages >= inputs.max_target_messages
        ):
            if not inputs.capture_windows:
                break
            continue
        if replay_start_ns is None:
            replay_start_ns = target_capture_ns
            first_target_ns = target_capture_ns
        replay_end_ns = target_capture_ns
        if inputs.max_replay_duration_s is not None:
            elapsed_ns = target_capture_ns - replay_start_ns
            if elapsed_ns > int(inputs.max_replay_duration_s * 1_000_000_000):
                break
        target_points_before = len(target_stream)
        offsets_s, point_time_reference_ns, point_time_reference = (
            _rosbag1_point_time_data(
                message, target_time_field, ros_timestamp_ns=timestamp_ns
            )
        )
        _record_rosbag1_point_time_mapping(
            point_time_clock_mapping_by_sensor,
            target_sensor,
            message,
            ros_timestamp_ns=timestamp_ns,
        )
        if offsets_s is not None:
            point_time_observed_by_sensor[target_sensor] = True
            (
                point_time_min_by_sensor[target_sensor],
                point_time_max_by_sensor[target_sensor],
            ) = _update_point_time_range(
                offsets_s,
                point_time_min=point_time_min_by_sensor[target_sensor],
                point_time_max=point_time_max_by_sensor[target_sensor],
            )
            if point_time_reference is not None:
                point_time_reference_by_sensor[target_sensor] = point_time_reference
        if target_deskew and odometry_track is not None:
            if offsets_s is None or point_time_reference_ns is None:
                raise DatasetError(
                    f"deskew requested for target sensor {target_sensor!r} but "
                    "per-point time offsets were not decoded"
                )
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
                    point_time_reference_ns,
                    t_base_source,
                    odometry_track,
                    inputs.max_target_points,
                    inject_time_offset_s=inputs.inject_time_offset_s,
                )
            )
        elif motion_compensated and odometry_track is not None:
            points = _pointcloud_xyz_to_vectors(message.xyz, inputs.max_target_points)
            capture_ns = _target_capture_ns(
                timestamp_ns,
                inject_time_offset_s=inputs.inject_time_offset_s,
            )
            t_world_base, _clamped, extrapolation_s = odometry_track.interpolate(capture_ns)
            if (
                extrapolation_tolerance_s is not None
                and extrapolation_s > extrapolation_tolerance_s
            ):
                msg = (
                    "target LiDAR message timestamp lies "
                    f"{extrapolation_s:.4f} s outside the odometry track "
                    f"(tolerance {extrapolation_tolerance_s:.4f} s); "
                    "motion-compensated replay premise is broken"
                )
                raise DatasetError(msg)
            t_world_source = t_world_base.compose(t_base_source)
            target_stream.extend(
                (frame_index, point, t_world_source, extrapolation_s, capture_ns)
                for point in points
            )
        else:
            points = _pointcloud_xyz_to_vectors(message.xyz, inputs.max_target_points)
            capture_ns = _target_capture_ns(
                timestamp_ns,
                inject_time_offset_s=inputs.inject_time_offset_s,
            )
            target_stream.extend(
                (frame_index, point, SE3.identity(), 0.0, capture_ns) for point in points
            )
        frame_index += 1
        target_messages += 1
        if target_window_index is not None:
            capture_window_stats[target_window_index][
                "selected_target_message_count"
            ] += 1
            capture_window_stats[target_window_index][
                "selected_target_point_count"
            ] += len(target_stream) - target_points_before

    replay_duration_s: float | None = None
    if replay_start_ns is not None and replay_end_ns is not None:
        replay_duration_s = (replay_end_ns - replay_start_ns) / 1_000_000_000

    target_frame_ids = sorted(
        {frame_index for frame_index, _point, _pose, _extrap, _capture in target_stream}
    )
    target_train_window_count = 0
    target_holdout_window_count = 0
    temporal_holdout_independent = False
    source_map_temporal_boundary_ns: int | None = None
    source_map_temporal_boundary_applied = False
    if len(target_frame_ids) >= 2:
        holdout_start_index = max(1, int(len(target_frame_ids) * 0.8))
        holdout_frame_ids = set(target_frame_ids[holdout_start_index:])
        holdout_entries = [
            entry for entry in target_stream if entry[0] in holdout_frame_ids
        ]
        source_map_temporal_boundary_ns = min(
            (entry[4] for entry in holdout_entries), default=None
        )
        target_train_window_count = holdout_start_index
        target_holdout_window_count = len(holdout_frame_ids)
        if source_map_temporal_boundary_ns is not None:
            eligible_source_windows = [
                window
                for window in source_record_windows
                if window[0] < source_map_temporal_boundary_ns
            ]
            if eligible_source_windows:
                source_records = [
                    record
                    for _timestamp, records in eligible_source_windows
                    for record in records
                ]
                source_record_windows = eligible_source_windows
                source_messages = len(eligible_source_windows)
                source_map_temporal_boundary_applied = True
                temporal_holdout_independent = True

    _refresh_capture_window_source_stats(
        capture_window_stats,
        inputs.capture_windows,
        source_record_windows,
    )

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
        "rosbag1_capture_window_start_timestamp_ns": (
            inputs.capture_window_start_timestamp_ns
        ),
        "rosbag1_capture_window_end_timestamp_ns": inputs.capture_window_end_timestamp_ns,
        "rosbag1_capture_window_count": len(inputs.capture_windows),
        "rosbag1_capture_windows": capture_window_stats,
        "rosbag1_capture_window_record_prefilter_margin_s": (
            inputs.capture_window_record_prefilter_margin_s
        ),
        "rosbag1_capture_window_selection_policy": (
            "union of inclusive LiDAR capture timestamp windows; overlapping windows "
            "use first-match provenance accounting; source/target message and "
            "source-point limits are applied per window"
        ),
        "rosbag1_capture_window_sampling_policy": inputs.capture_window_sampling_policy,
        "rosbag1_source_map_temporal_boundary_timestamp_ns": (
            source_map_temporal_boundary_ns
        ),
        "rosbag1_source_map_temporal_boundary_applied": (
            source_map_temporal_boundary_applied
        ),
        "rosbag1_source_map_window_count": len(source_record_windows),
        "rosbag1_target_train_window_count": target_train_window_count,
        "rosbag1_target_holdout_window_count": target_holdout_window_count,
        "rosbag1_temporal_holdout_independent": temporal_holdout_independent,
        "rosbag1_temporal_holdout_clock_note": (
            "boundary uses ROS bag record timestamps; Livox per-point offsets are "
            "mapped into that clock before interpreting temporal independence"
        ),
        "rosbag1_capture_clock": (
            "bag_record_timestamp_ns for message-level replay; Livox offset_time "
            "is added to the same clock for per-point deskew"
        ),
        "motion_compensated": motion_compensated,
        "point_time_deskew_enabled": _point_time_deskew_enabled(config),
        "point_time_observed_by_sensor": point_time_observed_by_sensor,
        "point_time_reference_by_sensor": point_time_reference_by_sensor,
        "point_time_clock_mapping_by_sensor": _finalize_point_time_mappings(
            point_time_clock_mapping_by_sensor
        ),
        "point_time_range_by_sensor": {
            sensor_name: {
                "min_s": point_time_min_by_sensor[sensor_name],
                "max_s": point_time_max_by_sensor[sensor_name],
            }
            for sensor_name in point_time_observed_by_sensor
            if point_time_min_by_sensor[sensor_name] is not None
            and point_time_max_by_sensor[sensor_name] is not None
        },
    }
    if config.dataset.odometry_topic:
        provenance.update(
            _odometry_replay_provenance(
                config.dataset.odometry_topic,
                odometry_track,
                motion_compensated,
            )
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
    if abs(inputs.inject_time_offset_s) > 1.0e-12:
        provenance["time_offset_injected_s"] = inputs.inject_time_offset_s
        provenance["time_offset_injection_warning"] = (
            "VALIDATION-ONLY: target capture timestamps were shifted at load time"
        )
    return (
        source_records,
        target_stream,
        provenance,
        motion_compensated,
        odometry_track,
        odometry_source_info,
    )


def _decode_rosbag1_online_lidar(
    topic: str,
    message_type: str,
    timestamp_ns: int,
    data: bytes,
    *,
    point_time_field: str | None,
) -> Any:
    """Decode a ROS1 LiDAR message with an optional declared time field."""

    if message_type == ROS1_POINTCLOUD2_TYPE:
        return decode_ros1_pointcloud2(
            topic,
            timestamp_ns,
            data,
            point_time_field=point_time_field,
        )
    return decode_bag_lidar_message(topic, message_type, timestamp_ns, data)


def _decode_rosbag2_online_lidar(
    topic: str,
    message_type: str,
    timestamp_ns: int,
    data: bytes,
    *,
    point_time_field: str | None,
) -> PointCloud2Message | LivoxCustomMessage:
    """Decode a ROS2 PointCloud2 or Livox CustomMsg for online replay."""

    return decode_rosbag2_lidar_message(
        topic,
        message_type,
        timestamp_ns,
        data,
        point_time_field=point_time_field,
    )


def _rosbag2_point_time_data(
    message: PointCloud2Message | LivoxCustomMessage,
    point_time_field: str | None,
    *,
    ros_timestamp_ns: int,
) -> tuple[Any | None, int | None, str | None]:
    """Map ROS2 LiDAR point-time fields into the recording clock domain."""

    if point_time_field is None:
        return None, None, None
    if isinstance(message, LivoxCustomMessage):
        if point_time_field != "offset_time":
            raise DatasetError(
                "Livox CustomMsg exposes per-point time as 'offset_time'; "
                f"got {point_time_field!r}"
            )
        if message.offset_time_ns is None:
            return None, message.timestamp_ns, "timebase_unavailable"
        return (
            message.offset_time_ns.astype(float) * 1.0e-9,
            message.point_time_reference_ns,
            "timebase_native_ros_epoch",
        )
    if isinstance(message, PointCloud2Message):
        return (
            message.point_time_offsets_s,
            message.timestamp_ns,
            "message_stamp",
        )
    raise DatasetError(f"unsupported ROS2 LiDAR message object: {type(message)!r}")


def _rosbag2_message_timestamp_ns(
    message: PointCloud2Message | LivoxCustomMessage,
    ros_timestamp_ns: int,
) -> int:
    """Return the capture-clock authority for a ROS2 LiDAR message."""

    if isinstance(message, LivoxCustomMessage):
        return message.point_time_reference_ns
    return message.timestamp_ns


def _record_rosbag2_point_time_mapping(
    mappings: dict[str, dict[str, Any]],
    sensor_name: str,
    message: PointCloud2Message | LivoxCustomMessage,
    *,
    ros_timestamp_ns: int,
) -> None:
    """Record how a Livox sensor clock maps into ROS2 bag time."""

    if not isinstance(message, LivoxCustomMessage) or message.timebase_ns <= 0:
        return
    header_offset_ns = int(message.timestamp_ns - message.timebase_ns)
    record_delay_ns = int(ros_timestamp_ns - message.timebase_ns)
    entry = mappings.setdefault(
        sensor_name,
        {
            "source": "livox_custommsg_timebase_and_header_stamp",
            "raw_reference": "Livox CustomMsg timebase",
            "mapped_reference": "ROS2 header/odometry clock",
            "mapping_policy": (
                "capture_ns = timebase_ns + offset_time_ns; header.stamp/timebase "
                "is already in the ROS epoch; bag record delay is retained separately"
            ),
            "sample_count": 0,
            "offset_min_ns": header_offset_ns,
            "offset_max_ns": header_offset_ns,
            "nominal_offset_ns": header_offset_ns,
            "residual_max_abs_ns": 0,
            "record_transport_delay_min_ns": record_delay_ns,
            "record_transport_delay_max_ns": record_delay_ns,
        },
    )
    entry["sample_count"] = int(entry["sample_count"]) + 1
    entry["offset_min_ns"] = min(int(entry["offset_min_ns"]), header_offset_ns)
    entry["offset_max_ns"] = max(int(entry["offset_max_ns"]), header_offset_ns)
    entry["record_transport_delay_min_ns"] = min(
        int(entry["record_transport_delay_min_ns"]), record_delay_ns
    )
    entry["record_transport_delay_max_ns"] = max(
        int(entry["record_transport_delay_max_ns"]), record_delay_ns
    )
    residual = abs(header_offset_ns - int(entry["nominal_offset_ns"]))
    entry["residual_max_abs_ns"] = max(
        int(entry["residual_max_abs_ns"]), residual
    )


def _rosbag1_point_time_data(
    message: Any,
    point_time_field: str | None,
    *,
    ros_timestamp_ns: int,
) -> tuple[Any | None, int | None, str | None]:
    """Return point offsets mapped into the ROS bag-record clock domain."""

    if point_time_field is None:
        return None, None, None
    if isinstance(message, LivoxCustomMessage):
        if point_time_field != "offset_time":
            raise DatasetError(
                "Livox CustomMsg exposes per-point time as 'offset_time'; "
                f"got {point_time_field!r}"
            )
        if message.offset_time_ns is None:
            return None, ros_timestamp_ns, "timebase_unavailable"
        return (
            message.offset_time_ns.astype(float) * 1.0e-9,
            ros_timestamp_ns,
            "timebase_mapped_to_ros_bag_timestamp",
        )
    if isinstance(message, PointCloud2Message):
        return (
            message.point_time_offsets_s,
            ros_timestamp_ns,
            "message_stamp_mapped_to_ros_bag_timestamp",
        )
    raise DatasetError(f"unsupported ROS1 LiDAR message object: {type(message)!r}")


def _record_rosbag1_point_time_mapping(
    mappings: dict[str, dict[str, Any]],
    sensor_name: str,
    message: Any,
    *,
    ros_timestamp_ns: int,
) -> None:
    """Record how Livox's raw sensor clock maps into ROS bag time."""

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


def _finalize_point_time_mappings(
    mappings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Add a conservative stability status to point-time clock evidence."""

    finalized: dict[str, dict[str, Any]] = {}
    for sensor_name, raw in mappings.items():
        item = dict(raw)
        residual_ns = int(item["residual_max_abs_ns"])
        item["residual_max_abs_s"] = residual_ns * 1.0e-9
        item["status"] = "stable" if residual_ns <= 20_000_000 else "unstable"
        finalized[sensor_name] = item
    return finalized


def _load_rosbag1_odometry_track(
    bag_path: Path,
    odometry_topic: str | None,
    *,
    preprocessing: OdometryPreprocessingConfig | None = None,
) -> tuple[OdometryTrack | None, OdometrySourceInfo | None]:
    """Load a ROS1 PoseStamped stream as an odometry interpolation track."""

    if not odometry_topic:
        return None, None
    samples: list[OdometryPoseSample] = []
    world_frame_id = ""
    for connection, timestamp_ns, data in iter_messages(
        bag_path,
        topics={odometry_topic},
    ):
        if connection.message_type != POSE_STAMPED_TYPE:
            continue
        message = decode_pose_stamped(connection.topic, timestamp_ns, data)
        if not world_frame_id:
            world_frame_id = message.frame_id
        samples.append(
            OdometryPoseSample(
                # The bag record timestamp is the ROS clock authority. Some
                # recordings contain stale or sensor-domain embedded stamps.
                timestamp_ns=timestamp_ns,
                pose=SE3(message.position, message.orientation_xyzw),
            )
        )
    if not samples:
        return None, None
    return (
        OdometryTrack(
            samples,
            burst_policy=(preprocessing.burst_policy if preprocessing else "preserve"),
            min_interval_s=(preprocessing.min_interval_s if preprocessing else 0.001),
        ),
        OdometrySourceInfo(
            topic=odometry_topic,
            world_frame_id=world_frame_id,
            child_frame_id="pose_stamped",
        ),
    )


def _load_rosbag2_online_pair(
    config: CalibrationConfig,
    inputs: _OnlineSolveInputs,
    *,
    source_sensor: str,
    target_sensor: str,
    t_base_source: SE3,
) -> tuple[
    list[LivoxPointRecord],
    list[OnlineTargetEntry],
    dict[str, Any],
    bool,
    OdometryTrack | None,
    OdometrySourceInfo | None,
]:
    """Stream one rosbag2 recording once, building a bounded source map and target stream."""

    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        return (
            [],
            [],
            {"rosbag2_path": str(bag_path), "rosbag2_exists": False},
            False,
            None,
            None,
        )

    source_topic = _lidar_topic(config, source_sensor)
    target_topic = _lidar_topic(config, target_sensor)
    odometry_topic = config.dataset.odometry_topic
    topics = {source_topic, target_topic}
    if odometry_topic:
        topics.add(odometry_topic)
    deskew_time_field = _deskew_time_field_map(config, source_sensor, target_sensor)
    source_time_field = deskew_time_field.get(source_sensor)
    target_time_field = deskew_time_field.get(target_sensor)
    evenly_spaced_sampling = bool(
        inputs.capture_windows
        and inputs.capture_window_sampling_policy == "evenly_spaced"
    )
    selected_source_record_timestamps: set[int] = set()
    selected_target_record_timestamps: set[int] = set()
    if evenly_spaced_sampling:
        (
            selected_source_record_timestamps,
            selected_target_record_timestamps,
        ) = _select_rosbag2_capture_record_timestamps(
            bag_path,
            source_topic=source_topic,
            target_topic=target_topic,
            source_point_time_field=source_time_field,
            target_point_time_field=target_time_field,
            windows=inputs.capture_windows,
            max_source_messages=inputs.max_source_messages,
            max_target_messages=inputs.max_target_messages,
        )

    odometry_track, odometry_source_info = _load_rosbag2_odometry_track(
        bag_path,
        odometry_topic,
        preprocessing=config.dataset.odometry_preprocessing,
    )
    motion_compensated = odometry_track is not None
    extrapolation_tolerance_s = _odometry_extrapolation_tolerance_s(config, motion_compensated)
    if (
        inputs.capture_window_start_timestamp_ns is not None
        and inputs.capture_window_end_timestamp_ns is not None
        and inputs.capture_window_end_timestamp_ns
        < inputs.capture_window_start_timestamp_ns
    ):
        raise DatasetError(
            "capture_window_end_timestamp_ns must be greater than or equal to "
            "capture_window_start_timestamp_ns"
        )

    point_time_deskew_enabled = _point_time_deskew_enabled(config)
    source_deskew = (
        motion_compensated
        and point_time_deskew_enabled
        and source_time_field is not None
    )
    target_deskew = (
        motion_compensated
        and point_time_deskew_enabled
        and target_time_field is not None
    )
    deskew_applied = source_deskew or target_deskew
    deskew_span_min: float | None = None
    deskew_span_max: float | None = None
    point_time_observed_by_sensor = {source_sensor: False, target_sensor: False}
    point_time_reference_by_sensor: dict[str, str] = {}
    point_time_clock_mapping_by_sensor: dict[str, dict[str, Any]] = {}
    point_time_min_by_sensor: dict[str, float | None] = {
        source_sensor: None,
        target_sensor: None,
    }
    point_time_max_by_sensor: dict[str, float | None] = {
        source_sensor: None,
        target_sensor: None,
    }

    source_records: list[LivoxPointRecord] = []
    source_record_windows: list[tuple[int, list[LivoxPointRecord]]] = []
    target_stream: list[OnlineTargetEntry] = []
    source_messages = 0
    target_messages = 0
    source_raw_point_count = 0
    target_raw_point_count = 0
    source_nonfinite_xyz_count = 0
    target_nonfinite_xyz_count = 0
    capture_window_stats = _new_capture_window_stats(inputs.capture_windows)
    frame_index = 0
    replay_start_ns: int | None = None
    replay_end_ns: int | None = None
    first_source_ns: int | None = None
    first_target_ns: int | None = None
    source_message_type: str | None = None
    target_message_type: str | None = None

    for connection, timestamp_ns, data in iter_rosbag2_messages(bag_path, topics=topics):
        if odometry_topic and connection.topic == odometry_topic:
            continue
        if connection.message_type not in ROS2_LIDAR_MESSAGE_TYPES:
            continue
        topic = connection.topic
        if topic == source_topic:
            if evenly_spaced_sampling and timestamp_ns not in selected_source_record_timestamps:
                continue
            message = _decode_rosbag2_online_lidar(
                topic,
                connection.message_type,
                timestamp_ns,
                data,
                point_time_field=source_time_field,
            )
            source_capture_ns = _rosbag2_message_timestamp_ns(message, timestamp_ns)
            source_window_index = _capture_window_index(
                source_capture_ns, inputs.capture_windows
            )
            if inputs.capture_windows and source_window_index is None:
                continue
            selected_source_messages = (
                source_messages
                if source_window_index is None
                else int(
                    capture_window_stats[source_window_index][
                        "selected_source_message_count"
                    ]
                )
            )
            selected_source_points = (
                len(source_records)
                if source_window_index is None
                else int(
                    capture_window_stats[source_window_index][
                        "selected_source_point_count"
                    ]
                )
            )
            if (
                inputs.max_source_messages is not None
                and selected_source_messages >= inputs.max_source_messages
            ):
                continue
            if (
                inputs.max_source_points is not None
                and selected_source_points >= inputs.max_source_points
            ):
                continue
            source_raw_point_count += int(
                getattr(message, "raw_point_count", message.point_count)
            )
            source_nonfinite_xyz_count += int(
                getattr(message, "nonfinite_xyz_count", 0)
            )
            source_message_type = connection.message_type
            first_source_ns = first_source_ns or source_capture_ns
            remaining = (
                None
                if inputs.max_source_points is None
                else max(0, inputs.max_source_points - selected_source_points)
            )
            offsets_s, point_time_reference_ns, point_time_reference = (
                _rosbag2_point_time_data(
                    message,
                    source_time_field,
                    ros_timestamp_ns=timestamp_ns,
                )
            )
            if offsets_s is not None:
                point_time_observed_by_sensor[source_sensor] = True
                (
                    point_time_min_by_sensor[source_sensor],
                    point_time_max_by_sensor[source_sensor],
                ) = _update_point_time_range(
                    offsets_s,
                    point_time_min=point_time_min_by_sensor[source_sensor],
                    point_time_max=point_time_max_by_sensor[source_sensor],
                )
                if point_time_reference is not None:
                    point_time_reference_by_sensor[source_sensor] = point_time_reference
            _record_rosbag2_point_time_mapping(
                point_time_clock_mapping_by_sensor,
                source_sensor,
                message,
                ros_timestamp_ns=timestamp_ns,
            )
            scan_records: list[LivoxPointRecord]
            if source_deskew and odometry_track is not None:
                if offsets_s is None or point_time_reference_ns is None:
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
                scan_records = _deskew_pointcloud_to_world_records(
                    message.xyz,
                    offsets_s,
                    point_time_reference_ns,
                    t_base_source,
                    odometry_track,
                    remaining,
                    extrapolation_tolerance_s,
                )
            elif motion_compensated and odometry_track is not None:
                t_world_base, _clamped, extrapolation_s = odometry_track.interpolate(
                    _rosbag2_message_timestamp_ns(message, timestamp_ns)
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
                scan_records = _transform_pointcloud_to_world_records(
                    message.xyz,
                    t_world_source,
                    remaining,
                )
            else:
                scan_records = _pointcloud_xyz_to_records(message.xyz, remaining)
            source_records.extend(scan_records)
            source_record_windows.append(
                (
                    _rosbag2_message_timestamp_ns(message, timestamp_ns),
                    scan_records,
                )
            )
            source_messages += 1
            if source_window_index is not None:
                capture_window_stats[source_window_index][
                    "selected_source_message_count"
                ] += 1
                capture_window_stats[source_window_index][
                    "selected_source_point_count"
                ] += len(scan_records)
            continue

        if topic != target_topic:
            continue

        if not _record_timestamp_may_be_in_capture_windows(
            timestamp_ns,
            inputs.capture_windows,
            margin_s=inputs.capture_window_record_prefilter_margin_s,
        ):
            continue
        if evenly_spaced_sampling and timestamp_ns not in selected_target_record_timestamps:
            continue

        message = _decode_rosbag2_online_lidar(
            topic,
            connection.message_type,
            timestamp_ns,
            data,
            point_time_field=target_time_field,
        )
        target_capture_ns = _rosbag2_message_timestamp_ns(message, timestamp_ns)
        target_window_index = _capture_window_index(
            target_capture_ns, inputs.capture_windows
        )
        if inputs.capture_windows and target_window_index is None:
            continue
        selected_target_messages = (
            target_messages
            if target_window_index is None
            else int(
                capture_window_stats[target_window_index][
                    "selected_target_message_count"
                ]
            )
        )
        if (
            inputs.max_target_messages is not None
            and selected_target_messages >= inputs.max_target_messages
        ):
            if not inputs.capture_windows:
                break
            continue
        if replay_start_ns is None:
            replay_start_ns = target_capture_ns
            first_target_ns = target_capture_ns
        replay_end_ns = target_capture_ns
        if inputs.max_replay_duration_s is not None:
            elapsed_ns = target_capture_ns - replay_start_ns
            if elapsed_ns > int(inputs.max_replay_duration_s * 1_000_000_000):
                break
        target_raw_point_count += int(
            getattr(message, "raw_point_count", message.point_count)
        )
        target_nonfinite_xyz_count += int(
            getattr(message, "nonfinite_xyz_count", 0)
        )
        target_message_type = connection.message_type
        target_points_before = len(target_stream)
        offsets_s, point_time_reference_ns, point_time_reference = (
            _rosbag2_point_time_data(
                message,
                target_time_field,
                ros_timestamp_ns=timestamp_ns,
            )
        )
        if offsets_s is not None:
            point_time_observed_by_sensor[target_sensor] = True
            (
                point_time_min_by_sensor[target_sensor],
                point_time_max_by_sensor[target_sensor],
            ) = _update_point_time_range(
                offsets_s,
                point_time_min=point_time_min_by_sensor[target_sensor],
                point_time_max=point_time_max_by_sensor[target_sensor],
            )
            if point_time_reference is not None:
                point_time_reference_by_sensor[target_sensor] = point_time_reference
        _record_rosbag2_point_time_mapping(
            point_time_clock_mapping_by_sensor,
            target_sensor,
            message,
            ros_timestamp_ns=timestamp_ns,
        )
        if target_deskew and odometry_track is not None:
            if offsets_s is None or point_time_reference_ns is None:
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
                    point_time_reference_ns,
                    t_base_source,
                    odometry_track,
                    inputs.max_target_points,
                    inject_time_offset_s=inputs.inject_time_offset_s,
                )
            )
        elif motion_compensated and odometry_track is not None:
            points = _pointcloud_xyz_to_vectors(message.xyz, inputs.max_target_points)
            capture_ns = _target_capture_ns(
                _rosbag2_message_timestamp_ns(message, timestamp_ns),
                inject_time_offset_s=inputs.inject_time_offset_s,
            )
            target_extrapolation_s = 0.0
            t_world_base, _clamped, target_extrapolation_s = odometry_track.interpolate(
                capture_ns
            )
            t_world_source = t_world_base.compose(t_base_source)
            target_stream.extend(
                (frame_index, point, t_world_source, target_extrapolation_s, capture_ns)
                for point in points
            )
        else:
            points = _pointcloud_xyz_to_vectors(message.xyz, inputs.max_target_points)
            target_stream.extend(
                (
                    frame_index,
                    point,
                    SE3.identity(),
                    0.0,
                    _rosbag2_message_timestamp_ns(message, timestamp_ns),
                )
                for point in points
            )
        frame_index += 1
        target_messages += 1
        if target_window_index is not None:
            capture_window_stats[target_window_index][
                "selected_target_message_count"
            ] += 1
            capture_window_stats[target_window_index][
                "selected_target_point_count"
            ] += len(target_stream) - target_points_before

    target_frame_ids = sorted(
        {frame_index for frame_index, _point, _pose, _extrap, _capture in target_stream}
    )
    target_train_window_count = 0
    target_holdout_window_count = 0
    temporal_holdout_independent = False
    source_map_temporal_boundary_ns: int | None = None
    source_map_temporal_boundary_applied = False
    if len(target_frame_ids) >= 2:
        holdout_start_index = max(1, int(len(target_frame_ids) * 0.8))
        holdout_frame_ids = set(target_frame_ids[holdout_start_index:])
        holdout_entries = [
            entry for entry in target_stream if entry[0] in holdout_frame_ids
        ]
        source_map_temporal_boundary_ns = min(
            (entry[4] for entry in holdout_entries), default=None
        )
        target_train_window_count = holdout_start_index
        target_holdout_window_count = len(holdout_frame_ids)
        if source_map_temporal_boundary_ns is not None:
            eligible_source_windows = [
                window
                for window in source_record_windows
                if window[0] < source_map_temporal_boundary_ns
            ]
            if eligible_source_windows:
                source_records = [
                    record
                    for _timestamp, records in eligible_source_windows
                    for record in records
                ]
                source_record_windows = eligible_source_windows
                source_messages = len(eligible_source_windows)
                source_map_temporal_boundary_applied = True
                temporal_holdout_independent = True

    _refresh_capture_window_source_stats(
        capture_window_stats,
        inputs.capture_windows,
        source_record_windows,
    )

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
        "rosbag2_source_raw_point_count": source_raw_point_count,
        "rosbag2_target_raw_point_count": target_raw_point_count,
        "rosbag2_source_nonfinite_xyz_count": source_nonfinite_xyz_count,
        "rosbag2_target_nonfinite_xyz_count": target_nonfinite_xyz_count,
        "rosbag2_replay_duration_s": replay_duration_s,
        "rosbag2_first_source_timestamp_ns": first_source_ns,
        "rosbag2_first_target_timestamp_ns": first_target_ns,
        "rosbag2_source_message_type": source_message_type,
        "rosbag2_target_message_type": target_message_type,
        "rosbag2_replay_start_timestamp_ns": replay_start_ns,
        "rosbag2_replay_end_timestamp_ns": replay_end_ns,
        "rosbag2_max_source_messages": inputs.max_source_messages,
        "rosbag2_max_target_messages": inputs.max_target_messages,
        "rosbag2_max_replay_duration_s": inputs.max_replay_duration_s,
        "rosbag2_capture_window_start_timestamp_ns": (
            inputs.capture_window_start_timestamp_ns
        ),
        "rosbag2_capture_window_end_timestamp_ns": inputs.capture_window_end_timestamp_ns,
        "rosbag2_capture_window_count": len(inputs.capture_windows),
        "rosbag2_capture_windows": capture_window_stats,
        "rosbag2_capture_window_record_prefilter_margin_s": (
            inputs.capture_window_record_prefilter_margin_s
        ),
        "rosbag2_capture_window_selection_policy": (
            "union of inclusive LiDAR capture timestamp windows; Livox uses "
            "timebase/header-mapped capture time and PointCloud2 uses the decoded "
            "message stamp; overlapping windows use first-match provenance accounting; "
            "source/target message and source-point limits are applied per window"
        ),
        "rosbag2_capture_window_sampling_policy": inputs.capture_window_sampling_policy,
        "rosbag2_source_map_temporal_boundary_timestamp_ns": (
            source_map_temporal_boundary_ns
        ),
        "rosbag2_source_map_temporal_boundary_applied": (
            source_map_temporal_boundary_applied
        ),
        "rosbag2_source_map_window_count": len(source_record_windows),
        "rosbag2_target_train_window_count": target_train_window_count,
        "rosbag2_target_holdout_window_count": target_holdout_window_count,
        "rosbag2_temporal_holdout_independent": temporal_holdout_independent,
        "rosbag2_temporal_holdout_clock_note": (
            "boundary compares ROS2 bag/header/Livox-mapped capture nanoseconds; "
            "validate clock-domain alignment before interpreting independence"
        ),
        "motion_compensated": motion_compensated,
        "point_time_deskew_enabled": point_time_deskew_enabled,
        "point_time_observed_by_sensor": point_time_observed_by_sensor,
        "point_time_reference_by_sensor": point_time_reference_by_sensor,
        "point_time_clock_mapping_by_sensor": _finalize_point_time_mappings(
            point_time_clock_mapping_by_sensor
        ),
        "point_time_range_by_sensor": {
            sensor_name: {
                "min_s": point_time_min_by_sensor[sensor_name],
                "max_s": point_time_max_by_sensor[sensor_name],
            }
            for sensor_name in point_time_observed_by_sensor
            if point_time_min_by_sensor[sensor_name] is not None
            and point_time_max_by_sensor[sensor_name] is not None
        },
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
    if abs(inputs.inject_time_offset_s) > 1.0e-12:
        provenance["time_offset_injected_s"] = inputs.inject_time_offset_s
        provenance["time_offset_injection_warning"] = (
            "VALIDATION-ONLY: target capture timestamps were shifted at load time"
        )
    return (
        source_records,
        target_stream,
        provenance,
        motion_compensated,
        odometry_track,
        odometry_source_info,
    )


def _load_rosbag2_odometry_track(
    bag_path: Path,
    odometry_topic: str | None,
    *,
    preprocessing: OdometryPreprocessingConfig | None = None,
) -> tuple[OdometryTrack | None, OdometrySourceInfo | None]:
    if not odometry_topic:
        return None, None
    samples: list[OdometryPoseSample] = []
    world_frame_id = ""
    child_frame_id = ""
    for connection, timestamp_ns, data in iter_rosbag2_messages(
        bag_path,
        topics={odometry_topic},
    ):
        if connection.message_type != ODOMETRY_TYPE:
            continue
        message = decode_odometry(connection.topic, timestamp_ns, data)
        if not world_frame_id:
            world_frame_id = message.frame_id
            child_frame_id = message.child_frame_id
        samples.append(
            OdometryPoseSample(
                timestamp_ns=message.timestamp_ns,
                pose=SE3.from_lists(list(message.position), list(message.orientation_xyzw)),
            )
        )
    if not samples:
        return None, None
    return (
        OdometryTrack(
            samples,
            burst_policy=(preprocessing.burst_policy if preprocessing else "preserve"),
            min_interval_s=(preprocessing.min_interval_s if preprocessing else 0.001),
        ),
        OdometrySourceInfo(
            topic=odometry_topic,
            world_frame_id=world_frame_id,
            child_frame_id=child_frame_id,
        ),
    )


def collect_rosbag2_cross_segment_scans(
    *,
    bag_path: Path,
    source_topic: str,
    source_point_time_field: str | None = None,
    use_point_time_offsets: bool = True,
    odometry_track: OdometryTrack,
    t_base_source: SE3,
    options: TrajectoryEvidenceOptions,
    extrapolation_tolerance_s: float | None,
    capture_window_start_timestamp_ns: int | None = None,
    capture_window_end_timestamp_ns: int | None = None,
    capture_window_record_prefilter_margin_s: float = 0.0,
    capture_windows: tuple[tuple[int | None, int | None], ...] | None = None,
) -> CrossSegmentScanCollection | None:
    """Stream source scans over the declared odometry/capture span for drift evaluation."""

    odometry_first_ns = odometry_track.first_timestamp_ns
    odometry_last_ns = odometry_track.last_timestamp_ns
    if (
        odometry_first_ns is None
        or odometry_last_ns is None
        or odometry_last_ns <= odometry_first_ns
    ):
        return None
    window_bounds = tuple(capture_windows or ())
    if not window_bounds and (
        capture_window_start_timestamp_ns is not None
        or capture_window_end_timestamp_ns is not None
    ):
        window_bounds = (
            (capture_window_start_timestamp_ns, capture_window_end_timestamp_ns),
        )
    if window_bounds:
        clipped_spans = [
            (
                max(
                    odometry_first_ns,
                    start_ns if start_ns is not None else odometry_first_ns,
                ),
                min(
                    odometry_last_ns,
                    end_ns if end_ns is not None else odometry_last_ns,
                ),
            )
            for start_ns, end_ns in window_bounds
        ]
        valid_spans = [
            (start_ns, end_ns) for start_ns, end_ns in clipped_spans if end_ns > start_ns
        ]
        if not valid_spans:
            return None
        first_ns = min(start_ns for start_ns, _end_ns in valid_spans)
        last_ns = max(end_ns for _start_ns, end_ns in valid_spans)
    else:
        first_ns = odometry_first_ns
        last_ns = odometry_last_ns
    if last_ns <= first_ns:
        return None
    capture_window_requested = bool(window_bounds)

    midpoint_ns = (first_ns + last_ns) // 2
    first_window_start_ns = first_ns
    first_window_end_ns = midpoint_ns
    second_window_start_ns = midpoint_ns + 1
    second_window_end_ns = last_ns

    source_candidates: list[tuple[int, int]] = []
    for connection, timestamp_ns, _data in iter_rosbag2_messages(
        bag_path,
        topics={source_topic},
    ):
        if (
            connection.message_type not in ROS2_LIDAR_MESSAGE_TYPES
            or connection.topic != source_topic
        ):
            continue
        if capture_window_requested:
            if not _record_timestamp_may_be_in_capture_windows(
                timestamp_ns,
                window_bounds,
                margin_s=capture_window_record_prefilter_margin_s,
            ):
                continue
            message = decode_rosbag2_lidar_message(
                connection.topic,
                connection.message_type,
                timestamp_ns,
                _data,
                point_time_field=source_point_time_field,
            )
            capture_ns = _rosbag2_message_timestamp_ns(message, timestamp_ns)
            if not _capture_timestamp_in_windows(capture_ns, window_bounds):
                continue
            if capture_ns < first_ns or capture_ns > last_ns:
                continue
            source_candidates.append((timestamp_ns, capture_ns))
        elif first_ns <= timestamp_ns <= last_ns:
            source_candidates.append((timestamp_ns, timestamp_ns))

    source_capture_timestamps_ns = sorted({capture for _record, capture in source_candidates})
    if len(source_capture_timestamps_ns) < 2:
        return None

    first_selected_capture_ns = set(
        _select_evenly_spaced_timestamps(
            source_capture_timestamps_ns,
            window_start_ns=first_window_start_ns,
            window_end_ns=first_window_end_ns,
            max_samples=options.max_scans_per_half,
        )
    )
    second_selected_capture_ns = set(
        _select_evenly_spaced_timestamps(
            source_capture_timestamps_ns,
            window_start_ns=second_window_start_ns,
            window_end_ns=second_window_end_ns,
            max_samples=options.max_scans_per_half,
        )
    )
    first_selected_record_ns = {
        record_ns
        for record_ns, capture_ns in source_candidates
        if capture_ns in first_selected_capture_ns
    }
    second_selected_record_ns = {
        record_ns
        for record_ns, capture_ns in source_candidates
        if capture_ns in second_selected_capture_ns
    }
    selected_record_ns = first_selected_record_ns | second_selected_record_ns
    if not first_selected_record_ns or not second_selected_record_ns:
        return None

    pose_track = OdometryTrack(list(odometry_track.samples))
    first_half_scans: list[OnlineSourceScan] = []
    second_half_scans: list[OnlineSourceScan] = []
    points_per_first_scan = max(
        1, options.max_points_per_half // max(len(first_selected_record_ns), 1)
    )
    points_per_second_scan = max(
        1, options.max_points_per_half // max(len(second_selected_record_ns), 1)
    )

    for connection, timestamp_ns, data in iter_rosbag2_messages(
        bag_path,
        topics={source_topic},
    ):
        if (
            connection.message_type not in ROS2_LIDAR_MESSAGE_TYPES
            or connection.topic != source_topic
        ):
            continue
        if timestamp_ns not in selected_record_ns:
            continue
        message = decode_rosbag2_lidar_message(
            connection.topic,
            connection.message_type,
            timestamp_ns,
            data,
            point_time_field=source_point_time_field,
        )
        if timestamp_ns in first_selected_record_ns:
            scan_records = _rosbag2_cross_segment_scan_records(
                message,
                timestamp_ns=timestamp_ns,
                point_time_field=source_point_time_field,
                use_point_time_offsets=use_point_time_offsets,
                t_base_source=t_base_source,
                pose_track=pose_track,
                max_points=points_per_first_scan,
                extrapolation_tolerance_s=extrapolation_tolerance_s,
            )
            if scan_records:
                first_half_scans.append(
                    OnlineSourceScan(
                        timestamp_ns=_rosbag2_message_timestamp_ns(message, timestamp_ns),
                        records=scan_records,
                    )
                )
        if timestamp_ns in second_selected_record_ns:
            scan_records = _rosbag2_cross_segment_scan_records(
                message,
                timestamp_ns=timestamp_ns,
                point_time_field=source_point_time_field,
                use_point_time_offsets=use_point_time_offsets,
                t_base_source=t_base_source,
                pose_track=pose_track,
                max_points=points_per_second_scan,
                extrapolation_tolerance_s=extrapolation_tolerance_s,
            )
            if scan_records:
                second_half_scans.append(
                    OnlineSourceScan(
                        timestamp_ns=_rosbag2_message_timestamp_ns(message, timestamp_ns),
                        records=scan_records,
                    )
                )

    if not first_half_scans or not second_half_scans:
        return None

    return CrossSegmentScanCollection(
        first_half_scans=sorted(first_half_scans, key=lambda scan: scan.timestamp_ns),
        second_half_scans=sorted(second_half_scans, key=lambda scan: scan.timestamp_ns),
        first_half_start_timestamp_ns=first_window_start_ns,
        first_half_end_timestamp_ns=first_window_end_ns,
        second_half_start_timestamp_ns=second_window_start_ns,
        second_half_end_timestamp_ns=second_window_end_ns,
    )


def _odometry_replay_provenance(
    odometry_topic: str,
    odometry_track: OdometryTrack | None,
    motion_compensated: bool,
) -> dict[str, Any]:
    return {
        "odometry_topic": odometry_topic,
        "odometry_message_count": odometry_track.message_count if odometry_track else 0,
        "odometry_raw_message_count": (
            odometry_track.raw_message_count if odometry_track else 0
        ),
        "odometry_burst_policy": (
            odometry_track.burst_policy if odometry_track else "preserve"
        ),
        "odometry_burst_min_interval_s": (
            odometry_track.burst_min_interval_s if odometry_track else 0.001
        ),
        "odometry_burst_removed_count": (
            odometry_track.burst_removed_count if odometry_track else 0
        ),
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


def evaluate_rosbag2_trajectory_window_drift(
    config_path: str | Path,
    *,
    reference_result_path: str | Path | None = None,
    use_point_time_offsets: bool | None = None,
    odometry_burst_policy: OdometryBurstPolicy | None = None,
    odometry_burst_min_interval_s: float | None = None,
) -> TrajectoryWindowDriftArtifact:
    """Recompute trajectory/map consistency for every configured capture window.

    This diagnostic deliberately does not use the estimated extrinsic. Source
    scans are motion-compensated into the odometry world frame, and each
    half-window is compared with an identity transform. It therefore provides
    evidence about map/odometry consistency that is independent of the online
    solver's final transform, while remaining a ground-truth-free hypothesis.
    """

    config_file = Path(config_path)
    config = load_config(config_file)
    override_notes: list[str] = []
    if use_point_time_offsets is not None:
        factor = config.pipeline.factors.get(_FACTOR_NAME)
        if factor is None:
            raise ConfigError(
                f"trajectory-window-drift requires the {_FACTOR_NAME!r} factor "
                "when --deskew overrides the config"
            )
        override_factor_options = dict(factor.options)
        override_factor_options["use_point_time_offsets"] = use_point_time_offsets
        factors = dict(config.pipeline.factors)
        factors[_FACTOR_NAME] = factor.model_copy(
            update={"options": override_factor_options}
        )
        pipeline = config.pipeline.model_copy(update={"factors": factors})
        config = config.model_copy(update={"pipeline": pipeline})
        override_notes.append(
            "override use_point_time_offsets=" + str(use_point_time_offsets).lower()
        )
    if odometry_burst_policy is not None or odometry_burst_min_interval_s is not None:
        configured = config.dataset.odometry_preprocessing
        burst_policy = odometry_burst_policy or (
            configured.burst_policy if configured is not None else "preserve"
        )
        min_interval_s = (
            odometry_burst_min_interval_s
            if odometry_burst_min_interval_s is not None
            else (configured.min_interval_s if configured is not None else 0.001)
        )
        preprocessing = OdometryPreprocessingConfig(
            burst_policy=burst_policy,
            min_interval_s=min_interval_s,
        )
        dataset = config.dataset.model_copy(
            update={"odometry_preprocessing": preprocessing}
        )
        config = config.model_copy(update={"dataset": dataset})
        override_notes.append(
            "override odometry_preprocessing="
            f"{burst_policy}/{min_interval_s:g}s"
        )
    if config.dataset.type != "rosbag2":
        raise ConfigError(
            "trajectory-window-drift currently requires dataset.type: rosbag2"
        )
    if not config.dataset.odometry_topic:
        raise ConfigError(
            "trajectory-window-drift requires dataset.odometry_topic for a pose track"
        )
    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        raise DatasetError(f"ROS bag dataset path does not exist: {bag_path}")

    frame_graph = FrameGraph.from_config(config)
    lidars = sorted(name for name, sensor in config.sensors.items() if sensor.type == "lidar")
    if len(lidars) < 2:
        raise ConfigError("trajectory-window-drift requires at least two LiDAR sensors")
    source_sensor, target_sensor = _resolve_online_lidar_pair(config, frame_graph, lidars)
    source_node = frame_graph.nodes.get(source_sensor)
    target_node = frame_graph.nodes.get(target_sensor)
    if source_node is None or target_node is None or target_node.parent is None:
        raise ConfigError(
            "trajectory-window-drift requires source and target LiDAR frames in the graph"
        )

    odometry_track, odometry_source_info = _load_rosbag2_odometry_track(
        bag_path,
        config.dataset.odometry_topic,
        preprocessing=config.dataset.odometry_preprocessing,
    )
    if odometry_track is None or odometry_source_info is None:
        raise DatasetError(
            f"no odometry messages found on {config.dataset.odometry_topic!r}"
        )
    first_odometry_ns = odometry_track.first_timestamp_ns
    last_odometry_ns = odometry_track.last_timestamp_ns
    if first_odometry_ns is None or last_odometry_ns is None:
        raise DatasetError("odometry track has no timestamp range")

    inputs = _solve_inputs(config)
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    factor_options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    trajectory_options = trajectory_evidence_options_from_factor(factor_options)
    extrapolation_tolerance_s = _odometry_extrapolation_tolerance_s(config, True)
    deskew_time_field = _deskew_time_field_map(config, source_sensor, target_sensor)
    source_point_time_field = deskew_time_field.get(source_sensor)
    declared_windows = inputs.capture_windows
    windows = declared_windows or ((first_odometry_ns, last_odometry_ns),)
    variable = f"T_{target_node.parent}_{target_sensor}"

    full_span_scans = collect_rosbag2_cross_segment_scans(
        bag_path=bag_path,
        source_topic=_lidar_topic(config, source_sensor),
        source_point_time_field=source_point_time_field,
        use_point_time_offsets=_point_time_deskew_enabled(config),
        odometry_track=odometry_track,
        t_base_source=source_node.transform_to_parent,
        options=trajectory_options,
        extrapolation_tolerance_s=extrapolation_tolerance_s,
        capture_window_record_prefilter_margin_s=inputs.capture_window_record_prefilter_margin_s,
        capture_windows=windows,
    )
    full_span_evidence = evaluate_trajectory_evidence(
        odometry_track=odometry_track,
        cross_segment_scans=full_span_scans,
        voxel_size_m=inputs.voxel_size_m,
        correspondence_gate_m=inputs.correspondence_gate_m,
        variable=variable,
        sensor=target_sensor,
        options=trajectory_options,
        max_odometry_extrapolation_s=extrapolation_tolerance_s,
    )

    window_artifacts: list[TrajectoryWindowDriftWindow] = []
    for index, window in enumerate(windows):
        window_scans = collect_rosbag2_cross_segment_scans(
            bag_path=bag_path,
            source_topic=_lidar_topic(config, source_sensor),
            source_point_time_field=source_point_time_field,
            use_point_time_offsets=_point_time_deskew_enabled(config),
            odometry_track=odometry_track,
            t_base_source=source_node.transform_to_parent,
            options=trajectory_options,
            extrapolation_tolerance_s=extrapolation_tolerance_s,
            capture_window_record_prefilter_margin_s=(
                inputs.capture_window_record_prefilter_margin_s
            ),
            capture_windows=(window,),
        )
        evidence = evaluate_trajectory_evidence(
            odometry_track=odometry_track,
            cross_segment_scans=window_scans,
            voxel_size_m=inputs.voxel_size_m,
            correspondence_gate_m=inputs.correspondence_gate_m,
            variable=variable,
            sensor=target_sensor,
            options=trajectory_options,
            max_odometry_extrapolation_s=extrapolation_tolerance_s,
        )
        window_artifacts.append(
            TrajectoryWindowDriftWindow(
                window_index=index,
                start_timestamp_ns=window[0],
                end_timestamp_ns=window[1],
                odometry=_trajectory_window_odometry_stats(odometry_track, window),
                cross_segment=_trajectory_window_cross_segment_evidence(
                    evidence.cross_segment
                ),
            )
        )

    interpretation, grade, interpretation_reason = interpret_trajectory_window_drift(
        [window.cross_segment.gate_status for window in window_artifacts],
        full_span_evidence.cross_segment.gate_status,
    )
    source_sha256: dict[str, str] = {}
    source_paths = [str(config_file), str(bag_path)]
    for path in (config_file, bag_path):
        digest = sha256_path(path)
        if digest is None:
            raise DatasetError(f"could not hash diagnostic input: {path}")
        source_sha256[str(path)] = digest
    notes = [
        "cross-segment evidence is computed in the odometry world frame with identity extrinsic",
        "interpretation is ground-truth-free and cannot uniquely attribute error to "
        "odometry, map geometry, or sensor timing",
    ]
    if not declared_windows:
        notes.append(
            "capture_windows was not declared; one diagnostic window covering the "
            "odometry span was used"
        )
    notes.extend(override_notes)
    if reference_result_path is not None:
        result_path = Path(reference_result_path)
        if not result_path.exists():
            raise DatasetError(f"reference result path does not exist: {result_path}")
        result_digest = sha256_path(result_path)
        if result_digest is None:
            raise DatasetError(f"could not hash reference result: {result_path}")
        source_paths.append(str(result_path))
        source_sha256[str(result_path)] = result_digest
        notes.append("reference result is provenance-only; all trajectory evidence was recomputed")

    return TrajectoryWindowDriftArtifact(
        config_path=str(config_file),
        dataset_path=str(bag_path),
        source_sensor=source_sensor,
        target_sensor=target_sensor,
        odometry_topic=odometry_source_info.topic,
        parameters=TrajectoryWindowDriftParameters(
            voxel_size_m=inputs.voxel_size_m,
            correspondence_gate_m=inputs.correspondence_gate_m,
            max_scans_per_half=trajectory_options.max_scans_per_half,
            max_points_per_half=trajectory_options.max_points_per_half,
            source_point_time_field=source_point_time_field,
            use_point_time_offsets=_point_time_deskew_enabled(config),
            odometry_burst_policy=odometry_track.burst_policy,
            odometry_burst_min_interval_s=odometry_track.burst_min_interval_s,
            capture_window_record_prefilter_margin_s=(
                inputs.capture_window_record_prefilter_margin_s
            ),
            capture_window_sampling_policy=inputs.capture_window_sampling_policy,
        ),
        thresholds=TrajectoryWindowDriftThresholds(
            max_cross_segment_rmse_m=trajectory_options.max_cross_segment_rmse_m,
            min_correspondences=trajectory_options.min_correspondences,
        ),
        windows=window_artifacts,
        full_span_cross_segment=_trajectory_window_cross_segment_evidence(
            full_span_evidence.cross_segment
        ),
        interpretation=interpretation,
        grade=grade,
        interpretation_reason=interpretation_reason,
        provenance=TrajectoryWindowDriftProvenance(
            source_paths=source_paths,
            source_sha256=source_sha256,
            tool_version=__version__,
            git_commit=git_commit(),
            notes=notes,
        ),
    )


def evaluate_rosbag2_capture_readiness(
    config_path: str | Path,
) -> CaptureReadinessArtifact:
    """Evaluate motion and geometry readiness before running calibration.

    The geometry DoF is intentionally a conservative proxy: selected source
    scans are deskewed into the odometry world frame, converted to a voxel
    plane map, and re-expressed through the configured initial target pose.
    It is not a calibration result and does not replace the post-solve holdout
    gates. Its purpose is to turn an under-excited recording into an actionable
    recapture instruction before a solver run.
    """

    config_file = Path(config_path)
    config = load_config(config_file)
    if config.dataset.type != "rosbag2":
        raise ConfigError("calibration-readiness currently requires dataset.type: rosbag2")
    if not config.dataset.odometry_topic:
        raise ConfigError(
            "calibration-readiness requires dataset.odometry_topic for motion excitation"
        )
    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        raise DatasetError(f"rosbag2 dataset path does not exist: {bag_path}")

    frame_graph = FrameGraph.from_config(config)
    lidars = sorted(name for name, sensor in config.sensors.items() if sensor.type == "lidar")
    if len(lidars) < 2:
        raise ConfigError("calibration-readiness requires at least two LiDAR sensors")
    source_sensor, target_sensor = _resolve_online_lidar_pair(config, frame_graph, lidars)
    source_node = frame_graph.nodes.get(source_sensor)
    target_node = frame_graph.nodes.get(target_sensor)
    if source_node is None or target_node is None or target_node.parent is None:
        raise ConfigError(
            "calibration-readiness requires source and target LiDAR frames in the graph"
        )

    odometry_track, odometry_source_info = _load_rosbag2_odometry_track(
        bag_path,
        config.dataset.odometry_topic,
        preprocessing=config.dataset.odometry_preprocessing,
    )
    if odometry_track is None or odometry_source_info is None:
        raise DatasetError(f"no odometry messages found on {config.dataset.odometry_topic!r}")
    first_odometry_ns = odometry_track.first_timestamp_ns
    last_odometry_ns = odometry_track.last_timestamp_ns
    if first_odometry_ns is None or last_odometry_ns is None:
        raise DatasetError("odometry track has no timestamp range")

    inputs = _solve_inputs(config)
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    factor_options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    trajectory_options = trajectory_evidence_options_from_factor(factor_options)
    readiness_options = _capture_readiness_options(factor_options)
    geometry_options = replace(
        trajectory_options,
        max_scans_per_half=max(
            1, (readiness_options["max_scans_per_window"] + 1) // 2
        ),
        max_points_per_half=max(1, readiness_options["max_points_per_window"] // 2),
    )
    extrapolation_tolerance_s = _odometry_extrapolation_tolerance_s(config, True)
    source_point_time_field = _deskew_time_field_map(config, source_sensor, target_sensor).get(
        source_sensor
    )
    declared_windows = inputs.capture_windows
    windows = declared_windows or ((first_odometry_ns, last_odometry_ns),)
    t_source_target_init = source_node.transform_to_parent.inverse().compose(
        target_node.transform_to_parent
    )
    variable = f"T_{target_node.parent}_{target_sensor}"
    thresholds = _capture_readiness_thresholds(factor_options)
    window_artifacts: list[CaptureReadinessWindow] = []
    for index, window in enumerate(windows):
        source_scans = collect_rosbag2_cross_segment_scans(
            bag_path=bag_path,
            source_topic=_lidar_topic(config, source_sensor),
            source_point_time_field=source_point_time_field,
            use_point_time_offsets=_point_time_deskew_enabled(config),
            odometry_track=odometry_track,
            t_base_source=source_node.transform_to_parent,
            options=geometry_options,
            extrapolation_tolerance_s=extrapolation_tolerance_s,
            capture_window_record_prefilter_margin_s=(
                inputs.capture_window_record_prefilter_margin_s
            ),
            capture_windows=(window,),
        )
        geometry = _capture_readiness_geometry(
            source_scans,
            initial_t_source_target=t_source_target_init,
            voxel_size_m=inputs.voxel_size_m,
            correspondence_gate_m=inputs.correspondence_gate_m,
            max_points=readiness_options["max_points_per_window"],
            variable=variable,
            sensor=target_sensor,
        )
        motion = summarize_capture_window_motion(list(odometry_track.samples), window)
        window_artifacts.append(
            classify_capture_readiness_window(
                window_index=index,
                requested_window=window,
                motion=motion,
                geometry=geometry,
                thresholds=thresholds,
            )
        )

    grade, decision, summary, recommendations = build_capture_readiness_recommendations(
        window_artifacts
    )
    source_sha256: dict[str, str] = {}
    source_paths = [str(config_file), str(bag_path)]
    for path in (config_file, bag_path):
        digest = sha256_path(path)
        if digest is None:
            raise DatasetError(f"could not hash readiness input: {path}")
        source_sha256[str(path)] = digest
    notes = [
        "readiness is ground-truth-free preflight evidence and is not a calibration result",
        "geometry DoF uses a source-world voxel-plane map and a virtual target re-expression",
        "run the solver only after readiness decision is proceed or warnings are reviewed",
    ]
    if not declared_windows:
        notes.append(
            "capture_windows was not declared; one readiness window covers the odometry span"
        )
    return CaptureReadinessArtifact(
        config_path=str(config_file),
        dataset_path=str(bag_path),
        source_sensor=source_sensor,
        target_sensor=target_sensor,
        odometry_topic=odometry_source_info.topic,
        parameters=CaptureReadinessParameters(
            voxel_size_m=inputs.voxel_size_m,
            correspondence_gate_m=inputs.correspondence_gate_m,
            max_scans_per_window=readiness_options["max_scans_per_window"],
            max_points_per_window=readiness_options["max_points_per_window"],
            source_point_time_field=source_point_time_field,
            use_point_time_offsets=_point_time_deskew_enabled(config),
            odometry_burst_policy=odometry_track.burst_policy,
            odometry_burst_min_interval_s=odometry_track.burst_min_interval_s,
            capture_window_record_prefilter_margin_s=(
                inputs.capture_window_record_prefilter_margin_s
            ),
            capture_window_sampling_policy=inputs.capture_window_sampling_policy,
        ),
        thresholds=thresholds,
        windows=window_artifacts,
        grade=grade,
        decision=decision,
        summary=summary,
        recommendations=recommendations,
        provenance=CaptureReadinessProvenance(
            source_paths=source_paths,
            source_sha256=source_sha256,
            tool_version=__version__,
            git_commit=git_commit(),
            notes=notes,
        ),
    )


def evaluate_continuous_time_lidar_pair(
    config_path: str | Path,
) -> ContinuousTimeLidarPairArtifact:
    """Jointly profile LiDAR extrinsic and scalar clock offset on a ROS bag pair.

    The source map and target stream use the existing online adapter and its
    temporal holdout boundary. The supplied odometry is interpolated at each
    target capture timestamp plus the profiled offset; no pose is extrapolated
    beyond the configured tolerance. This is a continuous-time refinement
    baseline, not yet a free trajectory estimator. Both ROS 1 bags and ROS 2
    recordings use the same solver and evaluation protocol; only the message
    adapter differs.
    """

    config_file = Path(config_path)
    config = load_config(config_file)
    if config.dataset.type not in ("rosbag1", "rosbag2"):
        raise ConfigError(
            "continuous-time-lidar requires dataset.type: rosbag1 or rosbag2"
        )
    if not config.dataset.odometry_topic:
        raise ConfigError(
            "continuous-time-lidar requires dataset.odometry_topic for timestamp profiling"
        )
    bag_path = Path(config.dataset.path)
    if not bag_path.exists():
        raise DatasetError(f"rosbag2 dataset path does not exist: {bag_path}")

    frame_graph = FrameGraph.from_config(config)
    lidars = sorted(name for name, sensor in config.sensors.items() if sensor.type == "lidar")
    if len(lidars) < 2:
        raise ConfigError("continuous-time-lidar requires at least two LiDAR sensors")
    source_sensor, target_sensor = _resolve_online_lidar_pair(config, frame_graph, lidars)
    source_node = frame_graph.nodes.get(source_sensor)
    target_node = frame_graph.nodes.get(target_sensor)
    if source_node is None or target_node is None or target_node.parent is None:
        raise ConfigError(
            "continuous-time-lidar requires source and target LiDAR frames in the graph"
        )

    inputs = _solve_inputs(config)
    factor = config.pipeline.factors.get(_FACTOR_NAME)
    factor_options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    continuous_options = _continuous_time_lidar_options(config, factor_options)
    if config.dataset.type == "rosbag1":
        (
            source_records,
            target_stream,
            replay_provenance,
            motion_compensated,
            odometry_track,
            odometry_source_info,
        ) = _load_rosbag1_online_pair(
            config,
            inputs,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
            t_base_source=source_node.transform_to_parent,
        )
    else:
        (
            source_records,
            target_stream,
            replay_provenance,
            motion_compensated,
            odometry_track,
            odometry_source_info,
        ) = _load_rosbag2_online_pair(
            config,
            inputs,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
            t_base_source=source_node.transform_to_parent,
        )
    if not source_records:
        raise DatasetError("continuous-time-lidar found no source points for the voxel map")
    if not target_stream:
        raise DatasetError("continuous-time-lidar found no target points to replay")
    if not motion_compensated or odometry_track is None or odometry_source_info is None:
        raise DatasetError(
            "continuous-time-lidar requires a motion-compensated target stream and odometry"
        )

    frame_ids = sorted({entry[0] for entry in target_stream})
    holdout_start = max(1, int(len(frame_ids) * continuous_options.holdout_start_fraction))
    holdout_frame_ids = set(frame_ids[holdout_start:])
    if not holdout_frame_ids and len(frame_ids) >= 2:
        holdout_frame_ids = {frame_ids[-1]}
    train_indices = tuple(
        index for index, entry in enumerate(target_stream) if entry[0] not in holdout_frame_ids
    )
    holdout_indices = tuple(
        index for index, entry in enumerate(target_stream) if entry[0] in holdout_frame_ids
    )
    if not train_indices:
        raise DatasetError("continuous-time-lidar temporal holdout left no train points")

    t_source_target_init = source_node.transform_to_parent.inverse().compose(
        target_node.transform_to_parent
    )
    variable = f"T_{target_node.parent}_{target_sensor}"
    target_points = [entry[1] for entry in target_stream]
    target_capture_timestamps_ns = [entry[4] for entry in target_stream]
    problem = ContinuousTimeLidarPairProblem(
        source_records=source_records,
        target_points=target_points,
        target_capture_timestamps_ns=target_capture_timestamps_ns,
        odometry_track=odometry_track,
        t_base_source=source_node.transform_to_parent,
        initial_t_source_target=t_source_target_init,
        variable=variable,
        sensor=target_sensor,
        voxel_size_m=inputs.voxel_size_m,
        correspondence_gate_m=inputs.correspondence_gate_m,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
    )
    solved = ContinuousTimeLidarPairSolver().solve(problem, continuous_options)
    refined_grade: Grade = "pass" if solved.status == "converged" else "warn"
    initial_transform = _continuous_time_transform_result(
        parent=source_sensor,
        child=target_sensor,
        transform=solved.initial_t_source_target,
        role="initial",
        grade="warn",
    )
    refined_transform = _continuous_time_transform_result(
        parent=source_sensor,
        child=target_sensor,
        transform=solved.refined_t_source_target,
        role="output",
        grade=refined_grade,
    )
    evaluation = solved.observability
    source_sha256: dict[str, str] = {}
    source_paths = [str(config_file), str(bag_path)]
    for path in (config_file, bag_path):
        digest = sha256_path(path)
        if digest is None:
            raise DatasetError(f"could not hash continuous-time input: {path}")
        source_sha256[str(path)] = digest
    notes = [
        "continuous-time profile re-solves the six-DoF extrinsic at every clock candidate",
        "target holdout is the latest capture frame group and is never used for optimization",
        (
            "trajectory_model is piecewise SE(3) interpolation of supplied odometry; "
            "trajectory knots are fixed"
        ),
        f"train_frame_count={len(frame_ids) - len(holdout_frame_ids)}",
        f"holdout_frame_count={len(holdout_frame_ids)}",
        f"holdout_start_fraction={continuous_options.holdout_start_fraction}",
        f"sampling_seed={continuous_options.sampling_seed}",
        f"source_record_cap={continuous_options.max_source_records}",
        f"target_point_cap_per_split={continuous_options.max_target_points_per_split}",
        f"outlier_policy={continuous_options.outlier_policy}",
        f"voxel_strategy={continuous_options.voxel_strategy}",
        (
            "replay_source_message_count="
            + str(replay_provenance.get(f"{config.dataset.type}_source_message_count", 0))
        ),
    ]
    if continuous_options.voxel_strategy == "adaptive":
        if continuous_options.adaptive_use_uniform_fallback:
            notes.append(
                "adaptive correspondences use a uniform voxel fallback and select the "
                "lower initial point-to-plane residual"
            )
        else:
            notes.append(
                "adaptive correspondences use the adaptive plane map without a "
                "uniform fallback"
            )
    return ContinuousTimeLidarPairArtifact(
        config_path=str(config_file),
        dataset_path=str(bag_path),
        source_sensor=source_sensor,
        target_sensor=target_sensor,
        odometry_topic=odometry_source_info.topic,
        variable=variable,
        trajectory_model=solved.trajectory_model,
        time_offset_sign_convention=solved.time_offset_sign_convention,
        initial_transform=initial_transform,
        refined_transform=refined_transform,
        initial_time_offset_sec=solved.initial_time_offset_sec,
        estimated_time_offset_sec=solved.estimated_time_offset_sec,
        initial_train_rmse_m=solved.initial_train_rmse_m,
        final_train_rmse_m=solved.final_train_rmse_m,
        final_holdout_rmse_m=solved.final_holdout_rmse_m,
        train_correspondence_count=solved.train_correspondence_count,
        holdout_correspondence_count=solved.holdout_correspondence_count,
        outlier_rejected_count=solved.outlier_rejected_count,
        observability=ContinuousTimeLidarPairObservability(
            rank=evaluation.rank if evaluation is not None else None,
            condition_number=(
                evaluation.normalized_condition_number_estimate
                if evaluation is not None
                else None
            ),
            weak_directions=list(evaluation.weak_directions) if evaluation is not None else [],
            residual_count=evaluation.residual_count if evaluation is not None else 0,
        ),
        status=solved.status,
        reason=solved.reason,
        options=ContinuousTimeLidarPairOptionsArtifact(
            initial_time_offset_sec=continuous_options.initial_time_offset_sec,
            max_abs_time_offset_sec=continuous_options.max_abs_time_offset_sec,
            initial_time_step_sec=continuous_options.initial_time_step_sec,
            minimum_time_step_sec=continuous_options.minimum_time_step_sec,
            max_iterations=continuous_options.max_iterations,
            min_correspondences=continuous_options.min_correspondences,
            max_odometry_extrapolation_s=continuous_options.max_odometry_extrapolation_s,
            max_source_records=continuous_options.max_source_records,
            max_target_points_per_split=continuous_options.max_target_points_per_split,
            sampling_seed=continuous_options.sampling_seed,
            holdout_start_fraction=continuous_options.holdout_start_fraction,
            outlier_policy=continuous_options.outlier_policy,
            outlier_mad_scale=continuous_options.outlier_mad_scale,
            outlier_min_threshold_m=continuous_options.outlier_min_threshold_m,
            outlier_min_inlier_fraction=continuous_options.outlier_min_inlier_fraction,
            voxel_strategy=continuous_options.voxel_strategy,
            adaptive_use_uniform_fallback=continuous_options.adaptive_use_uniform_fallback,
            adaptive_range_reference_m=continuous_options.adaptive_range_reference_m,
            adaptive_range_exponent=continuous_options.adaptive_range_exponent,
            adaptive_min_voxel_size_m=continuous_options.adaptive_min_voxel_size_m,
            adaptive_max_voxel_size_m=continuous_options.adaptive_max_voxel_size_m,
            adaptive_min_points_per_voxel=continuous_options.adaptive_min_points_per_voxel,
            correspondence_refinement_iterations=(
                continuous_options.correspondence_refinement_iterations
            ),
            fixed_extrinsic_max_iterations=continuous_options.solver.max_iterations,
            fixed_extrinsic_robust_loss=continuous_options.solver.robust_loss,
        ),
        iterations=[
            ContinuousTimeLidarPairIterationArtifact(
                iteration=item.iteration,
                candidate_offsets_sec=list(item.candidate_offsets_sec),
                candidate_train_rmse_m=list(item.candidate_train_rmse_m),
                candidate_holdout_rmse_m=list(item.candidate_holdout_rmse_m),
                selected_offset_sec=item.selected_offset_sec,
                train_rmse_m=item.train_rmse_m,
                holdout_rmse_m=item.holdout_rmse_m,
                accepted=item.accepted,
            )
            for item in solved.iterations
        ],
        provenance=ContinuousTimeLidarPairProvenance(
            source_paths=source_paths,
            source_sha256=source_sha256,
            tool_version=__version__,
            git_commit=git_commit(),
            notes=notes,
        ),
    )


def evaluate_rosbag2_continuous_time_lidar_pair(
    config_path: str | Path,
) -> ContinuousTimeLidarPairArtifact:
    """Backward-compatible ROS 2 entry point for continuous-time LiDAR pairing."""

    config = load_config(config_path)
    if config.dataset.type != "rosbag2":
        raise ConfigError(
            "evaluate_rosbag2_continuous_time_lidar_pair requires dataset.type: rosbag2"
        )
    return evaluate_continuous_time_lidar_pair(config_path)


def _continuous_time_lidar_options(
    config: CalibrationConfig,
    options: Mapping[str, Any],
) -> ContinuousTimeLidarPairOptions:
    """Parse continuous-time LiDAR options without changing config schema."""

    raw_outlier_policy = str(options.get("continuous_time_outlier_policy", "mad"))
    outlier_policy: Literal["none", "mad"] = (
        "none" if raw_outlier_policy == "none" else "mad"
    )
    raw_voxel_strategy = str(options.get("continuous_time_voxel_strategy", "uniform"))
    voxel_strategy: Literal["uniform", "adaptive"] = (
        "adaptive" if raw_voxel_strategy == "adaptive" else "uniform"
    )
    return ContinuousTimeLidarPairOptions(
        initial_time_offset_sec=_signed_float_option(
            dict(options), "continuous_time_initial_offset_s", default=0.0
        ),
        max_abs_time_offset_sec=_float_option(
            dict(options),
            "continuous_time_max_abs_offset_s",
            default=0.20,
            minimum=0.0,
        ),
        initial_time_step_sec=_float_option(
            dict(options),
            "continuous_time_initial_step_s",
            default=0.02,
            minimum=1.0e-6,
        ),
        minimum_time_step_sec=_float_option(
            dict(options),
            "continuous_time_minimum_step_s",
            default=0.001,
            minimum=1.0e-6,
        ),
        max_iterations=_int_option(
            dict(options), "continuous_time_max_iterations", default=8
        )
        or 8,
        min_correspondences=_int_option(
            dict(options), "continuous_time_min_correspondences", default=6
        )
        or 6,
        max_odometry_extrapolation_s=_float_option(
            dict(options),
            "continuous_time_max_odometry_extrapolation_s",
            default=0.25,
            minimum=0.0,
        ),
        max_source_records=_int_option(
            dict(options), "continuous_time_max_source_records", default=6000
        ),
        max_target_points_per_split=_int_option(
            dict(options),
            "continuous_time_max_target_points_per_split",
            default=6000,
        ),
        sampling_seed=_seed_option(
            dict(options),
            "continuous_time_sampling_seed",
            default=config.solver.seed or 0,
        ),
        holdout_start_fraction=_unit_interval_option(
            dict(options),
            "continuous_time_holdout_start_fraction",
            default=0.8,
        ),
        outlier_policy=outlier_policy,
        outlier_mad_scale=_float_option(
            dict(options),
            "continuous_time_outlier_mad_scale",
            default=3.5,
            minimum=0.0,
        ),
        outlier_min_threshold_m=_float_option(
            dict(options),
            "continuous_time_outlier_min_threshold_m",
            default=0.02,
            minimum=0.0,
        ),
        outlier_min_inlier_fraction=_float_option(
            dict(options),
            "continuous_time_outlier_min_inlier_fraction",
            default=0.25,
            minimum=1.0e-6,
        ),
        voxel_strategy=voxel_strategy,
        adaptive_use_uniform_fallback=_bool_option(
            dict(options),
            "continuous_time_adaptive_use_uniform_fallback",
            default=True,
        ),
        adaptive_range_reference_m=_float_option(
            dict(options),
            "continuous_time_adaptive_range_reference_m",
            default=10.0,
            minimum=1.0e-6,
        ),
        adaptive_range_exponent=_float_option(
            dict(options),
            "continuous_time_adaptive_range_exponent",
            default=0.5,
            minimum=0.0,
        ),
        adaptive_min_voxel_size_m=_float_option(
            dict(options),
            "continuous_time_adaptive_min_voxel_size_m",
            default=0.25,
            minimum=1.0e-6,
        ),
        adaptive_max_voxel_size_m=_float_option(
            dict(options),
            "continuous_time_adaptive_max_voxel_size_m",
            default=1.5,
            minimum=1.0e-6,
        ),
        adaptive_min_points_per_voxel=(
            _int_option(
                dict(options),
                "continuous_time_adaptive_min_points_per_voxel",
                default=3,
            )
            or 3
        ),
        correspondence_refinement_iterations=(
            _int_option(
                dict(options),
                "continuous_time_correspondence_refinement_iterations",
                default=1,
            )
            or 0
        ),
        solver=FixedTrajectorySe3SolverOptions(
            max_iterations=(
                _int_option(
                    dict(options),
                    "continuous_time_fixed_extrinsic_max_iterations",
                    default=max(1, config.solver.max_iterations),
                )
                or max(1, config.solver.max_iterations)
            ),
            convergence_tolerance=config.solver.convergence_tolerance,
            robust_loss=_robust_loss(config.solver.robust_loss),
        ),
    )


def _continuous_time_transform_result(
    *,
    parent: str,
    child: str,
    transform: SE3,
    role: Literal["initial", "output"],
    grade: Grade,
) -> TransformResult:
    """Package a continuous-time transform with explicit native provenance."""

    return TransformResult(
        parent=parent,
        child=child,
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        quality=TransformQuality(grade=grade),
        estimate_id=f"T_{parent}_{child}",
        provenance=TransformEstimateProvenance(
            producer="slac_native",
            execution_mode="online_stream",
            role_in_comparison=role,
            evidence_level=(
                "algorithmically_refined" if role == "output" else "unknown"
            ),
            tool_name="continuous_time_lidar_pair_solver",
            source="calibrex.solvers.continuous_time_lidar_pair_solver",
            notes=[
                "clock offset was profiled with extrinsic re-optimization",
                "supplied piecewise odometry trajectory remained fixed",
            ],
        ),
    )


def _capture_readiness_options(options: Mapping[str, Any]) -> dict[str, int]:
    """Read bounded source-scan limits for the readiness preflight."""

    max_scans = _int_option(
        dict(options), "readiness_max_scans_per_window", default=40
    ) or 40
    max_points = _int_option(
        dict(options), "readiness_max_points_per_window", default=4000
    ) or 4000
    return {
        "max_scans_per_window": max(1, max_scans),
        "max_points_per_window": max(1, max_points),
    }


def _capture_readiness_thresholds(options: Mapping[str, Any]) -> CaptureReadinessThresholds:
    """Read user-overridable readiness gates from factor options."""

    values = dict(options)
    min_rotation = _float_option(
        values, "readiness_min_endpoint_rotation_deg", default=10.0, minimum=0.0
    )
    warn_rotation = _float_option(
        values, "readiness_warn_endpoint_rotation_deg", default=25.0, minimum=min_rotation
    )
    return CaptureReadinessThresholds(
        min_pose_count=(
            _int_option(values, "readiness_min_pose_count", default=10) or 10
        ),
        min_time_span_s=_float_option(
            values, "readiness_min_time_span_s", default=5.0, minimum=0.0
        ),
        min_path_length_m=_float_option(
            values, "readiness_min_path_length_m", default=1.0, minimum=0.0
        ),
        min_endpoint_rotation_deg=min_rotation,
        warn_endpoint_rotation_deg=warn_rotation,
        min_plane_count=(
            _int_option(values, "readiness_min_plane_count", default=10) or 10
        ),
        min_normal_rank=(
            _int_option(values, "readiness_min_normal_rank", default=2) or 2
        ),
        max_normal_condition_number=_float_option(
            values,
            "readiness_max_normal_condition_number",
            default=100.0,
            minimum=1.0,
        ),
        min_observable_dof=(
            _int_option(values, "readiness_min_observable_dof", default=6) or 6
        ),
        max_angular_speed_dps=_float_option(
            values,
            "readiness_max_angular_speed_dps",
            default=120.0,
            minimum=0.0,
        ),
        max_linear_speed_mps=_float_option(
            values,
            "readiness_max_linear_speed_mps",
            default=3.0,
            minimum=0.0,
        ),
    )


def _capture_readiness_geometry(
    source_scans: CrossSegmentScanCollection | None,
    *,
    initial_t_source_target: SE3,
    voxel_size_m: float,
    correspondence_gate_m: float,
    max_points: int,
    variable: str,
    sensor: str,
) -> CaptureReadinessGeometry:
    """Build plane-normal and local six-DoF evidence from selected source scans."""

    if source_scans is None:
        return CaptureReadinessGeometry(
            source_scan_count=0,
            source_point_count=0,
            plane_count=0,
            normal_rank=0,
            normal_condition_number=None,
            normal_diversity_score=0.0,
            estimated_observable_dof=0,
            correspondence_count=0,
            normal_source="unavailable",
        )
    scans = [*source_scans.first_half_scans, *source_scans.second_half_scans]
    records = [record for scan in scans for record in scan.records]
    if not records:
        return CaptureReadinessGeometry(
            source_scan_count=len(scans),
            source_point_count=0,
            plane_count=0,
            normal_rank=0,
            normal_condition_number=None,
            normal_diversity_score=0.0,
            estimated_observable_dof=0,
            correspondence_count=0,
            normal_source="unavailable",
        )
    plane_map = build_voxel_plane_map(records, voxel_size_m)
    normals = [plane.normal for plane in plane_map.values() if plane.normal is not None]
    normal_rank, normal_condition, normal_diversity, eigenvalues = _normal_spectrum(normals)
    observations: list[LidarPointToPlaneObservation] = []
    inverse_initial = initial_t_source_target.inverse()
    for index in _subsample_indices(len(records), max_points):
        record = records[index]
        point_world = (record.point[0], record.point[1], record.point[2])
        plane = nearest_voxel_plane(
            point_world,
            plane_map,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
        )
        if plane is None or plane.normal is None:
            continue
        observations.append(
            LidarPointToPlaneObservation(
                point_lidar_m=inverse_initial.transform_point(point_world),
                plane_point_world_m=plane.centroid,
                plane_normal_world=plane.normal,
            )
        )
    evaluation = None
    if observations:
        evaluation = LidarRigPointToPlaneFactor(
            variable=variable,
            t_ego_lidar=initial_t_source_target,
            observations=observations,
            sensor=sensor,
        ).evaluate()
    return CaptureReadinessGeometry(
        source_scan_count=len(scans),
        source_point_count=len(records),
        plane_count=len(normals),
        normal_rank=normal_rank,
        normal_condition_number=normal_condition,
        normal_diversity_score=normal_diversity,
        normal_eigenvalues=eigenvalues,
        estimated_observable_dof=evaluation.rank if evaluation is not None else 0,
        weak_directions=list(evaluation.weak_directions) if evaluation is not None else [],
        correspondence_count=len(observations),
        normal_source="voxel_plane_map" if normals else "unavailable",
    )


def _normal_spectrum(
    normals: list[tuple[float, float, float]],
) -> tuple[int, float | None, float, list[float]]:
    """Return rank, condition, diversity, and eigenvalues of normal scatter."""

    if not normals:
        return 0, None, 0.0, []
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    for normal in normals:
        for row in range(3):
            for column in range(3):
                matrix[row][column] += normal[row] * normal[column]
    eigenvalues = _symmetric_eigenvalues_3x3(matrix)
    maximum = max(eigenvalues)
    if maximum <= 1.0e-12:
        return 0, None, 0.0, eigenvalues
    rank = sum(value > maximum * 1.0e-3 for value in eigenvalues)
    positive = [value for value in eigenvalues if value > maximum * 1.0e-3]
    condition = maximum / min(positive) if rank == 3 and positive else None
    diversity = max(0.0, min(1.0, eigenvalues[0] / maximum))
    return rank, condition, diversity, eigenvalues


def _symmetric_eigenvalues_3x3(matrix: list[list[float]]) -> list[float]:
    """Compute sorted eigenvalues of a real symmetric 3x3 matrix."""

    a, b, c = matrix[0]
    _b, d, e = matrix[1]
    _c, _e, f = matrix[2]
    off_diagonal = b * b + c * c + e * e
    if off_diagonal <= 1.0e-24:
        return sorted((max(0.0, a), max(0.0, d), max(0.0, f)))
    trace_third = (a + d + f) / 3.0
    centered_norm = ((a - trace_third) ** 2 + (d - trace_third) ** 2 + (f - trace_third) ** 2)
    centered_norm += 2.0 * off_diagonal
    scale = math.sqrt(centered_norm / 6.0)
    if scale <= 1.0e-12:
        return [0.0, 0.0, 0.0]
    normalized = [
        [(matrix[row][column] - (trace_third if row == column else 0.0)) / scale
         for column in range(3)]
        for row in range(3)
    ]
    determinant = _determinant3(normalized) / 2.0
    if determinant <= -1.0:
        angle = math.pi / 3.0
    elif determinant >= 1.0:
        angle = 0.0
    else:
        angle = math.acos(determinant) / 3.0
    first = trace_third + 2.0 * scale * math.cos(angle)
    third = trace_third + 2.0 * scale * math.cos(angle + 2.0 * math.pi / 3.0)
    second = 3.0 * trace_third - first - third
    return sorted(max(0.0, value) for value in (first, second, third))


def _determinant3(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _trajectory_window_cross_segment_evidence(
    evidence: Any,
) -> TrajectoryWindowCrossSegmentEvidence:
    """Convert the internal trajectory evidence row to the standalone schema."""

    return TrajectoryWindowCrossSegmentEvidence(
        rmse_m=evidence.rmse_m,
        first_half_scan_count=evidence.first_half_scan_count,
        second_half_scan_count=evidence.second_half_scan_count,
        first_half_point_count=evidence.first_half_point_count,
        second_half_point_count=evidence.second_half_point_count,
        first_half_start_timestamp_ns=evidence.first_half_start_timestamp_ns,
        first_half_end_timestamp_ns=evidence.first_half_end_timestamp_ns,
        second_half_start_timestamp_ns=evidence.second_half_start_timestamp_ns,
        second_half_end_timestamp_ns=evidence.second_half_end_timestamp_ns,
        sampling_policy=evidence.sampling_policy,
        correspondence_count=evidence.correspondence_count,
        gate_status=evidence.gate_status,
        gate_reason=evidence.gate_reason,
    )


def _trajectory_window_odometry_stats(
    odometry_track: OdometryTrack,
    window: tuple[int | None, int | None],
) -> TrajectoryWindowOdometryStats:
    """Summarize the retained odometry samples inside one capture window."""

    first_ns = odometry_track.first_timestamp_ns
    last_ns = odometry_track.last_timestamp_ns
    if first_ns is None or last_ns is None:
        return TrajectoryWindowOdometryStats(
            pose_count=0,
            time_span_s=0.0,
            path_length_m=0.0,
            endpoint_displacement_m=0.0,
            endpoint_rotation_deg=0.0,
        )
    start_ns = max(first_ns, window[0] if window[0] is not None else first_ns)
    end_ns = min(last_ns, window[1] if window[1] is not None else last_ns)
    samples = [
        sample for sample in odometry_track.samples if start_ns <= sample.timestamp_ns <= end_ns
    ]
    if not samples:
        return TrajectoryWindowOdometryStats(
            pose_count=0,
            first_timestamp_ns=None,
            last_timestamp_ns=None,
            time_span_s=0.0,
            path_length_m=0.0,
            endpoint_displacement_m=0.0,
            endpoint_rotation_deg=0.0,
        )
    path_length_m = sum(
        math.dist(left.pose.translation_m, right.pose.translation_m)
        for left, right in pairwise(samples)
    )
    relative = samples[0].pose.inverse().compose(samples[-1].pose)
    quaternion = relative.rotation_quat_xyzw
    scalar = min(1.0, max(-1.0, abs(quaternion[3])))
    endpoint_rotation_deg = math.degrees(2.0 * math.acos(scalar))
    return TrajectoryWindowOdometryStats(
        pose_count=len(samples),
        first_timestamp_ns=samples[0].timestamp_ns,
        last_timestamp_ns=samples[-1].timestamp_ns,
        time_span_s=(samples[-1].timestamp_ns - samples[0].timestamp_ns) / 1_000_000_000,
        path_length_m=path_length_m,
        endpoint_displacement_m=math.dist(
            samples[0].pose.translation_m, samples[-1].pose.translation_m
        ),
        endpoint_rotation_deg=endpoint_rotation_deg,
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
            producer="slac_native",
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
            slac_version=__version__,
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
                **_dataset_manifest_provenance(config_file),
            },
        ),
        frame_graph=frame_graph.snapshot(),
        solid_state=_solid_state_context_for_config(config, config_file),
        metrics={
            "online_lidar_point_to_plane_available": MetricResult(
                value=0.0, grade="warn", reason=reason
            )
        },
        quality=QualitySummary(grade="warn", warnings=[reason]),
    )


    result.metrics.update(solid_state_metrics_from_inspection(config, None, result))
    if result.solid_state is not None:
        result.run.provenance["solid_state_evaluation_config"] = (
            config.evaluation.solid_state.model_dump(mode="json")
        )
    result.save(output_dir / config.outputs.result)
    write_report_artifacts(result, output_dir, html_filename=config.outputs.report)
    return result


def _rosbag2_cross_segment_scan_records(
    message: PointCloud2Message | LivoxCustomMessage,
    *,
    timestamp_ns: int,
    point_time_field: str | None,
    use_point_time_offsets: bool,
    t_base_source: SE3,
    pose_track: OdometryTrack,
    max_points: int,
    extrapolation_tolerance_s: float | None,
) -> list[LivoxPointRecord]:
    """Transform one selected ROS2 source scan for trajectory evidence."""

    offsets_s, point_time_reference_ns, _point_time_reference = _rosbag2_point_time_data(
        message,
        point_time_field,
        ros_timestamp_ns=timestamp_ns,
    )
    if (
        use_point_time_offsets
        and offsets_s is not None
        and point_time_reference_ns is not None
    ):
        return _deskew_pointcloud_to_world_records(
            message.xyz,
            offsets_s,
            point_time_reference_ns,
            t_base_source,
            pose_track,
            max_points,
            extrapolation_tolerance_s,
        )
    t_world_base, _clamped, extrapolation_s = pose_track.interpolate(
        _rosbag2_message_timestamp_ns(message, timestamp_ns)
    )
    if extrapolation_tolerance_s is not None and extrapolation_s > extrapolation_tolerance_s:
        msg = (
            "cross-segment source LiDAR message timestamp lies "
            f"{extrapolation_s:.4f} s outside the odometry track "
            f"(tolerance {extrapolation_tolerance_s:.4f} s)"
        )
        raise DatasetError(msg)
    return _transform_pointcloud_to_world_records(
        message.xyz,
        t_world_base.compose(t_base_source),
        max_points,
    )


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
    t_source_target_init: SE3,
    batch_size: int,
    replay_provenance: dict[str, Any] | None = None,
    motion_compensated: bool = False,
    target_stream: list[OnlineTargetEntry] | None = None,
    odometry_track: OdometryTrack | None = None,
    odometry_source_info: OdometrySourceInfo | None = None,
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
            producer="slac_native",
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

    factor = config.pipeline.factors.get(_FACTOR_NAME)
    factor_options: dict[str, Any] = dict(factor.options) if factor is not None else {}
    temporal_options = temporal_evidence_options_from_factor(factor_options)
    trajectory_options = trajectory_evidence_options_from_factor(factor_options)
    dual_temporal_result: DualTemporalEvidenceResult | None = None
    trajectory_result: TrajectoryEvidenceResult | None = None
    trajectory_metrics: dict[str, MetricResult] = {}
    run_provenance: dict[str, Any] = {
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
        **_dataset_manifest_provenance(config_file),
        **(replay_provenance or {}),
    }
    if target_stream:
        source_window_count = _int_or_none(
            (replay_provenance or {}).get("rosbag1_source_message_count")
            or (replay_provenance or {}).get("rosbag2_source_message_count")
        )
        target_window_ids = sorted(
            {
                frame_index
                for frame_index, _point, _pose, _extrapolation, _capture_ns in target_stream
            }
        )
        temporal_holdout_independent = (
            replay_provenance or {}
        ).get(f"{config.dataset.type}_temporal_holdout_independent") is True
        independent_holdout = temporal_holdout_independent
        train_window_count = (
            _int_or_none(
                (replay_provenance or {}).get(
                    f"{config.dataset.type}_target_train_window_count"
                )
            )
            or max(1, int(len(target_window_ids) * (1.0 - session.holdout_ratio)))
        )
        holdout_window_count = (
            _int_or_none(
                (replay_provenance or {}).get(
                    f"{config.dataset.type}_target_holdout_window_count"
                )
            )
            or max(0, len(target_window_ids) - train_window_count)
        )
        holdout_window_ids = (
            target_window_ids[-holdout_window_count:]
            if holdout_window_count > 0
            else []
        )
        run_provenance["solid_state_evaluation"] = {
            "independent_holdout": independent_holdout,
            "temporal_holdout_independent": temporal_holdout_independent,
            "split_policy": (
                "source-map messages before the target holdout boundary; later "
                "target capture windows are reserved for evaluation"
                if config.dataset.type == "rosbag1"
                else (
                    "source-map messages before the target holdout boundary; later "
                    "target capture windows are reserved for evaluation, with each "
                    "target batch also carrying a deterministic point holdout"
                )
            ),
            "source_map_window_count": source_window_count or 0,
            "train_window_count": train_window_count,
            "holdout_window_count": holdout_window_count,
            "holdout_window_ids": holdout_window_ids,
            "known_bad_case_count": 0,
            "known_bad_evaluation_status": "not_yet_run",
        }
    if temporal_evidence_requested(temporal_options):
        if motion_compensated and odometry_track is not None and target_stream:
            target_points = [point for _fi, point, _pose, _extrap, _capture in target_stream]
            capture_timestamps_ns = [
                capture_ns for _fi, _point, _pose, _extrap, capture_ns in target_stream
            ]
            dual_temporal_result = evaluate_dual_temporal_evidence(
                source_records=session.source_records,
                target_points=target_points,
                capture_timestamps_ns=capture_timestamps_ns,
                adapted_t_source_target=session.current_estimate,
                anchored_t_source_target=t_source_target_init,
                odometry_track=odometry_track,
                t_base_source=t_base_source,
                variable=variable,
                sensor=target_sensor,
                voxel_size_m=session.voxel_size_m,
                correspondence_gate_m=session.correspondence_gate_m,
                holdout_ratio=session.holdout_ratio,
                seed=session.seed,
                options=temporal_options,
                max_odometry_extrapolation_s=session.gate_thresholds.max_odometry_extrapolation_s,
            )
            run_provenance["temporal_evidence"] = dual_temporal_evidence_to_provenance(
                dual_temporal_result
            )
            run_provenance["online_temporal_gate_status"] = dual_temporal_result.selected.verdict
            metrics.update(_temporal_evidence_metrics(dual_temporal_result))
        else:
            run_provenance["temporal_evidence_skipped_no_odometry"] = True

    if motion_compensated and odometry_track is not None and odometry_source_info is not None:
        cross_segment_scans: CrossSegmentScanCollection | None = None
        if config.dataset.type == "rosbag2":
            bag_path = Path(config.dataset.path)
            if bag_path.exists():
                cross_segment_scans = collect_rosbag2_cross_segment_scans(
                    bag_path=bag_path,
                    source_topic=_lidar_topic(config, source_sensor),
                    source_point_time_field=_deskew_time_field_map(
                        config, source_sensor, target_sensor
                    ).get(source_sensor),
                    use_point_time_offsets=_point_time_deskew_enabled(config),
                    odometry_track=odometry_track,
                    t_base_source=t_base_source,
                    options=trajectory_options,
                    extrapolation_tolerance_s=session.gate_thresholds.max_odometry_extrapolation_s,
                    capture_window_start_timestamp_ns=(
                        replay_provenance.get(
                            "rosbag2_capture_window_start_timestamp_ns"
                        )
                        if replay_provenance is not None
                        else None
                    ),
                    capture_window_end_timestamp_ns=(
                        replay_provenance.get("rosbag2_capture_window_end_timestamp_ns")
                        if replay_provenance is not None
                        else None
                    ),
                    capture_window_record_prefilter_margin_s=(
                        float(
                            replay_provenance.get(
                                "rosbag2_capture_window_record_prefilter_margin_s",
                                0.0,
                            )
                        )
                        if replay_provenance is not None
                        else 0.0
                    ),
                    capture_windows=_capture_windows_from_provenance(
                        replay_provenance, prefix="rosbag2"
                    ),
                )
        trajectory_result = evaluate_trajectory_evidence(
            odometry_track=odometry_track,
            cross_segment_scans=cross_segment_scans,
            voxel_size_m=session.voxel_size_m,
            correspondence_gate_m=session.correspondence_gate_m,
            variable=variable,
            sensor=target_sensor,
            options=trajectory_options,
            max_odometry_extrapolation_s=session.gate_thresholds.max_odometry_extrapolation_s,
        )
        run_provenance["trajectory_evidence"] = trajectory_evidence_to_provenance(
            trajectory_result
        )
        run_provenance["online_trajectory_gate_status"] = trajectory_result.verdict
        trajectory_metrics = trajectory_evidence_metrics(trajectory_result)
        metrics.update(trajectory_metrics)
    else:
        run_provenance["trajectory_evidence_skipped_no_odometry"] = True

    known_bad_metrics, known_bad_evidence = _online_known_bad_evidence(
        source_records=session.source_records,
        target_stream=target_stream or [],
        target_transform=session.current_estimate,
        variable=variable,
        sensor=target_sensor,
        voxel_size_m=session.voxel_size_m,
        correspondence_gate_m=session.correspondence_gate_m,
        min_detection_fraction=config.evaluation.solid_state.min_known_bad_detectable_fraction,
    )
    metrics.update(known_bad_metrics)
    run_provenance["online_known_bad_evidence"] = known_bad_evidence
    solid_state_evaluation = run_provenance.get("solid_state_evaluation")
    if isinstance(solid_state_evaluation, dict):
        solid_state_evaluation.update(
            {
                "known_bad_case_count": known_bad_evidence.get("case_count"),
                "known_bad_detectable_fraction": known_bad_evidence.get(
                    "detectable_fraction"
                ),
                "known_bad_evaluation_status": known_bad_evidence.get("status"),
            }
        )

    result = CalibrationResult(
        run=RunInfo(
            id=_run_id(config.project.name),
            slac_version=__version__,
            git_commit=git_commit(),
            config_sha256=sha256_path(config_file),
            dataset_sha256=sha256_path(Path(config.dataset.path)),
            status="warning",
            domain=config.project.domain,
            provenance=run_provenance,
        ),
        frame_graph=frame_graph.snapshot(),
        solid_state=_solid_state_context_for_config(config, config_file),
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
    _attach_online_capture_window(
        result,
        config,
        target_sensor,
        target_stream,
        replay_provenance,
    )
    result.metrics.update(solid_state_metrics_from_inspection(config, None, result))
    if result.solid_state is not None:
        result.run.provenance["solid_state_evaluation_config"] = (
            config.evaluation.solid_state.model_dump(mode="json")
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

    if (
        motion_compensated
        and odometry_track is not None
        and odometry_source_info is not None
        and trajectory_result is not None
    ):
        trajectory_path = output_dir / _TRAJECTORY_FILENAME
        trajectory_artifact = build_trajectory_artifact(
            run=_report_run_info(result),
            odometry_track=odometry_track,
            odometry_source=odometry_source_info,
            evidence=trajectory_result,
            metrics=trajectory_metrics,
        )
        write_mapping(
            trajectory_path,
            trajectory_artifact.model_dump(mode="json", exclude_none=True),
        )
        result.run.provenance["trajectory_path"] = str(trajectory_path)
        result.save(output_dir / config.outputs.result)

    return result


def _online_known_bad_evidence(
    *,
    source_records: list[LivoxPointRecord],
    target_stream: list[OnlineTargetEntry],
    target_transform: SE3,
    variable: str,
    sensor: str,
    voxel_size_m: float,
    correspondence_gate_m: float,
    min_detection_fraction: float,
) -> tuple[dict[str, MetricResult], dict[str, Any]]:
    """Score held-out online windows against six known-bad extrinsic probes."""

    frame_ids = sorted({entry[0] for entry in target_stream})
    if len(frame_ids) < 2 or not source_records:
        reason = "online known-bad evidence needs a source map and at least two target windows"
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
            {
                "status": "unavailable",
                "reason": reason,
                "case_count": 0,
                "detectable_fraction": None,
            },
        )

    holdout_start = max(1, int(len(frame_ids) * 0.8))
    holdout_frame_ids = set(frame_ids[holdout_start:])
    if not holdout_frame_ids:
        holdout_frame_ids = {frame_ids[-1]}
    holdout_entries = [entry for entry in target_stream if entry[0] in holdout_frame_ids]
    if len(holdout_entries) > 4000:
        step = (len(holdout_entries) + 3999) // 4000
        holdout_entries = holdout_entries[::step]
    target_points = [entry[1] for entry in holdout_entries]
    target_poses = [entry[2] for entry in holdout_entries]
    observations = build_rig_point_to_plane_observations(
        source_records=source_records,
        target_points=target_points,
        initial_t_source_target=target_transform,
        target_t_world_source=target_poses,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
    )
    if len(observations) < _MIN_HOLDOUT_OBSERVATIONS:
        reason = (
            "online known-bad holdout has too few point-to-plane correspondences "
            f"({len(observations)} < {_MIN_HOLDOUT_OBSERVATIONS})"
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
            {
                "status": "inconclusive",
                "reason": reason,
                "case_count": 0,
                "detectable_fraction": None,
                "holdout_window_count": len(holdout_frame_ids),
                "holdout_correspondence_count": len(observations),
            },
        )

    baseline_factor = LidarRigPointToPlaneFactor(
        variable=variable,
        t_ego_lidar=target_transform,
        observations=observations,
        sensor=sensor,
    )
    baseline_residuals = baseline_factor.residuals()
    baseline_rmse = _rmse_from_values(baseline_residuals)
    baseline_p90 = _p90_from_values(baseline_residuals)
    cases: list[dict[str, Any]] = []
    detected = 0
    for dof, candidate in _online_known_bad_candidates(target_transform):
        candidate_factor = LidarRigPointToPlaneFactor(
            variable=variable,
            t_ego_lidar=candidate,
            observations=observations,
            sensor=sensor,
        )
        candidate_residuals = candidate_factor.residuals()
        candidate_rmse = _rmse_from_values(candidate_residuals)
        candidate_p90 = _p90_from_values(candidate_residuals)
        worsened = (
            baseline_rmse is not None
            and candidate_rmse is not None
            and candidate_rmse > baseline_rmse + 1.0e-3
        ) or (
            baseline_p90 is not None
            and candidate_p90 is not None
            and candidate_p90 > baseline_p90 + 1.0e-3
        )
        if worsened:
            detected += 1
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
        if detectable_fraction is not None and detectable_fraction >= min_detection_fraction
        else "warn"
    )
    return (
        {
            "lidar_pair_holdout_point_to_plane_rmse_m": MetricResult(
                value=baseline_rmse,
                unit="m",
                grade="pass" if baseline_rmse is not None else "warn",
                reason=(
                    "final online extrinsic scored on target frames held out from "
                    "the source-map replay"
                ),
            ),
            "lidar_pair_known_bad_case_count": MetricResult(
                value=float(len(cases)),
                unit="cases",
                grade=grade,
                reason="six fixed roll/pitch/yaw/x/y/z perturbations scored on online holdout",
            ),
            "lidar_pair_known_bad_detectable_fraction": MetricResult(
                value=detectable_fraction,
                grade=grade,
                reason=(
                    "fraction of online six-DoF known-bad perturbations whose holdout "
                    f"RMSE or P90 increased; target >= {min_detection_fraction:g}"
                ),
            ),
        },
        {
            "status": "scored",
            "split_policy": "last_target_capture_windows_holdout",
            "holdout_window_count": len(holdout_frame_ids),
            "holdout_correspondence_count": len(observations),
            "case_count": len(cases),
            "detectable_case_count": detected,
            "detectable_fraction": detectable_fraction,
            "cases": cases,
        },
    )


def _online_known_bad_candidates(base: SE3) -> list[tuple[str, SE3]]:
    amount_m = 0.1
    amount_rad = math.radians(5.0)
    candidates: list[tuple[str, SE3]] = []
    for index, dof in enumerate(("x", "y", "z")):
        translation = [0.0, 0.0, 0.0]
        translation[index] = amount_m
        translation_tuple = (translation[0], translation[1], translation[2])
        candidates.append(
            (dof, SE3(translation_tuple, (0.0, 0.0, 0.0, 1.0)).compose(base))
        )
    for index, dof in enumerate(("roll", "pitch", "yaw")):
        rotation = [0.0, 0.0, 0.0]
        half = amount_rad / 2.0
        rotation[index] = math.sin(half)
        candidates.append(
            (
                dof,
                SE3(
                    (0.0, 0.0, 0.0),
                    (rotation[0], rotation[1], rotation[2], math.cos(half)),
                ).compose(base),
            )
        )
    return candidates


def _rmse_from_values(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _p90_from_values(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(abs(value) for value in values)
    index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return ordered[index]


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


def _temporal_evidence_metrics(
    dual_result: DualTemporalEvidenceResult,
) -> dict[str, MetricResult]:
    """Surface temporal probe/estimate gates as first-class metrics."""

    temporal_result = dual_result.selected
    metrics: dict[str, MetricResult] = {}
    if temporal_result.probe_detection_ratio is not None:
        metrics["online_temporal_probe_detection_ratio"] = MetricResult(
            value=temporal_result.probe_detection_ratio,
            grade=_grade_from_gate(
                temporal_result.probe_gate_status or "inconclusive"
            ),
            reason=temporal_result.probe_gate_reason
            or "time-offset perturbation probe detection ratio",
        )
    if temporal_result.estimate is not None:
        metrics["online_temporal_estimated_offset_s"] = MetricResult(
            value=temporal_result.estimate.estimated_offset_s,
            unit="s",
            grade=_grade_from_gate(
                temporal_result.estimate_gate_status or "inconclusive"
            ),
            reason=temporal_result.estimate_gate_reason
            or "1D holdout-RMSE time-offset estimate",
        )
        metrics["online_temporal_rmse_relative_improvement"] = MetricResult(
            value=temporal_result.estimate.relative_improvement,
            grade="pass",
            reason=(
                f"holdout RMSE improved by {temporal_result.estimate.relative_improvement:.4f} "
                f"at estimated offset {temporal_result.estimate.estimated_offset_s:.4f} s "
                f"({temporal_result.estimate.curve_sample_count} curve samples)"
            ),
        )
    if temporal_result.verdict is not None:
        metrics["online_temporal_gate_verdict"] = MetricResult(
            value={"pass": 1.0, "inconclusive": 0.5, "fail": 0.0}[temporal_result.verdict],
            grade=_grade_from_gate(temporal_result.verdict),
            reason=temporal_result.verdict_reason or "temporal evidence gate verdict",
        )
    if dual_result.separability is not None:
        separability = dual_result.separability
        metrics["online_temporal_separability"] = MetricResult(
            value=_separability_metric_value(separability.verdict),
            grade=_separability_grade(separability.verdict),
            reason=separability.reason,
        )
    return metrics


def _separability_metric_value(verdict: TemporalSeparabilityVerdict) -> float:
    if verdict == "degenerate":
        return 0.0
    if verdict == "separable":
        return 1.0
    return 0.5


def _separability_grade(verdict: TemporalSeparabilityVerdict) -> Grade:
    if verdict == "degenerate":
        return "warn"
    return "pass"


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
        slac_version=result.run.slac_version,
        git_commit=result.run.git_commit,
        created_at=result.run.created_at,
        dataset_type=dataset_type if isinstance(dataset_type, str) else None,
        dataset_path=dataset_path if isinstance(dataset_path, str) else None,
    )
