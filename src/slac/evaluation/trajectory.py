"""Ground-truth-free trajectory evidence for motion-compensated online calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from slac.core.geometry import SE3
from slac.core.report_artifacts import ReportRunInfo
from slac.core.result import MetricResult
from slac.core.trajectory import (
    TrajectoryArtifact,
    TrajectoryFrameSemantics,
    TrajectoryInterpolationHealth,
    TrajectoryOdometrySource,
    TrajectoryPoseSample,
    TrajectoryQualityBlock,
    TrajectorySubsampling,
    trajectory_quality_metric_from_result,
)
from slac.data.livox import LivoxPointRecord
from slac.data.odometry_track import OdometryPoseSample, OdometryTrack
from slac.evaluation.lidar import build_rig_point_to_plane_observations
from slac.graph.lidar_point_to_plane import LidarRigPointToPlaneFactor

TrajectoryGateStatus = Literal["pass", "fail", "inconclusive"]


@dataclass(frozen=True)
class OdometrySourceInfo:
    """Frame and topic metadata captured while loading an odometry track."""

    topic: str
    world_frame_id: str
    child_frame_id: str


@dataclass(frozen=True)
class OnlineSourceScan:
    """One source LiDAR message expressed as world-frame point records."""

    timestamp_ns: int
    records: list[LivoxPointRecord]


CROSS_SEGMENT_SAMPLING_POLICY = "evenly_spaced_time"


@dataclass(frozen=True)
class CrossSegmentScanCollection:
    """Source scans for leave-segment-out drift evaluation over the full track span."""

    first_half_scans: list[OnlineSourceScan]
    second_half_scans: list[OnlineSourceScan]
    first_half_start_timestamp_ns: int
    first_half_end_timestamp_ns: int
    second_half_start_timestamp_ns: int
    second_half_end_timestamp_ns: int
    sampling_policy: str = CROSS_SEGMENT_SAMPLING_POLICY


@dataclass(frozen=True)
class TrajectoryEvidenceOptions:
    """Factor options controlling post-run trajectory evidence."""

    max_speed_mps: float = 3.0
    max_angular_speed_dps: float = 120.0
    max_clamp_fraction: float = 0.05
    max_cross_segment_rmse_m: float = 0.30
    max_pose_samples: int = 500
    max_scans_per_half: int = 40
    max_points_per_half: int = 4000
    min_correspondences: int = 50
    min_pose_samples: int = 10


@dataclass(frozen=True)
class TrajectoryKinematicResult:
    """Kinematic sanity metrics derived from consecutive odometry poses."""

    linear_speed_p95_mps: float | None
    linear_speed_max_mps: float | None
    angular_speed_p95_dps: float | None
    angular_speed_max_dps: float | None
    gate_status: TrajectoryGateStatus
    gate_reason: str


@dataclass(frozen=True)
class TrajectoryInterpolationResult:
    """Interpolation-health metrics from the replayed odometry track."""

    clamp_fraction: float | None
    max_extrapolation_s: float
    gate_status: TrajectoryGateStatus
    gate_reason: str


@dataclass(frozen=True)
class TrajectoryCrossSegmentResult:
    """Leave-segment-out map-consistency drift proxy."""

    rmse_m: float | None
    first_half_scan_count: int
    second_half_scan_count: int
    first_half_point_count: int
    second_half_point_count: int
    first_half_start_timestamp_ns: int | None
    first_half_end_timestamp_ns: int | None
    second_half_start_timestamp_ns: int | None
    second_half_end_timestamp_ns: int | None
    sampling_policy: str | None
    correspondence_count: int
    gate_status: TrajectoryGateStatus
    gate_reason: str


@dataclass(frozen=True)
class TrajectoryEvidenceResult:
    """Aggregated trajectory evidence block for provenance and metrics."""

    options: TrajectoryEvidenceOptions
    kinematic: TrajectoryKinematicResult
    interpolation: TrajectoryInterpolationResult
    cross_segment: TrajectoryCrossSegmentResult
    verdict: TrajectoryGateStatus
    verdict_reason: str


def trajectory_evidence_options_from_factor(options: dict[str, Any]) -> TrajectoryEvidenceOptions:
    """Parse trajectory-evidence options from ``lidar_rig_point_to_plane`` factor options."""

    return TrajectoryEvidenceOptions(
        max_speed_mps=_float_option(
            options, "trajectory_gate_max_speed_mps", default=3.0, minimum=0.0
        ),
        max_angular_speed_dps=_float_option(
            options, "trajectory_gate_max_angular_speed_dps", default=120.0, minimum=0.0
        ),
        max_clamp_fraction=_float_option(
            options, "trajectory_gate_max_clamp_fraction", default=0.05, minimum=0.0
        ),
        max_cross_segment_rmse_m=_float_option(
            options, "trajectory_gate_max_cross_segment_rmse_m", default=0.30, minimum=0.0
        ),
    )


def evaluate_trajectory_evidence(
    *,
    odometry_track: OdometryTrack,
    cross_segment_scans: CrossSegmentScanCollection | None,
    voxel_size_m: float,
    correspondence_gate_m: float,
    variable: str,
    sensor: str,
    options: TrajectoryEvidenceOptions,
    max_odometry_extrapolation_s: float | None = None,
) -> TrajectoryEvidenceResult:
    """Run kinematic, interpolation, and cross-segment trajectory evidence."""

    kinematic = _evaluate_kinematic_gate(odometry_track, options)
    interpolation = _evaluate_interpolation_gate(
        odometry_track,
        options,
        max_odometry_extrapolation_s=max_odometry_extrapolation_s,
    )
    cross_segment = _evaluate_cross_segment_gate(
        cross_segment_scans=cross_segment_scans,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
        variable=variable,
        sensor=sensor,
        options=options,
    )
    verdict, verdict_reason = _combine_verdicts(
        kinematic_gate_status=kinematic.gate_status,
        kinematic_gate_reason=kinematic.gate_reason,
        interpolation_gate_status=interpolation.gate_status,
        interpolation_gate_reason=interpolation.gate_reason,
        cross_segment_gate_status=cross_segment.gate_status,
        cross_segment_gate_reason=cross_segment.gate_reason,
    )
    return TrajectoryEvidenceResult(
        options=options,
        kinematic=kinematic,
        interpolation=interpolation,
        cross_segment=cross_segment,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )


def build_trajectory_artifact(
    *,
    run: ReportRunInfo,
    odometry_track: OdometryTrack,
    odometry_source: OdometrySourceInfo,
    evidence: TrajectoryEvidenceResult,
    metrics: dict[str, MetricResult],
) -> TrajectoryArtifact:
    """Build the schema-validated trajectory artifact from an odometry track."""

    options = evidence.options
    total_count = odometry_track.message_count
    recorded_poses = _subsample_poses_evenly(odometry_track.samples, options.max_pose_samples)
    first_ns = odometry_track.first_timestamp_ns or 0
    last_ns = odometry_track.last_timestamp_ns or first_ns
    time_span_s = max(0.0, (last_ns - first_ns) / 1_000_000_000)
    return TrajectoryArtifact(
        run=run,
        odometry_source=TrajectoryOdometrySource(
            topic=odometry_source.topic,
            message_count=total_count,
            first_timestamp_ns=first_ns,
            last_timestamp_ns=last_ns,
            time_span_s=time_span_s,
        ),
        frame_semantics=TrajectoryFrameSemantics(
            world_frame_id=odometry_source.world_frame_id,
            child_frame_id=odometry_source.child_frame_id,
        ),
        pose_count_total=total_count,
        pose_count_recorded=len(recorded_poses),
        subsampling=TrajectorySubsampling(max_pose_samples=options.max_pose_samples),
        poses=[
            TrajectoryPoseSample(
                timestamp_ns=sample.timestamp_ns,
                translation=list(sample.pose.translation_m),
                rotation_quat_xyzw=list(sample.pose.rotation_quat_xyzw),
            )
            for sample in recorded_poses
        ],
        interpolation_health=TrajectoryInterpolationHealth(
            interpolation_count=odometry_track.interpolation_count,
            clamp_count=odometry_track.clamp_count,
            max_extrapolation_s=odometry_track.max_extrapolation_s,
        ),
        quality=_quality_block_from_evidence(evidence, metrics),
    )


def trajectory_evidence_to_provenance(result: TrajectoryEvidenceResult) -> dict[str, Any]:
    """Serialize a :class:`TrajectoryEvidenceResult` for run provenance."""

    options = result.options
    block: dict[str, Any] = {
        "trajectory_gate_max_speed_mps": options.max_speed_mps,
        "trajectory_gate_max_angular_speed_dps": options.max_angular_speed_dps,
        "trajectory_gate_max_clamp_fraction": options.max_clamp_fraction,
        "trajectory_gate_max_cross_segment_rmse_m": options.max_cross_segment_rmse_m,
        "kinematic_linear_speed_p95_mps": result.kinematic.linear_speed_p95_mps,
        "kinematic_linear_speed_max_mps": result.kinematic.linear_speed_max_mps,
        "kinematic_angular_speed_p95_dps": result.kinematic.angular_speed_p95_dps,
        "kinematic_angular_speed_max_dps": result.kinematic.angular_speed_max_dps,
        "kinematic_gate_status": result.kinematic.gate_status,
        "kinematic_gate_reason": result.kinematic.gate_reason,
        "interpolation_clamp_fraction": result.interpolation.clamp_fraction,
        "interpolation_max_extrapolation_s": result.interpolation.max_extrapolation_s,
        "interpolation_gate_status": result.interpolation.gate_status,
        "interpolation_gate_reason": result.interpolation.gate_reason,
        "cross_segment_rmse_m": result.cross_segment.rmse_m,
        "cross_segment_first_half_scan_count": result.cross_segment.first_half_scan_count,
        "cross_segment_second_half_scan_count": result.cross_segment.second_half_scan_count,
        "cross_segment_first_half_point_count": result.cross_segment.first_half_point_count,
        "cross_segment_second_half_point_count": result.cross_segment.second_half_point_count,
        "cross_segment_first_half_start_timestamp_ns": (
            result.cross_segment.first_half_start_timestamp_ns
        ),
        "cross_segment_first_half_end_timestamp_ns": (
            result.cross_segment.first_half_end_timestamp_ns
        ),
        "cross_segment_second_half_start_timestamp_ns": (
            result.cross_segment.second_half_start_timestamp_ns
        ),
        "cross_segment_second_half_end_timestamp_ns": (
            result.cross_segment.second_half_end_timestamp_ns
        ),
        "cross_segment_sampling_policy": result.cross_segment.sampling_policy,
        "cross_segment_correspondence_count": result.cross_segment.correspondence_count,
        "cross_segment_gate_status": result.cross_segment.gate_status,
        "cross_segment_gate_reason": result.cross_segment.gate_reason,
        "verdict": result.verdict,
        "verdict_reason": result.verdict_reason,
    }
    return block


def _evaluate_kinematic_gate(
    odometry_track: OdometryTrack,
    options: TrajectoryEvidenceOptions,
) -> TrajectoryKinematicResult:
    samples = odometry_track.samples
    if len(samples) < options.min_pose_samples:
        reason = (
            f"only {len(samples)} odometry poses "
            f"(need >= {options.min_pose_samples} for kinematic sanity)"
        )
        return TrajectoryKinematicResult(
            linear_speed_p95_mps=None,
            linear_speed_max_mps=None,
            angular_speed_p95_dps=None,
            angular_speed_max_dps=None,
            gate_status="inconclusive",
            gate_reason=reason,
        )

    linear_speeds: list[float] = []
    angular_speeds: list[float] = []
    for left, right in pairwise(samples):
        dt_s = max((right.timestamp_ns - left.timestamp_ns) / 1_000_000_000, 1.0e-9)
        translation_m = math.dist(left.pose.translation_m, right.pose.translation_m)
        linear_speeds.append(translation_m / dt_s)
        angular_speeds.append(_rotation_speed_dps(left.pose, right.pose, dt_s))

    linear_p95 = _percentile(linear_speeds, 95.0)
    linear_max = max(linear_speeds)
    angular_p95 = _percentile(angular_speeds, 95.0)
    angular_max = max(angular_speeds)

    violations: list[str] = []
    if linear_max > options.max_speed_mps:
        violations.append(
            f"max linear speed {linear_max:.3f} m/s exceeds "
            f"{options.max_speed_mps:.3f} m/s gate"
        )
    if angular_max > options.max_angular_speed_dps:
        violations.append(
            f"max angular speed {angular_max:.3f} deg/s exceeds "
            f"{options.max_angular_speed_dps:.3f} deg/s gate"
        )
    if violations:
        return TrajectoryKinematicResult(
            linear_speed_p95_mps=linear_p95,
            linear_speed_max_mps=linear_max,
            angular_speed_p95_dps=angular_p95,
            angular_speed_max_dps=angular_max,
            gate_status="fail",
            gate_reason="; ".join(violations),
        )
    return TrajectoryKinematicResult(
        linear_speed_p95_mps=linear_p95,
        linear_speed_max_mps=linear_max,
        angular_speed_p95_dps=angular_p95,
        angular_speed_max_dps=angular_max,
        gate_status="pass",
        gate_reason=(
            f"p95 linear {linear_p95:.3f} m/s, max {linear_max:.3f} m/s; "
            f"p95 angular {angular_p95:.3f} deg/s, max {angular_max:.3f} deg/s "
            f"within gates"
        ),
    )


def _evaluate_interpolation_gate(
    odometry_track: OdometryTrack,
    options: TrajectoryEvidenceOptions,
    *,
    max_odometry_extrapolation_s: float | None,
) -> TrajectoryInterpolationResult:
    interpolation_count = odometry_track.interpolation_count
    clamp_count = odometry_track.clamp_count
    max_extrapolation_s = odometry_track.max_extrapolation_s
    clamp_fraction = (
        clamp_count / interpolation_count if interpolation_count > 0 else None
    )

    violations: list[str] = []
    if clamp_fraction is not None and clamp_fraction > options.max_clamp_fraction:
        violations.append(
            f"clamp fraction {clamp_fraction:.4f} exceeds "
            f"{options.max_clamp_fraction:.4f} gate"
        )
    if (
        max_odometry_extrapolation_s is not None
        and max_extrapolation_s > max_odometry_extrapolation_s
    ):
        violations.append(
            f"max extrapolation {max_extrapolation_s:.4f} s exceeds "
            f"{max_odometry_extrapolation_s:.4f} s tolerance"
        )

    if violations:
        return TrajectoryInterpolationResult(
            clamp_fraction=clamp_fraction,
            max_extrapolation_s=max_extrapolation_s,
            gate_status="fail",
            gate_reason="; ".join(violations),
        )
    return TrajectoryInterpolationResult(
        clamp_fraction=clamp_fraction,
        max_extrapolation_s=max_extrapolation_s,
        gate_status="pass",
        gate_reason=(
            f"clamp fraction {clamp_fraction:.4f} <= {options.max_clamp_fraction:.4f}; "
            f"max extrapolation {max_extrapolation_s:.4f} s within tolerance"
            if clamp_fraction is not None
            else f"max extrapolation {max_extrapolation_s:.4f} s within tolerance"
        ),
    )


def _evaluate_cross_segment_gate(
    *,
    cross_segment_scans: CrossSegmentScanCollection | None,
    voxel_size_m: float,
    correspondence_gate_m: float,
    variable: str,
    sensor: str,
    options: TrajectoryEvidenceOptions,
) -> TrajectoryCrossSegmentResult:
    empty = TrajectoryCrossSegmentResult(
        rmse_m=None,
        first_half_scan_count=0,
        second_half_scan_count=0,
        first_half_point_count=0,
        second_half_point_count=0,
        first_half_start_timestamp_ns=None,
        first_half_end_timestamp_ns=None,
        second_half_start_timestamp_ns=None,
        second_half_end_timestamp_ns=None,
        sampling_policy=None,
        correspondence_count=0,
        gate_status="inconclusive",
        gate_reason="no cross-segment source scans collected",
    )
    if cross_segment_scans is None:
        return empty

    first_scans = cross_segment_scans.first_half_scans
    second_scans = cross_segment_scans.second_half_scans
    if len(first_scans) < 1 or len(second_scans) < 1:
        return TrajectoryCrossSegmentResult(
            rmse_m=None,
            first_half_scan_count=len(first_scans),
            second_half_scan_count=len(second_scans),
            first_half_point_count=0,
            second_half_point_count=0,
            first_half_start_timestamp_ns=cross_segment_scans.first_half_start_timestamp_ns,
            first_half_end_timestamp_ns=cross_segment_scans.first_half_end_timestamp_ns,
            second_half_start_timestamp_ns=cross_segment_scans.second_half_start_timestamp_ns,
            second_half_end_timestamp_ns=cross_segment_scans.second_half_end_timestamp_ns,
            sampling_policy=cross_segment_scans.sampling_policy,
            correspondence_count=0,
            gate_status="inconclusive",
            gate_reason="fewer than 2 cross-segment source scans across both halves",
        )

    first_records = _concat_scan_records(first_scans, options.max_points_per_half)
    second_records = _concat_scan_records(second_scans, options.max_points_per_half)
    second_points = [
        (record.point[0], record.point[1], record.point[2]) for record in second_records
    ]

    observations = build_rig_point_to_plane_observations(
        source_records=first_records,
        target_points=second_points,
        initial_t_source_target=SE3.identity(),
        target_t_world_source=None,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
    )
    correspondence_count = len(observations)
    if correspondence_count < options.min_correspondences:
        return TrajectoryCrossSegmentResult(
            rmse_m=None,
            first_half_scan_count=len(first_scans),
            second_half_scan_count=len(second_scans),
            first_half_point_count=len(first_records),
            second_half_point_count=len(second_records),
            first_half_start_timestamp_ns=cross_segment_scans.first_half_start_timestamp_ns,
            first_half_end_timestamp_ns=cross_segment_scans.first_half_end_timestamp_ns,
            second_half_start_timestamp_ns=cross_segment_scans.second_half_start_timestamp_ns,
            second_half_end_timestamp_ns=cross_segment_scans.second_half_end_timestamp_ns,
            sampling_policy=cross_segment_scans.sampling_policy,
            correspondence_count=correspondence_count,
            gate_status="inconclusive",
            gate_reason=(
                f"only {correspondence_count} cross-segment correspondences "
                f"(need >= {options.min_correspondences})"
            ),
        )

    factor = LidarRigPointToPlaneFactor(
        variable=variable,
        t_ego_lidar=SE3.identity(),
        observations=observations,
        sensor=sensor,
    )
    evaluation = factor.evaluate()
    rmse_m = evaluation.rmse_m
    if rmse_m is None:
        return TrajectoryCrossSegmentResult(
            rmse_m=None,
            first_half_scan_count=len(first_scans),
            second_half_scan_count=len(second_scans),
            first_half_point_count=len(first_records),
            second_half_point_count=len(second_records),
            first_half_start_timestamp_ns=cross_segment_scans.first_half_start_timestamp_ns,
            first_half_end_timestamp_ns=cross_segment_scans.first_half_end_timestamp_ns,
            second_half_start_timestamp_ns=cross_segment_scans.second_half_start_timestamp_ns,
            second_half_end_timestamp_ns=cross_segment_scans.second_half_end_timestamp_ns,
            sampling_policy=cross_segment_scans.sampling_policy,
            correspondence_count=correspondence_count,
            gate_status="inconclusive",
            gate_reason="cross-segment RMSE could not be computed",
        )
    if rmse_m > options.max_cross_segment_rmse_m:
        return TrajectoryCrossSegmentResult(
            rmse_m=rmse_m,
            first_half_scan_count=len(first_scans),
            second_half_scan_count=len(second_scans),
            first_half_point_count=len(first_records),
            second_half_point_count=len(second_records),
            first_half_start_timestamp_ns=cross_segment_scans.first_half_start_timestamp_ns,
            first_half_end_timestamp_ns=cross_segment_scans.first_half_end_timestamp_ns,
            second_half_start_timestamp_ns=cross_segment_scans.second_half_start_timestamp_ns,
            second_half_end_timestamp_ns=cross_segment_scans.second_half_end_timestamp_ns,
            sampling_policy=cross_segment_scans.sampling_policy,
            correspondence_count=correspondence_count,
            gate_status="fail",
            gate_reason=(
                f"cross-segment RMSE {rmse_m:.4f} m exceeds "
                f"{options.max_cross_segment_rmse_m:.4f} m gate"
            ),
        )
    return TrajectoryCrossSegmentResult(
        rmse_m=rmse_m,
        first_half_scan_count=len(first_scans),
        second_half_scan_count=len(second_scans),
        first_half_point_count=len(first_records),
        second_half_point_count=len(second_records),
        first_half_start_timestamp_ns=cross_segment_scans.first_half_start_timestamp_ns,
        first_half_end_timestamp_ns=cross_segment_scans.first_half_end_timestamp_ns,
        second_half_start_timestamp_ns=cross_segment_scans.second_half_start_timestamp_ns,
        second_half_end_timestamp_ns=cross_segment_scans.second_half_end_timestamp_ns,
        sampling_policy=cross_segment_scans.sampling_policy,
        correspondence_count=correspondence_count,
        gate_status="pass",
        gate_reason=(
            f"cross-segment RMSE {rmse_m:.4f} m <= "
            f"{options.max_cross_segment_rmse_m:.4f} m gate "
            f"({correspondence_count} correspondences)"
        ),
    )


def _quality_block_from_evidence(
    evidence: TrajectoryEvidenceResult,
    metrics: dict[str, MetricResult],
) -> TrajectoryQualityBlock:
    options = evidence.options
    kinematic = evidence.kinematic
    interpolation = evidence.interpolation
    cross_segment = evidence.cross_segment
    return TrajectoryQualityBlock(
        kinematic_linear_speed_p95_mps=trajectory_quality_metric_from_result(
            metrics["trajectory_kinematic_linear_speed_p95_mps"],
            threshold=options.max_speed_mps,
        ),
        kinematic_linear_speed_max_mps=trajectory_quality_metric_from_result(
            metrics["trajectory_kinematic_linear_speed_max_mps"],
            threshold=options.max_speed_mps,
        ),
        kinematic_angular_speed_p95_dps=trajectory_quality_metric_from_result(
            metrics["trajectory_kinematic_angular_speed_p95_dps"],
            threshold=options.max_angular_speed_dps,
        ),
        kinematic_angular_speed_max_dps=trajectory_quality_metric_from_result(
            metrics["trajectory_kinematic_angular_speed_max_dps"],
            threshold=options.max_angular_speed_dps,
        ),
        kinematic_gate_status=kinematic.gate_status,
        kinematic_gate_reason=kinematic.gate_reason,
        interpolation_clamp_fraction=trajectory_quality_metric_from_result(
            metrics["trajectory_interpolation_clamp_fraction"],
            threshold=options.max_clamp_fraction,
        ),
        interpolation_max_extrapolation_s=trajectory_quality_metric_from_result(
            metrics["trajectory_interpolation_max_extrapolation_s"],
        ),
        interpolation_gate_status=interpolation.gate_status,
        interpolation_gate_reason=interpolation.gate_reason,
        cross_segment_rmse_m=trajectory_quality_metric_from_result(
            metrics["trajectory_cross_segment_rmse_m"],
            threshold=options.max_cross_segment_rmse_m,
        ),
        cross_segment_first_half_scan_count=cross_segment.first_half_scan_count,
        cross_segment_second_half_scan_count=cross_segment.second_half_scan_count,
        cross_segment_first_half_point_count=cross_segment.first_half_point_count,
        cross_segment_second_half_point_count=cross_segment.second_half_point_count,
        cross_segment_first_half_start_timestamp_ns=cross_segment.first_half_start_timestamp_ns,
        cross_segment_first_half_end_timestamp_ns=cross_segment.first_half_end_timestamp_ns,
        cross_segment_second_half_start_timestamp_ns=cross_segment.second_half_start_timestamp_ns,
        cross_segment_second_half_end_timestamp_ns=cross_segment.second_half_end_timestamp_ns,
        cross_segment_sampling_policy=cross_segment.sampling_policy,
        cross_segment_correspondence_count=cross_segment.correspondence_count,
        cross_segment_gate_status=cross_segment.gate_status,
        cross_segment_gate_reason=cross_segment.gate_reason,
        verdict=evidence.verdict,
        verdict_reason=evidence.verdict_reason,
    )


def trajectory_evidence_metrics(
    evidence: TrajectoryEvidenceResult,
) -> dict[str, MetricResult]:
    """Surface trajectory evidence gates as first-class graded metrics."""

    kinematic = evidence.kinematic
    interpolation = evidence.interpolation
    cross_segment = evidence.cross_segment
    kinematic_grade = _grade_from_gate(kinematic.gate_status)
    interpolation_grade = _grade_from_gate(interpolation.gate_status)
    cross_segment_grade = _grade_from_gate(cross_segment.gate_status)

    metrics: dict[str, MetricResult] = {
        "trajectory_kinematic_linear_speed_p95_mps": MetricResult(
            value=kinematic.linear_speed_p95_mps,
            unit="m/s",
            grade=kinematic_grade,
            reason=kinematic.gate_reason,
        ),
        "trajectory_kinematic_linear_speed_max_mps": MetricResult(
            value=kinematic.linear_speed_max_mps,
            unit="m/s",
            grade=kinematic_grade,
            reason=kinematic.gate_reason,
        ),
        "trajectory_kinematic_angular_speed_p95_dps": MetricResult(
            value=kinematic.angular_speed_p95_dps,
            unit="deg/s",
            grade=kinematic_grade,
            reason=kinematic.gate_reason,
        ),
        "trajectory_kinematic_angular_speed_max_dps": MetricResult(
            value=kinematic.angular_speed_max_dps,
            unit="deg/s",
            grade=kinematic_grade,
            reason=kinematic.gate_reason,
        ),
        "trajectory_interpolation_clamp_fraction": MetricResult(
            value=interpolation.clamp_fraction,
            grade=interpolation_grade,
            reason=interpolation.gate_reason,
        ),
        "trajectory_interpolation_max_extrapolation_s": MetricResult(
            value=interpolation.max_extrapolation_s,
            unit="s",
            grade=interpolation_grade,
            reason=interpolation.gate_reason,
        ),
        "trajectory_cross_segment_rmse_m": MetricResult(
            value=cross_segment.rmse_m,
            unit="m",
            grade=cross_segment_grade,
            reason=cross_segment.gate_reason,
        ),
        "trajectory_gate_verdict": MetricResult(
            value={"pass": 1.0, "inconclusive": 0.5, "fail": 0.0}[evidence.verdict],
            grade=_grade_from_gate(evidence.verdict),
            reason=evidence.verdict_reason,
        ),
    }
    return metrics


def _grade_from_gate(status: TrajectoryGateStatus) -> Literal["pass", "warn", "fail"]:
    if status == "pass":
        return "pass"
    if status == "inconclusive":
        return "warn"
    return "fail"


def _combine_verdicts(
    *,
    kinematic_gate_status: TrajectoryGateStatus,
    kinematic_gate_reason: str,
    interpolation_gate_status: TrajectoryGateStatus,
    interpolation_gate_reason: str,
    cross_segment_gate_status: TrajectoryGateStatus,
    cross_segment_gate_reason: str,
) -> tuple[TrajectoryGateStatus, str]:
    statuses = [
        kinematic_gate_status,
        interpolation_gate_status,
        cross_segment_gate_status,
    ]
    reasons = [
        f"kinematic: {kinematic_gate_reason}",
        f"interpolation: {interpolation_gate_reason}",
        f"cross_segment: {cross_segment_gate_reason}",
    ]
    verdict = _worst_status(statuses)
    return verdict, "; ".join(reasons)


def _worst_status(statuses: list[TrajectoryGateStatus]) -> TrajectoryGateStatus:
    order = {"pass": 0, "inconclusive": 1, "fail": 2}
    return max(statuses, key=lambda status: order[status])


def _subsample_poses_evenly(
    samples: tuple[OdometryPoseSample, ...] | list[OdometryPoseSample],
    max_pose_samples: int,
) -> list[OdometryPoseSample]:
    ordered = list(samples)
    if len(ordered) <= max_pose_samples:
        return ordered
    if max_pose_samples <= 1:
        return [ordered[0]]
    first_ns = ordered[0].timestamp_ns
    last_ns = ordered[-1].timestamp_ns
    if last_ns <= first_ns:
        step = max(1, len(ordered) // max_pose_samples)
        return ordered[::step][:max_pose_samples]
    span_ns = last_ns - first_ns
    selected: list[OdometryPoseSample] = []
    cursor = 0
    for sample_index in range(max_pose_samples):
        target_ns = first_ns + (span_ns * sample_index) // max(max_pose_samples - 1, 1)
        while cursor + 1 < len(ordered) and ordered[cursor + 1].timestamp_ns <= target_ns:
            cursor += 1
        selected.append(ordered[cursor])
    return selected


def _select_evenly_spaced_timestamps(
    timestamps_ns: list[int],
    *,
    window_start_ns: int,
    window_end_ns: int,
    max_samples: int,
) -> list[int]:
    """Return up to ``max_samples`` bag timestamps evenly spaced in time."""

    in_window = sorted(
        timestamp_ns
        for timestamp_ns in timestamps_ns
        if window_start_ns <= timestamp_ns <= window_end_ns
    )
    if len(in_window) <= max_samples:
        return in_window
    if max_samples <= 1:
        return [in_window[0]]
    span_ns = window_end_ns - window_start_ns
    selected: list[int] = []
    cursor = 0
    for sample_index in range(max_samples):
        target_ns = window_start_ns + (span_ns * sample_index) // max(max_samples - 1, 1)
        while cursor + 1 < len(in_window) and in_window[cursor + 1] <= target_ns:
            cursor += 1
        selected.append(in_window[cursor])
    return selected


def _concat_scan_records(
    scans: list[OnlineSourceScan],
    max_points: int,
) -> list[LivoxPointRecord]:
    records: list[LivoxPointRecord] = []
    for scan in scans:
        remaining = max_points - len(records)
        if remaining <= 0:
            break
        if len(scan.records) <= remaining:
            records.extend(scan.records)
        else:
            step = max(1, (len(scan.records) + remaining - 1) // remaining)
            records.extend(scan.records[::step][:remaining])
    return records


def _rotation_speed_dps(left: SE3, right: SE3, dt_s: float) -> float:
    delta = left.inverse().compose(right)
    _x, _y, _z, w = delta.rotation_quat_xyzw
    angle_rad = 2.0 * math.acos(min(1.0, abs(w)))
    return math.degrees(angle_rad / dt_s)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = rank - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _float_option(
    options: dict[str, Any], key: str, *, default: float, minimum: float
) -> float:
    try:
        value = float(options[key])
    except (KeyError, TypeError, ValueError):
        return default
    return max(value, minimum)
