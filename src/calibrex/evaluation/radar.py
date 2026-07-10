"""Radar extrinsic velocity-consistency evaluation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from calibrex.core.config import CalibrationConfig
from calibrex.core.exceptions import DatasetError
from calibrex.core.geometry import (
    SE3,
    Vector3,
    quaternion_conjugate_xyzw,
    rotate_vector_xyzw,
)
from calibrex.core.result import CalibrationResult, Grade, MetricResult
from calibrex.data.base import TimestampedRecord
from calibrex.data.inspect import DatasetInspection
from calibrex.data.nuscenes import (
    NuScenesDataset,
    NuScenesEgoPose,
    find_nuscenes_version_path,
    read_nuscenes_ego_poses,
)
from calibrex.data.nuscenes_radar import NuScenesRadarPCD, read_nuscenes_radar_pcd
from calibrex.evaluation.holdout import split_indices

_MIN_EGO_SPEED_MPS = 0.5
_MIN_INLIER_COUNT = 5
_MIN_RANGE_M = 1.0
_STATIC_DYN_PROP = {1, 3, 5, 7}
_STATIC_VCOMP_THRESHOLD_MPS = 0.5
_MAX_FRAMES_PER_SENSOR = 30
_MANDATORY_YAW_PERTURBATIONS_DEG = (-10.0, -5.0, 5.0, 10.0)
_PERTURBATION_DETECT_MARGIN_MPS = 0.05


@dataclass(frozen=True)
class RadarVelocityConsistencyStats:
    """Aggregate radar velocity-consistency diagnostics."""

    median_abs_residual_mps: float | None
    inlier_rmse_mps: float | None
    return_count: int
    inlier_count: int
    frame_count: int
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "median_abs_residual_mps": self.median_abs_residual_mps,
            "inlier_rmse_mps": self.inlier_rmse_mps,
            "return_count": self.return_count,
            "inlier_count": self.inlier_count,
            "frame_count": self.frame_count,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RadarVelocityObservation:
    """One adapter-independent radar Doppler observation.

    ``line_of_sight_radar`` is expressed in the radar frame. Grouping by
    ``frame_id`` keeps all returns from one acquisition in the same split.
    """

    frame_id: str
    line_of_sight_radar: Vector3
    measured_radial_velocity_mps: float
    ego_velocity_mps: Vector3


@dataclass(frozen=True)
class RadarVelocityResidual:
    """Signed velocity residual for one radar observation."""

    frame_id: str
    measured_radial_velocity_mps: float
    predicted_radial_velocity_mps: float
    residual_mps: float


@dataclass(frozen=True)
class RadarVelocityResidualSplit:
    """Frame-disjoint deterministic train/holdout radar residuals."""

    train: tuple[RadarVelocityResidual, ...]
    holdout: tuple[RadarVelocityResidual, ...]
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    holdout_ratio: float
    seed: int


@dataclass(frozen=True)
class RadarStaticReturnCounts:
    """Adapter-neutral support counts for one radar frame."""

    frame_id: str
    total_return_count: int
    range_eligible_count: int
    static_return_count: int


@dataclass(frozen=True)
class RadarStaticReturnDiagnostics:
    """Aggregate static-return support used by Doppler evidence."""

    frame_count: int
    total_return_count: int
    range_eligible_count: int
    static_return_count: int
    static_fraction: float | None
    status: Literal["supported", "inconclusive"]
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "total_return_count": self.total_return_count,
            "range_eligible_count": self.range_eligible_count,
            "static_return_count": self.static_return_count,
            "static_fraction": self.static_fraction,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RadarYawObservabilityStats:
    """Candidate-independent support for detecting a radar yaw error."""

    frame_count: int
    observation_count: int
    yaw_sensitivity_rms_mps_per_rad: float | None
    yaw_sensitivity_median_abs_mps_per_rad: float | None
    status: Literal["supported", "inconclusive"]
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "los_doppler_yaw_jacobian/v0.1",
            "identifiable_dof": "yaw",
            "translation_observable": False,
            "frame_count": self.frame_count,
            "observation_count": self.observation_count,
            "yaw_sensitivity_rms_mps_per_rad": self.yaw_sensitivity_rms_mps_per_rad,
            "yaw_sensitivity_median_abs_mps_per_rad": (
                self.yaw_sensitivity_median_abs_mps_per_rad
            ),
            "status": self.status,
            "reason": self.reason,
        }


RadarPolicyStatus = Literal["pass", "fail", "inconclusive"]


@dataclass(frozen=True)
class RadarVelocityPolicy:
    """Conservative thresholds for yaw-only Radar-LiDAR evidence."""

    policy_id: str = "slac.falsification.radar_lidar_yaw/v0.1"
    min_holdout_frames: int = 5
    min_holdout_static_returns: int = 50
    min_static_fraction: float = 0.20
    min_yaw_sensitivity_rms_mps_per_rad: float = 0.57
    min_mandatory_detectable_fraction: float = 1.0
    max_holdout_median_abs_residual_mps: float | None = None
    max_holdout_rmse_mps: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "min_holdout_frames": self.min_holdout_frames,
            "min_holdout_static_returns": self.min_holdout_static_returns,
            "min_static_fraction": self.min_static_fraction,
            "min_yaw_sensitivity_rms_mps_per_rad": (
                self.min_yaw_sensitivity_rms_mps_per_rad
            ),
            "min_mandatory_detectable_fraction": (
                self.min_mandatory_detectable_fraction
            ),
            "max_holdout_median_abs_residual_mps": (
                self.max_holdout_median_abs_residual_mps
            ),
            "max_holdout_rmse_mps": self.max_holdout_rmse_mps,
        }


@dataclass(frozen=True)
class RadarPolicyRule:
    """One applied Radar-LiDAR yaw evidence gate."""

    rule_id: str
    status: RadarPolicyStatus
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"rule_id": self.rule_id, "status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class RadarPolicyAssessment:
    """Three-valued assessment of Radar-LiDAR yaw evidence."""

    status: RadarPolicyStatus
    reason: str
    rules: tuple[RadarPolicyRule, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "scope": "yaw_only_translation_unobservable",
            "rules": [rule.as_dict() for rule in self.rules],
        }


def assess_radar_velocity_policy(
    *,
    holdout_summary: Mapping[str, int | float | None],
    holdout_static: RadarStaticReturnDiagnostics,
    observability_by_sensor: Mapping[str, RadarYawObservabilityStats],
    mandatory_required_count: int,
    mandatory_supported_count: int,
    mandatory_detectable_fraction: float | None,
    policy: RadarVelocityPolicy,
) -> RadarPolicyAssessment:
    """Apply fail-closed Radar-LiDAR yaw evidence gates."""

    rules: list[RadarPolicyRule] = []
    holdout_frames = max(
        (diagnostic.frame_count for diagnostic in observability_by_sensor.values()),
        default=0,
    )
    support_ok = (
        holdout_frames >= policy.min_holdout_frames
        and holdout_static.static_return_count >= policy.min_holdout_static_returns
        and holdout_static.static_fraction is not None
        and holdout_static.static_fraction >= policy.min_static_fraction
    )
    rules.append(
        RadarPolicyRule(
            rule_id="holdout_support_gate",
            status="pass" if support_ok else "inconclusive",
            reason=(
                "holdout frame and static-return support satisfy the policy"
                if support_ok
                else "holdout frame count, static returns, or static fraction are insufficient"
            ),
        )
    )
    observability_ok = bool(observability_by_sensor) and all(
        diagnostic.status == "supported"
        and diagnostic.yaw_sensitivity_rms_mps_per_rad is not None
        and diagnostic.yaw_sensitivity_rms_mps_per_rad
        >= policy.min_yaw_sensitivity_rms_mps_per_rad
        for diagnostic in observability_by_sensor.values()
    )
    rules.append(
        RadarPolicyRule(
            rule_id="yaw_observability_gate",
            status="pass" if observability_ok else "inconclusive",
            reason=(
                "all radar sensors have sufficient holdout yaw sensitivity"
                if observability_ok
                else "one or more radar sensors lack holdout yaw sensitivity"
            ),
        )
    )
    controls_ok = (
        mandatory_required_count > 0
        and mandatory_supported_count == mandatory_required_count
        and mandatory_detectable_fraction is not None
        and mandatory_detectable_fraction
        >= policy.min_mandatory_detectable_fraction
    )
    rules.append(
        RadarPolicyRule(
            rule_id="known_bad_controls",
            status="pass" if controls_ok else "inconclusive",
            reason=(
                "all mandatory yaw controls are detectable on holdout support"
                if controls_ok
                else "mandatory yaw controls do not establish falsification power"
            ),
        )
    )
    median = holdout_summary.get("median_abs_residual_mps")
    rmse = holdout_summary.get("rmse_mps")
    median_value = float(median) if isinstance(median, (float, int)) else None
    rmse_value = float(rmse) if isinstance(rmse, (float, int)) else None
    median_limit = policy.max_holdout_median_abs_residual_mps
    rmse_limit = policy.max_holdout_rmse_mps
    thresholds_declared = (
        median_limit is not None and rmse_limit is not None
    )
    residuals_available = median_value is not None and rmse_value is not None
    if not thresholds_declared or not residuals_available:
        residual_status: RadarPolicyStatus = "inconclusive"
        residual_reason = "holdout residual thresholds or measurements are unavailable"
    elif (
        median_value is not None
        and rmse_value is not None
        and median_limit is not None
        and rmse_limit is not None
        and (median_value > median_limit or rmse_value > rmse_limit)
    ):
        residual_status = "fail"
        residual_reason = "candidate holdout residual exceeds the declared policy threshold"
    else:
        residual_status = "pass"
        residual_reason = "candidate holdout residual satisfies the declared policy threshold"
    rules.append(
        RadarPolicyRule(
            rule_id="holdout_velocity_consistency",
            status=residual_status,
            reason=residual_reason,
        )
    )
    prerequisites = rules[:3]
    if any(rule.status != "pass" for rule in prerequisites):
        status: RadarPolicyStatus = "inconclusive"
        reason = next(rule.reason for rule in prerequisites if rule.status != "pass")
    else:
        status = residual_status
        reason = residual_reason
    return RadarPolicyAssessment(status=status, reason=reason, rules=tuple(rules))


def summarize_radar_static_returns(
    frames: Iterable[RadarStaticReturnCounts],
) -> RadarStaticReturnDiagnostics:
    """Summarize static support without depending on a dataset adapter."""

    frame_list = list(frames)
    total = sum(frame.total_return_count for frame in frame_list)
    range_eligible = sum(frame.range_eligible_count for frame in frame_list)
    static = sum(frame.static_return_count for frame in frame_list)
    fraction = static / range_eligible if range_eligible else None
    supported = static >= _MIN_INLIER_COUNT and fraction is not None
    return RadarStaticReturnDiagnostics(
        frame_count=len(frame_list),
        total_return_count=total,
        range_eligible_count=range_eligible,
        static_return_count=static,
        static_fraction=fraction,
        status="supported" if supported else "inconclusive",
        reason=None if supported else "insufficient static radar returns",
    )


def summarize_radar_yaw_observability(
    observations: Iterable[RadarVelocityObservation],
    *,
    t_ego_radar: SE3,
    frame_ids: Iterable[str] | None = None,
    min_frame_count: int = 1,
    min_observation_count: int = _MIN_INLIER_COUNT,
    min_yaw_sensitivity_rms_mps_per_rad: float = 0.1,
) -> RadarYawObservabilityStats:
    """Summarize LOS Doppler sensitivity to a left-composed yaw perturbation."""

    selected_frames = set(frame_ids) if frame_ids is not None else None
    sensitivities_by_frame: dict[str, list[float]] = {}
    inv_rotation = quaternion_conjugate_xyzw(t_ego_radar.rotation_quat_xyzw)
    for observation in observations:
        if selected_frames is not None and observation.frame_id not in selected_frames:
            continue
        unit_los = _unit_vector(*observation.line_of_sight_radar)
        vx, vy, _vz = observation.ego_velocity_mps
        yaw_velocity = rotate_vector_xyzw(inv_rotation, (-vy, vx, 0.0))
        sensitivity = abs(_dot(unit_los, yaw_velocity))
        sensitivities_by_frame.setdefault(observation.frame_id, []).append(sensitivity)
    values = [value for frame in sensitivities_by_frame.values() for value in frame]
    frame_count = len(sensitivities_by_frame)
    if frame_count < min_frame_count or len(values) < min_observation_count:
        return RadarYawObservabilityStats(
            frame_count=frame_count,
            observation_count=len(values),
            yaw_sensitivity_rms_mps_per_rad=None,
            yaw_sensitivity_median_abs_mps_per_rad=None,
            status="inconclusive",
            reason="insufficient holdout frames or radar returns for yaw observability",
        )
    frame_mean_squares = [
        sum(value * value for value in frame_values) / len(frame_values)
        for frame_values in sensitivities_by_frame.values()
    ]
    rms = math.sqrt(sum(frame_mean_squares) / len(frame_mean_squares))
    supported = rms >= min_yaw_sensitivity_rms_mps_per_rad
    return RadarYawObservabilityStats(
        frame_count=frame_count,
        observation_count=len(values),
        yaw_sensitivity_rms_mps_per_rad=rms,
        yaw_sensitivity_median_abs_mps_per_rad=_median(values),
        status="supported" if supported else "inconclusive",
        reason=None if supported else "insufficient yaw sensitivity",
    )


def compute_radar_velocity_residuals(
    observations: Iterable[RadarVelocityObservation],
    *,
    t_ego_radar: SE3,
) -> tuple[RadarVelocityResidual, ...]:
    """Compute Doppler residuals without depending on a dataset or ROS adapter."""

    inv_rotation = quaternion_conjugate_xyzw(t_ego_radar.rotation_quat_xyzw)
    residuals: list[RadarVelocityResidual] = []
    for observation in observations:
        line_of_sight = observation.line_of_sight_radar
        norm = _vector_norm(line_of_sight)
        if norm < 1e-9:
            raise ValueError("radar line of sight must be non-zero")
        unit_line_of_sight = (
            line_of_sight[0] / norm,
            line_of_sight[1] / norm,
            line_of_sight[2] / norm,
        )
        ego_velocity_radar = rotate_vector_xyzw(
            inv_rotation,
            observation.ego_velocity_mps,
        )
        predicted = -_dot(ego_velocity_radar, unit_line_of_sight)
        residuals.append(
            RadarVelocityResidual(
                frame_id=observation.frame_id,
                measured_radial_velocity_mps=observation.measured_radial_velocity_mps,
                predicted_radial_velocity_mps=predicted,
                residual_mps=observation.measured_radial_velocity_mps - predicted,
            )
        )
    return tuple(residuals)


def split_radar_velocity_residuals(
    residuals: Iterable[RadarVelocityResidual],
    *,
    holdout_ratio: float = 0.2,
    seed: int = 0,
) -> RadarVelocityResidualSplit:
    """Split residuals deterministically while keeping radar frames disjoint."""

    residual_list = list(residuals)
    frame_ids = tuple(sorted({residual.frame_id for residual in residual_list}))
    train_indices, holdout_indices = split_indices(
        len(frame_ids),
        holdout_ratio,
        seed=seed,
    )
    train_frame_ids = tuple(frame_ids[index] for index in train_indices)
    holdout_frame_ids = tuple(frame_ids[index] for index in holdout_indices)
    train_frames = set(train_frame_ids)
    holdout_frames = set(holdout_frame_ids)
    return RadarVelocityResidualSplit(
        train=tuple(
            residual for residual in residual_list if residual.frame_id in train_frames
        ),
        holdout=tuple(
            residual for residual in residual_list if residual.frame_id in holdout_frames
        ),
        train_frame_ids=train_frame_ids,
        holdout_frame_ids=holdout_frame_ids,
        holdout_ratio=holdout_ratio,
        seed=seed,
    )


def radar_velocity_metrics_from_result(
    config: CalibrationConfig,
    result: CalibrationResult,
    inspection: DatasetInspection,
) -> dict[str, MetricResult]:
    """Score radar radial velocity consistency for nuScenes radar extrinsics."""

    _ = inspection

    if config.dataset.type != "nuscenes":
        return {}
    if "radar_lidar_velocity_consistency" not in config.evaluation.metrics:
        return {}
    radar_sensors = _configured_radar_sensors(config)
    if not radar_sensors:
        return {}

    radar_config = config.evaluation.radar
    seed_value = result.run.provenance.get("seed")
    split_seed = seed_value if isinstance(seed_value, int) else 0
    policy = RadarVelocityPolicy(
        min_holdout_frames=radar_config.min_holdout_frames,
        min_holdout_static_returns=radar_config.min_holdout_static_returns,
        min_static_fraction=radar_config.min_static_fraction,
        min_yaw_sensitivity_rms_mps_per_rad=(
            radar_config.min_yaw_sensitivity_rms_mps_per_rad
        ),
        min_mandatory_detectable_fraction=(
            radar_config.min_mandatory_detectable_fraction
        ),
        max_holdout_median_abs_residual_mps=(
            radar_config.max_holdout_median_abs_residual_mps
        ),
        max_holdout_rmse_mps=radar_config.max_holdout_rmse_mps,
    )
    stats, provenance = summarize_nuscenes_radar_velocity_consistency(
        dataset_path=config.dataset.path,
        radar_sensors=radar_sensors,
        extrinsics_lookup=lambda sensor_name: _resolve_radar_extrinsic(
            result,
            sensor_name,
            use_dataset_reference=radar_config.use_dataset_reference,
        ),
        max_frames_per_sensor=radar_config.max_frames_per_sensor,
        holdout_ratio=config.evaluation.holdout_ratio,
        split_seed=split_seed,
        policy=policy,
    )
    if provenance:
        provenance["candidate_source"] = (
            "dataset_reference"
            if radar_config.use_dataset_reference
            else "configured_or_optimized_candidate"
        )
        result.run.provenance["radar_velocity_consistency"] = provenance

    if stats.status == "scored" and stats.median_abs_residual_mps is not None:
        assessment_payload = provenance.get("assessment")
        assessment_status = (
            assessment_payload.get("status")
            if isinstance(assessment_payload, Mapping)
            else "inconclusive"
        )
        grade_by_status: dict[str, Grade] = {
            "pass": "pass",
            "fail": "fail",
            "inconclusive": "warn",
        }
        grade = grade_by_status.get(str(assessment_status), "warn")
        residual_summary = provenance.get("residual_summary")
        train = _summary_float(residual_summary, "train", "median_abs_residual_mps")
        holdout = _summary_float(
            residual_summary,
            "holdout",
            "median_abs_residual_mps",
        )
        return {
            "radar_lidar_velocity_consistency": MetricResult(
                train=train,
                holdout=holdout,
                value=stats.median_abs_residual_mps,
                grade=grade,
                unit="m/s",
                reason=(
                    f"{assessment_status}: median absolute static-target radial velocity "
                    f"residual over {stats.inlier_count} inliers in "
                    f"{stats.frame_count} radar frame(s)"
                ),
            )
        }
    return {
        "radar_lidar_velocity_consistency": MetricResult(
            value=None,
            unit="m/s",
            grade="warn",
            reason=stats.reason or "radar velocity consistency could not be scored",
        )
    }


def summarize_nuscenes_radar_velocity_consistency(
    *,
    dataset_path: str | Path,
    radar_sensors: Mapping[str, str],
    extrinsics_lookup: Callable[[str], SE3 | None],
    max_frames_per_sensor: int = _MAX_FRAMES_PER_SENSOR,
    holdout_ratio: float = 0.2,
    split_seed: int = 0,
    policy: RadarVelocityPolicy | None = None,
) -> tuple[RadarVelocityConsistencyStats, dict[str, object]]:
    """Compute radar velocity consistency and optional yaw perturbation probe."""

    input_diagnostics = _RadarInputDiagnostics(sensor_total=len(radar_sensors))
    version_path = find_nuscenes_version_path(dataset_path)
    if version_path is None:
        input_diagnostics.record(kind="failure", code="metadata_not_found")
        return _unavailable_stats("nuScenes metadata tables were not found"), {
            "input_diagnostics": input_diagnostics.as_dict()
        }

    dataset = NuScenesDataset(dataset_path)
    ego_poses = read_nuscenes_ego_poses(dataset_path)
    if not ego_poses:
        input_diagnostics.record(kind="failure", code="empty_ego_pose_table")
        return _unavailable_stats("nuScenes ego_pose table is empty"), {
            "input_diagnostics": input_diagnostics.as_dict()
        }

    residuals: list[float] = []
    frame_count = 0
    return_count = 0
    inlier_count = 0
    scored_sensor = False
    baseline_value: float | None = None
    observations_by_sensor: dict[str, list[RadarVelocityObservation]] = {}
    extrinsics_by_sensor: dict[str, SE3] = {}
    static_counts: list[RadarStaticReturnCounts] = []
    static_counts_by_sensor: dict[str, list[RadarStaticReturnCounts]] = {
        sensor_name: [] for sensor_name in radar_sensors
    }

    for sensor_name, channel in sorted(radar_sensors.items()):
        observations_by_sensor[sensor_name] = []
        extrinsic = extrinsics_lookup(sensor_name)
        if extrinsic is None:
            input_diagnostics.record(
                kind="skip",
                code="missing_extrinsic",
                sensor=sensor_name,
                channel=channel,
            )
            continue
        extrinsics_by_sensor[sensor_name] = extrinsic
        records = sorted(dataset.records(channel), key=lambda record: record.timestamp_ns)
        if len(records) < 2:
            input_diagnostics.record(
                kind="skip",
                code="insufficient_records",
                sensor=sensor_name,
                channel=channel,
            )
            continue
        input_diagnostics.sensor_eligible += 1
        scored_sensor = True
        sampled = _sample_records(records, max_frames_per_sensor)
        for index in range(1, len(sampled)):
            input_diagnostics.attempted_transition_count += 1
            current = sampled[index]
            previous = sampled[index - 1]
            pose_current = ego_poses.get(str(current.metadata.get("ego_pose_token")))
            pose_previous = ego_poses.get(str(previous.metadata.get("ego_pose_token")))
            if pose_current is None or pose_previous is None:
                input_diagnostics.record(
                    kind="skip",
                    code="missing_ego_pose",
                    sensor=sensor_name,
                    channel=channel,
                    frame_id=f"{sensor_name}:{current.timestamp_ns}",
                )
                continue
            ego_velocity = _ego_velocity_mps(pose_previous, pose_current)
            if _vector_norm(ego_velocity) < _MIN_EGO_SPEED_MPS:
                input_diagnostics.record(
                    kind="exclusion",
                    code="insufficient_ego_motion",
                    sensor=sensor_name,
                    channel=channel,
                    frame_id=f"{sensor_name}:{current.timestamp_ns}",
                )
                continue
            payload_path = current.payload_path
            if payload_path is None:
                input_diagnostics.record(
                    kind="skip",
                    code="missing_payload_reference",
                    sensor=sensor_name,
                    channel=channel,
                    frame_id=f"{sensor_name}:{current.timestamp_ns}",
                )
                continue
            payload_ref = _safe_dataset_relative_path(dataset_path, payload_path)
            if not Path(payload_path).exists():
                input_diagnostics.record(
                    kind="failure",
                    code="payload_not_found",
                    sensor=sensor_name,
                    channel=channel,
                    frame_id=f"{sensor_name}:{current.timestamp_ns}",
                    payload_ref=payload_ref,
                )
                continue
            try:
                pcd = read_nuscenes_radar_pcd(payload_path)
            except (OSError, DatasetError, ValueError, KeyError, IndexError) as exc:
                input_diagnostics.record(
                    kind="failure",
                    code="payload_read_error",
                    sensor=sensor_name,
                    channel=channel,
                    frame_id=f"{sensor_name}:{current.timestamp_ns}",
                    payload_ref=payload_ref,
                    error_type=type(exc).__name__,
                )
                continue
            frame_id = f"{sensor_name}:{current.timestamp_ns}"
            frame_static_counts = _radar_static_return_counts_from_pcd(
                pcd=pcd,
                frame_id=frame_id,
            )
            static_counts.append(frame_static_counts)
            static_counts_by_sensor[sensor_name].append(frame_static_counts)
            return_count += pcd.point_count
            frame_observations = _radar_velocity_observations_from_pcd(
                pcd=pcd,
                ego_velocity_mps=ego_velocity,
                frame_id=frame_id,
            )
            if not frame_observations:
                input_diagnostics.record(
                    kind="exclusion",
                    code="no_eligible_static_returns",
                    sensor=sensor_name,
                    channel=channel,
                    frame_id=frame_id,
                    payload_ref=payload_ref,
                )
                continue
            input_diagnostics.accepted_frame_count += 1
            frame_count += 1
            inlier_count += len(frame_observations)
            observations_by_sensor[sensor_name].extend(frame_observations)

    baseline_residuals: list[RadarVelocityResidual] = []
    for sensor_name, observations in observations_by_sensor.items():
        if sensor_name not in extrinsics_by_sensor:
            continue
        baseline_residuals.extend(
            compute_radar_velocity_residuals(
                observations,
                t_ego_radar=extrinsics_by_sensor[sensor_name],
            )
        )
    residuals = [abs(residual.residual_mps) for residual in baseline_residuals]

    if not scored_sensor:
        return _unavailable_stats("no radar extrinsics or sample_data were available"), {
            "input_diagnostics": input_diagnostics.as_dict()
        }

    support_provenance: dict[str, object] = {
        "status": "inconclusive",
        "input_diagnostics": input_diagnostics.as_dict(),
        "static_return_diagnostics": summarize_radar_static_returns(
            static_counts
        ).as_dict(),
    }
    if inlier_count < _MIN_INLIER_COUNT or not residuals:
        return (
            RadarVelocityConsistencyStats(
                median_abs_residual_mps=None,
                inlier_rmse_mps=None,
                return_count=return_count,
                inlier_count=inlier_count,
                frame_count=frame_count,
                status="inconclusive",
                reason=(
                    "insufficient static radar returns or ego motion for velocity consistency"
                ),
            ),
            support_provenance,
        )

    median_abs = _median(residuals)
    inlier_rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    baseline_value = median_abs
    provenance: dict[str, object] = {
        "status": "scored",
        "frame_count": frame_count,
        "return_count": return_count,
        "inlier_count": inlier_count,
        "median_abs_residual_mps": median_abs,
        "inlier_rmse_mps": inlier_rmse,
        "input_diagnostics": input_diagnostics.as_dict(),
    }
    residual_split = split_radar_velocity_residuals(
        baseline_residuals,
        holdout_ratio=holdout_ratio,
        seed=split_seed,
    )
    provenance["split"] = {
        "policy": "seeded_frame_holdout/v0.1",
        "holdout_ratio": holdout_ratio,
        "seed": split_seed,
        "train_frame_ids": list(residual_split.train_frame_ids),
        "holdout_frame_ids": list(residual_split.holdout_frame_ids),
    }
    provenance["residual_summary"] = {
        "train": _residual_summary(residual_split.train),
        "holdout": _residual_summary(residual_split.holdout),
    }
    provenance["static_return_diagnostics"] = support_provenance[
        "static_return_diagnostics"
    ]
    train_frame_ids = set(residual_split.train_frame_ids)
    holdout_frame_ids = set(residual_split.holdout_frame_ids)
    train_static = summarize_radar_static_returns(
        frame for frame in static_counts if frame.frame_id in train_frame_ids
    )
    holdout_static = summarize_radar_static_returns(
        frame for frame in static_counts if frame.frame_id in holdout_frame_ids
    )
    provenance["static_return_summary"] = {
        "train": train_static.as_dict(),
        "holdout": holdout_static.as_dict(),
    }
    observability_by_sensor = {
        sensor_name: summarize_radar_yaw_observability(
            observations,
            t_ego_radar=extrinsics_by_sensor[sensor_name],
            frame_ids=residual_split.holdout_frame_ids,
        )
        for sensor_name, observations in observations_by_sensor.items()
        if sensor_name in extrinsics_by_sensor
    }
    provenance["holdout_yaw_observability"] = {
        sensor_name: diagnostic.as_dict()
        for sensor_name, diagnostic in observability_by_sensor.items()
    }

    perturbations = [
        probe
        for amount_deg in _MANDATORY_YAW_PERTURBATIONS_DEG
        if (
            probe := _yaw_perturbation_probe(
                observations_by_sensor=observations_by_sensor,
                extrinsics_by_sensor=extrinsics_by_sensor,
                baseline_residuals=baseline_residuals,
                residual_split=residual_split,
                baseline_median_abs_mps=baseline_value,
                amount_deg=amount_deg,
            )
        )
        is not None
    ]
    challenge_probes = [
        {
            **probe,
            "supported": probe["comparison_support"] == "holdout",
            "diagnostic_detectable": probe["detectable"],
            "detectable": (
                probe["detectable"]
                if probe["comparison_support"] == "holdout"
                else None
            ),
        }
        for probe in perturbations
    ]
    supported_count = sum(bool(probe["supported"]) for probe in challenge_probes)
    detectable_count = sum(probe["detectable"] is True for probe in challenge_probes)
    required_count = len(_MANDATORY_YAW_PERTURBATIONS_DEG)
    provenance["known_bad_rotation_probes"] = {
        "challenge_id": "radar_lidar_mandatory_yaw_controls/v0.1",
        "mandatory_amounts_deg": list(_MANDATORY_YAW_PERTURBATIONS_DEG),
        "detect_margin_mps": _PERTURBATION_DETECT_MARGIN_MPS,
        "required_count": required_count,
        "scored_count": len(perturbations),
        "supported_count": supported_count,
        "detectable_count": detectable_count,
        "detectable_fraction": (
            detectable_count / required_count
            if supported_count == required_count
            else None
        ),
        "probes": challenge_probes,
    }
    legacy_positive_five = next(
        (probe for probe in perturbations if probe["amount_deg"] == 5.0),
        None,
    )
    if legacy_positive_five is not None:
        provenance["perturbation_probe"] = legacy_positive_five
    active_policy = policy or RadarVelocityPolicy()
    assessment = assess_radar_velocity_policy(
        holdout_summary=_residual_summary(residual_split.holdout),
        holdout_static=holdout_static,
        observability_by_sensor=observability_by_sensor,
        mandatory_required_count=required_count,
        mandatory_supported_count=supported_count,
        mandatory_detectable_fraction=(
            detectable_count / required_count
            if supported_count == required_count
            else None
        ),
        policy=active_policy,
    )
    provenance["policy"] = active_policy.as_dict()
    provenance["pooled_assessment"] = assessment.as_dict()
    sensor_evaluations: dict[str, dict[str, object]] = {}
    for sensor_name in sorted(radar_sensors):
        extrinsic = extrinsics_by_sensor.get(sensor_name)
        observations = observations_by_sensor.get(sensor_name, [])
        if extrinsic is None or not observations:
            sensor_evaluations[sensor_name] = _unavailable_radar_sensor_evaluation(
                reason=(
                    "candidate extrinsic unavailable"
                    if extrinsic is None
                    else "no eligible radar observations"
                ),
                policy=active_policy,
            )
            continue
        sensor_evaluations[sensor_name] = _evaluate_radar_sensor(
            sensor_name=sensor_name,
            observations=observations,
            static_counts=static_counts_by_sensor[sensor_name],
            extrinsic=extrinsic,
            holdout_ratio=holdout_ratio,
            split_seed=split_seed,
            policy=active_policy,
        )
    provenance["sensor_evaluations"] = sensor_evaluations
    provenance["assessment"] = aggregate_radar_sensor_assessments(sensor_evaluations)

    return (
        RadarVelocityConsistencyStats(
            median_abs_residual_mps=median_abs,
            inlier_rmse_mps=inlier_rmse,
            return_count=return_count,
            inlier_count=inlier_count,
            frame_count=frame_count,
            status="scored",
        ),
        provenance,
    )


@dataclass(frozen=True)
class _FrameResidualStats:
    return_count: int
    inlier_count: int
    abs_residuals_mps: list[float]


@dataclass
class _RadarInputDiagnostics:
    sensor_total: int
    sensor_eligible: int = 0
    attempted_transition_count: int = 0
    accepted_frame_count: int = 0
    counts_by_code: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    truncated_count: int = 0

    def record(
        self,
        *,
        kind: Literal["skip", "failure", "exclusion"],
        code: str,
        sensor: str | None = None,
        channel: str | None = None,
        frame_id: str | None = None,
        payload_ref: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.counts_by_code[code] = self.counts_by_code.get(code, 0) + 1
        if len(self.events) >= 100:
            self.truncated_count += 1
            return
        self.events.append(
            {
                "kind": kind,
                "code": code,
                "sensor": sensor,
                "channel": channel,
                "frame_id": frame_id,
                "payload_ref": payload_ref,
                "error_type": error_type,
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": "slac.nuscenes_radar_input_diagnostics/v0.1",
            "sensor_total": self.sensor_total,
            "sensor_eligible": self.sensor_eligible,
            "attempted_transition_count": self.attempted_transition_count,
            "accepted_frame_count": self.accepted_frame_count,
            "counts_by_code": dict(sorted(self.counts_by_code.items())),
            "events": self.events,
            "truncated_count": self.truncated_count,
        }


def score_radar_velocity_frame(
    *,
    pcd: NuScenesRadarPCD,
    ego_velocity_mps: Vector3,
    t_ego_radar: SE3,
) -> _FrameResidualStats | None:
    """Score one radar frame against ego velocity and a candidate extrinsic."""

    observations = _radar_velocity_observations_from_pcd(
        pcd=pcd,
        ego_velocity_mps=ego_velocity_mps,
        frame_id="frame",
    )
    if not observations:
        return None
    residuals = compute_radar_velocity_residuals(
        observations,
        t_ego_radar=t_ego_radar,
    )
    return _FrameResidualStats(
        return_count=int(pcd.point_count),
        inlier_count=len(residuals),
        abs_residuals_mps=[abs(residual.residual_mps) for residual in residuals],
    )


def _radar_velocity_observations_from_pcd(
    *,
    pcd: NuScenesRadarPCD,
    ego_velocity_mps: Vector3,
    frame_id: str,
) -> tuple[RadarVelocityObservation, ...]:
    if pcd.point_count == 0:
        return ()
    x = pcd.array("x")
    y = pcd.array("y")
    z = pcd.array("z")
    vx = pcd.array("vx")
    vy = pcd.array("vy")
    dyn_prop = pcd.fields.get("dyn_prop")
    vx_comp = pcd.fields.get("vx_comp")
    vy_comp = pcd.fields.get("vy_comp")

    observations: list[RadarVelocityObservation] = []
    return_count = int(pcd.point_count)
    for index in range(return_count):
        px = float(x[index])
        py = float(y[index])
        pz = float(z[index])
        range_m = math.sqrt(px * px + py * py + pz * pz)
        if range_m < _MIN_RANGE_M:
            continue
        if not _is_static_return(
            dyn_prop=dyn_prop,
            index=index,
            vx_comp=vx_comp,
            vy_comp=vy_comp,
        ):
            continue
        measured = _radial_velocity(float(vx[index]), float(vy[index]), px, py, pz)
        observations.append(
            RadarVelocityObservation(
                frame_id=frame_id,
                line_of_sight_radar=(px, py, pz),
                measured_radial_velocity_mps=measured,
                ego_velocity_mps=ego_velocity_mps,
            )
        )
    return tuple(observations)


def _radar_static_return_counts_from_pcd(
    *,
    pcd: NuScenesRadarPCD,
    frame_id: str,
) -> RadarStaticReturnCounts:
    x = pcd.array("x")
    y = pcd.array("y")
    z = pcd.array("z")
    dyn_prop = pcd.fields.get("dyn_prop")
    vx_comp = pcd.fields.get("vx_comp")
    vy_comp = pcd.fields.get("vy_comp")
    range_eligible_count = 0
    static_return_count = 0
    for index in range(int(pcd.point_count)):
        range_m = math.sqrt(
            float(x[index]) ** 2 + float(y[index]) ** 2 + float(z[index]) ** 2
        )
        if range_m < _MIN_RANGE_M:
            continue
        range_eligible_count += 1
        if _is_static_return(
            dyn_prop=dyn_prop,
            index=index,
            vx_comp=vx_comp,
            vy_comp=vy_comp,
        ):
            static_return_count += 1
    return RadarStaticReturnCounts(
        frame_id=frame_id,
        total_return_count=int(pcd.point_count),
        range_eligible_count=range_eligible_count,
        static_return_count=static_return_count,
    )


def _yaw_perturbation_probe(
    *,
    observations_by_sensor: Mapping[str, Sequence[RadarVelocityObservation]],
    extrinsics_by_sensor: Mapping[str, SE3],
    baseline_residuals: Sequence[RadarVelocityResidual],
    residual_split: RadarVelocityResidualSplit,
    baseline_median_abs_mps: float,
    amount_deg: float,
) -> dict[str, object] | None:
    yaw_rad = math.radians(amount_deg)
    half = yaw_rad / 2.0
    perturbation = SE3((0.0, 0.0, 0.0), (0.0, 0.0, math.sin(half), math.cos(half)))
    perturbed_residuals: list[RadarVelocityResidual] = []
    for sensor_name, observations in observations_by_sensor.items():
        baseline_extrinsic = extrinsics_by_sensor[sensor_name]
        perturbed_extrinsic = perturbation.compose(baseline_extrinsic)
        perturbed_residuals.extend(
            compute_radar_velocity_residuals(
                observations,
                t_ego_radar=perturbed_extrinsic,
            )
        )

    comparison_frames = set(residual_split.holdout_frame_ids)
    comparison_support = "holdout"
    if not comparison_frames:
        comparison_frames = {residual.frame_id for residual in baseline_residuals}
        comparison_support = "all_fallback_no_holdout"
    baseline_comparison = [
        abs(residual.residual_mps)
        for residual in baseline_residuals
        if residual.frame_id in comparison_frames
    ]
    perturbed_comparison = [
        abs(residual.residual_mps)
        for residual in perturbed_residuals
        if residual.frame_id in comparison_frames
    ]
    if len(perturbed_comparison) < _MIN_INLIER_COUNT or not baseline_comparison:
        return None
    baseline_comparison_median = _median(baseline_comparison)
    perturbed_median = _median(perturbed_comparison)
    detectable = (
        perturbed_median
        > baseline_comparison_median + _PERTURBATION_DETECT_MARGIN_MPS
    )
    return {
        "dof": "yaw_deg",
        "amount_deg": amount_deg,
        "baseline_median_abs_residual_mps": baseline_median_abs_mps,
        "comparison_baseline_median_abs_residual_mps": baseline_comparison_median,
        "perturbed_median_abs_residual_mps": perturbed_median,
        "detectable": detectable,
        "comparison_support": comparison_support,
        "frame_ids": sorted(comparison_frames),
    }


def _residual_summary(
    residuals: Sequence[RadarVelocityResidual],
) -> dict[str, int | float | None]:
    absolute = [abs(residual.residual_mps) for residual in residuals]
    if not absolute:
        return {
            "count": 0,
            "median_abs_residual_mps": None,
            "rmse_mps": None,
        }
    return {
        "count": len(absolute),
        "median_abs_residual_mps": _median(absolute),
        "rmse_mps": math.sqrt(sum(value * value for value in absolute) / len(absolute)),
    }


def _evaluate_radar_sensor(
    *,
    sensor_name: str,
    observations: Sequence[RadarVelocityObservation],
    static_counts: Sequence[RadarStaticReturnCounts],
    extrinsic: SE3,
    holdout_ratio: float,
    split_seed: int,
    policy: RadarVelocityPolicy,
) -> dict[str, object]:
    baseline = compute_radar_velocity_residuals(observations, t_ego_radar=extrinsic)
    split = split_radar_velocity_residuals(
        baseline,
        holdout_ratio=holdout_ratio,
        seed=split_seed,
    )
    train_frames = set(split.train_frame_ids)
    holdout_frames = set(split.holdout_frame_ids)
    train_static = summarize_radar_static_returns(
        item for item in static_counts if item.frame_id in train_frames
    )
    holdout_static = summarize_radar_static_returns(
        item for item in static_counts if item.frame_id in holdout_frames
    )
    observability = summarize_radar_yaw_observability(
        observations,
        t_ego_radar=extrinsic,
        frame_ids=split.holdout_frame_ids,
    )
    raw_probes = [
        probe
        for amount_deg in _MANDATORY_YAW_PERTURBATIONS_DEG
        if (
            probe := _yaw_perturbation_probe(
                observations_by_sensor={sensor_name: observations},
                extrinsics_by_sensor={sensor_name: extrinsic},
                baseline_residuals=baseline,
                residual_split=split,
                baseline_median_abs_mps=_median(
                    abs(item.residual_mps) for item in baseline
                ),
                amount_deg=amount_deg,
            )
        )
        is not None
    ]
    probes = [
        {
            **probe,
            "supported": probe["comparison_support"] == "holdout",
            "diagnostic_detectable": probe["detectable"],
            "detectable": (
                probe["detectable"]
                if probe["comparison_support"] == "holdout"
                else None
            ),
        }
        for probe in raw_probes
    ]
    required_count = len(_MANDATORY_YAW_PERTURBATIONS_DEG)
    supported_count = sum(probe["supported"] is True for probe in probes)
    detectable_count = sum(probe["detectable"] is True for probe in probes)
    detectable_fraction = (
        detectable_count / required_count if supported_count == required_count else None
    )
    challenge: dict[str, object] = {
        "challenge_id": "radar_lidar_mandatory_yaw_controls/v0.1",
        "mandatory_amounts_deg": list(_MANDATORY_YAW_PERTURBATIONS_DEG),
        "required_count": required_count,
        "supported_count": supported_count,
        "detectable_count": detectable_count,
        "detectable_fraction": detectable_fraction,
        "probes": probes,
    }
    assessment = assess_radar_velocity_policy(
        holdout_summary=_residual_summary(split.holdout),
        holdout_static=holdout_static,
        observability_by_sensor={sensor_name: observability},
        mandatory_required_count=required_count,
        mandatory_supported_count=supported_count,
        mandatory_detectable_fraction=detectable_fraction,
        policy=policy,
    )
    return {
        "split": {
            "policy": "seeded_frame_holdout/v0.1",
            "holdout_ratio": holdout_ratio,
            "seed": split_seed,
            "train_frame_ids": list(split.train_frame_ids),
            "holdout_frame_ids": list(split.holdout_frame_ids),
        },
        "residual_summary": {
            "train": _residual_summary(split.train),
            "holdout": _residual_summary(split.holdout),
        },
        "static_return_summary": {
            "train": train_static.as_dict(),
            "holdout": holdout_static.as_dict(),
        },
        "holdout_yaw_observability": observability.as_dict(),
        "known_bad_rotation_probes": challenge,
        "assessment": assessment.as_dict(),
    }


def _unavailable_radar_sensor_evaluation(
    *,
    reason: str,
    policy: RadarVelocityPolicy,
) -> dict[str, object]:
    return {
        "split": {
            "policy": "seeded_frame_holdout/v0.1",
            "train_frame_ids": [],
            "holdout_frame_ids": [],
        },
        "policy": policy.as_dict(),
        "assessment": {
            "status": "inconclusive",
            "reason": reason,
            "scope": "yaw_only_translation_unobservable",
            "rules": [],
        },
    }


def aggregate_radar_sensor_assessments(
    sensor_evaluations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    statuses: dict[str, str] = {}
    for sensor_name, evaluation in sorted(sensor_evaluations.items()):
        assessment = evaluation.get("assessment")
        status = assessment.get("status") if isinstance(assessment, Mapping) else None
        statuses[sensor_name] = str(status or "inconclusive")
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    inconclusive = sorted(
        name for name, status in statuses.items() if status != "pass" and status != "fail"
    )
    if failed:
        status = "fail"
        reason = f"Radar yaw evidence rejected sensor candidates: {', '.join(failed)}"
    elif inconclusive or not statuses:
        status = "inconclusive"
        reason = (
            f"Radar yaw evidence is inconclusive for sensors: {', '.join(inconclusive)}"
            if inconclusive
            else "no required Radar sensors were evaluated"
        )
    else:
        status = "pass"
        reason = "all required Radar sensors satisfy the yaw-only policy"
    return {
        "status": status,
        "reason": reason,
        "scope": "yaw_only_translation_unobservable",
        "aggregation": "all_required_pass_any_conclusive_fail",
        "required_sensors": sorted(statuses),
        "sensor_statuses": statuses,
        "failed_sensors": failed,
        "inconclusive_sensors": inconclusive,
        "rules": [
            {
                "rule_id": "sensor_policy_aggregation",
                "status": status,
                "reason": reason,
            }
        ],
    }


def _ego_velocity_mps(previous: NuScenesEgoPose, current: NuScenesEgoPose) -> Vector3:
    dt_s = (current.timestamp_ns - previous.timestamp_ns) / 1_000_000_000.0
    if dt_s <= 0.0:
        return (0.0, 0.0, 0.0)
    v_global = (
        (current.translation_m[0] - previous.translation_m[0]) / dt_s,
        (current.translation_m[1] - previous.translation_m[1]) / dt_s,
        (current.translation_m[2] - previous.translation_m[2]) / dt_s,
    )
    inv_rotation = quaternion_conjugate_xyzw(current.rotation_quat_xyzw)
    return rotate_vector_xyzw(inv_rotation, v_global)


def _resolve_radar_extrinsic(
    result: CalibrationResult,
    sensor_name: str,
    *,
    use_dataset_reference: bool = False,
) -> SE3 | None:
    transform_name = f"T_ego_{sensor_name}"
    mappings = (
        (result.reference_extrinsics,)
        if use_dataset_reference
        else (
            result.transforms,
            result.candidate_extrinsics,
            result.reference_extrinsics,
        )
    )
    for mapping in mappings:
        payload = mapping.get(transform_name)
        if payload is None:
            continue
        if hasattr(payload, "translation_m"):
            return SE3.from_lists(list(payload.translation_m), list(payload.rotation_quat_xyzw))
        if isinstance(payload, Mapping):
            translation = payload.get("translation_m")
            rotation = payload.get("rotation_quat_xyzw")
            if isinstance(translation, list) and isinstance(rotation, list):
                return SE3.from_lists(translation, rotation)
    return None


def _configured_radar_sensors(config: CalibrationConfig) -> dict[str, str]:
    sensors: dict[str, str] = {}
    for name, sensor in config.sensors.items():
        if sensor.type != "radar":
            continue
        topic = sensor.topic or name.upper()
        sensors[name] = topic
    return sensors


def _sample_records(
    records: Sequence[TimestampedRecord],
    max_count: int,
) -> list[TimestampedRecord]:
    if len(records) <= max_count:
        return list(records)
    if max_count < 2:
        return list(records[:max_count])
    step = max(1, (len(records) - 1) // (max_count - 1))
    sampled = [records[index] for index in range(0, len(records), step)]
    if sampled[-1] is not records[-1]:
        sampled.append(records[-1])
    return sampled[:max_count]


def _is_static_return(
    *,
    dyn_prop: object | None,
    index: int,
    vx_comp: object | None,
    vy_comp: object | None,
) -> bool:
    if dyn_prop is not None:
        try:
            value = int(dyn_prop[index])  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            return False
        return value in _STATIC_DYN_PROP
    if vx_comp is not None and vy_comp is not None:
        speed = math.hypot(float(vx_comp[index]), float(vy_comp[index]))  # type: ignore[index]
        return speed <= _STATIC_VCOMP_THRESHOLD_MPS
    return False


def _radial_velocity(vx: float, vy: float, x: float, y: float, z: float) -> float:
    ux, uy, _uz = _unit_vector(x, y, z)
    return vx * ux + vy * uy


def _unit_vector(x: float, y: float, z: float) -> Vector3:
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0)
    return (x / norm, y / norm, z / norm)


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vector_norm(value: Vector3) -> float:
    return math.sqrt(_dot(value, value))


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _unavailable_stats(reason: str) -> RadarVelocityConsistencyStats:
    return RadarVelocityConsistencyStats(
        median_abs_residual_mps=None,
        inlier_rmse_mps=None,
        return_count=0,
        inlier_count=0,
        frame_count=0,
        status="unavailable",
        reason=reason,
    )


def _summary_float(payload: object, group: str, key: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    group_payload = payload.get(group)
    if not isinstance(group_payload, Mapping):
        return None
    value = group_payload.get(key)
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _safe_dataset_relative_path(
    dataset_path: str | Path,
    payload_path: str | Path,
) -> str:
    root = Path(dataset_path).resolve()
    payload = Path(payload_path).resolve()
    try:
        return payload.relative_to(root).as_posix()
    except ValueError:
        return "<outside_dataset_root>"
