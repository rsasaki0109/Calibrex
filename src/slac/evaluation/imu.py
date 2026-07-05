"""LiDAR-IMU extrinsic rotation candidate evaluation (ADR 0007 Pillar 3)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from slac.core.config import CalibrationConfig
from slac.core.geometry import (
    SE3,
    Vector3,
    normalize_quaternion_xyzw,
    quaternion_multiply_xyzw,
    rotate_vector_xyzw,
)
from slac.core.report_artifacts import EvidenceCaseItem
from slac.core.result import CalibrationResult, Grade, MetricResult
from slac.data.inspect import DatasetInspection
from slac.data.odometry_track import OdometryPoseSample, OdometryTrack
from slac.data.ros_messages import ImuMessage
from slac.data.rosbag2 import (
    IMU_TYPE,
    ODOMETRY_TYPE,
    decode_imu,
    decode_odometry,
    iter_messages,
)

LIDAR_IMU_FACTOR_NAME = "lidar_imu_rotation_consistency"
LIDAR_IMU_METRIC_PREFIX = "lidar_imu_"
_MANDATORY_ROTATION_DEG = (5.0, 10.0)
_MANDATORY_AXES = ("roll", "pitch", "yaw")
_MANDATORY_CASE_COUNT = 12.0
_DELTA_EPSILON = 1.0e-9
_GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class ImuEvidenceOptions:
    """Factor options controlling LiDAR-IMU rotation evidence."""

    holdout_block_stride: int = 5
    min_excitation_p95_dps: float = 5.0
    imu_gate_max_holdout_rotation_rate_rmse_dps: float = 5.0
    imu_gate_max_gravity_alignment_deg: float = 10.0
    known_bad_detect_margin: float = 0.20
    gravity_lowpass_window_s: float = 0.5
    imu_rotation_rate_smoothing_intervals: int = 2
    comparison_method: str = (
        "pose_interval_mean_imu_vs_odometry_log_symmetric_ma"
    )


@dataclass(frozen=True)
class _ImuSample:
    timestamp_ns: int
    angular_velocity: Vector3
    linear_acceleration: Vector3


@dataclass(frozen=True)
class _RotationInterval:
    timestamp_ns: int
    block_index: int
    omega_odometry_rad_s: Vector3
    omega_imu_rad_s: Vector3


@dataclass(frozen=True)
class ImuEvidencePayload:
    """Rotation and gravity evidence metrics, cases, and protocol metadata."""

    metrics: dict[str, MetricResult]
    cases: list[EvidenceCaseItem]
    protocol: dict[str, object]
    evidence_gates: dict[str, float | int]


def imu_evidence_options_from_factor(options: Mapping[str, Any]) -> ImuEvidenceOptions:
    """Parse LiDAR-IMU evidence options from factor options."""

    return ImuEvidenceOptions(
        holdout_block_stride=_int_option(options, "holdout_block_stride", default=5, minimum=2),
        min_excitation_p95_dps=_float_option(
            options, "min_excitation_p95_dps", default=5.0, minimum=0.0
        ),
        imu_gate_max_holdout_rotation_rate_rmse_dps=_float_option(
            options,
            "imu_gate_max_holdout_rotation_rate_rmse_dps",
            default=5.0,
            minimum=0.0,
        ),
        imu_gate_max_gravity_alignment_deg=_float_option(
            options,
            "imu_gate_max_gravity_alignment_deg",
            default=10.0,
            minimum=0.0,
        ),
        known_bad_detect_margin=_float_option(
            options, "known_bad_detect_margin", default=0.20, minimum=0.0
        ),
        gravity_lowpass_window_s=_float_option(
            options, "gravity_lowpass_window_s", default=0.5, minimum=0.01
        ),
        imu_rotation_rate_smoothing_intervals=_int_option(
            options, "imu_rotation_rate_smoothing_intervals", default=2, minimum=0
        ),
    )


def imu_metrics_from_result(
    config: CalibrationConfig,
    result: CalibrationResult,
    inspection: DatasetInspection,
) -> dict[str, MetricResult]:
    """Build LiDAR-IMU rotation-consistency and gravity-support evidence metrics."""

    if not _uses_imu_evaluation(config):
        return {}

    options = _evidence_options(config)
    skip_reason = _skip_reason(config, inspection)
    if skip_reason is not None:
        result.run.provenance["imu_evidence_skipped"] = skip_reason
        return {}

    imu_sensor_name, imu_topic = _configured_imu_sensor(config)
    root_name, candidate = _imu_candidate_transform(config, result, imu_sensor_name)
    if candidate is None:
        result.run.provenance["imu_evidence_skipped"] = "no imu frame candidate rotation in graph"
        return {}

    bag_path = Path(config.dataset.path)
    odometry_topic = config.dataset.odometry_topic
    if odometry_topic is None:
        result.run.provenance["imu_evidence_skipped"] = "no odometry topic configured"
        return {}

    odometry_track = _load_odometry_track(bag_path, odometry_topic)
    if odometry_track is None or odometry_track.message_count < 2:
        result.run.provenance["imu_evidence_skipped"] = "odometry track unavailable or too short"
        return {}

    imu_samples = _load_imu_samples(bag_path, imu_topic)
    if len(imu_samples) < 2:
        result.run.provenance["imu_evidence_skipped"] = f"no imu samples on topic {imu_topic!r}"
        return {}

    payload = _evaluate_imu_evidence(
        odometry_track=odometry_track,
        imu_samples=imu_samples,
        r_root_imu=candidate,
        options=options,
        imu_sensor_name=imu_sensor_name,
        root_name=root_name,
        imu_topic=imu_topic,
        odometry_topic=odometry_topic,
    )
    _store_imu_evidence_provenance(result, payload)
    return payload.metrics


def _evaluate_imu_evidence(
    *,
    odometry_track: OdometryTrack,
    imu_samples: list[_ImuSample],
    r_root_imu: SE3,
    options: ImuEvidenceOptions,
    imu_sensor_name: str,
    root_name: str,
    imu_topic: str,
    odometry_topic: str,
) -> ImuEvidencePayload:
    intervals = _build_rotation_intervals(
        odometry_track,
        imu_samples,
        r_root_imu,
        options=options,
    )
    if not intervals:
        return ImuEvidencePayload(
            _unavailable_metrics("no overlapping odometry pose intervals with imu samples"),
            [],
            {},
            _gate_dict(options),
        )

    raw_intervals = intervals
    intervals = _smooth_rotation_intervals(
        intervals,
        half_window=options.imu_rotation_rate_smoothing_intervals,
    )

    train_intervals, holdout_intervals = _split_train_holdout(
        intervals,
        block_stride=options.holdout_block_stride,
    )
    baseline_holdout_rmse = _rotation_rate_rmse_dps(holdout_intervals)
    baseline_holdout_mean = _rotation_rate_mean_abs_dps(holdout_intervals)
    train_rmse = _rotation_rate_rmse_dps(train_intervals)
    train_mean = _rotation_rate_mean_abs_dps(train_intervals)

    excitation_values = [
        _vector_norm_dps(interval.omega_odometry_rad_s) for interval in raw_intervals
    ]
    excitation_p95 = _percentile(excitation_values, 95.0)
    excitation_max = max(excitation_values) if excitation_values else 0.0
    excitation_p95_per_axis = {
        axis: _percentile(
            [
                math.degrees(abs(interval.omega_odometry_rad_s[index]))
                for interval in raw_intervals
            ],
            95.0,
        )
        for index, axis in enumerate(("x", "y", "z"))
    }

    holdout_grade: Grade = (
        "pass"
        if baseline_holdout_rmse is not None
        and baseline_holdout_rmse <= options.imu_gate_max_holdout_rotation_rate_rmse_dps
        else "warn"
    )

    cases, probe_metrics, probe_table = _known_bad_probes(
        odometry_track=odometry_track,
        imu_samples=imu_samples,
        baseline_r_root_imu=r_root_imu,
        holdout_intervals=holdout_intervals,
        baseline_holdout_rmse=baseline_holdout_rmse,
        options=options,
    )

    angle_deg, gravity_grade = _gravity_support(
        odometry_track=odometry_track,
        imu_samples=imu_samples,
        r_root_imu=r_root_imu,
        options=options,
    )

    support_grade: Grade = (
        "pass"
        if excitation_p95 >= options.min_excitation_p95_dps and len(intervals) >= 4
        else "warn"
    )

    metrics = {
        "lidar_imu_pose_interval_count": MetricResult(
            value=float(len(intervals)),
            unit="intervals",
            reason="odometry pose intervals with averaged imu samples",
        ),
        "lidar_imu_train_rotation_rate_rmse_dps": MetricResult(
            train=train_rmse,
            unit="deg/s",
            grade="pass",
            reason="train contiguous blocks rotation-rate RMSE",
        ),
        "lidar_imu_holdout_rotation_rate_rmse_dps": MetricResult(
            holdout=baseline_holdout_rmse,
            unit="deg/s",
            grade=holdout_grade,
            reason="holdout contiguous blocks rotation-rate RMSE",
        ),
        "lidar_imu_holdout_rotation_rate_mean_abs_dps": MetricResult(
            holdout=baseline_holdout_mean,
            unit="deg/s",
            grade=holdout_grade,
            reason="holdout mean absolute rotation-rate residual",
        ),
        "lidar_imu_train_rotation_rate_mean_abs_dps": MetricResult(
            train=train_mean,
            unit="deg/s",
            grade="pass",
            reason="train mean absolute rotation-rate residual",
        ),
        "lidar_imu_odometry_angular_excitation_p95_dps": MetricResult(
            value=excitation_p95,
            unit="deg/s",
            grade=support_grade,
            reason="p95 odometry-derived angular speed over pose intervals",
        ),
        "lidar_imu_odometry_angular_excitation_max_dps": MetricResult(
            value=excitation_max,
            unit="deg/s",
            grade="pass",
            reason="max odometry-derived angular speed over pose intervals",
        ),
        "lidar_imu_imu_sample_count": MetricResult(
            value=float(len(imu_samples)),
            unit="samples",
            grade="pass",
            reason=f"imu samples averaged within pose intervals from {imu_topic}",
        ),
        "lidar_imu_gravity_alignment_deg": MetricResult(
            value=angle_deg,
            unit="deg",
            grade=gravity_grade,
            reason="angle between low-pass acceleration and world gravity (supporting-only)",
        ),
    }
    metrics.update(probe_metrics)

    comparison_method = (
        f"{options.comparison_method}{options.imu_rotation_rate_smoothing_intervals}"
        if options.imu_rotation_rate_smoothing_intervals > 0
        else "pose_interval_mean_imu_vs_odometry_log"
    )
    protocol = {
        "protocol_id": "lidar_imu_rotation_gravity_holdout/v0.1",
        "comparison_method": comparison_method,
        "holdout_split": f"contiguous_blocks_stride_{options.holdout_block_stride}",
        "imu_rotation_rate_smoothing_intervals": options.imu_rotation_rate_smoothing_intervals,
        "imu_sensor": imu_sensor_name,
        "root_frame": root_name,
        "imu_topic": imu_topic,
        "odometry_topic": odometry_topic,
        "candidate_rotation_note": (
            "R_root_imu candidate rotation evaluated; OS1 internal IMU is approximately "
            "rotation-aligned with the os1 sensor frame (R_velo_imu ≈ R_velo_os1)"
        ),
        "gravity_channel": "supporting_only",
        "gravity_lowpass_window_s": options.gravity_lowpass_window_s,
        "odometry_angular_excitation_p95_dps_per_axis": excitation_p95_per_axis,
        "known_bad_probe_axes_deg": list(_MANDATORY_ROTATION_DEG),
        "known_bad_probe_margin_relative": options.known_bad_detect_margin,
        "known_bad_probe_table": probe_table,
    }

    return ImuEvidencePayload(
        metrics=metrics,
        cases=cases,
        protocol=protocol,
        evidence_gates=_gate_dict(options),
    )


def _known_bad_probes(
    *,
    odometry_track: OdometryTrack,
    imu_samples: list[_ImuSample],
    baseline_r_root_imu: SE3,
    holdout_intervals: list[_RotationInterval],
    baseline_holdout_rmse: float | None,
    options: ImuEvidenceOptions,
) -> tuple[list[EvidenceCaseItem], dict[str, MetricResult], list[dict[str, object]]]:
    if not holdout_intervals:
        return [], _unavailable_probe_metrics(
            "no holdout pose intervals for known-bad rotation probes"
        ), []

    if baseline_holdout_rmse is None:
        return [], _unavailable_probe_metrics(
            "holdout rotation-rate RMSE unavailable for known-bad probes"
        ), []

    cases: list[EvidenceCaseItem] = []
    probe_table: list[dict[str, object]] = []
    detectable = 0
    mandatory_detectable = 0
    for axis in ("roll", "pitch", "yaw"):
        for amount_deg in _MANDATORY_ROTATION_DEG:
            for sign in (-1.0, 1.0):
                signed_amount = sign * amount_deg
                perturbed = _perturb_rotation(baseline_r_root_imu, axis, signed_amount)
                perturbed_intervals = _recompute_imu_omega_for_probe(
                    holdout_intervals,
                    baseline_r_root_imu,
                    perturbed,
                )
                perturbed_rmse = _rotation_rate_rmse_dps(perturbed_intervals)
                relative_increase = (
                    (perturbed_rmse - baseline_holdout_rmse)
                    / max(baseline_holdout_rmse, _DELTA_EPSILON)
                    if perturbed_rmse is not None
                    else None
                )
                if baseline_holdout_rmse <= _DELTA_EPSILON:
                    detected = perturbed_rmse is not None and perturbed_rmse > _DELTA_EPSILON
                else:
                    detected = (
                        relative_increase is not None
                        and relative_increase >= options.known_bad_detect_margin
                    )
                if detected:
                    detectable += 1
                    mandatory_detectable += 1
                probe_table.append(
                    {
                        "axis": axis,
                        "angle_deg": signed_amount,
                        "baseline_holdout_rmse_dps": baseline_holdout_rmse,
                        "perturbed_holdout_rmse_dps": perturbed_rmse,
                        "holdout_rmse_relative_delta": relative_increase,
                        "holdout_rmse_delta_dps": (
                            perturbed_rmse - baseline_holdout_rmse
                            if perturbed_rmse is not None
                            else None
                        ),
                        "detected": detected,
                    }
                )
                cases.append(
                    EvidenceCaseItem(
                        family="lidar_imu",
                        case_id=f"{axis}_{signed_amount:+.0f}deg",
                        check="Known-Bad Controls",
                        status="pass" if detected else "warn",
                        dof=axis,
                        amount=signed_amount,
                        unit="deg",
                        convention="left-multiplied rotation perturbation of R_root_imu",
                        metric_values={
                            "lidar_imu_holdout_rotation_rate_rmse_dps": perturbed_rmse,
                        },
                        delta_values={
                            "holdout_rmse_relative_increase": relative_increase,
                            "holdout_rmse_delta_dps": (
                                perturbed_rmse - baseline_holdout_rmse
                                if perturbed_rmse is not None
                                else None
                            ),
                        },
                    )
                )

    case_count = float(len(cases))
    detectable_fraction = detectable / case_count if case_count else 0.0
    probe_grade: Grade = _known_bad_fraction_grade(detectable_fraction)
    metrics = {
        "lidar_imu_known_bad_case_count": MetricResult(
            value=case_count,
            unit="cases",
            grade="pass",
            reason="mandatory ±5°/±10° rotation probes about roll/pitch/yaw",
        ),
        "lidar_imu_known_bad_detectable_fraction": MetricResult(
            value=detectable_fraction,
            grade=probe_grade,
            reason="fraction of known-bad rotation probes increasing holdout RMSE",
        ),
        "lidar_imu_known_bad_mandatory_case_count": MetricResult(
            value=_MANDATORY_CASE_COUNT,
            unit="cases",
            grade="pass",
            reason="declared mandatory rotation probe count",
        ),
        "lidar_imu_known_bad_mandatory_detectable_count": MetricResult(
            value=float(mandatory_detectable),
            unit="cases",
            grade=probe_grade,
            reason="mandatory rotation probes detected under holdout RMSE margin",
        ),
    }
    return cases, metrics, probe_table


def _gravity_support(
    *,
    odometry_track: OdometryTrack,
    imu_samples: list[_ImuSample],
    r_root_imu: SE3,
    options: ImuEvidenceOptions,
) -> tuple[float | None, Grade]:
    filtered = _lowpass_imu_samples(imu_samples, window_s=options.gravity_lowpass_window_s)
    if not filtered:
        return None, "warn"

    angles_deg: list[float] = []
    gravity_world: Vector3 = (0.0, 0.0, _GRAVITY_MPS2)
    gravity_dir = _normalize(gravity_world)
    if gravity_dir is None:
        return None, "warn"
    for sample in filtered:
        pose, _clamped, _extrap = odometry_track.interpolate(sample.timestamp_ns)
        accel_root = rotate_vector_xyzw(
            quaternion_multiply_xyzw(
                pose.rotation_quat_xyzw,
                r_root_imu.rotation_quat_xyzw,
            ),
            sample.linear_acceleration,
        )
        accel_dir = _normalize(accel_root)
        if accel_dir is None:
            continue
        dot = max(-1.0, min(1.0, _dot(accel_dir, gravity_dir)))
        angles_deg.append(math.degrees(math.acos(dot)))

    if not angles_deg:
        return None, "warn"
    mean_angle = sum(angles_deg) / len(angles_deg)
    if mean_angle <= options.imu_gate_max_gravity_alignment_deg:
        return mean_angle, "pass"
    if mean_angle <= options.imu_gate_max_gravity_alignment_deg * 2.0:
        return mean_angle, "warn"
    return mean_angle, "warn"


def _build_rotation_intervals(
    odometry_track: OdometryTrack,
    imu_samples: list[_ImuSample],
    r_root_imu: SE3,
    *,
    options: ImuEvidenceOptions,
) -> list[_RotationInterval]:
    _ = options
    samples = odometry_track.samples
    intervals: list[_RotationInterval] = []
    block_index = 0
    for left, right in pairwise(samples):
        dt_s = max((right.timestamp_ns - left.timestamp_ns) / 1_000_000_000, 1.0e-9)
        midpoint_ns = (left.timestamp_ns + right.timestamp_ns) // 2
        omega_odometry = _rotation_rate_rad_s(left.pose, right.pose, dt_s)
        interval_imu = [
            sample
            for sample in imu_samples
            if left.timestamp_ns <= sample.timestamp_ns <= right.timestamp_ns
        ]
        if not interval_imu:
            continue
        mean_imu = _mean_vector(sample.angular_velocity for sample in interval_imu)
        omega_imu = rotate_vector_xyzw(r_root_imu.rotation_quat_xyzw, mean_imu)
        intervals.append(
            _RotationInterval(
                timestamp_ns=midpoint_ns,
                block_index=block_index,
                omega_odometry_rad_s=omega_odometry,
                omega_imu_rad_s=omega_imu,
            )
        )
        block_index += 1
    return intervals


def _smooth_rotation_intervals(
    intervals: list[_RotationInterval],
    *,
    half_window: int,
) -> list[_RotationInterval]:
    if half_window <= 0 or not intervals:
        return intervals
    smoothed: list[_RotationInterval] = []
    for index, interval in enumerate(intervals):
        start = max(0, index - half_window)
        end = min(len(intervals), index + half_window + 1)
        window = intervals[start:end]
        smoothed.append(
            _RotationInterval(
                timestamp_ns=interval.timestamp_ns,
                block_index=interval.block_index,
                omega_odometry_rad_s=_mean_vector(
                    candidate.omega_odometry_rad_s for candidate in window
                ),
                omega_imu_rad_s=_mean_vector(
                    candidate.omega_imu_rad_s for candidate in window
                ),
            )
        )
    return smoothed


def _recompute_imu_omega_for_probe(
    holdout_intervals: list[_RotationInterval],
    baseline_r_root_imu: SE3,
    perturbed_r_root_imu: SE3,
) -> list[_RotationInterval]:
    delta = perturbed_r_root_imu.compose(baseline_r_root_imu.inverse())
    return [
        _RotationInterval(
            timestamp_ns=interval.timestamp_ns,
            block_index=interval.block_index,
            omega_odometry_rad_s=interval.omega_odometry_rad_s,
            omega_imu_rad_s=rotate_vector_xyzw(
                delta.rotation_quat_xyzw,
                interval.omega_imu_rad_s,
            ),
        )
        for interval in holdout_intervals
    ]


def _split_train_holdout(
    intervals: list[_RotationInterval],
    *,
    block_stride: int,
) -> tuple[list[_RotationInterval], list[_RotationInterval]]:
    train: list[_RotationInterval] = []
    holdout: list[_RotationInterval] = []
    for interval in intervals:
        if interval.block_index % block_stride == block_stride - 1:
            holdout.append(interval)
        else:
            train.append(interval)
    return train, holdout


def _rotation_rate_rmse_dps(intervals: Sequence[_RotationInterval]) -> float | None:
    if not intervals:
        return None
    squared = [
        _vector_norm_dps(_residual_rad_s(interval)) ** 2 for interval in intervals
    ]
    return math.sqrt(sum(squared) / len(squared))


def _rotation_rate_mean_abs_dps(intervals: Sequence[_RotationInterval]) -> float | None:
    if not intervals:
        return None
    values = [_vector_norm_dps(_residual_rad_s(interval)) for interval in intervals]
    return sum(values) / len(values)


def _residual_rad_s(interval: _RotationInterval) -> Vector3:
    return (
        interval.omega_odometry_rad_s[0] - interval.omega_imu_rad_s[0],
        interval.omega_odometry_rad_s[1] - interval.omega_imu_rad_s[1],
        interval.omega_odometry_rad_s[2] - interval.omega_imu_rad_s[2],
    )


def _rotation_rate_rad_s(left: SE3, right: SE3, dt_s: float) -> Vector3:
    delta = left.inverse().compose(right)
    x, y, z, w = delta.rotation_quat_xyzw
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = min(1.0, max(-1.0, w))
    angle = 2.0 * math.acos(w)
    if angle < 1.0e-12 or dt_s < 1.0e-12:
        return (0.0, 0.0, 0.0)
    sin_half = math.sin(angle / 2.0)
    if abs(sin_half) < 1.0e-12:
        return (0.0, 0.0, 0.0)
    scale = angle / sin_half / dt_s
    return (x * scale, y * scale, z * scale)


def _perturb_rotation(base: SE3, axis: str, degrees: float) -> SE3:
    radians = math.radians(degrees)
    half = radians / 2.0
    sin_half = math.sin(half)
    cos_half = math.cos(half)
    if axis == "roll":
        delta_q = (sin_half, 0.0, 0.0, cos_half)
    elif axis == "pitch":
        delta_q = (0.0, sin_half, 0.0, cos_half)
    else:
        delta_q = (0.0, 0.0, sin_half, cos_half)
    delta_q = normalize_quaternion_xyzw(delta_q)
    perturbed_q = quaternion_multiply_xyzw(delta_q, base.rotation_quat_xyzw)
    return SE3(base.translation_m, perturbed_q)


def _lowpass_imu_samples(
    samples: list[_ImuSample],
    *,
    window_s: float,
) -> list[_ImuSample]:
    if not samples:
        return []
    window_ns = int(window_s * 1_000_000_000)
    filtered: list[_ImuSample] = []
    for sample in samples:
        window = [
            candidate
            for candidate in samples
            if abs(candidate.timestamp_ns - sample.timestamp_ns) <= window_ns // 2
        ]
        filtered.append(
            _ImuSample(
                timestamp_ns=sample.timestamp_ns,
                angular_velocity=sample.angular_velocity,
                linear_acceleration=_mean_vector(
                    candidate.linear_acceleration for candidate in window
                ),
            )
        )
    return filtered


def _load_odometry_track(bag_path: Path, odometry_topic: str) -> OdometryTrack | None:
    samples: list[OdometryPoseSample] = []
    for connection, timestamp_ns, data in iter_messages(bag_path, topics={odometry_topic}):
        if connection.message_type != ODOMETRY_TYPE:
            continue
        message = decode_odometry(connection.topic, timestamp_ns, data)
        samples.append(
            OdometryPoseSample(
                timestamp_ns=message.timestamp_ns,
                pose=SE3.from_lists(list(message.position), list(message.orientation_xyzw)),
            )
        )
    if not samples:
        return None
    return OdometryTrack(samples)


def _load_imu_samples(bag_path: Path, imu_topic: str) -> list[_ImuSample]:
    samples: list[_ImuSample] = []
    for connection, timestamp_ns, data in iter_messages(bag_path, topics={imu_topic}):
        if connection.message_type != IMU_TYPE:
            continue
        message = decode_imu(connection.topic, timestamp_ns, data)
        samples.append(_imu_sample_from_message(message))
    return sorted(samples, key=lambda sample: sample.timestamp_ns)


def _imu_sample_from_message(message: ImuMessage) -> _ImuSample:
    return _ImuSample(
        timestamp_ns=message.timestamp_ns,
        angular_velocity=message.angular_velocity,
        linear_acceleration=message.linear_acceleration,
    )


def _configured_imu_sensor(config: CalibrationConfig) -> tuple[str, str]:
    for name, sensor in config.sensors.items():
        if sensor.type == "imu" and sensor.topic:
            return name, sensor.topic
    msg = "no imu sensor with topic configured"
    raise ValueError(msg)


def _imu_candidate_transform(
    config: CalibrationConfig,
    result: CalibrationResult,
    imu_sensor_name: str,
) -> tuple[str, SE3 | None]:
    frame = config.frames.get(imu_sensor_name)
    if frame is None or frame.parent is None:
        return "", None
    root_name = frame.parent
    transform = result.transforms.get(imu_sensor_name)
    if transform is not None:
        return root_name, transform.as_se3()
    if frame.transform is not None:
        initial = frame.transform.initial
        return root_name, SE3.from_lists(
            list(initial.translation),
            list(initial.rotation_quat_xyzw),
        )
    return root_name, None


def _uses_imu_evaluation(config: CalibrationConfig) -> bool:
    if any(
        metric_name.startswith(LIDAR_IMU_METRIC_PREFIX)
        for metric_name in config.evaluation.metrics
    ):
        return True
    factor = config.pipeline.factors.get(LIDAR_IMU_FACTOR_NAME)
    return factor is not None and factor.enabled


def _evidence_options(config: CalibrationConfig) -> ImuEvidenceOptions:
    factor = config.pipeline.factors.get(LIDAR_IMU_FACTOR_NAME)
    if factor is not None:
        return imu_evidence_options_from_factor(factor.options)
    return ImuEvidenceOptions()


def _skip_reason(config: CalibrationConfig, inspection: DatasetInspection) -> str | None:
    if config.dataset.type != "rosbag2":
        return f"dataset type {config.dataset.type!r} is not rosbag2"
    imu_sensors = [name for name, sensor in config.sensors.items() if sensor.type == "imu"]
    if not imu_sensors:
        return "no imu sensor configured"
    if not any(config.sensors[name].topic for name in imu_sensors):
        return "imu sensor has no topic"
    if config.dataset.odometry_topic is None:
        return "no odometry topic configured"
    if not inspection.exists:
        return "dataset path missing"
    return None


def _store_imu_evidence_provenance(result: CalibrationResult, payload: ImuEvidencePayload) -> None:
    if "lidar_imu_holdout_rotation_rate_rmse_dps" not in payload.metrics:
        return
    existing_cases = result.run.provenance.get("evidence_cases")
    if isinstance(existing_cases, list):
        merged = [*existing_cases, *[case.model_dump() for case in payload.cases]]
    else:
        merged = [case.model_dump() for case in payload.cases]
    result.run.provenance["evidence_cases"] = merged
    result.run.provenance["lidar_imu_evidence"] = {
        **payload.protocol,
        "evidence_gates": payload.evidence_gates,
    }


def _gate_dict(options: ImuEvidenceOptions) -> dict[str, float | int]:
    return {
        "imu_gate_max_holdout_rotation_rate_rmse_dps": (
            options.imu_gate_max_holdout_rotation_rate_rmse_dps
        ),
        "imu_gate_max_gravity_alignment_deg": options.imu_gate_max_gravity_alignment_deg,
        "known_bad_detect_margin": options.known_bad_detect_margin,
        "min_excitation_p95_dps": options.min_excitation_p95_dps,
        "holdout_block_stride": options.holdout_block_stride,
        "imu_rotation_rate_smoothing_intervals": options.imu_rotation_rate_smoothing_intervals,
    }


def _unavailable_metrics(reason: str) -> dict[str, MetricResult]:
    return {
        "lidar_imu_holdout_rotation_rate_rmse_dps": MetricResult(
            value=None,
            unit="deg/s",
            grade="warn",
            reason=reason,
        ),
    }


def _unavailable_probe_metrics(reason: str) -> dict[str, MetricResult]:
    return {
        "lidar_imu_known_bad_detectable_fraction": MetricResult(
            value=None,
            grade="warn",
            reason=reason,
        ),
    }


def _known_bad_fraction_grade(detectable_fraction: float) -> Grade:
    if detectable_fraction >= 0.5:
        return "pass"
    if detectable_fraction >= 0.1:
        return "warn"
    return "fail"


def _mean_vector(vectors: Iterable[Vector3]) -> Vector3:
    items = list(vectors)
    if not items:
        return (0.0, 0.0, 0.0)
    return (
        sum(vector[0] for vector in items) / len(items),
        sum(vector[1] for vector in items) / len(items),
        sum(vector[2] for vector in items) / len(items),
    )


def _vector_norm_dps(vector_rad_s: Vector3) -> float:
    return math.degrees(
        math.sqrt(vector_rad_s[0] ** 2 + vector_rad_s[1] ** 2 + vector_rad_s[2] ** 2)
    )


def _normalize(vector: Vector3) -> Vector3 | None:
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if norm < 1.0e-12:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


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
    options: Mapping[str, Any], key: str, *, default: float, minimum: float
) -> float:
    try:
        value = float(options[key])
    except (KeyError, TypeError, ValueError):
        return default
    return max(value, minimum)


def _int_option(
    options: Mapping[str, Any], key: str, *, default: int, minimum: int
) -> int:
    try:
        value = int(options[key])
    except (KeyError, TypeError, ValueError):
        return default
    return max(value, minimum)
