"""nuScenes adapter for Radar lever-arm and clock-offset evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from calibrex.core.config import CalibrationConfig
from calibrex.core.geometry import (
    SE3,
    QuaternionXYZW,
    Vector3,
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    rotate_vector_xyzw,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import (
    CalibrationResult,
    MetricResult,
    ObservabilityResult,
    TimeOffsetQuality,
    TimeOffsetResult,
)
from calibrex.data.inspect import DatasetInspection
from calibrex.data.nuscenes import NuScenesDataset, read_nuscenes_ego_poses
from calibrex.data.nuscenes_radar import NuScenesRadarPCD, read_nuscenes_radar_pcd
from calibrex.solvers.radar_joint_spatiotemporal_solver import (
    RadarJointSpatiotemporalOptions,
    RadarJointSpatiotemporalResult,
    RadarJointSpatiotemporalSolver,
)
from calibrex.solvers.radar_spatiotemporal_lever_arm_solver import (
    RadarSpatiotemporalLeverArmOptions,
    RadarSpatiotemporalLeverArmResult,
    RadarSpatiotemporalLeverArmSolver,
    RadarVelocityMeasurement,
    ReferenceKinematicSample,
)
from calibrex.solvers.radar_trajectory_rotation_solver import (
    RadarTrajectoryRotationOptions,
    RadarTrajectoryRotationPair,
    RadarTrajectoryRotationResult,
    RadarTrajectoryRotationSolver,
)

_MIN_STATIC_RETURNS = 5
_MIN_RANGE_M = 1.0
_STATIC_DYN_PROP = {1, 3, 5, 7}
_STATIC_VCOMP_THRESHOLD_MPS = 0.5


@dataclass(frozen=True)
class NuScenesRadarSpatiotemporalEvidence:
    """Adapter output and input accounting for one nuScenes Radar channel."""

    status: str
    reason: str
    result: RadarSpatiotemporalLeverArmResult | None
    rotation_result: RadarTrajectoryRotationResult | None
    joint_result: RadarJointSpatiotemporalResult | None
    channel: str
    rotation_ego_radar_xyzw: QuaternionXYZW
    translation_ego_radar_m: Vector3
    measurement_count: int
    reference_sample_count: int
    rejected_scan_count: int
    raw_input_files: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "channel": self.channel,
            "rotation_ego_radar_xyzw": list(self.rotation_ego_radar_xyzw),
            "translation_ego_radar_m": list(self.translation_ego_radar_m),
            "measurement_count": self.measurement_count,
            "reference_sample_count": self.reference_sample_count,
            "rejected_scan_count": self.rejected_scan_count,
            "result": self.result.as_dict() if self.result is not None else None,
            "rotation_result": (
                self.rotation_result.as_dict() if self.rotation_result is not None else None
            ),
            "joint_result": self.joint_result.as_dict() if self.joint_result is not None else None,
            "raw_input_files": list(self.raw_input_files),
            "metrics_origin": "recomputed",
            "dataset_license": "nuScenes terms of use; user-supplied local download",
        }


def radar_spatiotemporal_metrics_from_result(
    config: CalibrationConfig,
    result: CalibrationResult,
    inspection: DatasetInspection,
) -> dict[str, MetricResult]:
    """Run optional public-data Radar lever-arm/time evidence and apply outputs."""

    del inspection
    factor = config.pipeline.factors.get("radar_spatiotemporal_velocity")
    if config.dataset.type != "nuscenes" or factor is None or not factor.enabled:
        return {}
    options = factor.options
    sensor_name = str(options.get("sensor", "radar_front"))
    channel = str(options.get("channel", "RADAR_FRONT"))
    transform_name = f"T_ego_{sensor_name}"
    transform = result.reference_extrinsics.get(transform_name) or result.transforms.get(
        transform_name
    )
    if transform is None:
        return {
            "radar_spatiotemporal_available": MetricResult(
                value=0.0,
                unit="bool",
                grade="warn",
                reason=f"missing candidate rotation {transform_name}",
            )
        }
    t_ego_radar = SE3.from_lists(transform.translation_m, transform.rotation_quat_xyzw)
    time_config = config.time_offsets.get(sensor_name)
    evidence = summarize_nuscenes_radar_spatiotemporal(
        dataset_path=config.dataset.path,
        channel=channel,
        rotation_ego_radar_xyzw=t_ego_radar.rotation_quat_xyzw,
        translation_ego_radar_m=t_ego_radar.translation_m,
        initial_time_offset_sec=time_config.initial_sec if time_config is not None else 0.0,
        joint_known_bad_rotation_deg=float(options.get("joint_known_bad_rotation_deg", 5.0)),
        joint_max_initial_residual_mps=float(options.get("joint_max_initial_residual_mps", 3.0)),
        options=RadarSpatiotemporalLeverArmOptions(
            max_abs_time_offset_sec=float(options.get("max_abs_time_offset_sec", 0.1)),
            time_offset_step_sec=float(options.get("time_offset_step_sec", 0.002)),
            holdout_ratio=config.evaluation.holdout_ratio,
            split_seed=config.solver.seed or 0,
            min_train_measurements=int(options.get("min_train_measurements", 12)),
            rank_tolerance=float(options.get("rank_tolerance", 1.0e-8)),
            max_condition_number=float(options.get("max_condition_number", 1.0e6)),
            joint_rank_tolerance=float(options.get("joint_rank_tolerance", 1.0e-8)),
            max_joint_condition_number=float(options.get("max_joint_condition_number", 1.0e6)),
            known_bad_translation_m=float(options.get("known_bad_translation_m", 0.1)),
            known_bad_time_offset_sec=float(options.get("known_bad_time_offset_sec", 0.02)),
            known_bad_margin_mps=float(options.get("known_bad_margin_mps", 0.02)),
        ),
        max_frames=int(options.get("max_frames", 64)),
    )
    result.run.provenance["radar_spatiotemporal_velocity"] = evidence.as_dict()
    if evidence.raw_input_files:
        result.run.provenance.setdefault("raw_input_files", []).extend(evidence.raw_input_files)
    solved = evidence.result
    if solved is not None:
        weak_directions: list[str] = []
        if solved.lever_arm_rank < 3:
            weak_directions.append("radar_lever_arm")
        if solved.joint_rank < 4:
            weak_directions.append("radar_clock_lever_arm_coupling")
        result.observability = ObservabilityResult(
            rank=solved.joint_rank,
            condition_number=solved.joint_condition_number,
            weak_directions=weak_directions,
            grade="pass" if solved.status == "converged" else "warn",
        )
        if solved.status == "converged" and solved.translation_body_radar_m is not None:
            if transform_name in result.transforms:
                result.transforms[transform_name].translation_m = list(
                    solved.translation_body_radar_m
                )
            if solved.time_offset_sec is not None:
                result.time_offsets[sensor_name] = TimeOffsetResult(
                    seconds=solved.time_offset_sec,
                    quality=TimeOffsetQuality(grade="pass"),
                )
    return _evidence_metrics(evidence)


def summarize_nuscenes_radar_spatiotemporal(
    *,
    dataset_path: str | Path,
    channel: str,
    rotation_ego_radar_xyzw: QuaternionXYZW,
    translation_ego_radar_m: Vector3 = (0.0, 0.0, 0.0),
    initial_time_offset_sec: float = 0.0,
    joint_known_bad_rotation_deg: float = 5.0,
    joint_max_initial_residual_mps: float = 3.0,
    options: RadarSpatiotemporalLeverArmOptions,
    max_frames: int = 64,
) -> NuScenesRadarSpatiotemporalEvidence:
    """Build scan velocities and reference kinematics, then run the native solver."""

    root = Path(dataset_path)
    dataset = NuScenesDataset(root)
    records = sorted(dataset.records(channel), key=lambda item: item.timestamp_ns)
    poses = read_nuscenes_ego_poses(root)
    if len(records) < 3 or not poses:
        return NuScenesRadarSpatiotemporalEvidence(
            "unavailable",
            "nuScenes metadata, ego poses, or Radar records are unavailable",
            None,
            None,
            None,
            channel,
            rotation_ego_radar_xyzw,
            translation_ego_radar_m,
            0,
            0,
            0,
            _metadata_manifests(root),
        )
    sampled = _uniform_sample(records, max_frames)
    origin_ns = sampled[0].timestamp_ns
    reference: list[ReferenceKinematicSample] = []
    measurements: list[RadarVelocityMeasurement] = []
    manifests: list[dict[str, object]] = list(_metadata_manifests(root))
    rejected = 0
    previous_pose = None
    for record in sampled:
        pose = poses.get(str(record.metadata.get("ego_pose_token")))
        if pose is None:
            rejected += 1
            continue
        timestamp = (record.timestamp_ns - origin_ns) * 1.0e-9
        if previous_pose is not None:
            dt = (pose.timestamp_ns - previous_pose.timestamp_ns) * 1.0e-9
            if dt > 0.0:
                reference.append(_kinematic_sample(previous_pose, pose, timestamp, dt))
        previous_pose = pose
        payload = Path(record.payload_path) if record.payload_path else None
        if payload is None or not payload.exists():
            rejected += 1
            continue
        try:
            pcd = read_nuscenes_radar_pcd(payload)
            velocity = estimate_planar_radar_ego_velocity(pcd)
        except (OSError, ValueError, KeyError, IndexError):
            velocity = None
        if velocity is None:
            rejected += 1
            continue
        measurements.append(
            RadarVelocityMeasurement(f"{channel}:{record.timestamp_ns}", timestamp, velocity)
        )
        manifests.append(_manifest(payload, "nuscenes_radar_pcd"))
    joint_solved = RadarJointSpatiotemporalSolver().solve(
        measurements,
        reference,
        SE3(translation_ego_radar_m, rotation_ego_radar_xyzw),
        RadarJointSpatiotemporalOptions(
            holdout_ratio=options.holdout_ratio,
            split_seed=options.split_seed,
            min_train_measurements=options.min_train_measurements,
            max_initial_residual_mps=joint_max_initial_residual_mps,
            max_abs_time_offset_sec=options.max_abs_time_offset_sec,
            initial_time_offset_sec=initial_time_offset_sec,
            rank_tolerance=options.joint_rank_tolerance,
            max_condition_number=options.max_joint_condition_number,
            known_bad_translation_m=options.known_bad_translation_m,
            known_bad_rotation_deg=joint_known_bad_rotation_deg,
            known_bad_time_offset_sec=options.known_bad_time_offset_sec,
            known_bad_margin_mps=options.known_bad_margin_mps,
        ),
    )
    joint_rejected = set(joint_solved.rejected_initial_residual_ids)
    comparison_measurements = [
        item for item in measurements if item.measurement_id not in joint_rejected
    ]
    rotation_pairs = _rotation_pairs(comparison_measurements, reference, translation_ego_radar_m)
    rotation_solved = RadarTrajectoryRotationSolver().solve(
        rotation_pairs,
        RadarTrajectoryRotationOptions(
            holdout_ratio=options.holdout_ratio,
            split_seed=options.split_seed,
        ),
    )
    solved = RadarSpatiotemporalLeverArmSolver().solve(
        comparison_measurements, reference, rotation_ego_radar_xyzw, options
    )
    status = "scored" if solved.status == "converged" else "inconclusive"
    return NuScenesRadarSpatiotemporalEvidence(
        status,
        solved.reason,
        solved,
        rotation_solved,
        joint_solved,
        channel,
        rotation_ego_radar_xyzw,
        translation_ego_radar_m,
        len(measurements),
        len(reference),
        rejected,
        tuple(manifests),
    )


def estimate_planar_radar_ego_velocity(pcd: NuScenesRadarPCD) -> Vector3 | None:
    """Estimate 2D Radar-frame ego velocity from static raw Doppler returns."""

    rows: list[tuple[float, float]] = []
    values: list[float] = []
    for index in range(pcd.point_count):
        x = float(pcd.array("x")[index])
        y = float(pcd.array("y")[index])
        z = float(pcd.array("z")[index])
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < _MIN_RANGE_M or not _is_static(pcd, index):
            continue
        measured = (float(pcd.array("vx")[index]) * x + float(pcd.array("vy")[index]) * y) / norm
        rows.append((x / norm, y / norm))
        values.append(-measured)
    if len(rows) < _MIN_STATIC_RETURNS:
        return None
    matrix: np.ndarray[Any, np.dtype[np.float64]] = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(matrix) < 2:
        return None
    velocity, _residuals, _rank, _spectrum = np.linalg.lstsq(
        matrix, np.asarray(values, dtype=np.float64), rcond=None
    )
    return (float(velocity[0]), float(velocity[1]), 0.0)


def _rotation_pairs(
    measurements: list[RadarVelocityMeasurement],
    reference: list[ReferenceKinematicSample],
    translation_ego_radar_m: Vector3,
) -> list[RadarTrajectoryRotationPair]:
    """Form the Wise velocity relation at the configured Radar origin."""

    pairs: list[RadarTrajectoryRotationPair] = []
    ordered = sorted(reference, key=lambda sample: sample.timestamp_sec)
    for measurement in measurements:
        kinematic = _interpolate_kinematics(ordered, measurement.timestamp_sec)
        if kinematic is None:
            continue
        linear, angular = kinematic
        tx, ty, tz = translation_ego_radar_m
        wx, wy, wz = angular
        lever_velocity = (
            wy * tz - wz * ty,
            wz * tx - wx * tz,
            wx * ty - wy * tx,
        )
        velocity_at_origin: Vector3 = (
            linear[0] + lever_velocity[0],
            linear[1] + lever_velocity[1],
            linear[2] + lever_velocity[2],
        )
        pairs.append(
            RadarTrajectoryRotationPair(
                measurement.measurement_id,
                measurement.velocity_radar_mps,
                velocity_at_origin,
                measurement.weight,
            )
        )
    return pairs


def _interpolate_kinematics(
    samples: list[ReferenceKinematicSample], timestamp_sec: float
) -> tuple[Vector3, Vector3] | None:
    if len(samples) < 2 or timestamp_sec < samples[0].timestamp_sec:
        return None
    for left, right in pairwise(samples):
        if left.timestamp_sec <= timestamp_sec <= right.timestamp_sec:
            span = right.timestamp_sec - left.timestamp_sec
            alpha = (timestamp_sec - left.timestamp_sec) / span if span > 0.0 else 0.0
            linear: Vector3 = _lerp_vector(
                left.linear_velocity_body_mps,
                right.linear_velocity_body_mps,
                alpha,
            )
            angular: Vector3 = _lerp_vector(
                left.angular_velocity_body_radps,
                right.angular_velocity_body_radps,
                alpha,
            )
            return linear, angular
    return None


def _lerp_vector(left: Vector3, right: Vector3, alpha: float) -> Vector3:
    return (
        left[0] + alpha * (right[0] - left[0]),
        left[1] + alpha * (right[1] - left[1]),
        left[2] + alpha * (right[2] - left[2]),
    )


def _kinematic_sample(
    previous: Any, current: Any, timestamp: float, dt: float
) -> ReferenceKinematicSample:
    world_velocity = tuple(
        (current.translation_m[index] - previous.translation_m[index]) / dt for index in range(3)
    )
    inverse_current = quaternion_conjugate_xyzw(current.rotation_quat_xyzw)
    body_velocity = rotate_vector_xyzw(inverse_current, world_velocity)
    delta = quaternion_multiply_xyzw(
        quaternion_conjugate_xyzw(previous.rotation_quat_xyzw),
        current.rotation_quat_xyzw,
    )
    angular = _rotation_vector(delta, dt)
    return ReferenceKinematicSample(timestamp, body_velocity, angular)


def _rotation_vector(quaternion: QuaternionXYZW, dt: float) -> Vector3:
    x, y, z, w = quaternion
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1.0e-12:
        return (0.0, 0.0, 0.0)
    scale = 2.0 * math.atan2(vector_norm, max(-1.0, min(1.0, w))) / (vector_norm * dt)
    return (scale * x, scale * y, scale * z)


def _is_static(pcd: NuScenesRadarPCD, index: int) -> bool:
    dyn_prop = pcd.fields.get("dyn_prop")
    if dyn_prop is not None and int(dyn_prop[index]) not in _STATIC_DYN_PROP:
        return False
    vx_comp = pcd.fields.get("vx_comp")
    vy_comp = pcd.fields.get("vy_comp")
    return not (
        vx_comp is not None
        and vy_comp is not None
        and math.hypot(float(vx_comp[index]), float(vy_comp[index])) > _STATIC_VCOMP_THRESHOLD_MPS
    )


def _evidence_metrics(evidence: NuScenesRadarSpatiotemporalEvidence) -> dict[str, MetricResult]:
    solved = evidence.result
    if solved is None:
        return {
            "radar_spatiotemporal_available": MetricResult(
                value=0.0, unit="bool", grade="warn", reason=evidence.reason
            )
        }
    detectable = [probe.detectable for probe in solved.probes]
    fraction = sum(value is True for value in detectable) / len(detectable) if detectable else None
    converged = solved.status == "converged"
    metrics = {
        "radar_spatiotemporal_available": MetricResult(value=1.0, unit="bool", grade="pass"),
        "radar_spatiotemporal_holdout_rmse_mps": MetricResult(
            value=solved.holdout_rmse_mps,
            unit="m/s",
            grade="pass" if converged and solved.holdout_rmse_mps is not None else "warn",
            reason=solved.reason,
        ),
        "radar_spatiotemporal_lever_arm_rank": MetricResult(
            value=float(solved.lever_arm_rank),
            unit="rank",
            grade="pass" if solved.lever_arm_rank == 3 else "warn",
        ),
        "radar_spatiotemporal_joint_rank": MetricResult(
            value=float(solved.joint_rank),
            unit="rank",
            grade="pass" if solved.joint_rank == 4 else "warn",
            reason=solved.reason,
        ),
        "radar_spatiotemporal_joint_condition_number": MetricResult(
            value=solved.joint_condition_number,
            unit="ratio",
            grade="pass" if converged and solved.joint_condition_number is not None else "warn",
            reason=solved.reason,
        ),
        "radar_spatiotemporal_time_translation_coupling": MetricResult(
            value=solved.time_translation_subspace_coupling,
            unit="fraction",
            grade=(
                "pass"
                if converged and solved.time_translation_subspace_coupling is not None
                else "warn"
            ),
            reason=(
                "fraction of the scaled time Jacobian explained by the translation subspace; "
                "one indicates complete local confounding"
            ),
        ),
        "radar_spatiotemporal_time_curvature": MetricResult(
            value=solved.time_objective_curvature_mps2_per_sec2,
            unit="m2/s4",
            grade="pass"
            if solved.time_objective_curvature_mps2_per_sec2 is not None
            and solved.time_objective_curvature_mps2_per_sec2 > 0.0
            else "warn",
        ),
        "radar_spatiotemporal_known_bad_detectable_fraction": MetricResult(
            value=fraction,
            unit="fraction",
            grade="pass" if fraction == 1.0 else "warn",
        ),
    }
    rotation = evidence.rotation_result
    if rotation is not None:
        rotation_detectable = [probe.detectable for probe in rotation.probes]
        rotation_fraction = (
            sum(value is True for value in rotation_detectable) / len(rotation_detectable)
            if rotation_detectable
            else None
        )
        rotation_converged = rotation.status == "converged"
        metrics.update(
            {
                "radar_rotation_holdout_rmse_mps": MetricResult(
                    value=rotation.holdout_rmse_mps,
                    unit="m/s",
                    grade="pass" if rotation_converged else "warn",
                    reason=rotation.reason,
                ),
                "radar_rotation_information_rank": MetricResult(
                    value=float(rotation.information_rank),
                    unit="rank",
                    grade="pass" if rotation.information_rank == 3 else "warn",
                    reason=rotation.reason,
                ),
                "radar_rotation_condition_number": MetricResult(
                    value=rotation.information_condition_number,
                    unit="ratio",
                    grade="pass" if rotation_converged else "warn",
                    reason=rotation.reason,
                ),
                "radar_rotation_known_bad_detectable_fraction": MetricResult(
                    value=rotation_fraction,
                    unit="fraction",
                    grade="pass" if rotation_fraction == 1.0 else "warn",
                ),
            }
        )
    joint = evidence.joint_result
    if joint is not None:
        joint_detectable = [probe.detectable for probe in joint.probes]
        joint_fraction = (
            sum(value is True for value in joint_detectable) / len(joint_detectable)
            if joint_detectable
            else None
        )
        joint_converged = joint.status == "converged"
        common_split = set(joint.train_measurement_ids) == set(
            solved.train_measurement_ids
        ) and set(joint.holdout_measurement_ids) == set(solved.holdout_measurement_ids)
        metrics.update(
            {
                "radar_joint_spatiotemporal_holdout_rmse_mps": MetricResult(
                    value=joint.holdout_rmse_mps,
                    unit="m/s",
                    grade="pass" if joint_converged else "warn",
                    reason=joint.reason,
                ),
                "radar_joint_spatiotemporal_information_rank": MetricResult(
                    value=float(joint.information_rank),
                    unit="rank",
                    grade="pass" if joint.information_rank == 7 else "warn",
                    reason=joint.reason,
                ),
                "radar_joint_spatiotemporal_condition_number": MetricResult(
                    value=joint.information_condition_number,
                    unit="ratio",
                    grade="pass" if joint_converged else "warn",
                    reason="unit-dependent local train Jacobian diagnostic; not covariance",
                ),
                "radar_joint_spatiotemporal_known_bad_detectable_fraction": MetricResult(
                    value=joint_fraction,
                    unit="fraction",
                    grade="pass" if joint_fraction == 1.0 else "warn",
                ),
                "radar_joint_spatiotemporal_common_split_consistent": MetricResult(
                    value=1.0 if common_split else 0.0,
                    unit="bool",
                    grade="pass" if common_split else "warn",
                    reason="joint and staged estimators must use the same eligible IDs and split",
                ),
            }
        )
    return metrics


def _uniform_sample(records: list[Any], maximum: int) -> list[Any]:
    if len(records) <= maximum:
        return records
    return [records[index * len(records) // maximum] for index in range(maximum)]


def _metadata_manifests(root: Path) -> tuple[dict[str, object], ...]:
    candidates = [
        path
        for table in ("ego_pose", "sample_data", "calibrated_sensor", "sensor")
        for path in sorted(root.glob(f"v1.0-*/{table}.json"))
    ]
    return tuple(_manifest(path, "nuscenes_metadata") for path in candidates)


def _manifest(path: Path, role: str) -> dict[str, object]:
    return {
        "path": str(path),
        "role": role,
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
        "source_url": "https://www.nuscenes.org/nuscenes",
    }
