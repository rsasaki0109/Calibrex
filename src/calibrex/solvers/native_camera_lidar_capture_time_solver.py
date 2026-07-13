"""Native adapter for public Camera-LiDAR per-point capture-time evidence."""

from __future__ import annotations

import bisect
import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.capture_time import (
    ConstantBodyTwist,
    LidarCaptureTimePolicy,
    TimedLidarPoint,
)
from calibrex.core.config import CalibrationConfig, FactorConfig
from calibrex.core.exceptions import DatasetError
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.provenance import sha256_path
from calibrex.core.result import MetricResult, ObservabilityResult
from calibrex.data.inspect import DatasetInspection
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.data.rosbag1 import POINTCLOUD2_TYPE, decode_pointcloud2, iter_messages
from calibrex.solvers.base import SolverAdapter, SolverAdapterResult
from calibrex.solvers.camera_lidar_capture_time_solver import (
    CameraLidarCaptureTimeOptions,
    CameraLidarCaptureTimeSolver,
    CameraLidarTimedCapture,
)

NATIVE_CAMERA_LIDAR_CAPTURE_TIME_BACKEND = "native_camera_lidar_capture_time"
TIERS_LIDARS_CALI_SOURCE_URL = (
    "https://utufi.sharepoint.com/:u:/s/msteams_0ed7e9/"
    "Ea0qTMxHxR5GsHMX62HRjFMBxpdrOrp9fMSfKkxp2e5DAg?e=HmjOoT"
)
TIERS_DATASET_REPOSITORY_URL = "https://github.com/TIERS/tiers-lidars-dataset"
TIERS_LIDARS_CALI_SIZE_BYTES = 7_178_470_523
TIERS_LIDARS_CALI_FIRST_MIB_SHA256 = (
    "aa1f477d43e1d24236aed0f0860eadf4886efbf1f129c31b0a477eb54ab780f8"
)
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class _RawTimedScan:
    timestamp_ns: int
    points: tuple[TimedLidarPoint, ...]
    point_time_min_sec: float
    point_time_max_sec: float


@dataclass(frozen=True)
class _Extraction:
    captures: tuple[CameraLidarTimedCapture, ...]
    camera_message_count: int
    lidar_message_count: int
    pose_message_count: int
    paired_capture_count: int
    rejected_pair_count: int
    sampled_point_count: int
    point_time_field: str
    camera_topic: str
    lidar_topic: str
    pose_topic: str
    motion_proxy: str

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_message_count": self.camera_message_count,
            "lidar_message_count": self.lidar_message_count,
            "pose_message_count": self.pose_message_count,
            "paired_capture_count": self.paired_capture_count,
            "rejected_pair_count": self.rejected_pair_count,
            "sampled_point_count": self.sampled_point_count,
            "point_time_field": self.point_time_field,
            "camera_topic": self.camera_topic,
            "lidar_topic": self.lidar_topic,
            "pose_topic": self.pose_topic,
            "motion_proxy": self.motion_proxy,
            "selection": "uniform over complete LiDAR scan sequence after temporal pairing",
            "camera_pairing": "nearest camera header exposure timestamp",
        }


class NativeCameraLidarCaptureTimeSolver(SolverAdapter):
    """Apply measured LiDAR firing times at real camera exposure instants."""

    backend = NATIVE_CAMERA_LIDAR_CAPTURE_TIME_BACKEND

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        del frame_graph, inspection
        factor = _factor(config)
        options = factor.options if factor is not None else {}
        bag_path = Path(config.dataset.path)
        if not bag_path.exists():
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status="missing_dataset",
                metrics={
                    "camera_lidar_capture_time_input_available": MetricResult(
                        value=0.0,
                        unit="bool",
                        grade="fail",
                        reason=f"public ROS bag is missing: {bag_path}",
                    )
                },
                warnings=[f"public ROS bag is missing: {bag_path}"],
            )
        try:
            extraction = _extract_timed_captures(bag_path, options)
        except (DatasetError, OSError, ValueError) as exc:
            return SolverAdapterResult(
                backend=self.backend,
                available=True,
                status="invalid_capture_time_input",
                warnings=[str(exc)],
            )

        policy = LidarCaptureTimePolicy(
            point_offset_unit="seconds",
            stamp_reference="scan_end",
            sensor_time_offset_sec=0.0,
        )
        result = CameraLidarCaptureTimeSolver().solve(
            extraction.captures,
            policy,
            CameraLidarCaptureTimeOptions(
                min_train_captures=_int_option(options, "min_train_captures", 8),
                min_holdout_captures=_int_option(options, "min_holdout_captures", 3),
                holdout_ratio=config.evaluation.holdout_ratio,
                split_seed=config.solver.seed or 0,
                sensitivity_step_sec=_float_option(options, "sensitivity_step_sec", 0.001),
                min_time_sensitivity_mps=_float_option(options, "min_time_sensitivity_mps", 0.05),
                known_bad_margin_m=_float_option(options, "known_bad_margin_m", 0.001),
            ),
        )
        holdout = result.holdout_evaluation
        detectable = sum(probe.detectable is True for probe in result.probes)
        probe_count = len(result.probes)
        detectable_fraction = detectable / probe_count if probe_count else None
        evidence_pass = (
            result.status == "evidence_only"
            and result.time_observability_rank == 1
            and holdout.capture_count >= _int_option(options, "min_holdout_captures", 3)
            and holdout.mean_point_time_span_sec is not None
            and holdout.mean_point_time_span_sec
            >= _float_option(options, "min_point_time_span_sec", 0.08)
            and holdout.mean_point_time_span_sec
            <= _float_option(options, "max_point_time_span_sec", 0.12)
            and holdout.mean_abs_camera_to_lidar_stamp_sec is not None
            and holdout.mean_abs_camera_to_lidar_stamp_sec
            <= _float_option(options, "max_camera_lidar_stamp_delta_sec", 0.06)
            and result.time_sensitivity_rms_mps is not None
            and result.time_sensitivity_rms_mps
            >= _float_option(options, "min_time_sensitivity_mps", 0.05)
            and detectable_fraction is not None
            and detectable_fraction
            >= _float_option(options, "min_known_bad_detectable_fraction", 0.8)
        )
        metrics = {
            "camera_lidar_capture_time_input_available": MetricResult(
                value=1.0, unit="bool", grade="pass"
            ),
            "camera_lidar_capture_pair_count": MetricResult(
                value=float(len(extraction.captures)),
                unit="captures",
                grade="pass" if evidence_pass else "warn",
            ),
            "camera_lidar_capture_point_count": MetricResult(
                train=float(result.train_evaluation.point_count),
                holdout=float(holdout.point_count),
                unit="points",
                grade="pass" if evidence_pass else "warn",
            ),
            "camera_lidar_point_time_span_s": MetricResult(
                train=result.train_evaluation.mean_point_time_span_sec,
                holdout=holdout.mean_point_time_span_sec,
                unit="s",
                grade="pass" if evidence_pass else "warn",
            ),
            "camera_lidar_camera_stamp_delta_ms": MetricResult(
                train=_milliseconds(result.train_evaluation.mean_abs_camera_to_lidar_stamp_sec),
                holdout=_milliseconds(holdout.mean_abs_camera_to_lidar_stamp_sec),
                unit="ms",
                grade="pass" if evidence_pass else "warn",
            ),
            "camera_lidar_point_to_exposure_delta_ms": MetricResult(
                train=_milliseconds(result.train_evaluation.mean_abs_point_to_camera_time_sec),
                holdout=_milliseconds(holdout.mean_abs_point_to_camera_time_sec),
                unit="ms",
                grade="warn",
                reason="measured timing geometry; no accuracy threshold is declared",
            ),
            "camera_lidar_deskew_displacement_rmse_m": MetricResult(
                train=result.train_evaluation.deskew_displacement_rmse_m,
                holdout=holdout.deskew_displacement_rmse_m,
                unit="m",
                grade="warn",
                reason="motion-induced correction magnitude, not alignment error",
            ),
            "camera_lidar_time_sensitivity_mps": MetricResult(
                value=result.time_sensitivity_rms_mps,
                unit="m/s",
                grade="pass" if result.time_observability_rank == 1 else "warn",
            ),
            "camera_lidar_time_observability_rank": MetricResult(
                value=float(result.time_observability_rank),
                unit="rank",
                grade="pass" if result.time_observability_rank == 1 else "warn",
            ),
            "camera_lidar_time_known_bad_detectable_fraction": MetricResult(
                value=detectable_fraction,
                unit="fraction",
                grade="pass" if evidence_pass else "warn",
                reason=f"{detectable}/{probe_count} signed held-out time controls detected",
            ),
            "camera_lidar_time_offset_estimated": MetricResult(
                value=0.0,
                unit="bool",
                grade="warn",
                reason=(
                    "TIERS bag has firing times and motion but no declared cross-modal "
                    "point identities; no clock offset is estimated"
                ),
            ),
        }
        warnings = [
            "public TIERS run is timing-application evidence only; no camera-LiDAR "
            "clock offset is estimated without cross-modal geometric references",
            "VRPN UWBTest pose is used as a rig-motion proxy; its constant alignment "
            "to the Velodyne frame is not published, so deskew displacement is diagnostic",
        ]
        first_mib_sha256 = _sha256_prefix(bag_path, 1_048_576)
        input_verified = (
            bag_path.stat().st_size == TIERS_LIDARS_CALI_SIZE_BYTES
            and first_mib_sha256 == TIERS_LIDARS_CALI_FIRST_MIB_SHA256
        )
        raw_input = {
            "path": str(bag_path),
            "role": "camera_lidar_point_time_and_motion_rosbag1",
            "sha256": sha256_path(bag_path),
            "size_bytes": bag_path.stat().st_size,
            "expected_size_bytes": TIERS_LIDARS_CALI_SIZE_BYTES,
            "sha256_first_mib": first_mib_sha256,
            "expected_sha256_first_mib": TIERS_LIDARS_CALI_FIRST_MIB_SHA256,
            "source_url": TIERS_LIDARS_CALI_SOURCE_URL,
            "dataset_repository": TIERS_DATASET_REPOSITORY_URL,
            "license_spdx": "MIT",
        }
        return SolverAdapterResult(
            backend=self.backend,
            available=True,
            status="pass" if evidence_pass else "inconclusive",
            metrics=metrics,
            observability=ObservabilityResult(
                rank=result.time_observability_rank,
                condition_number=result.time_condition_number,
                weak_directions=(
                    [] if result.time_observability_rank == 1 else ["lidar_time_offset"]
                ),
                grade="pass" if result.time_observability_rank == 1 else "warn",
            ),
            provenance={
                "metrics_origin": "recomputed",
                "data_verified": input_verified,
                "raw_input_files": [raw_input],
                "native_camera_lidar_capture_time": {
                    "result": result.as_dict(),
                    "evidence_pass": evidence_pass,
                    "extraction": extraction.as_dict(),
                    "dataset": {
                        "name": "TIERS LidarsCali",
                        "license_spdx": "MIT",
                        "source_url": TIERS_LIDARS_CALI_SOURCE_URL,
                        "repository_url": TIERS_DATASET_REPOSITORY_URL,
                    },
                    "capture_time_semantics": {
                        "camera_reference": (
                            "color image header timestamp used as the exposure reference"
                        ),
                        "lidar_stamp_reference": "scan_end",
                        "point_offset": "measured Velodyne PointCloud2 `time` seconds",
                        "clock_convention": (
                            "lidar_sensor_time + dt_lidar = camera_reference_time"
                        ),
                    },
                    "motion_proxy_limitation": (
                        "VRPN UWBTest T_world_rig supplies local twist; unpublished "
                        "constant T_rig_velodyne prevents interpreting displacement as "
                        "absolute geometric accuracy"
                    ),
                    "spatial_transform_role": (
                        "the configured identity T_camera0_lidar0 is a frame-graph "
                        "placeholder; it is not consumed, estimated, or emitted as a "
                        "calibration result by this solver"
                    ),
                    "external_code_executed": False,
                },
            },
            warnings=warnings,
        )


def _extract_timed_captures(path: Path, options: dict[str, Any]) -> _Extraction:
    camera_topic = str(options.get("camera_topic", "/cam_1/color/image_raw"))
    lidar_topic = str(options.get("lidar_topic", "/velodyne_points"))
    pose_topic = str(options.get("pose_topic", "/vrpn_client_node/UWBTest/pose"))
    point_time_field = str(options.get("point_time_field", "time"))
    max_captures = _int_option(options, "max_captures", 24)
    max_points = _int_option(options, "max_points_per_capture", 128)
    min_range = _float_option(options, "min_point_range_m", 0.5)
    max_range = _float_option(options, "max_point_range_m", 30.0)
    max_pair_delta_ns = round(
        _float_option(options, "max_camera_lidar_stamp_delta_sec", 0.06) * 1.0e9
    )
    twist_window_ns = round(_float_option(options, "twist_window_sec", 0.04) * 1.0e9)
    camera_timestamps: list[int] = []
    poses: list[OdometryPoseSample] = []
    scans: list[_RawTimedScan] = []
    topics = {camera_topic, lidar_topic, pose_topic}
    for connection, bag_timestamp_ns, data in iter_messages(path, topics=topics):
        if connection.topic == camera_topic:
            timestamp_ns, _frame, _offset = _decode_ros1_header(data, bag_timestamp_ns)
            camera_timestamps.append(timestamp_ns)
        elif connection.topic == pose_topic:
            poses.append(_decode_pose_stamped(data, bag_timestamp_ns))
        elif connection.topic == lidar_topic:
            if connection.message_type != POINTCLOUD2_TYPE:
                raise ValueError(
                    f"{lidar_topic} must be sensor_msgs/PointCloud2, got {connection.message_type}"
                )
            message = decode_pointcloud2(
                lidar_topic,
                bag_timestamp_ns,
                data,
                point_time_field=point_time_field,
            )
            offsets = message.point_time_offsets_s
            if offsets is None:
                raise ValueError(f"{lidar_topic} did not decode measured point times")
            scans.append(
                _sample_scan(
                    message.timestamp_ns,
                    message.xyz,
                    offsets,
                    max_points,
                    min_range,
                    max_range,
                )
            )
    camera_timestamps = sorted(set(camera_timestamps))
    poses = sorted(poses, key=lambda sample: sample.timestamp_ns)
    scans = sorted(scans, key=lambda scan: scan.timestamp_ns)
    if not camera_timestamps or not scans or len(poses) < 2:
        raise ValueError(
            "TIERS capture-time extraction requires camera, timed LiDAR, and pose messages"
        )
    selected = [scans[index] for index in _uniform_indices(len(scans), max_captures)]
    track = OdometryTrack(poses)
    captures: list[CameraLidarTimedCapture] = []
    rejected = 0
    for scan in selected:
        camera_timestamp_ns = _nearest_timestamp(camera_timestamps, scan.timestamp_ns)
        if abs(camera_timestamp_ns - scan.timestamp_ns) > max_pair_delta_ns:
            rejected += 1
            continue
        twist = _twist_at(track, camera_timestamp_ns, twist_window_ns)
        if twist is None:
            rejected += 1
            continue
        captures.append(
            CameraLidarTimedCapture(
                capture_id=f"tiers-lidars-cali-{scan.timestamp_ns}",
                lidar_message_stamp_sec=scan.timestamp_ns * 1.0e-9,
                camera_exposure_time_sec=camera_timestamp_ns * 1.0e-9,
                points=scan.points,
                twist=twist,
                transform_body_lidar=SE3.identity(),
            )
        )
    return _Extraction(
        captures=tuple(captures),
        camera_message_count=len(camera_timestamps),
        lidar_message_count=len(scans),
        pose_message_count=len(poses),
        paired_capture_count=len(captures),
        rejected_pair_count=rejected,
        sampled_point_count=sum(len(capture.points) for capture in captures),
        point_time_field=point_time_field,
        camera_topic=camera_topic,
        lidar_topic=lidar_topic,
        pose_topic=pose_topic,
        motion_proxy="VRPN UWBTest local constant twist with T_proxy_velodyne=identity",
    )


def _sample_scan(
    timestamp_ns: int,
    xyz: FloatArray,
    offsets: FloatArray,
    max_points: int,
    min_range_m: float,
    max_range_m: float,
) -> _RawTimedScan:
    ranges = np.linalg.norm(xyz, axis=1)
    valid = np.flatnonzero(
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(offsets)
        & (ranges >= min_range_m)
        & (ranges <= max_range_m)
    )
    if not len(valid):
        raise ValueError(f"LiDAR scan {timestamp_ns} has no finite timed points")
    chosen = valid[_uniform_indices(len(valid), max_points)]
    points = tuple(
        TimedLidarPoint(
            point_id=f"{timestamp_ns}:{int(index)}",
            position_lidar_m=(
                float(xyz[index, 0]),
                float(xyz[index, 1]),
                float(xyz[index, 2]),
            ),
            capture_offset=float(offsets[index]),
        )
        for index in chosen
    )
    return _RawTimedScan(
        timestamp_ns,
        points,
        float(np.min(offsets[valid])),
        float(np.max(offsets[valid])),
    )


def _decode_ros1_header(data: bytes, fallback_ns: int) -> tuple[int, str, int]:
    if len(data) < 16:
        raise ValueError("truncated ROS 1 message header")
    _sequence, seconds, nanoseconds, frame_length = struct.unpack_from("<IIII", data, 0)
    end = 16 + frame_length
    if end > len(data):
        raise ValueError("truncated ROS 1 message frame_id")
    frame_id = data[16:end].decode("utf-8", errors="replace")
    timestamp_ns = int(seconds) * 1_000_000_000 + int(nanoseconds)
    return (timestamp_ns if timestamp_ns else fallback_ns, frame_id, end)


def _decode_pose_stamped(data: bytes, fallback_ns: int) -> OdometryPoseSample:
    timestamp_ns, _frame_id, offset = _decode_ros1_header(data, fallback_ns)
    if offset + 56 > len(data):
        raise ValueError("truncated geometry_msgs/PoseStamped payload")
    values = struct.unpack_from("<7d", data, offset)
    return OdometryPoseSample(
        timestamp_ns,
        SE3(
            (float(values[0]), float(values[1]), float(values[2])),
            (float(values[3]), float(values[4]), float(values[5]), float(values[6])),
        ),
    )


def _twist_at(
    track: OdometryTrack,
    timestamp_ns: int,
    window_ns: int,
) -> ConstantBodyTwist | None:
    half = max(1, window_ns // 2)
    left, left_clamped, _ = track.interpolate(timestamp_ns - half)
    right, right_clamped, _ = track.interpolate(timestamp_ns + half)
    if left_clamped or right_clamped:
        return None
    duration = 2.0 * half * 1.0e-9
    relative = left.inverse().compose(right)
    rotation = _rotation_vector(relative)
    return ConstantBodyTwist(
        (
            relative.translation_m[0] / duration,
            relative.translation_m[1] / duration,
            relative.translation_m[2] / duration,
        ),
        (
            rotation[0] / duration,
            rotation[1] / duration,
            rotation[2] / duration,
        ),
    )


def _rotation_vector(transform: SE3) -> tuple[float, float, float]:
    x, y, z, w = transform.rotation_quat_xyzw
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1.0e-15:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, w))
    scale = angle / vector_norm
    return (scale * x, scale * y, scale * z)


def _nearest_timestamp(timestamps: list[int], query: int) -> int:
    index = bisect.bisect_left(timestamps, query)
    candidates = timestamps[max(0, index - 1) : min(len(timestamps), index + 1)]
    return min(candidates, key=lambda value: (abs(value - query), value))


def _uniform_indices(count: int, maximum: int) -> NDArray[np.int64]:
    if count <= 0 or maximum <= 0:
        return np.empty(0, dtype=np.int64)
    retained = min(count, maximum)
    if retained == 1:
        return np.asarray([0], dtype=np.int64)
    return np.asarray(
        [index * (count - 1) // (retained - 1) for index in range(retained)],
        dtype=np.int64,
    )


def _sha256_prefix(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(size))
    return digest.hexdigest()


def _factor(config: CalibrationConfig) -> FactorConfig | None:
    return config.pipeline.factors.get("camera_lidar_capture_time")


def _float_option(options: dict[str, Any], name: str, default: float) -> float:
    return float(options.get(name, default))


def _int_option(options: dict[str, Any], name: str, default: int) -> int:
    return int(options.get(name, default))


def _milliseconds(value: float | None) -> float | None:
    return value * 1000.0 if value is not None else None
