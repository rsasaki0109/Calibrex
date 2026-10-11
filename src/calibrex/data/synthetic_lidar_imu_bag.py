"""A synthetic rosbag2 of a spinning LiDAR and an IMU moving inside a room.

The bag exercises the LiDAR-IMU bag front end end to end with a known truth:
a 16-beam spinning LiDAR (``sensor_msgs/PointCloud2`` with a per-point
``offset_time`` in seconds after the header stamp) ray-casts a box room plus
two slanted walls while the rig follows a smooth continuous trajectory, and a
``sensor_msgs/Imu`` reports gyro and specific force in its own frame.

Conventions match the joint LiDAR-IMU calibration: the body frame is the
LiDAR frame, ``R_lidar_imu`` maps IMU vectors into the LiDAR frame, the IMU
origin sits at ``lever_arm`` in the LiDAR frame, and IMU stamps run ahead of
the LiDAR clock by ``clock_offset_s`` (``t_imu = t_lidar + clock_offset_s``).
Every value is synthetic; nothing here is evidence about a real sensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.se3_manifold import (
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_matrix,
    so3_left_jacobian,
)
from calibrex.data.ros_cdr_writer import POINT_FIELD_FLOAT32, encode_imu, encode_pointcloud2
from calibrex.data.rosbag2 import IMU_TYPE, POINTCLOUD2_TYPE
from calibrex.data.rosbag2_writer import Rosbag2Writer

FloatArray: TypeAlias = NDArray[np.float64]

LIDAR_TOPIC = "/lidar/points"
IMU_TOPIC = "/imu/data"
_START_NS = 1_700_000_000_000_000_000
_SLANTED_WALLS: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((3.5, 3.0, 0.0), (-0.6, -0.8, 0.0)),
    ((-3.0, -3.5, 0.0), (0.5, 0.6, -0.62)),
)


@dataclass(frozen=True)
class SyntheticLidarImuTruth:
    """The injected IMU calibration relative to the LiDAR."""

    imu_rotation_vector_rad: tuple[float, float, float] = (0.04, -0.03, 0.05)
    gyro_bias_rad_s: tuple[float, float, float] = (0.008, -0.006, 0.01)
    lever_arm_m: tuple[float, float, float] = (0.10, -0.06, 0.15)
    clock_offset_s: float = 0.007
    accel_bias_m_s2: tuple[float, float, float] = (0.06, -0.05, 0.04)
    gravity_world_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(frozen=True)
class SyntheticLidarImuBagSpec:
    """Recording length, sensor rates and noise of the synthetic bag."""

    duration_s: float = 12.0
    lidar_rate_hz: float = 10.0
    azimuth_steps: int = 360
    beam_elevations_deg: tuple[float, ...] = tuple(np.linspace(-15.0, 15.0, 16).tolist())
    range_noise_m: float = 0.01
    imu_rate_hz: float = 200.0
    gyro_noise_rad_s: float = 0.003
    accel_noise_m_s2: float = 0.03
    room_half_extent_m: tuple[float, float, float] = (6.0, 5.0, 2.0)
    room_center_m: tuple[float, float, float] = (0.0, 0.0, 0.5)
    # A roll/pitch vibration of the whole rig (frequency in Hz, angular-rate
    # amplitude in rad/s), as on a wheeled platform: far above what the
    # trajectory spline represents, so the gyro and the accelerometer see it as
    # structured noise while the LiDAR barely does.
    vibration_hz: float = 0.0
    vibration_rate_rad_s: float = 0.0
    seed: int = 20261011
    truth: SyntheticLidarImuTruth = field(default_factory=SyntheticLidarImuTruth)


@dataclass(frozen=True)
class SyntheticLidarImuBag:
    """Where the bag was written and what it contains."""

    path: Path
    lidar_topic: str
    imu_topic: str
    lidar_frame: str
    imu_frame: str
    start_time_s: float
    spec: SyntheticLidarImuBagSpec


def smooth_rotation_vector(time_s: float) -> FloatArray:
    """Return the rotation vector of ``R_world_lidar`` (without vibration)."""

    return np.array(
        [
            0.5 * math.sin(1.3 * time_s),
            0.45 * math.sin(1.1 * time_s + 0.4),
            1.0 * math.sin(0.6 * time_s),
        ]
    )


def smooth_position(time_s: float) -> FloatArray:
    """Return the LiDAR origin in the world at a time after the start."""

    return np.array(
        [
            0.6 * math.sin(0.5 * time_s) + 0.05 * time_s,
            0.5 * math.sin(0.35 * time_s + 0.2),
            0.15 * math.sin(0.8 * time_s),
        ]
    )


def _rotation_vector_rate(time_s: float) -> FloatArray:
    return np.array(
        [
            0.5 * 1.3 * math.cos(1.3 * time_s),
            0.45 * 1.1 * math.cos(1.1 * time_s + 0.4),
            1.0 * 0.6 * math.cos(0.6 * time_s),
        ]
    )


def _acceleration(time_s: float) -> FloatArray:
    return np.array(
        [
            -0.6 * 0.25 * math.sin(0.5 * time_s),
            -0.5 * 0.35**2 * math.sin(0.35 * time_s + 0.2),
            -0.15 * 0.64 * math.sin(0.8 * time_s),
        ]
    )


def smooth_body_rate(time_s: float) -> FloatArray:
    """Return the LiDAR-frame angular velocity at a time after the start."""

    # ``so3_left_jacobian`` satisfies Exp(w + d) ~= Exp(w) Exp(V d): the body
    # rate of R = Exp(phi(t)) is V(phi) phi'.
    return np.asarray(
        so3_left_jacobian(smooth_rotation_vector(time_s)) @ _rotation_vector_rate(time_s),
        dtype=float,
    )


def _vibration(time_s: float, spec: SyntheticLidarImuBagSpec) -> FloatArray:
    """Return the body-frame vibration rotation vector (roll and pitch, out of phase)."""

    if spec.vibration_hz <= 0.0 or spec.vibration_rate_rad_s == 0.0:
        return np.zeros(3)
    angular = 2.0 * math.pi * spec.vibration_hz
    amplitude = spec.vibration_rate_rad_s / angular
    return amplitude * np.array(
        [math.sin(angular * time_s), math.sin(angular * time_s + 1.1), 0.0]
    )


def _rig_rotation(time_s: float, spec: SyntheticLidarImuBagSpec) -> FloatArray:
    return np.asarray(
        rotation_vector_to_matrix(smooth_rotation_vector(time_s))
        @ rotation_vector_to_matrix(_vibration(time_s, spec)),
        dtype=float,
    )


def _rig_body_rate(time_s: float, spec: SyntheticLidarImuBagSpec) -> FloatArray:
    """Body angular velocity of the vibrating rig, by central differences of R."""

    step = 1.0e-6
    before = _rig_rotation(time_s - step, spec)
    after = _rig_rotation(time_s + step, spec)
    return np.asarray(
        rotation_matrix_to_rotation_vector(before.T @ after) / (2.0 * step), dtype=float
    )


def write_synthetic_lidar_imu_bag(
    directory: str | Path,
    spec: SyntheticLidarImuBagSpec | None = None,
    *,
    overwrite: bool = False,
) -> SyntheticLidarImuBag:
    """Write the synthetic LiDAR + IMU rosbag2 and return its description."""

    settings = spec or SyntheticLidarImuBagSpec()
    truth = settings.truth
    rng = np.random.default_rng(settings.seed)
    path = Path(directory)
    with Rosbag2Writer(
        path,
        custom_data={"generator": __name__, "seed": str(settings.seed)},
        overwrite=overwrite,
    ) as writer:
        writer.add_topic(LIDAR_TOPIC, POINTCLOUD2_TYPE)
        writer.add_topic(IMU_TOPIC, IMU_TYPE)
        period = 1.0 / settings.lidar_rate_hz
        scan_starts = np.arange(0.0, settings.duration_s - period + 1.0e-9, period)
        imu_times = np.arange(0.0, settings.duration_s, 1.0 / settings.imu_rate_hz)
        events: list[tuple[float, str, bytes]] = []
        for start in scan_starts:
            events.append((float(start), LIDAR_TOPIC, _scan_payload(float(start), settings, rng)))
        for time in imu_times:
            stamp = float(time) + truth.clock_offset_s
            events.append((stamp, IMU_TOPIC, _imu_payload(float(time), stamp, settings, rng)))
        for stamp, topic, payload in sorted(events, key=lambda item: item[0]):
            writer.write(topic, _START_NS + round(stamp * 1.0e9), payload)
    return SyntheticLidarImuBag(
        path=path,
        lidar_topic=LIDAR_TOPIC,
        imu_topic=IMU_TOPIC,
        lidar_frame="lidar",
        imu_frame="imu",
        start_time_s=_START_NS * 1.0e-9,
        spec=settings,
    )


def _scan_payload(start: float, spec: SyntheticLidarImuBagSpec, rng: np.random.Generator) -> bytes:
    period = 1.0 / spec.lidar_rate_hz
    steps = spec.azimuth_steps
    elevations = np.radians(np.asarray(spec.beam_elevations_deg, dtype=float))
    azimuths = 2.0 * math.pi * np.arange(steps) / steps
    offsets = period * np.arange(steps) / steps
    points: list[FloatArray] = []
    times: list[FloatArray] = []
    for azimuth, offset in zip(azimuths, offsets, strict=True):
        time = start + float(offset)
        rotation = _rig_rotation(time, spec)
        origin = smooth_position(time)
        directions = np.stack(
            [
                np.cos(elevations) * math.cos(azimuth),
                np.cos(elevations) * math.sin(azimuth),
                np.sin(elevations),
            ],
            axis=1,
        )
        ranges = _ray_ranges(origin, directions @ rotation.T, spec)
        ranges = ranges + rng.normal(0.0, spec.range_noise_m, ranges.shape)
        points.append(directions * ranges[:, None])
        times.append(np.full(ranges.shape, offset))
    xyz = np.concatenate(points)
    record = np.zeros(
        xyz.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("offset_time", "<f4")]
    )
    record["x"], record["y"], record["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    record["offset_time"] = np.concatenate(times)
    return encode_pointcloud2(
        frame_id="lidar",
        timestamp_ns=_START_NS + round(start * 1.0e9),
        fields=(
            ("x", 0, POINT_FIELD_FLOAT32, 1),
            ("y", 4, POINT_FIELD_FLOAT32, 1),
            ("z", 8, POINT_FIELD_FLOAT32, 1),
            ("offset_time", 12, POINT_FIELD_FLOAT32, 1),
        ),
        point_step=16,
        data=record.tobytes(),
    )


def _ray_ranges(
    origin: FloatArray, directions: FloatArray, spec: SyntheticLidarImuBagSpec
) -> FloatArray:
    center = np.asarray(spec.room_center_m, dtype=float)
    half = np.asarray(spec.room_half_extent_m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.where(directions > 0.0, center + half, center - half)
        distances = (bound - origin) / directions
    distances = np.where(np.isfinite(distances) & (distances > 0.0), distances, np.inf)
    ranges = np.min(distances, axis=1)
    for point, normal in _SLANTED_WALLS:
        unit = np.asarray(normal, dtype=float) / float(np.linalg.norm(normal))
        facing = directions @ unit
        with np.errstate(divide="ignore", invalid="ignore"):
            hit = (np.asarray(point) - origin) @ unit / facing
        valid = (facing < -1.0e-6) & (hit > 0.0)
        ranges = np.where(valid & (hit < ranges), hit, ranges)
    return np.asarray(ranges, dtype=float)


def _imu_payload(
    time: float, stamp: float, spec: SyntheticLidarImuBagSpec, rng: np.random.Generator
) -> bytes:
    truth = spec.truth
    imu_rotation = rotation_vector_to_matrix(np.asarray(truth.imu_rotation_vector_rad))
    lever = np.asarray(truth.lever_arm_m, dtype=float)
    vibrating = spec.vibration_hz > 0.0 and spec.vibration_rate_rad_s != 0.0
    if vibrating:
        omega = _rig_body_rate(time, spec)
        step = 1.0e-4
        alpha = (_rig_body_rate(time + step, spec) - _rig_body_rate(time - step, spec)) / (
            2.0 * step
        )
    else:
        omega = smooth_body_rate(time)
        step = 1.0e-5
        alpha = (smooth_body_rate(time + step) - smooth_body_rate(time - step)) / (2.0 * step)
    rotation = _rig_rotation(time, spec)
    kinematic = (
        rotation.T @ (_acceleration(time) - np.asarray(truth.gravity_world_m_s2))
        + np.cross(alpha, lever)
        + np.cross(omega, np.cross(omega, lever))
    )
    gyro = (
        imu_rotation.T @ omega
        + np.asarray(truth.gyro_bias_rad_s)
        + rng.normal(0.0, spec.gyro_noise_rad_s, 3)
    )
    accel = (
        imu_rotation.T @ kinematic
        + np.asarray(truth.accel_bias_m_s2)
        + rng.normal(0.0, spec.accel_noise_m_s2, 3)
    )
    return encode_imu(
        frame_id="imu",
        timestamp_ns=_START_NS + round(stamp * 1.0e9),
        angular_velocity=(float(gyro[0]), float(gyro[1]), float(gyro[2])),
        linear_acceleration=(float(accel[0]), float(accel[1]), float(accel[2])),
    )
