"""A synthetic rig for ``calibrex demo check``: a hand-held IMU and 4D radar with a known truth.

The demo writes a small ROS 2 bag of that rig and two candidate calibrations (the true
mounting, and the same mounting turned a few degrees), so ``calibrex check`` can be seen
passing one and failing the other without a dataset download. The data is simulated: it
shows what the command prints, not how well it does on a real recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from calibrex.data.livox_ros2 import ImuSamples

GRAVITY = np.array([0.0, 0.0, -9.81])
DEMO_PAIR = "radar-imu"
DEMO_TURN_DEG = 4.0
"""The turned candidate is the true mounting turned this much about the radar's pitch axis."""
_START_S = 1.7e9
_RADAR_FIELDS = [("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1), ("v_doppler_mps", 12, 7, 1)]


def _attitude(t: np.ndarray) -> Rotation:
    roll = 0.3 * np.sin(1.1 * t) + 0.1 * np.sin(5.0 * t)
    pitch = 0.25 * np.sin(0.9 * t + 1.0) + 0.1 * np.sin(6.5 * t + 0.5)
    yaw = 0.7 * np.sin(0.3 * t) + 0.3 * np.sin(1.3 * t)
    return Rotation.from_euler("xyz", np.column_stack([roll, pitch, yaw]))


def _position(t: np.ndarray) -> np.ndarray:
    return np.column_stack([1.5 * np.sin(0.4 * t), 1.2 * np.cos(0.5 * t), 0.3 * np.sin(0.9 * t)])


@dataclass(frozen=True)
class Rig:
    imu: ImuSamples
    radar_times_s: np.ndarray  # radar clock
    radar_velocities: np.ndarray  # radar frame
    radar_to_imu: np.ndarray  # R_ir
    lever_m: np.ndarray
    offset_s: float  # t_imu = t_radar + offset
    gyro_bias: np.ndarray


def synthetic_rig(
    *,
    duration_s: float = 120.0,
    offset_s: float = 0.012,
    noise_mps: float = 0.01,
    seed: int = 0,
) -> Rig:
    """A hand-held IMU and radar; the IMU samples at 200 Hz and the radar at 10 Hz."""

    rng = np.random.default_rng(seed)
    radar_to_imu = Rotation.from_euler("xyz", [20.0, -35.0, 110.0], degrees=True).as_matrix()
    lever = np.array([0.05, 0.08, 0.07])
    bias = np.array([0.002, -0.001, 0.0015])
    step = 1e-3
    t = np.arange(0.0, duration_s, 1.0 / 200.0)

    def kinematics(times: np.ndarray) -> tuple[Rotation, np.ndarray, np.ndarray, np.ndarray]:
        attitude = _attitude(times)
        rate = (_attitude(times - step).inv() * _attitude(times + step)).as_rotvec() / (2 * step)
        velocity = (_position(times + step) - _position(times - step)) / (2 * step)
        acceleration = (
            _position(times + step) - 2 * _position(times) + _position(times - step)
        ) / step**2
        return attitude, rate, velocity, acceleration

    attitude, rate, _velocity, acceleration = kinematics(t)
    gyro = rate + bias + rng.normal(0.0, 0.002, rate.shape)
    accel = attitude.inv().apply(acceleration - GRAVITY) + rng.normal(0.0, 0.02, rate.shape)
    radar_stamps = np.arange(1.0, duration_s - 1.0, 0.1)
    when = radar_stamps + offset_s  # the IMU time the radar scan was taken at
    attitude_r, rate_r, velocity_r, _ = kinematics(when)
    world = velocity_r + attitude_r.apply(np.cross(rate_r, lever))
    radar = (attitude_r.inv().apply(world)) @ radar_to_imu  # R_ir^T R_wi^T v
    radar = radar + rng.normal(0.0, noise_mps, radar.shape)
    return Rig(ImuSamples(t, gyro, accel), radar_stamps, radar, radar_to_imu, lever, offset_s, bias)


def write_rig_bag(rig: Rig, directory: Path) -> Path:
    """The rig as a ROS 2 bag: ``/imu`` and a ``/radar/scan`` PointCloud2 with Doppler."""

    from calibrex.data.ros_cdr_writer import encode_imu, encode_pointcloud2
    from calibrex.data.rosbag2_writer import Rosbag2Writer

    rng = np.random.default_rng(5)
    bag = directory / "rig"
    with Rosbag2Writer(bag) as writer:
        writer.add_topic("/imu", "sensor_msgs/msg/Imu")
        writer.add_topic("/radar/scan", "sensor_msgs/msg/PointCloud2")
        for t, g, a in zip(rig.imu.times_s, rig.imu.gyro_rps, rig.imu.accel_mps2, strict=True):
            ns = round((t + _START_S) * 1e9)
            writer.write(
                "/imu",
                ns,
                encode_imu(
                    frame_id="imu",
                    timestamp_ns=ns,
                    angular_velocity=tuple(g),
                    linear_acceleration=tuple(a),
                ),
            )
        for t, v in zip(rig.radar_times_s, rig.radar_velocities, strict=True):
            points = rng.uniform([1.0, -4.0, -2.0], [12.0, 4.0, 2.0], (64, 3))
            directions = points / np.linalg.norm(points, axis=1, keepdims=True)
            doppler = -directions @ v
            data = np.column_stack([points, doppler]).astype("<f4").tobytes()
            ns = round((t + _START_S) * 1e9)
            writer.write(
                "/radar/scan",
                ns,
                encode_pointcloud2(
                    frame_id="radar",
                    timestamp_ns=ns,
                    fields=_RADAR_FIELDS,
                    point_step=16,
                    data=data,
                ),
            )
    return bag


def rig_candidates(rig: Rig, turn_deg: float = DEMO_TURN_DEG) -> dict[str, Rotation]:
    """``T_imu_radar`` rotations: the true one, and it turned about the radar's pitch axis."""

    good = Rotation.from_matrix(rig.radar_to_imu)
    turned = (Rotation.from_euler("y", turn_deg, degrees=True) * good.inv()).inv()
    return {"good": good, "turned": turned}


def write_frames(rig: Rig, rotation: Rotation, path: Path) -> Path:
    """A ``--tf`` frames file with the radar mounted on the IMU by ``rotation``."""

    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [
                    {
                        "name": "radar",
                        "parent": "imu",
                        "translation_m": rig.lever_m.tolist(),
                        "rotation_quat_xyzw": rotation.as_quat().tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_demo(directory: Path) -> tuple[Path, dict[str, Path]]:
    """Write the demo bag and its two candidate files; returns the bag and the files by name."""

    directory.mkdir(parents=True, exist_ok=True)
    rig = synthetic_rig()
    bag = directory / "rig"
    if not (bag / "metadata.yaml").exists():
        write_rig_bag(rig, directory)
    frames = {
        name: write_frames(rig, rotation, directory / f"{name}.yaml")
        for name, rotation in rig_candidates(rig).items()
    }
    return bag, frames
