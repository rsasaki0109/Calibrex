"""The radar-IMU rotation from Doppler ego velocity: synthetic rigs with a known truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from calibrex.data.livox_ros2 import ImuSamples
from calibrex.data.radar_pointcloud import doppler_field
from calibrex.evaluation.radar_imu_rotation import (
    RadarImuOptions,
    RadarVelocities,
    ego_velocity,
    evaluate_radar_imu_rotation,
)

GRAVITY = np.array([0.0, 0.0, -9.81])


def _attitude(t: np.ndarray) -> Rotation:
    roll = 0.3 * np.sin(1.1 * t) + 0.1 * np.sin(5.0 * t)
    pitch = 0.25 * np.sin(0.9 * t + 1.0) + 0.1 * np.sin(6.5 * t + 0.5)
    yaw = 0.7 * np.sin(0.3 * t) + 0.3 * np.sin(1.3 * t)
    return Rotation.from_euler("xyz", np.column_stack([roll, pitch, yaw]))


def _position(t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [1.5 * np.sin(0.4 * t), 1.2 * np.cos(0.5 * t), 0.3 * np.sin(0.9 * t)]
    )


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
    return Rig(
        ImuSamples(t, gyro, accel), radar_stamps, radar, radar_to_imu, lever, offset_s, bias
    )


def test_ego_velocity_ignores_moving_returns() -> None:
    rng = np.random.default_rng(3)
    velocity = np.array([1.2, -0.4, 0.3])
    points = rng.uniform([2.0, -3.0, -1.0], [10.0, 3.0, 2.0], (80, 3))
    directions = points / np.linalg.norm(points, axis=1, keepdims=True)
    doppler = -directions @ velocity + rng.normal(0.0, 0.02, 80)
    doppler[:15] += rng.uniform(1.0, 3.0, 15)  # a walking crowd

    found = ego_velocity(points, doppler, rng, RadarImuOptions())

    assert found is not None
    assert found[0] == pytest.approx(velocity, abs=0.03)
    assert found[1] >= 60


def test_the_rotation_clock_offset_and_lever_arm_are_recovered() -> None:
    rig = synthetic_rig()
    velocities = RadarVelocities(rig.radar_times_s, rig.radar_velocities, len(rig.radar_times_s))

    found = evaluate_radar_imu_rotation(
        velocities,
        rig.imu,
        reference_rotation=rig.radar_to_imu.T,
        gyro_bias=rig.gyro_bias + np.array([0.0008, 0.0, -0.0005]),  # a slightly wrong bias
    )

    assert found.rotation is not None
    error = Rotation.from_matrix(found.rotation @ rig.radar_to_imu.T).magnitude()
    assert np.degrees(error) < 1.0
    assert found.offset_s == pytest.approx(rig.offset_s, abs=0.003)
    assert found.lever_arm_m == pytest.approx(rig.lever_m, abs=0.02)
    assert found.gyro_bias_rps == pytest.approx(rig.gyro_bias, abs=6e-4)
    roll, pitch, yaw, offset = found.records
    assert offset.name == "time_offset" and offset.unit == "s"
    for record in (roll, pitch, yaw):
        assert record.status == "estimated"
        assert abs(record.error_to_reference) < 3.0 * record.std_reported + 0.1
        assert record.known_bad_control is not None
    # a 1 deg control is one-sided: it can go undetected when the held-out optimum lies
    # the other way, which makes the policy warn rather than pass
    assert found.policy_status in {"pass", "warn"}


def test_too_short_a_recording_is_inconclusive() -> None:
    rig = synthetic_rig(duration_s=30.0)
    velocities = RadarVelocities(rig.radar_times_s, rig.radar_velocities, len(rig.radar_times_s))

    found = evaluate_radar_imu_rotation(velocities, rig.imu)

    assert found.rotation is None and found.policy_status == "inconclusive"
    assert "at least 5" in found.policy_reasons[0]


def test_the_doppler_field_is_found_by_name_and_float_type() -> None:
    def fields(*items: tuple[str, int]) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name, datatype=kind) for name, kind in items]

    assert doppler_field(fields(("x", 7), ("v_doppler_mps", 7))) == "v_doppler_mps"
    assert doppler_field(fields(("x", 7), ("velocity", 8), ("doppler", 7))) == "doppler"
    assert doppler_field(fields(("x", 7), ("doppler", 4))) is None  # an integer field
    assert doppler_field(fields(("x", 7), ("intensity", 7))) is None


def _write_bag(rig: Rig, directory: Path) -> Path:
    from calibrex.data.ros_cdr_writer import encode_imu, encode_pointcloud2
    from calibrex.data.rosbag2_writer import Rosbag2Writer

    rng = np.random.default_rng(5)
    bag = directory / "rig"
    fields = [("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1), ("v_doppler_mps", 12, 7, 1)]
    with Rosbag2Writer(bag) as writer:
        writer.add_topic("/imu", "sensor_msgs/msg/Imu")
        writer.add_topic("/radar/scan", "sensor_msgs/msg/PointCloud2")
        for t, g, a in zip(rig.imu.times_s, rig.imu.gyro_rps, rig.imu.accel_mps2, strict=True):
            ns = round((t + 1.7e9) * 1e9)
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
            ns = round((t + 1.7e9) * 1e9)
            writer.write(
                "/radar/scan",
                ns,
                encode_pointcloud2(
                    frame_id="radar", timestamp_ns=ns, fields=fields, point_step=16, data=data
                ),
            )
    return bag


def test_calibrex_check_judges_radar_imu(tmp_path: Path) -> None:
    from calibrex.check.runner import CheckRunOptions, build_calibration_check

    rig = synthetic_rig()
    bag = _write_bag(rig, tmp_path)
    good = Rotation.from_matrix(rig.radar_to_imu)  # T_imu_radar
    turned = (Rotation.from_euler("y", 4.0, degrees=True) * good.inv()).inv()
    results = {}
    for name, rotation in (("good", good), ("turned", turned)):
        frames = tmp_path / f"{name}.yaml"
        frames.write_text(
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
            )
        )
        artifact = build_calibration_check(
            bag,
            tf_files=[frames],
            wired_pairs=frozenset({"radar-imu"}),
            run=CheckRunOptions(evidence_dir=tmp_path / f"evidence-{name}"),
        )
        results[name] = next(p for p in artifact.pairs if p.pair == "radar-imu")

    radar = next(t for t in artifact.topics if t.topic == "/radar/scan")
    assert radar.role == "radar"
    assert results["good"].status == "pass"
    assert results["turned"].status == "fail"
    pitch = next(a for a in results["turned"].axes if a.name == "pitch")
    assert pitch.status == "fail"
