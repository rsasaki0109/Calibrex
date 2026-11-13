"""The radar-IMU rotation from Doppler ego velocity: synthetic rigs with a known truth."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.check.demo import synthetic_rig, write_demo
from calibrex.data.livox_ros2 import ImuSamples
from calibrex.data.radar_pointcloud import doppler_field
from calibrex.evaluation.radar_imu_rotation import (
    RadarImuOptions,
    RadarVelocities,
    ego_velocity,
    evaluate_radar_imu_rotation,
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


def test_repeated_imu_stamps_are_kept_once() -> None:
    rig = synthetic_rig()
    repeat = np.sort(np.r_[np.arange(len(rig.imu.times_s)), np.arange(500, 20000, 200)])
    imu = ImuSamples(rig.imu.times_s[repeat], rig.imu.gyro_rps[repeat], rig.imu.accel_mps2[repeat])
    velocities = RadarVelocities(rig.radar_times_s, rig.radar_velocities, len(rig.radar_times_s))

    found = evaluate_radar_imu_rotation(velocities, imu, gyro_bias=rig.gyro_bias)

    assert found.rotation is not None
    error = Rotation.from_matrix(found.rotation @ rig.radar_to_imu.T).magnitude()
    assert np.degrees(error) < 1.0


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


def test_calibrex_check_judges_radar_imu(tmp_path: Path) -> None:
    from calibrex.check.runner import CheckRunOptions, build_calibration_check

    bag, frames = write_demo(tmp_path)
    results = {}
    for name, path in frames.items():
        artifact = build_calibration_check(
            bag,
            tf_files=[path],
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


def test_the_check_demo_passes_the_true_mounting_and_fails_the_turned_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrex.cli.main import main

    assert main(["demo", "check", "--output-dir", str(tmp_path / "demo")]) == 0

    out = capsys.readouterr().out
    assert "overall verdict: pass" in out and "overall verdict: fail" in out
    assert "pitch" in out and "[fail]" in out
    assert "synthetic" in out and "Your own bag:" in out
    assert (tmp_path / "demo" / "check_turned.html").exists()
