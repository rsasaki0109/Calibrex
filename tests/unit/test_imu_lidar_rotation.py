from __future__ import annotations

import functools
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.imu_lidar_rotation import ImuLidarRotationArtifact
from calibrex.core.validation import validate_file
from calibrex.data import livox_ros2
from calibrex.evaluation.imu_lidar_rotation import (
    evaluate_imu_lidar_rotation,
    score_imu_lidar_candidate,
)
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import (
    GyroSeries,
    ImuLidarRotationSolver,
    RotationOptions,
    rate_intervals,
)
from calibrex.solvers.scan_to_scan_odometry import deskew_points

EPOCH = 1.76e9
TRUE_ROTATION = Rotation.from_euler("xyz", [0.8, -1.5, 2.0], degrees=True).as_matrix()
TRUE_BIAS = np.array([0.004, -0.002, 0.003])
TRUE_DT = 0.012


def attitude(time: float, *, planar: bool) -> np.ndarray:
    local = time - EPOCH
    yaw = 0.8 * math.sin(0.3 * local) + 0.4 * math.sin(1.3 * local)
    if planar:
        return Rotation.from_euler("z", yaw).as_matrix()
    return Rotation.from_euler(
        "xyz", [0.25 * math.sin(0.9 * local), 0.2 * math.sin(0.7 * local + 1.0), yaw]
    ).as_matrix()


def synthetic(*, planar: bool, seed: int = 0) -> tuple[GyroSeries, list[OdometryWindow]]:
    rng = np.random.default_rng(seed)
    step = 0.005
    times = EPOCH + np.arange(0.0, 90.0, step)
    rotations = [attitude(t, planar=planar) for t in times]
    rates = np.array(
        [
            Rotation.from_matrix(rotations[k].T @ rotations[k + 1]).as_rotvec() / step
            for k in range(len(times) - 1)
        ]
    )
    # The finite-difference rate belongs to the middle of its interval.
    mid_times = times[:-1] + 0.5 * step
    gyro = rates @ TRUE_ROTATION + TRUE_BIAS + rng.normal(0.0, 0.003, rates.shape)
    series = GyroSeries(mid_times + TRUE_DT, gyro)
    windows = []
    for index in range(9):
        lidar_times = EPOCH + 0.5 + 10.0 * index + np.arange(0.0, 9.9, 0.1)
        poses = np.stack([np.eye(4)] * len(lidar_times))
        for k, t in enumerate(lidar_times):
            noise = Rotation.from_rotvec(rng.normal(0.0, 0.0005, 3)).as_matrix()
            poses[k, :3, :3] = attitude(t, planar=planar) @ noise
        windows.append(OdometryWindow(f"w{index}", index, lidar_times, poses))
    return series, windows


def _solve(planar: bool) -> Any:
    gyro, windows = synthetic(planar=planar)
    return ImuLidarRotationSolver().solve(gyro, rate_intervals(windows, gyro, RotationOptions()))


def test_three_dimensional_rotation_recovers_every_quantity() -> None:
    result = _solve(planar=False)

    assert result.rotation is not None
    error = Rotation.from_matrix(TRUE_ROTATION.T @ result.rotation).magnitude()
    assert math.degrees(error) < 0.05
    assert all(dof.status == "estimated" for dof in result.dofs)
    assert abs(result.time_offset_s - TRUE_DT) < 0.002
    assert np.allclose(result.gyro_bias_rps, TRUE_BIAS, atol=5e-4)


def test_yaw_only_motion_leaves_yaw_unobservable() -> None:
    result = _solve(planar=True)

    assert result.dof("yaw").status == "unobservable"
    assert result.dof("roll").status == result.dof("pitch").status == "estimated"


def test_initial_procrustes_handles_an_arbitrary_mounting() -> None:
    gyro, windows = synthetic(planar=False)
    flipped = GyroSeries(gyro.times_s, gyro.gyro_rps @ np.diag([-1.0, -1.0, 1.0]))

    result = ImuLidarRotationSolver().solve(
        flipped, rate_intervals(windows, flipped, RotationOptions())
    )

    assert result.rotation is not None
    expected = TRUE_ROTATION @ np.diag([-1.0, -1.0, 1.0])
    assert math.degrees(Rotation.from_matrix(expected.T @ result.rotation).magnitude()) < 0.1


def test_deskew_moves_late_points_by_the_fraction_of_the_motion() -> None:
    motion = np.eye(4)
    motion[:3, 3] = [1.0, 0.0, 0.0]
    points = np.zeros((3, 3))

    moved = deskew_points(points, np.array([0.0, 0.05, 0.1]), motion, 0.1)

    assert np.allclose(moved[:, 0], [0.0, 0.5, 1.0])


def test_livox_profiles_convert_absolute_point_times_and_g(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header_ns = 1_713_244_621_000_000_000
    cloud = SimpleNamespace(
        timestamp_ns=header_ns,
        xyz=np.zeros((2, 3)),
        point_time_offsets_s=np.array([header_ns, header_ns + 50_000_000], dtype=np.float64),
    )
    imu = SimpleNamespace(
        timestamp_ns=header_ns,
        angular_velocity=(0.0, 0.0, 0.1),
        linear_acceleration=(0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(
        livox_ros2,
        "iter_messages",
        lambda bag, topics: iter([(None, header_ns, b""), (None, header_ns + 5_000_000, b"")]),
    )
    monkeypatch.setattr(livox_ros2.ros_cdr, "decode_ros2_pointcloud2", lambda *a, **k: cloud)
    monkeypatch.setattr(livox_ros2.ros_cdr, "decode_ros2_imu", lambda *a, **k: imu)
    profile = livox_ros2.LIVOX_PROFILES["livox-ros-driver2"]

    _, _, offsets = next(livox_ros2.iter_livox_points("bag", profile))
    samples = livox_ros2.load_livox_imu("bag", profile)

    assert offsets is not None and np.allclose(offsets, [0.0, 0.05])
    assert np.allclose(samples.accel_mps2[:, 2], livox_ros2.STANDARD_GRAVITY_MPS2)


@functools.cache
def _evaluation() -> Any:
    gyro, windows = synthetic(planar=False, seed=5)
    return evaluate_imu_lidar_rotation(gyro, windows, reference_rotation=TRUE_ROTATION)


def _artifact(evaluation: Any, *, policy: str | None = None) -> ImuLidarRotationArtifact:
    result = evaluation.result
    return ImuLidarRotationArtifact.model_validate(
        {
            "solver_status": result.status,
            "policy_status": policy or evaluation.policy_status,
            "policy_reasons": list(evaluation.policy_reasons),
            "calibrated_dofs": [r.name for r in evaluation.records if r.status == "estimated"],
            "rotation_quat_xyzw": list(Rotation.from_matrix(result.rotation).as_quat()),
            "time_offset_s": result.time_offset_s,
            "gyro_bias_rps": list(result.gyro_bias_rps),
            "dofs": [record.model_dump() for record in evaluation.records],
            "options": {},
            "windows": {
                "scans_read": 900,
                "odometry_segments": 9,
                "unreliable_registrations": 0,
                "windows": 9,
                "rate_intervals": evaluation.interval_count,
                "imu_samples": 18000,
            },
            "train_windows": evaluation.train_windows,
            "holdout_windows": evaluation.holdout_windows,
            "jackknife_fits": evaluation.jackknife_fits,
            "provenance": {
                "generator": "pytest",
                "generator_version": "1",
                "dataset_family": "synthetic",
                "sequence_ids": ["synthetic"],
                "stream_profile": "synthetic",
                "input_sha256": "c" * 64,
                "input_digest_scope": "synthetic",
                "dataset_license": "synthetic",
            },
        }
    )


def test_evaluation_passes_with_three_dimensional_motion(tmp_path: Path) -> None:
    evaluation = _evaluation()
    records = {record.name: record for record in evaluation.records}

    assert evaluation.policy_status == "pass", evaluation.policy_reasons
    for name in ("roll", "pitch", "yaw", "time_offset"):
        control = records[name].known_bad_control
        assert control is not None and control.detected, name
    assert abs(records["yaw"].error_to_reference or 0.0) < 0.05

    path = tmp_path / "imu-lidar.yaml"
    _artifact(evaluation).save(path)
    assert validate_file(path).kind == "imu-lidar-rotation"


def test_artifact_refuses_a_pass_without_yaw() -> None:
    payload = _artifact(_evaluation()).model_dump(mode="python")
    for record in payload["dofs"]:
        if record["name"] == "yaw":
            record["status"] = "unobservable"
    payload["calibrated_dofs"].remove("yaw")

    with pytest.raises(ValueError, match="roll, pitch, and yaw"):
        ImuLidarRotationArtifact.model_validate(payload)


def test_cli_forwards_profile_and_reference_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(bags: Any, profile: str, **kwargs: Any) -> ImuLidarRotationArtifact:
        captured.update(kwargs, bags=bags, profile=profile)
        return _artifact(_evaluation())

    monkeypatch.setattr("calibrex.cli.main.run_livox_imu_lidar_rotation", fake_run)
    output = tmp_path / "out.yaml"

    exit_code = main(
        [
            "imu-lidar",
            "livox",
            "bag_a",
            "--profile",
            "rtk-slam",
            "--dataset-family",
            "rtk_slam",
            "--dataset-license",
            "custom",
            "--no-mid360-reference",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured["profile"] == "rtk-slam"
    assert captured["reference_rotation"] is None
    assert validate_file(output).valid


def _coning_case(model: str) -> tuple[float, np.ndarray]:
    """Fast multi-axis rotation: roll/pitch at 3.0/2.3 Hz plus a 1.1 Hz yaw sway."""

    rng = np.random.default_rng(0)

    def fast(time: float) -> np.ndarray:
        local = time - EPOCH
        return Rotation.from_euler(
            "xyz",
            [
                0.3 * math.sin(2.0 * math.pi * 3.0 * local),
                0.3 * math.sin(2.0 * math.pi * 2.3 * local + 1.0),
                math.sin(0.5 * local) + 0.5 * math.sin(2.0 * math.pi * 1.1 * local),
            ],
        ).as_matrix()

    step = 0.002
    times = EPOCH + np.arange(0.0, 40.0, step)
    rotations = [fast(t) for t in times]
    rates = np.array(
        [
            Rotation.from_matrix(rotations[k].T @ rotations[k + 1]).as_rotvec() / step
            for k in range(len(times) - 1)
        ]
    )
    gyro = GyroSeries(
        times[:-1] + 0.5 * step + TRUE_DT,
        rates @ TRUE_ROTATION + TRUE_BIAS + rng.normal(0.0, 0.003, rates.shape),
    )
    lidar_times = EPOCH + 0.5 + np.arange(0.0, 38.0, 0.1)
    poses = np.stack([np.eye(4)] * len(lidar_times))
    for k, t in enumerate(lidar_times):
        poses[k, :3, :3] = fast(t)
    options = RotationOptions(model=model)  # type: ignore[arg-type]
    result = ImuLidarRotationSolver().solve(
        gyro,
        rate_intervals([OdometryWindow("w", 0, lidar_times, poses)], gyro, options),
        options,
    )
    assert result.rotation is not None
    error = math.degrees(Rotation.from_matrix(TRUE_ROTATION.T @ result.rotation).magnitude())
    return error, result.gyro_bias_rps


def test_preintegration_removes_the_coning_bias_of_mean_rates() -> None:
    mean_rate_error, mean_rate_bias = _coning_case("mean_rate")
    preintegrated_error, preintegrated_bias = _coning_case("preintegrated")

    assert mean_rate_error > 0.08
    assert preintegrated_error < 0.03
    assert np.allclose(preintegrated_bias, TRUE_BIAS, atol=1e-3)
    assert not np.allclose(mean_rate_bias, TRUE_BIAS, atol=1e-3)


def test_gyro_rotation_model_maps_gyro_rotations_into_the_lidar_frame() -> None:
    from calibrex.solvers.imu_lidar_rotation_solver import gyro_rotation_model

    times = EPOCH + np.arange(0.0, 2.0, 0.005)
    rate = np.array([0.0, 0.0, 0.5])
    gyro = GyroSeries(times, np.tile(rate, (len(times), 1)))
    mounting = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()

    model = gyro_rotation_model(gyro, mounting, np.zeros(3), 0.0)
    rotations = model(EPOCH + 0.5, np.array([0.0, 0.1]))

    expected = mounting @ Rotation.from_rotvec(rate * 0.1).as_matrix() @ mounting.T
    assert np.allclose(rotations[0], np.eye(3), atol=1e-9)
    assert np.allclose(rotations[1], expected, atol=1e-6)


def test_deskew_with_rotations_applies_rotation_then_fractional_translation() -> None:
    from calibrex.solvers.scan_to_scan_odometry import deskew_points_with_rotations

    motion = np.eye(4)
    motion[:3, 3] = [0.2, 0.0, 0.0]
    quarter = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    points = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    moved = deskew_points_with_rotations(
        points, np.array([0.0, 0.05]), np.stack([np.eye(3), quarter]), motion, 0.1
    )

    assert np.allclose(moved, [[1.0, 0.0, 0.0], [0.1, 1.0, 0.0]])


def test_a_pass_is_refused_while_the_deskew_iteration_still_moves() -> None:
    from calibrex.evaluation.imu_lidar_rotation import _gate_convergence

    evaluation = _evaluation()
    assert evaluation.policy_status == "pass"
    steady = _gate_convergence(evaluation, np.zeros(3))
    moving = _gate_convergence(evaluation, np.array([0.0, 0.0, 1.0]))

    assert steady.policy_status == "pass"
    assert moving.policy_status == "warn"
    assert "has not converged" in moving.policy_reasons[0]


def test_candidate_score_ranks_the_true_extrinsic_first() -> None:
    gyro, windows = synthetic(planar=False, seed=7)
    true = score_imu_lidar_candidate(gyro, windows, TRUE_ROTATION, TRUE_DT)
    tilted = Rotation.from_euler("y", 4.0, degrees=True).as_matrix() @ TRUE_ROTATION
    wrong = score_imu_lidar_candidate(gyro, windows, tilted, TRUE_DT - 0.05)

    assert true.holdout_windows == 3
    assert np.allclose(true.refit_gyro_bias_rps, TRUE_BIAS, atol=2e-3)
    assert wrong.holdout_median_rate_residual_rps > 2.0 * true.holdout_median_rate_residual_rps


def test_candidates_with_shifted_segmentations_share_the_reference_holdout_spans() -> None:
    gyro, windows = synthetic(planar=False, seed=7)
    reference = score_imu_lidar_candidate(gyro, windows, TRUE_ROTATION, TRUE_DT)
    # Another candidate's odometry lost the first window, which shifts its
    # own every-third-window held-out choice onto different stretches.
    shifted = windows[1:]
    own = score_imu_lidar_candidate(gyro, shifted, TRUE_ROTATION, TRUE_DT)
    shared = score_imu_lidar_candidate(
        gyro, shifted, TRUE_ROTATION, TRUE_DT, holdout_spans_s=reference.holdout_spans_s
    )

    assert set(own.holdout_window_medians_rps).isdisjoint(reference.holdout_window_medians_rps)
    assert set(shared.holdout_window_medians_rps) == set(reference.holdout_window_medians_rps)
    assert shared.holdout_intervals == reference.holdout_intervals
