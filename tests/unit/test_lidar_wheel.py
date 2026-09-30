from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.lidar_wheel_odometry import load_lidar_wheel_odometry
from calibrex.core.validation import validate_file
from calibrex.solvers.lidar_wheel_solver import (
    WheelOdometry,
    parse_wheel_csv,
    solve_lidar_wheel,
    wheel_std,
)
from calibrex.solvers.vehicle_frame_solver import motions_from_poses

TRUE_R_VS = Rotation.from_euler("xyz", [-0.8, 0.5, -0.3], degrees=True).as_matrix()
LEVER = np.array([1.2, 0.0, 1.5])
SCALE = 1.03  # true speed = SCALE x wheel speed (tyre radius error)
OFFSET = 0.035  # t_wheel = t_lidar + OFFSET


def speed(time: np.ndarray) -> np.ndarray:
    return 8.0 + 2.0 * np.sin(0.05 * time) + 1.5 * np.sin(0.7 * time)


def yaw_rate(time: np.ndarray) -> np.ndarray:
    return 0.15 * np.sin(0.2 * time) + 0.05 * np.sin(1.1 * time)


def drive(seed: int = 0) -> tuple[np.ndarray, list[np.ndarray], WheelOdometry]:
    rng = np.random.default_rng(seed)
    fine = np.arange(0.0, 120.0, 0.001)
    heading = np.concatenate(
        [[0.0], np.cumsum(0.5 * (yaw_rate(fine[1:]) + yaw_rate(fine[:-1])) * 0.001)]
    )
    vx = speed(fine) * np.cos(heading)
    vy = speed(fine) * np.sin(heading)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * 0.001)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * 0.001)])
    times = np.arange(0.0, 120.0, 0.1)
    index = np.round(times / 0.001).astype(int)
    poses = []
    for k, i in enumerate(index):
        vehicle = np.eye(4)
        vehicle[:3, :3] = Rotation.from_euler(
            "ZYX", [heading[i], 0.002 * np.sin(3.0 * times[k]), 0.003 * np.sin(2.3 * times[k])]
        ).as_matrix()
        vehicle[:3, 3] = [x[i], y[i], 0.0]
        sensor = np.eye(4)
        sensor[:3, :3] = TRUE_R_VS
        sensor[:3, 3] = LEVER
        noise = np.eye(4)
        noise[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.0003, 3)).as_matrix()
        noise[:3, 3] = rng.normal(0, 0.005, 3)
        poses.append(vehicle @ sensor @ noise)
    wheel_times = np.arange(-1.0, 121.0, 0.02)
    wheel = WheelOdometry(
        wheel_times,
        speed(wheel_times - OFFSET) / SCALE + rng.normal(0, 0.02, len(wheel_times)),
        yaw_rate(wheel_times - OFFSET) + rng.normal(0, 0.002, len(wheel_times)),
    )
    return times, poses, wheel


def test_recovers_rotation_scale_offset_and_lever() -> None:
    times, poses, wheel = drive()
    result = solve_lidar_wheel(motions_from_poses(times, poses), wheel)

    assert result.rotation is not None
    error = np.degrees(np.abs(Rotation.from_matrix(result.rotation @ TRUE_R_VS.T).as_rotvec()))
    assert error.max() < 0.05
    assert result.speed_scale == pytest.approx(SCALE, abs=0.002)
    assert result.time_offset_s == pytest.approx(OFFSET, abs=0.005)
    assert result.lever_m == pytest.approx(LEVER[0], abs=0.1)
    assert np.all(np.isfinite(wheel_std(result)))


def test_parse_wheel_csv_accepts_headers_and_sorts() -> None:
    wheel = parse_wheel_csv("t,v,w\n2, 5.0, 0.1\n1, 4.0, 0.0\n")
    assert wheel.times_s.tolist() == [1.0, 2.0]
    assert wheel.at(np.array([1.5]))[0][0] == pytest.approx(4.5)
    with pytest.raises(ValueError):
        parse_wheel_csv("t,v,w\n1,2,3\n")


def test_trajectory_cli_estimates_scale_and_offset(tmp_path: Path) -> None:
    times, poses, wheel = drive(seed=1)
    trajectory = tmp_path / "lidar.tum"
    rows = []
    for time, pose in zip(times, poses, strict=True):
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        rows.append(" ".join(f"{value:.9f}" for value in [time, *pose[:3, 3], *quat]))
    trajectory.write_text("\n".join(rows), encoding="utf-8")
    wheel_csv = tmp_path / "wheel.csv"
    wheel_csv.write_text(
        "t,speed,yaw_rate\n"
        + "\n".join(
            f"{t:.4f},{v:.6f},{w:.6f}"
            for t, v, w in zip(wheel.times_s, wheel.speed_mps, wheel.yaw_rate_rps, strict=True)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.yaml"

    exit_code = main(
        ["lidar-wheel", "trajectory", "--trajectory", str(trajectory), "--wheel", str(wheel_csv),
         "--output", str(output)]
    )  # fmt: skip

    assert exit_code == 0
    assert validate_file(output).kind == "lidar-wheel-odometry"
    artifact = load_lidar_wheel_odometry(output)
    records = {item.name: item for item in artifact.parameters}
    assert records["speed_scale"].status == "estimated"
    assert artifact.speed_scale == pytest.approx(SCALE, abs=0.003)
    assert artifact.time_offset_s == pytest.approx(OFFSET, abs=0.005)
    # The mild synthetic dynamics constrain the clock offset only weakly, so its
    # 20 ms control need not be detected; the policy then says so (warn).
    assert records["time_offset"].known_bad_control is not None
    if not records["time_offset"].known_bad_control.detected:
        assert artifact.policy_status in {"warn", "inconclusive"}
