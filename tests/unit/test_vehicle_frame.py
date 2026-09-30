from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.evaluation.vehicle_frame import (
    VehicleFrameRunOptions,
    evaluate_vehicle_frame,
    reference_entry,
)
from calibrex.solvers.vehicle_frame_solver import (
    motions_from_poses,
    rotation_std_deg,
    solve_vehicle_frame,
)

TRUE_R_VS = Rotation.from_euler("xyz", [-0.8, 0.5, -0.3], degrees=True).as_matrix()
LEVER = np.array([1.2, 0.0, 1.5])  # sensor position in the vehicle frame (from the rear axle)


def drive(*, turning: bool, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """A bicycle-model drive with hills and small suspension pitch/roll, seen by the sensor."""

    rng = np.random.default_rng(seed)
    times = np.arange(0.0, 120.0, 0.1)
    heading = np.zeros(len(times))
    position = np.zeros((len(times), 3))
    poses = []
    for index, time in enumerate(times):
        speed = 8.0 + 2.0 * np.sin(0.05 * time)
        yaw_rate = 0.15 * np.sin(0.2 * time) if turning else 0.0
        if index:
            heading[index] = heading[index - 1] + yaw_rate * 0.1
            middle = 0.5 * (heading[index - 1] + heading[index])  # midpoint rule
            forward = np.array([np.cos(middle), np.sin(middle), 0.0])
            position[index] = position[index - 1] + forward * speed * 0.1
        grade = 0.03 * np.sin(0.02 * time)
        position[index, 2] = 0.0 if index == 0 else position[index - 1, 2] + grade * speed * 0.1
        vehicle = np.eye(4)
        vehicle[:3, :3] = Rotation.from_euler(
            "ZYX",  # intrinsic: heading, then pitch and roll about the body axes
            [heading[index], -grade + 0.002 * np.sin(3.0 * time), 0.003 * np.sin(2.3 * time)],
        ).as_matrix()
        vehicle[:3, 3] = position[index]
        sensor = np.eye(4)
        sensor[:3, :3] = TRUE_R_VS
        sensor[:3, 3] = LEVER
        pose = vehicle @ sensor
        noise = np.eye(4)
        noise[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.0003, 3)).as_matrix()
        noise[:3, 3] = rng.normal(0, 0.005, 3)
        poses.append(pose @ noise)
    return times, poses


def test_turning_drive_recovers_every_rotation_axis() -> None:
    times, poses = drive(turning=True)
    result = solve_vehicle_frame(motions_from_poses(times, poses))

    assert result.rotation is not None
    error = Rotation.from_matrix(result.rotation @ TRUE_R_VS.T).as_rotvec()
    assert np.degrees(np.abs(error)).max() < 0.05
    assert result.lever_m is not None and abs(result.lever_m - LEVER[0]) < 0.1


def test_straight_drive_leaves_roll_unconstrained() -> None:
    times, poses = drive(turning=False)
    result = solve_vehicle_frame(motions_from_poses(times, poses), initial=TRUE_R_VS)

    std = rotation_std_deg(result)
    assert std[0] > 5 * max(std[1], std[2])


def test_evaluation_estimates_pitch_and_yaw_and_detects_every_control() -> None:
    times, poses = drive(turning=True, seed=3)
    motions = motions_from_poses(times, poses, block_duration_s=5.0)
    evaluation = evaluate_vehicle_frame(motions, VehicleFrameRunOptions(block_duration_s=5.0))

    status = {record.name: record.status for record in evaluation.records}
    # Roll is seen only through the tilt of the turning axis, so it is the weakest axis.
    assert status["pitch"] == status["yaw"] == "estimated"
    assert evaluation.policy_status in {"pass", "inconclusive"}, evaluation.policy_reasons
    assert all(
        record.known_bad_control and record.known_bad_control.detected
        for record in evaluation.records
    )
    reference = reference_entry(
        "truth rotated by 0.5 deg of yaw",
        Rotation.from_euler("z", 0.5, degrees=True).as_matrix() @ TRUE_R_VS,
        evaluation,
        [item for item in motions if item.block in evaluation.holdout_blocks],
        VehicleFrameRunOptions(block_duration_s=5.0),
    )
    assert reference.difference_deg[2] == pytest.approx(-0.5, abs=0.05)
    assert reference.holdout_delta_chi2 is not None and reference.holdout_delta_chi2 > 9.0
