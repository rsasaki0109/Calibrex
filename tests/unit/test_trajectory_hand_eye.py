from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.solvers.trajectory_hand_eye_solver import (
    DofPrior,
    ReferenceTrajectory,
    SensorMotion,
    TrajectoryHandEyeOptions,
    TrajectoryHandEyeSolver,
    covered_motions,
)

EPOCH = 1_317_042_272.0  # KITTI-sized UNIX time: exposes float64 resolution bugs
TRUE_DT = 0.012
TRUE_SCALE = 1.02


def true_extrinsic() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [1.0, -2.0, 3.0], degrees=True).as_matrix()
    transform[:3, 3] = [0.8, -0.3, 0.9]
    return transform


def reference_pose(time: float, *, planar: bool) -> np.ndarray:
    """Ground-truth ``T_world_ins`` of a vehicle weaving along a road."""

    local = time - EPOCH
    yaw = 0.6 * math.sin(0.35 * local) + 0.2 * math.sin(1.1 * local)
    roll = 0.0 if planar else 0.08 * math.sin(0.9 * local)
    pitch = 0.0 if planar else 0.06 * math.sin(0.7 * local + 0.4)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    heading = 0.6 / 0.35 * (1.0 - math.cos(0.35 * local))
    pose[:3, 3] = [8.0 * local, 6.0 * math.sin(0.3 * local) + heading, 0.0]
    return pose


def synthetic(
    *,
    planar: bool,
    duration_s: float = 40.0,
    noise: float = 1.0,
    seed: int = 7,
) -> tuple[ReferenceTrajectory, list[SensorMotion]]:
    rng = np.random.default_rng(seed)
    x_true = true_extrinsic()
    reference_times = EPOCH + np.arange(-1.0, duration_s + 1.0, 0.05)
    reference_poses = np.stack([reference_pose(t, planar=planar) for t in reference_times])
    reference_poses[:, :3, 3] /= TRUE_SCALE
    sensor_times = EPOCH + np.arange(0.0, duration_s, 0.1)
    sensor_poses = [reference_pose(t + TRUE_DT, planar=planar) @ x_true for t in sensor_times]
    motions: list[SensorMotion] = []
    for step in (5, 10, 20):
        for index in range(0, len(sensor_times) - step, step):
            relative = np.linalg.inv(sensor_poses[index]) @ sensor_poses[index + step]
            relative[:3, :3] = (
                Rotation.from_rotvec(rng.normal(0.0, noise * 2e-4, 3)).as_matrix()
                @ relative[:3, :3]
            )
            relative[:3, 3] += rng.normal(0.0, noise * 0.003, 3)
            motions.append(
                SensorMotion(
                    motion_id=f"s{step}-{index}",
                    block=int((sensor_times[index] - EPOCH) // 5.0),
                    start_s=float(sensor_times[index]),
                    end_s=float(sensor_times[index + step]),
                    motion=relative,
                )
            )
    return ReferenceTrajectory(reference_times, reference_poses), motions


def _rotation_error_deg(estimate: np.ndarray) -> float:
    delta = np.linalg.inv(true_extrinsic()) @ estimate
    return math.degrees(float(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))


def test_recovers_every_dof_when_rotation_is_three_dimensional() -> None:
    reference, motions = synthetic(planar=False)

    result = TrajectoryHandEyeSolver().solve(reference, motions)

    assert result.status == "converged" and result.transform is not None
    assert {item.name: item.status for item in result.dofs} == dict.fromkeys(
        ("roll", "pitch", "yaw", "x", "y", "z", "time_offset", "reference_scale"), "estimated"
    )
    assert _rotation_error_deg(result.transform) < 0.05
    assert np.linalg.norm(result.transform[:3, 3] - true_extrinsic()[:3, 3]) < 0.02
    assert abs(result.time_offset_s - TRUE_DT) < 0.002
    assert abs(result.reference_scale - TRUE_SCALE) < 0.002


def test_planar_motion_recovers_yaw_but_leaves_z_unobservable() -> None:
    reference, motions = synthetic(planar=True)

    result = TrajectoryHandEyeSolver().solve(reference, motions)

    assert result.transform is not None
    statuses = {item.name: item.status for item in result.dofs}
    assert statuses["z"] == "unobservable"
    assert statuses["yaw"] == "estimated"
    assert statuses["x"] == statuses["y"] == "estimated"
    yaw_error = math.degrees(result.dof("yaw").value) - 3.0
    assert abs(yaw_error) < 0.1
    assert np.linalg.norm(result.transform[:2, 3] - true_extrinsic()[:2, 3]) < 0.03


def test_time_offset_moves_away_from_zero_at_unix_epoch_timestamps() -> None:
    """Regression: absolute timestamps froze the clock offset at its initial value."""

    reference, motions = synthetic(planar=False)

    result = TrajectoryHandEyeSolver().solve(reference, motions)

    assert result.dof("time_offset").status == "estimated"
    assert abs(result.time_offset_s - TRUE_DT) < 0.002
    assert abs(result.time_offset_s) > 0.005


def test_a_prior_constrains_the_unobservable_translation() -> None:
    reference, motions = synthetic(planar=True)
    options = TrajectoryHandEyeOptions(priors=(DofPrior("z", 0.9, 0.01, "cad"),))

    result = TrajectoryHandEyeSolver().solve(reference, motions, options)

    z = result.dof("z")
    assert z.status == "prior"
    assert abs(z.value - 0.9) < 0.02
    assert z.std_total < z.std_data


def test_priors_reject_unsupported_dofs() -> None:
    with pytest.raises(ValueError, match="priors are supported"):
        DofPrior("yaw", 0.0, 0.01, "cad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive sigma"):
        DofPrior("z", 0.0, 0.0, "cad")


def test_reference_gaps_are_never_interpolated() -> None:
    times = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    poses = np.stack([np.eye(4)] * 5)
    reference = ReferenceTrajectory(times, poses, max_gap_s=1.0)
    spanning = SensorMotion("across", 0, 0.15, 5.05, np.eye(4))
    inside = SensorMotion("inside", 0, 5.0, 5.1, np.eye(4))

    assert reference.covers(0.15) and not reference.covers(2.0)
    assert not reference.continuous(0.15, 5.05)
    kept = covered_motions(
        reference, [spanning, inside], TrajectoryHandEyeOptions(estimate_time_offset=False)
    )
    assert [item.motion_id for item in kept] == ["inside"]


def test_too_few_motions_is_reported_not_solved() -> None:
    reference, motions = synthetic(planar=False, duration_s=4.0)

    result = TrajectoryHandEyeSolver().solve(reference, motions[:2])

    assert result.status == "insufficient_motions"
    assert result.transform is None
