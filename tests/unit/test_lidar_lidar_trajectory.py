"""Per-point reference trajectory, target accumulation, and clock offset for LiDAR-LiDAR.

The scene imitates a Unitree Go2: a dense reference LiDAR on the back (10 Hz),
the built-in L1 under the chin as the target (15 Hz, inverted, a few hundred
points per scan, each with its own capture time), a walking trajectory that
turns at 10-60 deg/s with gait roll and pitch, and an L1 clock that lags
the reference by 37 ms.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.lidar_lidar_extrinsic import (
    LidarLidarExtrinsicArtifact,
    LidarLidarTimeOffset,
)
from calibrex.evaluation.lidar_lidar_map import (
    LidarLidarMapOptions,
    ReferenceTrajectory,
    TargetScan,
    build_trajectory_samples,
    estimate_time_offset,
    evaluate_lidar_lidar_map,
    preprocess_timed_scan,
)
from calibrex.solvers.lidar_lidar_map_solver import (
    MapSample,
    MapSampleOptions,
    perturb_extrinsic,
    solve_map_extrinsic,
)
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions

# T_reference_target: the L1 ahead of and below the back LiDAR, pitched 165 deg (inverted).
TRUE_X = np.eye(4)
TRUE_X[:3, :3] = Rotation.from_euler("xyz", [0.6, 164.9, -0.8], degrees=True).as_matrix()
TRUE_X[:3, 3] = [0.27, 0.01, -0.16]
CLOCK_OFFSET_S = 0.037  # t_reference = t_target + offset
REFERENCE_RATE_HZ = 10.0
TARGET_RATE_HZ = 15.0


def room(rng: np.random.Generator) -> np.ndarray:
    faces = []
    for axis, value, extent in [
        (0, 0.0, (8, 4)), (0, 12.0, (8, 4)), (1, 0.0, (12, 4)), (1, 8.0, (12, 4)),
        (2, 0.0, (12, 8)), (2, 4.0, (12, 8)),
    ]:  # fmt: skip
        uv = rng.uniform(0, 1, (5000, 2)) * extent
        other = [index for index in range(3) if index != axis]
        points = np.zeros((5000, 3))
        points[:, axis] = value
        points[:, other[0]], points[:, other[1]] = uv[:, 0], uv[:, 1]
        faces.append(points)
    for corner, size in (([3.0, 2.0, 0.0], [1.5, 1.0, 1.2]), ([8.0, 5.0, 0.0], [1.0, 1.4, 0.8])):
        box = rng.uniform(0, 1, (3000, 3)) * size
        axis = rng.integers(0, 3, 3000)
        box[np.arange(3000), axis] = rng.integers(0, 2, 3000) * np.asarray(size)[axis]
        faces.append(box + corner)
    return np.concatenate(faces)


def body_pose(times: np.ndarray) -> np.ndarray:
    """T_world_reference along a walking path with gait sway."""

    t = np.asarray(times, dtype=np.float64)
    poses = np.zeros((len(t), 4, 4))
    euler = np.column_stack(
        [
            np.radians(1.0) * np.sin(2 * math.pi * 1.5 * t),
            np.radians(0.8) * np.sin(2 * math.pi * 1.5 * t + 1.0),
            # A varying turn rate (0.2-1.0 rad/s): at a constant body twist a clock
            # offset is indistinguishable from an extrinsic rotation and translation.
            0.6 * t + 0.8 * np.sin(0.5 * t),
        ]
    )
    poses[:, :3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
    poses[:, :3, 3] = np.column_stack(
        [
            6.0 + 2.5 * np.sin(0.2 * t),
            4.0 + 1.5 * np.sin(0.31 * t),
            0.6 + 0.01 * np.sin(2 * math.pi * 1.5 * t),
        ]
    )
    poses[:, 3, 3] = 1.0
    return poses


def go2_scene(
    seed: int = 0, duration_s: float = 40.0, target_points: int = 300
) -> tuple[ReferenceTrajectory, list[TargetScan]]:
    rng = np.random.default_rng(seed)
    world = room(rng)
    reference_times = np.arange(0.0, duration_s, 1.0 / REFERENCE_RATE_HZ)
    poses = body_pose(reference_times)
    scans = tuple(
        world[rng.choice(len(world), 1500, replace=False)].astype(np.float32)
        for _ in reference_times
    )
    trajectory = ReferenceTrajectory(reference_times, poses, scans, float(reference_times[0]))
    targets = []
    period = 1.0 / TARGET_RATE_HZ
    for true_start in np.arange(0.0, duration_s, period):
        offsets = np.sort(rng.uniform(0.0, period, target_points))
        sensor = body_pose(true_start + offsets) @ TRUE_X
        chosen = world[rng.choice(len(world), target_points, replace=False)]
        local = np.einsum(
            "nji,nj->ni", sensor[:, :3, :3], chosen - sensor[:, :3, 3]
        ) + rng.normal(0.0, 0.01, (target_points, 3))
        # The target clock stamps the scan start CLOCK_OFFSET_S early.
        targets.append(TargetScan(float(true_start - CLOCK_OFFSET_S), local, offsets))
    return trajectory, targets


OPTIONS = LidarLidarMapOptions(
    target_motion="reference_trajectory",
    target_accumulate_scans=10,
    sample_every_scans=10,
    map_half_window_scans=2,
    block_duration_s=5.0,
    target_time_offset_s=CLOCK_OFFSET_S,
    solver=MapSampleOptions(iterations=12),
)


def rotation_error_deg(transform: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(
        Rotation.from_matrix(transform[:3, :3] @ TRUE_X[:3, :3].T).as_rotvec()
    )))  # fmt: skip


def initial_guess() -> np.ndarray:
    moved = perturb_extrinsic(TRUE_X, "pitch", np.radians(2.0))
    moved = perturb_extrinsic(moved, "yaw", np.radians(-1.5))
    return perturb_extrinsic(moved, "x", 0.06)


def test_interpolation_reproduces_poses_and_rejects_outside_times() -> None:
    times = np.arange(0.0, 2.0, 0.1)
    trajectory = ReferenceTrajectory(
        times, body_pose(times), tuple(np.zeros((1, 3), np.float32) for _ in times), 0.0
    )
    at_knots = trajectory.interpolate(times)
    assert np.allclose(at_knots, body_pose(times), atol=1e-12)
    between = trajectory.interpolate(np.array([0.55]))[0]
    truth = body_pose(np.array([0.55]))[0]
    assert np.linalg.norm(between[:3, 3] - truth[:3, 3]) < 2e-3
    assert np.isnan(trajectory.interpolate(np.array([-0.01, 1.95]))[:, 0, 0]).all()


def test_timed_preprocess_keeps_one_time_per_voxel() -> None:
    points = np.array([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 2.0, 0.0]])
    offsets = np.array([0.0, 0.02, 0.03, 0.05])
    options = ScanOdometryOptions(voxel_size_m=0.05, min_range_m=0.5, max_range_m=10.0)
    xyz, times = preprocess_timed_scan(points, offsets, options)
    assert len(xyz) == 2 and times is not None
    order = np.argsort(xyz[:, 0])
    assert np.allclose(times[order], [0.01, 0.05])


def test_per_point_jacobian_matches_finite_differences() -> None:
    rng = np.random.default_rng(3)
    trajectory, targets = go2_scene(seed=3, duration_s=4.0)
    sample = build_trajectory_samples(trajectory, targets, OPTIONS)[0]
    assert sample.point_poses is not None
    options = MapSampleOptions()
    residual, normals, points, poses = sample.residuals_with_poses(TRUE_X, options)
    from calibrex.solvers.lidar_lidar_map_solver import _jacobian

    jacobian = _jacobian(poses, TRUE_X, normals, points)
    keep = rng.choice(len(points), 20, replace=False)
    epsilon = 1e-6
    for column, dof in enumerate(("roll", "pitch", "yaw", "x", "y", "z")):
        moved = perturb_extrinsic(TRUE_X, dof, epsilon)
        shifted = MapSample(
            0.0, 0, sample.reference_pose, points[keep], sample.map_points,
            point_poses=poses[keep],
        ).world_points(moved)
        base = MapSample(
            0.0, 0, sample.reference_pose, points[keep], sample.map_points,
            point_poses=poses[keep],
        ).world_points(TRUE_X)
        numeric = np.einsum("ij,ij->i", shifted - base, normals[keep]) / epsilon
        assert np.allclose(jacobian[keep, column], numeric, atol=1e-5)
    assert len(residual) == len(points)


def test_accumulated_trajectory_samples_recover_an_inverted_l1() -> None:
    trajectory, targets = go2_scene()
    samples = build_trajectory_samples(trajectory, targets, OPTIONS)
    assert len(samples) >= 50
    assert all(len(sample.target_points) == 3000 for sample in samples)
    evaluation = evaluate_lidar_lidar_map(samples, initial_guess(), OPTIONS,
                                          reference_transform=TRUE_X)  # fmt: skip
    transform = evaluation.result.transform
    assert transform is not None
    assert rotation_error_deg(transform) < 0.1
    assert np.linalg.norm(transform[:3, 3] - TRUE_X[:3, 3]) < 0.01
    assert all(record.status == "estimated" for record in evaluation.records)
    assert all(record.known_bad_control and record.known_bad_control.detected
               for record in evaluation.records)  # fmt: skip
    assert evaluation.policy_status == "pass"


def test_ignoring_the_clock_offset_biases_the_extrinsic() -> None:
    trajectory, targets = go2_scene()
    ignored = build_trajectory_samples(trajectory, targets, OPTIONS, time_offset_s=0.0)
    fit = solve_map_extrinsic(ignored, initial_guess(), OPTIONS.solver)
    assert fit.transform is not None
    # 37 ms at 0.2-1.0 rad/s of yaw is 0.4-2.1 deg of yaw.
    assert rotation_error_deg(fit.transform) > 0.5


def test_single_sparse_rigid_scans_are_worse_than_accumulated_trajectory() -> None:
    trajectory, targets = go2_scene()
    rigid = LidarLidarMapOptions(
        target_motion="reference_trajectory", target_accumulate_scans=1,
        sample_every_scans=1, map_half_window_scans=2, block_duration_s=5.0,
        target_time_offset_s=CLOCK_OFFSET_S, solver=MapSampleOptions(iterations=12),
        min_target_points=100,
    )  # fmt: skip
    # A rigid snapshot: every point of a scan at the scan's header time.
    snapshots = [TargetScan(scan.time_s, scan.points, None) for scan in targets]
    rigid_fit = solve_map_extrinsic(
        build_trajectory_samples(trajectory, snapshots, rigid), initial_guess(), rigid.solver
    )
    good_fit = solve_map_extrinsic(
        build_trajectory_samples(trajectory, targets, OPTIONS), initial_guess(), OPTIONS.solver
    )
    assert rigid_fit.transform is not None and good_fit.transform is not None
    assert rotation_error_deg(good_fit.transform) < 0.5 * rotation_error_deg(rigid_fit.transform)


def test_time_offset_profile_finds_the_l1_clock_lag() -> None:
    trajectory, targets = go2_scene(duration_s=30.0)
    options = LidarLidarMapOptions(
        target_motion="reference_trajectory", target_accumulate_scans=10,
        sample_every_scans=10, map_half_window_scans=2, block_duration_s=5.0,
        solver=MapSampleOptions(iterations=8),
    )  # fmt: skip
    estimate = estimate_time_offset(
        trajectory, targets, initial_guess(), options, half_width_s=0.08, step_s=0.02
    )
    assert not estimate.at_search_edge
    assert abs(estimate.value_s - CLOCK_OFFSET_S) < 0.006
    assert estimate.std_jackknife_s is not None and estimate.std_jackknife_s < 0.01
    assert len(estimate.candidates_s) == 9


def test_rigid_nearest_rejects_trajectory_only_settings() -> None:
    with pytest.raises(ValueError, match="reference_trajectory"):
        LidarLidarMapOptions(target_accumulate_scans=5)
    with pytest.raises(ValueError, match="reference_trajectory"):
        LidarLidarMapOptions(target_time_offset_s=0.01)


def test_cuda_backend_refuses_per_point_poses() -> None:
    trajectory, targets = go2_scene(duration_s=4.0)
    samples = build_trajectory_samples(trajectory, targets, OPTIONS)
    with pytest.raises(ValueError, match="per-point"):
        solve_map_extrinsic(samples, TRUE_X, OPTIONS.solver, backend="cuda")


def test_artifact_records_an_estimated_time_offset() -> None:
    offset = LidarLidarTimeOffset(
        value_s=0.037, source="estimated", std_jackknife_s=0.002, at_search_edge=False,
        search_center_s=0.0, search_half_width_s=0.08, search_step_s=0.02,
        profile=[{"offset_s": 0.04, "train_cost_per_point": 1.2}],  # type: ignore[list-item]
    )  # fmt: skip
    schema = LidarLidarExtrinsicArtifact.model_json_schema()
    assert "time_offset" in schema["properties"]
    assert offset.profile[0].offset_s == 0.04


def test_cli_go2_preset_sets_target_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = tmp_path / "initial.yaml"
    initial.write_text("T: [[1, 0, 0, 0.27], [0, 1, 0, 0], [0, 0, 1, -0.16], [0, 0, 0, 1]]\n")
    captured: dict[str, Any] = {}

    def fake_run(bag: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise ValueError("stop")

    monkeypatch.setattr("calibrex.evaluation.lidar_lidar_map.run_ros2_lidar_lidar_map", fake_run)
    exit_code = main(
        [
            "lidar-lidar", "ros2", "bag", "--reference-topic", "/livox/lidar",
            "--target-topic", "/utlidar/cloud", "--point-time-field", "offset_time",
            "--preset", "unitree-go2-l1", "--initial", str(initial),
            "--target-time-offset", "-609.25", "--time-offset-search", "0.1", "0.01",
            "--dataset-family", "go2", "--dataset-license", "private",
            "--output", str(tmp_path / "out.yaml"),
        ]
    )  # fmt: skip

    assert exit_code != 0
    options = captured["options"]
    assert options.target_motion == "reference_trajectory"
    assert options.target_accumulate_scans == 10
    assert options.target_time_offset_s == pytest.approx(-609.25)
    assert options.target_preprocess.voxel_size_m == pytest.approx(0.05)
    assert options.target_preprocess.min_range_m == pytest.approx(0.5)
    assert captured["target_point_time_field"] == "time"
    assert captured["time_offset_search"] == (0.1, 0.01)
    assert options.sensitivity_half_window_scans == 0
    assert "--preset" in captured["command"] and "--time-offset-search" in captured["command"]


def test_run_ros2_trajectory_path_writes_a_valid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calibrex.evaluation import lidar_lidar_map as module

    trajectory, targets = go2_scene(duration_s=30.0)
    monkeypatch.setattr(module, "read_reference_trajectory", lambda *a, **k: trajectory)
    monkeypatch.setattr(module, "read_target_scans", lambda *a, **k: targets)
    monkeypatch.setattr(module, "bag_input_digest", lambda bags: ("synthetic", "synthetic"))
    options = LidarLidarMapOptions(
        target_motion="reference_trajectory", target_accumulate_scans=10,
        sample_every_scans=10, map_half_window_scans=2, block_duration_s=5.0,
        target_time_offset_s=0.0, solver=MapSampleOptions(iterations=8),
    )  # fmt: skip
    artifact = module.run_ros2_lidar_lidar_map(
        tmp_path / "bag", reference_topic="/livox/lidar", target_topic="/utlidar/cloud",
        initial=initial_guess(), point_time_field="offset_time", dataset_family="synthetic",
        dataset_license="synthetic", reference_transform=TRUE_X, options=options,
        time_offset_search=(0.08, 0.02), command=["calibrex", "lidar-lidar", "ros2"],
    )  # fmt: skip
    assert artifact.time_offset is not None
    assert artifact.time_offset.source == "estimated"
    assert abs(artifact.time_offset.value_s - CLOCK_OFFSET_S) < 0.006
    assert len(artifact.time_offset.profile) == 9
    assert artifact.options["target_accumulate_scans"] == 10
    assert any("interpolated" in item for item in artifact.limitations)
    errors = [abs(record.error_to_reference or 0.0) for record in artifact.dofs]
    assert max(errors[:3]) < 0.1 and max(errors[3:]) < 0.01
    path = tmp_path / "out.yaml"
    artifact.save(path)
    from calibrex.core.lidar_lidar_extrinsic import load_lidar_lidar_extrinsic

    assert load_lidar_lidar_extrinsic(path).time_offset == artifact.time_offset


def test_cli_map_half_window_reaches_the_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = tmp_path / "initial.yaml"
    initial.write_text("T: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n")
    captured: dict[str, Any] = {}

    def fake_run(bag: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise ValueError("stop")

    monkeypatch.setattr("calibrex.evaluation.lidar_lidar_map.run_ros2_lidar_lidar_map", fake_run)
    arguments = [
        "lidar-lidar", "ros2", "bag", "--reference-topic", "/a", "--target-topic", "/b",
        "--initial", str(initial), "--dataset-family", "x", "--dataset-license", "y",
        "--output", str(tmp_path / "out.yaml"),
    ]  # fmt: skip
    assert main([*arguments, "--map-half-window-scans", "0"]) != 0
    assert captured["options"].map_half_window_scans == 0
    assert "--map-half-window-scans" in captured["command"]
    captured.clear()
    assert main([*arguments, "--map-half-window-scans", "-1"]) != 0
    assert captured == {}
