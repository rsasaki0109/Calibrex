from __future__ import annotations

import functools
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.ins_lidar_hand_eye import (
    InsLidarHandEyeArtifact,
    InsLidarHandEyeProvenance,
    InsLidarOdometrySummary,
    load_ins_lidar_hand_eye,
)
from calibrex.core.validation import validate_file
from calibrex.data.kitti_ins_lidar import load_kitti_ins_lidar_drive
from calibrex.evaluation import ins_lidar_hand_eye as evaluation
from calibrex.evaluation.ins_lidar_hand_eye import (
    InsLidarRunOptions,
    build_ins_lidar_artifact,
    build_sensor_motions,
    evaluate_ins_lidar_hand_eye,
)
from calibrex.solvers.scan_to_scan_odometry import (
    ScanOdometryOptions,
    preprocess_scan,
    register_point_to_plane,
    run_scan_to_scan_odometry,
)

_SPEC = importlib.util.spec_from_file_location(
    "trajectory_hand_eye_fixtures", Path(__file__).with_name("test_trajectory_hand_eye.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURES)


def _scene(seed: int = 3) -> np.ndarray:
    """Points on a ground plane, two facades, and a few boxes (world frame)."""

    rng = np.random.default_rng(seed)
    parts = [
        np.column_stack(
            [rng.uniform(-40, 40, 20000), rng.uniform(-12, 12, 20000), np.full(20000, -1.7)]
        ),
        np.column_stack(
            [rng.uniform(-40, 40, 6000), np.full(6000, 9.0), rng.uniform(-1.7, 6.0, 6000)]
        ),
        np.column_stack(
            [rng.uniform(-40, 40, 6000), np.full(6000, -9.0), rng.uniform(-1.7, 6.0, 6000)]
        ),
    ]
    for center in ((12.0, 4.0), (-8.0, -5.0), (25.0, -3.0), (-20.0, 6.0)):
        for axis in (0, 1):
            for side in (-1.0, 1.0):
                face = rng.uniform(-1.0, 1.0, (1500, 3))
                face[:, axis] = side
                face[:, :2] = face[:, :2] * 1.2 + np.array(center)
                face[:, 2] = face[:, 2] * 1.0 - 0.7
                parts.append(face)
    return np.vstack(parts)


def _view(scene: np.ndarray, pose: np.ndarray) -> np.ndarray:
    local = (scene - pose[:3, 3]) @ pose[:3, :3]
    return local[np.linalg.norm(local, axis=1) < 60.0]


def _motion(x: float, yaw_deg: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    transform[:3, 3] = [x, 0.1 * x, 0.0]
    return transform


def test_point_to_plane_registration_recovers_a_known_motion() -> None:
    scene = _scene()
    truth = _motion(0.8, 2.0)
    options = ScanOdometryOptions(voxel_size_m=0.3)
    target = preprocess_scan(_view(scene, np.eye(4)), options)
    source = preprocess_scan(_view(scene, truth), options)

    registration = register_point_to_plane(source, target, np.eye(4), options)

    error = np.linalg.inv(truth) @ registration.transform
    assert registration.converged
    assert np.linalg.norm(error[:3, 3]) < 0.01
    assert np.degrees(np.linalg.norm(Rotation.from_matrix(error[:3, :3]).as_rotvec())) < 0.05


def test_odometry_scales_the_constant_velocity_guess_across_frame_gaps() -> None:
    scene = _scene()
    step = _motion(0.9, 1.0)
    frame_numbers = [0, 1, 2, 4, 5]  # frame 3 dropped, like KITTI 0009
    poses = [np.linalg.matrix_power(step, number) for number in frame_numbers]
    times = [0.1 * number for number in frame_numbers]
    scans = [_view(scene, pose) for pose in poses]

    result = run_scan_to_scan_odometry(scans, ScanOdometryOptions(voxel_size_m=0.3), times_s=times)

    assert result.all_converged
    final_error = np.linalg.inv(poses[-1]) @ result.poses[-1]
    assert np.linalg.norm(final_error[:3, 3]) < 0.05


def _write_fake_kitti_drive(root: Path) -> Path:
    day = root / "2011_01_01"
    drive = day / "2011_01_01_drive_0001_sync"
    (drive / "oxts" / "data").mkdir(parents=True)
    (drive / "velodyne_points" / "data").mkdir(parents=True)
    (day / "calib_imu_to_velo.txt").write_text(
        "calib_time: fake\nR: 1 0 0 0 1 0 0 0 1\nT: -0.8 0.3 -0.8\n", encoding="utf-8"
    )
    stamps = [f"2011-01-01 12:00:00.{index}00000000" for index in range(5)]
    (drive / "oxts" / "timestamps.txt").write_text("\n".join(stamps) + "\n", encoding="utf-8")
    for index in range(5):
        values = [49.0, 8.4 + index * 1e-5, 110.0, 0.0, 0.0, 0.1, *([0.0] * 24)]
        (drive / "oxts" / "data" / f"{index:010d}.txt").write_text(
            " ".join(str(value) for value in values), encoding="utf-8"
        )
    velodyne_stamps = [
        f"2011-01-01 12:00:00.{index}50000000" if index != 2 else "" for index in range(5)
    ]
    (drive / "velodyne_points" / "timestamps.txt").write_text(
        "\n".join(velodyne_stamps) + "\n", encoding="utf-8"
    )
    for index in (0, 1, 3, 4):
        np.zeros((4, 4), dtype=np.float32).tofile(
            drive / "velodyne_points" / "data" / f"{index:010d}.bin"
        )
    return drive


def test_kitti_frames_are_associated_by_index_across_dropped_frames(tmp_path: Path) -> None:
    """Regression: positional pairing shifted every frame after a dropped one."""

    drive = load_kitti_ins_lidar_drive(_write_fake_kitti_drive(tmp_path))

    assert [frame.frame_index for frame in drive.velodyne_frames] == [0, 1, 3, 4]
    offsets = [frame.time_s - drive.velodyne_frames[0].time_s for frame in drive.velodyne_frames]
    assert np.allclose(offsets, [0.0, 0.1, 0.3, 0.4], atol=1e-6)
    assert drive.missing_velodyne_indices == (2,)
    assert np.allclose(drive.vendor_t_imu_lidar[:3, 3], [0.8, -0.3, 0.8])
    assert len(drive.input_sha256) == 64


@functools.cache
def _planar_evaluation() -> Any:
    reference, motions = _FIXTURES.synthetic(planar=True, duration_s=60.0)
    return evaluate_ins_lidar_hand_eye(
        reference,
        motions,
        InsLidarRunOptions(),
        reference_transform=_FIXTURES.true_extrinsic(),
    )


def _artifact(result: Any) -> InsLidarHandEyeArtifact:
    return build_ins_lidar_artifact(
        result,
        InsLidarRunOptions(),
        odometry=InsLidarOdometrySummary(
            frame_count=600, registration_count=599, converged_registrations=599
        ),
        provenance=InsLidarHandEyeProvenance(
            generator="pytest",
            generator_version="1",
            dataset_family="synthetic",
            drive_ids=["planar"],
            input_sha256="a" * 64,
            dataset_license="synthetic",
        ),
        reference_calibration="synthetic truth",
        limitations=["synthetic"],
    )


def test_planar_evaluation_is_an_honest_partial_calibration(tmp_path: Path) -> None:
    result = _planar_evaluation()
    records = {record.name: record for record in result.records}

    assert result.policy_status == "inconclusive"
    assert records["z"].status == "unobservable"
    assert result.jackknife_fits >= 3
    for name in ("roll", "pitch", "yaw", "x", "y"):
        assert records[name].status == "estimated", name
        control = records[name].known_bad_control
        assert control is not None and control.detected, name
    assert abs(records["yaw"].error_to_reference or 0.0) < 0.1
    assert records["yaw"].unit == "deg"

    artifact = _artifact(result)
    path = tmp_path / "ins-lidar.yaml"
    artifact.save(path)
    assert validate_file(path).kind == "ins-lidar-hand-eye"
    assert "z" not in load_ins_lidar_hand_eye(path).calibrated_dofs


def test_artifact_rejects_a_pass_with_an_unobservable_extrinsic_dof() -> None:
    payload = _artifact(_planar_evaluation()).model_dump(mode="python")
    payload["policy_status"] = "pass"

    with pytest.raises(ValueError, match="pass verdict"):
        InsLidarHandEyeArtifact.model_validate(payload)
    payload["policy_status"] = "inconclusive"
    payload["calibrated_dofs"] = ["roll"]
    with pytest.raises(ValueError, match="calibrated_dofs"):
        InsLidarHandEyeArtifact.model_validate(payload)


def test_reported_std_is_never_below_the_jackknife_spread() -> None:
    record = _artifact(_planar_evaluation()).dofs[0].model_dump(mode="python")
    record["std_jackknife"] = record["std_reported"] * 10.0 + 1.0

    with pytest.raises(ValueError, match="std_reported"):
        type(_artifact(_planar_evaluation()).dofs[0]).model_validate(record)


def test_sensor_motions_never_cross_blocks() -> None:
    times = [100.0 + 0.1 * index for index in range(120)]
    poses = [np.eye(4) for _ in times]

    motions = build_sensor_motions(times, poses, InsLidarRunOptions(block_duration_s=5.0))

    assert motions
    for motion in motions:
        assert int((motion.start_s - 100.0) // 5.0) == int((motion.end_s - 100.0) // 5.0)


def test_cli_parses_priors_and_reports_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(drives: Any, options: Any, **kwargs: Any) -> InsLidarHandEyeArtifact:
        captured["drives"] = drives
        captured["priors"] = options.solver.priors
        return _artifact(_planar_evaluation())

    monkeypatch.setattr("calibrex.cli.main.run_kitti_ins_lidar_hand_eye", fake_run)
    output = tmp_path / "out.yaml"

    exit_code = main(
        [
            "ins-lidar",
            "kitti",
            str(tmp_path / "a"),
            str(tmp_path / "b"),
            "--prior",
            "z=0.9:0.02",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    assert len(captured["drives"]) == 2
    assert [(prior.dof, prior.value, prior.sigma) for prior in captured["priors"]] == [
        ("z", 0.9, 0.02)
    ]
    assert '"status": "inconclusive"' in capsys.readouterr().out
    assert validate_file(output).valid
    assert main(["ins-lidar", "kitti", "x", "--prior", "yaw=1:1", "--output", str(output)]) == 2


def test_limitations_name_the_known_kitti_caveats() -> None:
    joined = " ".join(evaluation.KITTI_LIMITATIONS)

    assert "not independent" in joined and "planar" in joined and math.isfinite(1.0)
