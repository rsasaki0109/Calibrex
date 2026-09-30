from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.imu_lidar_translation import ImuLidarTranslationArtifact
from calibrex.core.validation import validate_file
from calibrex.evaluation.imu_lidar_translation import (
    ImuLidarTranslationEvaluation,
    evaluate_imu_lidar_translation,
    score_imu_lidar_translation_candidate,
)
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_translation_solver import (
    ImuPreintegrator,
    solve_translation,
    window_systems,
)

EPOCH = 1.76e9
TRUE_ROTATION = Rotation.from_euler("xyz", [0.8, -1.5, 2.0], degrees=True).as_matrix()
TRUE_TRANSLATION = np.array([0.011, 0.023, -0.044])
TRUE_DT = 0.012
ACCEL_BIAS = np.array([0.03, -0.02, 0.05])
GYRO_BIAS = np.array([0.004, -0.002, 0.003])
GRAVITY = np.array([0.0, 0.0, -9.81])
ROTATION_ARTIFACT = (
    Path(__file__).resolve().parents[2] / "docs/assets/mid360_imu_lidar_rotation_rtk_slam_seq2.yaml"
)


def attitude(time: float, *, planar: bool) -> np.ndarray:
    local = time - EPOCH
    yaw = 0.8 * math.sin(0.3 * local) + 0.4 * math.sin(1.3 * local)
    if planar:
        return Rotation.from_euler("z", yaw).as_matrix()
    # Hand-held-like motion: slow sway plus a faster wobble on every axis.
    roll = 0.35 * math.sin(1.1 * local) + 0.12 * math.sin(7.0 * local)
    pitch = 0.3 * math.sin(0.9 * local + 1.0) + 0.1 * math.sin(8.5 * local + 0.5)
    return Rotation.from_euler("xyz", [roll, pitch, yaw + 0.1 * math.sin(6.0 * local)]).as_matrix()


def position(time: float, *, planar: bool) -> np.ndarray:
    local = time - EPOCH
    z = 0.0 if planar else 0.15 * math.sin(0.7 * local)
    return np.array([0.6 * math.sin(0.4 * local), 0.5 * math.cos(0.5 * local), z])


def synthetic(
    *, planar: bool, seed: int = 0
) -> tuple[ImuPreintegrator, list[OdometryWindow]]:
    """IMU at TRUE_TRANSLATION in the LiDAR frame; IMU clock = LiDAR clock + TRUE_DT."""

    rng = np.random.default_rng(seed)
    imu_times = EPOCH + TRUE_DT + np.arange(0.0, 90.0, 0.005)
    step = 1e-3
    gyro, accel = [], []
    for imu_time in imu_times:
        lidar_time = imu_time - TRUE_DT

        def imu_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
            rotation = attitude(t, planar=planar)
            origin = position(t, planar=planar) + rotation @ TRUE_TRANSLATION
            return rotation @ TRUE_ROTATION, origin

        before, _ = imu_pose(lidar_time - step)
        now, now_position = imu_pose(lidar_time)
        after, _ = imu_pose(lidar_time + step)
        _, back = imu_pose(lidar_time - step)
        _, ahead = imu_pose(lidar_time + step)
        gyro.append(Rotation.from_matrix(before.T @ after).as_rotvec() / (2 * step))
        acceleration = (ahead - 2.0 * now_position + back) / step**2
        accel.append(now.T @ (acceleration - GRAVITY) + ACCEL_BIAS)
    gyro_array = np.array(gyro) + GYRO_BIAS + rng.normal(0.0, 0.002, (len(imu_times), 3))
    accel_array = np.array(accel) + rng.normal(0.0, 0.02, (len(imu_times), 3))
    imu = ImuPreintegrator(imu_times, gyro_array, accel_array, GYRO_BIAS)
    windows = []
    for index in range(9):
        lidar_times = EPOCH + 0.5 + 10.0 * index + np.arange(0.0, 9.9, 0.1)
        poses = np.stack([np.eye(4)] * len(lidar_times))
        for k, t in enumerate(lidar_times):
            noise = Rotation.from_rotvec(rng.normal(0.0, 0.0003, 3)).as_matrix()
            poses[k, :3, :3] = attitude(t, planar=planar) @ noise
            poses[k, :3, 3] = position(t, planar=planar) + rng.normal(0.0, 0.002, 3)
        windows.append(OdometryWindow(f"w{index}", index, lidar_times, poses))
    return imu, windows


@functools.cache
def _evaluation(planar: bool) -> ImuLidarTranslationEvaluation:
    imu, windows = synthetic(planar=planar, seed=3)
    return evaluate_imu_lidar_translation(
        imu, windows, TRUE_ROTATION, TRUE_DT, reference_translation=TRUE_TRANSLATION
    )


def test_solver_recovers_the_lever_arm_from_three_dimensional_motion() -> None:
    imu, windows = synthetic(planar=False)
    result = solve_translation(window_systems(windows, imu, TRUE_ROTATION, TRUE_DT))

    assert result.status == "converged"
    assert result.translation_m is not None
    assert np.allclose(result.translation_m, TRUE_TRANSLATION, atol=0.003)


def test_evaluation_passes_and_detects_every_control() -> None:
    evaluation = _evaluation(planar=False)

    assert evaluation.policy_status == "pass", evaluation.policy_reasons
    for record in evaluation.records:
        assert record.known_bad_control is not None and record.known_bad_control.detected
        assert abs(record.error_to_reference or 0.0) < 3.0 * record.std_reported + 0.002
    assert evaluation.gravity_norm_median_mps2 == pytest.approx(9.81, abs=0.05)


def test_yaw_only_motion_leaves_the_vertical_lever_arm_unobservable() -> None:
    evaluation = _evaluation(planar=True)
    records = {record.name: record for record in evaluation.records}

    assert evaluation.policy_status in {"inconclusive", "warn"}
    assert records["z"].status == "unobservable"


def _artifact(evaluation: ImuLidarTranslationEvaluation) -> ImuLidarTranslationArtifact:
    translation = evaluation.result.translation_m
    assert translation is not None
    return ImuLidarTranslationArtifact.model_validate(
        {
            "solver_status": evaluation.result.status,
            "policy_status": evaluation.policy_status,
            "policy_reasons": list(evaluation.policy_reasons),
            "calibrated_dofs": [r.name for r in evaluation.records if r.status == "estimated"],
            "translation_m": list(translation),
            "rotation_input": {
                "policy_status": "pass",
                "rotation_quat_xyzw": list(Rotation.from_matrix(TRUE_ROTATION).as_quat()),
                "time_offset_s": TRUE_DT,
                "gyro_bias_rps": list(GYRO_BIAS),
            },
            "axes": [record.model_dump() for record in evaluation.records],
            "segment_sensitivity": [fit.model_dump() for fit in evaluation.segment_fits],
            "options": {},
            "windows": {
                "windows": 9,
                "segments": evaluation.segments,
                "scan_rows": evaluation.scan_rows,
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


def test_artifact_validates_and_refuses_a_pass_without_every_axis(tmp_path: Path) -> None:
    artifact = _artifact(_evaluation(planar=False))
    path = tmp_path / "translation.yaml"
    artifact.save(path)
    assert validate_file(path).kind == "imu-lidar-translation"

    payload = artifact.model_dump(mode="python")
    payload["axes"][2]["status"] = "unobservable"
    payload["calibrated_dofs"].remove("z")
    with pytest.raises(ValueError, match="x, y, and z"):
        ImuLidarTranslationArtifact.model_validate(payload)


def test_reported_std_cannot_hide_the_segment_sensitivity() -> None:
    payload = _artifact(_evaluation(planar=False)).model_dump(mode="python")
    payload["axes"][0]["std_segment_sensitivity"] = payload["axes"][0]["std_reported"] + 0.01

    with pytest.raises(ValueError, match="segment-sensitivity"):
        ImuLidarTranslationArtifact.model_validate(payload)


def test_cli_passes_the_rotation_artifact_and_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rotation_path = ROTATION_ARTIFACT
    captured: dict[str, Any] = {}

    def fake_run(bags: Any, profile: str, rotation: Any, **kwargs: Any) -> Any:
        captured.update(kwargs, bags=bags, profile=profile, rotation=rotation)
        return _artifact(_evaluation(planar=False))

    monkeypatch.setattr("calibrex.cli.main.run_livox_imu_lidar_translation", fake_run)
    output = tmp_path / "out.yaml"
    exit_code = main(
        [
            "imu-lidar",
            "livox-translation",
            "bag_a",
            "--profile",
            "rtk-slam",
            "--rotation",
            str(rotation_path),
            "--dataset-family",
            "rtk_slam",
            "--dataset-license",
            "custom",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured["rotation"].schema_version == "slac.imu_lidar_rotation/v0.1"
    assert len(captured["rotation_artifact_sha256"]) == 64
    assert np.allclose(captured["reference_translation"], [0.011, 0.02329, -0.04412])
    assert validate_file(output).valid


def test_candidate_score_prefers_the_true_lever_arm_on_shared_spans() -> None:
    imu, windows = synthetic(planar=False, seed=4)
    spans = [(float(w.times_s[0]), float(w.times_s[-1])) for w in windows[1::3]]
    true = score_imu_lidar_translation_candidate(
        imu, windows, TRUE_ROTATION, TRUE_DT, TRUE_TRANSLATION, spans
    )
    wrong = score_imu_lidar_translation_candidate(
        imu, windows, TRUE_ROTATION, TRUE_DT, TRUE_TRANSLATION + np.array([0.0, 0.0, 0.05]), spans
    )
    # A candidate whose own segmentation split windows differently is still
    # scored on exactly the same spans.
    halves = [
        OdometryWindow(f"{w.window_id}{part}", w.block, w.times_s[cut], w.poses[cut])
        for w in windows
        for part, cut in (("a", slice(0, 50)), ("b", slice(50, None)))
    ]
    split = score_imu_lidar_translation_candidate(
        imu, halves, TRUE_ROTATION, TRUE_DT, TRUE_TRANSLATION, spans
    )

    assert set(true.holdout_span_rms_m) == {round(start, 1) for start, _ in spans}
    assert set(split.holdout_span_rms_m) == set(true.holdout_span_rms_m)
    for key, value in true.holdout_span_rms_m.items():
        assert wrong.holdout_span_rms_m[key] > value
