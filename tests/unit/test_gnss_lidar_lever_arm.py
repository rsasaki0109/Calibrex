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
from calibrex.core.gnss_lidar_lever_arm import (
    GnssLidarLeverArmArtifact,
    GnssLidarLeverArmProvenance,
    GnssLidarSegmentSummary,
)
from calibrex.core.validation import validate_file
from calibrex.data.rtk_slam import load_rtk_track, rtk_slam_reference_lever_arm
from calibrex.evaluation.gnss_lidar_lever_arm import (
    GnssLidarRunOptions,
    build_gnss_lidar_artifact,
    collect_windows,
    evaluate_gnss_lidar_lever_arm,
)
from calibrex.solvers.gnss_lever_arm_solver import (
    GnssLeverArmSolver,
    GnssTrackModel,
    OdometryWindow,
    weighted_procrustes,
)
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions

EPOCH = 1.76e9
LEVER = np.array([0.034, 0.0, 0.046])
DT = 0.03

_SPEC = importlib.util.spec_from_file_location(
    "ins_lidar_fixtures", Path(__file__).with_name("test_ins_lidar_hand_eye.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_SCENES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCENES)


def walking_pose(time: float, *, rotate: bool = True) -> np.ndarray:
    local = time - EPOCH
    pose = np.eye(4)
    if rotate:
        pose[:3, :3] = Rotation.from_euler(
            "xyz",
            [
                0.25 * math.sin(0.9 * local),
                0.2 * math.sin(0.7 * local + 1.0),
                0.8 * math.sin(0.3 * local) + 0.4 * math.sin(1.3 * local),
            ],
        ).as_matrix()
    pose[:3, 3] = [
        1.2 * local + 0.15 * math.sin(2.0 * math.pi * 0.9 * local),
        3.0 * math.sin(0.2 * local),
        0.3 * math.sin(0.5 * local) + 0.03 * math.sin(2.0 * math.pi * 1.8 * local),
    ]
    return pose


def synthetic(
    seed: int,
    *,
    gnss_noise: float = 0.02,
    odometry_walk: float = 0.002,
    rotate: bool = True,
    windows: int = 11,
) -> tuple[GnssTrackModel, list[OdometryWindow]]:
    rng = np.random.default_rng(seed)
    gnss_times = EPOCH + np.arange(0.0, 10.0 * windows + 10.0, 0.1)
    antenna = np.array(
        [
            walking_pose(t, rotate=rotate)[:3, 3] + walking_pose(t, rotate=rotate)[:3, :3] @ LEVER
            for t in gnss_times - DT
        ]
    )
    track = GnssTrackModel(
        gnss_times,
        antenna + rng.normal(0.0, gnss_noise, antenna.shape),
        np.full(len(gnss_times), max(gnss_noise, 0.02)),
    )
    result: list[OdometryWindow] = []
    for index in range(windows):
        times = EPOCH + 0.53 + 10.0 * index + np.arange(0.0, 10.0, 0.1)
        drift = np.eye(4)
        drift[:3, :3] = Rotation.from_rotvec(rng.normal(0.0, 0.3, 3)).as_matrix()
        drift[:3, 3] = rng.normal(0.0, 5.0, 3)
        poses = np.stack([drift @ walking_pose(t, rotate=rotate) for t in times])
        poses[:, :3, 3] += np.cumsum(rng.normal(0.0, odometry_walk, (len(times), 3)), axis=0)
        result.append(OdometryWindow(f"w{index}", index, times, poses))
    return track, result


def test_noise_free_data_recover_the_lever_arm_and_clock_offset_exactly() -> None:
    track, windows = synthetic(0, gnss_noise=0.0, odometry_walk=0.0)

    result = GnssLeverArmSolver().solve(track, windows)

    # Walking accelerations (~4 m/s^2 vertically) leave a few millimetres of
    # linear-interpolation error between 10 Hz poses even without noise.
    assert result.lever_arm_m is not None
    assert np.allclose(result.lever_arm_m, LEVER, atol=2e-3)
    assert abs(result.time_offset_s - DT) < 1e-3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noisy_walking_data_are_consistent_with_their_uncertainty(seed: int) -> None:
    """Regression: interpolating noisy GNSS biased the clock offset by ~15 ms."""

    track, windows = synthetic(seed)

    result = GnssLeverArmSolver().solve(track, windows)

    assert result.lever_arm_m is not None
    for dof, truth in zip(result.dofs, [*LEVER, DT], strict=True):
        assert abs(dof.value - truth) < 4.0 * dof.std + 1e-3, dof.name
    assert abs(result.time_offset_s - DT) < 0.005


def test_pure_translation_leaves_the_lever_arm_unobservable() -> None:
    track, windows = synthetic(0, rotate=False)

    result = GnssLeverArmSolver().solve(track, windows)

    assert all(result.dof(name).status == "unobservable" for name in ("x", "y", "z"))


def test_weighted_procrustes_recovers_a_rotation() -> None:
    rng = np.random.default_rng(4)
    truth = Rotation.from_rotvec([0.3, -0.2, 1.1]).as_matrix()
    source = rng.normal(size=(30, 3))

    estimate = weighted_procrustes(source, source @ truth.T, np.ones(30))

    assert np.allclose(estimate, truth, atol=1e-9)


def test_rtk_track_keeps_fixed_epochs_and_shares_an_origin(tmp_path: Path) -> None:
    rtk = tmp_path / "rtk.txt"
    rtk.write_text(
        "# timestamp lat lon height status blt_std\n"
        "100.0 48.0 9.0 300.0 4 0.02\n"
        "100.1 48.00001 9.0 300.0 5 0.30\n"
        "100.2 48.00001 9.0 300.0 4 0.005\n",
        encoding="utf-8",
    )

    track = load_rtk_track(rtk)
    shifted = load_rtk_track(rtk, origin=(47.99999, 9.0, 300.0))

    assert track.rejected_epochs == 1
    assert np.allclose(track.enu_m[1], [0.0, 1.112, 0.0], atol=0.002)
    assert np.allclose(track.sigma_m, [0.02, 0.01])
    assert np.allclose(shifted.enu_m[0], [0.0, 1.112, 0.0], atol=0.002)


def test_reference_lever_arm_follows_the_mid360_convention(tmp_path: Path) -> None:
    calib = tmp_path / "calib.yaml"
    calib.write_text(
        "lidar0:\n  T_lidar_imu:\n"
        "  - [1.0, 0.0, 0.0, -0.011]\n  - [0.0, 1.0, 0.0, -0.023]\n"
        "  - [0.0, 0.0, 1.0,  0.044]\n  - [0.0, 0.0, 0.0,  1.0]\n"
        "reference_offsets:\n  gnss_antenna_phase_center: [0.023, -0.023, 0.090]\n",
        encoding="utf-8",
    )

    assert np.allclose(rtk_slam_reference_lever_arm(calib), [0.034, 0.0, 0.046])


def test_windows_restart_at_gnss_gaps() -> None:
    scene = _SCENES._scene()
    step = _SCENES._motion(0.4, 1.0)
    times = [EPOCH + 0.1 * index for index in range(60)]
    scans = [
        (t, _SCENES._view(scene, np.linalg.matrix_power(step, i))) for i, t in enumerate(times)
    ]
    covered = np.r_[np.arange(-1.0, 2.6, 0.1), np.arange(3.6, 7.0, 0.1)] + EPOCH
    track = GnssTrackModel(covered, np.zeros((len(covered), 3)), np.full(len(covered), 0.02))
    options = GnssLidarRunOptions(
        window_duration_s=1.0,
        odometry=ScanOdometryOptions(voxel_size_m=0.3, local_map_scans=3),
    )

    segmenter = collect_windows(track, scans, options)

    assert segmenter.segments == 2
    assert segmenter.scans_covered < segmenter.scans_read
    for window in segmenter.windows:
        assert not (window.times_s[0] < EPOCH + 3.0 < window.times_s[-1])


@functools.cache
def _evaluation() -> Any:
    track, windows = synthetic(3, windows=15)
    return evaluate_gnss_lidar_lever_arm(track, windows, reference_lever_arm=LEVER)


def _artifact() -> GnssLidarLeverArmArtifact:
    return build_gnss_lidar_artifact(
        _evaluation(),
        GnssLidarRunOptions(),
        segments=GnssLidarSegmentSummary(
            scans_read=1500,
            scans_in_gnss_coverage=1500,
            odometry_segments=15,
            unreliable_registrations=0,
            windows=15,
            gnss_epochs_used=1600,
            gnss_epochs_rejected=0,
        ),
        provenance=GnssLidarLeverArmProvenance(
            generator="pytest",
            generator_version="1",
            dataset_family="synthetic",
            sequence_ids=["walking"],
            input_sha256="b" * 64,
            input_digest_scope="synthetic",
            dataset_license="synthetic",
        ),
        reference="synthetic truth",
        limitations=["synthetic"],
    )


def test_evaluation_detects_lever_arm_controls_on_held_out_windows(tmp_path: Path) -> None:
    evaluation = _evaluation()
    records = {record.name: record for record in evaluation.records}

    assert evaluation.jackknife_fits >= 3
    for name in ("x", "y", "z"):
        control = records[name].known_bad_control
        assert control is not None and control.detected, name
        assert abs(records[name].error_to_reference or 0.0) < 4.0 * records[name].std_reported

    path = tmp_path / "gnss-lidar.yaml"
    _artifact().save(path)
    assert validate_file(path).kind == "gnss-lidar-lever-arm"


def test_artifact_refuses_a_pass_without_every_lever_arm_component() -> None:
    payload = _artifact().model_dump(mode="python")
    for record in payload["dofs"]:
        if record["name"] == "z":
            record["status"] = "unobservable"
    payload["calibrated_dofs"] = [
        record["name"] for record in payload["dofs"] if record["status"] == "estimated"
    ]
    payload["policy_status"] = "pass"

    with pytest.raises(ValueError, match="pass verdict"):
        GnssLidarLeverArmArtifact.model_validate(payload)


def test_cli_pairs_bags_with_rtk_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(sequences: Any, calib: Any, **kwargs: Any) -> GnssLidarLeverArmArtifact:
        captured["sequences"] = sequences
        return _artifact()

    monkeypatch.setattr("calibrex.cli.main.run_rtk_slam_lever_arm", fake_run)
    output = tmp_path / "out.yaml"
    base = ["gnss-lidar", "rtk-slam", "--calib", "c.yaml", "--output", str(output)]

    assert main([*base, "--bag", "a", "--rtk", "a.txt", "--bag", "b", "--rtk", "b.txt"]) == 0
    assert [(str(bag), str(rtk)) for bag, rtk in captured["sequences"]] == [
        ("a", "a.txt"),
        ("b", "b.txt"),
    ]
    assert validate_file(output).valid
    assert main([*base, "--bag", "a", "--rtk", "a.txt", "--bag", "b"]) == 2
