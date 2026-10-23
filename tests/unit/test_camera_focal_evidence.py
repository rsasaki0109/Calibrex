"""Exact rate fixtures separate focal evidence policy from image tracking."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.core.camera_focal_scale import (
    camera_focal_scale_json_schema,
    load_camera_focal_scale,
)
from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
from calibrex.evaluation import camera_focal, visual_rotation
from calibrex.evaluation.camera_focal import estimate_focal_scale
from calibrex.evaluation.visual_rotation import CameraModel
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries


def _constant_rate_windows(
    *, missing_holdout_axes: tuple[int, ...] = (), scale: float = 1.0,
) -> tuple[GyroSeries, list[OdometryWindow]]:
    times = np.arange(-0.5, 24.5, 0.01)
    rate = np.tile([0.12, 0.16, 0.20], (len(times), 1))
    windows = []
    for index in range(12):
        angular_rate = np.array([0.12, 0.16, 0.20])
        if index % 3 == 1:
            angular_rate[list(missing_holdout_axes)] = 0.0
        rate[(times >= 2 * index) & (times < 2 * index + 2)] = angular_rate
        local = np.arange(21) * 0.05
        poses = np.repeat(np.eye(4)[None], len(local), axis=0)
        # Each window has constant body rate, so Log(R_i.T R_j)/dt is exact.
        camera_rate = angular_rate * np.array([1 / scale, 1 / scale, 1])
        poses[:, :3, :3] = Rotation.from_rotvec(local[:, None] * camera_rate).as_matrix()
        windows.append(OdometryWindow(str(index), index, 2 * index + 0.25 + local, poses))
    return GyroSeries(times, rate), windows


@pytest.mark.parametrize("scale", [0.97, 1.0, 1.03])
def test_exact_rates_recover_training_and_heldout_ratios(scale: float) -> None:
    gyro, windows = _constant_rate_windows(scale=scale)
    result = estimate_focal_scale(gyro, windows, np.eye(3), 0.0, np.zeros(3))

    assert result.status == "pass"
    assert result.ratio == pytest.approx([1 / scale, 1 / scale, 1], abs=1e-10)
    assert result.holdout_ratio == pytest.approx(result.ratio, abs=1e-10)
    assert np.all(result.ratio_std < 1e-10)


@pytest.mark.parametrize("axes", [(0,), (1,), (2,), (0, 1, 2)])
def test_missing_heldout_excitation_cannot_pass(axes: tuple[int, ...]) -> None:
    gyro, windows = _constant_rate_windows(missing_holdout_axes=axes)
    result = estimate_focal_scale(gyro, windows, np.eye(3), 0.0, np.zeros(3))

    assert result.ratio == pytest.approx(np.ones(3), abs=1e-10)
    assert np.all(np.isfinite(result.ratio_std))
    assert np.all(np.isnan(result.holdout_ratio[list(axes)]))
    assert result.status == "inconclusive"
    assert "held-out" in " ".join(result.reasons)


@pytest.mark.parametrize("missing_holdout", [False, True])
def test_artifacts_require_finite_evidence_and_preserve_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_holdout: bool,
) -> None:
    from calibrex.data import livox_ros2, ros2_camera_imu

    gyro, windows = _constant_rate_windows(missing_holdout_axes=(1,) if missing_holdout else ())
    result = estimate_focal_scale(gyro, windows, np.eye(3), 0.0, np.zeros(3))
    monkeypatch.setattr(camera_focal, "estimate_focal_scale", lambda *args: result)
    monkeypatch.setattr(visual_rotation, "track_camera_rotations",
                        lambda *args: SimpleNamespace(windows=lambda: windows))
    monkeypatch.setattr(ros2_camera_imu, "iter_gray_images", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(ros2_camera_imu, "load_ros2_imu", lambda *args: SimpleNamespace(
        times_s=gyro.times_s, gyro_rps=gyro.gyro_rps,
    ))

    def unexpected_digest(*args: object) -> None:
        pytest.fail("incomplete evidence must be rejected before constructing the artifact")

    monkeypatch.setattr(livox_ros2, "bag_input_digest", unexpected_digest if missing_holdout
                        else lambda *args: ("synthetic-digest", None))
    rotation = load_imu_lidar_rotation(
        Path("docs/assets/hilti2022_camera_imu/cam1_exp21.yaml")
    )
    def run() -> object:
        return camera_focal.run_camera_focal_check(
            "synthetic-unused-bag", image_topic="/image", imu_topic="/imu",
            camera=CameraModel(400, 400, 320, 240), rotation_artifact=rotation,
            rotation_artifact_sha256="a" * 64, dataset_family="synthetic",
            dataset_license="synthetic",
        )

    if missing_holdout:
        with pytest.raises(ValueError, match=r"camera focal evidence is incomplete.*held-out"):
            run()
    else:
        artifact = run()
        assert isinstance(artifact, camera_focal.CameraFocalScaleArtifact)
        path = tmp_path / "focal.json"
        artifact.save(path)
        jsonschema.validate(json.loads(path.read_text()), camera_focal_scale_json_schema())
        loaded = load_camera_focal_scale(path)
        assert loaded.policy_status == "pass"
        assert loaded.rate_ratio == pytest.approx(np.ones(3), abs=1e-10)
        assert loaded.provenance.dataset_family == "synthetic"
        assert loaded.provenance.rotation_artifact_sha256 == "a" * 64
