from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.imu_lidar_rotation import ImuLidarRotationArtifact
from calibrex.data.ros2_camera_imu import to_gray
from calibrex.evaluation.visual_rotation import (
    CameraModel,
    VisualRotationOptions,
    pure_rotation_ransac,
    relative_rotation,
    track_camera_rotations,
)


def _bearings(count: int, rng: np.random.Generator) -> np.ndarray:
    rays = np.column_stack([rng.uniform(-0.6, 0.6, (count, 2)), np.ones(count)])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def test_to_gray_handles_mono_color_and_row_padding() -> None:
    mono = to_gray(bytes([1, 2, 0, 3, 4, 0]), height=2, width=2, step=3, encoding="mono8")
    color = to_gray(bytes([255, 0, 0] * 2), height=1, width=2, step=6, encoding="rgb8")
    mono16 = to_gray(
        np.array([0x1234, 0xFF00], "<u2").tobytes(), height=1, width=2, step=4, encoding="mono16"
    )

    assert mono.tolist() == [[1, 2], [3, 4]]
    assert color.tolist() == [[76, 76]]
    assert mono16.tolist() == [[0x12, 0xFF]]


def test_pure_rotation_ransac_rejects_outliers() -> None:
    rng = np.random.default_rng(0)
    truth = Rotation.from_euler("xyz", [1.0, -2.0, 0.5], degrees=True).as_matrix()
    previous = _bearings(200, rng)
    current = previous @ truth.T
    current[:40] = _bearings(40, rng)  # 20 % outliers

    fit = pure_rotation_ransac(previous, current, 0.002, 200, rng)

    assert fit is not None
    rotation, inliers = fit
    assert Rotation.from_matrix(rotation.T @ truth).magnitude() < 1e-6
    assert inliers[40:].all() and inliers[:40].sum() < 5


def test_kalibr_camera_models_parse() -> None:
    camera = CameraModel.from_kalibr(
        {
            "camera_model": "pinhole",
            "intrinsics": [350.0, 351.0, 360.0, 250.0],
            "distortion_model": "equidistant",
            "distortion_coeffs": [-0.03, -0.009, 0.009, -0.004],
        }
    )
    assert camera.distortion_model == "equidistant"
    with pytest.raises(ValueError, match="unsupported camera model"):
        CameraModel.from_kalibr({"camera_model": "omni", "intrinsics": [1, 1, 0, 0]})


def test_essential_matrix_recovers_rotation_with_parallax() -> None:
    pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    truth = Rotation.from_euler("xyz", [0.8, 1.5, -1.0], degrees=True).as_matrix()
    points = np.column_stack([rng.uniform(-2, 2, (300, 2)), rng.uniform(1.5, 4.0, 300)])
    translation = np.array([0.08, 0.0, 0.02])
    moved = points @ truth.T + translation
    previous = points[:, :2] / points[:, 2:]
    current = moved[:, :2] / moved[:, 2:]

    fit = relative_rotation(previous, current, 350.0, VisualRotationOptions(), rng)

    assert fit is not None
    rotation, model, _ = fit
    assert model == "essential"
    assert np.degrees(Rotation.from_matrix(rotation.T @ truth).magnitude()) < 0.05


def test_tracking_a_rotating_camera_recovers_the_orientation() -> None:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(2)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (1200, 1600), dtype=np.uint8), (0, 0), 3)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    camera = CameraModel(400.0, 400.0, 320.0, 240.0)
    matrix = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]])
    # A texture at infinity seen by a camera rotating about every axis: no
    # parallax, the degenerate case for the essential matrix.
    base = np.array([[1, 0, -480.0], [0, 1, -360.0], [0, 0, 1]])
    frames, truth = [], []
    for index in range(30):
        angles = np.radians(
            [2.0 * np.sin(0.3 * index), 1.5 * np.sin(0.25 * index + 1), 0.1 * index]
        )
        rotation = Rotation.from_rotvec(angles).as_matrix()  # R_world_camera
        warp = matrix @ rotation.T @ np.linalg.inv(matrix) @ np.linalg.inv(base)
        image = cv2.warpPerspective(
            texture, np.linalg.inv(warp), (640, 480), flags=cv2.WARP_INVERSE_MAP
        )
        frames.append((0.1 * index, image))
        truth.append(rotation)

    track = track_camera_rotations(frames, camera)

    assert track.stats.failed == 0
    # Per-pair rotations are what the calibration uses (as mean rates).
    errors = [
        np.degrees(
            Rotation.from_matrix(
                (track.orientations[k].T @ track.orientations[k + 1]).T
                @ (truth[k].T @ truth[k + 1])
            ).magnitude()
        )
        for k in range(len(truth) - 1)
    ]
    assert np.median(errors) < 0.1
    # Degenerate essential fits can still be off by about a degree on a pair;
    # the robust (Huber) rate alignment downweights such pairs.
    assert max(errors) < 1.5


def test_cli_passes_the_camchain_reference_and_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camchain = tmp_path / "camchain.yaml"
    camchain.write_text(
        "cam0:\n"
        "  camera_model: pinhole\n"
        "  intrinsics: [350, 350, 360, 250]\n"
        "  distortion_model: equidistant\n"
        "  distortion_coeffs: [0, 0, 0, 0]\n"
        "  rostopic: /cam0/image_raw\n"
        "  timeshift_cam_imu: 0.002\n"
        "  T_cam_imu: [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_run(bags: Any, **kwargs: Any) -> ImuLidarRotationArtifact:
        captured.update(kwargs, bags=bags)
        raise ValueError("stop here")

    monkeypatch.setattr(
        "calibrex.evaluation.camera_imu_rotation.run_ros2_camera_imu_rotation", fake_run
    )
    exit_code = main(
        [
            "camera-imu",
            "rotation",
            "bag_a",
            "--camchain",
            str(camchain),
            "--imu-topic",
            "/imu",
            "--dataset-family",
            "hilti2022",
            "--dataset-license",
            "custom",
            "--output",
            str(tmp_path / "out.yaml"),
        ]
    )

    assert exit_code != 0
    assert captured["image_topic"] == "/cam0/image_raw"
    assert np.allclose(captured["reference_rotation"], [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert "timeshift_cam_imu 0.002" in captured["reference"]
