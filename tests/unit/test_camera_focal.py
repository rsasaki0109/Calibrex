from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.evaluation.camera_focal import estimate_focal_scale
from calibrex.evaluation.visual_rotation import CameraModel, track_camera_rotations
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries


def attitude(time: np.ndarray | float) -> Rotation:
    t = np.atleast_1d(time)
    return Rotation.from_rotvec(
        np.column_stack(
            [0.15 * np.sin(1.3 * t), 0.18 * np.sin(1.1 * t + 1.0), 0.12 * np.sin(0.9 * t + 2.0)]
        )
    )


def test_a_wrong_focal_length_shows_in_the_x_and_y_rate_ratios() -> None:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (1400, 1800), dtype=np.uint8), (0, 0), 3)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    true_f = 400.0
    matrix = np.array([[true_f, 0, 320.0], [0, true_f, 240.0], [0, 0, 1.0]])
    base = np.array([[1, 0, -580.0], [0, 1, -460.0], [0, 0, 1]])
    frames = []
    for time in np.arange(0.0, 45.0, 0.1):
        rotation = attitude(time).as_matrix()[0]  # R_world_camera
        warp = matrix @ rotation.T @ np.linalg.inv(matrix) @ base
        image = cv2.warpPerspective(
            texture, np.linalg.inv(warp), (640, 480), flags=cv2.WARP_INVERSE_MAP
        )
        frames.append((float(time), image))
    gyro_times = np.arange(-1.0, 46.0, 0.005)
    step = 1e-3
    rates = (attitude(gyro_times - step).inv() * attitude(gyro_times + step)).as_rotvec() / (
        2 * step
    )
    gyro = GyroSeries(gyro_times, rates)
    # Track with a focal length 3 % too long.
    used = CameraModel(1.03 * true_f, 1.03 * true_f, 320.0, 240.0)
    track = track_camera_rotations(frames, used)

    result = estimate_focal_scale(gyro, track.windows(), np.eye(3), 0.0, np.zeros(3))

    assert result.ratio[0] == pytest.approx(1 / 1.03, abs=0.006)
    assert result.ratio[1] == pytest.approx(1 / 1.03, abs=0.006)
    assert result.ratio[2] == pytest.approx(1.0, abs=0.006)
