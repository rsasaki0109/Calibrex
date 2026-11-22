from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from calibrex.evaluation.intrinsics_guide import IntrinsicsEstimate
from calibrex.evaluation.rotation_self_calibration import SelfCalibration
from calibrex.evaluation.stereo_self_calibration import baseline_direction, relative_rotation

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 320, 240
K_LEFT = np.array([[280.0, 0.0, 162.0], [0.0, 278.7, 117.3], [0.0, 0.0, 1.0]])
D_LEFT = np.array([-0.12, 0.05, 0.0008, -0.0005, 0.0])
K_RIGHT = np.array([[284.0, 0.0, 157.5], [0.0, 283.0, 121.0], [0.0, 0.0, 1.0]])
D_RIGHT = np.array([-0.10, 0.03, -0.0004, 0.0006, 0.0])
R_LR = Rotation.from_euler("yx", [2.0, 1.0], degrees=True).as_matrix()
"""The right camera's axes in the left camera's frame."""
BASELINE_M = 0.10
ROOM_HALF_M = 8.0
CELL_M = 0.5
NEAR_ROOM_HALF_M = 2.0
NEAR_CELL_M = 0.15
SUBSAMPLES = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))
_RAYS: dict[str, np.ndarray] = {}


def _rays(camera: str) -> np.ndarray:
    if camera not in _RAYS:
        k, d = (K_LEFT, D_LEFT) if camera == "left" else (K_RIGHT, D_RIGHT)
        u, v = np.meshgrid(np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float))
        rays = []
        for du, dv in SUBSAMPLES:
            pixels = np.stack([u.ravel() + du, v.ravel() + dv], 1).reshape(-1, 1, 2)
            normalized = cv2.undistortPointsIter(
                pixels, k, d, None, None, (cv2.TERM_CRITERIA_COUNT, 20, 0)
            ).reshape(-1, 2)
            rays.append(np.column_stack([normalized, np.ones(len(normalized))]))
        _RAYS[camera] = np.concatenate(rays)
    return _RAYS[camera]


def _hash(a: np.ndarray, b: np.ndarray, face: np.ndarray) -> np.ndarray:
    x = (a * 73856093) ^ (b * 19349663) ^ (face * 83492791)
    x = (x ^ (x >> 13)) * 1274126177
    return ((x ^ (x >> 16)) & 0xFFFF) / 65535.0


def render(
    camera: str, rig: np.ndarray, room_half_m: float = ROOM_HALF_M, cell_m: float = CELL_M
) -> np.ndarray:
    """One camera of a rig turned about its middle, in a textured cube room."""

    offset = np.array([-BASELINE_M / 2, 0.0, 0.0] if camera == "left" else [BASELINE_M / 2, 0, 0])
    rotation = rig if camera == "left" else rig @ R_LR
    center = rig @ offset
    directions = _rays(camera) @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(
            directions > 0,
            (room_half_m - center) / directions,
            (-room_half_m - center) / directions,
        )
    exits[~np.isfinite(exits)] = np.inf
    axis = np.argmin(exits, axis=1)
    rows = np.arange(len(axis))
    hit = center + directions * exits[rows, axis][:, None]
    face = axis * 2 + (directions[rows, axis] > 0)
    a = np.where(axis == 0, hit[:, 1], hit[:, 0]) + 50
    b = np.where(axis == 2, hit[:, 1], hit[:, 2]) + 50
    coarse = _hash(
        np.floor(a / cell_m).astype(np.int64), np.floor(b / cell_m).astype(np.int64), face
    )
    fine = _hash(
        np.floor(3 * a / cell_m).astype(np.int64) + 7,
        np.floor(3 * b / cell_m).astype(np.int64) + 11,
        face + 9,
    )
    image = (40 + 150 * coarse + 50 * fine).reshape(len(SUBSAMPLES), HEIGHT, WIDTH).mean(axis=0)
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.5), 0, 255).astype(np.uint8)


def turning(time_s: float) -> np.ndarray:
    rotvec = [
        0.45 * np.sin(0.9 * time_s),
        0.6 * np.sin(0.55 * time_s + 0.3),
        0.35 * np.sin(0.75 * time_s + 1.0),
    ]
    return Rotation.from_rotvec(rotvec).as_matrix()


def _self_calibration(times: np.ndarray, rotations: np.ndarray) -> SelfCalibration:
    estimate = IntrinsicsEstimate(320, 240, 1, 1, 1, 1, (0, 0, 0, 0), {}, 0.0, len(times))
    return SelfCalibration(estimate, (0, 0, 0), len(times), 0, 0, (), tuple(times), rotations)


def test_the_relative_rotation_aligns_the_cameras_rotations() -> None:
    times = np.arange(0.0, 8.0, 0.2)
    world_left = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    world_right = Rotation.from_euler("x", -50, degrees=True).as_matrix()
    # Each camera's keyframe orientations in its own arbitrary world frame.
    left = np.array([world_left @ turning(t) for t in times])
    right = np.array([world_right @ turning(t) @ R_LR for t in times[::2] + 0.05])

    rotation, rms, intervals = relative_rotation(
        _self_calibration(times, left), _self_calibration(times[::2] + 0.05, right)
    )

    assert Rotation.from_matrix(rotation.T @ R_LR).magnitude() < np.radians(0.05)
    # The left orientations are interpolated at the right keyframe times.
    assert rms < 0.2 and intervals >= 10


def test_the_baseline_direction_comes_from_near_matches() -> None:
    def estimate(k: np.ndarray, d: np.ndarray) -> IntrinsicsEstimate:
        return IntrinsicsEstimate(
            WIDTH, HEIGHT, k[0, 0], k[1, 1], k[0, 2], k[1, 2], tuple(d[:4]), {}, 0.0, 0
        )

    pairs = [
        (render("left", turning(t), NEAR_ROOM_HALF_M, NEAR_CELL_M),
         render("right", turning(t), NEAR_ROOM_HALF_M, NEAR_CELL_M))
        for t in (0.5, 2.0, 4.0, 6.0)
    ]  # fmt: skip
    direction, matches, parallax = baseline_direction(
        pairs, estimate(K_LEFT, D_LEFT), estimate(K_RIGHT, D_RIGHT), R_LR
    )

    assert direction is not None and matches > 500 and parallax > 2.0
    expected = -(R_LR.T @ np.array([1.0, 0.0, 0.0]))
    assert np.degrees(np.arccos(np.clip(direction @ expected, -1.0, 1.0))) < 3.0
    # A rotation known to 0.5 deg is too coarse for this parallax: no direction.
    unsure, _, _ = baseline_direction(
        pairs, estimate(K_LEFT, D_LEFT), estimate(K_RIGHT, D_RIGHT), R_LR, 0.5
    )
    assert unsure is None


def test_cli_calibrates_a_side_by_side_stereo_rig(tmp_path: Path) -> None:
    from calibrex.cli.main import main

    video = tmp_path / "stereo.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (2 * WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("no lossless video codec in this OpenCV build")
    for time_s in np.arange(0.0, 8.0, 0.1):
        rig = turning(time_s)
        frame = np.hstack([render("left", rig), render("right", rig)])
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    output = tmp_path / "camchain.yaml"

    status = main(
        ["intrinsics", "rotate", "--video", str(video), "--side-by-side", "--baseline-m",
         str(BASELINE_M), "--output", str(output), "--json"]
    )  # fmt: skip

    assert status == 0
    chain = yaml.safe_load(output.read_text())
    for name, k in (("cam0", K_LEFT), ("cam1", K_RIGHT)):
        fx, fy, cx, cy = chain[name]["intrinsics"]
        # Each lens is 5 cm from the rotation centre at 8-14 m, which lengthens the focal
        # length by about 0.4 % and moves the principal point by a few pixels.
        assert fx == pytest.approx(k[0, 0], rel=0.012)
        assert fy == pytest.approx(k[1, 1], rel=0.012)
        assert cx == pytest.approx(k[0, 2], abs=5.0)
        assert cy == pytest.approx(k[1, 2], abs=5.0)
    # 8-14 m away the 10 cm baseline gives under 2 px of parallax: no direction, no
    # T_cn_cnm1, and a warning saying what to record.
    assert "T_cn_cnm1" not in chain["cam1"]
