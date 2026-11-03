"""Camera-LiDAR extrinsics from a checkerboard on synthetic images and scans."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.evaluation.board_camera_lidar import (
    BoardPose,
    Checkerboard,
    calibrate_board,
    detect_board,
    extract_board_plane,
)
from calibrex.evaluation.visual_rotation import CameraModel

cv2 = pytest.importorskip("cv2")

BOARD = Checkerboard(7, 5, 0.08)
CAMERA = CameraModel(800.0, 800.0, 640.0, 480.0)
SIZE = (1280, 960)
# LiDAR x forward, y left, z up; the camera's optical frame: x right, y down, z forward
AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
TRUE = np.eye(4)
TRUE[:3, :3] = Rotation.from_rotvec(np.radians([2.0, -1.0, 3.0])).as_matrix() @ AXES
TRUE[:3, 3] = [0.10, -0.05, 0.08]
ROUGH = np.eye(4)
ROUGH[:3, :3] = AXES


def _board_pose(yaw_deg: float, pitch_deg: float, centre: tuple[float, float, float]):
    rotation = Rotation.from_euler("yx", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    middle = np.array([3 * BOARD.square_m, 2 * BOARD.square_m, 0.0])
    return rotation, np.asarray(centre) - rotation @ middle


def _render(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    ppm, margin = 1000.0, 2 * BOARD.square_m  # texture: 1 mm per pixel, a white border
    width = int(((BOARD.columns + 1) * BOARD.square_m + 2 * margin) * ppm)
    height = int(((BOARD.rows + 1) * BOARD.square_m + 2 * margin) * ppm)
    texture = np.full((height, width), 255, dtype=np.uint8)
    offset = margin + BOARD.square_m
    for row in range(-1, BOARD.rows):
        for column in range(-1, BOARD.columns):
            if (row + column) % 2 == 0:
                x0 = int((offset + column * BOARD.square_m) * ppm)
                y0 = int((offset + row * BOARD.square_m) * ppm)
                size = int(BOARD.square_m * ppm)
                texture[y0 : y0 + size, x0 : x0 + size] = 0
    k = np.array([[CAMERA.fx, 0, CAMERA.cx], [0, CAMERA.fy, CAMERA.cy], [0, 0, 1]])
    board_to_image = k @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    texture_to_board = np.array([[1 / ppm, 0, -offset], [0, 1 / ppm, -offset], [0, 0, 1]])
    image = cv2.warpPerspective(
        texture, board_to_image @ texture_to_board, SIZE, borderValue=128, flags=cv2.INTER_AREA
    )
    return np.asarray(image, dtype=np.uint8)


def _scan(rotation, translation, rng, *, range_bias_m=0.0) -> np.ndarray:
    """Board points (with a white border) and a wall and floor behind it, in the LiDAR frame."""

    (x0, y0), (x1, y1) = BOARD.squares_extent()
    grid = np.array(
        [
            [x, y, 0.0]
            for x in np.arange(x0 - 0.05, x1 + 0.05, 0.015)
            for y in np.arange(y0 - 0.05, y1 + 0.05, 0.015)
        ]
    )
    board = grid @ rotation.T + translation
    board += rotation[:, 2] * (rng.normal(0.0, 0.003, (len(board), 1)) + range_bias_m)
    wall = np.column_stack([rng.uniform(-3, 3, 4000), rng.uniform(-2, 2, 4000), np.full(4000, 6.0)])
    floor = np.column_stack([rng.uniform(-3, 3, 4000), np.full(4000, 1.2), rng.uniform(1, 6, 4000)])
    camera_points = np.vstack([board, wall, floor])
    return (camera_points - TRUE[:3, 3]) @ TRUE[:3, :3]


def test_board_spec_parses_inner_corners_and_square() -> None:
    assert Checkerboard.parse("7x5x0.08") == BOARD
    assert Checkerboard.parse(" 9 X 6 x .025 ") == Checkerboard(9, 6, 0.025)
    with pytest.raises(ValueError, match="COLUMNSxROWSxSQUARE_M"):
        Checkerboard.parse("7x5")


def test_detect_board_recovers_the_board_plane_and_corners() -> None:
    """The corners, as a set: a board with an even number of squares each way looks the same
    turned by 180 deg, so OpenCV may number them from the opposite corner."""

    rotation, translation = _board_pose(25.0, -15.0, (0.1, -0.05, 2.0))
    view = detect_board(_render(rotation, translation), BOARD, CAMERA)

    assert view is not None
    assert view.reprojection_px < 0.5
    detected = BOARD.object_points() @ view.rotation.T + view.translation
    truth = BOARD.object_points() @ rotation.T + translation
    nearest = np.min(np.linalg.norm(detected[:, None, :] - truth[None, :, :], axis=2), axis=1)
    assert nearest.max() < 0.004
    normal = np.asarray(view.plane.normal)
    assert math.degrees(math.acos(abs(float(normal @ rotation[:, 2])))) < 0.2
    assert abs(abs(view.plane.offset_m) - abs(float(rotation[:, 2] @ translation))) < 0.003
    blank = np.full((SIZE[1], SIZE[0]), 128, dtype=np.uint8)
    assert detect_board(blank, BOARD, CAMERA) is None


def test_the_board_plane_is_found_among_wall_and_floor() -> None:
    rotation, translation = _board_pose(-20.0, 10.0, (-0.2, 0.1, 2.5))
    rng = np.random.default_rng(1)
    view = detect_board(_render(rotation, translation), BOARD, CAMERA)
    assert view is not None

    found = extract_board_plane(_scan(rotation, translation, rng), view, BOARD, ROUGH)

    assert found is not None
    normal_camera = TRUE[:3, :3] @ np.asarray(found.plane.normal)
    assert math.degrees(math.acos(abs(float(normal_camera @ rotation[:, 2])))) < 0.5
    assert found.rms_m < 0.005


def test_a_noisy_board_plane_does_not_depend_on_the_point_order() -> None:
    """Range noise wider than the RANSAC band must not leave the plane to the sample drawn."""

    rotation, translation = _board_pose(15.0, -10.0, (0.1, 0.0, 3.0))
    rng = np.random.default_rng(6)
    view = detect_board(_render(rotation, translation), BOARD, CAMERA)
    assert view is not None
    camera_points = _scan(rotation, translation, rng) @ TRUE[:3, :3].T + TRUE[:3, 3]
    board = np.abs((camera_points - translation) @ rotation[:, 2]) < 0.01
    camera_points[board] += rotation[:, 2] * rng.normal(0.0, 0.02, (int(board.sum()), 1))
    points = (camera_points - TRUE[:3, 3]) @ TRUE[:3, :3]

    planes = []
    for seed in range(4):
        order = np.random.default_rng(seed).permutation(len(points))
        found = extract_board_plane(points[order], view, BOARD, TRUE, margin_m=0.15, gate_m=0.15)
        assert found is not None
        normal = np.asarray(found.plane.normal)
        sign = 1.0 if normal @ np.ones(3) > 0 else -1.0
        planes.append(np.append(sign * normal, sign * found.plane.offset_m))
    np.testing.assert_allclose(planes, [planes[0]] * len(planes), atol=1e-6)


def _poses(angles, rng, *, range_bias_m=0.0) -> list[BoardPose]:
    poses = []
    for index, (yaw, pitch) in enumerate(angles):
        centre = (rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2), rng.uniform(2.0, 3.5))
        rotation, translation = _board_pose(yaw, pitch, centre)
        view = detect_board(_render(rotation, translation), BOARD, CAMERA)
        assert view is not None, (yaw, pitch)
        bias = rng.normal(0.0, range_bias_m) if range_bias_m else 0.0
        poses.append(
            BoardPose(f"{index:02d}", view, _scan(rotation, translation, rng, range_bias_m=bias))
        )
    return poses


def test_tilted_board_poses_recover_all_six_axes() -> None:
    rng = np.random.default_rng(2)
    angles = [(y, p) for y in (-30.0, 0.0, 30.0) for p in (-25.0, 0.0, 25.0)]

    result = calibrate_board(_poses(angles, rng), BOARD, ROUGH)

    assert result.status == "converged" and result.transform_camera_lidar is not None
    error = Rotation.from_matrix(result.transform_camera_lidar[:3, :3] @ TRUE[:3, :3].T)
    assert math.degrees(error.magnitude()) < 0.2
    np.testing.assert_allclose(result.transform_camera_lidar[:3, 3], TRUE[:3, 3], atol=0.01)
    assert [axis.name for axis in result.axes if axis.observed] == [
        "roll",
        "pitch",
        "yaw",
        "x",
        "y",
        "z",
    ]
    assert len(result.poses_used) == 9 and result.poses_rejected == ()


def test_boards_turned_only_left_and_right_leave_the_vertical_unobserved() -> None:
    """Normals in one plane do not constrain the translation along the third direction."""

    rng = np.random.default_rng(3)
    angles = [(y, p) for y in (-35.0, -20.0, 0.0, 20.0, 35.0) for p in (-1.0, 1.0)]

    result = calibrate_board(_poses(angles, rng, range_bias_m=0.01), BOARD, ROUGH)

    assert result.status == "converged"
    by_name = {axis.name: axis for axis in result.axes}
    assert not by_name["y"].observed  # the camera's vertical
    assert by_name["y"].value == pytest.approx(ROUGH[1, 3], abs=0.005)  # kept at the prior
    (direction,) = result.unobserved_directions
    assert abs(direction[1]) > 0.9
    assert by_name["x"].observed and by_name["z"].observed
    assert result.normal_singular_values is not None
    assert result.normal_singular_values[2] < 0.1 * result.normal_singular_values[0]


def test_too_few_poses_give_no_solution() -> None:
    rng = np.random.default_rng(4)
    result = calibrate_board(_poses([(-20.0, 10.0), (20.0, -10.0)], rng), BOARD, ROUGH)
    assert result.status == "insufficient_poses" and result.transform_camera_lidar is None
