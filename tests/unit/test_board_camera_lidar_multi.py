"""Several checkerboards in one image (a single shot): find every board, solve from all.

KITTI's calibration recordings show a dozen boards of different sizes at once, the vehicle
standing still. These tests render such a shot (boards of three patterns and square sizes,
turned left/right and up/down) into one 1392 x 512 image and a LiDAR scan of the same
boards in front of a wall, and check that ``find_boards`` returns every board with its own
grid and that ``calibrate_board`` with a board per pose recovers the extrinsic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.evaluation.board_camera_lidar import (
    BoardPose,
    Checkerboard,
    board_view,
    calibrate_board,
    find_boards,
)
from calibrex.evaluation.visual_rotation import CameraModel

cv2 = pytest.importorskip("cv2")

CAMERA = CameraModel(720.0, 720.0, 696.0, 256.0)
SIZE = (1392, 512)
AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
TRUE = np.eye(4)
TRUE[:3, :3] = Rotation.from_rotvec(np.radians([1.5, -2.0, 1.0])).as_matrix() @ AXES
TRUE[:3, 3] = [-0.004, -0.076, -0.27]  # a KITTI-like lever arm: the LiDAR behind the camera
START = np.eye(4)
START[:3, :3] = AXES
# (columns, rows, square_m, centre in the camera frame, yaw, pitch)
SHOT = [
    (7, 5, 0.10, (-2.6, -0.3, 6.5), 30.0, 15.0),
    (5, 4, 0.10, (-1.0, 0.4, 5.5), -20.0, -20.0),
    (8, 6, 0.08, (0.3, -0.5, 5.0), 10.0, 25.0),
    (6, 4, 0.10, (1.8, 0.3, 6.0), -35.0, 10.0),
    (7, 5, 0.08, (3.2, -0.2, 7.0), 25.0, -25.0),
]


def _pose(board: Checkerboard, centre, yaw: float, pitch: float):
    rotation = Rotation.from_euler("yx", [yaw, pitch], degrees=True).as_matrix()
    middle = np.array(
        [(board.columns - 1) * board.square_m / 2, (board.rows - 1) * board.square_m / 2, 0.0]
    )
    return rotation, np.asarray(centre, dtype=np.float64) - rotation @ middle


def _paint(image: np.ndarray, board: Checkerboard, rotation, translation) -> None:
    square_px = 60
    margin = square_px  # a white border of one square
    columns, rows = board.columns + 1, board.rows + 1
    texture = np.full(
        (rows * square_px + 2 * margin, columns * square_px + 2 * margin), 255, np.uint8
    )
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                y0, x0 = margin + row * square_px, margin + column * square_px
                texture[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    scale = board.square_m / square_px
    offset = (margin + square_px) * scale  # the first inner corner
    texture_to_board = np.array([[scale, 0, -offset], [0, scale, -offset], [0, 0, 1]])
    k = np.array([[CAMERA.fx, 0, CAMERA.cx], [0, CAMERA.fy, CAMERA.cy], [0, 0, 1]])
    homography = (
        k @ np.column_stack([rotation[:, 0], rotation[:, 1], translation]) @ texture_to_board
    )
    warped = cv2.warpPerspective(texture, homography, SIZE, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(
        np.full_like(texture, 255), homography, SIZE, flags=cv2.INTER_NEAREST
    )
    image[mask > 0] = warped[mask > 0]


def _shot():
    image = np.full((SIZE[1], SIZE[0]), 110, np.uint8)
    truths = []
    for columns, rows, square, centre, yaw, pitch in SHOT:
        board = Checkerboard(columns, rows, square)
        rotation, translation = _pose(board, centre, yaw, pitch)
        _paint(image, board, rotation, translation)
        truths.append((board, rotation, translation))
    return image, truths


def _matching_truth(found, truths):
    """The rendered board whose projected centre is nearest the found grid's centre."""

    k = np.array([[CAMERA.fx, 0, CAMERA.cx], [0, CAMERA.fy, CAMERA.cy], [0, 0, 1]])
    centre = found.pixels.mean(axis=0)
    distances = []
    for board, rotation, translation in truths:
        middle = (
            rotation
            @ np.array(
                [(board.columns - 1) * board.square_m / 2, (board.rows - 1) * board.square_m / 2, 0]
            )
            + translation
        )
        projected = k @ middle
        distances.append(np.linalg.norm(projected[:2] / projected[2] - centre))
    return truths[int(np.argmin(distances))]


def test_find_boards_returns_every_board_of_a_single_shot() -> None:
    image, truths = _shot()

    found = find_boards(image)

    assert len(found) == len(SHOT)
    for item in found:
        board, rotation, translation = _matching_truth(item, truths)
        # the grid may come out turned by 90 deg: the same corners, the other way round
        assert {item.columns, item.rows} == {board.columns, board.rows}
        view = board_view(item.pixels, item.board(board.square_m), CAMERA)
        assert view is not None and view.reprojection_px < 0.5
        normal = np.asarray(view.plane.normal)
        assert math.degrees(math.acos(min(1.0, abs(float(normal @ rotation[:, 2]))))) < 0.5
        # PnP on a board of 12 px squares at 6 m: its distance within 0.5 % of the range
        offset = abs(float(rotation[:, 2] @ translation))
        assert abs(abs(view.plane.offset_m) - offset) < 0.005 * float(np.linalg.norm(translation))
    blank = np.full((SIZE[1], SIZE[0]), 110, np.uint8)
    assert find_boards(blank) == []


def _scan(truths, rng) -> np.ndarray:
    """Every board (with a white border) and a wall behind them, in the LiDAR frame."""

    parts = []
    for board, rotation, translation in truths:
        (x0, y0), (x1, y1) = board.squares_extent()
        grid = np.array(
            [
                [x, y, 0.0]
                for x in np.arange(x0 - 0.05, x1 + 0.05, 0.02)
                for y in np.arange(y0 - 0.05, y1 + 0.05, 0.02)
            ]
        )
        points = grid @ rotation.T + translation
        points += rotation[:, 2] * rng.normal(0.0, 0.01, (len(points), 1))
        parts.append(points)
    wall = np.column_stack([rng.uniform(-5, 5, 6000), rng.uniform(-2, 2, 6000), np.full(6000, 9.0)])
    camera_points = np.vstack([*parts, wall])
    return (camera_points - TRUE[:3, 3]) @ TRUE[:3, :3]


def test_one_shot_of_several_boards_recovers_the_extrinsic() -> None:
    image, truths = _shot()
    rng = np.random.default_rng(7)
    cloud = _scan(truths, rng)
    poses = []
    for index, item in enumerate(find_boards(image)):
        board, _rotation, _translation = _matching_truth(item, truths)
        own = item.board(board.square_m)
        view = board_view(item.pixels, own, CAMERA)
        assert view is not None
        poses.append(BoardPose(f"board{index}", view, cloud, board=own))

    result = calibrate_board(poses, None, START)

    assert result.status == "converged" and result.transform_camera_lidar is not None
    error = Rotation.from_matrix(result.transform_camera_lidar[:3, :3] @ TRUE[:3, :3].T)
    assert math.degrees(error.magnitude()) < 0.3
    # five boards leave stds of about 2 cm: each translation axis within three of its own
    by_name = {axis.name: axis for axis in result.axes}
    for index, name in enumerate(("x", "y", "z")):
        error = abs(result.transform_camera_lidar[index, 3] - TRUE[index, 3])
        assert error <= 3.0 * by_name[name].std, (name, error, by_name[name].std)
    assert by_name["z"].observed and all(by_name[name].observed for name in ("roll", "pitch"))
    assert len(result.poses_used) == len(SHOT)


def test_a_pose_without_a_board_needs_one_from_the_calibration() -> None:
    image, truths = _shot()
    item = find_boards(image)[0]
    board, _rotation, _translation = _matching_truth(item, truths)
    view = board_view(item.pixels, item.board(board.square_m), CAMERA)
    assert view is not None
    with pytest.raises(ValueError, match="every pose needs a board"):
        calibrate_board([BoardPose("a", view, np.zeros((10, 3)))], None, START)


def test_a_given_pattern_finds_only_boards_of_that_pattern() -> None:
    image, _truths = _shot()

    found = find_boards(image, pattern=(8, 6))

    assert [(item.columns, item.rows) for item in found] == [(8, 6)]
    assert find_boards(image, pattern=(9, 7)) == []
