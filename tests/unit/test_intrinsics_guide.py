from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from calibrex.evaluation.board_camera_lidar import Checkerboard
from calibrex.evaluation.intrinsics_guide import (
    TILTS,
    GuideOptions,
    IntrinsicsGuide,
    Tilt,
    camchain_entry,
    draw_guide,
    run_guided_calibration,
    status_lines,
    view_tilts,
)

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 480, 360
TRUE_K = np.array([[420.0, 0.0, 243.0], [0.0, 418.0, 176.0], [0.0, 0.0, 1.0]])
TRUE_DISTORTION = np.array([-0.12, 0.05, 0.0008, -0.0005, 0.0])
BOARD = Checkerboard(8, 6, 0.03)
TILT_ROTATION: dict[Tilt, tuple[str, float]] = {
    "left": ("y", 1.0),
    "right": ("y", -1.0),
    "up": ("x", -1.0),
    "down": ("x", 1.0),
}

_RAYS: np.ndarray | None = None
SUPERSAMPLE = tuple((-0.375 + 0.25 * i, -0.375 + 0.25 * j) for i in range(4) for j in range(4))


def _rays() -> np.ndarray:
    """Normalized rays of 4 x 4 sub-pixel samples per pixel under the true (distorted) camera."""

    global _RAYS
    if _RAYS is None:
        u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
        samples = []
        for du, dv in SUPERSAMPLE:
            pixels = np.stack([u.ravel() + du, v.ravel() + dv], axis=1).reshape(-1, 1, 2)
            normalized = cv2.undistortPointsIter(
                pixels, TRUE_K, TRUE_DISTORTION, None, None, (cv2.TERM_CRITERIA_COUNT, 20, 0)
            ).reshape(-1, 2)
            samples.append(np.column_stack([normalized, np.ones(len(normalized))]))
        _RAYS = np.concatenate(samples)
    return _RAYS


def render(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """The board (one white square of margin) seen by the true camera; grey elsewhere."""

    rays = _rays()
    # Board frame: p_board = R^T (s ray - t); intersect z = 0.
    in_board = rays @ rotation  # R^T ray, row-wise
    origin = -(rotation.T @ translation)
    scale = -origin[2] / in_board[:, 2]
    points = origin[:2] + scale[:, None] * in_board[:, :2]
    s = BOARD.square_m
    column = np.floor(points[:, 0] / s + 1.0)
    row = np.floor(points[:, 1] / s + 1.0)
    inside = (
        (scale > 0)
        & (points[:, 0] >= -2 * s)
        & (points[:, 0] <= (BOARD.columns + 1) * s)
        & (points[:, 1] >= -2 * s)
        & (points[:, 1] <= (BOARD.rows + 1) * s)
    )
    squares = (column >= 0) & (column <= BOARD.columns) & (row >= 0) & (row <= BOARD.rows)
    image = np.full(len(rays), 110, dtype=np.float64)
    image[inside] = 235.0
    dark = inside & squares & ((column + row) % 2 == 0)
    image[dark] = 25.0
    image = image.reshape(len(SUPERSAMPLE), HEIGHT, WIDTH).mean(axis=0)
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.4), 0, 255).astype(np.uint8)


def pose(
    center_px: tuple[float, float], depth_m: float, tilt: Tilt | None, angle_deg: float = 35.0
) -> tuple[np.ndarray, np.ndarray]:
    """A board pose centred on a pixel, at a depth, tilted one way (``None``: facing)."""

    rotation = np.eye(3)
    if tilt is not None:
        axis, sign = TILT_ROTATION[tilt]
        rotation = Rotation.from_euler(axis, sign * angle_deg, degrees=True).as_matrix()
    center_board = np.array(
        [(BOARD.columns - 1) * BOARD.square_m / 2, (BOARD.rows - 1) * BOARD.square_m / 2, 0.0]
    )
    ray = np.linalg.solve(TRUE_K, [center_px[0], center_px[1], 1.0])
    translation = depth_m * ray - rotation @ center_board
    return rotation, translation


def test_tilt_directions_follow_the_board_edge_turned_away() -> None:
    options = GuideOptions()
    for tilt in TILTS:
        rotation, _ = pose((240, 180), 0.5, tilt)
        assert view_tilts(rotation, options) == {tilt}
    assert view_tilts(np.eye(3), options) == set()


def test_the_rendered_board_is_found_where_the_true_camera_projects_it() -> None:
    rotation, translation = pose((200, 150), 0.45, "left")
    guide = IntrinsicsGuide(BOARD, WIDTH, HEIGHT)
    frame = guide.update(render(rotation, translation))
    assert frame.corners is not None
    expected, _ = cv2.projectPoints(
        BOARD.object_points(), cv2.Rodrigues(rotation)[0], translation, TRUE_K, TRUE_DISTORTION
    )
    expected = expected.reshape(-1, 2)
    # 8 x 6 inner corners are 9 x 7 squares: the board looks the same turned half a turn,
    # so the corners may come in either order (the calibration does not mind).
    error = min(
        np.max(np.linalg.norm(frame.corners - order, axis=1))
        for order in (expected, expected[::-1])
    )
    assert error < 0.3


def test_a_frame_is_kept_only_when_the_board_is_still_and_adds_a_cell() -> None:
    guide = IntrinsicsGuide(BOARD, WIDTH, HEIGHT)
    assert guide.update(np.full((HEIGHT, WIDTH), 110, np.uint8)).advice.kind == "find"
    image = render(*pose((240, 180), 0.5, None))
    first = guide.update(image)
    assert not first.still and not first.captured and first.advice.kind == "hold"
    second = guide.update(image)
    assert second.still and second.captured
    assert second.advice.kind == "move" and second.advice.target_cell is not None
    column, row = second.advice.target_cell
    assert guide.coverage[row, column] == 0
    # The same view again adds nothing.
    assert not guide.update(image).captured
    moved = guide.update(render(*pose((250, 185), 0.5, None)))
    assert not moved.still and moved.advice.kind == "hold"


def _follow(guide: IntrinsicsGuide, steps: int = 200) -> list[str]:
    """A person who does what the guide says: two identical frames per step."""

    rng = np.random.default_rng(4)
    kinds = []
    center, tilt = (240.0, 180.0), None
    depth = 0.5
    for _ in range(steps):
        image = render(*pose(center, depth, tilt))
        guide.update(image)
        frame = guide.update(image)
        kinds.append(frame.advice.kind)
        advice = frame.advice
        if advice.kind == "done":
            break
        tilt = None
        if advice.kind == "find":
            center, depth = (240.0, 180.0), 0.6
        elif advice.kind == "move":
            # Over the target cell, as far as the whole board stays in view.
            columns, rows = guide.options.grid
            column, row = advice.target_cell
            depth = 0.6
            half_x = (BOARD.columns + 1) * BOARD.square_m / 2 * TRUE_K[0, 0] / depth + 12
            half_y = (BOARD.rows + 1) * BOARD.square_m / 2 * TRUE_K[1, 1] / depth + 12
            center = (
                float(np.clip((column + 0.5) * WIDTH / columns, half_x, WIDTH - half_x)),
                float(np.clip((row + 0.5) * HEIGHT / rows, half_y, HEIGHT - half_y)),
            )
        elif advice.kind == "tilt":
            center, tilt, depth = (240.0, 180.0), advice.tilt, 0.55
        elif advice.kind == "vary":
            center = (rng.uniform(90, WIDTH - 90), rng.uniform(80, HEIGHT - 80))
            tilt = TILTS[int(rng.integers(4))]
            depth = float(rng.uniform(0.45, 0.8))
    return kinds


def test_following_the_advice_calibrates_the_camera() -> None:
    guide = IntrinsicsGuide(BOARD, WIDTH, HEIGHT)
    kinds = _follow(guide)

    assert kinds[-1] == "done", kinds
    assert guide.coverage.all() and guide.tilts == set(TILTS)
    estimate = guide.estimate
    assert estimate is not None
    # The corners carry the renderer's ~0.1 px aliasing, correlated within a view.
    assert estimate.fx == pytest.approx(TRUE_K[0, 0], rel=0.01)
    assert estimate.fy == pytest.approx(TRUE_K[1, 1], rel=0.01)
    assert estimate.cx == pytest.approx(TRUE_K[0, 2], abs=3.0)
    assert estimate.cy == pytest.approx(TRUE_K[1, 2], abs=3.0)
    assert len(guide.views) >= guide.options.min_views_done
    assert estimate.distortion[0] == pytest.approx(TRUE_DISTORTION[0], abs=0.02)
    assert estimate.rms_px < 0.3
    # Coverage first, then the tilts, then precision.
    assert kinds.index("tilt") > kinds.index("move")
    assert all(line for line in status_lines(guide)[:2])


def test_the_estimate_is_a_kalibr_camchain_entry() -> None:
    from calibrex.evaluation.visual_rotation import CameraModel

    guide = IntrinsicsGuide(BOARD, WIDTH, HEIGHT)
    _follow(guide)
    assert guide.estimate is not None
    entry = camchain_entry(guide.estimate)
    camera = CameraModel.from_kalibr(entry)
    assert (camera.fx, camera.cy) == (guide.estimate.fx, guide.estimate.cy)
    assert entry["distortion_model"] == "radtan" and entry["resolution"] == [WIDTH, HEIGHT]


def test_the_overlay_marks_the_target_and_the_text() -> None:
    guide = IntrinsicsGuide(BOARD, WIDTH, HEIGHT)
    image = render(*pose((240, 180), 0.5, None))
    guide.update(image)
    frame = guide.update(image)
    canvas = draw_guide(image, guide, frame)
    assert canvas.shape == (HEIGHT, WIDTH, 3)
    column, row = frame.advice.target_cell
    columns, rows = guide.options.grid
    x0, y0 = int(column * WIDTH / columns), int(row * HEIGHT / rows)
    # The target cell's border is drawn in yellow (BGR 0, 220, 255).
    border = canvas[y0 + 1, x0 + 5 : x0 + 30]
    assert np.any(np.all(np.abs(border.astype(int) - [0, 220, 255]) < 40, axis=1))


def test_the_live_loop_runs_headless_records_and_saves(tmp_path: Path) -> None:
    guide = IntrinsicsGuide(BOARD, WIDTH, HEIGHT)
    images: list[np.ndarray] = []
    original = guide.update

    def recording(gray: np.ndarray) -> object:
        images.append(gray)
        return original(gray)

    guide.update = recording  # type: ignore[method-assign]
    _follow(guide)
    saved: list[object] = []
    record = tmp_path / "guide.avi"

    run = run_guided_calibration(
        iter(images), BOARD, show=False, record=record, save=saved.append, stop_when_done=True
    )

    assert run.done and run.saved and saved == [run.estimate]
    assert run.frames <= len(images)
    assert record.exists() and record.stat().st_size > 0


def test_cli_guides_a_video_and_writes_a_camchain(tmp_path: Path) -> None:
    from calibrex.cli.main import main

    guide = IntrinsicsGuide(BOARD, WIDTH, HEIGHT)
    images: list[np.ndarray] = []
    original = guide.update

    def recording(gray: np.ndarray) -> object:
        images.append(gray)
        return original(gray)

    guide.update = recording  # type: ignore[method-assign]
    _follow(guide)
    video = tmp_path / "board.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("no lossless video codec in this OpenCV build")
    for image in images:
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()
    output = tmp_path / "camchain.yaml"

    status = main(
        ["intrinsics", "guide", "--board", "8x6x0.03", "--video", str(video), "--no-window",
         "--output", str(output), "--camera-name", "cam1"]
    )  # fmt: skip

    assert status == 0
    chain = yaml.safe_load(output.read_text())
    fx, _, _, cy = chain["cam1"]["intrinsics"]
    assert fx == pytest.approx(TRUE_K[0, 0], rel=0.01)
    assert math.isclose(cy, TRUE_K[1, 2], abs_tol=3.0)
