from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from calibrex.evaluation.board_camera_lidar import Checkerboard
from calibrex.evaluation.intrinsics_guide import TILTS, Tilt
from calibrex.evaluation.intrinsics_stereo_guide import (
    StereoGuide,
    draw_stereo,
    run_stereo_guide,
    stereo_camchain,
)

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 480, 360
CAMERAS = {
    "left": (
        np.array([[420.0, 0.0, 243.0], [0.0, 418.0, 176.0], [0.0, 0.0, 1.0]]),
        np.array([-0.12, 0.05, 0.0008, -0.0005, 0.0]),
    ),
    "right": (
        np.array([[426.0, 0.0, 236.0], [0.0, 425.0, 182.0], [0.0, 0.0, 1.0]]),
        np.array([-0.10, 0.03, -0.0004, 0.0006, 0.0]),
    ),
}
R_RL = Rotation.from_euler("yx", [-3.0, 1.0], degrees=True).as_matrix()
"""``X_right = R_RL X_left + T_RL``."""
T_RL = np.array([-0.06, 0.002, 0.001])
BOARD = Checkerboard(8, 6, 0.03)
TILT_ROTATION: dict[Tilt, tuple[str, float]] = {
    "left": ("y", 1.0),
    "right": ("y", -1.0),
    "up": ("x", -1.0),
    "down": ("x", 1.0),
}
SUPERSAMPLE = tuple((-0.25 + 0.5 * i, -0.25 + 0.5 * j) for i in range(2) for j in range(2))
_RAYS: dict[str, np.ndarray] = {}


def _rays(camera: str) -> np.ndarray:
    if camera not in _RAYS:
        k, d = CAMERAS[camera]
        u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
        samples = []
        for du, dv in SUPERSAMPLE:
            pixels = np.stack([u.ravel() + du, v.ravel() + dv], axis=1).reshape(-1, 1, 2)
            normalized = cv2.undistortPointsIter(
                pixels, k, d, None, None, (cv2.TERM_CRITERIA_COUNT, 20, 0)
            ).reshape(-1, 2)
            samples.append(np.column_stack([normalized, np.ones(len(normalized))]))
        _RAYS[camera] = np.concatenate(samples)
    return _RAYS[camera]


def render(camera: str, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """The board, posed in the left camera's frame, seen by one camera of the rig."""

    if camera == "right":
        rotation, translation = R_RL @ rotation, R_RL @ translation + T_RL
    rays = _rays(camera)
    in_board = rays @ rotation
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
    image[inside & squares & ((column + row) % 2 == 0)] = 25.0
    image = image.reshape(len(SUPERSAMPLE), HEIGHT, WIDTH).mean(axis=0)
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.4), 0, 255).astype(np.uint8)


def pose(
    camera: str, center_px: tuple[float, float], depth_m: float, tilt: Tilt | None
) -> tuple[np.ndarray, np.ndarray]:
    """A board pose, in the left camera's frame, centred on a pixel of ``camera``."""

    rotation = np.eye(3)
    if tilt is not None:
        axis, sign = TILT_ROTATION[tilt]
        rotation = Rotation.from_euler(axis, sign * 35.0, degrees=True).as_matrix()
    center_board = np.array(
        [(BOARD.columns - 1) * BOARD.square_m / 2, (BOARD.rows - 1) * BOARD.square_m / 2, 0.0]
    )
    k, _ = CAMERAS[camera]
    translation = depth_m * np.linalg.solve(k, [center_px[0], center_px[1], 1.0])
    translation = translation - rotation @ center_board
    if camera == "right":  # back into the left frame
        rotation, translation = R_RL.T @ rotation, R_RL.T @ (translation - T_RL)
    return rotation, translation


def _follow(guide: StereoGuide, steps: int = 400) -> tuple[list[str], list[tuple]]:
    """A person who does what the guide says; returns the advice and the frames shown."""

    rng = np.random.default_rng(5)
    kinds: list[str] = []
    shown: list[tuple] = []
    camera, center, tilt, depth = "left", (240.0, 180.0), None, 0.6
    for _ in range(steps):
        board = pose(camera, center, depth, tilt)
        pair = (render("left", *board), render("right", *board))
        guide.update(*pair)
        frame = guide.update(*pair)
        shown += [pair, pair]
        advice = frame.advice
        kinds.append(advice.kind)
        if advice.kind == "done":
            break
        camera = "right" if advice.text.startswith("Right:") else "left"
        tilt = None
        if advice.kind == "find":
            center, depth = (240.0, 180.0), 0.6
        elif advice.kind == "move":
            columns, rows = guide.options.grid
            column, row = advice.target_cell
            k, _ = CAMERAS[camera]
            depth = 0.6
            half_x = (BOARD.columns + 1) * BOARD.square_m / 2 * k[0, 0] / depth + 12
            half_y = (BOARD.rows + 1) * BOARD.square_m / 2 * k[1, 1] / depth + 12
            center = (
                float(np.clip((column + 0.5) * WIDTH / columns, half_x, WIDTH - half_x)),
                float(np.clip((row + 0.5) * HEIGHT / rows, half_y, HEIGHT - half_y)),
            )
        elif advice.kind == "tilt":
            center, tilt, depth = (240.0, 180.0), advice.tilt, 0.55
        else:  # vary: a pose both cameras see
            camera = "left"
            center = (rng.uniform(150, WIDTH - 110), rng.uniform(110, HEIGHT - 110))
            tilt = TILTS[int(rng.integers(4))] if rng.random() < 0.6 else None
            depth = float(rng.uniform(0.55, 0.9))
    return kinds, shown


@pytest.fixture(scope="module")
def followed() -> tuple[StereoGuide, list[str], list[tuple]]:
    guide = StereoGuide(BOARD, (WIDTH, HEIGHT), (WIDTH, HEIGHT))
    kinds, shown = _follow(guide)
    return guide, kinds, shown


def test_following_the_advice_calibrates_both_cameras_and_the_baseline(
    followed: tuple[StereoGuide, list[str], list[tuple]],
) -> None:
    guide, kinds, _ = followed

    assert kinds[-1] == "done", kinds
    for side in ("left", "right"):
        estimate = getattr(guide, side).estimate
        k, d = CAMERAS[side]
        assert estimate.fx == pytest.approx(k[0, 0], rel=0.01)
        assert estimate.cx == pytest.approx(k[0, 2], abs=3.0)
        assert estimate.distortion[0] == pytest.approx(d[0], abs=0.02)
    extrinsics = guide.extrinsics
    assert extrinsics is not None and extrinsics.views >= guide.min_stereo_views
    # A principal point 1 px off turns the relative rotation by 1 / f rad (0.14 deg).
    angle = np.degrees(Rotation.from_matrix(extrinsics.rotation @ R_RL.T).magnitude())
    assert angle < 0.4 and angle < 3.0 * extrinsics.rotation_std_deg
    # The intrinsics' errors (the renderer's aliasing) shift the baseline by a few mm,
    # mostly along the optical axis; the reported std carries them.
    assert extrinsics.baseline_m == pytest.approx(np.linalg.norm(T_RL), rel=0.02)
    error = np.abs(extrinsics.translation_m - T_RL)
    assert np.all(error < 0.003)
    assert np.all(error < 3.0 * extrinsics.translation_std_m)
    assert extrinsics.baseline_std_m <= guide.baseline_rel_std * extrinsics.baseline_m
    assert extrinsics.rms_px < 0.5


def test_the_camchain_has_both_cameras_and_the_baseline(
    followed: tuple[StereoGuide, list[str], list[tuple]],
) -> None:
    guide, _, shown = followed
    chain = stereo_camchain(guide)

    assert set(chain) == {"cam0", "cam1"}
    transform = np.asarray(chain["cam1"]["T_cn_cnm1"])
    assert transform.shape == (4, 4)
    np.testing.assert_allclose(transform[:3, 3], T_RL, atol=0.003)
    frame = guide.update(*shown[-1])
    canvas = draw_stereo(
        cv2.cvtColor(shown[-1][0], cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(shown[-1][1], cv2.COLOR_GRAY2BGR),
        guide,
        frame,
    )
    assert canvas.shape == (HEIGHT, 2 * WIDTH, 3)


def test_the_stereo_loop_and_the_cli_save_a_camchain(
    tmp_path: Path, followed: tuple[StereoGuide, list[str], list[tuple]]
) -> None:
    from calibrex.cli.main import main

    _, _, shown = followed
    saved: list[StereoGuide] = []
    run = run_stereo_guide(iter(shown), BOARD, show=False, save=saved.append, stop_when_done=True)
    assert run.done and run.saved and saved == [run.guide]

    video = tmp_path / "stereo.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (2 * WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("no lossless video codec in this OpenCV build")
    for left, right in shown:
        writer.write(cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_GRAY2BGR))
    writer.release()
    output = tmp_path / "camchain.yaml"

    status = main(
        ["intrinsics", "guide", "--board", "8x6x0.03", "--video", str(video), "--side-by-side",
         "--no-window", "--output", str(output)]
    )  # fmt: skip

    assert status == 0
    chain = yaml.safe_load(output.read_text())
    assert chain["cam0"]["intrinsics"][0] == pytest.approx(CAMERAS["left"][0][0, 0], rel=0.01)
    assert chain["cam1"]["intrinsics"][0] == pytest.approx(CAMERAS["right"][0][0, 0], rel=0.01)
    np.testing.assert_allclose(np.asarray(chain["cam1"]["T_cn_cnm1"])[:3, 3], T_RL, atol=0.003)
