from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from calibrex.evaluation.board_camera_lidar import Checkerboard
from calibrex.evaluation.intrinsics_guide import (
    TILTS,
    GuideOptions,
    IntrinsicsEstimate,
    IntrinsicsGuide,
    Tilt,
    camchain_entry,
)
from calibrex.evaluation.intrinsics_stereo_guide import stereo_extrinsics
from calibrex.evaluation.rotation_self_calibration import (
    SelfCalibrationOptions,
    project_points,
    self_calibrate,
    undistort_points,
)

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 480, 360
TRUE_K = np.array([[235.0, 0.0, 243.0], [0.0, 234.0, 176.0], [0.0, 0.0, 1.0]])
TRUE_D = np.array([0.04, -0.02, 0.006, -0.001])
"""Equidistant (Kannala-Brandt) k1..k4: about 110 deg across."""
PARAMS = np.array([235.0, 234.0, 243.0, 176.0, *TRUE_D])
BOARD = Checkerboard(8, 6, 0.03)
TILT_ROTATION: dict[Tilt, tuple[str, float]] = {
    "left": ("y", 1.0),
    "right": ("y", -1.0),
    "up": ("x", -1.0),
    "down": ("x", 1.0),
}
SUPERSAMPLE = tuple((-0.25 + 0.5 * i, -0.25 + 0.5 * j) for i in range(2) for j in range(2))
_RAYS: list[np.ndarray] = []


def _rays() -> np.ndarray:
    if not _RAYS:
        u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
        samples = []
        for du, dv in SUPERSAMPLE:
            pixels = np.stack([u.ravel() + du, v.ravel() + dv], axis=1)
            samples.append(undistort_points(PARAMS, pixels, "equidistant"))
        _RAYS.append(np.concatenate(samples))
    return _RAYS[0]


def test_the_equidistant_model_round_trips_and_matches_opencv() -> None:
    rng = np.random.default_rng(0)
    pixels = rng.uniform([10, 10], [WIDTH - 10, HEIGHT - 10], size=(200, 2))
    rays = undistort_points(PARAMS, pixels, "equidistant")
    np.testing.assert_allclose(project_points(PARAMS, rays, "equidistant"), pixels, atol=1e-6)
    opencv = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), TRUE_K, TRUE_D).reshape(-1, 2)
    np.testing.assert_allclose(rays[:, :2], opencv, atol=1e-6)


# ------------------------------------------------------------------- the board guide


def render_board(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rays = _rays()
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


def pose(center_px: tuple[float, float], depth_m: float, tilt: Tilt | None):
    rotation = np.eye(3)
    if tilt is not None:
        axis, sign = TILT_ROTATION[tilt]
        rotation = Rotation.from_euler(axis, sign * 35.0, degrees=True).as_matrix()
    center_board = np.array(
        [(BOARD.columns - 1) * BOARD.square_m / 2, (BOARD.rows - 1) * BOARD.square_m / 2, 0.0]
    )
    ray = undistort_points(PARAMS, np.array([center_px]), "equidistant")[0]
    return rotation, depth_m * ray - rotation @ center_board


def _follow(guide: IntrinsicsGuide, steps: int = 200) -> list[str]:
    rng = np.random.default_rng(4)
    kinds: list[str] = []
    center, tilt, depth = (240.0, 180.0), None, 0.4
    for _ in range(steps):
        image = render_board(*pose(center, depth, tilt))
        guide.update(image)
        advice = guide.update(image).advice
        kinds.append(advice.kind)
        if advice.kind == "done":
            break
        tilt = None
        if advice.kind == "find":
            center, depth = (240.0, 180.0), 0.4
        elif advice.kind == "move":
            columns, rows = guide.options.grid
            column, row = advice.target_cell
            depth = 0.4
            center = (
                float(np.clip((column + 0.5) * WIDTH / columns, 95, WIDTH - 95)),
                float(np.clip((row + 0.5) * HEIGHT / rows, 75, HEIGHT - 75)),
            )
        elif advice.kind == "tilt":
            center, tilt, depth = (240.0, 180.0), advice.tilt, 0.38
        else:
            center = (rng.uniform(100, WIDTH - 100), rng.uniform(80, HEIGHT - 80))
            tilt = TILTS[int(rng.integers(4))]
            depth = float(rng.uniform(0.3, 0.55))
    return kinds


def test_the_board_guide_calibrates_a_fisheye_lens() -> None:
    guide = IntrinsicsGuide(BOARD, WIDTH, HEIGHT, GuideOptions(distortion_model="equidistant"))
    kinds = _follow(guide)

    assert kinds[-1] == "done", kinds
    estimate = guide.estimate
    assert estimate is not None and estimate.model == "equidistant"
    assert estimate.fx == pytest.approx(TRUE_K[0, 0], rel=0.01)
    assert estimate.fy == pytest.approx(TRUE_K[1, 1], rel=0.01)
    assert estimate.cx == pytest.approx(TRUE_K[0, 2], abs=2.0)
    assert estimate.cy == pytest.approx(TRUE_K[1, 2], abs=2.0)
    assert set(estimate.std) == {"fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"}
    # The Jacobian stds are a few times smaller than the renderer's correlated error.
    assert 0.0 < estimate.std["fx"] < 0.003 * estimate.fx
    entry = camchain_entry(estimate)
    assert entry["distortion_model"] == "equidistant" and len(entry["distortion_coeffs"]) == 4


def test_the_stereo_baseline_works_with_fisheye_intrinsics() -> None:
    # Exact corners projected through two fisheye cameras, with 0.1 px noise.
    rng = np.random.default_rng(1)
    r_rl = Rotation.from_euler("yx", [-4.0, 1.0], degrees=True).as_matrix()
    t_rl = np.array([-0.08, 0.002, 0.001])
    estimate = IntrinsicsEstimate(
        WIDTH, HEIGHT, *PARAMS[:4], tuple(TRUE_D), {}, 0.0, 0, "equidistant"
    )
    views = []
    for _ in range(15):
        rotation = Rotation.from_rotvec(rng.normal(0, 0.35, 3)).as_matrix()
        center = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2), rng.uniform(0.4, 0.8)])
        translation = center - rotation @ np.array([0.105, 0.075, 0.0])
        corners = []
        for r, t in ((rotation, translation), (r_rl @ rotation, r_rl @ translation + t_rl)):
            camera = BOARD.object_points() @ r.T + t
            pixels = project_points(PARAMS, camera, "equidistant")
            corners.append(pixels + rng.normal(0, 0.1, pixels.shape))
        views.append((corners[0], corners[1]))

    extrinsics = stereo_extrinsics(views, BOARD, estimate, estimate)

    assert Rotation.from_matrix(extrinsics.rotation @ r_rl.T).magnitude() < np.radians(0.05)
    np.testing.assert_allclose(extrinsics.translation_m, t_rl, atol=0.0005)
    assert extrinsics.rms_px == pytest.approx(0.14, abs=0.05)


# ------------------------------------------------------------- the targetless rotation

ROOM_HALF_M, CELL_M = 8.0, 0.5


def _hash(a: np.ndarray, b: np.ndarray, face: np.ndarray) -> np.ndarray:
    x = (a * 73856093) ^ (b * 19349663) ^ (face * 83492791)
    x = (x ^ (x >> 13)) * 1274126177
    return ((x ^ (x >> 16)) & 0xFFFF) / 65535.0


def render_room(rotation: np.ndarray) -> np.ndarray:
    directions = _rays() @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(directions > 0, ROOM_HALF_M / directions, -ROOM_HALF_M / directions)
    exits[~np.isfinite(exits)] = np.inf
    axis = np.argmin(exits, axis=1)
    rows = np.arange(len(axis))
    hit = directions * exits[rows, axis][:, None]
    face = axis * 2 + (directions[rows, axis] > 0)
    a = np.where(axis == 0, hit[:, 1], hit[:, 0]) + 50
    b = np.where(axis == 2, hit[:, 1], hit[:, 2]) + 50
    coarse = _hash(
        np.floor(a / CELL_M).astype(np.int64), np.floor(b / CELL_M).astype(np.int64), face
    )
    fine = _hash(
        np.floor(3 * a / CELL_M).astype(np.int64) + 7,
        np.floor(3 * b / CELL_M).astype(np.int64) + 11,
        face + 9,
    )
    image = (40 + 150 * coarse + 50 * fine).reshape(len(SUPERSAMPLE), HEIGHT, WIDTH).mean(axis=0)
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.5), 0, 255).astype(np.uint8)


def _turning(time_s: float) -> np.ndarray:
    rotvec = [
        0.45 * np.sin(0.9 * time_s),
        0.6 * np.sin(0.55 * time_s + 0.3),
        0.35 * np.sin(0.75 * time_s + 1.0),
    ]
    return Rotation.from_rotvec(rotvec).as_matrix()


@pytest.fixture(scope="module")
def turning_frames() -> list[tuple[float, np.ndarray]]:
    return [(float(t), render_room(_turning(t))) for t in np.arange(0.0, 8.0, 0.1)]


def test_rotation_self_calibration_of_a_fisheye_lens(
    turning_frames: list[tuple[float, np.ndarray]],
) -> None:
    result = self_calibrate(turning_frames, SelfCalibrationOptions(distortion_model="equidistant"))

    estimate = result.estimate
    assert estimate.model == "equidistant"
    assert estimate.fx == pytest.approx(TRUE_K[0, 0], rel=0.01)
    assert estimate.fy == pytest.approx(TRUE_K[1, 1], rel=0.01)
    assert estimate.cx == pytest.approx(TRUE_K[0, 2], abs=2.0)
    assert estimate.cy == pytest.approx(TRUE_K[1, 2], abs=2.0)
    # The fisheye compresses the texture: the 2 x 2 supersampled renderer aliases more.
    assert result.estimate.rms_px < 0.8
    # Off the axis the fitted model maps pixels to the true directions.
    pixels = np.array([[40.0, 30.0], [440.0, 330.0], [240.0, 20.0]])
    fitted = undistort_points(
        np.array([estimate.fx, estimate.fy, estimate.cx, estimate.cy, *estimate.distortion]),
        pixels,
        "equidistant",
    )
    true = undistort_points(PARAMS, pixels, "equidistant")
    cosine = np.sum(fitted * true, axis=1) / (
        np.linalg.norm(fitted, axis=1) * np.linalg.norm(true, axis=1)
    )
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))).max() < 0.5


def test_cli_rotate_writes_an_equidistant_camchain(
    tmp_path: Path, turning_frames: list[tuple[float, np.ndarray]]
) -> None:
    from calibrex.cli.main import main

    video = tmp_path / "turning.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("no lossless video codec in this OpenCV build")
    for _, image in turning_frames:
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()
    output = tmp_path / "camchain.yaml"

    status = main(
        ["intrinsics", "rotate", "--video", str(video), "--distortion", "equidistant",
         "--output", str(output)]
    )  # fmt: skip

    assert status == 0
    entry = yaml.safe_load(output.read_text())["cam0"]
    assert entry["distortion_model"] == "equidistant"
    assert entry["intrinsics"][0] == pytest.approx(TRUE_K[0, 0], rel=0.01)


def test_the_turning_guide_reads_fisheye_turns(
    turning_frames: list[tuple[float, np.ndarray]],
) -> None:
    from calibrex.evaluation.rotation_guide import RotationGuide, RotationGuideOptions

    guide = RotationGuide(WIDTH, HEIGHT, RotationGuideOptions(distortion_model="equidistant"))
    for _, image in turning_frames:
        guide.update(image)
    true = np.array([Rotation.from_matrix(_turning(t)).as_rotvec() for t, _ in turning_frames])
    true_span = np.degrees(true.max(axis=0) - true.min(axis=0))
    # The guessed fisheye focal length (216 px for the true 235 px) reads pan and tilt
    # about 8 % short; roll is exact up to the turns' coupling.
    assert guide.span_deg["roll"] == pytest.approx(true_span[2], rel=0.15)
    assert guide.span_deg["pan"] == pytest.approx(true_span[1], rel=0.2)
    assert guide.span_deg["tilt"] == pytest.approx(true_span[0], rel=0.2)
