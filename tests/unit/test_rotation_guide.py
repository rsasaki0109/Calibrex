from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from calibrex.evaluation.rotation_guide import (
    RotationGuide,
    draw_rotation_guide,
    run_rotation_guide,
)
from calibrex.evaluation.rotation_self_calibration import self_calibrate

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 320, 240
TRUE_K = np.array([[280.0, 0.0, 162.0], [0.0, 278.7, 117.3], [0.0, 0.0, 1.0]])
TRUE_DISTORTION = np.array([-0.12, 0.05, 0.0008, -0.0005, 0.0])
ROOM_HALF_M = 8.0
CELL_M = 0.5
SUBSAMPLES = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))
_RAYS: list[np.ndarray] = []


def _rays() -> np.ndarray:
    if not _RAYS:
        u, v = np.meshgrid(np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float))
        rays = []
        for du, dv in SUBSAMPLES:
            pixels = np.stack([u.ravel() + du, v.ravel() + dv], 1).reshape(-1, 1, 2)
            normalized = cv2.undistortPointsIter(
                pixels, TRUE_K, TRUE_DISTORTION, None, None, (cv2.TERM_CRITERIA_COUNT, 20, 0)
            ).reshape(-1, 2)
            rays.append(np.column_stack([normalized, np.ones(len(normalized))]))
        _RAYS.append(np.concatenate(rays))
    return _RAYS[0]


def _hash(a: np.ndarray, b: np.ndarray, face: np.ndarray) -> np.ndarray:
    x = (a * 73856093) ^ (b * 19349663) ^ (face * 83492791)
    x = (x ^ (x >> 13)) * 1274126177
    return ((x ^ (x >> 16)) & 0xFFFF) / 65535.0


def render(rotation: np.ndarray) -> np.ndarray:
    """The camera (camera-to-world ``rotation``) at the centre of a textured cube room."""

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
    image = (40 + 150 * coarse + 50 * fine).reshape(len(SUBSAMPLES), HEIGHT, WIDTH).mean(axis=0)
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.5), 0, 255).astype(np.uint8)


_AXIS = {"tilt": 0, "pan": 1, "roll": 2}


def _follow(
    guide: RotationGuide, frames: int = 400
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """A person who sweeps the axis the guide names, about +-16 deg, then the next.

    Returns the advice, the images and the true span about each of tilt, pan and roll.
    """

    angles = np.zeros(3)
    phase = 0.0
    axis = "pan"
    kinds: list[str] = []
    images: list[np.ndarray] = []
    low, high = np.zeros(3), np.zeros(3)
    for _ in range(frames):
        low, high = np.minimum(low, angles), np.maximum(high, angles)
        image = render(Rotation.from_rotvec(np.radians(angles)).as_matrix())
        images.append(image)
        advice = guide.update(image).advice
        kinds.append(advice.axis or advice.kind)
        if advice.kind == "done":
            break
        if advice.axis is not None and advice.axis != axis:
            axis, phase = advice.axis, 0.0
        phase += 0.15
        angles[_AXIS[axis]] = 16.0 * np.sin(phase)
    return kinds, images, high - low


@pytest.fixture(scope="module")
def followed() -> tuple[RotationGuide, list[str], list[np.ndarray], np.ndarray]:
    guide = RotationGuide(WIDTH, HEIGHT)
    kinds, images, span = _follow(guide)
    return guide, kinds, images, span


def test_following_the_advice_turns_the_camera_about_every_axis(
    followed: tuple[RotationGuide, list[str], list[np.ndarray], np.ndarray],
) -> None:
    guide, kinds, images, true_span = followed

    assert kinds[-1] == "done", kinds
    # Pan first, then tilt, then roll.
    first = [kinds.index(axis) for axis in ("pan", "tilt", "roll")]
    assert first == sorted(first)
    # The spans come from a guessed focal length (256 px for the true 280 px): pan and
    # tilt read about 10 % short, roll is exact.
    tilt, pan, roll = true_span
    assert guide.span_deg["roll"] == pytest.approx(roll, abs=1.5)
    assert guide.span_deg["pan"] == pytest.approx(pan * 256 / 280, rel=0.1)
    assert guide.span_deg["tilt"] == pytest.approx(tilt * 256 / 280, rel=0.1)

    result = self_calibrate(enumerate(images))
    assert result.estimate.fx == pytest.approx(TRUE_K[0, 0], rel=0.02)
    # Enough for the calibration's own check: two axes turned 20 deg or more.
    assert sorted(result.rotation_span_deg)[1] > 20.0


def test_the_guide_asks_for_texture_and_for_slower_turning() -> None:
    guide = RotationGuide(WIDTH, HEIGHT)
    assert guide.update(np.full((HEIGHT, WIDTH), 120, np.uint8)).advice.kind == "texture"
    guide.update(render(np.eye(3)))
    jump = Rotation.from_euler("y", 25, degrees=True).as_matrix()
    assert guide.update(render(jump)).advice.kind == "slow"


def test_the_overlay_and_the_headless_loop(
    tmp_path: Path, followed: tuple[RotationGuide, list[str], list[np.ndarray], np.ndarray]
) -> None:
    guide, _, images, _ = followed
    frame = guide.update(images[-1])
    canvas = draw_rotation_guide(images[-1], guide, frame)
    assert canvas.shape == (HEIGHT, WIDTH, 3)

    record = tmp_path / "rotate.avi"
    run = run_rotation_guide(
        ((float(i), image) for i, image in enumerate(images)), show=False, record=record, stride=2
    )
    assert run.done and run.accepted
    assert len(run.recorded) == (len(images) + 1) // 2
    assert record.exists() and record.stat().st_size > 0


def test_cli_rotate_with_guidance_on_a_video(
    tmp_path: Path, followed: tuple[RotationGuide, list[str], list[np.ndarray], np.ndarray]
) -> None:
    from calibrex.cli.main import main

    _, _, images, _ = followed
    video = tmp_path / "turning.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("no lossless video codec in this OpenCV build")
    for image in images:
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()
    output = tmp_path / "camchain.yaml"
    record = tmp_path / "annotated.avi"

    status = main(
        ["intrinsics", "rotate", "--video", str(video), "--guide", "--no-window",
         "--record", str(record), "--output", str(output)]
    )  # fmt: skip

    assert status == 0
    fx = yaml.safe_load(output.read_text())["cam0"]["intrinsics"][0]
    assert fx == pytest.approx(TRUE_K[0, 0], rel=0.02)
    assert record.exists()
