from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from calibrex.evaluation.rotation_self_calibration import (
    project_points,
    self_calibrate,
    undistort_points,
)

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 320, 240
TRUE_K = np.array([[280.0, 0.0, 162.0], [0.0, 278.7, 117.3], [0.0, 0.0, 1.0]])
TRUE_DISTORTION = np.array([-0.12, 0.05, 0.0008, -0.0005, 0.0])
ROOM_HALF_M = 4.0
SUBSAMPLES = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))
_RAYS: np.ndarray | None = None


def _rays() -> np.ndarray:
    global _RAYS
    if _RAYS is None:
        u, v = np.meshgrid(np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float))
        rays = []
        for du, dv in SUBSAMPLES:
            pixels = np.stack([u.ravel() + du, v.ravel() + dv], 1).reshape(-1, 1, 2)
            normalized = cv2.undistortPointsIter(
                pixels, TRUE_K, TRUE_DISTORTION, None, None, (cv2.TERM_CRITERIA_COUNT, 20, 0)
            ).reshape(-1, 2)
            rays.append(np.column_stack([normalized, np.ones(len(normalized))]))
        _RAYS = np.concatenate(rays)
    return _RAYS


def _hash(a: np.ndarray, b: np.ndarray, face: np.ndarray) -> np.ndarray:
    x = (a * 73856093) ^ (b * 19349663) ^ (face * 83492791)
    x = (x ^ (x >> 13)) * 1274126177
    return ((x ^ (x >> 16)) & 0xFFFF) / 65535.0


def render(rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    """A textured cube room seen by the true (distorted) camera at ``rotation``/``center``."""

    directions = _rays() @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(
            directions > 0,
            (ROOM_HALF_M - center) / directions,
            (-ROOM_HALF_M - center) / directions,
        )
    exits[~np.isfinite(exits)] = np.inf
    axis = np.argmin(exits, axis=1)
    rows = np.arange(len(axis))
    hit = center + directions * exits[rows, axis][:, None]
    face = axis * 2 + (directions[rows, axis] > 0)
    a = np.where(axis == 0, hit[:, 1], hit[:, 0]) + 10
    b = np.where(axis == 2, hit[:, 1], hit[:, 2]) + 10
    grey = 40 + 150 * _hash(
        np.floor(a / 0.25).astype(np.int64), np.floor(b / 0.25).astype(np.int64), face
    )
    grey += 50 * _hash(
        np.floor(a / 0.08).astype(np.int64) + 7, np.floor(b / 0.08).astype(np.int64) + 11, face + 9
    )
    image = grey.reshape(len(SUBSAMPLES), HEIGHT, WIDTH).mean(axis=0)
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.5), 0, 255).astype(np.uint8)


def turning(time_s: float) -> np.ndarray:
    """Tilt, pan and roll of a camera turned by hand about its lens."""

    rotvec = [
        0.45 * np.sin(0.9 * time_s),
        0.6 * np.sin(0.55 * time_s + 0.3),
        0.35 * np.sin(0.75 * time_s + 1.0),
    ]
    return Rotation.from_rotvec(rotvec).as_matrix()


def frames(
    seconds: float, fps: float, motion=turning, lens_offset_m: float = 0.0
) -> list[tuple[float, np.ndarray]]:
    out = []
    for time_s in np.arange(0.0, seconds, 1.0 / fps):
        rotation = motion(time_s)
        out.append(
            (float(time_s), render(rotation, rotation @ np.array([0.0, 0.0, lens_offset_m])))
        )
    return out


def test_undistortion_inverts_the_projection() -> None:
    params = np.array([280.0, 278.7, 162.0, 117.3, -0.12, 0.05, 0.0008, -0.0005])
    rng = np.random.default_rng(0)
    rays = np.column_stack([rng.uniform(-0.6, 0.6, (200, 2)), np.ones(200)])
    pixels = project_points(params, rays)
    expected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), TRUE_K, TRUE_DISTORTION)
    assert np.allclose(pixels, expected.reshape(-1, 2), atol=1e-6)
    assert np.allclose(undistort_points(params, pixels), rays, atol=1e-6)


def test_a_camera_panned_only_is_flagged() -> None:
    def pan(time_s: float) -> np.ndarray:
        return Rotation.from_rotvec([0.0, 0.6 * np.sin(0.8 * time_s), 0.0]).as_matrix()

    result = self_calibrate(iter(frames(5.0, 8.0, motion=pan)))

    assert result.rotation_span_deg[1] > 40 and result.rotation_span_deg[0] < 5
    assert any("two axes" in warning for warning in result.warnings)


def test_cli_calibrates_a_camera_turned_about_its_lens_from_a_video(tmp_path: Path) -> None:
    from calibrex.cli.main import main

    video = tmp_path / "turning.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("no lossless video codec in this OpenCV build")
    for _, image in frames(8.0, 10.0):
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()
    output = tmp_path / "camchain.yaml"

    assert (
        main(["intrinsics", "rotate", "--video", str(video), "--output", str(output), "--json"])
        == 0
    )

    entry = yaml.safe_load(output.read_text())["cam0"]
    fx, fy, cx, cy = entry["intrinsics"]
    assert fx == pytest.approx(TRUE_K[0, 0], rel=0.01)
    assert fy == pytest.approx(TRUE_K[1, 1], rel=0.01)
    assert cx == pytest.approx(TRUE_K[0, 2], abs=2.0)
    assert cy == pytest.approx(TRUE_K[1, 2], abs=2.0)
    assert entry["distortion_coeffs"][0] == pytest.approx(TRUE_DISTORTION[0], abs=0.015)
    assert entry["distortion_model"] == "radtan" and entry["resolution"] == [WIDTH, HEIGHT]
