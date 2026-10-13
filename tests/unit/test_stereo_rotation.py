"""A stereo pair's relative rotation from its images, on rendered fisheye pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.transform import Rotation

from calibrex.check.estimators import rotation_errors_deg
from calibrex.core.stereo_rotation import StereoRotationProvenance
from calibrex.core.validation import validate_file
from calibrex.evaluation.fisheye_stereo import EquidistantCamera
from calibrex.evaluation.stereo_rotation import (
    StereoRotationOptions,
    estimate_stereo_rotation,
)

LEFT = EquidistantCamera(160.0, 160.5, 161.0, 118.0, (-0.03, -0.006, 0.004, -0.002), 320, 240)
RIGHT = EquidistantCamera(161.0, 161.2, 158.0, 121.5, (-0.035, -0.004, 0.003, -0.001), 320, 240)
# the right camera 0.1 m along the left camera's x, slightly turned
TRUTH = np.eye(4)
TRUTH[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.4, 0.2], degrees=True).as_matrix()
TRUTH[:3, 3] = [0.1, 0.001, -0.002]
ROOM_M = np.array([6.0, 4.0, 2.5])  # half-sizes of the room (x, y, z)
OPTIONS = StereoRotationOptions(patches_per_pair=150, min_patches=300)
PROVENANCE = StereoRotationProvenance(
    generator="tests",
    generator_version="0",
    left_topic="/left",
    right_topic="/right",
    left_frame="left",
    right_frame="right",
    intrinsics_source="test",
)


def _texture() -> np.ndarray:
    rng = np.random.default_rng(11)
    blurred = ndimage.gaussian_filter(rng.uniform(0.0, 1.0, (1024, 2048)), 2.0)
    return 128.0 + 40.0 * (blurred - blurred.mean()) / blurred.std()


def _render(camera: EquidistantCamera, rotation: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """The camera's image of a textured box-shaped room around the world origin."""

    rows, cols = np.mgrid[0 : camera.height, 0 : camera.width].astype(np.float64)
    rays = camera.unproject(cols.ravel(), rows.ravel()) @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = np.where(rays > 0.0, (ROOM_M - origin) / rays, (-ROOM_M - origin) / rays)
    reach = np.where(np.isfinite(reach) & (reach > 0.0), reach, np.inf)
    wall = np.argmin(reach, axis=1)
    points = origin + rays * reach[np.arange(len(rays)), wall][:, None]
    # each wall shows the texture over its two in-plane coordinates, offset per wall
    first = np.where(wall == 0, points[:, 1], points[:, 0])
    second = np.where(wall == 2, points[:, 1], points[:, 2])
    texture = _texture()
    scale = 120.0  # texels per metre
    u = first * scale + 300.0 * wall + 0.5 * texture.shape[1]
    v = second * scale + 150.0 * wall + 0.5 * texture.shape[0]
    image = ndimage.map_coordinates(texture, [v, u], order=1, mode="wrap")
    return np.clip(image, 0.0, 255.0).reshape(camera.height, camera.width)


@pytest.fixture(scope="module")
def pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(4)
    result = []
    for _ in range(8):
        world_left = Rotation.from_euler(
            "zyx",
            [rng.uniform(-180, 180), rng.uniform(-30, 30), rng.uniform(-20, 20)],
            degrees=True,
        ).as_matrix()
        left = _render(LEFT, world_left, np.zeros(3))
        right = _render(RIGHT, world_left @ TRUTH[:3, :3], world_left @ TRUTH[:3, 3])
        result.append((left, right))
    return result


def test_the_true_rotation_passes_with_every_axis_observed(
    pairs: list[tuple[np.ndarray, np.ndarray]], tmp_path: Path
) -> None:
    artifact = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS
    )
    assert artifact.policy_status == "pass"
    assert [axis.name for axis in artifact.axes] == ["roll", "pitch", "yaw"]
    for axis in artifact.axes:
        assert axis.observable, axis.reason
        assert abs(axis.candidate_error_deg) < 0.05
        assert axis.std_deg < 0.05
        assert axis.control_response == pytest.approx(1.0, abs=0.2)
    artifact.save(tmp_path / "stereo.yaml")
    assert validate_file(tmp_path / "stereo.yaml").valid


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_turned_candidate_reads_its_turn_about_the_left_camera(
    pairs: list[tuple[np.ndarray, np.ndarray]], axis: int
) -> None:
    candidate = TRUTH.copy()
    turn = Rotation.from_rotvec(np.radians(0.6) * np.eye(3)[axis]).as_matrix()
    candidate[:3, :3] = turn @ TRUTH[:3, :3]
    expected = rotation_errors_deg(candidate[:3, :3], TRUTH[:3, :3])  # the check's convention
    artifact = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, candidate, provenance=PROVENANCE, options=OPTIONS
    )
    errors = [entry.candidate_error_deg for entry in artifact.axes]
    assert errors == pytest.approx(list(expected), abs=0.06)


def test_too_few_pairs_are_inconclusive(pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    artifact = estimate_stereo_rotation(
        pairs[:3], LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS
    )
    assert artifact.policy_status == "inconclusive"
    assert artifact.axes == []
