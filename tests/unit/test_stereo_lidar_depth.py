"""Stereo-depth camera-LiDAR evidence on a rendered street scene."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from calibrex.evaluation.stereo_lidar_depth import (
    RectifiedStereo,
    StereoFrame,
    estimate_depth_offset,
    estimate_stereo_alignment,
)

FX, CX, CY, WIDTH, HEIGHT = 300.0, 200.0, 100.0, 400, 200
BASELINE_M, ROAD_M = 0.5, 1.5  # the road is 1.5 m below the cameras (y points down)
SIDES_M = (-3.5, 4.0)  # house fronts along the street, left and right (x)
K = np.array([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]])


def _projection(camera_x: float, cx: float = CX) -> np.ndarray:
    intrinsics = K.copy()
    intrinsics[0, 2] = cx
    return intrinsics @ np.c_[np.eye(3), [-camera_x, 0.0, 0.0]]


NOISE = np.random.default_rng(7).uniform(0.0, 255.0, (1024, 1024))


def _texture(a: np.ndarray, b: np.ndarray, phase: float) -> np.ndarray:
    """Random value noise on 12 cm cells (over two pixels on every wall; not periodic)."""

    cells = np.stack([b / 0.12 + 97.0 * phase, a / 0.12])
    return np.asarray(map_coordinates(NOISE, cells.reshape(2, -1), order=1, mode="wrap")).reshape(
        a.shape
    )


def _walls(index: int) -> list[tuple[float, float, float, float, float]]:
    """Camera-facing rectangles ``(Z, x0, x1, y0, y1)``, moving a little per frame."""

    shift = 0.3 * index
    return [
        (5.0, -3.0 + shift, -0.6 + shift, -1.0, ROAD_M),
        (8.0, 0.4 - shift, 4.0 - shift, -2.0, ROAD_M),
        (14.0, -6.0, 6.0, -4.0, ROAD_M),
    ]


def _render(
    camera_x: float, index: int, *, walls: bool, sides: bool = False, cx: float = CX
) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale image and depth of the scene seen from ``(camera_x, 0, 0)``."""

    v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    ray_x, ray_y = (u - cx) / FX, (v - CY) / FX
    depth = np.full(u.shape, 40.0)
    image = _texture(camera_x + ray_x * 40.0, ray_y * 40.0, index)
    road = np.where(ray_y > 0, ROAD_M / np.where(ray_y > 0, ray_y, 1.0), 1.0e3)
    hit = road < depth
    depth = np.where(hit, road, depth)
    image = np.where(hit, _texture(camera_x + ray_x * road, road, index), image)
    for side in SIDES_M if sides else ():
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(ray_x * side > 0, (side - camera_x) / ray_x, 1.0e3)
        y = ray_y * z
        hit = (z < depth) & (z > 4.0) & (z < 25.0) & (y >= -2.5) & (y <= ROAD_M)
        depth = np.where(hit, z, depth)
        image = np.where(hit, _texture(z, y, index + 3.0), image)
    for z, x0, x1, y0, y1 in _walls(index) if walls else []:
        x, y = camera_x + ray_x * z, ray_y * z
        hit = (z < depth) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        depth = np.where(hit, z, depth)
        image = np.where(hit, _texture(x, y, index), image)
    return np.asarray(image), depth


def _points(index: int, *, walls: bool, sides: bool = False) -> np.ndarray:
    """LiDAR returns (at the left camera) on the visible road and walls."""

    points = [
        np.c_[x.ravel(), np.full(x.size, ROAD_M), z.ravel()]
        for x, z in [np.meshgrid(np.arange(-6.0, 6.0, 0.15), np.arange(3.5, 30.0, 0.2))]
    ]
    for side in SIDES_M if sides else ():
        z, y = np.meshgrid(np.arange(4.0, 25.0, 0.08), np.arange(-2.5, ROAD_M, 0.08))
        points.append(np.c_[np.full(z.size, side), y.ravel(), z.ravel()])
    for z, x0, x1, y0, y1 in _walls(index) if walls else []:
        x, y = np.meshgrid(np.arange(x0, x1, 0.04 * z / 5), np.arange(y0, y1, 0.04 * z / 5))
        points.append(np.c_[x.ravel(), y.ravel(), np.full(x.size, z)])
    cloud = np.concatenate(points)
    # drop returns hidden behind a nearer surface (the LiDAR sits at the left camera)
    pixel = cloud @ K.T
    u = np.round(pixel[:, 0] / pixel[:, 2]).astype(int)
    v = np.round(pixel[:, 1] / pixel[:, 2]).astype(int)
    inside = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    cloud, u, v = cloud[inside], u[inside], v[inside]
    _, depth = _render(0.0, index, walls=walls, sides=sides)
    return np.asarray(cloud[cloud[:, 2] <= depth[v, u] + 0.05])


def _frames(*, walls: bool = True, sides: bool = False, right_cx: float = CX) -> list[StereoFrame]:
    return [
        StereoFrame(
            left=_render(0.0, index, walls=walls, sides=sides)[0],
            right=_render(BASELINE_M, index, walls=walls, sides=sides, cx=right_cx)[0],
            points=_points(index, walls=walls, sides=sides),
        )
        for index in range(6)
    ]


STEREO = RectifiedStereo(left=_projection(0.0), right=_projection(BASELINE_M))


@pytest.mark.parametrize("error_m", [0.0, 0.10, -0.15])
def test_the_candidates_error_along_the_optical_axis_is_measured(error_m: float) -> None:
    candidate = np.eye(4)
    candidate[2, 3] = error_m
    estimate = estimate_depth_offset(_frames(), STEREO, candidate)

    assert estimate.observable, estimate.reason
    assert estimate.offset_m == pytest.approx(error_m, abs=0.03)
    assert estimate.std_m is not None and estimate.std_m < 0.03
    assert estimate.control_response == pytest.approx(1.0, abs=0.3)


def test_a_constant_disparity_offset_of_the_rig_is_absorbed() -> None:
    # the right camera's principal point is 0.6 px off what its projection says
    estimate = estimate_depth_offset(_frames(right_cx=CX - 0.6), STEREO, np.eye(4))

    assert estimate.observable, estimate.reason
    assert estimate.disparity_offset_px == pytest.approx(0.6, abs=0.15)
    assert estimate.offset_m == pytest.approx(0.0, abs=0.03)


def test_a_road_without_camera_facing_surfaces_is_not_observable() -> None:
    # moving the candidate along the optical axis keeps a road point on the road
    estimate = estimate_depth_offset(_frames(walls=False), STEREO, np.eye(4))

    assert not estimate.observable
    assert estimate.offset_m is None
    assert "camera-facing" in estimate.reason


@pytest.fixture(scope="module")
def street() -> list[StereoFrame]:
    return _frames(sides=True)


@pytest.mark.parametrize(
    ("axis", "amount"),
    [(None, 0.0), (0, 0.10), (0, -0.08), (2, 0.12), (5, 0.4), (4, -0.4)],
)
def test_the_joint_fit_measures_lateral_depth_and_rotation_errors(
    street: list[StereoFrame], axis: int | None, amount: float
) -> None:
    from scipy.spatial.transform import Rotation

    candidate = np.eye(4)
    expected = np.zeros(6)
    if axis is not None and axis < 3:
        candidate[axis, 3] = amount
    elif axis is not None:
        rotation = np.zeros(3)
        rotation[axis - 3] = np.radians(amount)
        candidate[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    if axis is not None:
        expected[axis] = amount

    estimate = estimate_stereo_alignment(street, STEREO, candidate)

    by_name = {item.name: item for item in estimate.axes}
    # house fronts and walls facing the camera always pin x and z
    assert by_name["x"].observable, by_name["x"].reason
    assert by_name["z"].observable, by_name["z"].reason
    # whatever axis is trusted is right; y rests on the near road only (beyond about 5 m
    # it is seen within 0.3 of edge-on) and is trusted only when its control follows
    for index, name in enumerate(("x", "y", "z")):
        if by_name[name].observable:
            assert by_name[name].offset == pytest.approx(expected[index], abs=0.03), name
        else:
            assert "response" in by_name[name].reason
    for index, name in enumerate(("roll", "pitch", "yaw"), start=3):
        assert by_name[name].offset == pytest.approx(expected[index], abs=0.15)


def test_lateral_translation_is_not_trusted_without_walls_along_the_street() -> None:
    # without house fronts nothing faces sideways: x is not observable
    estimate = estimate_stereo_alignment(_frames(), STEREO, np.eye(4))

    by_name = {item.name: item for item in estimate.axes}
    assert not by_name["x"].observable
