from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("numba")

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    project_depth_pairs,
)
from calibrex.solvers.borer_rotation_only_solver import apply_local_euler_delta
from calibrex.solvers.numba_depth_to_depth_adapter import (
    NUMBA_DEPTH_PAIR_PROJECTOR_VERSION,
    NumbaDepthPairProjector,
    numba_depth_pair_projector_identity,
    project_depth_pairs_numba,
)


@pytest.mark.parametrize(
    "camera",
    [
        DepthToDepthCameraModel(
            width=96,
            height=64,
            fx=52.0,
            fy=51.0,
            cx=47.5,
            cy=31.5,
        ),
        DepthToDepthCameraModel(
            width=96,
            height=96,
            fx=75.0,
            fy=74.0,
            cx=47.5,
            cy=47.5,
            projection="mei",
            xi=1.4,
            distortion=(0.01, 0.02, 0.001, -0.002),
        ),
        DepthToDepthCameraModel(
            width=96,
            height=80,
            fx=58.0,
            fy=57.0,
            cx=47.5,
            cy=39.5,
            projection="double_sphere",
            xi=0.35,
            alpha=0.55,
        ),
    ],
)
def test_numba_projection_exactly_matches_numpy(
    camera: DepthToDepthCameraModel,
) -> None:
    rng = np.random.default_rng(20260811)
    points = rng.uniform(
        low=(-12.0, -8.0, -4.0),
        high=(12.0, 8.0, 25.0),
        size=(20_000, 3),
    )
    points[1] = points[0]
    depth = rng.uniform(0.1, 80.0, size=(camera.height, camera.width))
    depth[::11, ::13] = np.nan
    observation = DepthToDepthObservation(
        frame_id=camera.projection,
        depth_map=depth,
        lidar_points=points,
        camera=camera,
    )
    rotated = apply_local_euler_delta(
        SE3(
            translation_m=(0.12, -0.04, 0.08),
            rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        (1.2, -0.7, 0.4),
    )

    expected = project_depth_pairs(observation, rotated)
    observed = project_depth_pairs_numba(observation, rotated)

    assert observed.projected_count_before_visibility == (
        expected.projected_count_before_visibility
    )
    assert np.array_equal(observed.camera_depth, expected.camera_depth)
    assert np.array_equal(observed.lidar_range_m, expected.lidar_range_m)
    assert np.array_equal(observed.pixel_u, expected.pixel_u)
    assert np.array_equal(observed.pixel_v, expected.pixel_v)
    fused = NumbaDepthPairProjector((observation,))
    for use_z_buffer in (True, False):
        pairs = project_depth_pairs(observation, rotated, use_z_buffer=use_z_buffer)
        histogram = fused.histogram_depth_pairs(
            observation, rotated, use_z_buffer=use_z_buffer, bins=32,
            camera_range=(0.5, 40), lidar_range=(1, 20),
        )
        oracle = np.histogram2d(
            pairs.camera_depth, pairs.lidar_range_m, bins=32, range=((0.5, 40), (1, 20)),
        )[0]
        assert np.array_equal(histogram.histogram, oracle)
        assert histogram.visible_point_count == pairs.camera_depth.size
        assert histogram.projected_count_before_visibility == (
            pairs.projected_count_before_visibility
        )


def _edge_observation() -> DepthToDepthObservation:
    values = np.concatenate((
        np.array([0, -1, np.nan, np.inf]), np.linspace(1, 9, 9),
        np.nextafter(np.linspace(1, 9, 9), -np.inf),
        np.nextafter(np.linspace(1, 9, 9), np.inf),
    ))
    width = values.size
    camera = DepthToDepthCameraModel(width=width, height=2, fx=1, fy=1, cx=0, cy=0)
    points = np.array([[float(i), 0, 1] for i in range(width)])
    observation = DepthToDepthObservation(
        frame_id="edges", camera=camera, lidar_points=points,
        depth_map=np.tile(values, (2, 1)),
    )
    # Range features can be frozen independently of the geometry. Exercise
    # histogram edges using exact values, not rounded norms of test rays.
    object.__setattr__(observation, "lidar_range_m", np.roll(values, 2).copy())
    return observation


def test_sparse_histogram_edges_visibility_and_workspace_reset() -> None:
    observation = _edge_observation()
    projector = NumbaDepthPairProjector((observation,))
    for pose in (SE3.identity(), SE3.identity(), SE3((1000, 0, 0), (0, 0, 0, 1)), SE3.identity()):
        for z_buffer in (True, False):
            for limits in ((1, 9), (2, 8), (1, 9)):
                pairs = project_depth_pairs(observation, pose, use_z_buffer=z_buffer)
                actual = projector.histogram_depth_pairs(
                    observation, pose, use_z_buffer=z_buffer, bins=8,
                    camera_range=limits, lidar_range=limits,
                )
                expected = np.histogram2d(
                    pairs.camera_depth, pairs.lidar_range_m, bins=8, range=(limits, limits),
                )[0]
                assert np.array_equal(actual.histogram, expected)
                assert actual.visible_point_count == pairs.camera_depth.size
                assert actual.projected_count_before_visibility == (
                    pairs.projected_count_before_visibility
                )


def test_sparse_histogram_nearest_ties_and_invalid_nearest_occlude_farther_points() -> None:
    camera = DepthToDepthCameraModel(width=3, height=2, fx=1, fy=1, cx=0, cy=0)
    observation = DepthToDepthObservation(
        frame_id="ties", camera=camera, depth_map=np.array([[2, np.nan, 3], [3, 3, 3]]),
        lidar_points=np.array([[0, 0, 2], [0, 0, 2], [1, 0, 1], [2, 0, 2], [4, 0, 2]]),
    )
    object.__setattr__(observation, "lidar_range_m", np.array([2., 4., 3., 4., 5.]))
    projector = NumbaDepthPairProjector((observation,))
    actual = projector.histogram_depth_pairs(
        observation, SE3.identity(), use_z_buffer=True, bins=8,
        camera_range=(1, 9), lidar_range=(1, 9),
    )
    assert actual.visible_point_count == 2
    assert actual.projected_count_before_visibility == 5
    assert actual.histogram[1, 1] == 1  # equal-range tie chooses the first input
    assert actual.histogram[2, 4] == 1
    assert actual.histogram.sum() == 2


def test_prepared_snapshot_and_concurrent_different_histogram_configurations() -> None:
    observation = _edge_observation()
    snapshot = replace(observation, depth_map=observation.depth_map.copy())
    object.__setattr__(snapshot, "lidar_range_m", observation.lidar_range_m.copy())
    projector = NumbaDepthPairProjector((observation,))
    observation.depth_map[:] = 0
    observation.lidar_range_m[:] = 0

    def evaluate(index):
        limits = (1, 9) if index % 2 else (2, 8)
        actual = projector.histogram_depth_pairs(
            observation, SE3.identity(), use_z_buffer=True, bins=8,
            camera_range=limits, lidar_range=limits,
        )
        pairs = project_depth_pairs(snapshot, SE3.identity())
        expected = np.histogram2d(
            pairs.camera_depth, pairs.lidar_range_m, bins=8, range=(limits, limits),
        )[0]
        return (
            np.array_equal(actual.histogram, expected),
            actual.visible_point_count == pairs.camera_depth.size,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(a and b for a, b in executor.map(evaluate, range(24)))
    expected = project_depth_pairs(snapshot, SE3.identity())
    assert np.array_equal(
        projector(observation, SE3.identity()).camera_depth, expected.camera_depth,
    )
    unprepared = replace(observation, frame_id="unprepared")
    assert projector.histogram_depth_pairs(
        unprepared, SE3.identity(), use_z_buffer=True, bins=8,
        camera_range=(1, 9), lidar_range=(1, 9),
    ).visible_point_count == 0


def test_sparse_scratch_handles_image_growth_empty_frames_and_native_failure(monkeypatch) -> None:
    from calibrex.solvers import numba_depth_to_depth_adapter as adapter

    projector = NumbaDepthPairProjector()
    for width in (3, 100, 2, 100):
        camera = DepthToDepthCameraModel(width=width, height=2, fx=1, fy=1, cx=0, cy=0)
        observation = DepthToDepthObservation(
            frame_id=str(width), camera=camera, depth_map=np.ones((2, width)),
            lidar_points=np.array([[float(i), 0, 1] for i in range(width)]),
        )
        empty = replace(observation, lidar_points=np.empty((0, 3)))
        for item in (observation, empty, observation):
            actual = projector.histogram_depth_pairs(
                item, SE3.identity(), use_z_buffer=True, bins=8,
                camera_range=(0.5, 2), lidar_range=(0.5, 200),
            )
            assert actual.visible_point_count == item.lidar_points.shape[0]
            assert actual.histogram.sum() == item.lidar_points.shape[0]
    original = adapter._sparse_histogram_fused

    def fail(*args):
        args[-3][0] = 10_000  # simulate a native failure after touching scratch
        raise RuntimeError("native failure")

    monkeypatch.setattr(adapter, "_sparse_histogram_fused", fail)
    with pytest.raises(RuntimeError, match="native failure"):
        projector.histogram_depth_pairs(
            observation, SE3.identity(), use_z_buffer=True, bins=8,
            camera_range=(0.5, 2), lidar_range=(0.5, 200),
        )
    monkeypatch.setattr(adapter, "_sparse_histogram_fused", original)
    assert projector.histogram_depth_pairs(
        observation, SE3.identity(), use_z_buffer=True, bins=8,
        camera_range=(0.5, 2), lidar_range=(0.5, 200),
    ).visible_point_count == 100


def test_overflowed_camera_range_keeps_numpy_first_tie_semantics() -> None:
    camera = DepthToDepthCameraModel(width=4, height=4, fx=1, fy=1, cx=0, cy=0)
    with np.errstate(over="ignore"):
        observation = DepthToDepthObservation(
            frame_id="overflow", camera=camera, depth_map=np.ones((4, 4)),
            lidar_points=np.array([[1e308, 1e308, 1e308]]),
        )
        object.__setattr__(observation, "lidar_range_m", np.array([1.]))
        expected = project_depth_pairs(observation, SE3.identity())
        observed = project_depth_pairs_numba(observation, SE3.identity())
        projector = NumbaDepthPairProjector((observation,))
    assert np.array_equal(observed.camera_depth, expected.camera_depth)
    assert expected.camera_depth.size == 1
    assert projector.histogram_depth_pairs(
        observation, SE3.identity(), use_z_buffer=True, bins=8,
        camera_range=(0.5, 2.5), lidar_range=(0.5, 2.5),
    ).visible_point_count == 1


def test_numba_adapter_records_dependency_and_preserves_no_z_buffer() -> None:
    camera = DepthToDepthCameraModel(
        width=16,
        height=12,
        fx=8.0,
        fy=8.0,
        cx=7.5,
        cy=5.5,
        distortion=(),  # type: ignore[arg-type]
    )
    observation = DepthToDepthObservation(
        frame_id="fallback",
        depth_map=np.ones((12, 16)),
        lidar_points=np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]),
        camera=camera,
    )

    expected = project_depth_pairs(
        observation,
        SE3.identity(),
        use_z_buffer=False,
    )
    observed = project_depth_pairs_numba(
        observation,
        SE3.identity(),
        use_z_buffer=False,
    )

    assert np.array_equal(observed.camera_depth, expected.camera_depth)
    assert np.array_equal(observed.lidar_range_m, expected.lidar_range_m)
    assert numba_depth_pair_projector_identity().startswith(
        f"{NUMBA_DEPTH_PAIR_PROJECTOR_VERSION};numba="
    )
