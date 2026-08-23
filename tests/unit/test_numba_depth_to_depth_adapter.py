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
