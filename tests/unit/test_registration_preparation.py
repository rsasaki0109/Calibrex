from dataclasses import replace
from threading import Event
from time import sleep

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers import chen_medioni_point_to_plane_icp_solver as plane
from calibrex.solvers import native_registration_comparison_solver as native
from calibrex.solvers import robust_point_to_point_icp_solver as point


@pytest.mark.parametrize("voxel", [0.01, 0.5, 3.0])
@pytest.mark.parametrize("cap", [1, 31, 10000])
def test_centroids_keep_input_addition_order_ids_and_uniform_cap(voxel, cap):
    rng = np.random.default_rng(241)
    coordinates = rng.normal(size=(2003, 4)) * 4.0
    coordinates[0, :3] = np.nan
    coordinates[1, :3] = np.inf
    coordinates[2:5, :3] = (0.0, -0.0, 0.0)
    # Deliberate rounding inside a single voxel. Pairwise summation can change
    # the centroid from the scalar reference.
    if voxel == 3.0:
        coordinates[5:8, :3] = [(1.0, 1.0, 1.0), (1.0e-16, 1.0e-16, 1.0e-16),
                               (1.0e-16, 1.0e-16, 1.0e-16)]
    expected = native._scalar_voxel_centroids(coordinates, "cloud", voxel, cap)
    for values in (coordinates, coordinates.tolist(), coordinates[::-1][::-1]):
        actual = native.voxel_downsample_icp_points(values, "cloud", voxel, cap)
        assert [item.point_id for item in actual] == [item.point_id for item in expected]
        assert np.asarray([item.position_m for item in actual]).tobytes() == np.asarray(
            [item.position_m for item in expected]).tobytes()


@pytest.mark.parametrize("values,voxel", [
    ([(1.0e30, -1.0e30, 0.0), (1.0e30, -1.0e30, 0.0)], 0.5),
    ([(float(2**63), 0.0, 0.0), (-float(2**63), 0.0, 0.0)], 1.0),
    ([(0.1, -0.1, -0.0), (1.0, 0.0, 0.0)], float("inf")),
    ([(1.0e16, 0.0, 0.0), (-1.0e16, 0.0, 0.0), (1.0, 0.0, 0.0)], float("inf")),
    ([], 1.0),
    ([(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)], 1.0),
    ([(0.1, 0.2, 0.3, "ignored"), (0.2, 0.3, 0.4)], 0.5),
])
def test_centroids_preserve_empty_extra_fields_and_arbitrary_precision_keys(values, voxel):
    assert native.voxel_downsample_icp_points(values, "edge", voxel, 3) == (
        native._scalar_voxel_centroids(values, "edge", voxel, 3))


@pytest.mark.parametrize("values,voxel,error", [
    ([(1.0e308, 0.0, 0.0)], 1.0e-308, OverflowError),
    ([(1.0, 0.0, 0.0)], float("nan"), ValueError),
    ([(1.0, 0.0)], 0.5, IndexError),
    (np.ones((3, 2)), 0.5, IndexError),
])
def test_centroids_preserve_scalar_errors(values, voxel, error):
    with pytest.raises(error):
        native.voxel_downsample_icp_points(values, "bad", voxel, 3)


def _scalar_normals(target, options):
    neighborhoods, _ = plane._ExactNeighbors(target).query(
        target, min(options.normal_neighbor_count, len(target)))
    normals = np.zeros_like(target)
    valid = np.zeros(len(target), dtype=np.bool_)
    eigengaps = np.zeros(len(target))
    for index, neighbors in enumerate(neighborhoods):
        points = target[neighbors]
        centered = points - np.mean(points, axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
        gap = max(float(eigenvalues[1] - eigenvalues[0]), 0.0) / max(
            float(eigenvalues[2]), 1.0e-15)
        eigengaps[index] = gap
        if gap >= options.minimum_normal_eigengap:
            normals[index] = eigenvectors[:, 0]
            valid[index] = True
    return normals, valid, eigengaps


@pytest.mark.parametrize("count,neighbors,gap", [(53, 3, 0.02), (71, 12, 0.02),
                                               (17, 100, 0.02), (31, 5, 100.0),
                                               (5501, 12, 0.02)])
def test_batched_normals_are_bit_identical_across_batch_and_neighborhood_boundaries(
    count, neighbors, gap,
):
    rng = np.random.default_rng(86)
    target = rng.normal(size=(count, 3))
    options = plane.ChenMedioniPointToPlaneOptions(
        normal_neighbor_count=neighbors, minimum_normal_eigengap=gap)
    for actual, expected in zip(plane._estimate_target_normals(target, options),
                                _scalar_normals(target, options), strict=True):
        assert actual.tobytes() == expected.tobytes()


def test_batched_normals_keep_duplicate_and_planar_eigensystems():
    target = np.asarray([(x, y, 0.) for x in range(6) for y in range(6)] * 2)
    options = plane.ChenMedioniPointToPlaneOptions()
    for actual, expected in zip(plane._estimate_target_normals(target, options),
                                _scalar_normals(target, options), strict=True):
        assert actual.tobytes() == expected.tobytes()


def test_information_matrix_keeps_rows_signed_zeros_and_svd(monkeypatch):
    rng = np.random.default_rng(739)
    points = rng.normal(size=(67, 3))
    points[0] = (0.0, -0.0, 0.0)
    rows = []
    for x, y, z in points:
        rows.extend([[1., 0., 0., 0., z, -y], [0., 1., 0., -z, 0., x],
                     [0., 0., 1., y, -x, 0.]])
    expected = np.asarray(rows)
    original_svd = np.linalg.svd

    def checked_svd(matrix, *, compute_uv):
        assert matrix.tobytes() == expected.tobytes()
        return original_svd(matrix, compute_uv=compute_uv)

    monkeypatch.setattr(np.linalg, "svd", checked_svd)
    spectrum, rank, condition = point._information_diagnostics(
        points, point.RobustPointToPointIcpOptions())
    values = original_svd(expected, compute_uv=False)
    assert spectrum == tuple(values)
    assert rank == 6
    assert condition == values[0] / values[5]


@pytest.mark.parametrize("workers", [0, 2, 4])
def test_parallel_multi_start_preserves_complete_payload_and_direction_order(monkeypatch, workers):
    from tests.unit.test_robust_point_to_point_icp_solver import _clouds

    monkeypatch.setattr(point.os, "cpu_count", lambda: 4)
    # Exercise the automatic path without making every unit test a large scan.
    source, target = _clouds(SE3((.05, -.03, .02), (.002, -.003, .004, .99998)), count=1500)
    options = point.RobustPointToPointIcpOptions(max_iterations=4, multi_start_workers=1)
    expected = point.RobustPointToPointIcpSolver().solve(source, target, options=options).as_dict()
    actual = point.RobustPointToPointIcpSolver().solve(
        source, target, options=replace(options, multi_start_workers=workers)).as_dict()
    assert actual == expected
    assert [trial["direction"] for trial in actual["multi_start_trials"]] == [
        "x", "y", "z", "roll", "pitch", "yaw"]


def test_invalid_worker_count_is_rejected():
    with pytest.raises(ValueError, match="multi_start_workers"):
        point.RobustPointToPointIcpSolver().solve(
            [], [], options=point.RobustPointToPointIcpOptions(multi_start_workers=-1))


def test_multi_start_joins_running_workers_before_propagating_failure(monkeypatch):
    running = Event()
    finished = Event()

    def failing_solve(self, source, target, *, initial_transform, options):
        if initial_transform.translation_m[0] > 0.0:
            assert running.wait(timeout=2.0)
            raise RuntimeError("trial failed")
        running.set()
        sleep(0.02)
        finished.set()
        return point._empty("insufficient_correspondences", (), ())

    monkeypatch.setattr(point.RobustPointToPointIcpSolver, "solve", failing_solve)
    with pytest.raises(RuntimeError, match="trial failed"):
        point._multi_start_diagnostics(
            [], [], SE3.identity(), SE3.identity(), None,
            point.RobustPointToPointIcpOptions(multi_start_workers=2))
    assert finished.is_set()
