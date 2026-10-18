"""Reached-point native normals retain the original complete-map calculation."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.evaluation import lidar_lidar_map as evaluation
from calibrex.solvers.lidar_lidar_map_solver import MapSample, MapSampleOptions


def _eager(sample, options):
    sample._tree = cKDTree(sample.map_points)
    k = min(options.normal_neighbours, len(sample.map_points))
    _, indices = sample._tree.query(sample.map_points, k=k)
    neighbours = sample.map_points[indices]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
    values, vectors = np.linalg.eigh(covariance)
    sample._normals = vectors[:, :, 0]
    sample._planar = values[:, 0] <= options.max_planarity_ratio * values[:, 1]


@pytest.mark.parametrize("k", [3, 6, 10])
@pytest.mark.parametrize("ratio", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("workers", [1, 4])
def test_sparse_residuals_retain_every_native_bit(k, ratio, workers):
    rng = np.random.default_rng(812)
    points = rng.normal(size=(509, 3))
    points[4:9] = points[0]  # Native tie ordering and degenerate neighborhoods.
    queries = np.vstack([points[:23], points[:4], [[1000.0, 0.0, 0.0]]])
    options = MapSampleOptions(
        normal_neighbours=k, max_planarity_ratio=ratio, query_workers=workers,
    )
    lazy = MapSample(0.0, 0, np.eye(4), queries, points)
    eager = MapSample(0.0, 0, np.eye(4), queries, points)
    _eager(eager, options)
    for shift in (0.0, 0.03, -0.01, 0.0):
        transform = np.eye(4)
        transform[:3, 3] = shift
        for actual, expected in zip(
            lazy.residuals(transform, options), eager.residuals(transform, options), strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
    assert lazy._normal_cache is not None
    assert lazy._normal_cache.done.sum() < len(points) // 3
    assert lazy._normals is None and lazy._planar is None


def test_explicit_prepare_completes_partial_cache_and_retains_first_policy():
    rng = np.random.default_rng(613)
    points = rng.normal(size=(113, 3))
    sample = MapSample(0.0, 0, np.eye(4), points[:3], points)
    options = MapSampleOptions(normal_neighbours=6, max_planarity_ratio=0.0)
    sample.residuals(np.eye(4), options)
    sample.prepare(replace(options, normal_neighbours=10, max_planarity_ratio=1.0))
    expected = MapSample(0.0, 0, np.eye(4), points[:3], points)
    _eager(expected, options)
    np.testing.assert_array_equal(sample._normals, expected._normals)
    np.testing.assert_array_equal(sample._planar, expected._planar)
    assert sample._normal_cache.done.all()


def test_only_missing_reached_rows_are_queried_and_eager_callers_are_preserved():
    class RecordingTree(cKDTree):
        def query(self, values, *args, **kwargs):
            if "k" in kwargs:
                normal_rows.append(len(values))
            return super().query(values, *args, **kwargs)

    normal_rows = []
    points = np.random.default_rng(808).normal(size=(107, 3))
    sample = MapSample(0.0, 0, np.eye(4), points[[0, 0, 1]], points)
    sample._tree = RecordingTree(points)
    options = MapSampleOptions()
    first = sample.residuals(np.eye(4), options)
    sample.residuals(np.eye(4), options)
    assert normal_rows == [2]
    sample.prepare(options)
    assert normal_rows == [2, 105]
    for actual, expected in zip(sample.residuals(np.eye(4), options), first, strict=True):
        np.testing.assert_array_equal(actual, expected)
    sample._normals[:] = [1.0, 0.0, 0.0]
    sample._planar[:] = True
    _, normals, _ = sample.residuals(np.eye(4), options)
    np.testing.assert_array_equal(normals, np.tile([1.0, 0.0, 0.0], (3, 1)))
    assert normal_rows == [2, 105]


def test_failed_eigensolve_does_not_publish_partial_cache(monkeypatch):
    points = np.random.default_rng(552).normal(size=(113, 3))
    sample = MapSample(0.0, 0, np.eye(4), points[:3], points)
    options = MapSampleOptions()
    original = np.linalg.eigh

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("failed eigensolve")

    monkeypatch.setattr(np.linalg, "eigh", fail)
    with pytest.raises(np.linalg.LinAlgError, match="failed eigensolve"):
        sample.residuals(np.eye(4), options)
    assert sample._normal_cache is not None and not sample._normal_cache.done.any()
    assert sample._normals is None and sample._planar is None
    monkeypatch.setattr(np.linalg, "eigh", original)
    actual = sample.residuals(np.eye(4), options)
    eager = MapSample(0.0, 0, np.eye(4), points[:3], points)
    _eager(eager, options)
    for values, expected in zip(actual, eager.residuals(np.eye(4), options), strict=True):
        np.testing.assert_array_equal(values, expected)


def test_no_reached_rows_do_not_trigger_eigensolves(monkeypatch):
    points = np.random.default_rng(193).normal(size=(113, 3))
    sample = MapSample(0.0, 0, np.eye(4), np.full((7, 3), 1000.0), points)

    def fail(*args, **kwargs):
        pytest.fail("no map points were reached")

    monkeypatch.setattr(np.linalg, "eigh", fail)
    residual, normals, targets = sample.residuals(np.eye(4), MapSampleOptions())
    assert residual.shape == (0,) and normals.shape == targets.shape == (0, 3)
    assert not sample._normal_cache.done.any()


def test_complete_evaluation_and_every_jackknife_fit_match_eager(monkeypatch):
    rng = np.random.default_rng(973)
    faces = []
    for axis in range(3):
        for value in (-2.0, 2.0):
            points = rng.uniform(-2.0, 2.0, (300, 3))
            points[:, axis] = value
            faces.append(points)
    world = np.concatenate(faces)
    raw = []
    for index in range(9):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(rng.normal(0.0, 0.05, 3)).as_matrix()
        pose[:3, 3] = rng.normal(0.0, 0.1, 3)
        selected = world[rng.choice(len(world), 180, replace=False)]
        targets = (selected - pose[:3, 3]) @ pose[:3, :3]
        targets += rng.normal(0.0, 0.01, targets.shape)
        raw.append((float(index), index, pose, targets, world))
    options = evaluation.LidarLidarMapOptions(
        jackknife_groups=4, solver=MapSampleOptions(query_workers=4),
    )
    initial = np.eye(4)
    initial[:3, 3] = [0.04, -0.03, 0.02]
    original = evaluation.solve_map_extrinsic
    fits = []

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        fits.append(asdict(result))
        return result

    monkeypatch.setattr(evaluation, "solve_map_extrinsic", recorded)
    eager = [MapSample(*arguments) for arguments in raw]
    for sample in eager:
        _eager(sample, options.solver)
    expected = evaluation.evaluate_lidar_lidar_map(
        eager, initial, options, reference_transform=np.eye(4),
    )
    expected_fits = fits[:]
    fits.clear()
    lazy = [MapSample(*arguments) for arguments in raw]
    actual = evaluation.evaluate_lidar_lidar_map(
        lazy, initial, options, reference_transform=np.eye(4),
    )
    assert len(expected_fits) == len(fits) == 5
    np.testing.assert_equal(fits, expected_fits)
    np.testing.assert_equal(asdict(actual), asdict(expected))
    assert all(record.known_bad_control is not None for record in actual.records)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "4"])
def test_invalid_map_worker_budgets_are_rejected(workers):
    with pytest.raises(ValueError, match="positive integer"):
        MapSampleOptions(query_workers=workers)


@pytest.mark.parametrize("workers", [1, 4])
def test_cli_forwards_worker_budget_and_command_provenance(tmp_path, monkeypatch, workers):
    initial = tmp_path / "initial.yaml"
    initial.write_text("matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n")
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        raise ValueError("dispatch probe")

    monkeypatch.setattr(evaluation, "run_ros2_lidar_lidar_map", fake_run)
    args = [
        "lidar-lidar", "ros2", "bag", "--reference-topic", "/reference",
        "--target-topic", "/target", "--initial", str(initial), "--dataset-family", "synthetic",
        "--dataset-license", "test", "--output", str(tmp_path / "result.json"),
    ]
    if workers != 1:
        args += ["--registration-query-workers", str(workers)]
    assert main(args) == 2
    assert captured["options"].solver.query_workers == workers
    if workers != 1:
        index = captured["command"].index("--registration-query-workers")
        assert captured["command"][index + 1] == str(workers)
    else:
        assert "--registration-query-workers" not in captured["command"]


@pytest.mark.parametrize("workers", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_worker_budget(workers):
    with pytest.raises(SystemExit) as error:
        main(["lidar-lidar", "ros2", "bag", "--registration-query-workers", workers])
    assert error.value.code == 2


def test_bag_api_records_nondefault_query_budget(monkeypatch):
    monkeypatch.setattr(evaluation, "collect_map_samples", lambda *args, **kwargs: ([], 0))
    monkeypatch.setattr(evaluation, "bag_input_digest", lambda *args: ("a" * 64, "synthetic"))
    artifact = evaluation.run_ros2_lidar_lidar_map(
        "synthetic-bag", reference_topic="/reference", target_topic="/target",
        initial=np.eye(4), point_time_field=None, dataset_family="synthetic",
        dataset_license="test", options=evaluation.LidarLidarMapOptions(
            solver=MapSampleOptions(query_workers=4),
        ),
    )
    assert artifact.options["registration_query_workers"] == 4
