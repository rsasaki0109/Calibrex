from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from calibrex.core.config import load_config
from calibrex.core.geometry import SE3
from calibrex.data import livox
from calibrex.data.downloads import LIVOX_BASE_PCD_NAME, LIVOX_TARGET_PCD_NAME
from calibrex.evaluation.lidar import livox_pair_evidence_from_dataset
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration


def _map():
    rng = np.random.default_rng(91)
    points = rng.uniform(-3, 3, (400, 3))
    normals = rng.normal(size=(400, 3))
    records = [
        livox.LivoxPointRecord((*point, 0.0), tuple(normal))
        for point, normal in zip(points.tolist(), normals.tolist(), strict=True)
    ]
    return livox.build_voxel_plane_map(records, 1.0)


@pytest.mark.parametrize("count", [0, 17, 64, 4096, 4200])
@pytest.mark.parametrize("gate", [0.0, 0.75, 1.5, float("inf"), float("nan")])
@pytest.mark.parametrize("index_budget", [0, 8 * 1024 * 1024])
def test_batched_residuals_keep_scalar_order_bits_and_support(
    count, gate, index_budget, monkeypatch,
):
    monkeypatch.setattr(livox, "_DENSE_PLANE_INDEX_MAX_BYTES", index_budget)
    rng = np.random.default_rng(32)
    points = [tuple(row) for row in rng.uniform(-4, 4, (count, 3)).tolist()]
    options = {"plane_map": _map(), "voxel_size_m": 1.0, "correspondence_gate_m": gate}
    expected, excluded = livox._scalar_plane_residuals(points, **options)
    actual, unmatched = livox._point_to_plane_residuals(points, **options)
    assert unmatched == excluded
    assert np.asarray(actual, dtype="<f8").tobytes() == np.asarray(expected, dtype="<f8").tobytes()


def test_equal_distances_invalid_nearest_normal_and_voxel_neighborhood():
    first = livox._VoxelPlane(count=1, normal_x_sum=1.0, normal_count=1)
    second = livox._VoxelPlane(count=1, x_sum=2.0, normal_y_sum=1.0, normal_count=1)
    options = {"voxel_size_m": 1.0, "correspondence_gate_m": 5.0}
    points = [(1.0, 0.0, 0.0)] * 64
    assert livox._point_to_plane_residuals(
        points,
        plane_map={(2, 0, 0): second, (0, 0, 0): first},
        **options,
    ) == ([1.0] * 64, 0)
    first.normal_count = 0
    assert livox._point_to_plane_residuals(
        points,
        plane_map={(2, 0, 0): second, (0, 0, 0): first},
        **options,
    ) == ([], 64)
    # A nearby centroid outside the declared 27 voxel keys is ineligible.
    assert livox._point_to_plane_residuals(
        points,
        plane_map={(1000, 0, 0): second},
        **options,
    ) == ([], 64)


def test_sparse_large_extent_uses_exact_structured_fallback():
    extent = 2**62
    points = [(-float(extent), 0.0, 0.0), (float(extent), 0.0, 0.0)] * 32
    records = [livox.LivoxPointRecord((*point, 0.0), (1.0, 0.0, 0.0)) for point in points]
    options = {
        "plane_map": livox.build_voxel_plane_map(records, 1.0),
        "voxel_size_m": 1.0, "correspondence_gate_m": 1.5,
    }
    assert livox._point_to_plane_residuals(points, **options) == ([0.0] * 64, 0)
    assert livox._point_to_plane_residuals(points, **options) == (
        livox._scalar_plane_residuals(points, **options)
    )


@pytest.mark.parametrize("coordinate,voxel", [(1e20, 1.0), (-1e20, 1.0), (0.0, 1.0)])
def test_large_voxel_keys_and_empty_maps_retain_scalar_behavior(coordinate, voxel):
    records = [livox.LivoxPointRecord((coordinate, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
    options = {
        "plane_map": livox.build_voxel_plane_map(records, voxel),
        "voxel_size_m": voxel,
        "correspondence_gate_m": 1.5,
    }
    points = [(coordinate, -0.0, 0.0)] * 64
    assert livox._point_to_plane_residuals(points, **options) == (
        livox._scalar_plane_residuals(points, **options)
    )
    options["plane_map"] = {}
    assert livox._point_to_plane_residuals(points, **options) == ([], 64)


@pytest.mark.parametrize("coordinate,voxel", [(float("nan"), 1.0), (1.0, 0.0), (1e300, 1e-300)])
def test_unsupported_coordinates_keep_original_exceptions(coordinate, voxel):
    points = [(coordinate, 0.0, 0.0)] * 64
    options = {"plane_map": {}, "voxel_size_m": voxel, "correspondence_gate_m": 1.5}
    with pytest.raises((ValueError, OverflowError, ZeroDivisionError)) as expected:
        livox._scalar_plane_residuals(points, **options)
    with pytest.raises(type(expected.value)):
        livox._point_to_plane_residuals(points, **options)


def _write_pcd(path: Path, offset: float = 0.0) -> None:
    points = [(0.03 * i + offset, 0.02 * (i % 7), 1.0, 2.0, 0.0, 0.0, 1.0) for i in range(100)]
    header = (
        "FIELDS x y z intensity normal_x normal_y normal_z\n"
        "SIZE 4 4 4 4 4 4 4\nTYPE F F F F F F F\nCOUNT 1 1 1 1 1 1 1\n"
        f"POINTS {len(points)}\nDATA binary\n"
    )
    path.write_bytes(header.encode() + b"".join(struct.pack("<7f", *point) for point in points))


@pytest.mark.parametrize("sample_limit", [-1, 0, 1, 2, 4])
def test_scoped_statistics_match_uncached_with_malformed_and_extra_files(tmp_path, sample_limit):
    _write_pcd(tmp_path / "a.pcd")
    (tmp_path / "b.pcd").write_text("malformed")
    _write_pcd(tmp_path / "c.pcd", 0.1)
    _write_pcd(tmp_path / "d.pcd", 0.2)
    context = livox.LivoxPCDEvaluationContext(tmp_path)
    for transform in [None, SE3.identity(), SE3((0.1, -0.2, 0.05), (0.1, 0.2, 0.3, 0.9))]:
        options = {"sample_limit": sample_limit, "target_transform": transform}
        assert livox.summarize_livox_pcd(tmp_path, context=context, **options).as_dict() == (
            livox.summarize_livox_pcd(tmp_path, **options).as_dict()
        )
        assert (
            livox.summarize_livox_pair_point_to_plane(
                tmp_path,
                target_transform=transform,
                context=context,
            ).as_dict()
            == livox.summarize_livox_pair_point_to_plane(
                tmp_path,
                target_transform=transform,
            ).as_dict()
        )


def test_probe_context_reads_each_file_once_and_next_invocation_observes_replacement(
    tmp_path,
    monkeypatch,
):
    _write_pcd(tmp_path / "a.pcd")
    _write_pcd(tmp_path / "b.pcd", 0.1)
    original_read = livox.read_livox_binary_pcd_records
    original_index = livox._prepare_plane_index
    original_voxels = livox._voxel_centroids
    reads = []
    prepared = []
    voxel_inputs = []

    def read(path):
        reads.append(Path(path))
        return original_read(path)

    monkeypatch.setattr(livox, "read_livox_binary_pcd_records", read)

    def prepare(plane_map):
        prepared.append(plane_map)
        return original_index(plane_map)

    def voxelize(points, voxel_size_m):
        voxel_inputs.append(points)
        return original_voxels(points, voxel_size_m)

    monkeypatch.setattr(livox, "_prepare_plane_index", prepare)
    monkeypatch.setattr(livox, "_voxel_centroids", voxelize)
    config = load_config(
        "examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml"
    )
    first = livox_pair_evidence_from_dataset(str(tmp_path), SE3.identity(), config)
    assert len(first.cases) == 24
    assert reads == [tmp_path / "a.pcd", tmp_path / "b.pcd"]
    assert len(prepared) == 1
    # The source list keeps its identity across probes; it is accumulated once.
    assert sum(points is voxel_inputs[0] for points in voxel_inputs) == 1
    _write_pcd(tmp_path / "b.pcd", 50.0)
    second = livox_pair_evidence_from_dataset(str(tmp_path), SE3.identity(), config)
    assert reads == [tmp_path / "a.pcd", tmp_path / "b.pcd"] * 2
    assert len(prepared) == 2
    assert prepared[0] is not prepared[1]
    assert first.point_to_plane.matched_point_count > second.point_to_plane.matched_point_count


def test_context_rejects_different_dataset_and_queries_are_bounded(tmp_path, monkeypatch):
    context = livox.LivoxPCDEvaluationContext(tmp_path)
    with pytest.raises(ValueError, match="different dataset"):
        livox.summarize_livox_pcd(tmp_path / "other", context=context)
    searchsorted = np.searchsorted
    counts = []

    def search(keys, queries):
        counts.append(len(queries))
        return searchsorted(keys, queries)

    monkeypatch.setattr(np, "searchsorted", search)
    monkeypatch.setattr(livox, "_DENSE_PLANE_INDEX_MAX_BYTES", 0)
    livox._point_to_plane_residuals(
        [(0.1, -0.1, 0.0)] * 5000,
        plane_map=_map(),
        voxel_size_m=1.0,
        correspondence_gate_m=1.5,
    )
    assert counts == [4096 * 27, (5000 - 4096) * 27]


@pytest.mark.parametrize("budget", [0, 8, 64, 8 * 1024 * 1024])
def test_dense_index_stays_bounded_and_rejects_holes_and_clipped_voxels(budget, monkeypatch):
    monkeypatch.setattr(livox, "_DENSE_PLANE_INDEX_MAX_BYTES", budget)
    plane_map = {
        (-2, 0, 0): livox._VoxelPlane(count=1, x_sum=-1.5, normal_x_sum=1, normal_count=1),
        (2, 0, 0): livox._VoxelPlane(count=1, x_sum=2.5, normal_x_sum=1, normal_count=1),
    }
    index = livox._prepare_plane_index(plane_map)
    assert index is not None
    if index.dense_positions is not None:
        assert index.dense_positions.nbytes <= budget
    points = [(x, 0.0, 0.0) for x in (-10.0, -3.2, -1.1, 0.1, 1.1, 3.2, 10.0)] * 10
    options = {"plane_map": plane_map, "voxel_size_m": 1.0, "correspondence_gate_m": 20.0}
    expected, excluded = livox._scalar_plane_residuals(points, **options)
    actual, unmatched = livox._point_to_plane_residuals(points, index=index, **options)
    assert unmatched == excluded
    assert np.asarray(actual, dtype="<f8").tobytes() == np.asarray(expected, dtype="<f8").tobytes()


def test_standalone_residual_evaluation_observes_mutated_plane_map():
    plane = livox._VoxelPlane(count=1, normal_x_sum=1.0, normal_count=1)
    plane_map = {(0, 0, 0): plane}
    points = [(0.1, 0.0, 0.0)] * 64
    options = {"plane_map": plane_map, "voxel_size_m": 1.0, "correspondence_gate_m": 2.0}
    assert livox._point_to_plane_residuals(points, **options) == ([0.1] * 64, 0)
    plane.x_sum = 1.0
    assert livox._point_to_plane_residuals(points, **options) == ([0.9] * 64, 0)


@pytest.mark.parametrize("count", [0, 17, 255, 256, 4096])
@pytest.mark.parametrize("voxel", [0.01, 1.0, -0.5])
def test_array_voxels_preserve_scalar_sums_first_seen_order_and_centroid_bits(count, voxel):
    rng = np.random.default_rng(109)
    points = [(*row, float("nan")) for row in rng.uniform(-3, 3, (count, 3)).tolist()]
    expected = {}
    for point in points:
        key = livox._voxel_key(point, voxel)
        if key not in expected:
            expected[key] = livox._VoxelCentroid()
        expected[key].add(point)
    actual = livox._voxel_centroids(points, voxel)
    assert list(actual) == list(expected)
    for key in expected:
        a, b = actual[key], expected[key]
        assert a.count == b.count
        assert np.asarray([a.x_sum, a.y_sum, a.z_sum, *a.centroid]).tobytes() == np.asarray(
            [b.x_sum, b.y_sum, b.z_sum, *b.centroid],
        ).tobytes()


def test_unbuffered_voxel_accumulation_preserves_nonassociative_sums_and_zero_signs():
    points = [(1.0e16, -0.0, 0.0, 0.0), (1.0, 0.0, -0.0, 0.0)]
    points += [(1.0, -0.0, 0.0, 0.0)] * 298
    for voxel in (1.0e20, -1.0e20):
        expected = {}
        for point in points:
            key = livox._voxel_key(point, voxel)
            expected.setdefault(key, livox._VoxelCentroid()).add(point)
        actual = livox._voxel_centroids(points, voxel)
        assert list(actual) == list(expected)
        assert np.sum([p[0] for p in points]) != next(iter(expected.values())).x_sum
        for key in expected:
            assert np.asarray(actual[key].centroid).tobytes() == np.asarray(
                expected[key].centroid,
            ).tobytes()


@pytest.mark.parametrize("value", [1.0e30, -1.0e30, 1.0e12])
def test_large_voxel_keys_and_extent_products_retain_arbitrary_precision(value):
    points = [(value, -value, value, 0.0), (-value, value, -value, 0.0)] * 130
    result = livox._voxel_centroids(points, 0.25)
    assert list(result) == [livox._voxel_key(points[0], 0.25),
                            livox._voxel_key(points[1], 0.25)]
    assert all(v.count == 130 for v in result.values())


@pytest.mark.parametrize("voxel,error", [(0.0, ZeroDivisionError),
                                         (float("nan"), ValueError),
                                         (1.0e-300, OverflowError)])
def test_array_voxel_guards_preserve_scalar_invalid_input_exceptions(voxel, error):
    with pytest.raises(error):
        livox._voxel_centroids([(1.0e100, 0.0, 0.0, 0.0)] * 256, voxel)


@pytest.mark.parametrize("budget", [0, 8, 448, 496, 4096, 8 * 1024 * 1024])
def test_packed_neighborhoods_share_lookup_budget_and_keep_offset_order(budget, monkeypatch):
    monkeypatch.setattr(livox, "_DENSE_PLANE_INDEX_MAX_BYTES", budget)
    plane_map = _map()
    index = livox._prepare_plane_index(plane_map)
    assert index is not None
    size = index.dense_positions.nbytes if index.dense_positions is not None else 0
    packed = index.neighborhoods
    if packed is not None:
        size += sum(a.nbytes for a in (
            packed.lower, packed.upper, packed.offsets, packed.positions,
        ))
        assert not packed.offsets.flags.writeable
        assert not packed.positions.flags.writeable
        for key in plane_map:
            encoded = (
                ((key[0] - int(packed.lower[0])) * packed.widths[1]
                 + key[1] - int(packed.lower[1])) * packed.widths[2]
                + key[2] - int(packed.lower[2])
            )
            expected = [k for k in livox._neighbor_keys(key) if k in plane_map]
            positions = packed.positions[packed.offsets[encoded]:packed.offsets[encoded + 1]]
            assert [tuple(index.keys[p]) for p in positions] == expected
    assert size <= budget


def test_context_reuses_only_latest_transform_array_and_preserves_signed_zero(tmp_path):
    _write_pcd(tmp_path / "a.pcd")
    context = livox.LivoxPCDEvaluationContext(tmp_path)
    path = context._files[0]
    first_pose = SE3.identity()
    first = context._query_coordinates(path, first_pose)
    assert context._query_coordinates(path, first_pose) is first
    assert not first.flags.writeable
    other_pose = SE3((-0.0, 0.0, 0.0), (0.0, -0.0, 0.0, 1.0))
    other = context._query_coordinates(path, other_pose)
    assert other is not first
    assert len(context._queries) == 1
    expected = livox._transform_query_points(context.read_points(path), other_pose)
    assert other.tobytes() == np.asarray(expected).tobytes()
    assert context._queries[path][0] is other_pose


def test_dense_neighborhoods_keep_exact_residuals_and_prepared_arrays_are_readonly():
    plane_map = {
        (x, y, z): livox._VoxelPlane(
            count=1, x_sum=x + 0.5, y_sum=y + 0.5, z_sum=z + 0.5,
            normal_x_sum=1.0, normal_count=1,
        )
        for x in range(-3, 4) for y in range(-3, 4) for z in range(-3, 4)
    }
    index = livox._prepare_plane_index(plane_map)
    assert index is not None
    with pytest.raises(ValueError, match="read-only"):
        index.centroids[0, 0] = 100.0
    rng = np.random.default_rng(72)
    points = [tuple(row) for row in rng.uniform(-1, 1, (4200, 3)).tolist()]
    options = {"plane_map": plane_map, "voxel_size_m": 1.0, "correspondence_gate_m": 1.5}
    expected, excluded = livox._scalar_plane_residuals(points, **options)
    actual, unmatched = livox._point_to_plane_residuals(points, index=index, **options)
    assert unmatched == excluded
    assert np.asarray(actual, dtype="<f8").tobytes() == np.asarray(expected, dtype="<f8").tobytes()


def test_context_reuses_effective_samples_and_returns_independent_point_lists(
    tmp_path, monkeypatch,
):
    _write_pcd(tmp_path / "a.pcd")
    _write_pcd(tmp_path / "b.pcd", 0.1)
    context = livox.LivoxPCDEvaluationContext(tmp_path)
    original = livox._summarize_sample
    samples = []

    def summarize(path, points, **kwargs):
        samples.append(path)
        return original(path, points, **kwargs)

    monkeypatch.setattr(livox, "_summarize_sample", summarize)
    first = livox.summarize_livox_pcd(tmp_path, sample_limit=2, context=context)
    assert livox.summarize_livox_pcd(tmp_path, sample_limit=4, context=context) is first
    assert samples == [tmp_path / "a.pcd", tmp_path / "b.pcd"]
    limited = livox.summarize_livox_pcd(tmp_path, sample_limit=1, context=context)
    assert limited.sampled_file_count == 1
    assert len(samples) == 3
    returned = context.read_points(tmp_path / "a.pcd")
    returned.clear()
    assert len(context.read_points(tmp_path / "a.pcd")) == 100
    with pytest.raises(ValueError, match="not part"):
        context.read_points(tmp_path / "other.pcd")


def test_pipeline_decodes_each_capture_once_and_next_run_sees_replaced_input(tmp_path, monkeypatch):
    source = tmp_path / LIVOX_BASE_PCD_NAME
    target = tmp_path / LIVOX_TARGET_PCD_NAME
    _write_pcd(source)
    _write_pcd(target, 0.1)
    config = load_config(
        "examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml"
    )
    config.dataset.path = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    original = livox.read_livox_binary_pcd_records
    reads = []

    def read(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(livox, "read_livox_binary_pcd_records", read)
    first = run_calibration(config_path, CalibrationRunOptions(output_dir=tmp_path / "first"))
    assert first is not None
    assert reads == [source, target]
    _write_pcd(target, 50.0)
    second = run_calibration(config_path, CalibrationRunOptions(output_dir=tmp_path / "second"))
    assert second is not None
    assert reads == [source, target] * 2
    first_evidence = first.run.provenance["livox_pair_evidence"]["holdout_geometry"]
    second_evidence = second.run.provenance["livox_pair_evidence"]["holdout_geometry"]
    assert first_evidence["matched_point_count"] > second_evidence["matched_point_count"]


@pytest.mark.parametrize("count", [0, 17, 64, 5000])
@pytest.mark.parametrize("quaternion", [(0.0, -0.0, 0.0, 1.0), (0.1, -0.2, 0.3, 0.9)])
def test_batched_transform_keeps_hamilton_product_rounding(count, quaternion):
    rng = np.random.default_rng(45)
    points = [(*row, 7.0) for row in rng.normal(size=(count, 3)).tolist()]
    if points:
        points[0] = (-0.0, 0.0, -0.0, 7.0)
    transform = SE3((0.1, -0.2, -0.0), quaternion)
    expected = [transform.transform_point(point[:3]) for point in points]
    actual = livox._transform_query_points(points, transform)
    assert np.asarray(actual, dtype="<f8").tobytes() == np.asarray(expected, dtype="<f8").tobytes()
    assert [point[3] for point in livox._transform_points(points, transform)] == [7.0] * count
