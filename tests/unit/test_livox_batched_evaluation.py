from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from calibrex.core.config import load_config
from calibrex.core.geometry import SE3
from calibrex.data import livox
from calibrex.evaluation.lidar import livox_pair_evidence_from_dataset


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
def test_batched_residuals_keep_scalar_order_bits_and_support(count, gate):
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
    reads = []

    def read(path):
        reads.append(Path(path))
        return original_read(path)

    monkeypatch.setattr(livox, "read_livox_binary_pcd_records", read)
    config = load_config(
        "examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml"
    )
    first = livox_pair_evidence_from_dataset(str(tmp_path), SE3.identity(), config)
    assert len(first.cases) == 24
    assert reads == [tmp_path / "a.pcd", tmp_path / "b.pcd"]
    _write_pcd(tmp_path / "b.pcd", 50.0)
    second = livox_pair_evidence_from_dataset(str(tmp_path), SE3.identity(), config)
    assert reads == [tmp_path / "a.pcd", tmp_path / "b.pcd"] * 2
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
    livox._point_to_plane_residuals(
        [(0.1, -0.1, 0.0)] * 5000,
        plane_map=_map(),
        voxel_size_m=1.0,
        correspondence_gate_m=1.5,
    )
    assert counts == [4096 * 27, (5000 - 4096) * 27]


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
