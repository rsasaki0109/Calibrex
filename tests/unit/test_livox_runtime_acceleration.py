from __future__ import annotations

import dataclasses
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from calibrex.core.config import load_config
from calibrex.core.geometry import SE3
from calibrex.data import livox
from calibrex.data.inspect import inspect_dataset
from calibrex.evaluation.lidar import (
    build_rig_point_to_plane_observations,
    livox_pair_evidence_from_dataset,
)
from calibrex.evaluation.solid_state import (
    _livox_distance_bin_counts,
    solid_state_metrics_from_inspection,
)
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration


def _pcd(path: Path, points, *, normal=True, advertised=None, padding=0, tail=b""):
    fields = "x y z intensity" + (" normal_z padding normal_x normal_y" if normal else "")
    size = "4 4 4 4" + (f" 4 {padding or 4} 4 4" if normal else "")
    types = "F F F F" + (" F U F F" if normal else "")
    count = len(points) if advertised is None else advertised
    header = (f"FIELDS {fields}\nSIZE {size}\nTYPE {types}\n"
              f"POINTS {count}\nDATA binary\n").encode()
    content = bytearray(header)
    for point in points:
        content.extend(struct.pack("<ffff", *point[:4]))
        if normal:
            nx, ny, nz = point[4:7]
            content.extend(struct.pack("<f", nz))
            content.extend(bytes(padding or 4))
            content.extend(struct.pack("<ff", nx, ny))
    path.write_bytes(content + tail)


def _record_bits(records):
    return b"".join(struct.pack("<4d", *record.point) + (
        b"n" if record.normal_xyz is None else b"v" + struct.pack("<3d", *record.normal_xyz)
    ) for record in records)


@pytest.mark.parametrize("count", [0, 1, 255, 256, 257, 4097])
@pytest.mark.parametrize("normal,padding", [(False, 0), (True, 1), (True, 12)])
def test_binary_array_reader_matches_original_scalar_bits(
    count, normal, padding, tmp_path, monkeypatch,
):
    rng = np.random.default_rng(94)
    points = rng.normal(size=(count, 7)).astype(np.float32).tolist()
    for index, values in enumerate(points):
        if index % 37 == 0:
            values[:4] = [-0.0, 0.0, -0.0, -0.0]
        if index % 43 == 0:
            values[4:] = [0.0, -0.0, 0.0]
        if index % 47 == 1:
            values[0] = math.nan
        if index % 53 == 2:
            values[4] = math.inf
        if index % 59 == 3:
            values[3] = math.nan
        if index % 61 == 4:
            values[3] = math.inf
    path = tmp_path / "sample.pcd"
    _pcd(path, points, normal=normal, padding=padding, advertised=count + 17, tail=b"abc")
    actual = livox.read_livox_binary_pcd_records(path)
    monkeypatch.setattr(livox, "_PCD_ARRAY_MIN_POINTS", 10**12)
    expected = livox.read_livox_binary_pcd_records(path)
    assert _record_bits(actual) == _record_bits(expected)


def test_binary_count_and_signal_nan_payloads_preserve_scalar_bytes(tmp_path, monkeypatch):
    points = [[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]] * 512
    path = tmp_path / "sample.pcd"
    _pcd(path, points, advertised=256)
    data = bytearray(path.read_bytes())
    offset = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    for index, bits in enumerate([0x7F800001, 0x7FC00123, 0xFF800007, 0xFFC00789]):
        struct.pack_into("<I", data, offset + index * 32 + 12, bits)
    path.write_bytes(data)
    actual = livox.read_livox_binary_pcd_records(path)
    monkeypatch.setattr(livox, "_PCD_ARRAY_MIN_POINTS", 10**12)
    expected = livox.read_livox_binary_pcd_records(path)
    assert len(actual) == 256 and _record_bits(actual) == _record_bits(expected)


@pytest.mark.parametrize("count", [17, 255, 256, 4097])
def test_point_accumulator_preserves_first_extrema_and_ordered_sums(count):
    rng = np.random.default_rng(25)
    points = [tuple(row) for row in rng.normal(size=(count, 4)).tolist()]
    points[:3] = [(0.0, -0.0, 0.0, 1.0e16), (-0.0, 0.0, -0.0, 1.0),
                  (0.0, -0.0, 0.0, -1.0e16)]
    actual, scalar = livox._PointAccumulator(), livox._PointAccumulator()
    for batch in (points, [(0.0, -0.0, 0.0, 1.0)] * 256):
        actual.add_many(batch)
        for point in batch:
            scalar.add(point)
    assert actual.count == scalar.count
    fields = [field.name for field in dataclasses.fields(actual) if field.name != "count"]
    assert struct.pack("<9d", *(getattr(actual, name) for name in fields)) == struct.pack(
        "<9d", *(getattr(scalar, name) for name in fields)
    )


@pytest.mark.parametrize("intensity", [math.nan, math.inf, -math.inf])
def test_nonfinite_intensity_retains_original_accumulation(intensity):
    points = [(1.0, 2.0, 3.0, intensity)] * 256
    actual, scalar = livox._PointAccumulator(), livox._PointAccumulator()
    actual.add_many(points)
    for point in points:
        scalar.add(point)
    assert math.isnan(actual.intensity_sum) == math.isnan(scalar.intensity_sum)
    assert actual.intensity_min == scalar.intensity_min
    assert actual.intensity_max == scalar.intensity_max


def _config(tmp_path):
    config = load_config(
        "examples/public_datasets/livox_horizon_horizon_pcd_sample/online_accumulation_config.yaml"
    )
    config.dataset.path = str(tmp_path)
    return config


def _pair(tmp_path):
    points = [(i / 5, j / 5, 1.0, 0.5, 0.0, 0.0, 1.0)
              for i in range(-12, 13) for j in range(-12, 13)]
    _pcd(tmp_path / "base_horizon_100432.pcd", points)
    _pcd(tmp_path / "target_horizon_100538.pcd", points)


def test_cached_bins_records_and_new_context_follow_capture_boundary(tmp_path):
    path = tmp_path / "a.pcd"
    _pcd(path, [(value, 0.0, 0.0, 0.0) for value in (0.0, 1.0, 2.0, 3.0)], normal=False)
    context = livox.LivoxPCDEvaluationContext(tmp_path)
    config = _config(tmp_path)
    config.evaluation.solid_state.distance_bins_m = [0.0, 1.0, 2.0, 3.0]
    expected = _livox_distance_bin_counts(config, config.evaluation.solid_state)
    assert expected == [1, 1, 2]
    actual = context.distance_bin_counts(tmp_path, [0.0, 1.0, 2.0, 3.0])
    actual[0] = 100
    assert context.distance_bin_counts(tmp_path, [0.0, 1.0, 2.0, 3.0]) == expected
    assert context.distance_bin_counts(tmp_path, [0.0, 3.0]) == [4]
    records = context.read_records(path)
    records.clear()
    assert len(context.read_records(path)) == 4
    _pcd(path, [(7.0, 0.0, 0.0, 0.0)], normal=False)
    assert context.distance_bin_counts(tmp_path, [0.0, 3.0]) == [4]
    assert livox.LivoxPCDEvaluationContext(tmp_path).distance_bin_counts(
        tmp_path, [0.0, 3.0],
    ) == [0]
    with pytest.raises(ValueError, match="different dataset"):
        context.distance_bin_counts(tmp_path / "other", [0.0, 3.0])
    with pytest.raises(ValueError, match="not part"):
        context.read_records(tmp_path / "other.pcd")


def test_quality_metrics_match_legacy_bins_with_custom_edges(tmp_path):
    _pair(tmp_path)
    config = _config(tmp_path)
    config.evaluation.solid_state.distance_bins_m = [0.0, 0.5, 1.0, 2.0, 5.0]
    inspection = inspect_dataset(config.dataset)
    context = livox.LivoxPCDEvaluationContext(tmp_path)
    expected = solid_state_metrics_from_inspection(config, inspection)
    assert expected == solid_state_metrics_from_inspection(
        config, inspection, livox_context=context,
    )


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_probe_results_and_case_order_match_serial(tmp_path, workers):
    _pair(tmp_path)
    config = _config(tmp_path)
    scalar = livox_pair_evidence_from_dataset(
        str(tmp_path), SE3.identity(), config, probe_workers=1,
    )
    actual = livox_pair_evidence_from_dataset(
        str(tmp_path), SE3.identity(), config, probe_workers=workers,
    )
    assert actual == scalar and len(actual.cases) == 24


def test_shared_capture_pipeline_reads_once_and_next_run_observes_replacement(
    tmp_path, monkeypatch,
):
    import yaml

    _pair(tmp_path)
    config = _config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    reads = []
    original = livox.read_livox_binary_pcd_records

    def read(source):
        reads.append(Path(source))
        return original(source)

    monkeypatch.setattr(livox, "read_livox_binary_pcd_records", read)
    first = run_calibration(path, CalibrationRunOptions(output_dir=tmp_path / "first"))
    _pcd(tmp_path / "target_horizon_100538.pcd", [(10.0, 10.0, 10.0, 1.0)] * 300, normal=False)
    second = run_calibration(path, CalibrationRunOptions(output_dir=tmp_path / "second"))
    assert len(reads) == 4 and reads[:2] == reads[2:]
    assert first.run.provenance["raw_input_files"] != second.run.provenance["raw_input_files"]


def test_spatial_tree_import_is_deferred_until_finite_reference_requires_it():
    script = """
import sys
import numpy as np
from calibrex.solvers._exact_neighbors import _ExactNeighbors
assert 'scipy.spatial' not in sys.modules
_ExactNeighbors(np.empty((0, 3)))
_ExactNeighbors(np.asarray([[float('nan'), 0.0, 0.0]]))
assert 'scipy.spatial' not in sys.modules
index = _ExactNeighbors(np.asarray([[0.0, 0.0, 0.0]]))
assert 'scipy.spatial' in sys.modules
assert index.nearest(np.asarray([[1.0, 0.0, 0.0]]))[1].tolist() == [1.0]
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("gate", [0.0, 0.75, 1.5, math.inf, math.nan])
@pytest.mark.parametrize("budget", [0, 8 * 1024 * 1024])
def test_batched_plane_associations_preserve_plane_identity_and_ties(
    tmp_path, gate, budget, monkeypatch,
):
    _pair(tmp_path)
    records = livox.read_livox_binary_pcd_records(tmp_path / "base_horizon_100432.pcd")
    planes = livox.build_voxel_plane_map(records, 1.0)
    rng = np.random.default_rng(834)
    points = rng.uniform(-4.0, 4.0, (4097, 3))
    monkeypatch.setattr(livox, "_DENSE_PLANE_INDEX_MAX_BYTES", budget)
    actual = livox.nearest_voxel_planes(points, planes, voxel_size_m=1.0,
                                        correspondence_gate_m=gate)
    expected = [livox.nearest_voxel_plane(tuple(p), planes, voxel_size_m=1.0,
                                         correspondence_gate_m=gate) for p in points.tolist()]
    assert all(left is right for left, right in zip(actual, expected, strict=True))


def test_native_static_observation_builder_matches_per_point_identity_path(tmp_path):
    _pair(tmp_path)
    records = livox.read_livox_binary_pcd_records(tmp_path / "base_horizon_100432.pcd")
    points = [record.point[:3] for record in records]
    transform = SE3((0.1, -0.1, 0.2), (0.1, 0.2, -0.1, 0.96))
    options = {"source_records": records, "target_points": points,
               "initial_t_source_target": transform, "voxel_size_m": 1.0,
               "correspondence_gate_m": 1.5}
    actual = build_rig_point_to_plane_observations(**options)
    expected = build_rig_point_to_plane_observations(
        **options, target_t_world_source=[SE3.identity() for _ in points],
    )
    assert actual == expected
