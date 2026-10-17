"""Initial native transforms, plane geometry, row order and callback parity."""

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
from tests.unit.test_tum_diagnostic_batches import _raw_points

from calibrex.core import geometry
from calibrex.core.geometry import SE3
from calibrex.data import livox
from calibrex.data.tum_rgbd import TUMDepthPoseEntry, TUMImageEntry, TUMTrajectoryEntry
from calibrex.solvers import _tum_initial_geometry as initial
from calibrex.solvers import native_tum_joint_slac_solver as adapter


def _frame(index=0, angle=0.07):
    return TUMDepthPoseEntry(
        TUMImageEntry(float(index), f"depth/{index}.png"),
        TUMTrajectoryEntry(float(index), (0.01 * index, -0.03, 0.02), (angle, 0.02, -0.01, 1.0)),
        0.0,
    )


def _points(count):
    raw = _raw_points()
    return [raw[index % len(raw)] for index in range(count)]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("count", [0, 1, 63, 64, 4097])
@pytest.mark.parametrize("angle", [0.0, 1e-12, 0.3, math.pi - 1e-9])
def test_every_transformed_map_point_retains_scalar_float_bits(count, angle):
    frame, points = _frame(angle=angle), _points(count)
    expected = [frame.transform_world_camera.transform_point(point) for point in points]
    records = initial._frame_records(frame, points)
    actual = [record.point[:3] for record in records]
    assert np.asarray(actual, dtype="<f8").tobytes() == np.asarray(expected, dtype="<f8").tobytes()
    assert len(records) == count
    assert all(record.point[3] == 0.0 and record.normal_xyz is None for record in records)


def test_transform_scratch_is_chunked_without_removing_points(monkeypatch):
    frame, points = _frame(), _points(4097)
    expected = initial._frame_records(frame, points)
    calls = []
    original = initial._transform_points

    def transformed(points, transform):
        calls.append(len(points))
        return original(points, transform)

    monkeypatch.setattr(initial, "_CHUNK_ROWS", 17)
    monkeypatch.setattr(initial, "_transform_points", transformed)
    assert initial._frame_records(frame, points) == expected
    assert sum(calls) == len(points) and max(calls) <= 17


@pytest.mark.parametrize("replacement", ["frame", "point", "init", "rotate", "normalize"])
def test_replaced_geometry_keeps_original_per_point_callback_count(replacement, monkeypatch):
    frame, points = _frame(), _points(80)
    calls = []
    if replacement == "frame":
        original = TUMDepthPoseEntry.transform_world_camera.fget

        def pose(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(TUMDepthPoseEntry, "transform_world_camera", property(pose))
    else:
        owner, name = {
            "point": (SE3, "transform_point"),
            "init": (SE3, "__post_init__"),
            "rotate": (geometry, "rotate_vector_xyzw"),
            "normalize": (geometry, "normalize_quaternion_xyzw"),
        }[replacement]
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)
    expected = [
        livox.LivoxPointRecord((*frame.transform_world_camera.transform_point(point), 0.0))
        for point in points
    ]
    expected_calls = len(calls)
    calls.clear()
    assert initial._frame_records(frame, points) == expected
    assert len(calls) == expected_calls


def test_custom_numeric_inputs_keep_scalar_conversion_calls():
    calls = []

    class Number(float):
        def __float__(self):
            calls.append(1)
            return super().__float__()

    frame = _frame()
    points = [(Number(x), y, z) for x, y, z in _points(80)]
    expected = [
        livox.LivoxPointRecord((*frame.transform_world_camera.transform_point(point), 0.0))
        for point in points
    ]
    expected_calls = len(calls)
    calls.clear()
    assert initial._frame_records(frame, points) == expected
    assert len(calls) == expected_calls == len(points)


@pytest.mark.parametrize("count", [64, 145, 4099])
def test_captured_world_and_pair_matches_keep_all_scalar_geometry_and_ids(count):
    frame, points = _frame(), _points(count)
    planes = livox.build_voxel_plane_map(initial._frame_records(frame, points), 0.1)
    query = initial._query_snapshot(planes)
    assert query is not None
    world = [frame.transform_world_camera.transform_point(point) for point in points]
    expected = [
        livox.nearest_voxel_plane_match(point, planes, voxel_size_m=0.1, correspondence_gate_m=0.15)
        for point in world
    ]
    captured = initial._frame_matches(
        frame, points, query, voxel_size_m=0.1, correspondence_gate_m=0.15
    )
    assert captured is not None
    assert _canonical([(p.centroid, p.normal) if p else None for p in captured]) == _canonical(
        [(p.centroid, p.normal) if p else None for p in expected]
    )
    paired = initial._pair_matches(
        tuple(points),
        frame.transform_world_camera,
        query,
        voxel_size_m=0.1,
        correspondence_gate_m=0.15,
    )
    assert _canonical([asdict(p) if p else None for p in paired]) == _canonical(
        [asdict(p) if p else None for p in expected]
    )


@pytest.mark.parametrize("mutation", ["subclass", "property"])
def test_custom_plane_geometry_does_not_capture_properties(mutation, monkeypatch):
    planes = livox.build_voxel_plane_map(initial._frame_records(_frame(), _points(145)), 0.1)
    assert planes
    calls = []
    if mutation == "subclass":

        class CustomPlane(livox._VoxelPlane):
            pass

        key = next(iter(planes))
        planes[key] = CustomPlane(**vars(planes[key]))
    else:
        original = livox._VoxelPlane.normal.fget

        def normal(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(livox._VoxelPlane, "normal", property(normal))
    assert initial._query_snapshot(planes) is None
    assert not calls


def test_sparse_map_without_packed_neighborhoods_keeps_scalar_geometry():
    frame = _frame()
    points = _points(145)
    records = initial._frame_records(frame, points)
    records += [
        livox.LivoxPointRecord((r.point[0] + 1e6, *r.point[1:]), r.normal_xyz)
        for r in records.copy()
    ]
    planes = livox.build_voxel_plane_map(records, 0.1)
    query = initial._query_snapshot(planes)
    assert query is not None and query._index.neighborhoods is None
    expected = [
        livox.nearest_voxel_plane_match(
            frame.transform_world_camera.transform_point(point),
            planes,
            voxel_size_m=0.1,
            correspondence_gate_m=0.15,
        )
        for point in points
    ]
    actual = initial._frame_matches(
        frame, points, query, voxel_size_m=0.1, correspondence_gate_m=0.15
    )
    assert _canonical([(p.centroid, p.normal) if p else None for p in actual]) == _canonical(
        [(p.centroid, p.normal) if p else None for p in expected]
    )


def _fixture(monkeypatch):
    frames = tuple(_frame(index, angle=0.01 * index) for index in range(6))
    points = _points(145)
    monkeypatch.setattr(adapter, "read_tum_depth_png", lambda path: object())
    monkeypatch.setattr(adapter, "sample_tum_depth_points", lambda *a, **k: points.copy())
    options = replace(adapter.TUMJointSlacOptions(), map_frame_count=2, query_frame_count=4)
    return frames, options


@pytest.mark.parametrize("family", ["world", "pair"])
def test_complete_initial_problem_measurements_lineage_and_factors_are_exact(
    family, monkeypatch, tmp_path
):
    frames, options = _fixture(monkeypatch)
    intrinsics = adapter.TUMDepthIntrinsics()
    build = adapter._build_problem if family == "world" else adapter._build_xyz_pair_problem
    fast = build(tmp_path, frames, intrinsics, options)
    monkeypatch.setattr(initial, "_MIN_POINTS", 1_000_000)
    scalar = build(tmp_path, frames, intrinsics, options)
    assert _canonical([asdict(m) for m in fast.measurements]) == _canonical(
        [asdict(m) for m in scalar.measurements]
    )
    assert fast.parameter_blocks == scalar.parameter_blocks
    if family == "pair":
        assert _canonical([asdict(a) for a in fast.associations]) == _canonical(
            [asdict(a) for a in scalar.associations]
        )
        assert fast.pair_counts == scalar.pair_counts
    values = {block.name: block.initial_values for block in fast.parameter_blocks}
    for before, after in zip(scalar.data_factors, fast.data_factors, strict=True):
        assert (
            before.factor_id,
            before.variable_names,
            before.weight,
            before.sqrt_information,
        ) == (
            after.factor_id,
            after.variable_names,
            after.weight,
            after.sqrt_information,
        )
        assert (
            np.asarray(before.residuals(values)).tobytes()
            == np.asarray(after.residuals(values)).tobytes()
        )


@pytest.mark.parametrize("family", ["world", "pair"])
def test_initial_custom_matcher_is_still_called_for_every_query(family, monkeypatch, tmp_path):
    frames, options = _fixture(monkeypatch)
    name = "nearest_voxel_plane" if family == "world" else "nearest_voxel_plane_match"
    original, calls = getattr(adapter, name), []

    def matched(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, name, matched)
    build = adapter._build_problem if family == "world" else adapter._build_xyz_pair_problem
    build(tmp_path, frames, adapter.TUMDepthIntrinsics(), options)
    groups = len(frames) - (
        options.map_frame_count if family == "world" else options.xyz_pair_stride
    )
    assert len(calls) == groups * 145
