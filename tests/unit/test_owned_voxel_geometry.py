"""Owned normal-free geometry retains scalar plane and query float bytes."""

import numpy as np
import pytest

from calibrex.data import livox
from calibrex.data._voxel_geometry import _point_plane_geometry


def _reference(points, voxel):
    records = [livox.LivoxPointRecord((*row, 0.0)) for row in map(tuple, points.tolist())]
    return livox.VoxelPlaneQueryContext(livox.build_voxel_plane_map(records, voxel))


def _bits(context):
    return [(key, np.asarray((*plane.centroid, *plane.normal), dtype="<f8").tobytes())
            for key, plane in context._planes.items()]


@pytest.mark.parametrize("count", [64, 1000, 8193, 65536])
@pytest.mark.parametrize("voxel", [0.1, 1.0, 100.0])
def test_complete_snapshot_and_queries_keep_scalar_order_and_bytes(count, voxel):
    rng = np.random.default_rng(731)
    points = rng.uniform(-2.0, 2.0, (count, 3))
    reference = _reference(points, voxel)
    actual = livox.VoxelPlaneQueryContext._from_points(points, voxel)
    assert actual is not None
    assert _bits(actual) == _bits(reference)
    queries = rng.uniform(-2.1, 2.1, (129, 3))
    options = {"voxel_size_m": voxel, "correspondence_gate_m": 3.0}
    def matches(context):
        return [None if m is None else (m.voxel_key, np.asarray(
            (*m.centroid, *m.normal, m.centroid_distance_m), dtype="<f8",
        ).tobytes()) for m in context.nearest_matches(queries, **options)]
    assert matches(actual) == matches(reference)
    # Snapshot ownership does not depend on the caller's later array edits.
    before = _bits(actual)
    points[:] = 0.0
    assert _bits(actual) == before


@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e20])
def test_first_ties_collinear_dropped_and_twelve_point_prefix(scale):
    prefix = np.asarray([
        (-0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0),
    ]) * scale
    points = np.vstack((np.tile(prefix, (3, 1)), np.tile((10.0, 0.0, 0.0), (48, 1))))
    voxel = float(scale * 100)
    actual = livox.VoxelPlaneQueryContext._from_points(points, voxel)
    assert actual is not None
    assert _bits(actual) == _bits(_reference(points, voxel))


@pytest.mark.parametrize("change", [
    "float32", "object", "subclass", "small", "large", "shape", "nan", "inf",
    "huge", "key", "zero_voxel", "negative_voxel", "nan_voxel", "custom_voxel",
])
def test_unsupported_arrays_and_voxels_use_the_original_constructor(change):
    points, voxel = np.zeros((64, 3)), 1.0
    if change == "float32":
        points = points.astype(np.float32)
    elif change == "object":
        points = points.astype(object)
    elif change == "subclass":
        points = points.view(type("Array", (np.ndarray,), {}))
    elif change in ("small", "large"):
        points = np.zeros((63 if change == "small" else 65537, 3))
    elif change == "shape":
        points = np.zeros((64, 4))
    elif change in ("nan", "inf", "huge", "key"):
        points[0, 0] = {"nan": float("nan"), "inf": float("inf"),
                        "huge": 1e31, "key": 1e20}[change]
    elif change == "custom_voxel":
        voxel = np.float64(1.0)
    else:
        voxel = {"zero_voxel": 0.0, "negative_voxel": -1.0,
                 "nan_voxel": float("nan")}[change]
    assert _point_plane_geometry(points, voxel) is None


@pytest.mark.parametrize("attribute", ["add", "centroid", "normal", "fallback"])
def test_replaced_scalar_plane_operations_disable_native_capture(attribute, monkeypatch):
    if attribute == "fallback":
        monkeypatch.setattr(livox, "_fallback_plane_normal", lambda points: None)
    else:
        monkeypatch.setattr(livox._VoxelPlane, attribute, lambda *args: None)
    assert livox.VoxelPlaneQueryContext._from_points(np.zeros((64, 3)), 1.0) is None
