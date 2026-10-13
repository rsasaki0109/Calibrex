"""Captured immutable fields retain constructor values and custom behavior."""

import dataclasses
import math
import struct

import pytest

from calibrex.core.geometry import SE3
from calibrex.graph import lidar_point_to_plane as module
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    _PreparedObservationBuilder,
)


def _bits(observation):
    values = (
        *observation.point_lidar_m,
        *observation.plane_point_world_m,
        *observation.plane_normal_world,
        observation.weight,
    )
    return b"".join(struct.pack("!d", value) for value in values)


def test_prepared_constructor_retains_values_signed_zeros_and_frozen_fields():
    points = [(float(i), -0.0, 0.0) for i in range(65)]
    builder = _PreparedObservationBuilder(points)
    pose = SE3((0.2, -0.3, 0.1), (0.1, 0.2, -0.3, 0.8))
    centroid = (0.3, -0.4, 0.5)
    normals = [(sign, 0.9, -0.3) for sign in (-0.0, 0.0)]
    for point in points:
        for normal in normals:
            actual = builder.build(point, centroid, normal, pose)
            expected = LidarPointToPlaneObservation(point, centroid, normal, pose)
            assert dataclasses.asdict(actual) == dataclasses.asdict(expected)
            assert _bits(actual) == _bits(expected)
            assert actual.point_lidar_m is not point
            assert actual.t_world_ego is pose
    assert len(builder._planes) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        actual.weight = 2.0


def test_mutable_and_non_native_fields_use_constructor_and_observe_changes():
    point = (0.1, 0.2, 0.3)
    builder = _PreparedObservationBuilder([point])
    centroid, normal = [0.1, 0.2, 0.3], [0.1, 0.2, 0.9]
    for offset in (0.0, 0.5):
        centroid[0] += offset
        normal[1] -= offset
        assert _bits(builder.build(point, centroid, normal, SE3.identity())) == _bits(
            LidarPointToPlaneObservation(point, centroid, normal, SE3.identity())
        )
    assert not builder._planes
    changed_point = (0.1, 0.2, 0.4)
    assert _bits(builder.build(changed_point, centroid, normal, SE3.identity())) == _bits(
        LidarPointToPlaneObservation(changed_point, centroid, normal, SE3.identity())
    )


def test_retained_points_and_planes_are_bounded():
    points = [(float(i), 0.0, 0.0) for i in range(65537)]
    builder = _PreparedObservationBuilder(points)
    assert len(builder._points) == 65536
    assert id(points[-1]) not in builder._points
    normal = (0.1, -0.2, 0.9)
    centroids = [(float(i), 0.1, 0.2) for i in range(1025)]
    for centroid in centroids:
        actual = builder.build(points[0], centroid, normal, SE3.identity())
        assert _bits(actual) == _bits(
            LidarPointToPlaneObservation(points[0], centroid, normal, SE3.identity())
        )
    assert len(builder._planes) == 1024
    assert (id(centroids[0]), id(normal)) not in builder._planes


@pytest.mark.parametrize("method", ["__init__", "__post_init__"])
def test_custom_constructor_or_post_init_is_not_bypassed(monkeypatch, method):
    point, normal = (0.1, 0.2, 0.3), (0.2, 0.1, 0.9)
    builder = _PreparedObservationBuilder([point])
    original = getattr(LidarPointToPlaneObservation, method)
    calls = []

    def observed(self, *args, **kwargs):
        calls.append(method)
        original(self, *args, **kwargs)

    monkeypatch.setattr(LidarPointToPlaneObservation, method, observed)
    builder.build(point, point, normal, SE3.identity())
    assert calls == [method]
    assert not builder._planes


def test_nonfinite_fields_and_degenerate_normals_keep_constructor_semantics():
    point = (0.1, 0.2, 0.3)
    builder = _PreparedObservationBuilder([point])
    for value in (math.nan, math.inf, -math.inf):
        normal = (value, 0.2, 0.1)
        assert _bits(builder.build(point, point, normal, SE3.identity())) == _bits(
            LidarPointToPlaneObservation(point, point, normal, SE3.identity())
        )
    assert not builder._planes
    with pytest.raises(ValueError, match="non-zero"):
        builder.build(point, point, (0.0, 0.0, 0.0), SE3.identity())


def test_replaced_graph_constructor_alias_keeps_the_callers_native_class(monkeypatch):
    point = (0.1, 0.2, 0.3)
    builder = _PreparedObservationBuilder([point])

    class Replacement(LidarPointToPlaneObservation):
        pass

    monkeypatch.setattr(module, "LidarPointToPlaneObservation", Replacement)
    assert type(builder.build(point, point, (0.1, 0.2, 0.9), SE3.identity())) is (
        LidarPointToPlaneObservation
    )
