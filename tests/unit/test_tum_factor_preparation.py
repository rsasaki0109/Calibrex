"""Native construction reuse preserves values, callbacks and scope lifetimes."""

import math
import struct
from dataclasses import asdict, replace

import numpy as np
import pytest
from tests.unit.test_rgbd_transform_reuse import KINDS, factory, inputs, measurement
from tests.unit.test_tum_initial_geometry import _canonical, _fixture

from calibrex.core import geometry
from calibrex.core.geometry import SE3
from calibrex.data.tum_rgbd import TUMDepthPoseEntry, TUMImageEntry, TUMTrajectoryEntry
from calibrex.graph import joint_factors as factors
from calibrex.graph._evaluation_scope import (
    _RGBD_FACTORY_CACHE,
    _RGBD_FACTORY_CACHE_LIMIT,
    _with_rgbd_factory_cache,
)
from calibrex.solvers import _tum_initial_geometry as initial
from calibrex.solvers import native_tum_joint_slac_solver as adapter


def test_native_measurements_share_only_their_frame_and_retain_every_value(monkeypatch, tmp_path):
    frames, options = _fixture(monkeypatch)
    fast = adapter._build_problem(tmp_path, frames, adapter.TUMDepthIntrinsics(), options)
    monkeypatch.setattr(adapter, "_frame_initial_transforms", lambda frame: None)
    scalar = adapter._build_problem(tmp_path, frames, adapter.TUMDepthIntrinsics(), options)
    assert fast.measurements
    assert _canonical([asdict(m) for m in fast.measurements]) == _canonical(
        [asdict(m) for m in scalar.measurements]
    )
    assert fast.parameter_blocks == scalar.parameter_blocks
    for group in {m.observation_group for m in fast.measurements}:
        rows = [m for m in fast.measurements if m.observation_group == group]
        assert len(rows) > 1
        assert len({id(m.transform_world_body_initial) for m in rows}) == 1
        assert len({id(m.transform_body_sensor_initial) for m in rows}) == 1
    assert len({id(m.transform_world_body_initial) for m in fast.measurements}) == 4
    assert len({id(m.transform_world_body_initial) for m in scalar.measurements}) == len(
        scalar.measurements
    )
    values = {
        b.name: tuple(0.0001 * (i + 1) for i in range(b.dimension)) for b in fast.parameter_blocks
    }
    for before, after in zip(scalar.data_factors, fast.data_factors, strict=True):
        assert (before.factor_id, before.variable_names) == (after.factor_id, after.variable_names)
        assert (
            np.asarray(before.residuals(values)).tobytes()
            == np.asarray(after.residuals(values)).tobytes()
        )
    assert _RGBD_FACTORY_CACHE.get() is None


@pytest.mark.parametrize(
    "replacement",
    [
        "frame",
        "constructor",
        "init",
        "identity",
        "tuple3",
        "tuple4",
        "normalize",
        "rotate",
    ],
)
def test_custom_geometry_retains_all_original_construction_calls(
    replacement, monkeypatch, tmp_path
):
    frames, options = _fixture(monkeypatch)
    calls = []
    if replacement == "frame":
        original = TUMDepthPoseEntry.transform_world_camera.fget

        def wrapped(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(TUMDepthPoseEntry, "transform_world_camera", property(wrapped))
    elif replacement == "identity":
        original = SE3.identity.__func__

        def wrapped(cls):
            calls.append(1)
            return original(cls)

        monkeypatch.setattr(SE3, "identity", classmethod(wrapped))
    else:
        owner, name = {
            "constructor": (SE3, "__init__"),
            "init": (SE3, "__post_init__"),
            "tuple3": (geometry, "_tuple3"),
            "tuple4": (geometry, "_tuple4"),
            "normalize": (geometry, "normalize_quaternion_xyzw"),
            "rotate": (geometry, "rotate_vector_xyzw"),
        }[replacement]
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)
    actual = adapter._build_problem(tmp_path, frames, adapter.TUMDepthIntrinsics(), options)
    actual_calls = len(calls)
    assert actual_calls > 0
    calls.clear()
    monkeypatch.setattr(adapter, "_frame_initial_transforms", lambda frame: None)
    expected = adapter._build_problem(tmp_path, frames, adapter.TUMDepthIntrinsics(), options)
    assert len(calls) == actual_calls
    assert _canonical([asdict(m) for m in actual.measurements]) == _canonical(
        [asdict(m) for m in expected.measurements]
    )


@pytest.mark.parametrize("kind", KINDS)
def test_scoped_factory_signatures_retain_all_families_residual_bits(kind):
    world = SE3((-0.0, 0.2, -0.3), (0.03, -0.02, 0.01, 0.99))
    sensor = SE3((0.4, -0.0, 0.2), (-0.02, 0.01, 0.04, 0.98))
    changes = (
        {"transform_world_source_initial": world, "transform_world_target_initial": sensor}
        if kind == "pair"
        else {"transform_world_body_initial": world, "transform_body_sensor_initial": sensor}
    )
    rows = [replace(measurement(kind, i), **changes) for i in range(12)]
    expected = [factory(kind, row) for row in rows]

    @_with_rgbd_factory_cache
    def build():
        result = [factory(kind, row) for row in rows]
        assert len(_RGBD_FACTORY_CACHE.get()) == (2 if kind == "pair" else 1)
        return result

    actual = build()
    assert _RGBD_FACTORY_CACHE.get() is None
    for before, after in zip(expected, actual, strict=True):
        assert (before.factor_id, before.variable_names) == (after.factor_id, after.variable_names)
        assert (
            np.asarray(before.residuals(inputs())).tobytes()
            == np.asarray(after.residuals(inputs())).tobytes()
        )


def test_cached_signature_revalidates_replaced_leaves_and_preserves_signed_zero():
    world, sensor = SE3.identity(), SE3.identity()

    @_with_rgbd_factory_cache
    def build():
        positive = factors._rgbd_transform_signature(world, sensor)
        assert positive is not None
        assert factors._rgbd_transform_signature(world, sensor) is positive
        object.__setattr__(world, "translation_m", (-0.0, 0.0, 0.0))
        negative = factors._rgbd_transform_signature(world, sensor)
        assert negative is not None
        assert positive.key != negative.key
        assert struct.pack("<3d", *negative.leaves[0]) == struct.pack("<3d", -0.0, 0.0, 0.0)
        object.__setattr__(sensor, "rotation_quat_xyzw", (math.nan, 0.0, 0.0, 1.0))
        assert factors._rgbd_transform_signature(world, sensor) is None
        assert not negative.matches(world, sensor)

    build()


def test_factory_cache_is_bounded_isolated_and_cleared_on_error():
    held = []

    @_with_rgbd_factory_cache
    def nested():
        held.append(_RGBD_FACTORY_CACHE.get())
        factors._rgbd_transform_signature(SE3.identity())
        raise RuntimeError("builder failed")

    @_with_rgbd_factory_cache
    def build():
        cache = _RGBD_FACTORY_CACHE.get()
        held.append(cache)
        worlds = [
            SE3((float(i), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            for i in range(_RGBD_FACTORY_CACHE_LIMIT + 3)
        ]
        for world in worlds:
            factors._rgbd_transform_signature(world)
            assert len(cache) <= _RGBD_FACTORY_CACHE_LIMIT
        assert (id(worlds[0]), id(None)) not in cache
        with pytest.raises(RuntimeError, match="builder failed"):
            nested()
        assert _RGBD_FACTORY_CACHE.get() is cache
        assert held[-1] == {} and held[-1] is not cache

    build()
    assert held[0] == {} and _RGBD_FACTORY_CACHE.get() is None


def test_non_native_frame_values_keep_scalar_conversions():
    class Number(float):
        pass

    frame = adapter.TUMDepthPoseEntry(
        TUMImageEntry(0.0, "depth.png"),
        TUMTrajectoryEntry(0.0, (Number(0.0), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        0.0,
    )
    assert initial._frame_initial_transforms(frame) is None
