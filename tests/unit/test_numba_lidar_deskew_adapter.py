"""Exact captured replay rows, failure prefixes and native customization fallbacks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("numba")

from calibrex.core.geometry import SE3
from calibrex.data import numba_lidar_deskew_adapter as adapter
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.pipelines import online


def _fixture(count=4097, epoch=0, dtype=np.float32):
    samples = [OdometryPoseSample(epoch+i*1_000_000_000, SE3(
        (.03*i, -.02*i, .01*i), (.02*i, .03*i, -.01*i, 1.),
    )) for i in range(4)]
    xyz = np.random.default_rng(35).normal(size=(count, 3)).astype(dtype)
    offsets = np.linspace(-.02, 3.02, count)
    base = SE3((.1, -.2, .3), (.2, -.1, .3, .9))
    return samples, xyz, offsets, base


def _counters(left, right):
    for name in ("interpolation_count", "clamp_count", "max_extrapolation_s"):
        assert getattr(left, name) == getattr(right, name), name


@pytest.mark.parametrize("count", [64, 4097])
@pytest.mark.parametrize("epoch", [0, 1_650_000_000_000_000_000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("injection", [0., .0000000005, -.02])
def test_target_points_poses_timestamps_extrapolation_and_counters_match(
    count, epoch, dtype, injection,
):
    samples, xyz, offsets, base = _fixture(count, epoch, dtype)
    left, right = OdometryTrack(samples), OdometryTrack(samples)
    prepared = adapter._PreparedDeskew.capture(right)
    expected = online._deskew_target_entries(5, xyz, offsets, epoch, base, left, None, injection)
    actual = online._deskew_target_entries(
        5, xyz, offsets, epoch, base, right, None, injection, _prepared=prepared,
    )
    assert actual == expected
    _counters(left, right)
    assert prepared.rows == (count if count >= 4096 else 0)
    assert prepared.calls == (1 if count >= 4096 else 0)
    assert not prepared.context._pose_cache


@pytest.mark.parametrize("count", [64, 4097])
@pytest.mark.parametrize("limit", [None, 17, 63, 100])
@pytest.mark.parametrize("tolerance", [None, .1, 0., -.1, float("nan")])
def test_source_world_points_subsampling_and_failure_prefixes_match(count, limit, tolerance):
    epoch = 1_650_000_000_000_000_000
    samples, xyz, offsets, base = _fixture(count, epoch)
    offsets = np.linspace(0., 3.02, count)
    left, right = OdometryTrack(samples), OdometryTrack(samples)
    prepared = adapter._PreparedDeskew.capture(right)
    kwargs = {"xyz":xyz, "offsets_s":offsets, "reference_timestamp_ns":epoch,
              "t_base_source":base, "max_points":limit, "extrapolation_tolerance_s":tolerance}
    try:
        expected = online._deskew_pointcloud_to_world_records(odometry_track=left, **kwargs)
    except online.DatasetError as error:
        with pytest.raises(online.DatasetError) as actual:
            online._deskew_pointcloud_to_world_records(
                odometry_track=right, _prepared=prepared, **kwargs,
            )
        assert str(actual.value) == str(error)
    else:
        actual = online._deskew_pointcloud_to_world_records(
            odometry_track=right, _prepared=prepared, **kwargs,
        )
        np.testing.assert_array_equal(np.asarray([r.point for r in actual]).view(np.uint64),
                                      np.asarray([r.point for r in expected]).view(np.uint64))
    _counters(left, right)


@pytest.mark.parametrize("hook", ["timestamp", "compose", "record", "sample_replacement"])
def test_native_customization_and_stale_snapshots_keep_original_order(hook, monkeypatch):
    samples, xyz, offsets, base = _fixture(4097)
    left, right = OdometryTrack(samples), OdometryTrack(samples)
    prepared = adapter._PreparedDeskew.capture(right)
    visited = []
    if hook == "timestamp":
        native = online._target_capture_ns
        def observed(timestamp, **kwargs):
            visited.append(timestamp)
            return native(timestamp, **kwargs)
        monkeypatch.setattr(online, "_target_capture_ns", observed)
    elif hook == "compose":
        native = SE3.compose
        def observed(self, other):
            visited.append(self)
            return native(self, other)
        monkeypatch.setattr(SE3, "compose", observed)
    elif hook == "record":
        for track in (left, right):
            def observe(track):
                native = track._record_extrapolation
                def recorded(value):
                    visited.append(value)
                    native(value)
                return recorded
            track._record_extrapolation = observe(track)
    else:
        for track in (left, right):
            track.samples = tuple(replace(s, pose=SE3((.1, .2, .3), (0., 0., 0., 1.)))
                                  for s in samples)
            track._timestamps_ns = tuple(s.timestamp_ns for s in track.samples)
    expected = online._deskew_target_entries(5, xyz, offsets, 0, base, left, None)
    first = len(visited)
    actual = online._deskew_target_entries(
        5, xyz, offsets, 0, base, right, None, _prepared=prepared,
    )
    assert actual == expected
    _counters(left, right)
    assert prepared.calls == 0
    if hook != "sample_replacement":
        assert first == len(visited)-first == len(xyz)


def test_nan_offsets_preserve_the_scalar_exception_prefix_and_budget_fallback(monkeypatch):
    samples, xyz, offsets, base = _fixture(4097)
    left, right = OdometryTrack(samples), OdometryTrack(samples)
    prepared = adapter._PreparedDeskew.capture(right)
    offsets[4] = np.nan
    for track, extra in ((left, {}), (right, {"_prepared":prepared})):
        with pytest.raises(ValueError):
            online._deskew_target_entries(5, xyz, offsets, 0, base, track, None, **extra)
    _counters(left, right)
    assert left.interpolation_count == 4 and prepared.calls == 0
    monkeypatch.setattr(adapter, "_MAX_DESKEW_ROW_BYTES", 1)
    assert prepared.batch(xyz, np.zeros(64), 0, range(64), right, base) is None


def test_factory_is_opt_in_and_identity_records_cpu_execution():
    from calibrex.core.config import load_config

    config = load_config(
        "examples/public_datasets/glim_versatile/online_continuous_time_compiled_queries_config.yaml"
    )
    samples, _xyz, _offsets, _base = _fixture(64)
    track = OdometryTrack(samples)
    assert online._compiled_deskew_context(config, track, False) is None
    assert online._compiled_deskew_context(config, track, True) is None
    config = load_config(
        "examples/public_datasets/glim_versatile/online_continuous_time_compiled_dense_capture_config.yaml"
    )
    prepared = online._compiled_deskew_context(config, track, True)
    assert prepared is not None
    identity = prepared.identity()
    assert identity.startswith("calibrex.numba_lidar_deskew/v0.1;")
    assert ";fastmath=false;cache=true;point_pose_cpu=true;source_sha256=" in identity
    assert len(identity.rsplit("=", 1)[1]) == 64
    config.pipeline.factors["lidar_rig_point_to_plane"].options.pop("continuous_time_query_backend")
    assert online._compiled_deskew_context(config, track, True) is None
