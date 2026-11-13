"""The query-thread setting changes how long a KD-tree query takes and nothing else."""

from __future__ import annotations

import numpy as np
import pytest

from calibrex.core import query_threads as setting
from calibrex.core.query_threads import query_threads, query_workers
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions, run_scan_to_scan_odometry


def _room(rng: np.random.Generator, count: int = 6000) -> np.ndarray:
    """Points on the floor and three walls of a room, seen from inside."""

    planes = []
    for axis, value in ((2, -1.5), (0, 8.0), (1, 6.0), (1, -6.0)):
        points = rng.uniform([-8.0, -6.0, -1.5], [8.0, 6.0, 3.0], (count // 4, 3))
        points[:, axis] = value
        planes.append(points)
    return np.vstack(planes)


def _scans() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    world = _room(rng)
    scans = []
    for step in range(5):
        yaw = 0.02 * step
        cos, sin = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        moved = (world - [0.1 * step, 0.02 * step, 0.0]) @ rotation  # world -> sensor
        scans.append(moved + rng.normal(0.0, 0.005, moved.shape))
    return scans


def test_the_odometry_is_identical_on_one_and_on_several_threads() -> None:
    scans = _scans()
    options = ScanOdometryOptions(voxel_size_m=0.3, min_range_m=0.5, local_map_scans=3)

    with query_threads(1):
        single = run_scan_to_scan_odometry(scans, options)
    with query_threads(4):
        several = run_scan_to_scan_odometry(scans, options)

    assert len(single.poses) == 5 and all(r.correspondences > 200 for r in single.registrations)
    assert np.linalg.norm(single.poses[-1][:3, 3]) > 0.2  # the odometry really moved
    for a, b in zip(single.poses, several.poses, strict=True):
        assert np.array_equal(a, b)  # bit for bit


def test_the_setting_is_scoped_and_an_explicit_option_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(setting.ENV, raising=False)
    assert query_workers() == 1  # unset: one thread, as before
    with query_threads(3):
        assert query_workers() == 3
        assert query_workers(8) == 8  # an estimator option that asks for more is kept
        with query_threads(None):  # None leaves the setting
            assert query_workers() == 3
        with query_threads(0):  # 0: every available CPU
            assert query_workers() == setting.available_threads() >= 1
        assert query_workers() == 3
    assert query_workers() == 1


def test_the_environment_sets_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(setting.ENV, "2")
    assert query_workers() == 2
    monkeypatch.setenv(setting.ENV, "0")
    assert query_workers() == setting.available_threads()
    monkeypatch.setenv(setting.ENV, "many")
    with pytest.raises(ValueError, match="must be an integer"):
        query_workers()
    with query_threads(5):  # a set value does not read the environment
        assert query_workers() == 5


def test_check_estimate_and_drift_take_threads(capsys: pytest.CaptureFixture[str]) -> None:
    from calibrex.cli.main import main

    for command in ("check", "estimate", "drift"):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        assert "--threads N" in capsys.readouterr().out
