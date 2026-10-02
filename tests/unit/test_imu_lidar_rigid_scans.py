"""imu-lidar estimator on a synthetic rotating rig, with and without per-point time.

A pure-rotation sensor sits in a walled room with two pillars.  Every sweep
samples random room points; with a per-point time field each point is seen
at its own capture time, so the sweep is skewed by the rotation.  The rigid
mode (``deskew="none"``) ignores the time field.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.core.imu_lidar_rotation import ImuLidarRotationArtifact
from calibrex.data.livox_ros2 import ImuSamples, LivoxStreamProfile
from calibrex.evaluation import imu_lidar_rotation as rotation_module
from calibrex.evaluation.imu_lidar_rotation import (
    ImuLidarRunOptions,
    run_livox_imu_lidar_rotation,
)
from calibrex.evaluation.odometry_windows import WindowingOptions
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions

EPOCH = 1.76e9
SWEEP_S = 0.1
TRUE_R_LIDAR_IMU = Rotation.from_euler("xyz", [0.8, -1.5, 2.0], degrees=True).as_matrix()
TRUE_DT = 0.012
WINDOWING = WindowingOptions(
    window_duration_s=2.0,
    min_window_scans=10,
    odometry=ScanOdometryOptions(
        voxel_size_m=0.3, min_range_m=1.0, max_range_m=30.0, local_map_scans=3
    ),
)


def attitude(times: Any, speed: float) -> Any:
    local = (np.asarray(times) - EPOCH) * speed
    yaw = 0.8 * np.sin(0.3 * local) + 0.4 * np.sin(1.3 * local)
    euler = np.stack([0.25 * np.sin(0.9 * local), 0.2 * np.sin(0.7 * local + 1.0), yaw], axis=-1)
    return Rotation.from_euler("xyz", euler).as_matrix()


def room(rng: np.random.Generator, count: int) -> Any:
    half = np.array([6.0, 5.0, 2.5])
    walls = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            points = rng.uniform(-half, half, (count // 6, 3))
            points[:, axis] = sign * half[axis]
            walls.append(points)
    for cx, cy in ((2.5, 1.5), (-3.0, -2.0)):
        angle = rng.uniform(0.0, 2.0 * math.pi, count // 12)
        walls.append(
            np.c_[
                cx + 0.6 * np.cos(angle),
                cy + 0.6 * np.sin(angle),
                rng.uniform(-2.5, 2.5, count // 12),
            ]
        )
    return np.vstack(walls)


def recording(speed: float, seconds: float = 30.0) -> tuple[list[Any], ImuSamples]:
    rng = np.random.default_rng(0)
    world = room(rng, 3000)
    scans = []
    for index in range(int(seconds / SWEEP_S)):
        stamp = EPOCH + index * SWEEP_S
        points = world[rng.choice(len(world), 2000, replace=False)]
        offsets = rng.uniform(0.0, SWEEP_S, len(points))
        rotations = attitude(stamp + offsets, speed)
        scans.append((stamp, np.einsum("nji,nj->ni", rotations, points), offsets))
    step = 0.005
    times = EPOCH + np.arange(-1.0, seconds + 1.0, step)
    rotations = attitude(times, speed)
    rates = (
        Rotation.from_matrix(np.einsum("nji,njk->nik", rotations[:-1], rotations[1:])).as_rotvec()
        / step
    )
    gyro = rates @ TRUE_R_LIDAR_IMU + np.array([0.004, -0.002, 0.003])
    gyro += rng.normal(0.0, 0.003, gyro.shape)
    imu = ImuSamples(times[:-1] + 0.5 * step + TRUE_DT, gyro, np.zeros_like(gyro))
    return scans, imu


@pytest.fixture(scope="module")
def bag_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("rig") / "bag"
    path.mkdir()
    (path / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n", encoding="utf-8")
    return path


def run(
    monkeypatch: pytest.MonkeyPatch,
    bag: Path,
    speed: float,
    options: ImuLidarRunOptions,
    *,
    time_field: str | None = "t",
) -> ImuLidarRotationArtifact:
    scans, imu = recording(speed)
    seen_fields: list[str | None] = []

    def iter_scans(
        _bag: Any, stream: LivoxStreamProfile, _max: Any, _store: Any
    ) -> Iterator[tuple[float, Any, Any]]:
        seen_fields.append(stream.point_time_field)
        for stamp, xyz, offsets in scans:
            yield stamp, xyz, (None if stream.point_time_field is None else offsets)

    monkeypatch.setattr(rotation_module, "_load_imu", lambda *_: imu)
    monkeypatch.setattr(rotation_module, "_iter_scans", iter_scans)
    profile = LivoxStreamProfile("synthetic", "/p", "/i", time_field, "offset_s", "mps2")
    artifact = run_livox_imu_lidar_rotation(
        [bag],
        profile,
        options,
        dataset_family="synthetic",
        dataset_license="synthetic",
        reference_rotation=TRUE_R_LIDAR_IMU,
    )
    run.fields = seen_fields  # type: ignore[attr-defined]
    return artifact


def axis_errors(artifact: ImuLidarRotationArtifact) -> dict[str, float]:
    return {
        d.name: abs(d.error_to_reference)
        for d in artifact.dofs
        if d.name in {"roll", "pitch", "yaw"} and d.error_to_reference is not None
    }


def rigid_options() -> ImuLidarRunOptions:
    return ImuLidarRunOptions(deskew="none", windowing=WINDOWING)


@pytest.mark.parametrize("speed", [0.3, 1.0])
def test_rigid_scans_recover_the_extrinsic_and_absorb_half_a_sweep_in_the_offset(
    monkeypatch: pytest.MonkeyPatch, bag_dir: Path, speed: float
) -> None:
    artifact = run(monkeypatch, bag_dir, speed, rigid_options())

    # Rigid scans are stamped at the sweep start, so the time offset absorbs ~half a sweep.
    assert artifact.time_offset_s == pytest.approx(TRUE_DT + 0.5 * SWEEP_S, abs=0.015)
    # The rotation stays within the rigid-scan floor of `calibrex check` (1.5 deg) at both
    # rates; the bias grows with the rate only through the rate's change within a sweep.
    errors = axis_errors(artifact)
    assert set(errors) == {"roll", "pitch", "yaw"}
    assert max(errors.values()) < 0.5
    assert artifact.options["deskew"] == "none"
    # Rigid mode makes one odometry pass and no gyro passes or feedback check.
    assert [p["deskew"] for p in artifact.deskew_passes] == ["rigid_none"]
    assert artifact.deskew_feedback_ratio is None
    assert any("rigid snapshots" in text for text in artifact.limitations)
    assert any("scan header stamp" in text for text in artifact.limitations)
    # The per-point time field is ignored even though the stream profile names one.
    assert run.fields == [None]  # type: ignore[attr-defined]
    # The artifact validates and survives a save / load round trip.
    again = ImuLidarRotationArtifact.model_validate(artifact.model_dump(mode="json"))
    assert again == artifact


def test_default_gyro_mode_keeps_its_artifact_shape(
    monkeypatch: pytest.MonkeyPatch, bag_dir: Path
) -> None:
    assert ImuLidarRunOptions().deskew == "gyro"
    # Constant-velocity pass only (no gyro passes) keeps this fast; the default path is
    # otherwise unchanged: the time field is read, "deskew" is not written to the options.
    options = ImuLidarRunOptions(gyro_deskew_passes=0, windowing=WINDOWING)
    artifact = run(monkeypatch, bag_dir, 0.3, options)
    assert run.fields == ["t"]  # type: ignore[attr-defined]
    assert "deskew" not in artifact.options
    assert [p["deskew"] for p in artifact.deskew_passes] == ["lidar_constant_velocity"]
    assert not any("rigid" in text for text in artifact.limitations)
    # With a true per-point time the half-sweep stamp offset does not appear.
    assert artifact.time_offset_s == pytest.approx(TRUE_DT, abs=0.015)
