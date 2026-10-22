"""Independent coordinate/unit controls; reference calibration never enters poses."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from calibrex.data.boreas_ins_lidar import (
    boreas_scan_time_s,
    read_boreas_ins_trajectory,
    read_boreas_lidar_xyz,
    read_boreas_reference_extrinsic,
)

COLUMNS = ["GPSTime", "easting", "northing", "altitude", "roll", "pitch", "heading"]


def csv_file(tmp_path, rows, columns=COLUMNS):
    path = tmp_path / "gps_post_process.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_seconds_units_position_and_passive_heading_without_calibration(tmp_path):
    epoch = 1606417097.5
    path = csv_file(
        tmp_path, [[epoch, 1, 2, 3, 0, 0, math.pi / 2], [epoch + 0.01, 4, 5, 6, 0, 0, math.pi / 2]]
    )
    trajectory = read_boreas_ins_trajectory(path)
    np.testing.assert_array_equal(trajectory.times_s, [epoch, epoch + 0.01])
    np.testing.assert_array_equal(trajectory.poses[:, :3, 3], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(trajectory.poses[0, :3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-15)
    assert list(tmp_path.iterdir()) == [path]  # No calibration or LiDAR pose file needed.
    assert trajectory.covers(epoch + 0.005)


def test_noncommuting_passive_rotation_order(tmp_path):
    path = csv_file(
        tmp_path,
        [
            [1, 0, 0, 0, math.pi / 2, math.pi / 2, math.pi / 2],
            [2, 0, 0, 0, math.pi / 2, math.pi / 2, math.pi / 2],
        ],
    )
    rotation = read_boreas_ins_trajectory(path).poses[0, :3, :3]
    rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    ry = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    rz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(rotation, rx @ ry @ rz, atol=1e-15)


def test_bounds_preserve_gap_guards(tmp_path):
    path = csv_file(tmp_path, [[i, 0, 0, 0, 0, 0, 0] for i in (1, 2, 4, 5)])
    trajectory = read_boreas_ins_trajectory(path, start_s=2, end_s=4, max_gap_s=0.5)
    np.testing.assert_array_equal(trajectory.times_s, [2, 4])
    assert not trajectory.covers(3)


@pytest.mark.parametrize("times", [[1, 1], [2, 1]])
def test_duplicate_and_unsorted_timestamps_are_not_silently_reordered(tmp_path, times):
    path = csv_file(tmp_path, [[t, 0, 0, 0, 0, 0, 0] for t in times])
    with pytest.raises(ValueError, match="strictly increasing"):
        read_boreas_ins_trajectory(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "invalid", ""])
def test_bad_reference_values_are_rejected(tmp_path, value):
    path = csv_file(tmp_path, [[1, 0, 0, 0, 0, 0, 0], [2, value, 0, 0, 0, 0, 0]])
    with pytest.raises(ValueError, match="Applanix row"):
        read_boreas_ins_trajectory(path)


def test_sensor_pose_csv_is_not_a_reference_trajectory(tmp_path):
    path = csv_file(tmp_path, [[1, 0, 0, 0, 0, 0, 0]], columns=["t", "x", "y", "z", "r", "p", "y"])
    with pytest.raises(ValueError, match="not sensor poses"):
        read_boreas_ins_trajectory(path)


@pytest.mark.parametrize(
    "kwargs", [{"start_s": 2, "end_s": 1}, {"start_s": float("nan")}, {"max_gap_s": 0}]
)
def test_invalid_bounds_and_gap_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        read_boreas_ins_trajectory(tmp_path / "not-read", **kwargs)


def test_binary_layout_and_microsecond_filename(tmp_path):
    path = tmp_path / "1606417097502930.bin"
    rows = np.array([[1, 2, 3, 100, 7, 0.02], [-1, -2, -3, 50, 3, 0.04]], dtype="<f4")
    path.write_bytes(rows.tobytes())
    np.testing.assert_array_equal(read_boreas_lidar_xyz(path), rows[:, :3])
    assert read_boreas_lidar_xyz(path).dtype == np.float64
    assert abs(boreas_scan_time_s(path) - 1606417097.502930) < 3e-7
    path.write_bytes(b"not-six-floats")
    with pytest.raises(ValueError, match="six float32"):
        read_boreas_lidar_xyz(path)
    with pytest.raises(ValueError, match="16-digit"):
        boreas_scan_time_s(tmp_path / "1606417097.bin")


def test_reference_extrinsic_direction_and_validation(tmp_path):
    path = tmp_path / "T_applanix_lidar.txt"
    expected = np.eye(4)
    expected[:3, 3] = [1, 2, 3]
    np.savetxt(path, expected)
    np.testing.assert_array_equal(read_boreas_reference_extrinsic(path), expected)
    expected[0, 0] = -1
    np.savetxt(path, expected)
    with pytest.raises(ValueError, match=r"SE\(3\)"):
        read_boreas_reference_extrinsic(path)
