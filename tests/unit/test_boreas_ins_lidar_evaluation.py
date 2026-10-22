"""The Boreas adapter preserves LiDAR-only registration and failure provenance."""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest

from calibrex.core.ins_lidar_hand_eye import InsLidarHandEyeArtifact
from calibrex.evaluation import boreas_ins_lidar as adapter
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryResult, ScanRegistration


def sequence(tmp_path):
    root = tmp_path / "boreas-synthetic-input"
    for name in ("lidar", "calib", "applanix"):
        (root / name).mkdir(parents=True)
    epoch = 1606417097.0
    for offset in range(3):
        points = np.array([[offset, 2, 3, 4, 5, 0.01]], dtype="<f4")
        (root / "lidar" / f"{int((epoch + offset) * 1e6)}.bin").write_bytes(points.tobytes())
    with (root / "applanix/gps_post_process.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["GPSTime", "easting", "northing", "altitude", "roll", "pitch", "heading"])
        writer.writerows([[epoch + offset, 0, 0, 0, 0, 0, 0] for offset in range(-1, 4)])
    np.savetxt(root / "calib/T_applanix_lidar.txt", np.eye(4))
    return root


def test_full_adapter_keeps_reference_out_of_registration_and_retains_failed_solve(
    tmp_path, monkeypatch
):
    root = sequence(tmp_path)
    calls = []
    registration = ScanRegistration(np.eye(4), True, 1, 300, 0.01, 1.0)

    def odometry(loader, count, options, *, times_s):
        assert len(times_s) == count == 3
        assert options.deskew_iterations == 1
        calls.append([loader(i).copy() for i in range(count)])
        return ScanOdometryResult(
            tuple(np.eye(4) for _ in range(count)),
            (registration, replace(registration, converged=False)),
        )

    monkeypatch.setattr(adapter, "odometry_from_loader", odometry)
    result = adapter.run_boreas_ins_lidar_hand_eye(root, command=["synthetic adapter check"])
    assert result.solver_status == "insufficient_motions" and result.policy_status == "fail"
    assert result.odometry.frame_count == 3 and result.odometry.converged_registrations == 1
    assert result.odometry.registration_count == 2
    assert result.provenance.dataset_family == "boreas"
    assert result.provenance.command == ["synthetic adapter check"]
    assert result.provenance.drive_ids == [root.name]
    np.testing.assert_array_equal(calls[0][2], [[2, 2, 3]])
    InsLidarHandEyeArtifact.model_validate(result.model_dump(mode="json"))
    before = result.provenance.input_sha256
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    np.savetxt(root / "calib/T_applanix_lidar.txt", transform)
    again = adapter.run_boreas_ins_lidar_hand_eye(root)
    assert again.provenance.input_sha256 != before
    assert again.solver_status == result.solver_status
    for left, right in zip(calls[0], calls[1], strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("max_frames", [0, 1, -1, True])
def test_invalid_frame_budget_cannot_silently_change_the_experiment(tmp_path, max_frames):
    with pytest.raises(ValueError, match="max_frames"):
        adapter.run_boreas_ins_lidar_hand_eye(tmp_path, max_frames=max_frames)


def test_ins_reference_must_cover_scan_times_before_registration(tmp_path, monkeypatch):
    root = sequence(tmp_path)
    path = root / "applanix/gps_post_process.csv"
    path.write_text(
        "GPSTime,easting,northing,altitude,roll,pitch,heading\n1,0,0,0,0,0,0\n2,0,0,0,0,0,0\n"
    )
    monkeypatch.setattr(
        adapter, "odometry_from_loader", lambda *a, **kw: pytest.fail("must not register")
    )
    with pytest.raises(ValueError, match="Applanix samples"):
        adapter.run_boreas_ins_lidar_hand_eye(root)
