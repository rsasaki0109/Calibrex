"""Backend routing and actual-work accounting across coverage gaps."""

from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pytest

from calibrex.cli.main import main
from calibrex.evaluation import odometry_windows as windows
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryResult, ScanRegistration


def test_backend_and_statistics_survive_short_segments_and_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Any] = []
    token = object()

    class FakeOdometry:
        def __init__(self, options: Any, *, rotation_model: Any, backend: str) -> None:
            assert backend == "cuda"
            assert rotation_model is token
            self.count = 0
            created.append(self)

        def add(self, *args: Any) -> None:
            self.count += 1

        def result(self) -> ScanOdometryResult:
            return ScanOdometryResult(
                tuple(np.eye(4) for _ in range(self.count)),
                tuple(ScanRegistration(np.eye(4), True, 1, 300, 0, 1)
                      for _ in range(self.count - 1)),
                "fake_cuda", {"gpu_query_rows": self.count * 11},
            )

    monkeypatch.setattr(windows, "IncrementalScanOdometry", FakeOdometry)
    options = windows.WindowingOptions(min_window_scans=2, odometry_backend="cuda")
    scans = [(float(i), np.ones((4, 3))) for i in range(4)]
    segmenter = windows.collect_odometry_windows(
        scans, options, covers=lambda t: t != 1, rotation_model=token,
    )
    # The first one-scan segment contributes actual work even though it cannot
    # contribute a calibration window. The second segment creates a fresh backend.
    assert [instance.count for instance in created] == [1, 2]
    assert segmenter.segments == 1
    assert len(segmenter.windows) == 1
    report = segmenter.execution_report()
    assert report == {
        "requested_backend": "cuda", "executed_backends": ["fake_cuda"],
        "statistics": {"gpu_query_rows": 33},
    }
    report["statistics"]["gpu_query_rows"] = 0
    assert segmenter.execution_statistics["gpu_query_rows"] == 33


def test_default_windowing_does_not_import_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cupy" or "cuda_scan_odometry_adapter" in name:
            pytest.fail("CPU windowing imported optional CUDA")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    segmenter = windows.collect_odometry_windows(
        [(0.0, np.ones((5, 3)))], windows.WindowingOptions(),
    )
    assert segmenter.execution_report() == {
        "requested_backend": "cpu", "executed_backends": ["numpy_scipy"], "statistics": {},
    }


def test_invalid_windowing_backend_fails_before_reading() -> None:
    with pytest.raises(ValueError, match="unknown odometry backend"):
        windows.WindowingOptions(odometry_backend="unknown")


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_joint_cli_records_and_forwards_backend_and_geometry(
    monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    seen: dict[str, Any] = {}

    def capture(bag: Any, profile: Any, options: Any, **kwargs: Any) -> Any:
        seen.update(options=options, **kwargs)
        raise ValueError("intentional stop after option routing")

    monkeypatch.setattr("calibrex.check.estimators.detect_point_time", lambda *_: ("t", "offset_s"))
    monkeypatch.setattr("calibrex.evaluation.lidar_imu_bag_calibration.calibrate_lidar_imu_bag",
                        capture)
    assert main([
        "imu-lidar", "joint", "public.bag", "--lidar-topic", "/p", "--imu-topic", "/i",
        "--dataset-family", "public", "--dataset-license", "upstream", "--output", "unused.json",
        "--odometry-backend", backend, "--odometry-voxel-size-m", "0.03",
        "--odometry-local-map-scans", "3", "--odometry-geometry-quantum-m", "1e-9",
        "--odometry-query-workers", "8",
        "--odometry-motion-translation-quantum-m", "1e-6",
        "--odometry-motion-rotation-quantum-rad", "1e-7",
    ]) == 2
    options = seen["options"].rotation.windowing
    assert options.odometry_backend == backend
    assert options.odometry.voxel_size_m == 0.03
    assert options.odometry.local_map_scans == 3
    assert options.odometry.query_workers == 8
    assert options.odometry.geometry_quantum_m == 1e-9
    assert options.odometry.motion_translation_quantum_m == 1e-6
    assert options.odometry.motion_rotation_quantum_rad == 1e-7
    command = seen["command"]
    for flag, value in (
        ("--odometry-backend", backend), ("--odometry-voxel-size-m", "0.03"),
        ("--odometry-local-map-scans", "3"), ("--odometry-geometry-quantum-m", "1e-09"),
        ("--odometry-query-workers", "8"),
        ("--odometry-motion-translation-quantum-m", "1e-06"),
        ("--odometry-motion-rotation-quantum-rad", "1e-07"),
    ):
        assert command[command.index(flag) + 1] == value
