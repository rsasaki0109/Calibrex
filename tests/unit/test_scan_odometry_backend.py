"""Default CPU isolation, optional-dependency errors and CLI provenance."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from calibrex.cli.main import main
from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    register_point_to_plane,
    run_scan_to_scan_odometry,
)


def test_cpu_odometry_never_imports_optional_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name == "cupy" or "cuda_scan_odometry_adapter" in name:
            raise ImportError("optional GPU dependency unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert run_scan_to_scan_odometry([]).execution_backend == "numpy_scipy"
    with pytest.raises(ValueError, match=r"calibrex\[cuda\]"):
        IncrementalScanOdometry(backend="cuda")
    with pytest.raises(ValueError, match=r"calibrex\[cuda\]"):
        register_point_to_plane(np.empty((0, 3)), np.empty((0, 3)), np.eye(4), backend="cuda")


def test_unknown_odometry_backend_rejected_before_geometry() -> None:
    with pytest.raises(ValueError, match="unknown odometry backend"):
        IncrementalScanOdometry(backend="invalid")
    with pytest.raises(ValueError, match="unknown odometry backend"):
        register_point_to_plane(np.empty((0, 3)), np.empty((0, 3)), np.eye(4), backend="invalid")


def test_lidar_cli_forwards_cuda_odometry_and_records_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    from pathlib import Path

    folder = Path(str(tmp_path))
    initial = folder / "initial.yaml"
    initial.write_text("matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n")
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        raise ValueError("intentional test stop")

    monkeypatch.setattr("calibrex.evaluation.lidar_lidar_map.run_ros2_lidar_lidar_map", fake_run)
    assert (
        main(
            [
                "lidar-lidar",
                "ros2",
                "public.bag",
                "--reference-topic",
                "/a",
                "--target-topic",
                "/b",
                "--initial",
                str(initial),
                "--odometry-backend",
                "cuda",
                "--dataset-family",
                "public",
                "--dataset-license",
                "test-only",
                "--output",
                str(folder / "result.json"),
            ]
        )
        == 2
    )
    assert seen["odometry_backend"] == "cuda"
    assert seen["backend"] == "cpu"
    command = seen["command"]
    assert command[command.index("--odometry-backend") + 1] == "cuda"
