"""CPU-only selection and CLI tests do not require the optional CUDA runtime."""
import sys

import numpy as np
import pytest

from calibrex.cli.main import main
from calibrex.solvers.lidar_lidar_map_solver import solve_map_extrinsic


def test_default_backend_does_not_import_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "calibrex.solvers.cuda_lidar_map_adapter", None)
    result = solve_map_extrinsic([], np.eye(4))
    assert result.status == "insufficient_samples"
    assert result.execution_backend == "numpy_scipy"
    with pytest.raises(ValueError, match="optional calibrex"):
        solve_map_extrinsic([], np.eye(4), backend="cuda")


def test_unknown_backend_is_rejected_before_solving():
    with pytest.raises(ValueError, match="backend must be"):
        solve_map_extrinsic([], np.eye(4), backend="unrecognized")


def test_cli_forwards_requested_gpu_backend_and_records_command(tmp_path, monkeypatch):
    initial = tmp_path / "initial.yaml"
    initial.write_text("matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n")
    captured = {}
    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        raise ValueError("intentional dispatch probe")
    monkeypatch.setattr("calibrex.evaluation.lidar_lidar_map.run_ros2_lidar_lidar_map", fake_run)
    assert main([
        "lidar-lidar", "ros2", str(tmp_path), "--reference-topic", "/reference",
        "--target-topic", "/target", "--initial", str(initial),
        "--dataset-family", "synthetic", "--dataset-license", "test",
        "--registration-backend", "cuda", "--output", str(tmp_path / "result.yaml"),
    ]) == 2
    assert captured["backend"] == "cuda"
    assert captured["command"][-2:] == ["--initial", str(initial)]
    assert "--registration-backend" in captured["command"]
