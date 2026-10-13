"""The public replay comparator must not hide quality or scope changes."""

from __future__ import annotations

from copy import deepcopy

import pytest
from tools import benchmark_cuda_scan_odometry as benchmark


def _pair() -> tuple[dict, dict]:
    cpu = {"execution_backend": "numpy_scipy", "poses": [[[1.0]]],
           "registrations": [{"execution_backend": "numpy_scipy", "converged": False,
                              "iterations": 30, "correspondences": 5015, "rmse_m": .001}]}
    gpu = deepcopy(cpu)
    identity = ("calibrex.cuda_scan_odometry/v0.1;fp64=true;query_fma=false;"
                "cupy=14.2.0;cuda=12090;driver=12090;cc=7.5;"
                f"query_sha256={'a' * 64};system_sha256={'b' * 64}")
    gpu["execution_backend"] = gpu["registrations"][0]["execution_backend"] = identity
    return cpu, gpu


def test_comparison_checks_all_poses_and_retains_failure_decisions() -> None:
    cpu, gpu = _pair()
    assert benchmark._compare(cpu, gpu) == []
    for key, value in (("converged", True), ("iterations", 29), ("correspondences", 5014),
                       ("rmse_m", .0011)):
        changed = deepcopy(gpu)
        changed["registrations"][0][key] = value
        assert benchmark._compare(cpu, changed) == [f".registrations[0].{key}"]
    gpu["poses"][0][0][0] += .01
    assert benchmark._compare(cpu, gpu) == [".poses[0][0][0]"]


def test_comparison_rejects_hidden_backend_suffix_and_type_changes() -> None:
    cpu, gpu = _pair()
    gpu["execution_backend"] += ";hidden_change=true"
    assert benchmark._compare(cpu, gpu) == [".execution_backend"]
    cpu, gpu = _pair()
    gpu["registrations"][0]["converged"] = 0
    assert benchmark._compare(cpu, gpu) == [".registrations[0].converged"]


def test_benchmark_does_not_overwrite_existing_output(tmp_path) -> None:
    existing = tmp_path / "result.json"
    existing.write_text("keep this")
    with pytest.raises(ValueError, match="new or empty"):
        benchmark.main(["--bag", "public-bag", "--output-dir", str(tmp_path)])
    assert existing.read_text() == "keep this"
    with pytest.raises(ValueError, match="seconds"):
        benchmark.main(["--bag", "public-bag", "--output-dir", str(tmp_path), "--seconds", "nan"])
