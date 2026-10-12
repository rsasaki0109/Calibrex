"""Protect the numerical audit and non-destructive benchmark preflight."""

from copy import deepcopy

import pytest
from tools import benchmark_cuda_lidar_map as benchmark


def test_comparator_preserves_counts_decisions_and_every_numeric_field():
    cpu = {"result": {"execution_backend": "numpy_scipy", "transform": [[1.0, 0.0]],
                      "covariance": [[1e-10]], "correspondences": 300},
           "policy_status": "pass", "control": {"detected": True, "score": 10.0}}
    gpu = deepcopy(cpu)
    gpu["result"]["execution_backend"] = (
        "calibrex.cuda_lidar_map/v0.1;fp64=true;query_fma=false;cupy=14.2.0;"
        "cuda=12090;driver=13000;cc=7.5;query_sha256=" + "0" * 64
    )
    assert not benchmark._compare(cpu, gpu)
    gpu["result"]["transform"][0][1] = 1e-5
    assert benchmark._compare(cpu, gpu) == [".result.transform[0][1]"]
    gpu["result"]["transform"][0][1] = 1e-10
    assert not benchmark._compare(cpu, gpu)
    gpu["control"]["detected"] = False
    assert benchmark._compare(cpu, gpu) == [".control.detected"]
    gpu["control"]["detected"] = True
    gpu["result"]["correspondences"] += 1
    assert benchmark._compare(cpu, gpu) == [".result.correspondences"]
    gpu["result"]["correspondences"] -= 1
    gpu["result"]["execution_backend"] += ";hidden_change=true"
    assert benchmark._compare(cpu, gpu) == [".result.execution_backend"]


def test_existing_outputs_and_invalid_budgets_are_preserved(tmp_path):
    destination = tmp_path / "output"
    destination.mkdir()
    evidence = destination / "preserve.txt"
    evidence.write_text("evidence")
    base = ["--sequence", str(tmp_path), "--output-dir", str(destination)]
    for option, value in (("--half-window", "6"), ("--repetitions", "0")):
        with pytest.raises(ValueError, match="repetitions must"):
            benchmark.main([*base, option, value])
    with pytest.raises(ValueError, match="new or empty"):
        benchmark.main(base)
    assert evidence.read_text() == "evidence"
