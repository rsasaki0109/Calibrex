"""Source integrity and quality guards for the complete public-data comparison."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.benchmark_bspline_lidar_imu import (
    _backend,
    _compare,
    _manifest,
    _time_controls,
    _time_normal_preparation,
    _validate_backend_execution,
    _verify_sources,
    _write,
)


def test_controls_observer_preserves_result_and_exception() -> None:
    def controls(fail: bool) -> list[int]:
        if fail:
            raise ValueError("original error")
        return [3, 7]

    holder = SimpleNamespace(_controls=controls)
    timings = _time_controls(holder)
    assert holder._controls(False) == [3, 7]
    with pytest.raises(ValueError, match="original error"):
        holder._controls(True)
    assert timings["calls"] == 2 and timings["seconds"] > 0


def test_normal_timing_counts_nested_cache_completion_once() -> None:
    class Target:
        def _ensure(self, value: int) -> int:
            return value + 1

        def _ensure_all(self) -> int:
            return self._ensure(5)

    timings = _time_normal_preparation(Target)
    assert Target()._ensure_all() == 6
    assert timings["calls"] == 1
    assert Target()._ensure(7) == 8
    assert timings["calls"] == 2
    assert timings["seconds"] >= 0


@pytest.mark.parametrize("comparison,prior,current", [
    ("source", "cpu", "cpu"), ("cpu-cuda", "cpu", "cuda"), ("cuda-cuda", "cuda", "cuda"),
])
def test_comparison_selects_each_actual_worker_backend(
    comparison: str, prior: str, current: str,
) -> None:
    assert _backend(comparison, "prior") == prior
    assert _backend(comparison, "current") == current


def test_gpu_source_comparison_requires_the_same_explicit_workload(tmp_path: Path) -> None:
    from jsonschema import ValidationError

    protocol = json.loads(Path("docs/assets/cuda_joint_lidar_imu/protocol.json").read_bytes())
    protocol["comparison"] = "cuda-cuda"
    _write(tmp_path / "protocol.json", protocol)
    protocol.pop("odometry_options")
    with pytest.raises(ValidationError):
        _write(tmp_path / "protocol.json", protocol)


def _evidence() -> dict:
    return {
        "t_lidar_imu_rotation_quat_xyzw": [0, 0, 0, 1],
        "t_lidar_imu_translation_m": [0.01, 0.02, 0.03],
        "imu_clock_offset_s": 0.007,
        "policy_status": "inconclusive",
        "controls": [
            {"parameter": "imu_clock_offset", "sign": 1, "detected": False, "delta_chi2": 1.0}
        ],
        "blocks": [
            {
                "name": "imu_clock_offset",
                "verdict": "weak",
                "estimate": [0.007],
                "formal_std": [0.1],
                "last_round_change": [0.0001],
            }
        ],
        "separability": [],
        "converged": False,
        "noise_sigmas": {"gyro": 0.03},
        "holdout_whitened_rms": {"gyro": 1.0, "accel": 1.01, "plane": 0.99},
    }


def test_quality_accepts_numeric_roundoff_but_rejects_decision_or_physical_changes() -> None:
    prior = _evidence()
    assert _compare(prior, copy.deepcopy(prior))["passed"]
    current = copy.deepcopy(prior)
    current["imu_clock_offset_s"] += 1e-10
    assert _compare(prior, current)["passed"]
    for path, value in (
        (("policy_status",), "pass"),
        (("controls", 0, "detected"), True),
        (("blocks", 0, "verdict"), "estimated"),
        (("t_lidar_imu_translation_m", 0), 0.1),
        (("imu_clock_offset_s",), 0.009),
        (("holdout_whitened_rms", "gyro"), 1.001),
        (("blocks", 0, "formal_std", 0), 0.2),
        (("controls", 0, "delta_chi2"), 2.0),
    ):
        current = copy.deepcopy(prior)
        parent = current
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        assert not _compare(prior, current)["passed"], path


@pytest.mark.parametrize("mutation", ["change", "remove", "extra"])
def test_source_manifest_rejects_changed_missing_or_extra_source(
    tmp_path: Path, mutation: str
) -> None:
    source = tmp_path / "src/calibrex/core/frozen.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n", encoding="utf-8")
    _write(tmp_path / "manifest.json", _manifest(tmp_path / "src"))
    assert _verify_sources(tmp_path)
    if mutation == "change":
        source.write_text("x = 2\n", encoding="utf-8")
    elif mutation == "remove":
        source.unlink()
    else:
        source.with_name("extra.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source tree changed"):
        _verify_sources(tmp_path)


def test_record_schema_rejects_incomplete_measurement(tmp_path: Path) -> None:
    from jsonschema import ValidationError

    with pytest.raises(ValidationError):
        _write(
            tmp_path / "measurement.json",
            {"schema_version": "slac.bspline_lidar_imu_comparison/v0.1", "record_type": "worker"},
        )


@pytest.mark.parametrize("mutation", ["identity", "queries", "systems", "negative", "missing"])
def test_gpu_comparison_requires_actual_work_in_every_odometry_pass(mutation: str) -> None:
    report = {
        "requested_backend": "cuda",
        "executed_backends": ["calibrex.cuda_scan_odometry/v0.1;test"],
        "statistics": {"gpu_query_rows": 1000, "gpu_fused_system_calls": 10},
    }
    result = {"options": {
        "odometry_backend": "cuda", "initial_odometry_execution": [report],
        "segment_odometry_execution": [copy.deepcopy(report)],
    }}
    _validate_backend_execution(result, "cuda")
    if mutation == "identity":
        report["executed_backends"] = ["numpy_scipy"]
    elif mutation in ("queries", "systems"):
        report["statistics"]["gpu_query_rows" if mutation == "queries"
                              else "gpu_fused_system_calls"] = 0
    elif mutation == "negative":
        report["statistics"]["scalar_query_repairs"] = -1
    else:
        result["options"]["segment_odometry_execution"] = []
    with pytest.raises(ValueError):
        _validate_backend_execution(result, "cuda")
