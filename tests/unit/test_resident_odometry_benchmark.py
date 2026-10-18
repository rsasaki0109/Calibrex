"""The public comparison binds resident work to source hashes and all output bits."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.benchmark_bspline_lidar_imu import (
    _resident_backends,
    _resident_hashes,
    _same_float_bits,
    _validate_resident_pair,
    _worker_arguments,
)


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "physics",
        "signed_zero",
        "fit",
        "identity",
        "native_identity",
        "work",
        "option",
        "empty_cache",
        "missing_target",
        "extra",
        "common_work",
    ],
)
def test_resident_comparison_permits_only_declared_execution_metadata(mutation: str) -> None:
    native_hashes = ("a" * 64, "b" * 64)
    hashes = {"query_sha256": "c" * 64, "system_sha256": "d" * 64, "geometry_sha256": "e" * 64}
    base = "calibrex.cuda_scan_odometry/v0.1;fp64=true;query_fma=false;"
    native_identity = (
        base + "query_sha256=" + native_hashes[0] + ";system_sha256=" + native_hashes[1]
    )
    resident_identity = base + ";".join(f"{key}={value}" for key, value in hashes.items())
    resident_identity += ";geometry_pipeline=resident-v1"
    report = {
        "requested_backend": "cuda",
        "executed_backends": [native_identity],
        "statistics": {
            "gpu_query_rows": 1000,
            "gpu_fused_system_calls": 10,
            "gpu_normal_rows": 100,
            "scalar_query_repairs": 4,
            "serialized_tree_captures": 2,
        },
    }
    prior = {
        "physics": [0.0, 0.3],
        "options": {
            "odometry_backend": "cuda",
            "initial_odometry_execution": [report],
            "segment_odometry_execution": [copy.deepcopy(report)],
        },
    }
    current = copy.deepcopy(prior)
    current["options"].update(
        resident_odometry=True,
        resident_geometry_execution={
            "source_cache_bytes": 2400,
            "source_captures": 2,
            "source_cache_hits": 2,
            "preprocess_calls": 10,
            "preprocess_rows": 1000,
            "transforms": 5,
        },
    )
    for name in ("initial_odometry_execution", "segment_odometry_execution"):
        current["options"][name][0]["executed_backends"] = [resident_identity]
        current["options"][name][0]["statistics"].update(
            gpu_grid_captures=2,
            scalar_query_repairs=1,
            serialized_tree_captures=0,
        )
    left, right = (
        [{"seconds": 1.0, "covariance": [[0.1]]}],
        [{"seconds": 2.0, "covariance": [[0.1]]}],
    )
    before = copy.deepcopy(current)
    _validate_resident_pair(prior, current, left, right, native_hashes, hashes)
    assert current == before
    if mutation == "none":
        return
    changed = current["options"]["segment_odometry_execution"][0]
    if mutation == "physics":
        current["physics"][1] += 1e-15
    elif mutation == "signed_zero":
        current["physics"][0] = -0.0
    elif mutation == "fit":
        right[0]["covariance"][0][0] += 1e-15
    elif mutation == "identity":
        changed["executed_backends"] = [resident_identity.replace("c" * 64, "f" * 64)]
    elif mutation == "native_identity":
        prior["options"]["segment_odometry_execution"][0]["executed_backends"] = [base]
    elif mutation == "work":
        current["options"]["resident_geometry_execution"]["source_cache_hits"] = 0
    elif mutation == "option":
        current["options"]["resident_odometry"] = False
    elif mutation == "empty_cache":
        current["options"]["resident_geometry_execution"]["source_cache_bytes"] = 0
    elif mutation == "missing_target":
        changed["statistics"]["gpu_grid_captures"] = 0
    elif mutation == "extra":
        current["extra"] = 1
    elif mutation == "common_work":
        changed["statistics"]["gpu_normal_rows"] += 1
    with pytest.raises(ValueError):
        _validate_resident_pair(prior, current, left, right, native_hashes, hashes)


def test_resident_hashes_and_worker_flags_need_no_device_import() -> None:
    hashes = _resident_hashes(Path(__file__).resolve().parents[2])
    assert set(hashes) == {"query_sha256", "system_sha256", "geometry_sha256"}
    assert all(len(value) == 64 for value in hashes.values())
    arguments = SimpleNamespace(
        comparison="cuda-cuda",
        prior_fit_plane_backend="cuda",
        current_fit_plane_backend="cuda",
        current_resident_odometry=True,
    )
    assert _resident_backends(arguments) == {"prior": False, "current": True}
    assert "--current-resident-odometry" in _worker_arguments(arguments)
    assert "--prior-resident-odometry" not in _worker_arguments(arguments)
    assert not _same_float_bits(0.0, -0.0)
