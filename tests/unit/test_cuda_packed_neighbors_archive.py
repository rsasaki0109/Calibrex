"""Complete public-result and source/observer audits of packed target queries."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import ValidationError
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_packed_neighbors"


def _rebind(root: Path) -> None:
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != path}
    path.write_text(json.dumps(index), encoding="utf-8")


def test_complete_public_pair_retains_quality_and_shared_gpu_observation() -> None:
    result = audit(ROOT)
    assert result["workers"] == 2 and result["quality_passed"]
    assert result["gpu_observation_samples"] == 23
    assert result["foreign_gpu_samples"] == 13
    assert result["unknown_gpu_samples"] == 0
    for name in ("whole_speedup", "registration_speedup", "normal_speedup"):
        assert result[name] > 0
    prior = json.loads((ROOT / "00-prior/measurement.json").read_bytes())
    current = json.loads((ROOT / "00-current/measurement.json").read_bytes())
    for left, right in zip(prior["rounds"], current["rounds"], strict=True):
        assert {k: v for k, v in left.items() if k != "seconds"} == {
            k: v for k, v in right.items() if k != "seconds"
        }
    old = json.loads((ROOT / "00-prior/result.json").read_bytes())
    new = json.loads((ROOT / "00-current/result.json").read_bytes())
    assert old["controls"] == new["controls"] and len(new["controls"]) == 32
    assert old["holdout_whitened_rms"] == new["holdout_whitened_rms"]


@pytest.mark.parametrize("mutation", ["rows", "bytes", "missing", "zero_time",
                                      "missing_time"])
def test_packing_and_registration_evidence_cannot_be_rebound(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    measurement_path = root / "00-current/measurement.json"
    measurement = json.loads(measurement_path.read_bytes())
    if mutation in ("rows", "bytes", "missing"):
        path = root / "00-current/result.json"
        result = json.loads(path.read_bytes())
        counters = result["options"]["initial_odometry_execution"][0]["statistics"]
        if mutation == "missing":
            counters.pop("gpu_packed_reference_rows")
        else:
            counters[f"gpu_packed_reference_{mutation}"] = 0
        path.write_text(json.dumps(result), encoding="utf-8")
        measurement["provenance"]["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == "zero_time":
        measurement["registration_seconds"] = 0
    else:
        measurement.pop("registration_seconds")
    measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
    _rebind(root)
    with pytest.raises(ValueError, match=r"packed reference|registration"):
        audit(root)


@pytest.mark.parametrize("mutation", ["observer", "schema", "launcher", "command",
                                      "offset", "negative", "exit"])
def test_observer_scope_and_bindings_cannot_be_rebound(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "observation.json"
    record = json.loads(path.read_bytes())
    if mutation in ("observer", "schema", "launcher"):
        record["provenance"][mutation + "_sha256"] = "0" * 64
    elif mutation == "command":
        command = record["provenance"]["benchmark_command"]
        command[command.index("--odometry-voxel-size-m") + 1] = "999"
    elif mutation == "offset":
        record["samples"][-1]["offset_seconds"] = 0
    elif mutation == "negative":
        record["samples"][0]["unknown"] = -1
    else:
        record["exit_code"] = -1
    path.write_text(json.dumps(record), encoding="utf-8")
    _rebind(root)
    with pytest.raises((ValueError, ValidationError), match=r"GPU observation|unknown"):
        audit(root)
