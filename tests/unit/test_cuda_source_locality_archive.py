"""CPU-only audit of dense registration scheduling and complete public evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_source_locality"


def test_complete_source_locality_archive_audits_registration_and_quality() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6 and result["quality_passed"]
    assert result["registration_speedup"] > 0 and result["whole_speedup"] > 0
    repeated = result["monitored_repeat"]
    assert repeated["quality_passed"] and repeated["workers"] == 6
    assert repeated["gpu_observation_samples"] == 52
    assert repeated["foreign_gpu_samples"] == 32
    assert repeated["unknown_gpu_samples"] == 0


@pytest.mark.parametrize("mutation", ["observer", "launcher", "schema", "interval",
                                      "offset", "negative", "command", "geometry"])
def test_monitored_repeat_rejects_rebound_false_observation(
    tmp_path: Path, mutation: str,
) -> None:
    from jsonschema import ValidationError

    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    repeated = root / "repeat-v2"
    path = repeated / "observation.json"
    record = json.loads(path.read_bytes())
    if mutation in ("observer", "launcher", "schema"):
        record["provenance"][mutation + "_sha256"] = "0" * 64
    elif mutation == "interval":
        record["provenance"]["sampling_interval_seconds"] = 1
    elif mutation == "offset":
        record["samples"][-1]["offset_seconds"] = 0
    elif mutation == "negative":
        record["samples"][0]["foreign"] = -1
    else:
        command = record["provenance"]["benchmark_command"]
        option = "--max-seconds" if mutation == "command" else "--odometry-voxel-size-m"
        command[command.index(option) + 1] = "999"
    path.write_text(json.dumps(record), encoding="utf-8")
    for folder in (repeated, root):
        index_path = folder / "publication-index.json"
        index = json.loads(index_path.read_bytes())
        index["files"] = {
            p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file() and p != index_path
        }
        index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises((ValueError, ValidationError), match=r"GPU observation|foreign"):
        audit(root)


@pytest.mark.parametrize("mutation", ["calls", "rows", "kernel", "missing",
                                      "zero_time", "outside_time", "missing_time",
                                      "observed_calls"])
def test_archive_rejects_rebound_false_ordering_or_registration_evidence(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "00-current/measurement.json"
    record = json.loads(path.read_bytes())
    if mutation in ("calls", "rows", "kernel", "missing"):
        result_path = root / "00-current/result.json"
        result = json.loads(result_path.read_bytes())
        counters = result["options"]["initial_odometry_execution"][0]["statistics"]
        key = {"calls": "gpu_source_order_calls", "rows": "gpu_source_order_rows",
               "kernel": "gpu_source_order_kernel_calls",
               "missing": "gpu_source_order_calls"}[mutation]
        if mutation == "missing":
            counters.pop(key)
        else:
            counters[key] = 0
        result_path.write_text(json.dumps(result), encoding="utf-8")
        record["provenance"]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    elif mutation == "zero_time":
        record["registration_seconds"] = 0
    elif mutation == "outside_time":
        record["registration_seconds"] = record["pipeline_seconds"] + 1
    elif mutation == "missing_time":
        record.pop("registration_seconds")
        record.pop("registration_calls")
    else:
        record["registration_calls"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match=r"source ordering|registration"):
        audit(root)
