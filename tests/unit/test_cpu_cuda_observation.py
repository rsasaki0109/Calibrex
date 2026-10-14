"""The observer binds effective CPU/CUDA sources rather than an ignored ref."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_joint_lidar_imu"


@pytest.mark.parametrize("mutation", [None, "effective_baseline", "comparison"])
def test_cpu_cuda_observer_uses_effective_source_and_checks_comparison(
    tmp_path: Path, mutation: str | None,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "protocol.json"
    protocol = json.loads(path.read_bytes())
    assert protocol["comparison"] == "cpu-cuda"
    if mutation == "effective_baseline":
        protocol["baseline_commit"] = "0" * 40
        path.write_text(json.dumps(protocol), encoding="utf-8")
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_bytes())
        summary["provenance"]["protocol_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    observer = b"# Synthetic observer fixture for CPU-only binding checks.\n"
    schema = b'{"type":"object"}\n'
    (root / "observer.py.txt").write_bytes(observer)
    (root / "observation.schema.json").write_bytes(schema)
    command = ["<PYTHON>", "--comparison",
               "cuda-cuda" if mutation == "comparison" else "cpu-cuda",
               "--baseline-commit", "b89e50bbd9e48a05a2acb6010ac46e50ece9220f"]
    for option, key in (("--max-seconds", "max_seconds"), ("--max-rounds", "max_rounds"),
                        ("--repeats", "repeats")):
        command.extend((option, str(protocol[key])))
    for key, value in protocol["odometry_options"].items():
        command.extend(("--odometry-" + key.replace("_", "-"), str(value)))
    observed = {"samples": [{"offset_seconds": 0, "own": 0, "foreign": 0, "unknown": 0}],
                "exit_code": 0, "provenance": {
                    "observer_sha256": hashlib.sha256(observer).hexdigest(),
                    "schema_sha256": hashlib.sha256(schema).hexdigest(),
                    "launcher_sha256": protocol["launcher_sha256"],
                    "sampling_interval_seconds": 10, "benchmark_command": command}}
    (root / "observation.json").write_text(json.dumps(observed), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["provenance"]["input_protocol_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    if mutation is None:
        result = audit(root)
        assert result["quality_passed"] and result["gpu_observation_samples"] == 1
    else:
        with pytest.raises(ValueError, match=r"observation.*command|same declared source revision"):
            audit(root)
