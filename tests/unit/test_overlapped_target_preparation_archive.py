"""Scheduling-only changes keep all physical evidence and executed GPU work exact."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/overlapped_target_preparation"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_complete_overlap_comparison_keeps_all_outputs_exact_without_a_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name in ("cupy", "numba") or name.startswith(("cupy.", "numba.")):
            raise AssertionError("archive audit imported an optional accelerator")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert json.loads((ROOT / "protocol.json").read_bytes())["exact_outputs_required"] is True
    result = audit(ROOT)
    assert result["workers"] == 4 and result["quality_passed"]
    summary = json.loads((ROOT / "summary.json").read_bytes())
    times = summary["whole_median_seconds"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{times['prior']:.3f} s | {times['current']:.3f} s" in readme
    assert f"{result['whole_speedup']:.3f}x" in readme
    assert "Not adopted" in readme
    assert "51 passed" in (ROOT / "prototype-validation/pytest.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("mutation", [
    "trajectory", "execution_work", "ratio", "baseline_source", "prototype_source",
])
def test_rebound_hashes_do_not_hide_quality_work_or_source_changes(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path, record_path = root / "00-current/result.json", root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    if mutation == "trajectory":
        value = record["rounds"][-1]["trajectory_positions"][1][0]
        record["rounds"][-1]["trajectory_positions"][1][0] = float(np.nextafter(value, np.inf))
    elif mutation == "execution_work":
        result["options"]["initial_odometry_execution"][0]["statistics"]["gpu_query_calls"] += 1
    elif mutation == "ratio":
        path = root / "summary.json"
        summary = json.loads(path.read_bytes())
        summary["whole_speedup"] *= 2
        _write(path, summary)
    elif mutation == "baseline_source":
        path = root / "baseline-validation.json"
        receipt = json.loads(path.read_bytes())
        receipt["source_commit"] = "0" * 40
        _write(path, receipt)
    else:
        path = root / "prototype-validation/receipt.json"
        receipt = json.loads(path.read_bytes())
        receipt["provenance"]["source_commit"] = "0" * 40
        _write(path, receipt)
    _write(result_path, result)
    record["provenance"]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    _write(record_path, record)
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file() and p != path
    }
    _write(path, index)
    with pytest.raises(ValueError):
        audit(root)
