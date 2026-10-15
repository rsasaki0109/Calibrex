"""Audit full control evidence and the separately measured scoring scope."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/holdout_evaluation_reuse"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_complete_holdout_comparison_audits_without_optional_accelerators(
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
    assert result["control_speedup"] > 1
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    times = result["control_median_seconds"]
    assert f"{times['prior']:.6f} s | {times['current']:.6f} s" in readme
    assert f"{result['control_speedup']:.3f}x" in readme


@pytest.mark.parametrize("mutation", [
    "control_delta", "missing_timer", "score_count", "scope", "baseline_source",
])
def test_rebound_hashes_do_not_hide_changed_controls_or_invalid_observations(
    tmp_path: Path, mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path, record_path = root / "00-current/result.json", root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    if mutation == "control_delta":
        value = result["controls"][0]["delta_chi2"]
        result["controls"][0]["delta_chi2"] = float(np.nextafter(value, np.inf))
    elif mutation == "missing_timer":
        record.pop("control_evaluation_seconds")
    elif mutation == "score_count":
        record["control_score_calls"] -= 1
    elif mutation == "scope":
        record["control_evaluation_seconds"] = record["pipeline_seconds"] * 2
    else:
        path = root / "baseline-validation.json"
        receipt = json.loads(path.read_bytes())
        receipt["source_commit"] = "0" * 40
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
