"""CPU-only audit of selective signed-control evaluation on public data."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/selective_bspline_controls"


def test_complete_selective_controls_archive_retains_quality_and_timings() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6 and result["quality_passed"]
    assert result["controls_speedup"] > 0
    assert result["whole_speedup"] > 0


@pytest.mark.parametrize("mutation", ["missing", "zero", "outside", "calls"])
def test_archive_rejects_rebound_false_controls_timing(
    tmp_path: Path, mutation: str,
) -> None:
    from jsonschema import ValidationError

    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "00-current/measurement.json"
    record = json.loads(path.read_bytes())
    if mutation == "missing":
        record.pop("controls_seconds")
        record.pop("controls_calls")
    elif mutation == "zero":
        record["controls_seconds"] = 0
    elif mutation == "outside":
        record["controls_seconds"] = record["pipeline_seconds"] + 1
    else:
        record["controls_calls"] = 2
    path.write_text(json.dumps(record), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["files"] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file() and p != index_path}
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises((ValueError, ValidationError), match=r"controls"):
        audit(root)
