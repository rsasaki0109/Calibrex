"""CPU-only checks of the query diagnostic's scope, coverage and source bindings."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import jsonschema
import pytest
from tools.audit_cuda_spatial_candidates import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_spatial_candidates"


def _rebind(root: Path) -> None:
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.iterdir() if p.is_file() and p != path}
    index["provenance"]["measurement_sha256"] = index["files"]["measurement.json"]
    path.write_text(json.dumps(index), encoding="utf-8")


def test_archive_is_complete_and_has_no_whole_calibration_speedup_claim() -> None:
    result = audit(ROOT)
    assert result["candidates"] == 18
    assert result["exact_query_parity"]
    assert "whole_speedup" not in result
    assert result["scope"] == "warmed GPU query kernels; not complete calibration"


@pytest.mark.parametrize("mutation", ["median", "batch", "duplicate", "parity", "count"])
def test_audit_rejects_rebound_false_candidate_evidence(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "measurement.json"
    record = json.loads(path.read_bytes())
    if mutation == "median":
        record["records"][0]["kernel_median_seconds"] *= 2
    elif mutation == "batch":
        record["records"][0]["batch_seconds_per_launch"].pop()
    elif mutation == "duplicate":
        record["records"][1] = record["records"][0]
    elif mutation == "parity":
        record["records"][0]["exact_distance_and_index_parity"] = False
    else:
        record["records"][1]["query_rows"] += 1
    path.write_text(json.dumps(record), encoding="utf-8")
    _rebind(root)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        audit(root)


def test_audit_rejects_rebound_shader_that_does_not_match_its_generator(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "heap.cu.txt"
    path.write_bytes(path.read_bytes().replace(b"best[0]", b"best[1]"))
    measurement = root / "measurement.json"
    record = json.loads(measurement.read_bytes())
    record["provenance"]["shader_sha256"]["heap"] = hashlib.sha256(path.read_bytes()).hexdigest()
    measurement.write_text(json.dumps(record), encoding="utf-8")
    _rebind(root)
    with pytest.raises(ValueError, match="candidate shader"):
        audit(root)


def test_audit_rejects_rebound_import_manifest_that_disagrees_with_archived_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    path = root / "sources-manifest.json"
    manifest = json.loads(path.read_bytes())
    key = next(iter(manifest))
    manifest[key] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    measurement = root / "measurement.json"
    record = json.loads(measurement.read_bytes())
    record["provenance"]["executed_sources"][key] = "0" * 64
    measurement.write_text(json.dumps(record), encoding="utf-8")
    _rebind(root)
    with pytest.raises(ValueError, match="imported source"):
        audit(root)
