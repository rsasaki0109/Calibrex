"""Complete field-solve evidence and coherent metadata tampering checks."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import jsonschema
import pytest
from tools.audit_rgbd_field_columns import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/rgbd_field_columns"


def _load(path):
    return json.loads(
        gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    )


def _write(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def _refresh(root):
    index = _load(root / "index.json")
    for name in index["files"]:
        path = root / name
        index["files"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    _write(root / "index.json", index)


def test_complete_published_field_solver_work_and_results_are_exact():
    report = audit(ARCHIVE)
    assert report["complete_result_and_histories_exact"]
    assert len(report["attempts"]) == 6


@pytest.mark.parametrize("mutation", ["omit", "timing", "value", "status", "imports"])
def test_rehashed_evidence_tampering_is_rejected(tmp_path, mutation):
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    if mutation in ("omit", "timing"):
        path = root / "attempts.json"
        value = _load(path)
        if mutation == "omit":
            value["attempts"].pop()
        else:
            value["attempts"][0]["optimizer_seconds"] *= 0.5
        _write(path, value)
    else:
        folder = root / "01-current"
        result = _load(folder / "result.json.gz")
        traces = _load(folder / "traces.json.gz")
        measurement = _load(folder / "measurement.json")
        if mutation == "value":
            result["payload"]["train_rmse"] += 0.1
        elif mutation == "status":
            result["payload"]["status"] = measurement["status"] = "converged"
        else:
            for value in (result, traces, measurement):
                value["provenance"]["imports"]["calibrex.core.io"] = "outside.py"
        _write(folder / "result.json.gz", result)
        _write(folder / "traces.json.gz", traces)
        for key, name in (("result_sha256", "result.json.gz"), ("traces_sha256", "traces.json.gz")):
            measurement[key] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        _write(folder / "measurement.json", measurement)
    _refresh(root)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        audit(root)
