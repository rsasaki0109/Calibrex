"""Rejected candidates retain complete work and resist coherently rehashed edits."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import jsonschema
import pytest
from tools.audit_tum_diagnostic_execution import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/tum_diagnostic_execution"


def _load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


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


def test_complete_production_and_original_work_match_without_promoting_the_candidate():
    result = audit(ARCHIVE)
    assert result["production_workers"] == 6
    assert result["complete_results_exact"]
    assert result["fixed_population_events_per_run"] == 492
    assert result["runtime_candidate_adopted"] is False
    assert result["whole_speedup"] < 1.02


@pytest.mark.parametrize(
    "change", ["omit", "timing", "result", "imports", "population", "source", "input", "trace"],
)
def test_rehashed_candidate_evidence_edits_are_rejected(tmp_path, change):
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    if change in ("omit", "timing"):
        path = root / "attempts.json"
        value = _load(path)
        if change == "omit":
            value["attempts"].pop()
        else:
            value["attempts"][0]["api_seconds"] *= 0.5
        _write(path, value)
        summary = _load(root / "summary.json")
        summary["attempts"] = value["attempts"]
        _write(root / "summary.json", summary)
    elif change in ("source", "input"):
        path = root / ("sources-current-manifest.json" if change == "source" else "inputs.json")
        value = _load(path)
        if change == "source":
            value["calibrex/data/livox.py"] = "0" * 64
        else:
            value[0]["sha256"] = "0" * 64
        _write(path, value)
    elif change == "trace":
        import io
        import tarfile

        path = root / "supplementary-traces.tar.gz"
        with tarfile.open(path, "r:gz") as bundle:
            members = {m.name: bundle.extractfile(m).read() for m in bundle}
        name = "01-current/history.json.gz"
        value = json.loads(gzip.decompress(members[name]))
        value["payload"]["events"][0]["residual"]["sha256"] = "0" * 64
        members[name] = gzip.compress((json.dumps(value) + "\n").encode(), mtime=0)
        with tarfile.open(path, "w:gz") as bundle:
            for name, raw in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                bundle.addfile(info, io.BytesIO(raw))
    else:
        folder = root / "01-current"
        result = _load(folder / "result.json.gz")
        measurement = _load(folder / "measurement.json")
        if change == "result":
            result["payload"]["metrics"]["tum_joint_point_to_plane_rmse_m"]["holdout"] += 0.1
        elif change == "imports":
            for value in (result, measurement):
                value["provenance"]["imports"]["calibrex.core.io"] = "outside.py"
        else:
            for value in (result, measurement):
                value["provenance"]["verified_inputs"].pop("depth.txt")
        _write(folder / "result.json.gz", result)
        measurement["result_sha256"] = hashlib.sha256(
            (folder / "result.json.gz").read_bytes()
        ).hexdigest()
        _write(folder / "measurement.json", measurement)
    _refresh(root)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        audit(root)
