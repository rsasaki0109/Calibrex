"""Direct default/parallel controls and coherently rehashed evidence edits."""

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import jsonschema
import pytest
from tools.audit_tum_scoped_evaluation import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/tum_scoped_evaluation"


def _load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def _write(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def _refresh(root):
    index = _load(root / "index.json")
    for name in index["files"]:
        p = root / name
        index["files"][name] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size,
        }
    _write(root / "index.json", index)


def test_all_nine_direct_workers_and_original_histories_match_accepted_reference():
    result = audit(ARCHIVE)
    assert result["production_workers"] == 9
    assert result["complete_results_and_histories_exact"]
    assert 1.9 < result["direct_example_speedup"] < 2.1
    assert result["arithmetic_at_same_workers_speedup"] > 1.2


@pytest.mark.parametrize(
    "change", ["omit", "timing", "result", "imports", "population", "source", "input",
               "example", "trace"],
)
def test_rehashed_execution_and_calibration_edits_are_rejected(tmp_path, change):
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    if change in ("omit", "timing"):
        value = _load(root / "attempts.json")
        if change == "omit":
            value["attempts"].pop()
        else:
            value["attempts"][0]["api_seconds"] *= 0.5
        _write(root / "attempts.json", value)
        summary = _load(root / "summary.json")
        summary["attempts"] = value["attempts"]
        _write(root / "summary.json", summary)
    elif change in ("source", "input"):
        p = root / ("sources-current-manifest.json" if change == "source" else "inputs.json")
        value = _load(p)
        if change == "source":
            value["calibrex/data/livox.py"] = "0" * 64
        else:
            value[0]["sha256"] = "0" * 64
        _write(p, value)
    elif change == "example":
        import yaml

        p = root / "example-current.yaml"
        value = yaml.safe_load(p.read_bytes())
        value["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"][
            "window_workers"
        ] = 2
        p.write_bytes(yaml.safe_dump(value).encode())
        protocol = _load(root / "provenance.json")
        protocol["example_sha256"]["current"] = hashlib.sha256(p.read_bytes()).hexdigest()
        _write(root / "provenance.json", protocol)
    elif change == "trace":
        p = root / "supplementary-traces.tar.gz"
        with tarfile.open(p, "r:gz") as bundle:
            members = {m.name: bundle.extractfile(m).read() for m in bundle}
        name = "01-current/history.json.gz"
        value = json.loads(gzip.decompress(members[name]))
        value["payload"]["events"][0]["residual"]["sha256"] = "0" * 64
        members[name] = gzip.compress((json.dumps(value) + "\n").encode(), mtime=0)
        with tarfile.open(p, "w:gz") as bundle:
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
            for record in (result, measurement):
                record["provenance"]["imports"]["calibrex.core.io"] = "outside.py"
        else:
            for record in (result, measurement):
                record["provenance"]["verified_inputs"].pop("depth.txt")
        _write(folder / "result.json.gz", result)
        measurement["result_sha256"] = hashlib.sha256(
            (folder / "result.json.gz").read_bytes()
        ).hexdigest()
        _write(folder / "measurement.json", measurement)
    _refresh(root)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        audit(root)
