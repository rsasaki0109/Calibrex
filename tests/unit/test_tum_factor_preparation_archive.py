"""Complete factor-preparation evidence and coherent metadata tampering checks."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import jsonschema
import pytest
from tools.audit_tum_factor_preparation import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/tum_factor_preparation"


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


def test_complete_published_preparation_work_and_results_are_exact():
    report = audit(ARCHIVE)
    assert report["complete_result_and_histories_exact"]
    assert len(report["attempts"]) == 9


@pytest.mark.parametrize(
    "mutation",
    [
        "omit",
        "timing",
        "value",
        "status",
        "imports",
        "frontend",
        "population",
        "execution",
        "config",
    ],
)
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
    elif mutation == "execution":
        path = root / "01-current/execution.json"
        value = _load(path)
        value["execution"]["kernel_launches"] -= 1
        _write(path, value)
    elif mutation == "config":
        import yaml

        path = root / "config-cuda.yaml"
        value = yaml.safe_load(path.read_bytes())
        value["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"][
            "query_points_per_frame"
        ] -= 1
        path.write_bytes(yaml.safe_dump(value, sort_keys=False).encode())
        protocol = _load(root / "provenance.json")
        protocol["config_sha256"]["cuda"] = hashlib.sha256(path.read_bytes()).hexdigest()
        _write(root / "provenance.json", protocol)
    else:
        folder = root / "01-current"
        result = _load(folder / "result.json.gz")
        traces = _load(folder / "traces.json.gz")
        measurement = _load(folder / "measurement.json")
        frontend = _load(folder / "frontend.json.gz")
        if mutation == "value":
            result["payload"]["train_rmse"] += 0.1
        elif mutation == "status":
            result["payload"]["status"] = measurement["status"] = "converged"
        elif mutation in ("frontend", "population"):
            frontend["payload"][1]["points_sha256"] = "0" * 64
            if mutation == "population":
                frontend["payload"][1]["count"] -= 1
        else:
            for value in (result, traces, measurement, frontend):
                value["provenance"]["imports"]["calibrex.core.io"] = "outside.py"
        _write(folder / "result.json.gz", result)
        _write(folder / "traces.json.gz", traces)
        _write(folder / "frontend.json.gz", frontend)
        for key, name in (
            ("result_sha256", "result.json.gz"),
            ("traces_sha256", "traces.json.gz"),
            ("frontend_sha256", "frontend.json.gz"),
        ):
            measurement[key] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        _write(folder / "measurement.json", measurement)
    _refresh(root)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        audit(root)
