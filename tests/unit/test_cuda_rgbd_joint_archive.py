"""Independent CPU/CUDA workload, timing and coherent evidence corruption checks."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import jsonschema
import pytest
from tools.audit_cuda_rgbd_joint import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/cuda_rgbd_joint"


def _load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def _write(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def _rehash(root):
    index = _load(root / "index.json")
    for name in index["files"]:
        path = root / name
        index["files"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    _write(root / "index.json", index)


def test_complete_published_cpu_cuda_work_and_results_are_exact():
    report = audit(ARCHIVE)
    assert report["complete_result_and_histories_exact"] and report["complete_frontend_exact"]
    assert len(report["attempts"]) == 6


@pytest.mark.parametrize(
    "change", ["omit", "timing", "result", "trace", "frontend", "execution", "config", "imports"]
)
def test_coherently_rehashed_corruption_is_rejected(tmp_path, change):
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    if change in ("omit", "timing"):
        value = _load(root / "attempts.json")
        if change == "omit":
            value["attempts"].pop()
        else:
            value["attempts"][0]["optimizer_seconds"] *= 0.5
        _write(root / "attempts.json", value)
    elif change == "config":
        path = root / "config-cuda.yaml"
        path.write_bytes(
            path.read_bytes().replace(
                b"query_points_per_frame: 20000", b"query_points_per_frame: 10000"
            )
        )
    else:
        folder = root / "01-cuda"
        files = {
            name: _load(folder / name)
            for name in (
                "measurement.json",
                "execution.json",
                "result.json.gz",
                "traces.json.gz",
                "frontend.json.gz",
            )
        }
        if change == "result":
            files["result.json.gz"]["payload"]["train_rmse"] += 0.1
        elif change == "trace":
            files["traces.json.gz"]["payload"][0]["arrays"][0]["sha256"] = "0" * 64
        elif change == "frontend":
            files["frontend.json.gz"]["payload"][1]["count"] -= 1
        elif change == "execution":
            for name in ("measurement.json", "execution.json"):
                files[name]["execution"]["linearizations"] = 1
        else:
            for name in (
                "measurement.json",
                "result.json.gz",
                "traces.json.gz",
                "frontend.json.gz",
            ):
                files[name]["provenance"]["imports"]["calibrex.core.io"] = "outside.py"
        for name in ("execution.json", "result.json.gz", "traces.json.gz", "frontend.json.gz"):
            _write(folder / name, files[name])
        for key, name in (
            ("result_sha256", "result.json.gz"),
            ("traces_sha256", "traces.json.gz"),
            ("frontend_sha256", "frontend.json.gz"),
        ):
            files["measurement.json"][key] = hashlib.sha256(
                (folder / name).read_bytes()
            ).hexdigest()
        _write(folder / "measurement.json", files["measurement.json"])
    _rehash(root)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        audit(root)
