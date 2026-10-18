from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
from jsonschema import ValidationError
from tools.benchmark_radar_doppler_batches import audit, sha

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/radar_doppler_ordered_batches"


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def write(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def reindex(folder: Path) -> None:
    value = read(folder / "index.json")
    value["files"] = {
        p.relative_to(folder).as_posix(): {"sha256": sha(p), "size_bytes": p.stat().st_size}
        for p in folder.rglob("*")
        if p.is_file() and p.name != "index.json"
    }
    value["provenance"]["protocol_sha256"] = sha(folder / "protocol.json")
    write(folder / "index.json", value)


def clone(tmp_path: Path) -> Path:
    return Path(shutil.copytree(ARCHIVE, tmp_path / "archive"))


def test_complete_published_workflow_and_every_irls_system_are_exact() -> None:
    report = audit(ARCHIVE)
    assert report == read(ARCHIVE / "summary.json")
    assert report["payload"]["verified_runs"] == 6
    assert report["payload"]["irls_iterations"] == 17658
    assert report["payload"]["whole_speedup"] > 2.0


@pytest.mark.parametrize("field", ["options", "returns_per_scan", "repetitions"])
def test_rehashed_protocol_cannot_reduce_default_work(tmp_path: Path, field: str) -> None:
    folder = clone(tmp_path)
    protocol = read(folder / "protocol.json")
    if field == "options":
        protocol["payload"][field]["max_iterations"] = 1
    else:
        protocol["payload"][field] = 1
    write(folder / "protocol.json", protocol)
    reindex(folder)
    with pytest.raises((ValueError, ValidationError)):
        audit(folder)


@pytest.mark.parametrize("change", ["scan", "history", "imports", "revision", "counts"])
def test_worker_hashes_do_not_replace_semantic_checks(tmp_path: Path, change: str) -> None:
    folder = clone(tmp_path)
    worker = folder / "01-prior"
    measurement = read(worker / "measurement.json")
    payload = measurement["payload"]
    if change == "scan":
        value = json.loads(gzip.decompress((worker / "result.json.gz").read_bytes()))
        value["payload"]["scans"][0]["measurement_id"] = "different-scan"
        (worker / "result.json.gz").write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
        payload["result_sha256"] = sha(worker / "result.json.gz")
    elif change == "history":
        raw = gzip.decompress((worker / "history.bin.gz").read_bytes()) + b"I"
        (worker / "history.bin.gz").write_bytes(gzip.compress(raw, mtime=0))
        payload["history_sha256"] = sha(worker / "history.bin.gz")
        payload["history_raw_sha256"] = hashlib.sha256(raw).hexdigest()
    elif change == "imports":
        payload["imports"]["calibrex"]["sha256"] = "0" * 64
    elif change == "revision":
        measurement["provenance"]["git_commit"] = "0" * 40
        measurement["provenance"]["git_commit_short"] = "0000000"
    else:
        payload["returns"] -= 256
    write(worker / "measurement.json", measurement)
    reindex(folder)
    with pytest.raises(ValueError):
        audit(folder)


def test_rehashed_capsule_cannot_change_an_unrelated_runtime_module(tmp_path: Path) -> None:
    folder = clone(tmp_path)
    path = folder / "sources-current.tar.gz"
    members = []
    with tarfile.open(path) as bundle:
        for member in bundle.getmembers():
            if member.isfile():
                stream = bundle.extractfile(member)
                assert stream is not None
                data = stream.read()
                if member.name == "src/calibrex/__init__.py":
                    data += b"\n# unrelated change\n"
                members.append((member.name, data))
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    manifest = read(folder / "sources-current-manifest.json")
    manifest["calibrex/__init__.py"] = hashlib.sha256(
        dict(members)["src/calibrex/__init__.py"]
    ).hexdigest()
    write(folder / "sources-current-manifest.json", manifest)
    protocol = read(folder / "protocol.json")
    for name in ("sources-current.tar.gz", "sources-current-manifest.json"):
        protocol["provenance"]["files"][name] = sha(folder / name)
    write(folder / "protocol.json", protocol)
    reindex(folder)
    with pytest.raises(ValueError, match="runtime source changes"):
        audit(folder)


def test_rehashed_summary_cannot_overstate_the_speedup(tmp_path: Path) -> None:
    folder = clone(tmp_path)
    summary = read(folder / "summary.json")
    summary["payload"]["whole_speedup"] *= 10
    write(folder / "summary.json", summary)
    reindex(folder)
    with pytest.raises(ValueError, match="saved summary"):
        audit(folder)


def test_published_index_covers_validation_and_log_bytes(tmp_path: Path) -> None:
    folder = clone(tmp_path)
    (folder / "01-prior/stdout.txt").write_bytes(b"unrecorded output")
    with pytest.raises(ValueError, match="published file bytes"):
        audit(folder)
