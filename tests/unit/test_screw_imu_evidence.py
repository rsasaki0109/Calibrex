"""Complete timing outcomes, frozen APIs and safe CPU evidence replay."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from tools import benchmark_screw_imu_evaluation as evidence

ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/screw_imu_evaluation"
ARCHIVE = ASSETS / "complete-evidence.tar.gz"


def test_complete_archive_replays_the_target_miss_and_every_full_outcome(tmp_path):
    receipt = json.loads((ASSETS / "receipt.json").read_bytes())
    assert ARCHIVE.stat().st_size == receipt["bytes"]
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == receipt["sha256"]
    result = evidence.audit_archive(ARCHIVE, temporary_directory=tmp_path)
    assert result["all_complete_inputs_and_outcomes_equal"]
    assert result["meets_2x_api_target"] is False
    assert result["paired_median_api_ratio"] == 1.4524178953950906
    assert [row["seed"] for row in result["pairs"]] == list(evidence.previous.SEEDS)


def test_only_one_runtime_module_changes_and_public_signatures_are_identical():
    with tarfile.open(ARCHIVE) as archive:
        def read(name):
            return archive.extractfile(name).read()

        manifests = {variant: json.loads(read(variant + "/manifest.json"))
                     for variant in ("baseline", "candidate")}
        assert len(manifests["baseline"]) == len(manifests["candidate"]) == 401
        assert manifests["baseline"].keys() == manifests["candidate"].keys()
        changed = [name for name in manifests["baseline"]
                   if manifests["baseline"][name] != manifests["candidate"][name]]
        assert changed == ["calibrex/core/continuous_time_sparse.py"]

        def signatures(variant):
            tree = ast.parse(read(variant + "/src/" + changed[0]))
            return {node.name: ast.dump(node.args) for node in tree.body
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")}

        assert signatures("baseline") == signatures("candidate")
        protocol = json.loads(read("protocol.json"))
        assert protocol["commits"] == {
            "baseline": "0f76b0ec6bda2d975ea58d4b4bbd52d174892dae",
            "candidate": "6db6df77403461e86813c5c6f207cd7c9943a24e",
        }


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "a\\escape", "a//b", "CON.txt"])
def test_reader_rejects_unsafe_names_before_creating_files(name, tmp_path):
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for member_name, raw in (("index.json", b'{"files": {}}'), (name, b"data")):
            member = tarfile.TarInfo(member_name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    destination = tmp_path / "unpacked"
    destination.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        evidence._unpack_archive(path, destination)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("mode", ["duplicate", "missing", "digest"])
def test_reader_rejects_incomplete_or_changed_payloads(mode, tmp_path):
    path = tmp_path / "invalid.tar.gz"
    payload = b"original"
    index = {"files": {"payload.txt": {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }}}
    entries = [("index.json", json.dumps(index).encode())]
    if mode != "missing":
        entries.append(("payload.txt", b"modified" if mode == "digest" else payload))
    if mode == "duplicate":
        entries.append(entries[-1])
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in entries:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    destination = tmp_path / "unpacked"
    destination.mkdir()
    with pytest.raises(ValueError, match=r"duplicate|coverage|differs"):
        evidence._unpack_archive(path, destination)
