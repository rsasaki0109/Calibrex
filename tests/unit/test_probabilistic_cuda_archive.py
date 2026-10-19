"""Portable, CPU-only replay of complete GPU performance evidence."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from tools.audit_probabilistic_refinement_ablation import audit

ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "docs/assets/probabilistic_cuda_refinement/comparison-v1.tar.gz"
)


def test_complete_publication_is_auditable_without_cupy(monkeypatch):
    native = builtins.__import__

    def forbid(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise AssertionError("archive audit must not import a GPU runtime")
        return native(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid)
    result = audit(ARCHIVE)
    assert result["workers"] == 6 and result["fits_per_worker"] == 8
    assert result["exact_complete_results"] and result["passes_gate"]
    assert result["prior_cuda_whole_ratio"] >= 2.0
    assert result["numpy_cuda_whole_ratio"] >= 2.0


def test_result_tamper_is_detected_even_after_inventory_is_rehashed(tmp_path):
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        files = {}
        for member in archive:
            stream = archive.extractfile(member)
            assert stream is not None
            files[member.name] = stream.read()
    name = "00-prior/results/full_uncertainty.seed-0000.yaml"
    files[name] += b"\n# modified after measurement\n"
    index = json.loads(files["publication-index.json"])
    index["files"][name] = hashlib.sha256(files[name]).hexdigest()
    files["publication-index.json"] = json.dumps(index).encode()
    path = tmp_path / "tampered.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        for key, data in files.items():
            entry = tarfile.TarInfo(key)
            entry.size = len(data)
            output.addfile(entry, io.BytesIO(data))
    with pytest.raises(ValueError, match="saved result differs"):
        audit(path)


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", r"C:\outside.json"])
def test_archive_rejects_unsafe_member_paths_without_extraction(tmp_path, name):
    path = tmp_path / "invalid.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        entry = tarfile.TarInfo(name)
        entry.size = 2
        output.addfile(entry, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="unexpected publication entry"):
        audit(path)
