"""CPU-only retained evidence audit and safe archive handling."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from tools.audit_online_edge_cuda_archive import _local_references, audit_archive

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/assets/online_edge_cuda/complete-evidence.tar.gz"


def test_complete_archive_audits_without_cuda_or_core_imports(tmp_path):
    script = """
import runpy, sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'calibrex'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['audit', sys.argv[1], '--temporary-root', sys.argv[2]]
runpy.run_path('tools/audit_online_edge_cuda_archive.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ARCHIVE), str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(result.stdout)
    assert value["passes_gate"] is True
    assert value["observation_count"] == 6
    assert value["provenance"]["source_commit"] is None
    assert value["provenance"]["source_kind"] == "working-copy"
    assert value["provenance"]["indexed_files"] == 445


@pytest.mark.parametrize("name", ["../escaped", "C:/escaped", "unsafe\\escaped"])
def test_archive_rejects_unsafe_paths(tmp_path, name):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        member.size = 1
        stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        audit_archive(archive, tmp_path / "audit")


def test_schema_references_cannot_read_external_resources():
    with pytest.raises(ValueError, match="local references"):
        _local_references({"properties": {"x": {"$ref": "https://invalid.example/schema"}}})
