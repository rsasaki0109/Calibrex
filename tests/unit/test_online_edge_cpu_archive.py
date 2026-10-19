"""All CPU cohorts audit without importing optional runtimes or archived code."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "archive,commit",
    [
        ("complete-evidence.tar.gz", "bb90e12f2fa2b811a10ad92dfe222aaffef7831f"),
        ("initial-image-jit.tar.gz", "ccf3f872663c724dc9bc1d00a4279f35853b6bab"),
    ],
)
def test_complete_cpu_archive_without_accelerator_or_core_imports(tmp_path, archive, commit):
    script = """
import runpy, sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numba', 'cupy', 'calibrex'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['audit', sys.argv[1], '--temporary-root', sys.argv[2]]
runpy.run_path('tools/audit_online_edge_cpu_archive.py', run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "docs/assets/online_edge_cpu" / archive),
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(result.stdout)
    assert value["passes_gate"] is True
    assert value["complete_results_exact"] is True
    assert value["prepared_snapshots_exact"] is True
    assert value["observation_count"] == 6
    assert value["provenance"]["source_kind"] == "git-commit"
    assert value["provenance"]["source_commit"] == commit
    assert value["provenance"]["indexed_files"] == 449
