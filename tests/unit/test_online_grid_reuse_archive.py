"""Public CPU evidence replays without runtimes and rejects rebound alterations."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import jsonschema
import pytest
from tools import audit_online_grid_reuse_archive as archive_audit
from tools import benchmark_online_grid_reuse as bench
from tools.audit_online_edge_cuda_archive import audit_archive

ASSETS = bench.ROOT / "docs/assets/online_grid_reuse"
ARCHIVE = ASSETS / "complete-evidence.tar.gz"


def test_complete_archive_replays_without_runtime_or_core_imports(tmp_path):
    script = """
import runpy, sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numba', 'cupy', 'calibrex'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['audit', sys.argv[1], '--temporary-root', sys.argv[2]]
runpy.run_path('tools/audit_online_grid_reuse_archive.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ARCHIVE), str(tmp_path)],
        cwd=bench.ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(result.stdout)
    summary = bench.base.read(ASSETS / "summary.json")
    assert {key: item for key, item in value.items() if key != "provenance"} == {
        key: item for key, item in summary.items() if key != "provenance"
    }
    inventory = bench.base.read(ASSETS / "inventory.json")
    assert value["provenance"]["archive_sha256"] == inventory["sha256"] == bench.base.sha(ARCHIVE)
    assert value["provenance"]["indexed_files"] == inventory["indexed_files"] == 871
    assert value["provenance"]["grid_archive_auditor_sha256"] == bench.base.sha(
        bench.ROOT / "tools/audit_online_grid_reuse_archive.py"
    )


def test_semantic_tampering_is_rejected_even_after_artifact_hashes_are_rebound(tmp_path):
    def check(folder):
        original = bench.audit(folder)
        attempts_path = folder / "attempts.json"
        root = folder / "candidate/01-numba"
        paths = {name: root / f"{name}.json" for name in ("result", "measurement", "timing")}
        source_path = folder / "candidate/source/src/calibrex/solvers/numba_online_edge_adapter.py"
        snapshots = {
            path: path.read_bytes() for path in (*paths.values(), attempts_path, source_path)
        }
        cases = {
            "ulp": "complete quality",
            "work": "complete calibration work",
            "frame": "timed update population",
            "timing": "timing boundaries",
            "missing": "incomplete counterbalance",
            "source": "source",
        }
        for case, message in cases.items():
            try:
                attempts = bench.base.read(attempts_path)
                result, measured, timing = [bench.base.read(paths[name]) for name in paths]
                if case == "ulp":
                    step = result["result"]["timeline"][0]
                    step["center_objective"] = math.nextafter(step["center_objective"], math.inf)
                elif case == "work":
                    measured["point_rows"] -= 1
                elif case == "frame":
                    timing["updates"][0]["frame_id"] = "substituted-frame"
                elif case == "timing":
                    attempts["observations"][1]["seconds"] = 0.001
                elif case == "missing":
                    attempts["observations"].pop()
                else:
                    source_path.write_bytes(source_path.read_bytes() + b"\n# substituted source\n")
                bench.write(paths["result"], result)
                measured["result_sha256"] = bench.base.sha(paths["result"])
                bench.write(paths["measurement"], measured)
                timing["measurement_sha256"] = bench.base.sha(paths["measurement"])
                bench.write(paths["timing"], timing)
                for name, path in paths.items():
                    attempts["observations"][1]["artifact_sha256"][f"{name}.json"] = bench.base.sha(
                        path
                    )
                bench.write(attempts_path, attempts)
                with pytest.raises(ValueError, match=message):
                    bench.audit(folder)
            finally:
                for path, content in snapshots.items():
                    path.write_bytes(content)
        return original

    value = audit_archive(ARCHIVE, tmp_path, _audit=check)
    assert value["complete_quality_bits_equal"] is True


def test_published_summary_uses_the_closed_schema():
    jsonschema.validate(
        bench.base.read(ASSETS / "summary.json"), bench.base.read(ASSETS / "summary.schema.json")
    )
    assert bench.base.read(ASSETS / "summary.schema.json") == bench.record_schemas()["summary"]
    assert callable(archive_audit.audit)
