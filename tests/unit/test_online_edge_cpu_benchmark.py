"""Small CPU evidence contracts, source binding and disk preflight."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from tools import benchmark_online_edge_cpu as bench

from calibrex.core.geometry import SE3
from calibrex.solvers import levinson_thrun_online_solver as online

ROOT = Path(__file__).resolve().parents[2]


def test_closed_cpu_execution_schema_preserves_native_quality():
    schema = bench.schemas(ROOT)["result"]
    execution = {field.name: 0 for field in fields(online.OnlineCpuExecution)}
    execution.update(backend="numba", runtime="test", kernel_sha256="a" * 64, policy=bench.POLICY)
    quality = online.LevinsonThrunOnlineSolver().solve([], SE3.identity()).as_dict()
    value = {
        "result": {**quality, "execution": execution},
        "provenance": {
            "created_at": "2026-10-19T00:00:00+00:00",
            "source_commit": "a" * 40,
            "base_git_commit": "a" * 40,
            "source_kind": "git-commit",
            "variant": "numba",
            "source_manifest_sha256": "b" * 64,
            "acquisition_receipt_sha256": "b" * 64,
            "launcher_sha256": "b" * 64,
            "protocol_sha256": "b" * 64,
            "scope": "test",
        },
    }
    jsonschema.validate(value, schema)
    for change in ("negative", "missing", "unknown"):
        changed = copy.deepcopy(value)
        if change == "negative":
            changed["result"]["execution"]["objective_evaluations"] = -1
        elif change == "missing":
            del changed["result"]["execution"]["kernel_sha256"]
        else:
            changed["result"]["execution"]["hidden_work"] = 1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(changed, schema)


def test_protocol_fixes_counterbalance_and_complete_toolchain():
    schema = bench.schemas(ROOT)["records"]
    value = {
        "kind": "protocol",
        "source_commit": "a" * 40,
        "base_git_commit": "a" * 40,
        "source_kind": "git-commit",
        "source_manifest_sha256": "b" * 64,
        "toolchain_sha256": dict.fromkeys(bench.TOOLCHAIN, "b" * 64),
        "receipt_sha256": "b" * 64,
        "order": list(bench.ORDER),
        "blas_threads": 1,
        "preparation_workers": 3,
        "warmups": 0,
        "host_reserved": False,
        "whole_worker_gate": 2.0,
    }
    value["provenance"] = bench.base._provenance(value)
    jsonschema.validate(value, schema)
    for key, item in (("order", ["numpy", "numba"]), ("toolchain_sha256", {})):
        changed = {**value, key: item}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(changed, schema)


def test_cpu_source_hashes_match_runtime_kernels():
    pytest.importorskip("numba")
    from calibrex.solvers.numba_online_edge_adapter import IMAGE_KERNEL_SHA256, KERNEL_SHA256

    assert bench.kernel_digests(ROOT) == {
        "_interpolate": KERNEL_SHA256,
        "_image_response": IMAGE_KERNEL_SHA256,
    }


def test_low_disk_space_stops_before_source_or_dataset_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.shutil, "disk_usage", lambda root: SimpleNamespace(free=0))
    output = tmp_path / "cohort"
    args = argparse.Namespace(
        output=output, fixture=Path("unused"), receipt=Path("unused"), commit="HEAD"
    )
    with pytest.raises(ValueError, match="20 GiB free"):
        bench.run(args)
    assert not output.exists()


def test_cpu_audit_helpers_do_not_import_runtime_or_execute_snapshot():
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'numba', 'calibrex'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
from tools import benchmark_online_edge_cpu as bench
assert len(bench.schemas(bench.ROOT)) == 3
assert len(bench.kernel_digests(bench.ROOT)) == 2
"""
    value = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert value.returncode == 0, value.stdout + value.stderr
