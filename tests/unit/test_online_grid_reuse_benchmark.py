"""The maintained-commit comparison retains exact work and failed attempts."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import jsonschema
import pytest
from tools import benchmark_online_grid_reuse as bench


def test_summary_schema_rejects_missing_clocks_work_and_quality():
    metrics = dict.fromkeys(bench.METRICS, 2.0)
    value = {
        "medians_seconds": {"baseline": metrics, "candidate": metrics},
        "ratios": metrics,
        "passes_solve_gate": True,
        "observations": 6,
        "complete_quality_bits_equal": True,
        "prepared_bits_equal": True,
        "quality_sha256": "a" * 64,
        "provenance": {},
    }
    bench._validate(value, "summary")
    for change in ("clock", "quality", "work", "unknown"):
        altered = copy.deepcopy(value)
        if change == "clock":
            del altered["ratios"]["whole_worker"]
        elif change == "quality":
            altered["complete_quality_bits_equal"] = False
        elif change == "work":
            altered["observations"] = 5
        else:
            altered["selected_warmups"] = 1
        with pytest.raises(jsonschema.ValidationError):
            bench._validate(altered, "summary")


def test_timing_keeps_first_update_and_all_thirty_frame_ids():
    value = {
        "solve_seconds": [5.0],
        "updates": [{"frame_id": str(index), "seconds": 0.01} for index in range(30)],
        "measurement_sha256": "a" * 64,
        "provenance": {},
    }
    bench._validate(value, "timing")
    value["updates"].pop(0)
    with pytest.raises(jsonschema.ValidationError):
        bench._validate(value, "timing")


def test_quality_hash_preserves_signed_zero_and_one_ulp():
    first = {"score": 0.0, "execution": {"runtime": "old"}}
    second = {"score": 0.0, "execution": {"runtime": "new"}}
    assert bench._quality_digest(first) == bench._quality_digest(second)
    second["score"] = -0.0
    assert bench._quality_digest(first) != bench._quality_digest(second)
    first["score"], second["score"] = 1.0, 1.0000000000000002
    assert bench._quality_digest(first) != bench._quality_digest(second)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_timings_are_rejected(value):
    with pytest.raises(ValueError, match="timing"):
        bench._finite(value)


@pytest.mark.parametrize(
    "failure,expected", [(OSError("launch"), -2), (subprocess.TimeoutExpired("worker", 600), -1)]
)
def test_failed_launches_and_timeouts_keep_their_logs(tmp_path, monkeypatch, failure, expected):
    def fail(*args, **kwargs):
        kwargs["stderr"].write(b"retained failure\n")
        raise failure

    monkeypatch.setattr(bench.subprocess, "run", fail)
    assert bench.execute_worker(["worker"], {}, tmp_path) == expected
    assert (tmp_path / "stderr.txt").read_bytes() == b"retained failure\n"
    assert (tmp_path / "stdout.txt").exists()


def test_low_space_stops_before_freezing_or_reading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    args = argparse.Namespace(output=tmp_path / "run")
    with pytest.raises(ValueError, match="20 GiB"):
        bench.run(args)
    assert not args.output.exists()


def test_cpu_audit_and_runtime_identity_do_not_import_optional_backends(tmp_path):
    source = tmp_path / "src/calibrex/solvers"
    source.mkdir(parents=True)
    target = source / "numba_online_edge_adapter.py"
    for version in ("v0.1", "v0.2"):
        target.write_text(f'runtime = f"calibrex.numba_online_edge/{version};numba={{version}}"\n')
        assert bench.cpu.runtime_prefix(tmp_path) == f"calibrex.numba_online_edge/{version};numba="
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'numba', 'calibrex'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
from tools import benchmark_online_grid_reuse as bench
assert len(bench.record_schemas()) == 5
assert bench.cpu.runtime_prefix(bench.ROOT).startswith('calibrex.numba_online_edge/')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=bench.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_partial_or_reordered_attempts_are_rejected(tmp_path):
    source = "a" * 40
    protocol = {
        "commits": dict.fromkeys(("baseline", "candidate"), source),
        "source_protocol_sha256": dict.fromkeys(("baseline", "candidate"), "b" * 64),
        "toolchain_sha256": dict.fromkeys(bench.TOOLCHAIN, "b" * 64),
        "schema_sha256": {},
        "receipt_sha256": "b" * 64,
        "order": list(bench.ORDER),
        "blas_threads": 1,
        "preparation_workers": 3,
        "host_reserved": False,
        "warmups": 0,
        "solve_gate": 2.0,
        "provenance": {},
    }
    bench.write(tmp_path / "protocol.json", protocol)
    attempts = {
        "observations": [],
        "provenance": {"protocol_sha256": bench.base.sha(tmp_path / "protocol.json")},
    }
    bench.write(tmp_path / "attempts.json", attempts)
    with pytest.raises(ValueError, match="incomplete counterbalance"):
        bench.audit(tmp_path)
    attempts["provenance"]["protocol_sha256"] = "c" * 64
    bench.write(tmp_path / "attempts.json", attempts)
    with pytest.raises(ValueError, match="attempt lineage"):
        bench.audit(tmp_path)


def test_all_record_schemas_are_valid_json_schemas():
    for schema in bench.record_schemas().values():
        jsonschema.Draft202012Validator.check_schema(json.loads(json.dumps(schema)))
