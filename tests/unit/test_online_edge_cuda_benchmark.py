"""CPU-only schema and control checks for the complete-worker benchmark."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import tarfile
from dataclasses import fields
from pathlib import Path

import jsonschema
import pytest
from tests.unit.test_levinson_thrun_online_solver import _frames_for_truths
from tools import benchmark_online_edge_cuda as bench

from calibrex.core.geometry import SE3
from calibrex.solvers import levinson_thrun_online_solver as online

SCHEMAS = Path(__file__).resolve().parents[2] / "tools/schemas"


def test_git_archive_normalizes_text_and_preserves_binary(monkeypatch):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, data in {
            "src/calibrex/example.py": b"s = '\\r\\n'\r\n",
            "src/calibrex/example.json": b"{\r\n}\r\n",
            "src/calibrex/example.bin": b"\x00\r\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *args: payload.getvalue())
    value = bench._git_files("test", ("src/calibrex",))
    assert value["src/calibrex/example.py"] == b"s = '\\r\\n'\n"
    assert value["src/calibrex/example.json"] == b"{\n}\n"
    assert value["src/calibrex/example.bin"] == b"\x00\r\n"


def _artifact():
    result = online.LevinsonThrunOnlineSolver().solve([], SE3.identity()).as_dict()
    execution = {field.name: 0 for field in fields(online.OnlineEdgeExecution)}
    execution.update(
        backend="cuda",
        runtime="test",
        kernel_sha256="a" * 64,
        capture_sha256="b" * 64,
        policy="test",
    )
    result["execution"] = execution
    return {
        "result": result,
        "provenance": {
            "created_at": "2026-10-19T00:00:00+00:00",
            "source_commit": "c" * 40,
            "base_git_commit": "c" * 40,
            "source_kind": "git-commit",
            "source_manifest_sha256": "a" * 64,
            "acquisition_receipt_sha256": "a" * 64,
            "launcher_sha256": "a" * 64,
            "scope": "unit",
            "protocol_sha256": "a" * 64,
            "variant": "cuda",
        },
    }


def test_execution_schema_rejects_invalid_or_hidden_counters():
    schema = json.loads((SCHEMAS / "online_edge_cuda_result.schema.json").read_bytes())
    original = _artifact()
    jsonschema.validate(original, schema)
    for change in ("negative", "missing", "unknown"):
        value = copy.deepcopy(original)
        execution = value["result"]["execution"]
        if change == "negative":
            execution["grid_calls"] = -1
        elif change == "missing":
            del execution["kernel_sha256"]
        else:
            execution["hidden_fallback"] = 1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(value, schema)


def test_comparison_preserves_every_quality_field_and_input():
    value = _artifact()["result"]
    original = copy.deepcopy(value)
    compared = bench._comparable(value)
    assert value == original
    assert "execution" not in compared
    assert compared == {key: item for key, item in original.items() if key != "execution"}
    value["reason"] = "changed"
    assert bench._comparable(value) != compared


def test_working_copy_cannot_claim_a_committed_source():
    schema = json.loads((SCHEMAS / "online_edge_cuda_result.schema.json").read_bytes())
    value = _artifact()
    value["provenance"]["source_kind"] = "working-copy"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)
    value["provenance"]["source_commit"] = None
    jsonschema.validate(value, schema)
    value["provenance"]["source_kind"] = "git-commit"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)


def test_record_schema_rejects_nonpositive_duration():
    schema = json.loads((SCHEMAS / "online_edge_cuda_records.schema.json").read_bytes())
    value = {
        "kind": "attempts",
        "observations": [{"variant": "cuda", "worker": "02-cuda", "seconds": 1.0, "exit_code": 0}],
        "provenance": {
            "source_commit": "a" * 40,
            "base_git_commit": "a" * 40,
            "source_kind": "git-commit",
            "created_at": "test",
        },
    }
    jsonschema.validate(value, schema)
    value["observations"][0]["seconds"] = 0.0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)


def test_fixture_requires_complete_named_subset(tmp_path):
    receipt = {
        "complete": True,
        "frame_indices": list(range(15, 486, 10)),
        "size": 11320688364,
        "etag": "e933e363d73c5e057e18e6890463b47a-1350",
        "files": {"../escaped.bin": {"size": 0, "sha256": "a" * 64}},
    }
    with pytest.raises(ValueError, match="pinned 48-pair"):
        bench.verify_fixture(tmp_path, receipt)


@pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="requires numba")
def test_compiled_cpu_control_keeps_complete_native_step(monkeypatch):
    from tools.online_edge_fused_control import NativeRawGridControl, install

    frames = _frames_for_truths([SE3.identity()] * 3)
    options = online.LevinsonThrunOnlineOptions(window_size=3)
    expected = online._evaluate_online_step(frames, SE3.identity(), options)
    original = online._project_response
    monkeypatch.setattr(online, "_project_response", original)
    install(online)
    assert online._evaluate_online_step(frames, SE3.identity(), options) == expected
    scorer = NativeRawGridControl(online)
    assert online._evaluate_online_step(frames, SE3.identity(), options, scorer=scorer) == expected
    assert scorer.pose_evaluations == 728
    assert scorer.point_rows == 728 * sum(len(frame.lidar_points) for frame in frames)
