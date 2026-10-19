"""CPU-only replay checks must reject rehashed semantic substitutions."""

from __future__ import annotations

import math

import pytest
from tools import benchmark_offline_edge_preparation as bench


@pytest.fixture
def cohort(tmp_path):
    source = tmp_path / "source/src/calibrex/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# synthetic source, never executed\r\n")
    manifest = bench.base._manifest(tmp_path / "source/src")
    bench.write(tmp_path / "source/manifest.json", manifest)
    tool_hashes = {}
    for name in bench.TOOLCHAIN:
        path = tmp_path / "toolchain" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic tool, never executed\n", encoding="utf-8")
        tool_hashes[name] = bench.base.sha(path)
    bench.write(tmp_path / "receipt.json", {"scope": "synthetic test only"})
    bench.write(tmp_path / "result.schema.json", {"type": "object"})
    protocol = {
        "order": list(bench.ORDER),
        "workers": bench.WORKERS,
        "estimate_gate": 2.0,
        "blas_threads": 1,
        "source_kind": "git-commit",
        "source_commit": "1" * 40,
        "base_git_commit": "1" * 40,
        "toolchain_sha256": tool_hashes,
        "source_manifest_sha256": bench.base.sha(tmp_path / "source/manifest.json"),
        "receipt_sha256": bench.base.sha(tmp_path / "receipt.json"),
        "result_schema_sha256": bench.base.sha(tmp_path / "result.schema.json"),
    }
    bench.write(tmp_path / "protocol.json", protocol)
    protocol_hash = bench.base.sha(tmp_path / "protocol.json")
    attempts = {"observations": [], "provenance": {"protocol_sha256": protocol_hash}}
    for index, label in enumerate(bench.ORDER):
        name = f"{index:02d}-{label}"
        folder = tmp_path / name
        folder.mkdir()
        bench.write(
            folder / "result.json",
            {
                "options": {"preparation_workers": bench.WORKERS[label]},
                "score": 1.0,
                "provenance": {"scope": "synthetic test only"},
            },
        )
        bench.write(
            folder / "measurement.json",
            {
                "provenance": {"protocol_sha256": protocol_hash},
                "result_sha256": bench.base.sha(folder / "result.json"),
                "estimate_seconds": 2.0 if label == "serial" else 1.0,
                "preparation": {
                    "seconds": 1.0 if label == "serial" else 0.5,
                    "frames": 48,
                    "points": 42506,
                    "capture_sha256": "0" * 64,
                    "dropped": [],
                },
                "score_calls": 429,
                "imported_source_sha256": manifest,
            },
        )
        (folder / "stdout.txt").write_text("", encoding="utf-8")
        (folder / "stderr.txt").write_text("", encoding="utf-8")
        attempts["observations"].append(
            {
                "worker": name,
                "exit_code": 0,
                "seconds": 3.0,
                "artifact_sha256": {path.name: bench.base.sha(path) for path in folder.iterdir()},
            }
        )
    bench.write(tmp_path / "attempts.json", attempts)
    return tmp_path


def _rebind(folder, worker):
    measurement_path = folder / worker / "measurement.json"
    measurement = bench.base.read(measurement_path)
    measurement["result_sha256"] = bench.base.sha(folder / worker / "result.json")
    bench.write(measurement_path, measurement)
    attempts = bench.base.read(folder / "attempts.json")
    for observation in attempts["observations"]:
        if observation["worker"] == worker:
            observation["artifact_sha256"] = {
                path.name: bench.base.sha(path) for path in (folder / worker).iterdir()
            }
    bench.write(folder / "attempts.json", attempts)


def test_complete_cpu_only_replay(cohort):
    result = bench.audit(cohort)
    assert result["passes_estimate_gate"] is True
    assert result["ratios"]["estimate"] == 2.0
    bench.write(cohort / "summary.json", result)
    assert bench.audit(cohort) == result


def test_git_verification_normalizes_only_source_line_endings(cohort, monkeypatch):
    source = cohort / "source/src/calibrex/__init__.py"
    files = {"src/calibrex/__init__.py": source.read_bytes().replace(b"\r\n", b"\n")}
    files.update({name: (cohort / "toolchain" / name).read_bytes() for name in bench.TOOLCHAIN})
    files[bench.RESULT_SCHEMA] = (cohort / "result.schema.json").read_bytes()
    monkeypatch.setattr(bench.base, "_git_files", lambda *args: files)
    assert bench.audit(cohort, verify_git=True)["passes_estimate_gate"] is True


@pytest.mark.parametrize("change", ["quality", "prepared", "work", "imports", "timing"])
def test_rehashed_semantic_changes_are_rejected(cohort, change):
    worker = "01-parallel"
    if change == "quality":
        path = cohort / worker / "result.json"
        result = bench.base.read(path)
        result["score"] = math.nextafter(result["score"], math.inf)
        bench.write(path, result)
    else:
        path = cohort / worker / "measurement.json"
        result = bench.base.read(path)
        if change == "prepared":
            result["preparation"]["capture_sha256"] = "f" * 64
        elif change == "work":
            result["score_calls"] -= 1
        elif change == "imports":
            result["imported_source_sha256"]["calibrex/__init__.py"] = "f" * 64
        else:
            result["estimate_seconds"] = result["preparation"]["seconds"] / 2
        bench.write(path, result)
    _rebind(cohort, worker)
    with pytest.raises(ValueError):
        bench.audit(cohort)


@pytest.mark.parametrize("change", ["missing", "order", "source", "summary"])
def test_missing_attempts_source_and_saved_summary_are_rejected(cohort, change):
    if change in {"missing", "order"}:
        attempts = bench.base.read(cohort / "attempts.json")
        if change == "missing":
            attempts["observations"].pop()
        else:
            attempts["observations"].reverse()
        bench.write(cohort / "attempts.json", attempts)
    elif change == "source":
        (cohort / "source/src/calibrex/__init__.py").write_text("# changed\n", encoding="utf-8")
    else:
        summary = bench.audit(cohort)
        summary["ratios"]["estimate"] += 0.1
        bench.write(cohort / "summary.json", summary)
    with pytest.raises(ValueError):
        bench.audit(cohort)
