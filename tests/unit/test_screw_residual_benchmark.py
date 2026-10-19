"""The complete-fit audit rejects altered or partial evidence without a solver."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from tools import audit_screw_residual_archive as archive
from tools import benchmark_screw_residual_evaluation as bench


def _put(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _cohort(folder: Path) -> None:
    for variant in ("baseline", "candidate"):
        source = folder / variant / "src/calibrex/__init__.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"# minimal audit fixture\n")
        _put(folder / variant / "manifest.json", {"calibrex/__init__.py": bench.sha(source)})
    _put(folder / "geometry.json", {"provenance": {"scope": "test"}})
    _put(folder / "result.schema.json", {"type": "object"})
    for path in bench.TOOLS:
        target = folder / "toolchain" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"raise RuntimeError('archive tool executed')\n")
    protocol = {
        "order": list(bench.ORDER),
        "seeds": list(bench.SEEDS),
        "gate_api_ratio": 2.0,
        "blas_threads": 1,
        "commits": {"baseline": "a" * 40, "candidate": "b" * 40},
        "manifests": {
            v: bench.sha(folder / v / "manifest.json") for v in ("baseline", "candidate")
        },
        "tools": {p: bench.sha(folder / "toolchain" / p) for p in bench.TOOLS},
        "host_reserved": False,
        "warmups": 0,
        "geometry_sha256": bench.sha(folder / "geometry.json"),
        "result_schema_sha256": bench.sha(folder / "result.schema.json"),
        "provenance": {"created_at": "2026-10-20T00:00:00Z", "scope": "test, no speed claim"},
    }
    _put(folder / "protocol.json", protocol)
    observations = []
    for case in ("joint", "geometry"):
        for index, variant in enumerate(bench.ORDER):
            name = f"{case}-{index:02d}-{variant}"
            directory = folder / name
            origin = {
                "created_at": "2026-10-20T00:00:00Z",
                "protocol_sha256": bench.sha(folder / "protocol.json"),
                "source_commit": protocol["commits"][variant],
            }
            value = {"value": index // 2, "provenance": origin}
            for filename in ("result.json", "fit-input.json", "fit-result.json"):
                _put(directory / filename, value)
            _put(
                directory / "measurement.json",
                {
                    "api_seconds": 6.0 if variant == "baseline" else 2.0,
                    "fit_seconds": 1.5,
                    "fit_status": "converged",
                    "iterations": 2,
                    "point_to_plane_count": 480,
                    **dict.fromkeys(
                        ("result_quality_sha256", "input_quality_sha256", "fit_quality_sha256"),
                        bench.quality(value),
                    ),
                    "imported": json.loads((folder / variant / "manifest.json").read_bytes()),
                    "provenance": origin,
                },
            )
            (folder / f"{name}.log").write_bytes(b"complete\n")
            observations.append(
                {
                    "case": case,
                    "variant": variant,
                    "exit_code": 0,
                    "seconds": 10.0 if variant == "baseline" else 4.0,
                    "files": {p.name: bench.sha(p) for p in directory.glob("*.json")},
                    "log_sha256": bench.sha(folder / f"{name}.log"),
                }
            )
    _put(
        folder / "attempts.json",
        {
            "observations": observations,
            "provenance": {"created_at": "2026-10-20T00:00:00Z"},
        },
    )
    _put(folder / "summary.json", bench.audit(folder))


def test_complete_audit_recomputes_paired_ratios(tmp_path: Path) -> None:
    _cohort(tmp_path)
    result = bench.audit(tmp_path)
    for case in result["cases"].values():
        assert case["paired_median_api_ratio"] == 3.0
        assert case["paired_median_fresh_worker_ratio"] == 2.5
        assert case["passes_api_gate"]


@pytest.mark.parametrize("change", ["partial", "failed", "result", "log", "source", "coverage"])
def test_complete_audit_rejects_tampering(tmp_path: Path, change: str) -> None:
    _cohort(tmp_path)
    path = tmp_path / "attempts.json"
    attempts = json.loads(path.read_bytes())
    if change == "partial":
        attempts["observations"].pop()
    elif change == "failed":
        attempts["observations"][0]["exit_code"] = 1
    elif change == "coverage":
        attempts["observations"][0]["files"].pop("measurement.json")
    elif change == "result":
        _put(tmp_path / "joint-00-baseline/result.json", {"value": 999})
    elif change == "log":
        (tmp_path / "joint-00-baseline.log").write_bytes(b"altered\n")
    else:
        (tmp_path / "baseline/src/calibrex/__init__.py").write_bytes(b"# altered\n")
    _put(path, attempts)
    with pytest.raises(ValueError):
        bench.audit(tmp_path)


def test_quality_excludes_provenance_only() -> None:
    assert bench.quality({"x": 0.0, "provenance": {"a": 1}}) == bench.quality({"x": 0.0})
    assert bench.quality({"x": 0.0}) != bench.quality({"x": -0.0})


def _bundle(folder: Path, target: Path) -> None:
    _put(folder / "schemas/records.schema.json", archive.record_schema())
    # This source must never be executed by an evidence audit.
    (folder / "untrusted.py").write_text(
        "raise RuntimeError('archive executed')\n", encoding="utf-8"
    )
    paths = [p for p in folder.rglob("*") if p.is_file()]
    _put(
        folder / "index.json",
        {
            "files": {
                p.relative_to(folder).as_posix(): {"size": p.stat().st_size, "sha256": bench.sha(p)}
                for p in paths
            },
            "provenance": {"protocol_sha256": bench.sha(folder / "protocol.json")},
        },
    )
    with tarfile.open(target, "w:gz") as bundle:
        for path in [*paths, folder / "index.json"]:
            bundle.add(path, arcname=path.relative_to(folder).as_posix(), recursive=False)


def test_indexed_archive_replays_without_executing_source(tmp_path: Path) -> None:
    folder = tmp_path / "cohort"
    _cohort(folder)
    archive.validate_records(folder)
    target = tmp_path / "evidence.tar.gz"
    _bundle(folder, target)
    result = archive.audit(target, tmp_path / "audit")
    assert result["cases"]["joint"]["paired_median_api_ratio"] == 3.0
    assert result["provenance"]["archive_sha256"] == bench.sha(target)


def test_archive_rejects_forged_summary_with_matching_index(tmp_path: Path) -> None:
    folder = tmp_path / "cohort"
    _cohort(folder)
    summary = json.loads((folder / "summary.json").read_bytes())
    summary["cases"]["joint"]["paired_median_api_ratio"] = 5.0
    _put(folder / "summary.json", summary)
    target = tmp_path / "evidence.tar.gz"
    _bundle(folder, target)
    with pytest.raises(ValueError, match="published summary differs"):
        archive.audit(target, tmp_path / "audit")
