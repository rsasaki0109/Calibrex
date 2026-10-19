"""The complete-fit audit rejects altered or partial evidence without a solver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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
    protocol = {
        "order": list(bench.ORDER),
        "seeds": list(bench.SEEDS),
        "gate_api_ratio": 2.0,
        "blas_threads": 1,
        "commits": {"baseline": "a" * 40, "candidate": "b" * 40},
        "manifests": {
            v: bench.sha(folder / v / "manifest.json") for v in ("baseline", "candidate")
        },
        "tools": {},
        "geometry_sha256": bench.sha(folder / "geometry.json"),
        "result_schema_sha256": bench.sha(folder / "result.schema.json"),
        "provenance": {"scope": "small complete-cohort test, no speed claim"},
    }
    _put(folder / "protocol.json", protocol)
    observations = []
    for case in ("joint", "geometry"):
        for index, variant in enumerate(bench.ORDER):
            name = f"{case}-{index:02d}-{variant}"
            directory = folder / name
            origin = {
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
    _put(folder / "attempts.json", {"observations": observations, "provenance": {}})


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
