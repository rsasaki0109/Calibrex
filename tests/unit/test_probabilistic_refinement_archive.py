"""The complete comparison is independently auditable without CUDA."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_probabilistic_refinement_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/probabilistic_pose_batching"


def test_complete_comparison_audits_without_optional_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        assert not name.startswith(("cupy", "numba"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = audit(ROOT)
    assert result["workers"] == 4 and result["exact_complete_results"]
    assert result["earlier_dense"]["workers"] == 4


@pytest.mark.parametrize(
    "mutation", ["trace", "holdout", "count", "ratio", "capsule", "receipt", "earlier_ratio"]
)
def test_rebound_archive_rejects_changed_results_counts_ratios_and_sources(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    result_path, record_path = root / "00-current/result.json", root / "00-current/measurement.json"
    result = json.loads(result_path.read_bytes())
    record = json.loads(record_path.read_bytes())
    if mutation == "trace":
        result["trace"][1]["objective"] += 1e-12
    elif mutation == "holdout":
        result["final_holdout_evaluation"]["objective"] += 1e-12
    elif mutation == "count":
        record["trace_evaluations"] -= 1
    elif mutation == "ratio":
        path = root / "summary.json"
        summary = json.loads(path.read_bytes())
        summary["whole_speedup"] *= 2
        path.write_text(json.dumps(summary), encoding="utf-8")
    elif mutation == "capsule":
        path = root / "sources-current-manifest.json"
        value = json.loads(path.read_bytes())
        value[next(iter(value))] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        path = root / "protocol.json"
        value = json.loads(path.read_bytes())
        value["source_manifests"]["current"] = hashlib.sha256(
            (root / "sources-current-manifest.json").read_bytes()
        ).hexdigest()
        path.write_text(json.dumps(value), encoding="utf-8")
        path = root / "summary.json"
        value = json.loads(path.read_bytes())
        value["provenance"]["protocol_sha256"] = hashlib.sha256(
            (root / "protocol.json").read_bytes()
        ).hexdigest()
        path.write_text(json.dumps(value), encoding="utf-8")
    elif mutation == "receipt":
        path = root / "validation/receipt.json"
        value = json.loads(path.read_bytes())
        value["tests_passed"] = 999
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path = root / "earlier-dense-v2/summary.json"
        value = json.loads(path.read_bytes())
        value["whole_speedup"] *= 2
        path.write_text(json.dumps(value), encoding="utf-8")
        path = root / "earlier-dense-v2/publication-index.json"
        value = json.loads(path.read_bytes())
        value["files"]["summary.json"] = hashlib.sha256(
            (root / "earlier-dense-v2/summary.json").read_bytes()
        ).hexdigest()
        path.write_text(json.dumps(value), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    record["provenance"]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record), encoding="utf-8")
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    index["provenance"]["input_protocol_sha256"] = hashlib.sha256(
        (root / "protocol.json").read_bytes()
    ).hexdigest()
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p != index_path
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(
        ValueError, match="source capsule differs" if mutation == "capsule" else None
    ):
        audit(root)
