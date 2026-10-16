from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_lazy_runtime_benchmark import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/lazy_runtime_loading"


def _changed(tmp_path: Path, name: str, mutate) -> Path:
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    path = root / name
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_text(json.dumps(value))
    index_path = root / "index.json"
    index = json.loads(index_path.read_bytes())
    index["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index))
    return root


def test_complete_cli_archive_verifies_all_eighteen_workers() -> None:
    assert audit(ARCHIVE)["workers"] == 18


def test_audit_recomputes_speedup_even_when_inventory_is_updated(tmp_path: Path) -> None:
    def change(value: dict) -> None:
        value["cases"]["refine"]["speedup"] *= 2

    root = _changed(tmp_path, "summary.json", change)
    with pytest.raises(ValueError, match="reported timing"):
        audit(root)


def test_audit_checks_complete_trace_even_when_inventory_is_updated(tmp_path: Path) -> None:
    def change(value: dict) -> None:
        value["trace"][0]["objective"] += 0.25

    root = _changed(tmp_path, "refine-00-current/result.json", change)
    with pytest.raises(ValueError, match="complete CLI outputs"):
        audit(root)


def test_audit_rejects_imports_from_another_source_variant(tmp_path: Path) -> None:
    def change(value: dict) -> None:
        value["source"] = value["source"].replace("sources-current", "sources-prior")

    root = _changed(tmp_path, "refine-00-current/imports.json", change)
    with pytest.raises(ValueError, match="source variant"):
        audit(root)
