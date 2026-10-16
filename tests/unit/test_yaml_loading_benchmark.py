from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from tools.benchmark_yaml_loading import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/compiled_yaml_loading"


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    shutil.copytree(ARCHIVE, root)
    return root


def changed(root: Path, name: str, value: object) -> None:
    path = root / name
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())
    index = json.loads((root / "index.json").read_bytes())
    index["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "index.json").write_bytes((json.dumps(index, indent=2) + "\n").encode())


def test_complete_yaml_comparison_is_auditable_without_libyaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    summary = audit(ARCHIVE)
    assert summary["workers"] == 18
    assert set(summary["cases"]) == {"help", "validate", "refine"}
    assert all(case["exact_outputs"] for case in summary["cases"].values())


@pytest.mark.parametrize("failure", ["missing", "reordered", "failed"])
def test_no_attempt_can_be_dropped_after_inventory_rebinding(
    archive_copy: Path,
    failure: str,
) -> None:
    value = json.loads((archive_copy / "attempts.json").read_bytes())
    if failure == "missing":
        value.pop()
    elif failure == "reordered":
        value[0], value[1] = value[1], value[0]
    else:
        value[0]["exit_code"] = 1
    changed(archive_copy, "attempts.json", value)
    with pytest.raises(ValueError, match="incomplete, failed or reordered attempts"):
        audit(archive_copy)


def test_changed_objective_is_detected_after_inventory_rebinding(archive_copy: Path) -> None:
    name = "refine-00-current/result.json"
    value = json.loads((archive_copy / name).read_bytes())
    value["trace"][0]["objective"] += 0.001
    changed(archive_copy, name, value)
    with pytest.raises(ValueError, match="complete CLI outputs differ"):
        audit(archive_copy)


def test_changed_input_hash_is_not_normalized(archive_copy: Path) -> None:
    name = "refine-00-current/result.json"
    value = json.loads((archive_copy / name).read_bytes())
    value["provenance"]["correspondence_artifact_sha256"] = "0" * 64
    changed(archive_copy, name, value)
    with pytest.raises(ValueError, match="result input binding differs"):
        audit(archive_copy)


def test_imports_must_belong_to_the_source_capsule(archive_copy: Path) -> None:
    name = "refine-00-current/imports.json"
    value = json.loads((archive_copy / name).read_bytes())
    value["modules"]["calibrex.core.io"] = "C:/outside/io.py"
    changed(archive_copy, name, value)
    with pytest.raises(ValueError):
        audit(archive_copy)


def test_original_input_identity_must_match_the_fixture(archive_copy: Path) -> None:
    name = "input-transcoding.json"
    value = json.loads((archive_copy / name).read_bytes())
    value["original_sha256"]["correspondence"] = "0" * 64
    changed(archive_copy, name, value)
    with pytest.raises(ValueError, match="YAML fixture conversion binding differs"):
        audit(archive_copy)


def test_reported_speed_must_use_all_attempts(archive_copy: Path) -> None:
    name = "summary.json"
    value = json.loads((archive_copy / name).read_bytes())
    value["cases"]["refine"]["speedup"] += 0.1
    changed(archive_copy, name, value)
    with pytest.raises(ValueError, match="reported timing differs from all workers"):
        audit(archive_copy)
