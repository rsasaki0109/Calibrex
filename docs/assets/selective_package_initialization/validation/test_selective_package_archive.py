from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_lazy_runtime_benchmark import SELECTIVE_INITIALIZATION_CHANGES, audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/selective_package_initialization"


def checked(root: Path) -> dict:
    return audit(root, expected_runtime_changes=SELECTIVE_INITIALIZATION_CHANGES)


def write(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def rebind(root: Path, *names: str) -> None:
    path = root / "index.json"
    value = json.loads(path.read_bytes())
    for name in names:
        value["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    write(path, value)


@pytest.fixture
def archive_copy(tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def test_complete_cli_results_and_parser_definitions_match() -> None:
    report = checked(ARCHIVE)
    assert report == json.loads((ARCHIVE / "summary.json").read_bytes())
    assert report["workers"] == 18
    assert all(case["exact_outputs"] for case in report["cases"].values())


def test_every_archive_byte_is_indexed() -> None:
    assert json.loads((ARCHIVE / "index.json").read_bytes())["files"] == {
        p.relative_to(ARCHIVE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARCHIVE.rglob("*")
        if p.is_file() and p.name != "index.json"
    }


@pytest.mark.parametrize("part", ["summary", "trace", "identity", "result_commit", "parser"])
def test_mutations_fail_after_rebinding_the_byte_index(archive_copy: Path, part: str) -> None:
    if part == "summary":
        name, error = "summary.json", "reported timing"
        value = json.loads((archive_copy / name).read_bytes())
        value["cases"]["help"]["speedup"] *= 2
    elif part == "trace":
        name, error = "refine-00-current/result.json", "complete CLI outputs"
        value = json.loads((archive_copy / name).read_bytes())
        value["trace"][0]["objective"] += 0.25
    elif part == "identity":
        name, error = "source-identities.json", "actual full/short source identity"
        value = json.loads((archive_copy / name).read_bytes())
        value["variants"]["current"]["git_commit_short"] = "f" * 8
    elif part == "result_commit":
        name, error = "refine-00-current/result.json", "actual source commit"
        value = json.loads((archive_copy / name).read_bytes())
        value["provenance"]["git_commit"] = "f" * 8
    else:
        name, error = "parser-current.json.gz", "complete parser definitions"
        path = archive_copy / name
        rows = json.loads(gzip.decompress(path.read_bytes()))
        rows[0]["actions"][0]["default"] = "changed"
        path.write_bytes(gzip.compress((json.dumps(rows) + "\n").encode(), mtime=0))
        parity_path = archive_copy / "parser-parity.json"
        value = json.loads(parity_path.read_bytes())
        value["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        write(parity_path, value)
        rebind(archive_copy, name, "parser-parity.json")
        with pytest.raises(ValueError, match=error):
            checked(archive_copy)
        return
    write(archive_copy / name, value)
    rebind(archive_copy, name)
    with pytest.raises(ValueError, match=error):
        checked(archive_copy)


def test_compressed_prepared_input_retains_its_raw_hash_pin(archive_copy: Path) -> None:
    name = "correspondence.json.gz"
    path = archive_copy / name
    path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()) + b" ", mtime=0))
    rebind(archive_copy, name)
    with pytest.raises(ValueError, match="published prepared input differs"):
        checked(archive_copy)


def test_selective_package_changes_require_their_explicit_audit_profile() -> None:
    with pytest.raises(ValueError, match="changes extend beyond runtime loading"):
        audit(ARCHIVE)
