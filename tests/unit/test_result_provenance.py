"""Result-level provenance contract and legacy compatibility tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest
import yaml

from calibrex.cli.main import main
from calibrex.core.exceptions import ResultError
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import (
    CalibrationResult,
    FrameGraphSnapshot,
    RunInfo,
    build_result_provenance,
    load_result,
    result_json_schema,
    save_result,
)
from calibrex.core.validation import validate_file


def _empty_result() -> CalibrationResult:
    return CalibrationResult(
        run=RunInfo(id="empty", slac_version="test"),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
    )


def test_empty_new_result_is_blocked_at_save_boundary(tmp_path: Path) -> None:
    result = _empty_result()

    with pytest.raises(ResultError, match="production provenance"):
        save_result(result, tmp_path / "empty-explicit.yaml")


def test_legacy_result_load_is_honest_and_non_admissible(tmp_path: Path) -> None:
    source = read_mapping(Path("examples/precomputed/result.yaml"))
    legacy_path = tmp_path / "legacy.yaml"
    write_mapping(legacy_path, source)

    loaded = load_result(legacy_path)

    assert loaded.provenance_status == "legacy"
    assert loaded.production_valid is False
    assert loaded.run.provenance == source["run"]["provenance"]
    assert "provenance_version" not in loaded.run.provenance
    with pytest.raises(ResultError, match="legacy"):
        load_result(legacy_path, allow_legacy=False)
    report = validate_file(legacy_path, kind="result")
    assert report.valid is False
    assert report.production_valid is False
    assert report.admissibility == "blocked"
    assert report.provenance_issues


def test_result_schema_rejects_empty_provenance_but_keeps_legacy_maps_readable() -> None:
    schema = result_json_schema()
    legacy = yaml.safe_load(Path("examples/precomputed/result.yaml").read_text(encoding="utf-8"))
    jsonschema.validate(legacy, schema)
    empty = dict(legacy)
    empty["run"] = dict(legacy["run"])
    empty["run"]["provenance"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(empty, schema)


def test_result_schema_rejects_incomplete_production_provenance() -> None:
    schema = result_json_schema()
    payload = _empty_result().model_dump(mode="json")
    payload["run"]["provenance"] = build_result_provenance(
        producer="calibrex",
        tool_name="calibrex",
        tool_version="test",
        command=["pytest", "test_result_provenance.py"],
        config_sha256="a" * 64,
        input_sha256="b" * 64,
        git_commit="deadbee",
    )
    jsonschema.validate(payload, schema)

    invalid_cases = (
        ("producer", ""),
        ("command", []),
        ("git_commit", None),
        ("config_sha256", None),
        ("input_sha256", None),
    )
    for field_name, value in invalid_cases:
        invalid = deepcopy(payload)
        invalid["run"]["provenance"][field_name] = value
        if field_name == "git_commit":
            invalid["run"]["provenance"]["git_commit_unavailable_reason"] = None
        elif field_name == "config_sha256":
            invalid["run"]["provenance"]["config_digest_unavailable_reason"] = None
        elif field_name == "input_sha256":
            invalid["run"]["provenance"]["input_digest_unavailable_reason"] = None
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_build_result_provenance_has_stable_required_fields() -> None:
    provenance = build_result_provenance(
        producer="calibrex",
        tool_name="calibrex",
        tool_version="test",
        command=["pytest", "test_result_provenance.py"],
        config_sha256="a" * 64,
        input_sha256="b" * 64,
        git_commit="deadbee",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert {
        "provenance_version",
        "producer",
        "tool_name",
        "tool_version",
        "generated_at",
        "git_commit",
        "command",
        "config_sha256",
        "input_sha256",
    } <= provenance.keys()


def test_representative_cli_result_contains_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "calibration"
    assert (
        main(
            [
                "calibrate",
                "examples/configs/minimal.yaml",
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    result_path = output_dir / "result.yaml"
    result = load_result(result_path)
    assert result.production_valid
    for key in (
        "producer",
        "tool_name",
        "tool_version",
        "generated_at",
        "command",
        "config_sha256",
        "input_sha256",
    ):
        assert result.run.provenance.get(key)
    assert main(["validate", str(result_path), "--kind", "result", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["production_valid"] is True
    assert report["admissibility"] == "admissible"
