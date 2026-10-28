"""Replay retained P0 coverage and verify its complete digest-bound outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema

from calibrex.core.validation import validate_file
from calibrex.evaluation.uncertainty_coverage import (
    UncertaintyCoverageReport,
    summarize_coverage,
)

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/p0_reliability"


def test_retained_manifest_and_coverage_replay() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for folder, count in (("known_planes", 20), ("synthetic_bag", 5)):
        directory = ROOT / folder
        assert validate_file(directory / "coverage.json").valid
        payload = json.loads((directory / "coverage.json").read_text())
        jsonschema.validate(payload, json.loads((directory / "coverage.schema.json").read_text()))
        report = UncertaintyCoverageReport.model_validate(payload)
        assert len(report.trials) == count
        replay = summarize_coverage(report.trials, [a.axis for a in report.axes],
                                    nominal=report.nominal_coverage,
                                    control_std_scale=report.control_std_scale)
        assert replay == report.axes
        for trial in report.trials:
            assert trial.artifact_sha256 == hashlib.sha256(
                (directory / f"result-{trial.seed}.json").read_bytes()).hexdigest()
        for path, digest in report.source_sha256.items():
            snapshot = ROOT / "measured_source" / (path + ".txt")
            if snapshot.exists():
                assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
