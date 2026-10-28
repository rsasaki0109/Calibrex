"""Recompute the frozen interval model and held-out evidence from retained draws."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from calibrex.core.empirical_intervals import EmpiricalIntervalEvaluation, EmpiricalIntervalModel
from calibrex.core.validation import validate_file
from calibrex.evaluation.empirical_intervals import (
    evaluate_empirical_intervals,
    fit_empirical_intervals,
    interval_scope,
)
from calibrex.evaluation.uncertainty_coverage import UncertaintyCoverageReport

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/assets/empirical_intervals"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_model_and_evaluation_replay_every_retained_draw() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    for path, expected in manifest["files"].items():
        assert digest(DATA / path) == expected, path
    model_path = DATA / "relative-model.json"
    evaluation_path = DATA / "relative-evaluation.json"
    assert validate_file(model_path).valid and validate_file(evaluation_path).valid
    model = EmpiricalIntervalModel.model_validate_json(model_path.read_text())
    evaluation = EmpiricalIntervalEvaluation.model_validate_json(evaluation_path.read_text())
    reports = {}
    scopes = {}
    for cohort, n in (("calibration", 20), ("evaluation", 10)):
        path = DATA / cohort / "coverage.json"
        report = UncertaintyCoverageReport.model_validate_json(path.read_text())
        assert len(report.trials) == n
        for trial in report.trials:
            assert digest(DATA / cohort / f"result-{trial.seed}.json") == trial.artifact_sha256
        reports[cohort] = report
        scopes[cohort] = interval_scope(
            report, json.loads((DATA / cohort / "protocol.json").read_text())
        )
    replay_model = fit_empirical_intervals(
        reports["calibration"], scope_sha256=scopes["calibration"],
        report_sha256=digest(DATA / "calibration/coverage.json"),
        nominal=model.nominal_coverage, command=model.command,
        source_sha256=model.source_sha256, git_commit=model.git_commit,
    )
    assert replay_model == model
    replay_evaluation = evaluate_empirical_intervals(
        model, reports["evaluation"], scope_sha256=scopes["evaluation"],
        model_sha256=digest(model_path), report_sha256=digest(DATA / "evaluation/coverage.json"),
        command=evaluation.command, source_sha256=evaluation.source_sha256,
        git_commit=evaluation.git_commit, control_width_scale=evaluation.control_width_scale,
    )
    # Float summation differs between Python 3.11 and 3.12. Keep all counts,
    # decisions, fitted widths and provenance exact; tolerate only rounding
    # in the diagnostic mean of formal interval widths.
    assert replay_evaluation.model_dump(exclude={"axes"}) == evaluation.model_dump(exclude={"axes"})
    for actual, expected in zip(replay_evaluation.axes, evaluation.axes, strict=True):
        excluded = {"mean_formal_half_width"}
        assert actual.model_dump(exclude=excluded) == expected.model_dump(exclude=excluded)
        assert actual.mean_formal_half_width == pytest.approx(
            expected.mean_formal_half_width, rel=1e-14, abs=0.0
        ), actual.axis
    assert len(evaluation.axes) == 13
    assert all(not axis.axis.startswith("gravity[") for axis in evaluation.axes)
    assert sum(axis.verdict == "undercoverage" for axis in evaluation.axes) == 3
    for path, expected in model.source_sha256.items():
        assert digest(DATA / "measured_source" / (path + ".txt")) == expected


def test_cli_fits_then_evaluates_and_refuses_to_overwrite(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    score = tmp_path / "score.json"
    tool = [sys.executable, str(ROOT / "tools/empirical_intervals.py")]
    fit = [*tool, "fit", str(DATA / "calibration/coverage.json"), "--output", str(model)]
    subprocess.run(fit, check=True, capture_output=True, text=True)
    before = digest(model)
    subprocess.run([*tool, "evaluate", str(DATA / "evaluation/coverage.json"),
                    "--model", str(model), "--output", str(score)], check=True,
                   capture_output=True, text=True)
    assert digest(model) == before
    assert validate_file(score).valid
    retry = subprocess.run(fit, capture_output=True, text=True)
    assert retry.returncode != 0 and "output exists" in retry.stderr
    assert digest(model) == before
