"""Fit frozen empirical error intervals, then evaluate on independent recordings.

Inputs are coverage.json files from evaluate_p0_uncertainty.py, with their
protocol.json and result files alongside. Output paths must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from calibrex.core.empirical_intervals import EmpiricalIntervalModel
from calibrex.evaluation.empirical_intervals import (
    evaluate_empirical_intervals,
    fit_empirical_intervals,
    interval_scope,
)
from calibrex.evaluation.uncertainty_coverage import UncertaintyCoverageReport


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Freeze a calibration-only model or score an already-frozen one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "evaluate"))
    parser.add_argument("report", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal", type=float)
    args = parser.parse_args()
    if (args.action == "evaluate") != (args.model is not None):
        parser.error("--model is required only for evaluate")
    if args.output.exists():
        parser.error("output exists; use a fresh path to retain prior evidence")
    if args.nominal is not None:
        if args.action != "fit":
            parser.error("--nominal belongs to fit; evaluation uses the frozen model")
        if not 0 < args.nominal < 1:
            parser.error("nominal must be between zero and one")
    report = UncertaintyCoverageReport.model_validate_json(args.report.read_text())
    protocol = json.loads((args.report.parent / "protocol.json").read_text())
    scope = interval_scope(report, protocol)
    for trial in report.trials:
        if trial.artifact_sha256 is not None:
            result = args.report.parent / f"result-{trial.seed}.json"
            if _digest(result) != trial.artifact_sha256:
                raise ValueError(f"result digest mismatch for seed {trial.seed}")
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__).resolve(),
        root / "src/calibrex/core/empirical_intervals.py",
        root / "src/calibrex/evaluation/empirical_intervals.py",
        root / "src/calibrex/evaluation/uncertainty_coverage.py",
    ]
    sources = {str(p.relative_to(root)): _digest(p) for p in files}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "fit":
        model = fit_empirical_intervals(
            report,
            scope_sha256=scope,
            report_sha256=_digest(args.report),
            nominal=0.95 if args.nominal is None else args.nominal,
            command=sys.argv,
            source_sha256=sources,
            git_commit=commit,
        )
        model.save(args.output)
        print(
            f"Frozen intervals: {sum(a.half_width is not None for a in model.intervals)}"
            f"/{len(model.intervals)}; calibration attempts: {len(model.calibration_seeds)}"
        )
    else:
        model = EmpiricalIntervalModel.model_validate_json(args.model.read_text())
        evaluation = evaluate_empirical_intervals(
            model,
            report,
            scope_sha256=scope,
            model_sha256=_digest(args.model),
            report_sha256=_digest(args.report),
            command=sys.argv,
            source_sha256=sources,
            git_commit=commit,
        )
        evaluation.save(args.output)
        for axis in evaluation.axes:
            print(
                f"{axis.axis}: formal={axis.formal_covered}/{axis.attempts} "
                f"empirical={axis.covered}/{axis.attempts} "
                f"width={axis.half_width} {axis.verdict}"
            )


if __name__ == "__main__":
    main()
