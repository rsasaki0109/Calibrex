#!/usr/bin/env python3
"""Apply the committed v3 gate rule to the KITTI-360 development batch.

The rule is fixed in ``docs/benchmarks/kitti360_camera_lidar_v3_prep.md`` (committed before
any KITTI-360 check result). Development values come from the 22 spent segments' published
scores: the default rule for the primary claim, the opt-in 0.5 deg cross-start agreement
rule for the secondary warning claim. KITTI raw sets no gate. This tool only derives
numbers; it reads no evaluation data and states no verdict.

    python tools/kitti360_camera_lidar_v3_gates.py PUBLISHED_DIR OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calibrex.kitti360_camera_lidar_v3_gates/v0.1"
EVALUATION_SEGMENTS = 42
MISSED_1DEG_MAX = 0.25


def _tool(name: str) -> ModuleType:
    key = f"_kitti360_v3_gates_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def round_up(value: float, step: float) -> float:
    return round(math.ceil(round(value / step, 9)) * step, 10)


def round_down(value: float, step: float) -> float:
    return round(math.floor(round(value / step, 9)) * step, 10)


def upper_bound(misses: int, trials: int) -> float:
    """One-sided 95 % exact binomial upper bound; the rule of three when none is missed."""

    if trials < 1:
        raise ValueError("no development trials")
    if misses == 0:
        return 3.0 / trials
    if misses == trials:
        return 1.0
    return float(beta.ppf(0.95, misses + 1, trials - misses))


def allowance(misses: int, trials: int) -> dict[str, Any]:
    bound = upper_bound(misses, trials)
    return {
        "development": f"{misses} of {trials}",
        "upper_bound_95": bound,
        "allowance": min(1.0, round_up(bound, 0.05)),
    }


def error_gate(worst: float | None) -> dict[str, Any]:
    if worst is None:
        raise ValueError("no judged development axis")
    return {
        "development_worst_deg": worst,
        "times_1_5": 1.5 * worst,
        "threshold_deg": round_up(1.5 * worst, 0.25),
    }


def _trials(scores: dict[str, Any], angle: int) -> list[dict[str, Any]]:
    return [
        trial
        for item in scores.values()
        if item["state"] == "scored"
        for name, trial in item["trials"].items()
        if name.endswith(f":{angle}") and trial["counted"]
    ]


def primary(scores: dict[str, Any]) -> dict[str, Any]:
    """Default rule: v2's requirements with gates from these development values."""

    metrics = _tool("kitti_camera_lidar_v3_development").summarize_scores(scores)
    three = metrics["trials"]["3deg"]
    one = metrics["trials"]["1deg"]
    scored_rate = metrics["scored_drives"] / metrics["drives"]
    return {
        "metrics": metrics,
        "gates": {
            "judged-axis-accuracy": error_gate(metrics["max_baseline_judged_error_deg"]),
            "turned-judged-axis-accuracy": error_gate(metrics["max_turned_judged_error_deg"]),
            "judged-axis-coverage": {
                "development": metrics["judged_fraction_scored"],
                "threshold": round_down(metrics["judged_fraction_scored"] / 1.5, 0.25),
            },
            "scored-segments-coverage": {
                "development": f"{metrics['scored_drives']} of {metrics['drives']}",
                "threshold_segments": math.floor(round(scored_rate / 1.5 * EVALUATION_SEGMENTS, 9)),
            },
            "turn-3deg-flagged": allowance(three["counted"] - three["flagged"], three["counted"]),
            "turn-1deg-flagged": {
                "development": f"{one['counted'] - one['flagged']} of {one['counted']}",
                "allowance": MISSED_1DEG_MAX,
            },
            "turn-3deg-never-judged-right": {"development": three["judged_right"], "allowed": 0},
            "vendor-calibration-not-flagged": {
                "development": metrics["flagged_baselines"],
                "allowed": 0,
            },
            "no-failed-segments": {"development": metrics["failed_drives"], "allowed": 0},
        },
    }


def secondary(scores: dict[str, Any]) -> dict[str, Any]:
    """Opt-in warning: run-level verdicts, binomial allowances and the 1.5x error rule."""

    metrics = _tool("kitti_camera_lidar_v3_development").summarize_scores(scores)
    scored = [item for item in scores.values() if item["state"] == "scored"]
    warned = [item for item in scored if item["verdict"] in ("warn", "fail")]
    gates: dict[str, Any] = {
        "baseline-warned": allowance(len(warned), len(scored)),
        "judged-axis-accuracy": error_gate(metrics["max_baseline_judged_error_deg"]),
        "turned-judged-axis-accuracy": error_gate(metrics["max_turned_judged_error_deg"]),
    }
    for angle in (3, 1):
        trials = _trials(scores, angle)
        missed = sum(trial["verdict"] not in ("warn", "fail") for trial in trials)
        gates[f"turn-{angle}deg-run-flagged"] = allowance(missed, len(trials))
    return {"metrics": metrics, "gates": gates}


def scores_from(published: Path) -> dict[str, dict[str, Any]]:
    batch = _tool("kitti360_camera_lidar_v3_batch")
    scores: dict[str, dict[str, Any]] = {"current": {}, "proposed": {}}
    for segment in batch.segments():
        attempt, report = batch._latest(published, segment)
        if report is None or report["status"] != "PASS_DEVELOPMENT":
            for rule in scores:
                scores[rule][segment] = {
                    "state": "failed",
                    "reason": str(report and report["status"]),
                }
            continue
        path = published / segment / f"attempt_{attempt}" / "comparison" / "comparison.json"
        comparison = json.loads(path.read_text())["scores"]
        for rule in scores:
            scores[rule][segment] = comparison[rule][segment]
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("published", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse overwrite")
    scores = scores_from(args.published)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    rule_doc = ROOT / "docs/benchmarks/kitti360_camera_lidar_v3_prep.md"
    result = {
        "schema_version": SCHEMA,
        "analysis": "gate derivation from spent KITTI-360 development segments; no verdict",
        "primary": primary(scores["current"]),
        "secondary": secondary(scores["proposed"]),
        "provenance": {
            "generator": "tools/kitti360_camera_lidar_v3_gates.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rule_document_sha256": hashlib.sha256(rule_doc.read_bytes()).hexdigest(),
            "git_commit": commit,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "published_manifest_sha256": hashlib.sha256(
                (args.published / "manifest.json").read_bytes()
            ).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k]["gates"] for k in ("primary", "secondary")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
