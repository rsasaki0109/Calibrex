#!/usr/bin/env python3
"""Score the KITTI-360 v3 camera-LiDAR check audit against its frozen pre-registration.

Reads the published batch (per-segment comparisons under the default rule and the opt-in
0.5 deg cross-start agreement rule) and the thresholds in the pre-registration's
``scoring`` section, and states one verdict per claim:

* ``supported``: every requirement passes;
* ``refuted``: a requirement other than a coverage requirement fails;
* ``inconclusive``: otherwise (a coverage requirement fails or something has nothing to
  measure); the audit is then incomplete.

The secondary claim shares the primary's failed-segment and coverage requirements.

    python tools/kitti360_camera_lidar_v3_score.py PREREGISTRATION PUBLISHED_DIR OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calibrex.kitti360_camera_lidar_v3_audit/v0.1"
TOLERANCE = 1e-9
COVERAGE = ("judged-axis-coverage", "scored-segments-coverage")
SHARED = ("no-failed-segments", *COVERAGE)


def _tool(name: str) -> ModuleType:
    key = f"_kitti360_v3_score_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _missed(trials: list[dict[str, Any]], *, run_level: bool) -> float | None:
    if not trials:
        return None
    if run_level:
        return sum(t["verdict"] not in ("warn", "fail") for t in trials) / len(trials)
    return sum(not t["flagged"] for t in trials) / len(trials)


def observe(current: dict[str, Any], proposed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Observed values of every requirement, by claim."""

    summarize = _tool("kitti_camera_lidar_v3_development").summarize_scores
    trials = _tool("kitti360_camera_lidar_v3_gates")._trials
    base = summarize(current)
    alt = summarize(proposed)
    scored = [item for item in proposed.values() if item["state"] == "scored"]
    warned = sum(item["verdict"] in ("warn", "fail") for item in scored)
    primary = {
        "no-failed-segments": len(base["failed_drives"]),
        "vendor-calibration-not-flagged": len(base["flagged_baselines"]),
        "judged-axis-accuracy": base["max_baseline_judged_error_deg"],
        "judged-axis-coverage": base["judged_fraction_scored"],
        "turned-judged-axis-accuracy": base["max_turned_judged_error_deg"],
        "turn-3deg-flagged": _missed(trials(current, 3), run_level=False),
        "turn-3deg-never-judged-right": base["trials"]["3deg"]["judged_right"]
        if base["trials"]["3deg"]["counted"]
        else None,
        "turn-1deg-flagged": _missed(trials(current, 1), run_level=False),
        "scored-segments-coverage": base["scored_drives"],
    }
    secondary = {
        **{name: primary[name] for name in SHARED},
        "baseline-warned": warned / len(scored) if scored else None,
        "judged-axis-accuracy": alt["max_baseline_judged_error_deg"],
        "turned-judged-axis-accuracy": alt["max_turned_judged_error_deg"],
        "turn-3deg-run-flagged": _missed(trials(proposed, 3), run_level=True),
        "turn-1deg-run-flagged": _missed(trials(proposed, 1), run_level=True),
    }
    return {"primary": primary, "secondary": secondary}


def judge(observed: dict[str, Any], requirements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each requirement's status and the claim's verdict."""

    if set(observed) != set(requirements):
        raise ValueError(f"requirements differ: {sorted(set(observed) ^ set(requirements))}")
    results: dict[str, Any] = {}
    for name, rule in requirements.items():
        value = observed[name]
        ((kind, threshold),) = rule.items()
        if value is None:
            status = "not_evaluated"
        elif kind == "min":
            status = "pass" if value >= threshold - TOLERANCE else "fail"
        elif kind == "max":
            status = "pass" if value <= threshold + TOLERANCE else "fail"
        else:
            raise ValueError(f"{name}: a requirement is a single min or max")
        results[name] = {"status": status, "observed": value, kind: threshold}
    hard = [n for n, r in results.items() if n not in COVERAGE and r["status"] == "fail"]
    if hard:
        verdict = "refuted"
    elif all(r["status"] == "pass" for r in results.values()):
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "requirements": results}


def scores_for(published: Path, evaluation: dict[str, dict[str, Any]]) -> dict[str, Any]:
    batch = _tool("kitti360_camera_lidar_v3_batch")
    scorer = _tool("score_kitti_camera_lidar_v2")
    scores: dict[str, dict[str, Any]] = {"current": {}, "proposed": {}}
    for segment in evaluation:
        attempt, report = batch._latest(published, segment)
        path = published / segment / f"attempt_{attempt}" / "comparison" / "comparison.json"
        if report is None or report["status"] != "PASS_DEVELOPMENT" or not path.is_file():
            reason = "missing" if report is None else f"{report['status']}: {report['error']}"
            for rule in scores:
                scores[rule][segment] = asdict(scorer.DriveScore("failed", reason))
            continue
        comparison = json.loads(path.read_text())["scores"]
        for rule in scores:
            scores[rule][segment] = comparison[rule][segment]
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("published", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse overwrite")
    plan = yaml.safe_load(args.preregistration.read_text(encoding="utf-8"))
    evaluation = {item["id"]: item for item in plan["recordings"]["evaluation"]}
    scores = scores_for(args.published, evaluation)
    observed = observe(scores["current"], scores["proposed"])
    claims = {
        name: judge(observed[name], plan["scoring"][f"{name}_requirements"])
        for name in ("primary", "secondary")
    }
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    result = {
        "schema_version": SCHEMA,
        "claims": claims,
        "scores": scores,
        "provenance": {
            "generator": "tools/kitti360_camera_lidar_v3_score.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "preregistration_sha256": hashlib.sha256(args.preregistration.read_bytes()).hexdigest(),
            "git_commit": commit,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reference": "the KITTI-360 published calibration, not ground truth",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: claim["verdict"] for name, claim in claims.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
