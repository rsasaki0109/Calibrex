#!/usr/bin/env python3
"""Frozen development screen of one proposed v3 check rule on the 55 spent KITTI drives.

Pools single- or multi-drive ``compare`` reports (current versus proposed rule on identical
captured inputs), takes each spent drive exactly once and applies the criteria below. The
criterion values are v2's gate values; they were fixed before any proposed-rule replay of
real inputs. This is development on spent data: passing allows drafting a v3
pre-registration on unread data, never a SOTA claim.

    python tools/kitti_camera_lidar_v3_rule_screen.py OUT.json COMPARISON.json [...]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calibrex.camera_lidar_v3_rule_screen/v0.1"
LARGE_ERROR_DEG = 0.5  # the check's own observable rotation std bound
CRITERIA = {
    "baseline_flagged_max": 0,
    "baseline_judged_error_max_deg": 0.75,
    "turned_judged_error_max_deg": 1.5,
    "turn_3deg_missed_fraction_max": 0.1,
    "turn_1deg_missed_fraction_max": 0.25,
    "judged_fraction_scored_min": 0.5,
    "scored_drives_fraction_min": 2 / 3,
}


def _tool(name: str) -> ModuleType:
    key = f"_v3_screen_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def large_errors(scores: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Turned-run judged axes more than LARGE_ERROR_DEG from the injected reference."""

    return {
        (drive, trial, axis)
        for drive, item in scores.items()
        if item["state"] == "scored"
        for trial, record in item["trials"].items()
        for axis, error in record["judged_errors_deg"].items()
        if abs(error) > LARGE_ERROR_DEG
    }


def _rule(report: dict[str, Any]) -> dict[str, Any]:
    """Proposed overrides; reports from before ``proposed_options`` carry only the margin."""

    if "proposed_options" in report:
        return dict(report["proposed_options"])
    return {"start_scan_min_control_t": report["proposed_margin"]}


def screen(reports: list[dict[str, Any]]) -> dict[str, Any]:
    spent = _tool("kitti_camera_lidar_v3_replay").spent_drives()
    summarize = _tool("kitti_camera_lidar_v3_development").summarize_scores
    options = {json.dumps(_rule(r), sort_keys=True) for r in reports}
    if len(options) != 1:
        raise ValueError(f"reports compare different proposed rules: {sorted(options)}")
    scores: dict[str, dict[str, Any]] = {"current": {}, "proposed": {}}
    errors: dict[str, str] = {}
    for report in reports:
        for drive in report["drives"]:
            if drive in scores["current"]:
                raise ValueError(f"{drive} appears in more than one report")
            for rule in scores:
                scores[rule][drive] = report["scores"][rule][drive]
            if drive in report["errors"]:
                errors[drive] = report["errors"][drive]
    missing = sorted(set(spent) - set(scores["current"]))
    if missing or set(scores["current"]) != set(spent):
        raise ValueError(f"screen needs every spent drive exactly once; missing {missing}")
    metrics = {rule: summarize(values) for rule, values in scores.items()}
    current_large = large_errors(scores["current"])
    proposed_large = large_errors(scores["proposed"])
    p = metrics["proposed"]

    def missed(angle: str) -> float | None:
        trial = p["trials"][angle]
        return None if not trial["counted"] else 1 - trial["flagged"] / trial["counted"]

    checks = {
        "baseline_not_flagged": len(p["flagged_baselines"]) <= CRITERIA["baseline_flagged_max"],
        "baseline_judged_accuracy": (p["max_baseline_judged_error_deg"] or 0.0)
        <= CRITERIA["baseline_judged_error_max_deg"],
        "turned_judged_accuracy": (p["max_turned_judged_error_deg"] or 0.0)
        <= CRITERIA["turned_judged_error_max_deg"],
        "turn_3deg_flagged": (missed("3deg") is not None)
        and missed("3deg") <= CRITERIA["turn_3deg_missed_fraction_max"],
        "turn_1deg_flagged": (missed("1deg") is not None)
        and missed("1deg") <= CRITERIA["turn_1deg_missed_fraction_max"],
        "judged_axis_coverage": (p["judged_fraction_scored"] or 0.0)
        >= CRITERIA["judged_fraction_scored_min"],
        "scored_drives_coverage": p["scored_drives"] / len(spent)
        >= CRITERIA["scored_drives_fraction_min"],
        "fewer_large_errors": len(proposed_large) < len(current_large),
        "no_new_large_errors": not (proposed_large - current_large),
        "no_failed_drives": not errors and not p["failed_drives"],
    }
    return {
        "schema_version": SCHEMA,
        "analysis": "development screen on spent drives; not an audit or a SOTA claim",
        "proposed_options": json.loads(options.pop()),
        "criteria": CRITERIA,
        "large_error_deg": LARGE_ERROR_DEG,
        "metrics": metrics,
        "large_errors": {
            "current": sorted(map(list, current_large)),
            "proposed": sorted(map(list, proposed_large)),
            "new_under_proposed": sorted(map(list, proposed_large - current_large)),
            "removed_by_proposed": sorted(map(list, current_large - proposed_large)),
        },
        "checks": checks,
        "errors": errors,
        "verdict": "carry_forward" if all(checks.values()) else "not_carried_forward",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("comparisons", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse overwrite")
    reports = [json.loads(path.read_text()) for path in args.comparisons]
    result = screen(reports)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    result["provenance"] = {
        "generator": "tools/kitti_camera_lidar_v3_rule_screen.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.comparisons
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["verdict"], json.dumps(result["checks"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
