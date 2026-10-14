#!/usr/bin/env python3
"""Score ``calibrex check`` camera-camera runs against Kalibr for the Hilti 2022 audit.

The scorer implements exactly the metrics, gates and verdict rules of
``docs/benchmarks/hilti_camera_sota_preregistration.yaml`` (claim ``camera-camera``);
its thresholds are module constants and cannot be changed from the command line::

    python tools/score_hilti_camera_camera.py RUNS_DIR OUT/scores.json \\
        --recording exp05 --recording exp06 ...

``RUNS_DIR/<recording>/`` is the output directory of ``tools/hilti_camera_sota_run.py``
for that recording: ``manifest.json``, ``baseline.json`` and one ``inject_*.json`` per
pre-registered injection. A missing directory, manifest or artifact, or a non-zero exit
code, fails every unit of the recording.

* A *unit* is one gated camera pair on one recording, read from the baseline run (the
  candidate is Hilti's Kalibr calibration). It is ``scored`` when the pair has at least
  one judged axis (verdict ``pass``, ``warn`` or ``fail``), ``inconclusive`` when it
  has none, and ``failed`` when the run or the pair record is missing or the pair was
  skipped. Its metric is the largest absolute error of a judged axis against Kalibr.
* A *trial* is one injected turn on one recording. It is ``evaluable`` when the
  baseline scored at least one gated pair that contains the turned camera. It is
  ``detected`` when one of those pairs reads ``warn`` or ``fail`` in the injected run,
  and ``localized`` when the attribution group holding the camera is ``attributed`` to
  that camera.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GATED_PAIRS = (
    ("cam0", "cam1"),
    ("cam0", "cam3"),
    ("cam1", "cam3"),
    ("cam0", "cam4"),
    ("cam1", "cam4"),
)
REPORTED_PAIRS = (("cam0", "cam2"), ("cam1", "cam2"), ("cam2", "cam3"), ("cam2", "cam4"))
INJECTIONS = (("cam3", "y", 1.0), ("cam1", "y", 1.0))
ACCURACY_MAX_DEG = 0.75
SCORED_FRACTION_MIN = 0.7
DETECTION_RATE_MIN = 0.9
LOCALIZATION_RATE_MIN = 0.5
EVALUABLE_FRACTION_MIN = 0.7
TOLERANCE = 1e-9
REQUIREMENTS = (
    "accuracy-vs-kalibr",
    "no-failed-units",
    "scored-units-coverage",
    "detection",
    "localization",
    "evaluable-trials-coverage",
)


def frame(camera: str) -> str:
    return f"alphasense/{camera}/image_raw"


def injection_name(camera: str, axis: str, degrees: float) -> str:
    return f"inject_{camera}_{axis}{degrees:+g}"


@dataclass
class Unit:
    state: str
    reason: str | None = None
    verdict: str | None = None
    max_abs_error_deg: float | None = None
    judged_axes: dict[str, float] = field(default_factory=dict)


@dataclass
class Trial:
    evaluable: bool
    detected: bool | None = None
    localized: bool | None = None
    reason: str | None = None
    attribution: dict[str, Any] | None = None
    pair_verdicts: dict[str, str] = field(default_factory=dict)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (OSError, ValueError):
        return None


def _pair_record(artifact: dict[str, Any], left: str, right: str) -> dict[str, Any] | None:
    for record in artifact.get("pairs", []):
        if record.get("pair") == "camera-camera" and record.get("frames") == [
            frame(left),
            frame(right),
        ]:
            return record  # type: ignore[no-any-return]
    return None


def score_unit(artifact: dict[str, Any] | None, left: str, right: str, why: str) -> Unit:
    if artifact is None:
        return Unit("failed", why)
    record = _pair_record(artifact, left, right)
    if record is None:
        return Unit("failed", "no camera-camera record for the pair")
    status = record.get("status")
    if status == "skipped":
        return Unit("failed", f"skipped: {record.get('reason_code')}")
    if status == "inconclusive":
        return Unit("inconclusive", record.get("reason"), verdict=status)
    judged = {axis["name"]: float(axis["candidate_error"]) for axis in record.get("axes", [])}
    if not judged:
        return Unit("inconclusive", "no judged axis", verdict=status)
    return Unit(
        "scored",
        verdict=status,
        max_abs_error_deg=max(abs(value) for value in judged.values()),
        judged_axes=judged,
    )


def _group_of(artifact: dict[str, Any], frame_name: str) -> dict[str, Any] | None:
    report = artifact.get("rotation_attribution") or {}
    for group in report.get("groups", []):
        if frame_name in group.get("frames", []):
            return group  # type: ignore[no-any-return]
    return None


def score_trial(
    baseline_units: dict[str, Unit], injected: dict[str, Any] | None, camera: str, why: str
) -> Trial:
    pairs = [
        f"{left}-{right}"
        for left, right in GATED_PAIRS
        if camera in (left, right) and baseline_units[f"{left}-{right}"].state == "scored"
    ]
    if not pairs:
        return Trial(False, reason="no gated pair with the camera was scored on the baseline")
    if injected is None:
        return Trial(True, detected=False, localized=False, reason=why)
    verdicts = {}
    for key in pairs:
        left, right = key.split("-")
        record = _pair_record(injected, left, right)
        verdicts[key] = record.get("status", "missing") if record is not None else "missing"
    group = _group_of(injected, frame(camera))
    return Trial(
        True,
        detected=any(verdict in ("warn", "fail") for verdict in verdicts.values()),
        localized=group is not None
        and group.get("status") == "attributed"
        and group.get("frame") == frame(camera),
        attribution=(
            {"status": group.get("status"), "frame": group.get("frame")} if group else None
        ),
        pair_verdicts=verdicts,
    )


def _run_artifact(directory: Path, run: str) -> tuple[dict[str, Any] | None, str]:
    manifest = _load(directory / "manifest.json")
    if manifest is None:
        return None, "manifest missing"
    entry = manifest.get("runs", {}).get(run)
    if entry is None:
        return None, f"run {run} not in the manifest"
    if entry.get("exit_code") != 0:
        return None, f"run {run} exited with {entry.get('exit_code')}"
    artifact = _load(directory / f"{run}.json")
    return artifact, ("" if artifact is not None else f"{run}.json missing or unreadable")


def score(runs_dir: Path, recordings: list[str]) -> dict[str, Any]:
    units: dict[str, dict[str, Unit]] = {}
    reported: dict[str, dict[str, Unit]] = {}
    trials: dict[str, dict[str, Trial]] = {}
    for recording in recordings:
        directory = runs_dir / recording
        baseline, why = _run_artifact(directory, "baseline")
        units[recording] = {
            f"{left}-{right}": score_unit(baseline, left, right, why) for left, right in GATED_PAIRS
        }
        reported[recording] = {
            f"{left}-{right}": score_unit(baseline, left, right, why)
            for left, right in REPORTED_PAIRS
        }
        trials[recording] = {}
        for camera, axis, degrees in INJECTIONS:
            name = injection_name(camera, axis, degrees)
            injected, why_injected = _run_artifact(directory, name)
            trials[recording][name] = score_trial(units[recording], injected, camera, why_injected)
    every_unit = [unit for per in units.values() for unit in per.values()]
    scored = [unit for unit in every_unit if unit.state == "scored"]
    failed = [unit for unit in every_unit if unit.state == "failed"]
    every_trial = [trial for per in trials.values() for trial in per.values()]
    evaluable = [trial for trial in every_trial if trial.evaluable]
    detection_rate = (
        sum(bool(t.detected) for t in evaluable) / len(evaluable) if evaluable else None
    )
    localization_rate = (
        sum(bool(t.localized) for t in evaluable) / len(evaluable) if evaluable else None
    )
    worst = max((unit.max_abs_error_deg or 0.0) for unit in scored) if scored else None
    scored_fraction = len(scored) / len(every_unit) if every_unit else 0.0
    evaluable_fraction = len(evaluable) / len(every_trial) if every_trial else 0.0

    def requirement(failed_gate: bool | None, observed: float | None, threshold: float):
        status = "not_evaluated" if failed_gate is None else ("fail" if failed_gate else "pass")
        return {"status": status, "observed": observed, "threshold": threshold}

    requirements = {
        "accuracy-vs-kalibr": requirement(
            worst > ACCURACY_MAX_DEG + TOLERANCE if worst is not None else None,
            worst,
            ACCURACY_MAX_DEG,
        ),
        "no-failed-units": requirement(bool(failed), float(len(failed)), 0.0),
        "scored-units-coverage": requirement(
            scored_fraction < SCORED_FRACTION_MIN - TOLERANCE, scored_fraction, SCORED_FRACTION_MIN
        ),
        "detection": requirement(
            detection_rate < DETECTION_RATE_MIN - TOLERANCE if detection_rate is not None else None,
            detection_rate,
            DETECTION_RATE_MIN,
        ),
        "localization": requirement(
            localization_rate < LOCALIZATION_RATE_MIN - TOLERANCE
            if localization_rate is not None
            else None,
            localization_rate,
            LOCALIZATION_RATE_MIN,
        ),
        "evaluable-trials-coverage": requirement(
            evaluable_fraction < EVALUABLE_FRACTION_MIN - TOLERANCE,
            evaluable_fraction,
            EVALUABLE_FRACTION_MIN,
        ),
    }
    coverage = ("scored-units-coverage", "evaluable-trials-coverage")
    statuses = {name: item["status"] for name, item in requirements.items()}
    if any(status == "fail" for name, status in statuses.items() if name not in coverage):
        verdict = "refuted"
    elif all(status == "pass" for status in statuses.values()):
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {
        "units": {r: {k: asdict(u) for k, u in per.items()} for r, per in units.items()},
        "reported_units": {
            r: {k: asdict(u) for k, u in per.items()} for r, per in reported.items()
        },
        "trials": {r: {k: asdict(t) for k, t in per.items()} for r, per in trials.items()},
        "summary": {
            "units": len(every_unit),
            "scored_units": len(scored),
            "failed_units": len(failed),
            "trials": len(every_trial),
            "evaluable_trials": len(evaluable),
            "detection_rate": detection_rate,
            "localization_rate": localization_rate,
        },
        "requirements": requirements,
        "verdict": verdict,
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recording", action="append", required=True)
    args = parser.parse_args(argv)
    from calibrex import __version__
    from calibrex.core.provenance import sha256_path

    result = score(args.runs_dir, args.recording)
    result["provenance"] = {
        "generator": "tools/score_hilti_camera_camera.py",
        "calibrex_version": __version__,
        "git_commit": _git_commit(),
        "scorer_sha256": sha256_path(Path(__file__).resolve()),
        "python": platform.python_version(),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recordings": args.recording,
        "manifests_sha256": {
            recording: sha256_path(args.runs_dir / recording / "manifest.json")
            for recording in args.recording
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["verdict"])
    for name, item in result["requirements"].items():
        observed, threshold = item["observed"], item["threshold"]
        print(f"  {name}: {item['status']} (observed {observed}, threshold {threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
