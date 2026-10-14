#!/usr/bin/env python3
"""Post-hoc reading of the refuted Hilti camera-camera audit: was the turned axis judged?

An analysis, not part of the Calibrex package, and not part of the pre-registered audit:
it reads the committed runs of ``docs/assets/hilti2022_camera_sota/camera_camera/runs``
and never changes the verdict. It asks why the ``detection`` gate failed.

A *pair-trial* is one gated pair with the turned camera, on one recording and one
injection, whose pair was scored on the baseline (the units and trials of
``tools/score_hilti_camera_camera.py``). For each, the tool

* expresses the injected turn in the pair's parent-camera axes, the axes of the check's
  rotation errors, from the two runs' own candidates: ``rotvec(R_injected
  R_baseline^T)``. The axis it moves most is the *turned axis*;
* records whether the injected run judged that axis (``pass``, ``warn`` or ``fail``)
  or left it ``unchecked``, and how the baseline run treated it;
* reads both runs' evidence for that axis: the estimate's error to the candidate (also
  for an unchecked axis), its jackknife std and its known-bad control response, the two
  tests that decide whether the axis is judged. The *shift* is how far the estimate's
  error moved between the runs; a shift equal to the turn means the estimate saw it.

Per trial it lists the pairs that flagged the turn and the attribution. It also lists
the baseline's scored gated units on which the turned axis was ``unchecked``, with
their estimates, to show what judging such axes would have read on Kalibr's calibration.

::

    python tools/hilti_camera_camera_posthoc.py \\
        docs/assets/hilti2022_camera_sota/camera_camera/runs \\
        docs/assets/hilti2022_camera_sota/camera_camera/posthoc.json \\
        --recording exp05 --recording exp06 ...
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_hilti_camera_camera import (
    GATED_PAIRS,
    INJECTIONS,
    _group_of,
    _pair_record,
    _run_artifact,
    frame,
    injection_name,
    score_unit,
)

from calibrex import __version__
from calibrex.core.provenance import git_commit, sha256_path

AXIS_NAMES = ("roll", "pitch", "yaw")
MOVED_WITH_THE_TURN = 0.5
"""A shift of at least this fraction of the turn, in its direction, saw the turn."""


def turn_in_pair_deg(baseline: dict[str, Any], injected: dict[str, Any]) -> list[float]:
    """The injected turn about the pair's parent-camera axes, in degrees."""

    def rotation(record: dict[str, Any]) -> Rotation:
        return Rotation.from_quat(record["candidate_transform"]["rotation_quat_xyzw"])

    turn = rotation(injected) * rotation(baseline).inv()
    return [float(value) for value in np.degrees(turn.as_rotvec())]


def axis_state(record: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """How a run treated one axis of a pair: judged with a verdict, unchecked, or absent."""

    if record is None:
        return {"state": "no_record"}
    for axis in record.get("axes", []):
        if axis["name"] == name:
            return {
                "state": "judged",
                "status": axis["status"],
                "candidate_error_deg": axis["candidate_error"],
                "tolerance_deg": axis["tolerance"],
            }
    for axis in record.get("unchecked_axes", []):
        if axis["name"] == name:
            return {
                "state": "unchecked",
                "reason_code": axis.get("reason_code"),
                "reason": axis.get("reason"),
            }
    return {"state": "absent", "pair_status": record.get("status")}


def axis_estimate(run_dir: Path, record: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    """The stereo estimator's evidence for one axis, judged or not; ``None`` without it."""

    if record is None or not record.get("evidence_artifact"):
        return None
    evidence = yaml.safe_load((run_dir / record["evidence_artifact"]).read_text(encoding="utf-8"))
    axes = {axis["name"]: axis for axis in evidence.get("axes") or []}
    if name not in axes:
        return None
    axis, options = axes[name], evidence["options"]
    scale = axis.get("scale_std_deg") or 0.0
    # the estimator reports std_deg = hypot(jackknife std, row-scale std) and tests the
    # jackknife std alone against max_std_deg
    jackknife = math.sqrt(max(axis["std_deg"] ** 2 - scale**2, 0.0))
    response = axis["control_response"]
    recentred = evidence.get("recentred_by_deg")
    return {
        "candidate_error_deg": axis["candidate_error_deg"],
        "jackknife_std_deg": jackknife,
        "jackknife_above_bound": jackknife > options["max_std_deg"],
        "control_response": response,
        "control_failed": not options["min_response"] <= response <= options["max_response"],
        "observable": axis["observable"],
        "reason": axis["reason"],
        "recentred_by_deg": recentred[AXIS_NAMES.index(name)] if recentred else 0.0,
    }


def analyse(runs_dir: Path, recordings: list[str]) -> dict[str, Any]:
    pair_trials: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any] | None] = {}
    for recording in recordings:
        directory = runs_dir / recording
        baseline, why = _run_artifact(directory, "baseline")
        baselines[recording] = baseline
        for camera, axis, degrees in INJECTIONS:
            name = injection_name(camera, axis, degrees)
            injected, _ = _run_artifact(directory, name)
            flagged, scored_pairs = [], []
            for left, right in GATED_PAIRS:
                if camera not in (left, right):
                    continue
                if score_unit(baseline, left, right, why).state != "scored":
                    continue
                assert baseline is not None
                key = f"{left}-{right}"
                scored_pairs.append(key)
                base_record = _pair_record(baseline, left, right)
                inj_record = _pair_record(injected, left, right) if injected else None
                if inj_record is None:
                    pair_trials.append(
                        {"recording": recording, "injection": name, "pair": key, "missing": True}
                    )
                    continue
                assert base_record is not None
                turn = turn_in_pair_deg(base_record, inj_record)
                index = int(np.argmax(np.abs(turn)))
                turned = AXIS_NAMES[index]
                if inj_record.get("status") in ("warn", "fail"):
                    flagged.append(key)
                inj_estimate = axis_estimate(directory, inj_record, turned)
                base_estimate = axis_estimate(directory, base_record, turned)
                shift = (
                    inj_estimate["candidate_error_deg"] - base_estimate["candidate_error_deg"]
                    if inj_estimate is not None and base_estimate is not None
                    else None
                )
                pair_trials.append(
                    {
                        "recording": recording,
                        "injection": name,
                        "pair": key,
                        "turn_in_pair_deg": turn,
                        "turned_axis": turned,
                        "injected": axis_state(inj_record, turned),
                        "baseline": axis_state(base_record, turned),
                        "injected_pair_verdict": inj_record.get("status"),
                        "injected_estimate": inj_estimate,
                        "baseline_estimate": base_estimate,
                        "estimate_shift_deg": shift,
                        "shift_over_turn": shift / turn[index] if shift is not None else None,
                    }
                )
            if not scored_pairs:
                continue
            group = _group_of(injected, frame(camera)) if injected is not None else None
            trials.append(
                {
                    "recording": recording,
                    "injection": name,
                    "scored_pairs": scored_pairs,
                    "flagging_pairs": flagged,
                    "attribution": (
                        {"status": group.get("status"), "frame": group.get("frame")}
                        if group is not None
                        else None
                    ),
                }
            )

    complete = [item for item in pair_trials if not item.get("missing")]
    turned_axes = sorted({item["turned_axis"] for item in complete})
    judged = [item for item in complete if item["injected"]["state"] == "judged"]
    unchecked = [item for item in complete if item["injected"]["state"] == "unchecked"]
    hidden = [item for item in unchecked if item["baseline"]["state"] == "judged"]
    moved = [
        item
        for item in unchecked
        if item["shift_over_turn"] is not None and item["shift_over_turn"] >= MOVED_WITH_THE_TURN
    ]
    ambiguous = [
        item for item in trials if (item["attribution"] or {}).get("status") == "ambiguous"
    ]

    # what the baseline's unchecked turned axis read on Kalibr's calibration
    baseline_unchecked: list[dict[str, Any]] = []
    if len(turned_axes) == 1:
        for recording, baseline in baselines.items():
            for left, right in GATED_PAIRS:
                if score_unit(baseline, left, right, "").state != "scored":
                    continue
                assert baseline is not None
                record = _pair_record(baseline, left, right)
                if axis_state(record, turned_axes[0])["state"] != "unchecked":
                    continue
                estimate = axis_estimate(runs_dir / recording, record, turned_axes[0])
                baseline_unchecked.append(
                    {"recording": recording, "pair": f"{left}-{right}", "estimate": estimate}
                )
        baseline_unchecked.sort(
            key=lambda item: -abs((item["estimate"] or {}).get("candidate_error_deg", 0.0))
        )

    def shifts(items: list[dict[str, Any]]) -> list[float]:
        return sorted(round(abs(item["estimate_shift_deg"]), 3) for item in items)

    def count(items: list[dict[str, Any]], test: str) -> int:
        return sum(bool((item["injected_estimate"] or {}).get(test)) for item in items)

    baseline_errors = [
        abs(item["estimate"]["candidate_error_deg"])
        for item in baseline_unchecked
        if item["estimate"] is not None
    ]
    return {
        "pair_trials": pair_trials,
        "trials": trials,
        "baseline_unchecked_turned_axis": baseline_unchecked,
        "summary": {
            "pair_trials": len(pair_trials),
            "missing_injected_records": len(pair_trials) - len(complete),
            "turned_axis": dict(Counter(item["turned_axis"] for item in complete)),
            "turned_axis_judged": len(judged),
            "judged_verdicts": dict(Counter(item["injected"]["status"] for item in judged)),
            "turned_axis_unchecked": len(unchecked),
            "unchecked_reason_codes": dict(
                Counter(item["injected"]["reason_code"] for item in unchecked)
            ),
            "unchecked_jackknife_above_bound": count(unchecked, "jackknife_above_bound"),
            "unchecked_control_failed": count(unchecked, "control_failed"),
            "unchecked_pair_verdicts": dict(
                Counter(item["injected_pair_verdict"] for item in unchecked)
            ),
            "unchecked_estimate_moved_with_the_turn": len(moved),
            "unchecked_moved_abs_shift_deg": shifts(moved),
            "judged_abs_shift_deg": shifts(
                [item for item in judged if item["estimate_shift_deg"] is not None]
            ),
            "unchecked_but_judged_on_the_baseline": len(hidden),
            "hidden_reason_codes": dict(
                Counter(item["injected"]["reason_code"] for item in hidden)
            ),
            "hidden_jackknife_above_bound": count(hidden, "jackknife_above_bound"),
            "hidden_jackknife_std_deg": sorted(
                round(item["injected_estimate"]["jackknife_std_deg"], 3) for item in hidden
            ),
            "hidden_recentred_by_deg": sorted(
                round(item["injected_estimate"]["recentred_by_deg"], 3) for item in hidden
            ),
            "hidden_control_failed": count(hidden, "control_failed"),
            "hidden_moved_with_the_turn": sum(item in moved for item in hidden),
            "trials": len(trials),
            "ambiguous_trials": len(ambiguous),
            "ambiguous_trials_flagged_only_by": dict(
                Counter("+".join(item["flagging_pairs"]) or "none" for item in ambiguous)
            ),
            "baseline_scored_units_with_turned_axis_unchecked": len(baseline_unchecked),
            "their_abs_error_deg_max": round(max(baseline_errors), 3) if baseline_errors else None,
            "their_abs_error_over_0.5_deg": sum(error > 0.5 for error in baseline_errors),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recording", action="append", required=True)
    args = parser.parse_args(argv)
    result = analyse(args.runs_dir, args.recording)
    result["provenance"] = {
        "generator": "tools/hilti_camera_camera_posthoc.py",
        "post_hoc": "an analysis of the committed runs; it does not change the audit verdict",
        "calibrex_version": __version__,
        "git_commit": git_commit(),
        "tool_sha256": sha256_path(Path(__file__).resolve()),
        "python": platform.python_version(),
        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recordings": args.recording,
        "manifests_sha256": {
            recording: sha256_path(args.runs_dir / recording / "manifest.json")
            for recording in args.recording
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
