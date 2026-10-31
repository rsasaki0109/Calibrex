"""Frozen depth-bias versus translation study on three analytic scene layouts.

Stage A: frozen one-sided pose-only fit under zero/positive/negative camera-depth bias.
Stage B: bias/pose sensitivity at the reference (oracle diagnostic) and a fixed
bias profile. Stage C: separate seven-parameter pose plus bias fit. The reference
pose generates data, labels outcomes and evaluates the oracle diagnostic only; it
never enters an estimator start other than the declared truth-start controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from pydantic import BaseModel, ConfigDict
from tools.depth_bias_pose import (
    DISTINGUISHABLE_FRACTION,
    biased_observations,
    identifiability,
    shifted,
)
from tools.depth_bias_pose import solve as joint_solve
from tools.joint_metric_pose import depth_residual
from tools.joint_multistart_control import matrix
from tools.joint_occlusion_pose import solve as pose_only_solve
from tools.known_geometry_joint import reference
from tools.occlusion_geometry import SCENES
from tools.run_camera_lidar_development_matrix import pose_metrics

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta

POPULATIONS = ("exact_z_visible", "exact_z_all")
BIASES_M = (0.0, 0.25, -0.25)
STARTS = {"+1": 1, "-1": -1, "truth_pose": 0}
PROFILE_M = (-0.5, -0.25, 0.0, 0.25, 0.5)
BIAS_TOLERANCE_M = 0.05
ABSORPTION_BAND = (0.5, 1.5)
PRIOR = Path("docs/assets/occlusion_pose_20261029/evaluation.json")
CONDITIONS: dict[str, Any] = {
    "scenes": list(SCENES),
    "populations": list(POPULATIONS),
    "biases_m": list(BIASES_M),
    "starts": [*STARTS, "full_truth (joint only, nonzero bias)"],
    "profile_bias_m": list(PROFILE_M),
    "bias_tolerance_m": BIAS_TOLERANCE_M,
    "absorption_band": list(ABSORPTION_BAND),
    "distinguishable_fraction": DISTINGUISHABLE_FRACTION,
    "pose_gate": {"converged": True, "rotation_deg_below": 0.5, "translation_m_below": 0.2},
    "per_fit_budget": {"max_nfev": 100, "actual_residual_calls": 800},
    "identifiability_residual_evaluations": 15,
}


class BiasReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = (
        "synthetic depth-bias diagnostic on three already-used analytic layouts; "
        "not real-data, held-out or SOTA evidence"
    )
    source_commit: str
    plan_sha256: str
    environment: dict[str, Any]
    conditions: dict[str, Any]
    pose_only: list[dict[str, Any]] = []
    identifiability: list[dict[str, Any]] = []
    joint: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}


def start(sign: int) -> SE3:
    """The prior experiment's signed perturbation; sign 0 is the truth start."""
    return apply_local_se3_delta(
        reference(),
        tuple(sign * np.array([1.0, -0.5, 0.75])),
        tuple(sign * np.array([0.15, -0.1, 0.1])),
    )


def attempt(
    fit: Callable[..., dict[str, Any]], *args: Any
) -> tuple[dict[str, Any] | None, str | None, float]:
    began = time.perf_counter()
    try:
        return fit(*args), None, time.perf_counter() - began
    except Exception as error:  # a failed fit is a recorded outcome, never retried
        return None, f"{type(error).__name__}: {error}", time.perf_counter() - began


def outcome(
    batch: list[DepthToDepthObservation],
    result: dict[str, Any] | None,
    error: str | None,
    seconds: float,
    bias_m: float,
) -> dict[str, Any]:
    """Pose metrics, camera-frame translation error and final support of one fit."""
    truth = matrix(reference())
    if result is None:
        # The frozen solvers raise on call 801; any other error has no call count.
        calls = 801 if error and "hard residual budget exceeded" in error else None
        return {
            "result": None,
            "error": error,
            "seconds": seconds,
            "actual_residual_calls": calls,
            "metrics": pose_metrics(None, truth, False),
            "translation_error_camera_m": None,
            "population": None,
        }
    pose = SE3(
        tuple(result["transform"]["translation_m"]),
        tuple(result["transform"]["rotation_quat_xyzw"]),
    )
    estimated = matrix(pose)
    population = []
    for obs in batch:
        residual, valid = depth_residual(obs, pose)
        residual = np.where(valid, residual + bias_m, residual)
        population.append(
            {
                "points": len(residual),
                "supported": int(valid.sum()),
                "behind_over_20cm": int(((residual > 0.2) & valid).sum()),
                "in_front_over_20cm": int(((residual < -0.2) & valid).sum()),
            }
        )
    return {
        "result": result,
        "error": None,
        "seconds": seconds,
        "actual_residual_calls": result["actual_residual_calls"],
        "metrics": pose_metrics(estimated, truth, result["converged"]),
        "translation_error_camera_m": (estimated[:3, 3] - truth[:3, 3]).tolist(),
        "population": population,
    }


def reproduced(pose_only: list[dict[str, Any]], prior: dict[str, Any]) -> bool:
    """Zero-bias pose-only fits must equal the frozen one-sided occlusion results."""
    expected = {
        (c["scene"], c["condition"], c["sign"]): c["methods"]["asymmetric"] for c in prior["cases"]
    }
    rows = [r for r in pose_only if r["bias_m"] == 0.0]
    if len(rows) != len(expected):
        return False
    for row in rows:
        old = expected[(row["scene"], row["condition"], STARTS[row["start"]])]
        new = json.loads(json.dumps({k: row[k] for k in ("result", "metrics", "population")}))
        if any(new[k] != old[k] for k in new):
            return False
    return True


def summarize(
    pose_only: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    joint: list[dict[str, Any]],
    baseline_reproduced: bool,
) -> dict[str, Any]:
    """Apply the decision rules frozen in the protocol; no threshold is data-chosen."""

    def key(row: dict[str, Any]) -> tuple[str, str, float]:
        return row["scene"], row["condition"], row["bias_m"]

    def passed(row: dict[str, Any]) -> bool:
        return bool(row["metrics"]["recovered"])

    labels = {key(r): r["at_reference"]["distinguishable"] for r in sensitivity}
    truth_rows = {key(r): r for r in pose_only if r["start"] == "truth_pose"}
    profile_consistent = all(
        p["result"] == truth_rows[key(r)]["result"]
        for r in sensitivity
        for p in r["profile"]
        if p["assumed_bias_m"] == 0.0
    )
    stage_a: dict[str, Any] = {}
    for bias in BIASES_M:
        rows = [r for r in pose_only if r["bias_m"] == bias]
        perturbed = [r for r in rows if r["start"] != "truth_pose"]
        truth_start = [r for r in rows if r["start"] == "truth_pose"]
        stage_a[f"{bias:+.2f}"] = {
            "perturbed_pass": sum(map(passed, perturbed)),
            "perturbed_cases": len(perturbed),
            "truth_start_pass": sum(map(passed, truth_start)),
            "truth_start_cases": len(truth_start),
        }
    ratios = [r["absorption_ratio"] for r in pose_only if r["absorption_ratio"] is not None]
    median = statistics.median(ratios) if ratios else None
    absorption = {
        "cases": len(ratios),
        "min": min(ratios, default=None),
        "median": median,
        "max": max(ratios, default=None),
        "translation_absorbs_bias": bool(
            median is not None and ABSORPTION_BAND[0] <= median <= ABSORPTION_BAND[1]
        ),
    }
    stage_a_pass = {(*key(r), r["start"]): passed(r) for r in pose_only}
    zero_regressions = [
        {"scene": r["scene"], "condition": r["condition"], "start": r["start"]}
        for r in joint
        if r["bias_m"] == 0.0
        and ((stage_a_pass[(*key(r), r["start"])] and not passed(r)) or not r["bias_recovered"])
    ]

    def tally(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "cases": len(rows),
            "pose_pass": sum(map(passed, rows)),
            "bias_recovered": sum(bool(r["bias_recovered"]) for r in rows),
            "both": sum(passed(r) and bool(r["bias_recovered"]) for r in rows),
        }

    biased = [r for r in joint if r["bias_m"] != 0.0]
    distinguishable = [r for r in biased if labels[key(r)]]
    ambiguous = [r for r in biased if not labels[key(r)]]
    at_estimate = [r for r in joint if r["at_estimate"] is not None]
    if not (baseline_reproduced and profile_consistent):
        verdict = "invalid_reproduction_failed"
    elif not distinguishable:
        verdict = "not_identifiable"
    elif zero_regressions or any(not (passed(r) and r["bias_recovered"]) for r in distinguishable):
        verdict = "not_supported"
    else:
        verdict = "supported_synthetic_diagnostic"
    stages = {"pose_only": pose_only, "joint": joint}
    return {
        "baseline_reproduced": baseline_reproduced,
        "profile_zero_matches_stage_a": profile_consistent,
        "stage_a_pose_only": stage_a,
        "stage_a_absorption_ratio": absorption,
        "stage_b_distinguishable_conditions": sorted(
            [list(k) for k, v in labels.items() if v], key=str
        ),
        "stage_b_conditions": len(labels),
        "stage_c_zero_bias": tally([r for r in joint if r["bias_m"] == 0.0]),
        "stage_c_zero_bias_regressions": zero_regressions,
        "stage_c_biased_distinguishable": tally(distinguishable),
        "stage_c_biased_ambiguous": tally(ambiguous),
        "stage_c_reference_free_label_agreement": {
            "cases": len(at_estimate),
            "agree": sum(
                r["at_estimate"]["distinguishable"] == labels[key(r)] for r in at_estimate
            ),
        },
        "failed_fits": {
            name: sum(r["result"] is None for r in rows) for name, rows in stages.items()
        }
        | {"profile": sum(p["result"] is None for r in sensitivity for p in r["profile"])},
        "actual_residual_calls": {
            name: sum(r["actual_residual_calls"] or 0 for r in rows)
            for name, rows in stages.items()
        }
        | {
            "profile": sum(
                p["actual_residual_calls"] or 0 for r in sensitivity for p in r["profile"]
            ),
            "identifiability": sum(r["at_reference"]["residual_evaluations"] for r in sensitivity)
            + sum(r["at_estimate"]["residual_evaluations"] for r in at_estimate),
        },
        "verdict": verdict,
    }


def run(plan: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("refuse overwrite")
    spec = json.loads(plan.read_text())
    for p, sha in spec["source_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
            raise ValueError(f"changed source: {p}")
    if spec["conditions"] != json.loads(json.dumps(CONDITIONS)):
        raise ValueError("plan conditions differ from frozen constants")
    report = BiasReport(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        environment={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "tracked_changes": subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).splitlines(),
        },
        conditions=CONDITIONS,
    )

    def save() -> None:
        output.write_text(report.model_dump_json(indent=2) + "\n")

    cases = [
        (scene, population, bias)
        for scene in SCENES
        for population in POPULATIONS
        for bias in BIASES_M
    ]
    for scene, population, bias in cases:  # Stage A
        batch = biased_observations(scene, population, bias)
        for label, sign in STARTS.items():
            result, error, seconds = attempt(pose_only_solve, batch, start(sign))
            row = {"scene": scene, "condition": population, "bias_m": bias, "start": label}
            row |= outcome(batch, result, error, seconds, 0.0)
            shift = row["translation_error_camera_m"]
            row["absorption_ratio"] = shift[2] / bias if bias and shift else None
            report.pose_only.append(row)
            save()
            print("A", scene, population, bias, label, row["metrics"], flush=True)
    for scene, population, bias in cases:  # Stage B
        batch = biased_observations(scene, population, bias)
        profile = []
        for assumed in PROFILE_M:
            corrected = shifted(batch, -assumed)
            result, error, seconds = attempt(pose_only_solve, corrected, start(0))
            point = {"assumed_bias_m": assumed}
            point |= outcome(corrected, result, error, seconds, 0.0)
            profile.append(point)
        report.identifiability.append(
            {
                "scene": scene,
                "condition": population,
                "bias_m": bias,
                "at_reference": identifiability(batch, reference(), bias),
                "profile": profile,
            }
        )
        save()
        print("B", scene, population, bias, report.identifiability[-1]["at_reference"], flush=True)
    for scene, population, bias in cases:  # Stage C
        batch = biased_observations(scene, population, bias)
        starts = [(label, start(sign), 0.0) for label, sign in STARTS.items()]
        if bias:
            starts.append(("full_truth", start(0), bias))
        for label, initial, initial_bias in starts:
            result, error, seconds = attempt(joint_solve, batch, initial, initial_bias)
            estimate = result["bias_m"] if result else 0.0
            row = {
                "scene": scene,
                "condition": population,
                "bias_m": bias,
                "start": label,
                "initial_bias_m": initial_bias,
            }
            row |= outcome(batch, result, error, seconds, estimate)
            row["bias_error_m"] = estimate - bias if result else None
            row["bias_recovered"] = bool(result and abs(estimate - bias) < BIAS_TOLERANCE_M)
            row["at_estimate"] = None
            if result:
                pose = SE3(
                    tuple(result["transform"]["translation_m"]),
                    tuple(result["transform"]["rotation_quat_xyzw"]),
                )
                row["at_estimate"] = identifiability(batch, pose, estimate)
            report.joint.append(row)
            save()
            print("C", scene, population, bias, label, row["metrics"], row["bias_error_m"])
    report.summary = summarize(
        report.pose_only,
        report.identifiability,
        report.joint,
        reproduced(report.pose_only, json.loads(PRIOR.read_text())),
    )
    save()
    print(json.dumps(report.summary, indent=2))
    output.with_suffix(".schema.json").write_text(
        json.dumps(BiasReport.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)
