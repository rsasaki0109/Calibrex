"""Assemble the pre-registered KITTI camera-lidar v2 SOTA audit from ``scores.json``.

The gates are fixed by ``docs/benchmarks/kitti_camera_lidar_v2_preregistration.yaml`` and
mirror ``tools/score_kitti_camera_lidar_v2.py``; the preregistration's SHA-256 is pinned in
the protocol. Usage::

    python tools/build_kitti_camera_lidar_v2_audit.py \\
        SCORES.json docs/benchmarks/kitti_camera_lidar_v2_preregistration.yaml OUT_DIR

Nine benchmarks are written, each with the single method ``calibrex_check`` (KITTI's
calibration is the reference, not a method):

* ``drives``: one split per scored or failed drive (a not-usable drive has none); a failed
  drive is a failed trial;
* ``evaluation``: one split per evaluation drive; one not scored is a failed trial;
* ``baseline``: one split per scored drive; metric 1 when the check flags KITTI's own
  calibration (verdict warn or fail), else 0;
* ``axes``: one split per judged axis of a scored drive's baseline run; metric the absolute
  error to KITTI's calibration;
* ``coverage``: one split per scored drive; metric the number of its baseline axes not
  judged (a count, so that a mean of exactly 1.5 is exact);
* ``turned``: one split per judged axis of every turned run of a scored drive; metric the
  absolute error to KITTI's calibration;
* ``turn3`` and ``turn1``: one split per counted 3 deg (1 deg) turned trial; a trial whose
  turned axis is not flagged is a failed trial;
* ``turn3_judged_right``: one split per counted 3 deg trial; metric 1 when the check
  judges the turned axis ``pass``, else 0.

A requirement the scorer could not evaluate, and a coverage requirement that is not met,
is left without evidence, so the audit is ``incomplete`` (the preregistration's
``inconclusive``) unless another requirement is contradicted, in which case it is
``refuted``. The audit's verdict therefore equals the scorer's.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from calibrex import __version__
from calibrex.core.benchmark import (
    BenchmarkDefinition,
    BenchmarkMethodDefinition,
    BenchmarkMetricDefinition,
    BenchmarkProtocol,
    BenchmarkProvenance,
    BenchmarkSplit,
    BenchmarkTrial,
    BenchmarkTrialProvenance,
    aggregate_benchmark_definition,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.sota_audit import (
    SotaAuditProtocol,
    SotaAuditProvenance,
    SotaClaimRequirement,
    SotaClaimScope,
)
from calibrex.evaluation.sota_audit import audit_sota_claim

PROTOCOL_ID = "kitti-camera-lidar-check-v2"
AXIS_ERROR_MAX_DEG = 0.75
TURNED_AXIS_ERROR_MAX_DEG = 1.5
UNJUDGED_AXES_MEAN_MAX = 1.5  # judged-axis coverage: at least half of the 3 axes judged
MISSED_3DEG_MAX = 0.1
MISSED_1DEG_MAX = 0.25
SCORED_DRIVES_MIN = 15
DAYS = ("2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03")
CLAIM = (
    f"On {{drives}} KITTI raw drives of {', '.join(DAYS[:-1])} and {DAYS[-1]} that no "
    "Calibrex camera-LiDAR method had read, calibrex check --pairs camera-lidar on each "
    "drive's first 600 frames gives honest per-axis verdicts against that day's KITTI "
    "calibration (calib_velo_to_cam with the rectification of calib_cam_to_cam): it never "
    "flags that calibration (no warn or fail); every rotation axis it judges there "
    "(reported std at most 0.5 deg and its 2 deg known-bad control detected on held-out "
    f"frames at t >= 2.5) is within {AXIS_ERROR_MAX_DEG:g} deg of it, and it judges at "
    f"least {1 - UNJUDGED_AXES_MEAN_MAX / 3:.0%} of the axes; started at that calibration "
    "with the camera turned 1 or 3 deg about one of its axes, every axis it judges is "
    f"within {TURNED_AXIS_ERROR_MAX_DEG:g} deg of the day's calibration; it flags a 3 deg "
    "turn about an axis it judged on the drive in at least "
    f"{1 - MISSED_3DEG_MAX:.0%} of the trials and never judges such a turned axis right; "
    f"and it flags a 1 deg turn in at least {1 - MISSED_1DEG_MAX:.0%} of the trials."
).replace("%", " percent")
METHOD = BenchmarkMethodDefinition(
    method_id="calibrex_check",
    label="calibrex check camera-lidar (targetless edge alignment from the candidate)",
    implementation="calibrex_native",
    tool_name="calibrex",
)
# Benchmarks need two methods. KITTI's calibration is the reference the metrics are
# measured against, so its own error is zero by definition; it is never gated.
REFERENCE = BenchmarkMethodDefinition(
    method_id="kitti_reference",
    label="KITTI calib_velo_to_cam (the reference; error against itself is 0)",
    implementation="calibrex_native",
    tool_name="kitti",
)


def _metric(name: str, label: str, unit: str, interpretation: str) -> BenchmarkMetricDefinition:
    return BenchmarkMetricDefinition(
        name=name,
        label=label,
        unit=unit,
        direction="lower",
        primary=True,
        interpretation=interpretation,
    )


def _definition(
    benchmark_id: str,
    title: str,
    metric: BenchmarkMetricDefinition,
    rows: dict[str, float | str],
    digest: str,
    split_policy: str,
) -> BenchmarkDefinition:
    """``rows`` maps a split id to its metric value, or to a failure reason string."""

    splits = [
        BenchmarkSplit(
            split_id=key,
            seed=0,
            fit_count=1,
            holdout_count=1,
            fit_ids_sha256=hashlib.sha256(f"fit-{key}".encode()).hexdigest(),
            holdout_ids_sha256=hashlib.sha256(key.encode()).hexdigest(),
        )
        for key in rows
    ]
    provenance = BenchmarkTrialProvenance(
        command="calibrex check --pairs camera-lidar", config_sha256=digest, input_sha256=digest
    )
    trials = [
        BenchmarkTrial(
            method_id=REFERENCE.method_id,
            split_id=key,
            status="success",
            metrics={metric.name: 0.0},
            provenance=provenance,
        )
        for key in rows
    ] + [
        BenchmarkTrial(
            method_id=METHOD.method_id,
            split_id=key,
            status="failed" if isinstance(value, str) else "success",
            metrics={} if isinstance(value, str) else {metric.name: value},
            failure_reason=value if isinstance(value, str) else None,
            provenance=provenance,
        )
        for key, value in rows.items()
    ]
    return BenchmarkDefinition(
        benchmark_id=benchmark_id,
        title=title,
        protocol=BenchmarkProtocol(
            protocol_id=PROTOCOL_ID,
            dataset_id="kitti_raw_four_days_unused_drives_image_02_velodyne",
            dataset_source_sha256=digest,
            data_license="CC BY-NC-SA 3.0",
            split_policy=split_policy,
            splits=splits,
            initial_estimate_policy=(
                "the check starts at the candidate: KITTI's calibration, or it turned by the "
                "pre-registered injection"
            ),
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy="a download, conversion or check failure fails the drive",
        ),
        metrics=[metric],
        methods=[REFERENCE, METHOD],
        trials=trials,
        reference_method_id=REFERENCE.method_id,
        bootstrap_samples=2000,
        limitations=[
            "The reference is KITTI's target-based calibration, not ground truth.",
            "The drives come from four days of one car, each day with its own calibration.",
            "Only the axes the check judges are gated; the others are counted by coverage.",
            "Turned candidates are the reference turned about one camera axis at a time.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_kitti_camera_lidar_v2_audit.py",
            generator_version=__version__,
            command="python tools/build_kitti_camera_lidar_v2_audit.py SCORES.json PREREG OUT",
            data_verified=True,
        ),
    )


def _rows(scores: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {
        name: {}
        for name in (
            "drives",
            "evaluation",
            "baseline",
            "axes",
            "coverage",
            "turned",
            "turn3",
            "turn3_judged_right",
            "turn1",
        )
    }
    for drive in scores["drives"]:
        score = scores["scores"][drive]
        rows["evaluation"][drive] = 0.0 if score["state"] == "scored" else "not scored"
        if score["state"] == "failed":
            rows["drives"][drive] = score["reason"] or "failed"
            continue
        if score["state"] != "scored":
            continue
        rows["drives"][drive] = 0.0
        rows["baseline"][drive] = 1.0 if score["verdict"] in ("warn", "fail") else 0.0
        judged = [name for name, axis in score["axes"].items() if axis["judged"]]
        rows["coverage"][drive] = float(len(score["axes"]) - len(judged))
        for name in judged:
            rows["axes"][f"{drive}-{name}"] = abs(score["axes"][name]["error_deg"])
        for spec, trial in score["trials"].items():
            key = f"{drive}-{spec.replace(':', '+')}"
            for name, error in trial["judged_errors_deg"].items():
                rows["turned"][f"{key}-{name}"] = abs(error)
            if not trial["counted"]:
                continue
            degrees = float(spec.split(":")[1])
            flagged: float | str = 0.0 if trial["flagged"] else f"not flagged ({trial['outcome']})"
            if degrees == 3.0:
                rows["turn3"][key] = flagged
                rows["turn3_judged_right"][key] = 1.0 if trial["judged_right"] else 0.0
            else:
                rows["turn1"][key] = flagged
    return rows


BENCHMARKS = {
    "drives": (
        "KITTI camera-lidar check: drives that ran",
        ("drive_failed", "a failed drive fails", "1", "0 for a scored drive"),
        "one split per scored or failed evaluation drive; not-usable drives dropped",
    ),
    "evaluation": (
        "KITTI camera-lidar check: evaluation drives scored",
        ("drive_not_scored", "a drive not scored fails", "1", "0 for a scored drive"),
        "one split per evaluation drive",
    ),
    "baseline": (
        "KITTI camera-lidar check: verdicts on KITTI's own calibration",
        (
            "vendor_flagged",
            "1 when the check flags KITTI's calibration",
            "1",
            "1 for a baseline verdict of warn or fail, else 0",
        ),
        "one split per scored drive",
    ),
    "axes": (
        "KITTI camera-lidar check: judged axes against KITTI's calibration",
        (
            "axis_error_deg_abs",
            "absolute judged-axis error vs KITTI",
            "deg",
            "abs(component of rotvec(R_est R_ref^T)), camera optical axes",
        ),
        "one split per judged rotation axis of a scored drive's baseline run",
    ),
    "coverage": (
        "KITTI camera-lidar check: baseline axes not judged",
        ("unjudged_axes", "number of the drive's rotation axes not judged", "1", "3 - judged"),
        "one split per scored drive",
    ),
    "turned": (
        "KITTI camera-lidar check: judged axes of the turned runs against KITTI's calibration",
        (
            "axis_error_deg_abs",
            "absolute judged-axis error vs KITTI",
            "deg",
            "abs(component of rotvec(R_est R_ref^T)), camera optical axes",
        ),
        "one split per judged rotation axis of every turned run of a scored drive",
    ),
    "turn3": (
        "KITTI camera-lidar check: 3 deg turns of the camera flagged",
        ("turn_flagged", "a turn not flagged fails", "1", "0 for a flagged turn"),
        "one split per 3 deg turn about an axis judged on the drive's baseline run",
    ),
    "turn3_judged_right": (
        "KITTI camera-lidar check: 3 deg turns judged right",
        (
            "turn_judged_right",
            "1 when the check judges the turned axis pass",
            "1",
            "the turned axis's status in the turned run",
        ),
        "one split per 3 deg turn about an axis judged on the drive's baseline run",
    ),
    "turn1": (
        "KITTI camera-lidar check: 1 deg turns of the camera flagged",
        ("turn_flagged", "a turn not flagged fails", "1", "0 for a flagged turn"),
        "one split per 1 deg turn about an axis judged on the drive's baseline run",
    ),
}
# requirement -> (benchmark, metric or None, statistic, threshold, description)
GATES = {
    "no-failed-drives": ("drives", None, "failure_rate", 0.0, "no drive fails"),
    "vendor-calibration-not-flagged": (
        "baseline", "vendor_flagged", "metric_maximum", 0.0,
        "the check flags the day's KITTI calibration on no scored drive",
    ),
    "judged-axis-accuracy": (
        "axes", "axis_error_deg_abs", "metric_maximum", AXIS_ERROR_MAX_DEG,
        f"every judged baseline axis is within {AXIS_ERROR_MAX_DEG:g} deg of KITTI's "
        "calibration",
    ),
    "judged-axis-coverage": (
        "coverage", "unjudged_axes", "metric_mean", UNJUDGED_AXES_MEAN_MAX,
        f"on average at most {UNJUDGED_AXES_MEAN_MAX:g} of a scored drive's 3 baseline axes "
        "are not judged",
    ),
    "turned-judged-axis-accuracy": (
        "turned", "axis_error_deg_abs", "metric_maximum", TURNED_AXIS_ERROR_MAX_DEG,
        f"every axis judged in a turned run is within {TURNED_AXIS_ERROR_MAX_DEG:g} deg of "
        "KITTI's calibration",
    ),
    "turn-3deg-flagged": (
        "turn3", None, "failure_rate", MISSED_3DEG_MAX,
        f"at most {MISSED_3DEG_MAX:g} of the 3 deg turns about judged axes are not flagged",
    ),
    "turn-3deg-never-judged-right": (
        "turn3_judged_right", "turn_judged_right", "metric_maximum", 0.0,
        "the check never judges an axis turned 3 deg to be right",
    ),
    "turn-1deg-flagged": (
        "turn1", None, "failure_rate", MISSED_1DEG_MAX,
        f"at most {MISSED_1DEG_MAX:g} of the 1 deg turns about judged axes are not flagged",
    ),
    "scored-drives-coverage": (
        "evaluation", None, "failure_rate", None,
        f"at least {SCORED_DRIVES_MIN} of the evaluation drives are scored",
    ),
}  # fmt: skip


def build(
    scores: dict[str, Any], scores_path: Path, preregistration: Path, output_dir: Path
) -> Any:
    """Write the benchmarks, the protocol, and the audit; return the audit result."""

    scores_digest = sha256_path(scores_path)
    prereg_digest = sha256_path(preregistration)
    assert scores_digest is not None and prereg_digest is not None
    rows = _rows(scores)
    paths: dict[str, tuple[str, str]] = {}
    for name, (title, metric, policy) in BENCHMARKS.items():
        if not rows[name]:
            continue  # nothing to measure: its requirements stay without evidence
        definition = _definition(
            f"{PROTOCOL_ID}-{name}", title, _metric(*metric), rows[name], scores_digest, policy
        )
        path = output_dir / f"kitti_camera_lidar_v2_{name}_benchmark.yaml"
        aggregate_benchmark_definition(definition).save(path)
        digest = sha256_path(path)
        assert digest is not None
        paths[name] = (path.name, digest)
    drives = len(rows["evaluation"])
    common = {"dataset_family": "kitti_raw", "phase": "audit"}
    requirements = []
    for rid, (benchmark, metric, statistic, threshold, description) in GATES.items():
        if rid == "scored-drives-coverage":
            threshold = (drives - SCORED_DRIVES_MIN) / drives
        status = scores["requirements"][rid]["status"]
        coverage_unmet = rid.endswith("coverage") and status != "pass"
        if status == "not_evaluated" or coverage_unmet or benchmark not in paths:
            requirements.append(
                SotaClaimRequirement(
                    requirement_id=rid,
                    description=f"{description} (not met or not evaluable)",
                    **common,
                )
            )
            continue
        requirements.append(
            SotaClaimRequirement(
                requirement_id=rid,
                description=description,
                evidence_path=paths[benchmark][0],
                evidence_sha256=paths[benchmark][1],
                method_id=METHOD.method_id,
                metric=metric,
                statistic=statistic,
                comparison="less_equal",
                threshold=threshold,
                **common,
            )
        )
    requirements[0] = requirements[0].model_copy(
        update={"independent_rig": True, "rig_id": "kitti-raw-car"}
    )
    protocol = SotaAuditProtocol(
        protocol_id=PROTOCOL_ID,
        scope=SotaClaimScope(
            modalities=["camera", "lidar"],
            quantities=["rotation"],
            category="targetless_camera_lidar_check",
        ),
        claim_text=CLAIM.format(drives=drives),
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=requirements,
        provenance=SotaAuditProvenance(
            generator="tools/build_kitti_camera_lidar_v2_audit.py",
            generator_version=__version__,
            command=["python", "tools/build_kitti_camera_lidar_v2_audit.py", "SCORES.json"],
            source_sha256={"preregistration": prereg_digest, "scores": scores_digest},
        ),
    )
    protocol_path = output_dir / "kitti_camera_lidar_v2_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id=PROTOCOL_ID)
    result.save(output_dir / "kitti_camera_lidar_v2_sota_audit.yaml")
    return result


def main() -> None:
    scores_path, preregistration, output_dir = (Path(arg) for arg in sys.argv[1:4])
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build(scores, scores_path, preregistration, output_dir)
    print(result.verdict)
    for item in result.requirements:
        print(f"  {item.requirement_id}: {item.status} ({item.reason})")


if __name__ == "__main__":
    main()
