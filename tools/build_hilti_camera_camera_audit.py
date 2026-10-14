#!/usr/bin/env python3
"""Assemble the pre-registered Hilti camera-camera SOTA audit from ``scores.json``.

The gates are fixed by ``docs/benchmarks/hilti_camera_sota_preregistration.yaml``
(claim ``camera-camera``) and mirror ``tools/score_hilti_camera_camera.py``; the
pre-registration's SHA-256 is pinned in the protocol::

    python tools/build_hilti_camera_camera_audit.py \\
        SCORES.json docs/benchmarks/hilti_camera_sota_preregistration.yaml OUT_DIR

Four benchmarks are written, each with the single method ``calibrex_check`` (Kalibr is
the reference, not a method):

* ``units``: one split per gated unit that is scored or failed (an inconclusive unit
  has no split); the metric is the largest absolute judged-axis error against Kalibr;
* ``unit-coverage``: one split per gated unit; a unit that is not scored is a failed
  trial;
* ``trials``: one split per evaluable injection trial; the metrics are a missed
  detection and a missed localization (0 or 1);
* ``trial-coverage``: one split per trial; a trial that is not evaluable is a failed
  trial.

A coverage requirement whose scorer gate fails is left without evidence, so the audit
is ``incomplete`` (the pre-registration's ``inconclusive``) unless another requirement
is contradicted, in which case it is ``refuted``.
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

PROTOCOL_ID = "hilti-camera-camera-v1"
ACCURACY_MAX_DEG = 0.75
UNSCORED_FRACTION_MAX = 0.3
MISSED_DETECTION_MEAN_MAX = 0.1
MISSED_LOCALIZATION_MEAN_MAX = 0.5
UNEVALUABLE_FRACTION_MAX = 0.3
METHOD = BenchmarkMethodDefinition(
    method_id="calibrex_check",
    label="Calibrex targetless camera-camera check",
    implementation="calibrex_native",
    tool_name="calibrex",
)
# Benchmarks need two methods. Kalibr is the reference the metrics are measured against,
# so its own error is zero by definition; it is never gated.
REFERENCE = BenchmarkMethodDefinition(
    method_id="kalibr_reference",
    label="Hilti Kalibr calibration (the reference; error against itself is 0)",
    implementation="calibrex_native",
    tool_name="kalibr",
)
CLAIM = (
    "On twelve Hilti 2022 recordings not used in development, Calibrex's targetless "
    "camera-camera check reproduces Kalibr's rotation of five camera pairs within 0.75 deg "
    "on every judged axis, scores at least 70 percent of them, flags a 1 deg turn of a "
    "forward or a side camera in at least 90 percent of evaluable trials, and names the "
    "turned camera in at least half."
)


def _metric(name: str, label: str, unit: str, interpretation: str, primary: bool = False):
    return BenchmarkMetricDefinition(
        name=name,
        label=label,
        unit=unit,
        direction="lower",
        primary=primary,
        interpretation=interpretation,
    )


def _definition(
    benchmark_id: str,
    title: str,
    metrics: list[BenchmarkMetricDefinition],
    rows: dict[str, dict[str, float] | str],
    digest: str,
    split_policy: str,
) -> BenchmarkDefinition:
    """``rows`` maps a split id to its metric values, or to a failure reason string."""

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
        command="calibrex check --pairs camera-camera", config_sha256=digest, input_sha256=digest
    )
    reference_trials = [
        BenchmarkTrial(
            method_id=REFERENCE.method_id,
            split_id=key,
            status="success",
            metrics={metric.name: 0.0 for metric in metrics},
            provenance=provenance,
        )
        for key in rows
    ]
    trials = reference_trials + [
        BenchmarkTrial(
            method_id=METHOD.method_id,
            split_id=key,
            status="failed" if isinstance(value, str) else "success",
            metrics={} if isinstance(value, str) else value,
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
            dataset_id="hilti2022_unused_recordings_first_5gb",
            dataset_source_sha256=digest,
            data_license="Hilti SLAM Challenge 2022 terms (CC BY-NC-SA 3.0)",
            split_policy=split_policy,
            splits=splits,
            initial_estimate_policy=(
                "the candidate is Kalibr's calibration (or it turned by the injection); the "
                "check re-centres its rectification on its own estimate"
            ),
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy="a crashed run or a missing or skipped gated pair fails its units",
        ),
        metrics=metrics,
        methods=[REFERENCE, METHOD],
        trials=trials,
        reference_method_id=REFERENCE.method_id,
        bootstrap_samples=2000,
        limitations=[
            "The reference is Hilti's target-based Kalibr calibration, not ground truth.",
            "The pairs of one recording share cameras and a scene; they are not independent.",
            "The three exp23 parts are consecutive parts of one session.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_hilti_camera_camera_audit.py",
            generator_version=__version__,
            command=(
                "python tools/build_hilti_camera_camera_audit.py SCORES.json PREREGISTRATION OUT"
            ),
            data_verified=True,
        ),
    )


def build(scores: dict[str, Any], scores_path: Path, preregistration: Path, output_dir: Path):
    """Write the benchmarks, the protocol, and the audit; return the audit result."""

    scores_digest = sha256_path(scores_path)
    prereg_digest = sha256_path(preregistration)
    assert scores_digest is not None and prereg_digest is not None
    unit_rows: dict[str, dict[str, float] | str] = {}
    coverage_rows: dict[str, dict[str, float] | str] = {}
    for recording, units in scores["units"].items():
        for pair, unit in units.items():
            key = f"{recording}-{pair}"
            if unit["state"] == "scored":
                unit_rows[key] = {"max_abs_error_deg": unit["max_abs_error_deg"]}
                coverage_rows[key] = {"scored": 1.0}
            else:
                coverage_rows[key] = f"{unit['state']}: {unit['reason']}"
                if unit["state"] == "failed":
                    unit_rows[key] = unit["reason"] or "failed"
    trial_rows: dict[str, dict[str, float] | str] = {}
    trial_coverage_rows: dict[str, dict[str, float] | str] = {}
    for recording, trials in scores["trials"].items():
        for name, trial in trials.items():
            key = f"{recording}-{name}"
            if trial["evaluable"]:
                trial_rows[key] = {
                    "missed_detection": 0.0 if trial["detected"] else 1.0,
                    "missed_localization": 0.0 if trial["localized"] else 1.0,
                }
                trial_coverage_rows[key] = {"evaluable": 1.0}
            else:
                trial_coverage_rows[key] = f"not evaluable: {trial['reason']}"
    definitions = {
        "units": _definition(
            "hilti-camera-camera-units",
            "Hilti camera-camera rotation against Kalibr, per gated pair and recording",
            [
                _metric(
                    "max_abs_error_deg",
                    "largest absolute judged-axis error vs Kalibr",
                    "deg",
                    "max over the pair's judged axes of |candidate - estimate|",
                    primary=True,
                )
            ],
            unit_rows,
            scores_digest,
            "one split per gated pair and recording; inconclusive units dropped",
        ),
        "unit-coverage": _definition(
            "hilti-camera-camera-unit-coverage",
            "Hilti camera-camera gated units that were scored",
            [_metric("scored", "unit scored", "1", "1 for a scored unit", primary=True)],
            coverage_rows,
            scores_digest,
            "one split per gated pair and recording; a unit that is not scored fails",
        ),
        "trials": _definition(
            "hilti-camera-camera-trials",
            "Hilti camera-camera detection and localization of a 1 deg camera turn",
            [
                _metric(
                    "missed_detection",
                    "turn not flagged",
                    "1",
                    "1 unless a gated pair with the turned camera reads warn or fail",
                    primary=True,
                ),
                _metric(
                    "missed_localization",
                    "turned camera not named",
                    "1",
                    "1 unless the attribution is attributed to the turned camera",
                ),
            ],
            trial_rows,
            scores_digest,
            "one split per evaluable injection and recording",
        ),
        "trial-coverage": _definition(
            "hilti-camera-camera-trial-coverage",
            "Hilti camera-camera injection trials that were evaluable",
            [
                _metric(
                    "evaluable", "trial evaluable", "1", "1 for an evaluable trial", primary=True
                )
            ],
            trial_coverage_rows,
            scores_digest,
            "one split per injection and recording; a trial that is not evaluable fails",
        ),
    }
    paths = {}
    for name, definition in definitions.items():
        path = output_dir / f"hilti_camera_camera_{name.replace('-', '_')}_benchmark.yaml"
        aggregate_benchmark_definition(definition).save(path)
        digest = sha256_path(path)
        assert digest is not None
        paths[name] = (path.name, digest)
    common = {"dataset_family": "hilti2022", "phase": "audit"}

    def gate(rid, description, name, metric, statistic, threshold):
        return SotaClaimRequirement(
            requirement_id=rid,
            description=description,
            evidence_path=paths[name][0],
            evidence_sha256=paths[name][1],
            method_id=METHOD.method_id,
            metric=metric,
            statistic=statistic,
            comparison="less_equal",
            threshold=threshold,
            **common,
        )

    def coverage(rid, description, name, threshold):
        if scores["requirements"][rid]["status"] == "pass":
            return gate(rid, description, name, None, "failure_rate", threshold)
        return SotaClaimRequirement(
            requirement_id=rid, description=f"{description} (not met)", **common
        )

    first = gate(
        "accuracy-vs-kalibr",
        "every scored unit has every judged axis within 0.75 deg of Kalibr",
        "units",
        "max_abs_error_deg",
        "metric_maximum",
        ACCURACY_MAX_DEG,
    )
    first = first.model_copy(update={"independent_rig": True, "rig_id": "hilti2022-alphasense"})
    requirements = [
        first,
        gate("no-failed-units", "no gated unit fails", "units", None, "failure_rate", 0.0),
        coverage(
            "scored-units-coverage",
            "at least 70 percent of gated units are scored",
            "unit-coverage",
            UNSCORED_FRACTION_MAX,
        ),
    ]
    if trial_rows:
        requirements += [
            gate(
                "detection",
                "at least 90 percent of evaluable trials flag the turn",
                "trials",
                "missed_detection",
                "metric_mean",
                MISSED_DETECTION_MEAN_MAX,
            ),
            gate(
                "localization",
                "at least 50 percent of evaluable trials name the turned camera",
                "trials",
                "missed_localization",
                "metric_mean",
                MISSED_LOCALIZATION_MEAN_MAX,
            ),
        ]
    else:
        requirements += [
            SotaClaimRequirement(
                requirement_id=rid, description=f"{rid} (no evaluable trial)", **common
            )
            for rid in ("detection", "localization")
        ]
    requirements.append(
        coverage(
            "evaluable-trials-coverage",
            "at least 70 percent of injection trials are evaluable",
            "trial-coverage",
            UNEVALUABLE_FRACTION_MAX,
        )
    )
    protocol = SotaAuditProtocol(
        protocol_id=PROTOCOL_ID,
        scope=SotaClaimScope(
            modalities=["camera", "camera"],
            quantities=["rotation"],
            category="targetless_camera_camera",
        ),
        claim_text=CLAIM,
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=requirements,
        provenance=SotaAuditProvenance(
            generator="tools/build_hilti_camera_camera_audit.py",
            generator_version=__version__,
            command=["python", "tools/build_hilti_camera_camera_audit.py", "SCORES.json"],
            source_sha256={"preregistration": prereg_digest, "scores": scores_digest},
        ),
    )
    protocol_path = output_dir / "hilti_camera_camera_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id=PROTOCOL_ID)
    result.save(output_dir / "hilti_camera_camera_sota_audit.yaml")
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
