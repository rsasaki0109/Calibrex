"""Assemble the pre-registered Hilti camera-IMU v3 SOTA audit from ``scores.json``.

The gates are fixed by ``docs/benchmarks/hilti_camera_imu_v3_preregistration.yaml`` and
mirror ``tools/score_hilti_camera_imu_v3.py``; the preregistration's SHA-256 is pinned in
the protocol. Usage::

    python tools/build_hilti_camera_imu_v3_audit.py \\
        SCORES.json docs/benchmarks/hilti_camera_imu_v3_preregistration.yaml OUT_DIR

Six benchmarks are written, each with the single method ``calibrex_native`` (Kalibr is
the reference, not a method):

* ``axes``: one split per judged rotation axis of a scored unit; metric the absolute
  error to Kalibr;
* ``time``: one split per scored unit whose clock offset is judged;
* ``units``: one split per scored or failed unit (a not-usable unit has none); metric the
  number of the unit's axes not judged (a count, so that a mean of exactly 1.5 is exact);
  a failed unit is a failed trial;
* ``consistency``: one split per camera and axis judged on at least two recordings;
* ``relative``: one split per recording and axis judged for both cameras;
* ``recordings``: one split per evaluation recording; one not scored for both cameras is
  a failed trial.

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

PROTOCOL_ID = "hilti-camera-imu-v3"
GATED_CAMERAS = ("cam0", "cam1")
AXIS_ERROR_MAX_DEG = 1.0
TIME_ERROR_MAX_MS = 0.75
UNJUDGED_AXES_MEAN_MAX = 1.5  # judged-axis coverage: at least half of the 3 axes judged
CONSISTENCY_MAX_DEG = 1.0
RELATIVE_MAX_DEG = 0.75
SCORED_RECORDINGS_MIN = 3
CLAIM = (
    "On five Hilti 2022 recordings on which no camera-IMU estimate had been made, "
    "Calibrex's targetless camera-IMU rotation for the two forward Alphasense cameras "
    "(cam0, cam1) gives honest per-axis verdicts against Hilti's Kalibr calibration: every "
    "rotation axis it judges (reported std at most 0.3 deg and its 1 deg known-bad control "
    "detected on held-out windows) is within 1.0 deg, every clock offset it judges within "
    "0.75 ms; it judges at least half of the axes; an axis judged on several recordings "
    "spans at most 1.0 deg; and an axis judged for both cameras reproduces Kalibr's "
    "cam0-to-cam1 rotation within 0.75 deg."
)
METHOD = BenchmarkMethodDefinition(
    method_id="calibrex_native",
    label="Calibrex targetless camera-IMU rotation",
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
        command="calibrex camera-imu rotation", config_sha256=digest, input_sha256=digest
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
            dataset_id="hilti2022_camera_imu_unused_forward_cameras",
            dataset_source_sha256=digest,
            data_license="Hilti SLAM Challenge 2022 terms",
            split_policy=split_policy,
            splits=splits,
            initial_estimate_policy="none; the estimator does not read Kalibr's T_cam_imu",
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy="an estimator failure or policy fail on a usable unit fails the trial",
        ),
        metrics=[metric],
        methods=[REFERENCE, METHOD],
        trials=trials,
        reference_method_id=REFERENCE.method_id,
        bootstrap_samples=2000,
        limitations=[
            "The reference is Hilti's target-based Kalibr calibration, not ground truth.",
            "cam0 and cam1 share a scene and a Kalibr run; their units are not independent.",
            "Only the axes the method judges are gated; the others are counted by coverage.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_hilti_camera_imu_v3_audit.py",
            generator_version=__version__,
            command="python tools/build_hilti_camera_imu_v3_audit.py SCORES.json PREREG OUT",
            data_verified=True,
        ),
    )


def _rows(scores: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    axes: dict[str, float | str] = {}
    time: dict[str, float | str] = {}
    units: dict[str, float | str] = {}
    for camera in GATED_CAMERAS:
        for recording, unit in scores["units"][camera].items():
            key = f"{camera}-{recording}"
            if unit["state"] == "failed":
                units[key] = unit["reason"] or "failed"
                continue
            if unit["state"] != "scored":
                continue
            judged = [name for name, axis in unit["axes"].items() if axis["judged"]]
            units[key] = float(len(unit["axes"]) - len(judged))
            for name in judged:
                axes[f"{key}-{name}"] = abs(unit["axes"][name]["error_deg"])
            if unit["time"]["judged"]:
                time[key] = abs(unit["time"]["error_ms"])
    both = set(scores["recordings_scored_for_both_cameras"])
    return {
        "axes": axes,
        "time": time,
        "units": units,
        "consistency": {key: item["range_deg"] for key, item in scores["consistency"].items()},
        "relative": {
            key: abs(item["relative_deg"]) for key, item in scores["relative_cam0_cam1"].items()
        },
        "recordings": {
            recording: (0.0 if recording in both else "not scored for both cameras")
            for recording in scores["recordings"]
        },
    }


BENCHMARKS = {
    "axes": (
        "Hilti camera-IMU v3: judged rotation axes against Kalibr",
        (
            "axis_error_deg_abs",
            "absolute judged-axis error vs Kalibr",
            "deg",
            "abs(component of rotvec(R_est R_ref^T)), camera axes",
        ),
        "one split per judged axis of a scored gated unit",
    ),
    "time": (
        "Hilti camera-IMU v3: judged clock offsets against Kalibr",
        (
            "time_error_ms_abs",
            "absolute clock-offset difference to Kalibr",
            "ms",
            "abs(dt_est - timeshift_cam_imu)",
        ),
        "one split per scored gated unit whose clock offset is judged",
    ),
    "units": (
        "Hilti camera-IMU v3: units and the number of their axes not judged",
        (
            "unjudged_axes",
            "number of the unit's rotation axes not judged",
            "1",
            "3 - judged axes",
        ),
        "one split per scored or failed gated unit; not-usable units dropped",
    ),
    "consistency": (
        "Hilti camera-IMU v3: judged-axis consistency across recordings",
        (
            "range_deg",
            "span of the judged-axis errors across recordings",
            "deg",
            "per camera and axis, max - min over the recordings that judge it",
        ),
        "one split per camera and axis judged on at least two recordings",
    ),
    "relative": (
        "Hilti camera-IMU v3: relative cam0-to-cam1 rotation per judged axis",
        (
            "relative_axis_deg_abs",
            "relative cam0-cam1 error vs Kalibr, one axis",
            "deg",
            "abs(component of rotvec((R0 R1^T)(R0ref R1ref^T)^T)), cam0 axes",
        ),
        "one split per recording and axis judged for both cameras",
    ),
    "recordings": (
        "Hilti camera-IMU v3: recordings scored for both cameras",
        (
            "cameras_not_scored",
            "a recording not scored for both cameras fails",
            "1",
            "0 for a recording scored for both cameras",
        ),
        "one split per evaluation recording",
    ),
}
# requirement -> (benchmark, metric or None, statistic, threshold, description)
GATES = {
    "judged-axis-accuracy": (
        "axes", "axis_error_deg_abs", "metric_maximum", AXIS_ERROR_MAX_DEG,
        "every judged axis is within 1.0 deg of Kalibr",
    ),
    "judged-time-offset": (
        "time", "time_error_ms_abs", "metric_maximum", TIME_ERROR_MAX_MS,
        "every judged clock offset is within 0.75 ms of Kalibr",
    ),
    "judged-axis-coverage": (
        "units", "unjudged_axes", "metric_mean", UNJUDGED_AXES_MEAN_MAX,
        "at least half of the scored units' axes are judged",
    ),
    "no-failed-units": ("units", None, "failure_rate", 0.0, "no usable unit fails"),
    "judged-axis-consistency": (
        "consistency", "range_deg", "metric_maximum", CONSISTENCY_MAX_DEG,
        "an axis judged on several recordings spans at most 1.0 deg",
    ),
    "judged-axis-relative": (
        "relative", "relative_axis_deg_abs", "metric_maximum", RELATIVE_MAX_DEG,
        "an axis judged for both cameras reproduces Kalibr's cam0-cam1 rotation within 0.75 deg",
    ),
    "scored-recordings-coverage": (
        "recordings", None, "failure_rate", None,
        "at least 3 of the evaluation recordings are scored for both cameras",
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
        path = output_dir / f"hilti_camera_imu_v3_{name}_benchmark.yaml"
        aggregate_benchmark_definition(definition).save(path)
        digest = sha256_path(path)
        assert digest is not None
        paths[name] = (path.name, digest)
    recordings = len(rows["recordings"])
    common = {"dataset_family": "hilti2022", "phase": "audit"}
    requirements = []
    for rid, (benchmark, metric, statistic, threshold, description) in GATES.items():
        if rid == "scored-recordings-coverage":
            threshold = (recordings - SCORED_RECORDINGS_MIN) / recordings
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
        update={"independent_rig": True, "rig_id": "hilti2022-alphasense"}
    )
    protocol = SotaAuditProtocol(
        protocol_id=PROTOCOL_ID,
        scope=SotaClaimScope(
            modalities=["camera", "imu"],
            quantities=["rotation", "time_offset"],
            category="targetless_camera_imu",
        ),
        claim_text=CLAIM,
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=requirements,
        provenance=SotaAuditProvenance(
            generator="tools/build_hilti_camera_imu_v3_audit.py",
            generator_version=__version__,
            command=["python", "tools/build_hilti_camera_imu_v3_audit.py", "SCORES.json"],
            source_sha256={"preregistration": prereg_digest, "scores": scores_digest},
        ),
    )
    protocol_path = output_dir / "hilti_camera_imu_v3_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id=PROTOCOL_ID)
    result.save(output_dir / "hilti_camera_imu_v3_sota_audit.yaml")
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
