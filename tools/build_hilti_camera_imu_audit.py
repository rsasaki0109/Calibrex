"""Assemble the pre-registered Hilti camera-IMU SOTA audit from ``scores.json``.

The gates are fixed by ``docs/benchmarks/hilti_camera_imu_preregistration.yaml``
and mirror ``tools/score_hilti_camera_imu.py``; the preregistration's SHA-256
is pinned in the protocol.  Usage::

    python tools/build_hilti_camera_imu_audit.py \\
        SCORES.json docs/benchmarks/hilti_camera_imu_preregistration.yaml OUT_DIR

Three benchmarks are written, each with the single method ``calibrex_native``
(Kalibr is the reference, not a method):

* ``units``: one split per gated unit that is scored or failed (not-usable
  units are inconclusive and have no split); metrics are the rotation error,
  the absolute time error, and the constrained violation (0 or 1);
* ``consistency``: one split per gated camera;
* ``relative``: one split per evaluation recording; a recording not scored for
  both cameras is a failed trial.

The coverage rule "at least 3 of 4 recordings scored for both cameras" is the
``scored-recordings-coverage`` requirement.  When it is met it is a
failure-rate gate of 0.25 on the ``relative`` benchmark.  When it is not met
the requirement is left without evidence, so the audit is ``incomplete``
(the preregistration's ``inconclusive``) unless another requirement is
contradicted, in which case it is ``refuted``.
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

GATED_CAMERAS = ("cam0", "cam1")
ROTATION_ERROR_MAX_DEG = 1.0
TIME_ERROR_MAX_MS = 0.5
CONSISTENCY_MAX_DEG = 0.75
RELATIVE_MAX_DEG = 0.5
COVERAGE_FAILURE_RATE_MAX = 0.25
METHOD = BenchmarkMethodDefinition(
    method_id="calibrex_native",
    label="Calibrex targetless camera-IMU rotation",
    implementation="calibrex_native",
    tool_name="calibrex",
)
# Benchmarks need two methods.  Kalibr is the reference the metrics are measured
# against, so its own error is zero by definition; it is never gated.
REFERENCE = BenchmarkMethodDefinition(
    method_id="kalibr_reference",
    label="Hilti Kalibr calibration (the reference; error against itself is 0)",
    implementation="calibrex_native",
    tool_name="kalibr",
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
        command="calibrex camera-imu rotation", config_sha256=digest, input_sha256=digest
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
            protocol_id="hilti-camera-imu-v1",
            dataset_id="hilti2022_exp01_exp04_forward_cameras",
            dataset_source_sha256=digest,
            data_license="Hilti SLAM Challenge 2022 terms",
            split_policy=split_policy,
            splits=splits,
            initial_estimate_policy="none; the estimator does not read Kalibr's T_cam_imu",
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy="an estimator failure or policy fail on a usable unit fails the trial",
        ),
        metrics=metrics,
        methods=[REFERENCE, METHOD],
        trials=trials,
        reference_method_id=REFERENCE.method_id,
        bootstrap_samples=2000,
        limitations=[
            "The reference is Hilti's target-based Kalibr calibration, not ground truth.",
            "cam0 and cam1 share a scene and a Kalibr run; their units are not independent.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_hilti_camera_imu_audit.py",
            generator_version=__version__,
            command="python tools/build_hilti_camera_imu_audit.py SCORES.json PREREGISTRATION OUT",
            data_verified=True,
        ),
    )


def build(scores: dict[str, Any], scores_path: Path, preregistration: Path, output_dir: Path):
    """Write the benchmarks, the protocol, and the audit; return the audit result."""

    scores_digest = sha256_path(scores_path)
    prereg_digest = sha256_path(preregistration)
    assert scores_digest is not None and prereg_digest is not None
    unit_rows: dict[str, dict[str, float] | str] = {}
    for camera in GATED_CAMERAS:
        for recording, unit in scores["units"][camera].items():
            key = f"{camera}-{recording}"
            if unit["state"] == "scored":
                unit_rows[key] = {
                    "rotation_error_deg": unit["rotation_error_deg"],
                    "time_error_ms_abs": abs(unit["time_error_ms"]),
                    "constrained_violation": 0.0 if unit["constrained"] else 1.0,
                }
            elif unit["state"] == "failed":
                unit_rows[key] = unit["reason"] or "failed"
    consistency_rows: dict[str, dict[str, float] | str] = {
        camera: (
            {"consistency_deg": value}
            if (value := scores["consistency_deg"][camera]) is not None
            else "fewer than two scored recordings"
        )
        for camera in GATED_CAMERAS
    }
    relative_rows: dict[str, dict[str, float] | str] = {
        recording: ({"relative_deg": value} if value is not None else "not scored for both cameras")
        for recording, value in scores["relative_cam0_cam1_deg"].items()
    }
    definitions = {
        "units": _definition(
            "hilti-camera-imu-units",
            "Hilti camera-IMU rotation against Kalibr, per camera and recording",
            [
                _metric(
                    "rotation_error_deg",
                    "rotation angle error vs Kalibr",
                    "deg",
                    "geodesic angle of R_est R_ref^T",
                    primary=True,
                ),
                _metric(
                    "time_error_ms_abs",
                    "absolute clock-offset difference to Kalibr",
                    "ms",
                    "abs(dt_est - timeshift_cam_imu)",
                ),
                _metric(
                    "constrained_violation",
                    "not constrained by the data",
                    "1",
                    "1 unless all axes are estimated, std <= 0.3 deg, and >= 2 of 3 controls hit",
                ),
            ],
            unit_rows,
            scores_digest,
            "one split per gated camera and evaluation recording; not-usable units dropped",
        ),
        "consistency": _definition(
            "hilti-camera-imu-consistency",
            "Hilti camera-IMU cross-recording consistency",
            [
                _metric(
                    "consistency_deg",
                    "largest pairwise rotation angle across scored recordings",
                    "deg",
                    "per camera, max over pairs of scored recordings of the geodesic angle",
                    primary=True,
                )
            ],
            consistency_rows,
            scores_digest,
            "one split per gated camera",
        ),
        "relative": _definition(
            "hilti-camera-imu-relative",
            "Hilti camera-IMU relative cam0-to-cam1 rotation against Kalibr",
            [
                _metric(
                    "relative_deg",
                    "relative cam0-cam1 rotation error vs Kalibr",
                    "deg",
                    "geodesic angle of (R0 R1^T)(R0ref R1ref^T)^T",
                    primary=True,
                )
            ],
            relative_rows,
            scores_digest,
            "one split per evaluation recording",
        ),
    }
    paths = {}
    for name, definition in definitions.items():
        path = output_dir / f"hilti_camera_imu_{name}_benchmark.yaml"
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

    coverage_met = len(scores["recordings_scored_for_both_cameras"]) >= 3
    coverage = (
        gate(
            "scored-recordings-coverage",
            "at least 3 of 4 recordings scored for both cameras",
            "relative",
            None,
            "failure_rate",
            COVERAGE_FAILURE_RATE_MAX,
        )
        if coverage_met
        else SotaClaimRequirement(
            requirement_id="scored-recordings-coverage",
            description="at least 3 of 4 recordings scored for both cameras (not met)",
            **common,
        )
    )
    first = gate(
        "rotation-vs-kalibr",
        "every scored unit has rotation error <= 1.0 deg",
        "units",
        "rotation_error_deg",
        "metric_maximum",
        ROTATION_ERROR_MAX_DEG,
    )
    first = first.model_copy(update={"independent_rig": True, "rig_id": "hilti2022-alphasense"})
    protocol = SotaAuditProtocol(
        protocol_id="hilti-camera-imu-v1",
        scope=SotaClaimScope(
            modalities=["camera", "imu"],
            quantities=["rotation", "time_offset"],
            category="targetless_camera_imu",
        ),
        claim_text=(
            "On four Hilti 2022 recordings not used to develop the method, Calibrex's "
            "targetless camera-IMU rotation for the two forward cameras agrees with Kalibr "
            "within 1.0 deg and 0.5 ms, is constrained on all three axes, reproduces across "
            "recordings within 0.75 deg, and reproduces Kalibr's cam0-to-cam1 rotation "
            "within 0.5 deg."
        ),
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=[
            first,
            gate(
                "time-offset-vs-kalibr",
                "every scored unit has time error <= 0.5 ms",
                "units",
                "time_error_ms_abs",
                "metric_maximum",
                TIME_ERROR_MAX_MS,
            ),
            gate(
                "constrained",
                "every scored unit is constrained",
                "units",
                "constrained_violation",
                "metric_maximum",
                0.0,
            ),
            gate(
                "no-failed-units",
                "no usable unit fails",
                "units",
                None,
                "failure_rate",
                0.0,
            ),
            gate(
                "cross-recording-consistency",
                "per camera consistency <= 0.75 deg",
                "consistency",
                "consistency_deg",
                "metric_maximum",
                CONSISTENCY_MAX_DEG,
            ),
            gate(
                "relative-cam0-cam1",
                "every recording scored for both cameras has relative error <= 0.5 deg",
                "relative",
                "relative_deg",
                "metric_maximum",
                RELATIVE_MAX_DEG,
            ),
            coverage,
        ],
        provenance=SotaAuditProvenance(
            generator="tools/build_hilti_camera_imu_audit.py",
            generator_version=__version__,
            command=["python", "tools/build_hilti_camera_imu_audit.py", "SCORES.json"],
            source_sha256={"preregistration": prereg_digest, "scores": scores_digest},
        ),
    )
    protocol_path = output_dir / "hilti_camera_imu_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id="hilti-camera-imu-v1")
    result.save(output_dir / "hilti_camera_imu_sota_audit.yaml")
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
