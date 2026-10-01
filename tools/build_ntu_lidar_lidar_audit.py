"""Assemble the pre-registered NTU VIRAL LiDAR-LiDAR extrinsic audit.

Inputs are ``scores.json`` of ``tools/score_ntu_lidar_lidar.py`` for the
evaluation recordings.  Two benchmarks are written: one split per held-out
block (Calibrex against the concurrent-scan baseline and the design value),
and one split per recording (Calibrex's cross-recording consistency).
Requirements, thresholds, and coverage are those of
``docs/benchmarks/ntu_viral_lidar_lidar_preregistration.yaml``, whose SHA-256
is pinned in the protocol.

Usage::

    python tools/build_ntu_lidar_lidar_audit.py \\
        docs/assets/ntu_viral_lidar_lidar/scores.json docs/assets
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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

PREREGISTRATION = Path("docs/benchmarks/ntu_viral_lidar_lidar_preregistration.yaml")
COMMAND = [
    "python",
    "tools/build_ntu_lidar_lidar_audit.py",
    "docs/assets/ntu_viral_lidar_lidar/scores.json",
    "docs/assets",
]
HOLDOUT_METRIC = "holdout_block_chi2"
CONSISTENCY_METRIC = "cross_recording_consistency"
ROTATION_METRIC = "rotation_error_deg"
TRANSLATION_METRIC = "translation_error_m"
ROTATION_THRESHOLD_DEG = 1.0
TRANSLATION_THRESHOLD_M = 0.10
CONSISTENCY_THRESHOLD = 1.0
METHODS = {
    "calibrex_map": BenchmarkMethodDefinition(
        method_id="calibrex_map",
        label="Calibrex LiDAR-LiDAR by map registration",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    "scan_to_scan": BenchmarkMethodDefinition(
        method_id="scan_to_scan",
        label="Concurrent scan-to-scan registration",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    "ntu_design": BenchmarkMethodDefinition(
        method_id="ntu_design",
        label="NTU VIRAL design value inv(T_body_horz) T_body_vert",
        implementation="calibrex_native",
        tool_name="ntu_viral",
    ),
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(
        float(np.linalg.norm(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec()))
    )


def _estimate_gate(scores: dict) -> tuple[bool, str]:
    estimate = scores["estimates"]["calibrex_map"]
    if estimate is None:
        return False, "calibrex_map produced no estimate"
    matrix = np.asarray(estimate["matrix"])
    design = scores["design"]
    rotation = _rotation_error_deg(matrix, np.asarray(design["matrix"]))
    translation = float(np.linalg.norm(matrix[:3, 3] - np.asarray(design["translation_m"])))
    note = (
        f"rotation error to design {rotation:.3f} deg, "
        f"translation error to design {translation:.4f} m"
    )
    ok = rotation <= ROTATION_THRESHOLD_DEG and translation <= TRANSLATION_THRESHOLD_M
    return ok, note


def _accuracy_definition(
    scores: dict, failures: dict[str, str], digest: str
) -> BenchmarkDefinition:
    estimate = scores["estimates"]["calibrex_map"]
    matrix = np.asarray(estimate["matrix"]) if estimate is not None else None
    design = np.asarray(scores["design"]["matrix"])
    rotation = 0.0 if matrix is None else _rotation_error_deg(matrix, design)
    translation = 0.0 if matrix is None else float(np.linalg.norm(matrix[:3, 3] - design[:3, 3]))
    splits = [
        BenchmarkSplit(
            split_id="calibrex_map",
            seed=0,
            fit_count=1,
            holdout_count=1,
            fit_ids_sha256=_digest("calibrex_map"),
            holdout_ids_sha256=_digest("calibrex_map"),
        )
    ]
    provenance = BenchmarkTrialProvenance(
        command="score calibrex_map", config_sha256=digest, input_sha256=digest
    )
    failed = "calibrex_map" in failures
    trials = []
    for method in ("calibrex_map", "ntu_design"):
        if method == "ntu_design":
            metrics = {ROTATION_METRIC: 0.0, TRANSLATION_METRIC: 0.0}
            trials.append(
                BenchmarkTrial(
                    method_id=method,
                    split_id="calibrex_map",
                    status="success",
                    metrics=metrics,
                    provenance=provenance,
                )
            )
        elif failed:
            trials.append(
                BenchmarkTrial(
                    method_id=method,
                    split_id="calibrex_map",
                    status="failed",
                    failure_reason=failures[method],
                    provenance=provenance,
                )
            )
        else:
            trials.append(
                BenchmarkTrial(
                    method_id=method,
                    split_id="calibrex_map",
                    status="success",
                    metrics={ROTATION_METRIC: rotation, TRANSLATION_METRIC: translation},
                    provenance=provenance,
                )
            )
    return BenchmarkDefinition(
        benchmark_id="ntu-lidar-lidar-accuracy",
        title="NTU VIRAL LiDAR-LiDAR extrinsic accuracy against the design value",
        protocol=BenchmarkProtocol(
            protocol_id="ntu-lidar-lidar-v1",
            dataset_id="ntu_viral_evaluation_recordings",
            dataset_source_sha256=digest,
            data_license="CC BY-NC-SA 4.0",
            split_policy="one split: the pooled evaluation extrinsic against the design value",
            splits=splits,
            initial_estimate_policy="fitted methods start from the design value",
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy="a method without an estimate fails every split",
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name=ROTATION_METRIC,
                label="rotation error to the design value",
                unit="deg",
                direction="lower",
                primary=True,
                interpretation="geodesic angle between the fitted and design rotations",
            ),
            BenchmarkMetricDefinition(
                name=TRANSLATION_METRIC,
                label="translation error to the design value",
                unit="m",
                direction="lower",
                primary=False,
                interpretation="Euclidean norm of the fitted minus design translation",
            ),
        ],
        methods=[METHODS["calibrex_map"], METHODS["ntu_design"]],
        trials=trials,
        reference_method_id="ntu_design",
        bootstrap_samples=2000,
        limitations=[
            "The design value is a rounded CAD figure, not an independent measurement.",
            "With a single split the distribution is one number; the gates read that value.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_ntu_lidar_lidar_audit.py",
            generator_version=__version__,
            command=" ".join(COMMAND),
            data_verified=True,
        ),
    )


def _holdout_definition(scores: dict, failures: dict[str, str], digest: str) -> BenchmarkDefinition:
    values = scores["holdout_block_scores"]
    keys = sorted(values["calibrex_map"])
    splits = [
        BenchmarkSplit(
            split_id=key,
            seed=0,
            fit_count=1,
            holdout_count=1,
            fit_ids_sha256=_digest(f"fit-{key}"),
            holdout_ids_sha256=_digest(key),
        )
        for key in keys
    ]
    trials = []
    for method in METHODS:
        provenance = BenchmarkTrialProvenance(
            command=f"score {method}", config_sha256=digest, input_sha256=digest
        )
        for key in keys:
            if method in failures:
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=key,
                        status="failed",
                        failure_reason=failures[method],
                        provenance=provenance,
                    )
                )
            else:
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=key,
                        status="success",
                        metrics={HOLDOUT_METRIC: float(values[method][key])},
                        provenance=provenance,
                    )
                )
    return BenchmarkDefinition(
        benchmark_id="ntu-lidar-lidar-holdout",
        title="NTU VIRAL LiDAR-LiDAR extrinsic, held-out blocks",
        protocol=BenchmarkProtocol(
            protocol_id="ntu-lidar-lidar-v1",
            dataset_id="ntu_viral_evaluation_recordings",
            dataset_source_sha256=digest,
            data_license="CC BY-NC-SA 4.0",
            split_policy="10-second blocks per recording; every third sorted key held out",
            splits=splits,
            initial_estimate_policy=(
                "each method's own start; fitted methods start from the design value"
            ),
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy="a method without an estimate fails every split",
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name=HOLDOUT_METRIC,
                label="held-out point-to-plane chi-square per point",
                unit="1",
                direction="lower",
                primary=True,
                interpretation="mean of min((r / 0.1 m)^2, 25), lost points count 25",
            )
        ],
        methods=[METHODS[method] for method in METHODS],
        trials=trials,
        reference_method_id="scan_to_scan",
        bootstrap_samples=2000,
        limitations=[
            "The reference LiDAR's odometry supplies the map and drifts over the map window.",
            "The reported metric is dominated by the ordinary fraction of target points "
            "that has no map support and is penalized at the clip value.",
            "The design value is a rounded CAD figure, not an independent measurement.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_ntu_lidar_lidar_audit.py",
            generator_version=__version__,
            command=" ".join(COMMAND),
            data_verified=True,
        ),
    )


def _consistency_definition(
    scores: dict, failures: dict[str, str], digest: str
) -> BenchmarkDefinition:
    consistency = scores["consistency"]
    names = sorted(consistency)
    values = {
        "calibrex_map": {name: float(consistency[name]["normalized"]) for name in names},
        "scan_to_scan": {name: float(consistency[name]["normalized"]) for name in names},
        "ntu_design": dict.fromkeys(names, 0.0),
    }
    splits = [
        BenchmarkSplit(
            split_id=name,
            seed=0,
            fit_count=1,
            holdout_count=1,
            fit_ids_sha256=_digest(f"fit-{name}"),
            holdout_ids_sha256=_digest(name),
        )
        for name in names
    ]
    trials = []
    for method in METHODS:
        provenance = BenchmarkTrialProvenance(
            command=f"score {method}", config_sha256=digest, input_sha256=digest
        )
        for name in names:
            if method in failures or not math.isfinite(values[method][name]):
                reason = failures.get(method, "consistency is undefined")
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=name,
                        status="failed",
                        failure_reason=reason,
                        provenance=provenance,
                    )
                )
            else:
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=name,
                        status="success",
                        metrics={CONSISTENCY_METRIC: values[method][name]},
                        provenance=provenance,
                    )
                )
    return BenchmarkDefinition(
        benchmark_id="ntu-lidar-lidar-consistency",
        title="NTU VIRAL LiDAR-LiDAR extrinsic, cross-recording consistency",
        protocol=BenchmarkProtocol(
            protocol_id="ntu-lidar-lidar-v1",
            dataset_id="ntu_viral_evaluation_recordings",
            dataset_source_sha256=digest,
            data_license="CC BY-NC-SA 4.0",
            split_policy="one split per evaluation recording",
            splits=splits,
            initial_estimate_policy=(
                "each method's own start; fitted methods start from the design value"
            ),
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy="a method without an estimate fails every split",
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name=CONSISTENCY_METRIC,
                label="cross-recording extrinsic consistency",
                unit="1",
                direction="lower",
                primary=True,
                interpretation=(
                    "max(rotation / 0.3 deg, translation / 3 cm) to the extrinsic "
                    "fitted on the other recordings"
                ),
            )
        ],
        methods=[METHODS[method] for method in METHODS],
        trials=trials,
        reference_method_id="ntu_design",
        bootstrap_samples=2000,
        limitations=[
            "Consistency compares fits across recordings, not to an independent measurement.",
            "The distribution value for ntu_design is fixed at zero by construction.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_ntu_lidar_lidar_audit.py",
            generator_version=__version__,
            command=" ".join(COMMAND),
            data_verified=True,
        ),
    )


def main() -> None:
    scores_path, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    preregistration = sha256_path(PREREGISTRATION)
    scores_digest = sha256_path(scores_path)
    assert preregistration is not None and scores_digest is not None
    ok, note = _estimate_gate(scores)
    print("estimate requirement:", ok, note)
    failures = {} if ok else {"calibrex_map": "estimate gate failed: " + note}
    accuracy = aggregate_benchmark_definition(_accuracy_definition(scores, failures, scores_digest))
    holdout = aggregate_benchmark_definition(_holdout_definition(scores, failures, scores_digest))
    consistency = aggregate_benchmark_definition(
        _consistency_definition(scores, failures, scores_digest)
    )
    paths = {}
    for name, benchmark in (
        ("accuracy", accuracy),
        ("holdout", holdout),
        ("consistency", consistency),
    ):
        path = output_dir / f"ntu_viral_lidar_lidar_{name}_benchmark.yaml"
        benchmark.save(path)
        digest = sha256_path(path)
        assert digest is not None
        paths[name] = (path.name, digest)
    common = {"dataset_family": "ntu_viral", "phase": "audit"}
    protocol = SotaAuditProtocol(
        protocol_id="ntu-lidar-lidar-v1",
        scope=SotaClaimScope(
            modalities=["lidar", "lidar"],
            quantities=["rotation", "translation"],
            category="targetless_lidar_lidar",
        ),
        claim_text=(
            "On three NTU VIRAL recordings not used to develop the method, Calibrex's "
            "LiDAR-LiDAR extrinsic is accurate against the dataset's design value and "
            "reproduces across recordings."
        ),
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=[
            SotaClaimRequirement(
                requirement_id="calibrex-rotation-vs-design",
                description="Calibrex rotation error to the design value <= 1.0 deg",
                independent_rig=True,
                rig_id="ntu-viral-uav",
                evidence_path=paths["accuracy"][0],
                evidence_sha256=paths["accuracy"][1],
                method_id="calibrex_map",
                metric=ROTATION_METRIC,
                statistic="metric_mean",
                comparison="less_equal",
                threshold=ROTATION_THRESHOLD_DEG,
                **common,
            ),
            SotaClaimRequirement(
                requirement_id="calibrex-translation-vs-design",
                description="Calibrex translation error to the design value <= 0.10 m",
                evidence_path=paths["accuracy"][0],
                evidence_sha256=paths["accuracy"][1],
                method_id="calibrex_map",
                metric=TRANSLATION_METRIC,
                statistic="metric_mean",
                comparison="less_equal",
                threshold=TRANSLATION_THRESHOLD_M,
                **common,
            ),
            SotaClaimRequirement(
                requirement_id="calibrex-not-worse-than-scan-to-scan",
                description=(
                    "paired 95 % CI lower bound of the improvement over scan_to_scan >= 0"
                ),
                evidence_path=paths["holdout"][0],
                evidence_sha256=paths["holdout"][1],
                method_id="calibrex_map",
                reference_method_id="scan_to_scan",
                metric=HOLDOUT_METRIC,
                statistic="paired_improvement_ci95_low",
                comparison="greater_equal",
                threshold=0.0,
                **common,
            ),
            SotaClaimRequirement(
                requirement_id="calibrex-cross-recording-consistency",
                description="Calibrex cross-recording consistency <= 1.0",
                evidence_path=paths["consistency"][0],
                evidence_sha256=paths["consistency"][1],
                method_id="calibrex_map",
                metric=CONSISTENCY_METRIC,
                statistic="metric_mean",
                comparison="less_equal",
                threshold=CONSISTENCY_THRESHOLD,
                **common,
            ),
        ],
        provenance=SotaAuditProvenance(
            generator="tools/build_ntu_lidar_lidar_audit.py",
            generator_version=__version__,
            command=COMMAND,
            source_sha256={"preregistration": preregistration, "scores": scores_digest},
        ),
    )
    protocol_path = output_dir / "ntu_viral_lidar_lidar_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id="ntu-lidar-lidar-v1")
    result.save(output_dir / "ntu_viral_lidar_lidar_sota_audit.yaml")
    print(result.verdict)
    for item in result.requirements:
        print(f"  {item.requirement_id}: {item.status} ({item.reason})")


if __name__ == "__main__":
    main()
