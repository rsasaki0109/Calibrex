#!/usr/bin/env python3
"""Score the frozen GrandTour v2 audit and build schema-valid evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tools.build_grand_tour_lidar_lidar_audit import (
    METHODS,
    _digest,
    chordal_mean,
    same_time_median,
)
from tools.grand_tour_lidar_lidar_v2_evaluate import PREREG

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
from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact
from calibrex.core.provenance import sha256_path
from calibrex.core.sota_audit import (
    SotaAuditProtocol,
    SotaAuditProvenance,
    SotaClaimRequirement,
    SotaClaimScope,
)
from calibrex.evaluation.sota_audit import audit_sota_claim

AUDIT_ID = "grand-tour-lidar-lidar-v2"
ROTATION_FLOOR_DEG = (0.16, 0.03, 0.40)
AXES = ("roll", "pitch", "yaw")
SAME_TIME_BOUND_M = 0.055
ROTATION_RATIO_BOUND = 1.5
SPARSE_COVERAGE_FLOOR = 0.50
GATES = (
    ("dense-complete", "dense", "failure_rate", None, "less_equal", 0.0),
    (
        "dense-same-time",
        "dense",
        "metric_maximum",
        "same_time_median_m",
        "less_equal",
        SAME_TIME_BOUND_M,
    ),
    (
        "dense-rotation-reproducibility",
        "dense",
        "metric_maximum",
        "rotation_ratio",
        "less_equal",
        ROTATION_RATIO_BOUND,
    ),
    ("sparse-honesty", "sparse", "failure_rate", None, "less_equal", 0.0),
    (
        "sparse-axis-coverage",
        "sparse",
        "metric_mean",
        "estimated_fraction",
        "greater_equal",
        SPARSE_COVERAGE_FLOOR,
    ),
)


def rotation_ratio(
    matrix: np.ndarray, std: np.ndarray, peers: list[np.ndarray], peer_std: list[np.ndarray]
) -> float:
    """Largest left-tangent residual relative to independent reported uncertainty."""

    mean = chordal_mean(peers)
    delta = Rotation.from_matrix(matrix[:3, :3] @ mean[:3, :3].T).as_rotvec(degrees=True)
    combined = np.sqrt(std**2 + np.sum(np.asarray(peer_std) ** 2, axis=0) / len(peers) ** 2)
    return float(np.max(np.abs(delta) / combined))


def load_artifact(path: Path) -> LidarLidarExtrinsicArtifact | None:
    return (
        LidarLidarExtrinsicArtifact.model_validate(yaml.safe_load(path.read_text()))
        if path.exists()
        else None
    )


def judged(artifact: LidarLidarExtrinsicArtifact, axes: tuple[str, ...]) -> bool:
    records = {dof.name: dof for dof in artifact.dofs}
    return all(
        name in records
        and records[name].status == "estimated"
        and records[name].known_bad_control is not None
        and records[name].known_bad_control.detected
        for name in axes
    )


def dense_rotation(
    artifact: LidarLidarExtrinsicArtifact | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if (
        artifact is None
        or artifact.transform is None
        or artifact.policy_status == "fail"
        or artifact.samples.holdout_correspondences < 100
        or not judged(artifact, AXES)
    ):
        return None
    mapped = {dof.name: dof.std_reported for dof in artifact.dofs}
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(artifact.transform.rotation_quat_xyzw).as_matrix()
    return matrix, np.maximum(ROTATION_FLOOR_DEG, [mapped[a] for a in AXES])


def score(directory: Path, missions: list[str]) -> dict:
    dense = {m: load_artifact(directory / f"{m}_calibrex_dense.yaml") for m in missions}
    scales = {m: dense_rotation(a) for m, a in dense.items()}
    rows = {}
    for mission in missions:
        eligible = scales[mission]
        peers = [v for key, v in scales.items() if key != mission and v is not None]
        path = directory / f"{mission}_calibrex_dense_same_time.json"
        median = same_time_median(path, "estimate")
        dense_metrics = None
        if (
            eligible is not None
            and len(peers) == len(missions) - 1
            and median is not None
            and math.isfinite(median)
        ):
            matrix, std = eligible
            dense_metrics = {
                "same_time_median_m": median,
                "rotation_ratio": rotation_ratio(
                    matrix, std, [x[0] for x in peers], [x[1] for x in peers]
                ),
            }
        sparse = load_artifact(directory / f"{mission}_calibrex_sparse.yaml")
        sparse_metrics = None
        if (
            sparse is not None
            and sparse.transform is not None
            and sparse.policy_status in ("pass", "inconclusive")
            and sparse.samples.holdout_correspondences >= 100
            and judged(sparse, tuple(sparse.calibrated_dofs))
        ):
            sparse_metrics = {"estimated_fraction": len(sparse.calibrated_dofs) / 6}
        rows[mission] = {
            "dense": {
                "status": dense[mission].policy_status if dense[mission] else "missing",
                "metrics": dense_metrics,
            },
            "sparse": {
                "status": sparse.policy_status if sparse else "missing",
                "metrics": sparse_metrics,
            },
        }
    return rows


def build(directory: Path) -> str:
    prereg = yaml.safe_load(PREREG.read_text())
    pins = {m["mission"]: m["sha256"] for m in prereg["recordings"]["evaluation"]}
    scores = score(directory, list(pins))
    scores_path = directory / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
    pre_sha = sha256_path(PREREG)
    assert pre_sha is not None
    run = json.loads((directory / "runs.json").read_text())
    if run["preregistration_sha256"] != pre_sha or "finished_at" not in run:
        raise ValueError("scores require a completed run of the frozen pre-registration")
    benchmark_paths = {}
    for method, metrics in {
        "dense": [
            BenchmarkMetricDefinition(
                name="same_time_median_m",
                label="Same-time median",
                unit="m",
                direction="lower",
                primary=True,
                interpretation="median full-density point-to-plane residual",
            ),
            BenchmarkMetricDefinition(
                name="rotation_ratio",
                label="Largest standardized rotation deviation",
                unit="ratio",
                direction="lower",
                interpretation="leave-one-mission-out deviation divided by combined uncertainty",
            ),
        ],
        "sparse": [
            BenchmarkMetricDefinition(
                name="estimated_fraction",
                label="Estimated axes with detected controls",
                unit="fraction",
                direction="higher",
                primary=True,
                interpretation="estimated axes divided by six",
            )
        ],
    }.items():
        method_id = "calibrex_" + method
        reference_id = "frozen_gate_targets"
        targets = (
            {"same_time_median_m": SAME_TIME_BOUND_M, "rotation_ratio": ROTATION_RATIO_BOUND}
            if method == "dense"
            else {"estimated_fraction": SPARSE_COVERAGE_FLOOR}
        )
        trials = []
        for mission, items in scores.items():
            item = items[method]
            provenance = BenchmarkTrialProvenance(
                command=f"{AUDIT_ID} {mission} {method_id}",
                config_sha256=pre_sha,
                input_sha256=_digest(json.dumps(pins[mission], sort_keys=True)),
            )
            trials.append(
                BenchmarkTrial(
                    method_id=method_id,
                    split_id=mission,
                    provenance=provenance,
                    status="success" if item["metrics"] is not None else "failed",
                    metrics=item["metrics"] or {},
                    failure_reason=None
                    if item["metrics"] is not None
                    else f"missing or invalid required evidence ({item['status']})",
                )
            )
            trials.append(
                BenchmarkTrial(
                    method_id=reference_id,
                    split_id=mission,
                    provenance=provenance,
                    status="success",
                    metrics=targets,
                )
            )
        benchmark = BenchmarkDefinition(
            benchmark_id=f"{AUDIT_ID}-{method}",
            title=f"GrandTour LiDAR-LiDAR v2 {method}",
            protocol=BenchmarkProtocol(
                protocol_id=AUDIT_ID,
                dataset_id="grand_tour_evaluation_missions_v2",
                dataset_source_sha256=_digest(json.dumps(pins, sort_keys=True)),
                data_license="MIT",
                split_policy="one split per frozen mission",
                splits=[
                    BenchmarkSplit(
                        split_id=m,
                        seed=0,
                        fit_count=1,
                        holdout_count=1,
                        fit_ids_sha256=_digest(m),
                        holdout_ids_sha256=_digest(m),
                    )
                    for m in pins
                ],
                initial_estimate_policy="seed from dataset static tf",
                tuning_policy="frozen before accessing evaluation sensor data",
                failure_policy=prereg["failure_policy"],
            ),
            metrics=metrics,
            methods=[
                METHODS[method_id].model_copy(update={"label": f"Calibrex GrandTour v2 {method}"}),
                BenchmarkMethodDefinition(
                    method_id=reference_id,
                    label="Fixed protocol gate targets",
                    implementation="imported",
                    tool_name="preregistration",
                ),
            ],
            trials=trials,
            reference_method_id=reference_id,
            limitations=["Fixed gate targets are constants supplied to the benchmark schema."],
            bootstrap_samples=2000,
            provenance=BenchmarkProvenance(
                generator=__file__,
                generator_version=__version__,
                command=f"python -m tools.build_grand_tour_lidar_lidar_v2_audit {directory}",
                data_verified=True,
            ),
        )
        path = directory / f"{AUDIT_ID}-{method}-benchmark.yaml"
        aggregate_benchmark_definition(benchmark).save(path)
        benchmark_paths[method] = path
    requirements = []
    for req_id, method, statistic, metric, comparison, threshold in GATES:
        path = benchmark_paths[method]
        requirements.append(
            SotaClaimRequirement(
                requirement_id=req_id,
                phase="audit",
                description=req_id.replace("-", " "),
                dataset_family="grand_tour",
                independent_rig=req_id == "dense-complete",
                rig_id="anymal-d-boxi",
                evidence_path=path.name,
                evidence_sha256=sha256_path(path),
                method_id=f"calibrex_{method}",
                statistic=statistic,
                metric=metric,
                comparison=comparison,
                threshold=threshold,
            )
        )
    protocol = SotaAuditProtocol(
        protocol_id=AUDIT_ID,
        scope=SotaClaimScope(**prereg["scope"]),
        claim_text=prereg["claim"],
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=requirements,
        provenance=SotaAuditProvenance(
            generator=__file__,
            generator_version=__version__,
            command=["python", "-m", "tools.build_grand_tour_lidar_lidar_v2_audit", str(directory)],
            source_sha256={
                "preregistration": pre_sha,
                "scores": sha256_path(scores_path),
                "runs": sha256_path(directory / "runs.json"),
            },
        ),
    )
    protocol_path = directory / f"{AUDIT_ID}-protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id=AUDIT_ID)
    result.save(directory / f"{AUDIT_ID}-audit.yaml")
    print(result.verdict)
    for item in result.requirements:
        print(item.requirement_id, item.status, item.reason)
    return str(result.verdict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory)
