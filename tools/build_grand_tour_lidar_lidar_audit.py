#!/usr/bin/env python3
"""Score the pre-registered GrandTour LiDAR-LiDAR evaluation and build its SOTA audit.

An evaluation harness, not part of the Calibrex package. Its inputs are the directory
written by ``tools/grand_tour_lidar_lidar_evaluate.py`` and the pre-registration, whose
SHA-256 is pinned in the protocol. Every threshold is a module constant that the
pre-registration repeats; ``tests/unit/test_grand_tour_lidar_lidar_preregistration.py``
keeps the two equal.

Two benchmarks are written, each split by mission:

* ``dense``: ``calibrex_dense`` (Hesai XT-32 -> MID360, default options) against the
  dataset's static tf. Metrics:
    - the same-time median point-to-plane distance;
    - cross-mission consistency, ``max(rotation / 0.3 deg, translation / 3 cm)`` between a
      mission's estimate and the chordal mean of the other missions' estimates. The tf is
      one value for every mission, so its consistency is 0 by construction.
* ``sparse``: ``calibrex_sparse`` (the MID360 thinned to an L1-like 2,000 points within
  20 m, per-point reference trajectory, accumulated scans) against ``sparse_rigid`` (the
  same thinned target as rigid single scans). Metrics:
    - rotation and translation to the same mission's ``calibrex_dense`` estimate;
    - the same-time median.

A Calibrex trial fails when its run has no estimate or its verdict is not ``pass``
(``fail``, ``warn`` or ``inconclusive``). A sparse trial also fails when its mission has
no dense estimate to compare against.

    python tools/build_grand_tour_lidar_lidar_audit.py \\
        docs/assets/grand_tour_lidar_lidar/evaluation
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
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

PREREGISTRATION = Path("docs/benchmarks/grand_tour_lidar_lidar_preregistration.yaml")
AUDIT_ID = "grand-tour-lidar-lidar-v1"
CONSISTENCY_ROTATION_DEG = 0.3
CONSISTENCY_TRANSLATION_M = 0.03
CONSISTENCY_THRESHOLD = 1.0
SAME_TIME_THRESHOLD_M = 0.035
SPARSE_ROTATION_THRESHOLD_DEG = 0.25
SPARSE_TRANSLATION_THRESHOLD_M = 0.01
SAME_TIME = "same_time_median_m"
CONSISTENCY = "cross_mission_consistency"
ROTATION = "rotation_to_dense_deg"
TRANSLATION = "translation_to_dense_m"
METHODS = {
    "calibrex_dense": BenchmarkMethodDefinition(
        method_id="calibrex_dense",
        label="Calibrex lidar-lidar, Hesai XT-32 -> MID360, default options",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    "dataset_tf": BenchmarkMethodDefinition(
        method_id="dataset_tf",
        label="GrandTour static tf inv(T_box_hesai) T_box_livox",
        implementation="calibrex_native",
        tool_name="grand_tour",
    ),
    "calibrex_sparse": BenchmarkMethodDefinition(
        method_id="calibrex_sparse",
        label="Calibrex lidar-lidar, L1-like MID360, per-point trajectory, accumulated",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    "sparse_rigid": BenchmarkMethodDefinition(
        method_id="sparse_rigid",
        label="Calibrex lidar-lidar, L1-like MID360, rigid single scans",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def artifact_matrix(path: Path) -> tuple[np.ndarray | None, str]:
    """``(T_reference_target or None, policy_status)`` of a lidar-lidar artifact."""

    if not path.exists():
        return None, "missing"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    transform = data.get("transform")
    if transform is None:
        return None, str(data.get("policy_status"))
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(transform["rotation_quat_xyzw"]).as_matrix()
    matrix[:3, 3] = transform["translation_m"]
    return matrix, str(data["policy_status"])


def rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(
        float(np.linalg.norm(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec()))
    )


def chordal_mean(matrices: list[np.ndarray]) -> np.ndarray:
    """Rotation by the chordal L2 mean, translation by the arithmetic mean."""

    total = np.sum([matrix[:3, :3] for matrix in matrices], axis=0)
    left, _, right = np.linalg.svd(total)
    rotation = left @ np.diag([1.0, 1.0, np.linalg.det(left @ right)]) @ right
    mean = np.eye(4)
    mean[:3, :3] = rotation
    mean[:3, 3] = np.mean([matrix[:3, 3] for matrix in matrices], axis=0)
    return mean


def consistency(matrix: np.ndarray, others: list[np.ndarray]) -> float:
    mean = chordal_mean(others)
    return max(
        rotation_deg(matrix, mean) / CONSISTENCY_ROTATION_DEG,
        float(np.linalg.norm(matrix[:3, 3] - mean[:3, 3])) / CONSISTENCY_TRANSLATION_M,
    )


def same_time_median(path: Path, candidate: str) -> float | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    values = [row[candidate]["median_point_to_plane_m"] for row in rows]
    return float(np.median(values)) if values else None


def score(directory: Path, missions: list[str]) -> dict[str, Any]:
    """Per-mission metrics and failure reasons for every method."""

    dense: dict[str, np.ndarray] = {}
    status: dict[tuple[str, str], str] = {}
    matrices: dict[tuple[str, str], np.ndarray | None] = {}
    for mission in missions:
        for method in ("calibrex_dense", "calibrex_sparse", "sparse_rigid"):
            matrix, policy = artifact_matrix(directory / f"{mission}_{method}.yaml")
            matrices[(mission, method)] = matrix
            status[(mission, method)] = policy
        if matrices[(mission, "calibrex_dense")] is not None:
            dense[mission] = matrices[(mission, "calibrex_dense")]  # type: ignore[assignment]
    scores: dict[str, Any] = {"missions": {}}
    for mission in missions:
        entry: dict[str, Any] = {}
        tf_same_time = same_time_median(
            directory / f"{mission}_calibrex_dense_same_time.json", "dataset_tf"
        )
        entry["dataset_tf"] = {
            "metrics": {SAME_TIME: tf_same_time, CONSISTENCY: 0.0},
            "failure": None if tf_same_time is not None else "no same-time check",
        }
        for method in ("calibrex_dense", "calibrex_sparse", "sparse_rigid"):
            matrix = matrices[(mission, method)]
            policy = status[(mission, method)]
            failure = None
            metrics: dict[str, float | None] = {
                SAME_TIME: same_time_median(
                    directory / f"{mission}_{method}_same_time.json", "estimate"
                )
            }
            if matrix is None:
                failure = f"no estimate (verdict {policy})"
            elif policy != "pass":
                failure = f"verdict {policy}"
            if method == "calibrex_dense" and matrix is not None:
                others = [dense[item] for item in dense if item != mission]
                metrics[CONSISTENCY] = consistency(matrix, others) if others else None
                if not others:
                    failure = failure or "no other mission to compare against"
            if method != "calibrex_dense":
                reference = dense.get(mission)
                if reference is None:
                    failure = failure or "the mission has no dense estimate"
                elif matrix is not None:
                    metrics[ROTATION] = rotation_deg(matrix, reference)
                    metrics[TRANSLATION] = float(np.linalg.norm(matrix[:3, 3] - reference[:3, 3]))
            if any(value is None for value in metrics.values()) and failure is None:
                failure = "a metric could not be computed"
            entry[method] = {"policy_status": policy, "metrics": metrics, "failure": failure}
        scores["missions"][mission] = entry
    return scores


def _definition(
    benchmark_id: str,
    title: str,
    methods: list[str],
    metrics: list[BenchmarkMetricDefinition],
    reference: str,
    scores: dict[str, Any],
    digest: str,
    limitations: list[str],
) -> BenchmarkDefinition:
    missions = list(scores["missions"])
    splits = [
        BenchmarkSplit(
            split_id=mission,
            seed=0,
            fit_count=1,
            holdout_count=1,
            fit_ids_sha256=_digest(mission),
            holdout_ids_sha256=_digest(mission),
        )
        for mission in missions
    ]
    names = [metric.name for metric in metrics]
    trials = []
    for mission in missions:
        provenance = BenchmarkTrialProvenance(
            command=f"tools/grand_tour_lidar_lidar_evaluate.py {mission}",
            config_sha256=digest,
            input_sha256=digest,
        )
        for method in methods:
            item = scores["missions"][mission][method]
            if item["failure"] is not None:
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=mission,
                        status="failed",
                        failure_reason=item["failure"],
                        provenance=provenance,
                    )
                )
            else:
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=mission,
                        status="success",
                        metrics={name: float(item["metrics"][name]) for name in names},
                        provenance=provenance,
                    )
                )
    return BenchmarkDefinition(
        benchmark_id=benchmark_id,
        title=title,
        protocol=BenchmarkProtocol(
            protocol_id=AUDIT_ID,
            dataset_id="grand_tour_evaluation_missions",
            dataset_source_sha256=digest,
            data_license="MIT",
            split_policy="one split per pre-registered evaluation mission",
            splits=splits,
            initial_estimate_policy="every Calibrex run starts from the dataset's static tf",
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy=(
                "no estimate, or a verdict other than pass, fails the mission; a sparse "
                "trial also fails without a dense estimate of its mission"
            ),
        ),
        metrics=metrics,
        methods=[METHODS[method] for method in methods],
        trials=trials,
        reference_method_id=reference,
        bootstrap_samples=2000,
        limitations=limitations,
        provenance=BenchmarkProvenance(
            generator="tools/build_grand_tour_lidar_lidar_audit.py",
            generator_version=__version__,
            command=f"python tools/build_grand_tour_lidar_lidar_audit.py {digest}",
            data_verified=True,
        ),
    )


def requirements(paths: dict[str, tuple[str, str]]) -> list[SotaClaimRequirement]:
    common = {"dataset_family": "grand_tour", "phase": "audit"}
    dense, sparse = paths["dense"], paths["sparse"]
    return [
        SotaClaimRequirement(
            requirement_id="dense-pass-every-mission",
            description="calibrex_dense passes (all six DoFs, every control) on every mission",
            independent_rig=True,
            rig_id="anymal-d-boxi",
            evidence_path=dense[0],
            evidence_sha256=dense[1],
            method_id="calibrex_dense",
            statistic="failure_rate",
            comparison="less_equal",
            threshold=0.0,
            **common,
        ),
        SotaClaimRequirement(
            requirement_id="dense-cross-mission-consistency",
            description=f"calibrex_dense cross-mission consistency <= {CONSISTENCY_THRESHOLD} "
            "on every mission (0.3 deg / 3 cm)",
            evidence_path=dense[0],
            evidence_sha256=dense[1],
            method_id="calibrex_dense",
            metric=CONSISTENCY,
            statistic="metric_maximum",
            comparison="less_equal",
            threshold=CONSISTENCY_THRESHOLD,
            **common,
        ),
        SotaClaimRequirement(
            requirement_id="dense-same-time-alignment",
            description="calibrex_dense same-time median point-to-plane "
            f"<= {SAME_TIME_THRESHOLD_M} m on every mission",
            evidence_path=dense[0],
            evidence_sha256=dense[1],
            method_id="calibrex_dense",
            metric=SAME_TIME,
            statistic="metric_maximum",
            comparison="less_equal",
            threshold=SAME_TIME_THRESHOLD_M,
            **common,
        ),
        SotaClaimRequirement(
            requirement_id="sparse-pass-every-mission",
            description="calibrex_sparse passes on every mission",
            evidence_path=sparse[0],
            evidence_sha256=sparse[1],
            method_id="calibrex_sparse",
            statistic="failure_rate",
            comparison="less_equal",
            threshold=0.0,
            **common,
        ),
        SotaClaimRequirement(
            requirement_id="sparse-rotation-to-dense",
            description="calibrex_sparse rotation to the mission's dense estimate "
            f"<= {SPARSE_ROTATION_THRESHOLD_DEG} deg on every mission",
            evidence_path=sparse[0],
            evidence_sha256=sparse[1],
            method_id="calibrex_sparse",
            metric=ROTATION,
            statistic="metric_maximum",
            comparison="less_equal",
            threshold=SPARSE_ROTATION_THRESHOLD_DEG,
            **common,
        ),
        SotaClaimRequirement(
            requirement_id="sparse-translation-to-dense",
            description="calibrex_sparse translation to the mission's dense estimate "
            f"<= {SPARSE_TRANSLATION_THRESHOLD_M} m on every mission",
            evidence_path=sparse[0],
            evidence_sha256=sparse[1],
            method_id="calibrex_sparse",
            metric=TRANSLATION,
            statistic="metric_maximum",
            comparison="less_equal",
            threshold=SPARSE_TRANSLATION_THRESHOLD_M,
            **common,
        ),
    ]


def build(directory: Path) -> str:
    prereg = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    missions = [item["mission"] for item in prereg["recordings"]["evaluation"]]
    scores = score(directory, missions)
    scores_path = directory / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    digest = sha256_path(scores_path)
    preregistration = sha256_path(PREREGISTRATION)
    assert digest is not None and preregistration is not None
    same_time = BenchmarkMetricDefinition(
        name=SAME_TIME,
        label="same-time median point-to-plane",
        unit="m",
        direction="lower",
        primary=True,
        interpretation="median over the pre-registered frames of the median "
        "|point-to-plane| of single target scans against the nearest Hesai scan",
    )
    dense = _definition(
        "grand-tour-lidar-lidar-dense",
        "GrandTour Hesai -> MID360: dense estimate",
        ["calibrex_dense", "dataset_tf"],
        [
            same_time,
            BenchmarkMetricDefinition(
                name=CONSISTENCY,
                label="cross-mission consistency",
                unit="ratio",
                direction="lower",
                primary=False,
                interpretation="max(rotation / 0.3 deg, translation / 3 cm) to the chordal "
                "mean of the other missions' estimates",
            ),
        ],
        "dataset_tf",
        scores,
        digest,
        [
            "The dataset tf is a single value, so its consistency is 0 by construction.",
            "Same-time scans are about 10 ms apart; motion between them is not corrected.",
        ],
    )
    sparse = _definition(
        "grand-tour-lidar-lidar-sparse",
        "GrandTour Hesai -> L1-like MID360",
        ["calibrex_sparse", "sparse_rigid"],
        [
            BenchmarkMetricDefinition(
                name=ROTATION,
                label="rotation to the dense estimate",
                unit="deg",
                direction="lower",
                primary=True,
                interpretation="geodesic angle to the same mission's calibrex_dense rotation",
            ),
            BenchmarkMetricDefinition(
                name=TRANSLATION,
                label="translation to the dense estimate",
                unit="m",
                direction="lower",
                primary=False,
                interpretation="Euclidean distance to the same mission's dense translation",
            ),
            same_time,
        ],
        "sparse_rigid",
        scores,
        digest,
        [
            "The sparse target is an emulation: a random 2,000-point subset of each MID360 "
            "scan within 20 m, not a Unitree L1.",
            "The dense estimate is Calibrex's own, not ground truth.",
        ],
    )
    paths = {}
    for name, definition in (("dense", dense), ("sparse", sparse)):
        path = directory / f"grand_tour_lidar_lidar_{name}_benchmark.yaml"
        aggregate_benchmark_definition(definition).save(path)
        path_digest = sha256_path(path)
        assert path_digest is not None
        paths[name] = (path.name, path_digest)
    protocol = SotaAuditProtocol(
        protocol_id=AUDIT_ID,
        scope=SotaClaimScope(
            modalities=["lidar", "lidar"],
            quantities=["rotation", "translation"],
            category="targetless_lidar_lidar",
        ),
        claim_text=str(prereg["claim"]).strip(),
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=requirements(paths),
        provenance=SotaAuditProvenance(
            generator="tools/build_grand_tour_lidar_lidar_audit.py",
            generator_version=__version__,
            command=["python", "tools/build_grand_tour_lidar_lidar_audit.py", str(directory)],
            source_sha256={"preregistration": preregistration, "scores": digest},
        ),
    )
    protocol_path = directory / "grand_tour_lidar_lidar_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id=AUDIT_ID)
    result.save(directory / "grand_tour_lidar_lidar_sota_audit.yaml")
    print(result.verdict)
    for item in result.requirements:
        print(f"  {item.requirement_id}: {item.status} ({item.reason})")
    return str(result.verdict)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
