"""Assemble the pre-registered KITTI LiDAR-vehicle SOTA audit.

Inputs are the motion cache and ``scores.json`` of
``tools/score_kitti_lidar_vehicle.py`` for the evaluation drives.  Two
benchmarks are written: one split per held-out block (Calibrex against
KITTI's calib_imu_to_velo), and one split per drive (Calibrex's pitch/yaw
agreement with the OXTS-motion vehicle frame).  Requirements, thresholds,
and coverage are those of
``docs/benchmarks/kitti_lidar_vehicle_preregistration.yaml``, whose SHA-256 is
pinned in the protocol.

Usage::

    python tools/build_kitti_lidar_vehicle_audit.py CACHE.pkl \\
        docs/assets/kitti_lidar_vehicle_audit/scores.json docs/assets
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

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
from calibrex.evaluation.vehicle_frame import VehicleFrameRunOptions, evaluate_vehicle_frame
from calibrex.solvers.vehicle_frame_solver import VehicleMotion

PREREGISTRATION = Path("docs/benchmarks/kitti_lidar_vehicle_preregistration.yaml")
COMMAND = [
    "python",
    "tools/build_kitti_lidar_vehicle_audit.py",
    "CACHE.pkl",
    "docs/assets/kitti_lidar_vehicle_audit/scores.json",
    "docs/assets",
]
HOLDOUT_METRIC = "holdout_block_nonholonomic_chi2"
AGREEMENT_METRIC = "pitch_yaw_agreement_deg"
METHODS = {
    "calibrex_native": BenchmarkMethodDefinition(
        method_id="calibrex_native",
        label="Calibrex motion-only LiDAR-to-vehicle rotation",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    "kitti_vendor": BenchmarkMethodDefinition(
        method_id="kitti_vendor",
        label="KITTI calib_imu_to_velo rotation (OXTS frame as vehicle frame)",
        implementation="calibrex_native",
        tool_name="kitti_calibration",
    ),
}


def _estimate_ok(cache: dict[str, dict[str, object]]) -> tuple[bool, str]:
    motions: list[VehicleMotion] = []
    for position, name in enumerate(sorted(cache)):
        for item in cache[name]["lidar"]:  # type: ignore[attr-defined]
            motions.append(
                VehicleMotion(
                    item.block + position * 100_000,
                    item.start_s,
                    item.end_s,
                    item.velocity_mps,
                    item.angular_rate_rps,
                )
            )
    evaluation = evaluate_vehicle_frame(motions, VehicleFrameRunOptions())
    status = {record.name: record.status for record in evaluation.records}
    ok = status.get("pitch") == "estimated" and status.get("yaw") == "estimated"
    return ok, f"pooled evaluation: {status}"


def _vendor_agreement(cache: dict[str, dict[str, object]]) -> dict[str, float]:
    """Per drive: KITTI's rotation against the OXTS-motion vehicle frame (not gated)."""

    import numpy as np
    from scipy.spatial.transform import Rotation

    from calibrex.solvers.vehicle_frame_solver import solve_vehicle_frame

    values = {}
    for name in sorted(cache):
        vendor = np.asarray(cache[name]["vendor"])[:3, :3]
        oxts = solve_vehicle_frame(cache[name]["oxts"])  # type: ignore[arg-type]
        if oxts.rotation is None:
            values[name] = float("nan")
            continue
        difference = np.degrees(
            Rotation.from_matrix(vendor @ (oxts.rotation @ vendor).T).as_rotvec()
        )
        values[name] = float(np.hypot(difference[1], difference[2]))
    return values


def _definition(
    benchmark_id: str,
    title: str,
    metric: BenchmarkMetricDefinition,
    methods: list[str],
    values: dict[str, dict[str, float]],
    failures: dict[str, str],
    reference: str,
    digest: str,
    split_policy: str,
) -> BenchmarkDefinition:
    # A block in which no method has a moving interval to score (the vehicle stood
    # still) cannot be compared; it is dropped for every method alike.
    all_keys = sorted(next(iter(values.values())))
    keys = [
        key
        for key in all_keys
        if any(math.isfinite(float(values[method][key])) for method in values)
    ]
    dropped = [key for key in all_keys if key not in keys]
    splits = [
        BenchmarkSplit(
            split_id=key,
            seed=0,
            fit_count=1,
            holdout_count=1,
            fit_ids_sha256=hashlib.sha256(f"fit-{key}".encode()).hexdigest(),
            holdout_ids_sha256=hashlib.sha256(key.encode()).hexdigest(),
        )
        for key in keys
    ]
    trials = []
    for method in methods:
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
                        metrics={metric.name: float(values[method][key])},
                        provenance=provenance,
                    )
                )
    return BenchmarkDefinition(
        benchmark_id=benchmark_id,
        title=title,
        protocol=BenchmarkProtocol(
            protocol_id="kitti-lidar-vehicle-v1",
            dataset_id="kitti_raw_2011_09_26_evaluation_drives",
            dataset_source_sha256=digest,
            data_license="CC BY-NC-SA 3.0",
            split_policy=split_policy,
            splits=splits,
            initial_estimate_policy="each method's own rotation; levers fitted on train blocks",
            tuning_policy="procedure and thresholds pre-registered before the evaluation runs",
            failure_policy="a method without an estimate fails every split",
        ),
        metrics=[metric],
        methods=[METHODS[method] for method in methods],
        trials=trials,
        reference_method_id=reference,
        bootstrap_samples=2000,
        limitations=[
            "The vehicle frame is defined by the non-holonomic motion model.",
            *(
                [
                    "Dropped for every method, with no moving interval to score: "
                    + ", ".join(dropped)
                ]
                if dropped
                else []
            ),
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_kitti_lidar_vehicle_audit.py",
            generator_version=__version__,
            command=" ".join(COMMAND),
            data_verified=True,
        ),
    )


def main() -> None:
    cache_path, scores_path, output_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    with cache_path.open("rb") as stream:
        cache = pickle.load(stream)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    preregistration = sha256_path(PREREGISTRATION)
    scores_digest = sha256_path(scores_path)
    assert preregistration is not None and scores_digest is not None
    ok, note = _estimate_ok(cache)
    print("estimate requirement:", ok, note)
    failures = {} if ok else {"calibrex_native": "pitch or yaw not estimated: " + note}
    holdout = aggregate_benchmark_definition(
        _definition(
            "kitti-lidar-vehicle-holdout",
            "KITTI LiDAR-vehicle rotation, held-out blocks",
            BenchmarkMetricDefinition(
                name=HOLDOUT_METRIC,
                label="held-out non-holonomic chi-square per residual",
                unit="1",
                direction="lower",
                primary=True,
                interpretation="mean of min(r^2, 25) over normalized residuals of a held-out block",
            ),
            ["calibrex_native", "kitti_vendor"],
            scores["holdout_block_scores"],
            failures,
            "kitti_vendor",
            scores_digest,
            "10-second blocks per drive; every third sorted key held out; one split per block",
        )
    )
    agreement = aggregate_benchmark_definition(
        _definition(
            "kitti-lidar-vehicle-agreement",
            "KITTI LiDAR-vehicle pitch/yaw agreement with the OXTS-motion vehicle frame",
            BenchmarkMetricDefinition(
                name=AGREEMENT_METRIC,
                label="pitch/yaw agreement with the OXTS-motion vehicle frame",
                unit="deg",
                direction="lower",
                primary=True,
                interpretation="per drive, norm of the pitch and yaw differences",
            ),
            ["calibrex_native", "kitti_vendor"],
            {
                "calibrex_native": scores["pitch_yaw_agreement_deg"],
                "kitti_vendor": _vendor_agreement(cache),
            },
            failures,
            "calibrex_native",
            scores_digest,
            "one split per evaluation drive",
        )
    )
    paths = {}
    for name, benchmark in (("holdout", holdout), ("agreement", agreement)):
        path = output_dir / f"kitti_lidar_vehicle_{name}_benchmark.yaml"
        benchmark.save(path)
        digest = sha256_path(path)
        assert digest is not None
        paths[name] = (path.name, digest)
    common = {"dataset_family": "kitti_raw", "phase": "audit"}
    protocol = SotaAuditProtocol(
        protocol_id="kitti-lidar-vehicle-v1",
        scope=SotaClaimScope(
            modalities=["lidar", "vehicle"],
            quantities=["rotation"],
            category="targetless_lidar_vehicle",
        ),
        claim_text=(
            "On eight KITTI raw drives not used to develop the method, Calibrex's motion-only "
            "LiDAR-to-vehicle rotation fits held-out LiDAR motion at least as well as KITTI's "
            "calib_imu_to_velo, and its pitch and yaw agree with an independent vehicle frame "
            "from the OXTS body-frame velocities."
        ),
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=[
            SotaClaimRequirement(
                requirement_id="calibrex-estimate",
                description="Calibrex estimates pitch and yaw on the pooled evaluation drives",
                independent_rig=True,
                rig_id="kitti-2011-09-26",
                evidence_path=paths["holdout"][0],
                evidence_sha256=paths["holdout"][1],
                method_id="calibrex_native",
                statistic="failure_rate",
                comparison="less_equal",
                threshold=0.0,
                **common,
            ),
            SotaClaimRequirement(
                requirement_id="calibrex-not-worse-than-vendor",
                description=(
                    "paired 95 % CI lower bound of the improvement over calib_imu_to_velo >= 0"
                ),
                evidence_path=paths["holdout"][0],
                evidence_sha256=paths["holdout"][1],
                method_id="calibrex_native",
                reference_method_id="kitti_vendor",
                metric=HOLDOUT_METRIC,
                statistic="paired_improvement_ci95_low",
                comparison="greater_equal",
                threshold=0.0,
                **common,
            ),
            SotaClaimRequirement(
                requirement_id="calibrex-agrees-with-oxts-motion-frame",
                description="mean per-drive pitch/yaw agreement <= 0.5 deg",
                evidence_path=paths["agreement"][0],
                evidence_sha256=paths["agreement"][1],
                method_id="calibrex_native",
                metric=AGREEMENT_METRIC,
                statistic="metric_mean",
                comparison="less_equal",
                threshold=0.5,
                **common,
            ),
        ],
        provenance=SotaAuditProvenance(
            generator="tools/build_kitti_lidar_vehicle_audit.py",
            generator_version=__version__,
            command=COMMAND,
            source_sha256={"preregistration": preregistration, "scores": scores_digest},
        ),
    )
    protocol_path = output_dir / "kitti_lidar_vehicle_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id="kitti-lidar-vehicle-v1")
    result.save(output_dir / "kitti_lidar_vehicle_sota_audit.yaml")
    print(result.verdict)
    for item in result.requirements:
        print(f"  {item.requirement_id}: {item.status} ({item.reason})")


if __name__ == "__main__":
    main()
