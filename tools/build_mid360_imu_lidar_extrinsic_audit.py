"""Assemble the pre-registered MID360 IMU-LiDAR extrinsic SOTA audit from scores.

Inputs are the per-method JSON scores of
``tools/score_mid360_imu_lidar_extrinsics.py``, one directory per evaluation
recording, plus an optional ``failures.json`` there that maps a method id to
the reason it has no estimate.  Every method was scored on the same held-out
spans of one common odometry, so each span becomes one paired benchmark split.

Two benchmarks are written per recording because a paired comparison is made
against the benchmark's reference method: the complete extrinsic against
LI-Init, and the lever arm alone against LI-Init's lever arm combined with
Calibrex's rotation and clock offset.

The requirements, thresholds, and coverage are the ones fixed in
``docs/benchmarks/mid360_imu_lidar_translation_preregistration.yaml``; its
SHA-256 is pinned in the protocol's provenance.

Usage (the first round, v1)::

    python tools/build_mid360_imu_lidar_extrinsic_audit.py \\
        docs/assets/mid360_imu_lidar_extrinsic_scores docs/assets

and the second round (v2, one unseen recording)::

    python tools/build_mid360_imu_lidar_extrinsic_audit.py \\
        docs/assets/mid360_imu_lidar_extrinsic_v2_scores docs/assets --round v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

ROUNDS: dict[str, dict[str, Any]] = {
    "v1": {
        "command": [
            "python",
            "tools/build_mid360_imu_lidar_extrinsic_audit.py",
            "docs/assets/mid360_imu_lidar_extrinsic_scores",
            "docs/assets",
        ],
        "preregistration": Path(
            "docs/benchmarks/mid360_imu_lidar_translation_preregistration.yaml"
        ),
        "recordings": ("stadtgarten_seq1", "construction_seq1"),
        "prefix": "mid360_imu_lidar_extrinsic",
        "benchmark_prefix": "mid360_imu_lidar",
        "protocol_id": "mid360-imu-lidar-extrinsic-vs-li-init-v1",
        "claim": (
            "On two hand-held MID360 recordings of the RTK-SLAM dataset not used to develop "
            "the method, Calibrex's full IMU-LiDAR extrinsic predicts held-out "
            "accelerometer-integrated motion at least as well as LI-Init's, and so does its "
            "lever arm alone."
        ),
    },
    "v2": {
        "command": [
            "python",
            "tools/build_mid360_imu_lidar_extrinsic_audit.py",
            "docs/assets/mid360_imu_lidar_extrinsic_v2_scores",
            "docs/assets",
            "--round",
            "v2",
        ],
        "preregistration": Path(
            "docs/benchmarks/mid360_imu_lidar_translation_v2_preregistration.yaml"
        ),
        "recordings": ("construction_seq2",),
        "prefix": "mid360_imu_lidar_extrinsic_v2",
        "benchmark_prefix": "mid360_imu_lidar_v2",
        "protocol_id": "mid360-imu-lidar-extrinsic-vs-li-init-v2",
        "claim": (
            "On a hand-held MID360 recording of the RTK-SLAM dataset not used to develop "
            "the method (construction_seq2), Calibrex's full IMU-LiDAR extrinsic with the "
            "per-segment-gravity lever arm predicts held-out accelerometer-integrated "
            "motion at least as well as LI-Init's, and so does its lever arm alone."
        ),
    },
}
METRIC = "holdout_span_position_rms_m"
DATASET_LICENSE = "see https://huggingface.co/datasets/Willyzw/rtk-slam-dataset"
RIG_ID = "rtk-slam-mid360"
METHODS = {
    "calibrex_native": BenchmarkMethodDefinition(
        method_id="calibrex_native",
        label="Calibrex rotation + accelerometer lever arm",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    "li_init": BenchmarkMethodDefinition(
        method_id="li_init",
        label="LI-Init 66b157a (container)",
        implementation="subprocess",
        tool_name="li_init",
    ),
    "li_init_lever_arm": BenchmarkMethodDefinition(
        method_id="li_init_lever_arm",
        label="LI-Init lever arm with Calibrex rotation and clock offset",
        implementation="subprocess",
        tool_name="li_init",
    ),
    "design_reference": BenchmarkMethodDefinition(
        method_id="design_reference",
        label="MID360 design value (identity, (11.0, 23.29, -44.12) mm, dt = 0)",
        implementation="calibrex_native",
        tool_name="design_reference",
    ),
}
BENCHMARKS = {
    "extrinsic": (("calibrex_native", "li_init", "design_reference"), "li_init"),
    "lever_arm": (("calibrex_native", "li_init_lever_arm"), "li_init_lever_arm"),
}


def _benchmark(
    recording: str,
    kind: str,
    scores: dict[str, dict[str, Any]],
    failures: dict[str, str],
    preregistration_sha256: str,
    command: list[str],
) -> BenchmarkDefinition:
    method_ids, reference = BENCHMARKS[kind]
    present = [method for method in method_ids if method in scores]
    if not present:
        raise SystemExit(f"{recording}: no method has a score")
    keys = {method: set(scores[method]["holdout_span_rms_m"]) for method in present}
    spans = sorted(set.union(*keys.values()))
    if any(len(found) != len(spans) for found in keys.values()):
        raise SystemExit(f"{recording}: methods were scored on different held-out spans")
    splits = [
        BenchmarkSplit(
            split_id=f"span-{key}",
            seed=0,
            fit_count=2 * len(spans),
            holdout_count=1,
            fit_ids_sha256=hashlib.sha256(f"train-for-{key}".encode()).hexdigest(),
            holdout_ids_sha256=hashlib.sha256(key.encode()).hexdigest(),
        )
        for key in spans
    ]
    trials = []
    for method in method_ids:
        provenance = BenchmarkTrialProvenance(
            command=f"score {method} on rtk_slam {recording}",
            config_sha256=preregistration_sha256,
            input_sha256=hashlib.sha256(recording.encode()).hexdigest(),
        )
        score = scores.get(method)
        for key in spans:
            if score is None:
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=f"span-{key}",
                        status="failed",
                        failure_reason=failures.get(method, "no estimate"),
                        provenance=provenance,
                    )
                )
            else:
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=f"span-{key}",
                        status="success",
                        metrics={METRIC: float(score["holdout_span_rms_m"][key])},
                        provenance=provenance,
                    )
                )
    return BenchmarkDefinition(
        benchmark_id=f"mid360-imu-lidar-{kind}-{recording}",
        title=f"MID360 IMU-LiDAR {kind.replace('_', ' ')}, RTK-SLAM {recording}",
        protocol=BenchmarkProtocol(
            protocol_id="mid360-imu-lidar-common-odometry-span-v1",
            dataset_id=f"rtk_slam_{recording}",
            dataset_source_sha256=hashlib.sha256(recording.encode()).hexdigest(),
            data_license=DATASET_LICENSE,
            split_policy=(
                "one common odometry deskewed with the design reference; every third "
                "10-second window is a held-out span; one split per span"
            ),
            splits=splits,
            initial_estimate_policy="each method's own output; no shared initialization",
            tuning_policy="procedure and thresholds pre-registered before either method ran",
            failure_policy="a method without an estimate fails every span of that recording",
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name=METRIC,
                label="held-out span position RMS",
                unit="m",
                direction="lower",
                primary=True,
                interpretation=(
                    "RMS of accelerometer-integrated IMU motion against LiDAR odometry "
                    "on a held-out span; components clipped at 50 mm"
                ),
            )
        ],
        methods=[METHODS[method] for method in method_ids],
        trials=trials,
        reference_method_id=reference,
        bootstrap_samples=2000,
        limitations=[
            "LI-Init fits on the whole recording; Calibrex's own train/holdout split need "
            "not coincide with the scoring spans.",
            "The design reference is a sanity row, not a competitor.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools/build_mid360_imu_lidar_extrinsic_audit.py",
            generator_version=__version__,
            command=" ".join(command),
            data_verified=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--round", choices=sorted(ROUNDS), default="v1")
    args = parser.parse_args()
    config = ROUNDS[args.round]
    score_dir, output_dir = args.score_dir, args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration_sha256 = sha256_path(config["preregistration"])
    assert preregistration_sha256 is not None
    evidence: dict[tuple[str, str], tuple[str, str]] = {}
    for recording in config["recordings"]:
        folder = score_dir / recording
        scores = {
            path.stem.removeprefix("score_"): json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(folder.glob("score_*.json"))
        }
        failures_path = folder / "failures.json"
        failures = (
            json.loads(failures_path.read_text(encoding="utf-8"))
            if failures_path.exists()
            else {}
        )
        for method in failures:
            scores.pop(method, None)
        for kind in BENCHMARKS:
            benchmark = aggregate_benchmark_definition(
                _benchmark(
                    recording, kind, scores, failures, preregistration_sha256, config["command"]
                )
            )
            path = output_dir / f"{config['benchmark_prefix']}_{kind}_benchmark_{recording}.yaml"
            benchmark.save(path)
            digest = sha256_path(path)
            assert digest is not None
            evidence[(recording, kind)] = (path.name, digest)

    requirements = []
    for recording in config["recordings"]:
        extrinsic_path, extrinsic_digest = evidence[(recording, "extrinsic")]
        lever_path, lever_digest = evidence[(recording, "lever_arm")]
        common = {"dataset_family": "rtk_slam", "phase": "audit"}
        requirements += [
            SotaClaimRequirement(
                requirement_id=f"calibrex-estimate-{recording}",
                description=f"Calibrex returns a passing lever-arm estimate ({recording})",
                independent_rig=True,
                rig_id=RIG_ID,
                evidence_path=extrinsic_path,
                evidence_sha256=extrinsic_digest,
                method_id="calibrex_native",
                statistic="failure_rate",
                comparison="less_equal",
                threshold=0.0,
                **common,
            ),
            SotaClaimRequirement(
                requirement_id=f"calibrex-extrinsic-not-worse-than-li-init-{recording}",
                description="paired 95 % CI lower bound of the improvement over LI-Init >= 0",
                evidence_path=extrinsic_path,
                evidence_sha256=extrinsic_digest,
                method_id="calibrex_native",
                reference_method_id="li_init",
                metric=METRIC,
                statistic="paired_improvement_ci95_low",
                comparison="greater_equal",
                threshold=0.0,
                **common,
            ),
            SotaClaimRequirement(
                requirement_id=f"calibrex-lever-arm-not-worse-than-li-init-{recording}",
                description=(
                    "paired 95 % CI lower bound of the improvement over LI-Init's lever arm "
                    "(same rotation and clock offset) >= 0"
                ),
                evidence_path=lever_path,
                evidence_sha256=lever_digest,
                method_id="calibrex_native",
                reference_method_id="li_init_lever_arm",
                metric=METRIC,
                statistic="paired_improvement_ci95_low",
                comparison="greater_equal",
                threshold=0.0,
                **common,
            ),
            SotaClaimRequirement(
                requirement_id=f"calibrex-absolute-fit-{recording}",
                description="mean held-out span position RMS <= 0.010 m",
                evidence_path=extrinsic_path,
                evidence_sha256=extrinsic_digest,
                method_id="calibrex_native",
                metric=METRIC,
                statistic="metric_mean",
                comparison="less_equal",
                threshold=0.010,
                **common,
            ),
        ]
    protocol = SotaAuditProtocol(
        protocol_id=config["protocol_id"],
        scope=SotaClaimScope(
            modalities=["imu", "lidar"],
            quantities=["rotation", "translation", "time_offset"],
            category="targetless_imu_lidar",
        ),
        claim_text=config["claim"],
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=requirements,
        provenance=SotaAuditProvenance(
            generator="tools/build_mid360_imu_lidar_extrinsic_audit.py",
            generator_version=__version__,
            command=config["command"],
            source_sha256={"preregistration": preregistration_sha256},
        ),
    )
    protocol_path = output_dir / f"{config['prefix']}_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id=config["protocol_id"])
    result.save(output_dir / f"{config['prefix']}_sota_audit.yaml")
    print(result.verdict)
    for item in result.requirements:
        print(f"  {item.requirement_id}: {item.status} ({item.reason})")


if __name__ == "__main__":
    main()
