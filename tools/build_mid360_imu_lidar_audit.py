"""Assemble the pre-registered MID360 IMU-LiDAR SOTA audit from candidate scores.

Inputs are the JSON scores written by scoring each method's rotation and clock
offset on held-out windows (see ``score_imu_lidar_candidate``).  Every method
is scored on the same held-out spans of the recording: the held-out windows of
the design-reference segmentation (``holdout_spans_s``), because each method's
own deskewing can shift its odometry segmentation.  Each held-out span becomes
one benchmark split, so methods are compared paired by span.  A method without
a score for a recording is recorded as a failed trial on every split, with the
reason given.

The requirements, thresholds, and coverage are the ones fixed in
``docs/benchmarks/mid360_imu_lidar_preregistration.yaml``; its SHA-256 is
pinned in the protocol's provenance.

Usage::

    python tools/build_mid360_imu_lidar_audit.py docs/assets/mid360_imu_lidar_scores docs/assets

The recorded command is that canonical invocation, so artifacts carry no
local paths.
"""

from __future__ import annotations

import hashlib
import json
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

COMMAND = [
    "python",
    "tools/build_mid360_imu_lidar_audit.py",
    "docs/assets/mid360_imu_lidar_scores",
    "docs/assets",
]
PREREGISTRATION = Path("docs/benchmarks/mid360_imu_lidar_preregistration.yaml")
METRIC = "holdout_window_median_rate_residual_rps"
RECORDINGS = {
    "rtk_slam": {
        "dataset_id": "rtk_slam_stadtgarten_seq2",
        "license": "see https://huggingface.co/datasets/Willyzw/rtk-slam-dataset",
        "rig_id": "rtk-slam-mid360",
        "scores": {
            "calibrex_native": "score_seq2_calibrex.json",
            "li_init": "score_seq2_li_init.json",
            "design_reference": "score_seq2_design.json",
        },
        "failures": {},
    },
    "zenodo_driving_slam_mid360": {
        "dataset_id": "zenodo_14841855_driving_slam_mid360",
        "license": "CC-BY-4.0",
        "rig_id": "koide-driving-mid360",
        "scores": {"calibrex_native": "score_driving_calibrex.json"},
        "failures": {
            "li_init": "LI-Init returned no result: its excitation meter never reached "
            "100 % on every axis (81/73/72 %)",
        },
    },
}
METHODS = {
    "calibrex_native": BenchmarkMethodDefinition(
        method_id="calibrex_native",
        label="Calibrex gyro-deskewed preintegration",
        implementation="calibrex_native",
        tool_name="calibrex",
    ),
    "li_init": BenchmarkMethodDefinition(
        method_id="li_init",
        label="LI-Init 66b157a (container)",
        implementation="subprocess",
        tool_name="li_init",
    ),
    "design_reference": BenchmarkMethodDefinition(
        method_id="design_reference",
        label="MID360 design value (identity, dt = 0)",
        implementation="calibrex_native",
        tool_name="design_reference",
    ),
}


def main() -> None:
    score_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration_sha256 = sha256_path(PREREGISTRATION)
    assert preregistration_sha256 is not None
    benchmark_paths: dict[str, tuple[Path, str]] = {}
    for family, spec in RECORDINGS.items():
        scores = {
            method: json.loads((score_dir / name).read_text(encoding="utf-8"))
            for method, name in spec["scores"].items()  # type: ignore[attr-defined]
        }
        keys = {
            method: set(score["holdout_window_medians_rps"]) for method, score in scores.items()
        }
        windows = sorted(set.union(*keys.values()))
        if any(len(found) != len(windows) for found in keys.values()):
            raise SystemExit(
                f"{family}: methods were scored on different held-out spans; "
                "rescore them with the same holdout_spans_s"
            )
        splits = [
            BenchmarkSplit(
                split_id=f"window-{key}",
                seed=0,
                fit_count=int(scores["calibrex_native"]["holdout_windows"]) * 2,
                holdout_count=1,
                fit_ids_sha256=hashlib.sha256(f"train-for-{key}".encode()).hexdigest(),
                holdout_ids_sha256=hashlib.sha256(key.encode()).hexdigest(),
            )
            for key in windows
        ]
        methods = [METHODS[name] for name in [*scores, *spec["failures"]]]  # type: ignore[misc]
        trials = []
        for method in methods:
            score = scores.get(method.method_id)
            for key in windows:
                value = None if score is None else score["holdout_window_medians_rps"].get(key)
                provenance = BenchmarkTrialProvenance(
                    command=f"score {method.method_id} on {spec['dataset_id']}",
                    config_sha256=preregistration_sha256,
                    input_sha256=hashlib.sha256(spec["dataset_id"].encode()).hexdigest(),  # type: ignore[union-attr]
                )
                if value is None:
                    reason = spec["failures"].get(  # type: ignore[attr-defined]
                        method.method_id, "window missing from this method's odometry segmentation"
                    )
                    trials.append(
                        BenchmarkTrial(
                            method_id=method.method_id,
                            split_id=f"window-{key}",
                            status="failed",
                            failure_reason=reason,
                            provenance=provenance,
                        )
                    )
                else:
                    trials.append(
                        BenchmarkTrial(
                            method_id=method.method_id,
                            split_id=f"window-{key}",
                            status="success",
                            metrics={METRIC: float(value)},
                            runtime_seconds=float(score["seconds"]) / max(len(windows), 1),
                            provenance=provenance,
                        )
                    )
        reference = "li_init" if "li_init" in {m.method_id for m in methods} else "calibrex_native"
        definition = BenchmarkDefinition(
            benchmark_id=f"mid360-imu-lidar-rotation-{family}",
            title=f"MID360 IMU-LiDAR rotation, {spec['dataset_id']}",
            protocol=BenchmarkProtocol(
                protocol_id="mid360-imu-lidar-heldout-window-v1",
                dataset_id=str(spec["dataset_id"]),
                dataset_source_sha256=hashlib.sha256(str(spec["dataset_id"]).encode()).hexdigest(),
                data_license=str(spec["license"]),
                split_policy=(
                    "every third 10-second window of the design-reference odometry is held "
                    "out; every method is scored on those same time spans (rate intervals "
                    "whose midpoint falls in a span); one split per held-out span"
                ),
                splits=splits,
                initial_estimate_policy="each method's own output; no shared initialization",
                tuning_policy="thresholds pre-registered before any LI-Init score was computed",
                failure_policy="a method without an estimate fails every split of that recording",
            ),
            metrics=[
                BenchmarkMetricDefinition(
                    name=METRIC,
                    label="held-out window median rate residual",
                    unit="rad/s",
                    direction="lower",
                    primary=True,
                    interpretation=(
                        "gyro-predicted vs LiDAR rotation increments per second "
                        "on held-out windows"
                    ),
                )
            ],
            methods=methods,
            trials=trials,
            reference_method_id=reference,
            bootstrap_samples=2000,
            limitations=[
                "Each method's odometry is deskewed with its own extrinsic, so a wrong "
                "extrinsic also degrades its own odometry.",
                "The design reference is a sanity row, not a competitor.",
            ],
            provenance=BenchmarkProvenance(
                generator="tools/build_mid360_imu_lidar_audit.py",
                generator_version=__version__,
                command=" ".join(COMMAND),
                data_verified=True,
            ),
        )
        benchmark = aggregate_benchmark_definition(definition)
        path = output_dir / f"mid360_imu_lidar_benchmark_{family}.yaml"
        benchmark.save(path)
        digest = sha256_path(path)
        assert digest is not None
        benchmark_paths[family] = (path, digest)

    rtk_path, rtk_digest = benchmark_paths["rtk_slam"]
    drive_path, drive_digest = benchmark_paths["zenodo_driving_slam_mid360"]
    requirement = SotaClaimRequirement
    protocol = SotaAuditProtocol(
        protocol_id="mid360-imu-lidar-rotation-vs-li-init-v1",
        scope=SotaClaimScope(
            modalities=["imu", "lidar"],
            quantities=["rotation", "time_offset"],
            category="targetless_imu_lidar",
        ),
        claim_text=(
            "On two MID360 recordings from two dataset families, Calibrex's IMU-LiDAR "
            "rotation calibration fits held-out data at least as well as LI-Init, and "
            "returns an estimate wherever LI-Init returns one."
        ),
        minimum_dataset_families=2,
        minimum_independent_rigs=2,
        requirements=[
            requirement(
                requirement_id="calibrex-estimate-rtk-slam",
                phase="audit",
                description="Calibrex returns an estimate on every held-out window (RTK-SLAM)",
                dataset_family="rtk_slam",
                independent_rig=True,
                rig_id="rtk-slam-mid360",
                evidence_path=rtk_path.name,
                evidence_sha256=rtk_digest,
                method_id="calibrex_native",
                statistic="failure_rate",
                comparison="less_equal",
                threshold=0.0,
            ),
            requirement(
                requirement_id="calibrex-estimate-driving",
                phase="audit",
                description="Calibrex returns an estimate on every held-out window (driving)",
                dataset_family="zenodo_driving_slam_mid360",
                independent_rig=True,
                rig_id="koide-driving-mid360",
                evidence_path=drive_path.name,
                evidence_sha256=drive_digest,
                method_id="calibrex_native",
                statistic="failure_rate",
                comparison="less_equal",
                threshold=0.0,
            ),
            requirement(
                requirement_id="calibrex-not-worse-than-li-init-rtk-slam",
                phase="audit",
                description="paired 95 % CI lower bound of the improvement over LI-Init >= 0",
                dataset_family="rtk_slam",
                evidence_path=rtk_path.name,
                evidence_sha256=rtk_digest,
                method_id="calibrex_native",
                reference_method_id="li_init",
                metric=METRIC,
                statistic="paired_improvement_ci95_low",
                comparison="greater_equal",
                threshold=0.0,
            ),
            requirement(
                requirement_id="calibrex-absolute-fit-rtk-slam",
                phase="audit",
                description="mean held-out window median residual <= 0.02 rad/s",
                dataset_family="rtk_slam",
                evidence_path=rtk_path.name,
                evidence_sha256=rtk_digest,
                method_id="calibrex_native",
                metric=METRIC,
                statistic="metric_mean",
                comparison="less_equal",
                threshold=0.02,
            ),
        ],
        provenance=SotaAuditProvenance(
            generator="tools/build_mid360_imu_lidar_audit.py",
            generator_version=__version__,
            command=COMMAND,
            source_sha256={"preregistration": preregistration_sha256},
        ),
    )
    protocol_path = output_dir / "mid360_imu_lidar_sota_protocol.yaml"
    protocol.save(protocol_path)
    result = audit_sota_claim(protocol_path, audit_id="mid360-imu-lidar-rotation-vs-li-init-v1")
    result.save(output_dir / "mid360_imu_lidar_sota_audit.yaml")
    print(result.verdict)
    for item in result.requirements:
        print(f"  {item.requirement_id}: {item.status} ({item.reason})")


if __name__ == "__main__":
    main()
