from __future__ import annotations

from pathlib import Path

import pytest

from calibrex.cli.main import main
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
from calibrex.core.camera_lidar_sota_audit import (
    CameraLidarClaimRequirement,
    CameraLidarSotaAuditProtocol,
    CameraLidarSotaAuditProvenance,
    CameraLidarSotaAuditResult,
    load_camera_lidar_sota_audit_result,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.validation import validate_file
from calibrex.evaluation.camera_lidar_sota_audit import _evaluate_requirement


def test_sota_audit_refutes_one_contradicted_frozen_gate(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.yaml"
    _benchmark().save(benchmark_path)
    digest = sha256_path(benchmark_path)
    assert digest is not None
    protocol = CameraLidarSotaAuditProtocol(
        protocol_id="fixture-training-free-audit",
        declared_category="training_free_targetless",
        claim_text="fixture claim",
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=[
            CameraLidarClaimRequirement(
                requirement_id="hit-gate",
                phase="phase1",
                description="hit rate",
                dataset_family="fixture",
                independent_rig=True,
                rig_id="fixture-rig",
                evidence_path=str(benchmark_path),
                evidence_sha256=digest,
                method_id="candidate",
                metric="hit",
                statistic="metric_mean",
                comparison="greater_equal",
                threshold=1.0,
            ),
            CameraLidarClaimRequirement(
                requirement_id="failure-gate",
                phase="phase5",
                description="zero failures",
                evidence_path=str(benchmark_path),
                evidence_sha256=digest,
                method_id="candidate",
                statistic="failure_rate",
                comparison="less_equal",
                threshold=0.0,
            ),
            CameraLidarClaimRequirement(
                requirement_id="tail-gate",
                phase="phase5",
                description="p95 hit gate",
                dataset_family="optional-fixture",
                independent_rig=True,
                rig_id="optional-rig",
                evidence_path=str(benchmark_path),
                evidence_sha256=digest,
                method_id="candidate",
                metric="hit",
                statistic="metric_p95",
                comparison="less_than",
                threshold=0.51,
                required=False,
            ),
            CameraLidarClaimRequirement(
                requirement_id="paired-ci-gate",
                phase="phase5",
                description="paired CI must support improvement",
                evidence_path=str(benchmark_path),
                evidence_sha256=digest,
                method_id="candidate",
                reference_method_id="reference",
                metric="hit",
                statistic="paired_improvement_ci95_low",
                comparison="greater_equal",
                threshold=0.0,
            ),
            CameraLidarClaimRequirement(
                requirement_id="runtime-p95-gate",
                phase="phase5",
                description="runtime tail",
                evidence_path=str(benchmark_path),
                evidence_sha256=digest,
                method_id="candidate",
                statistic="runtime_p95",
                comparison="less_equal",
                threshold=2.0,
            ),
        ],
        provenance=CameraLidarSotaAuditProvenance(
            generator="pytest",
            generator_version="1",
            source_sha256={"benchmark": digest},
        ),
    )
    protocol_path = tmp_path / "audit-protocol.yaml"
    output = tmp_path / "audit-result.yaml"
    protocol.save(protocol_path)

    exit_code = main(
        [
            "camera-lidar",
            "audit-sota",
            str(protocol_path),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 2
    assert validate_file(output, "camera-lidar-sota-audit-result").valid
    text = output.read_text(encoding="utf-8")
    assert "verdict: refuted" in text
    assert "status: contradicted" in text
    assert "requirement_id: tail-gate" in text
    assert "observed_value: 0.5" in text
    assert "requirement_id: paired-ci-gate" in text
    assert "observed_value: -0.5" in text
    assert "requirement_id: runtime-p95-gate" in text
    assert "observed_value: 1.5" in text
    result = load_camera_lidar_sota_audit_result(output)
    assert result.achieved_dataset_families == []
    assert result.achieved_independent_rig_count == 0
    forged = result.model_dump(mode="python")
    forged["verdict"] = "incomplete"
    with pytest.raises(ValueError, match="incomplete SOTA verdict"):
        CameraLidarSotaAuditResult.model_validate(forged)


def test_sota_audit_rejects_unverified_benchmark_data(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "unverified.yaml"
    _benchmark(data_verified=False).save(benchmark_path)
    digest = sha256_path(benchmark_path)
    assert digest is not None
    requirement = CameraLidarClaimRequirement(
        requirement_id="verified-data",
        phase="phase0",
        description="verified benchmark input",
        evidence_path=str(benchmark_path),
        evidence_sha256=digest,
        method_id="candidate",
        metric="hit",
        statistic="metric_mean",
        comparison="greater_equal",
        threshold=0.0,
    )

    result = _evaluate_requirement(requirement, base_path=tmp_path)

    assert result.status == "contradicted"
    assert "data_verified=false" in result.reason


def test_sota_audit_protocol_rejects_nonfinite_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be finite"):
        CameraLidarClaimRequirement(
            requirement_id="nan",
            phase="phase5",
            description="invalid threshold",
            evidence_path="/tmp/evidence.yaml",
            evidence_sha256="a" * 64,
            method_id="candidate",
            metric="hit",
            statistic="metric_mean",
            comparison="greater_equal",
            threshold=float("nan"),
        )


def _benchmark(*, data_verified: bool = True):
    definition = BenchmarkDefinition(
        benchmark_id="fixture",
        title="fixture",
        protocol=BenchmarkProtocol(
            protocol_id="fixture",
            dataset_id="fixture",
            dataset_source_sha256="a" * 64,
            data_license="fixture",
            split_policy="fixture",
            splits=[
                BenchmarkSplit(
                    split_id="s0",
                    seed=0,
                    fit_count=1,
                    holdout_count=1,
                    fit_ids_sha256="b" * 64,
                    holdout_ids_sha256="c" * 64,
                )
            ],
            initial_estimate_policy="fixture",
            tuning_policy="none",
            failure_policy="retain",
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name="hit",
                label="hit",
                unit="ratio",
                direction="higher",
                primary=True,
                interpretation="fixture",
            )
        ],
        methods=[
            BenchmarkMethodDefinition(
                method_id="reference",
                label="reference",
                implementation="calibrex_native",
                tool_name="fixture",
            ),
            BenchmarkMethodDefinition(
                method_id="candidate",
                label="candidate",
                implementation="calibrex_native",
                tool_name="fixture",
            ),
        ],
        trials=[
            BenchmarkTrial(
                method_id=method_id,
                split_id="s0",
                status="success",
                metrics={"hit": value},
                runtime_seconds=1.0 + value,
                provenance=BenchmarkTrialProvenance(
                    command="pytest",
                    config_sha256="d" * 64,
                    input_sha256="e" * 64,
                ),
            )
            for method_id, value in (("reference", 1.0), ("candidate", 0.5))
        ],
        reference_method_id="reference",
        bootstrap_samples=10,
        limitations=["fixture"],
        provenance=BenchmarkProvenance(
            generator="pytest",
            generator_version="1",
            command="pytest",
            data_verified=data_verified,
        ),
    )
    return aggregate_benchmark_definition(definition)
