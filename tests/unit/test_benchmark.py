from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from calibrex.core.benchmark import (
    BenchmarkMethodDefinition,
    BenchmarkMetricDefinition,
    BenchmarkProtocol,
    BenchmarkProvenance,
    BenchmarkSplit,
    BenchmarkTrial,
    BenchmarkTrialProvenance,
    aggregate_benchmark,
    update_benchmark_table_in_markdown,
)
from calibrex.core.exceptions import BenchmarkError

SHA = sha256(b"benchmark-test").hexdigest()


def _trial(
    method_id: str,
    split_id: str,
    value: float,
    *,
    status: str = "success",
) -> BenchmarkTrial:
    metrics = {"error_m": value} if status == "success" else {}
    return BenchmarkTrial(
        method_id=method_id,
        split_id=split_id,
        status=status,
        metrics=metrics,
        runtime_seconds=value + 1.0,
        peak_memory_mb=100.0 + value,
        failure_reason="did not converge" if status == "failed" else None,
        provenance=BenchmarkTrialProvenance(
            command=f"run {method_id} {split_id}",
            config_sha256=SHA,
            input_sha256=SHA,
            output_sha256=SHA if status == "success" else None,
        ),
    )


def _aggregate(trials: list[BenchmarkTrial]):
    return aggregate_benchmark(
        benchmark_id="example",
        title="Example benchmark",
        protocol=BenchmarkProtocol(
            protocol_id="example/v0.1",
            dataset_id="example",
            dataset_source_sha256=SHA,
            data_license="CC-BY-4.0",
            split_policy="seeded shared split",
            splits=[
                BenchmarkSplit(
                    split_id="seed-1",
                    seed=1,
                    fit_count=8,
                    holdout_count=2,
                    fit_ids_sha256=SHA,
                    holdout_ids_sha256=SHA,
                ),
                BenchmarkSplit(
                    split_id="seed-2",
                    seed=2,
                    fit_count=8,
                    holdout_count=2,
                    fit_ids_sha256=SHA,
                    holdout_ids_sha256=SHA,
                ),
            ],
            initial_estimate_policy="same estimate",
            tuning_policy="train-only",
            failure_policy="retain every failure",
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name="error_m",
                label="Error",
                unit="m",
                direction="lower",
                primary=True,
                interpretation="Held-out error; lower is better.",
            )
        ],
        methods=[
            BenchmarkMethodDefinition(
                method_id="reference",
                label="Reference",
                implementation="calibrex_native",
                tool_name="calibrex",
            ),
            BenchmarkMethodDefinition(
                method_id="candidate",
                label="Candidate",
                implementation="adapter",
                tool_name="candidate-tool",
            ),
        ],
        trials=trials,
        reference_method_id="reference",
        limitations=["Synthetic test fixture."],
        provenance=BenchmarkProvenance(
            generator="test",
            generator_version="1",
            command="pytest",
            data_verified=True,
        ),
        bootstrap_samples=1000,
        bootstrap_seed=11,
    )


def test_aggregate_benchmark_computes_failure_rate_ranks_and_paired_ci() -> None:
    benchmark = _aggregate(
        [
            _trial("reference", "seed-1", 2.0),
            _trial("reference", "seed-2", 4.0),
            _trial("candidate", "seed-1", 1.0),
            _trial("candidate", "seed-2", 3.0),
        ]
    )

    candidate = benchmark.method_summaries["candidate"]
    assert candidate.failure_rate == 0.0
    assert candidate.metrics["error_m"].distribution.mean == 2.0
    assert candidate.metrics["error_m"].distribution.p90 == pytest.approx(2.8)
    assert candidate.metrics["error_m"].distribution.p95 == pytest.approx(2.9)
    assert candidate.metrics["error_m"].rank == 1
    comparison = benchmark.paired_comparisons[0]
    assert comparison.paired_count == 2
    assert comparison.raw_delta_mean == -1.0
    assert comparison.improvement_mean == 1.0
    assert comparison.improvement_ci95_low == 1.0
    assert comparison.improvement_ci95_high == 1.0


def test_aggregate_benchmark_keeps_failures_in_denominator() -> None:
    benchmark = _aggregate(
        [
            _trial("reference", "seed-1", 2.0),
            _trial("reference", "seed-2", 4.0),
            _trial("candidate", "seed-1", 1.0),
            _trial("candidate", "seed-2", 0.0, status="failed"),
        ]
    )

    candidate = benchmark.method_summaries["candidate"]
    assert candidate.trial_count == 2
    assert candidate.success_count == 1
    assert candidate.failure_count == 1
    assert candidate.failure_rate == 0.5
    assert benchmark.paired_comparisons[0].paired_count == 1


def test_aggregate_benchmark_rejects_incomplete_shared_matrix() -> None:
    with pytest.raises(ValueError, match="incomplete shared trial matrix"):
        _aggregate(
            [
                _trial("reference", "seed-1", 2.0),
                _trial("reference", "seed-2", 4.0),
                _trial("candidate", "seed-1", 1.0),
            ]
        )


def test_aggregate_benchmark_rejects_success_without_all_metrics() -> None:
    trial = _trial("candidate", "seed-2", 1.0)
    trial.metrics = {}
    with pytest.raises(ValueError, match="missing metrics"):
        _aggregate(
            [
                _trial("reference", "seed-1", 2.0),
                _trial("reference", "seed-2", 4.0),
                _trial("candidate", "seed-1", 1.0),
                trial,
            ]
        )


def test_update_benchmark_table_requires_matching_markers(tmp_path: Path) -> None:
    benchmark = _aggregate(
        [
            _trial("reference", "seed-1", 2.0),
            _trial("reference", "seed-2", 4.0),
            _trial("candidate", "seed-1", 1.0),
            _trial("candidate", "seed-2", 3.0),
        ]
    )
    path = tmp_path / "README.md"
    path.write_text("no generated markers\n", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="must contain exactly one"):
        update_benchmark_table_in_markdown(path, benchmark)
