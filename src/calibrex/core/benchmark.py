"""Typed, reproducible N-way benchmark artifacts and aggregation.

Unlike report comparison, a benchmark represents repeated executions on a
shared split matrix. It keeps failures in the population, computes descriptive
statistics per method, and uses paired bootstrap deltas only where both methods
produced a metric on the same split.
"""

from __future__ import annotations

import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.exceptions import BenchmarkError
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

BENCHMARK_SCHEMA_VERSION: Literal["slac.benchmark/v0.1"] = "slac.benchmark/v0.1"
BENCHMARK_DEFINITION_SCHEMA_VERSION: Literal["slac.benchmark_definition/v0.1"] = (
    "slac.benchmark_definition/v0.1"
)

MetricDirection = Literal["lower", "higher"]
BenchmarkTrialStatus = Literal["success", "failed"]
BenchmarkImplementation = Literal["calibrex_native", "adapter", "subprocess", "imported"]


class BenchmarkMetricDefinition(StrictModel):
    """One metric that every successful benchmark trial must report."""

    name: str
    label: str
    unit: str
    display_unit: str | None = None
    display_scale: float = Field(default=1.0, gt=0)
    direction: MetricDirection
    primary: bool = False
    interpretation: str


class BenchmarkMethodDefinition(StrictModel):
    """Method identity and implementation provenance."""

    method_id: str
    label: str
    implementation: BenchmarkImplementation
    tool_name: str
    tool_version: str | None = None
    source_commit: str | None = None
    paper_doi: str | None = None
    license_spdx: str | None = None
    adapter_version: str | None = None


class BenchmarkSplit(StrictModel):
    """One prespecified fit/holdout split shared by every method."""

    split_id: str
    seed: int
    fit_count: int = Field(ge=1)
    holdout_count: int = Field(ge=1)
    fit_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    holdout_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkProtocol(StrictModel):
    """Dataset and split contract for an N-way benchmark."""

    protocol_id: str
    dataset_id: str
    dataset_version: str | None = None
    dataset_doi: str | None = None
    dataset_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_license: str
    split_policy: str
    splits: list[BenchmarkSplit] = Field(min_length=1)
    initial_estimate_policy: str
    tuning_policy: str
    failure_policy: str


class BenchmarkTrialProvenance(StrictModel):
    """Execution provenance for one method on one split."""

    command: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_host: str | None = None
    notes: list[str] = Field(default_factory=list)


class BenchmarkTrial(StrictModel):
    """One method execution on one shared split."""

    method_id: str
    split_id: str
    status: BenchmarkTrialStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    runtime_seconds: float | None = Field(default=None, ge=0)
    peak_memory_mb: float | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    provenance: BenchmarkTrialProvenance


class DistributionSummary(StrictModel):
    """Descriptive distribution and deterministic bootstrap mean interval."""

    count: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean_ci95_low: float | None = None
    mean_ci95_high: float | None = None


class BenchmarkMetricSummary(StrictModel):
    """Aggregate metric values for one method."""

    metric: str
    distribution: DistributionSummary
    rank: int | None = Field(default=None, ge=1)


class BenchmarkMethodSummary(StrictModel):
    """Failure, runtime, memory, and metric aggregates for one method."""

    method_id: str
    trial_count: int = Field(ge=1)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    runtime_seconds: DistributionSummary
    peak_memory_mb: DistributionSummary
    metrics: dict[str, BenchmarkMetricSummary]


class PairedBootstrapComparison(StrictModel):
    """Paired candidate-vs-reference delta on common successful splits."""

    reference_method_id: str
    candidate_method_id: str
    metric: str
    direction: MetricDirection
    paired_count: int = Field(ge=0)
    raw_delta_mean: float | None = None
    improvement_mean: float | None = None
    improvement_ci95_low: float | None = None
    improvement_ci95_high: float | None = None


class BenchmarkProvenance(StrictModel):
    """Provenance for the generated benchmark artifact."""

    generator: str
    generator_version: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    command: str
    source_artifacts: dict[str, str] = Field(default_factory=dict)
    data_verified: bool


class BenchmarkDefinition(StrictModel):
    """Raw shared trial matrix consumed by the benchmark aggregator."""

    schema_version: Literal["slac.benchmark_definition/v0.1"] = BENCHMARK_DEFINITION_SCHEMA_VERSION
    benchmark_id: str
    title: str
    protocol: BenchmarkProtocol
    metrics: list[BenchmarkMetricDefinition] = Field(min_length=1)
    methods: list[BenchmarkMethodDefinition] = Field(min_length=2)
    trials: list[BenchmarkTrial] = Field(min_length=2)
    reference_method_id: str
    bootstrap_samples: int = Field(default=2000, ge=1)
    bootstrap_seed: int = 0
    limitations: list[str] = Field(min_length=1)
    provenance: BenchmarkProvenance

    def save(self, path: str | Path) -> None:
        """Write the raw benchmark definition as JSON or YAML."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


class BenchmarkArtifact(StrictModel):
    """Schema-valid N-way benchmark including raw trials and aggregates."""

    schema_version: Literal["slac.benchmark/v0.1"] = BENCHMARK_SCHEMA_VERSION
    benchmark_id: str
    title: str
    protocol: BenchmarkProtocol
    metrics: list[BenchmarkMetricDefinition] = Field(min_length=1)
    methods: list[BenchmarkMethodDefinition] = Field(min_length=2)
    trials: list[BenchmarkTrial] = Field(min_length=2)
    method_summaries: dict[str, BenchmarkMethodSummary]
    paired_comparisons: list[PairedBootstrapComparison] = Field(default_factory=list)
    reference_method_id: str
    bootstrap_samples: int = Field(ge=1)
    bootstrap_seed: int
    limitations: list[str] = Field(min_length=1)
    provenance: BenchmarkProvenance

    def save(self, path: str | Path) -> None:
        """Write the benchmark as JSON or YAML."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def aggregate_benchmark(
    *,
    benchmark_id: str,
    title: str,
    protocol: BenchmarkProtocol,
    metrics: list[BenchmarkMetricDefinition],
    methods: list[BenchmarkMethodDefinition],
    trials: list[BenchmarkTrial],
    reference_method_id: str,
    limitations: list[str],
    provenance: BenchmarkProvenance,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> BenchmarkArtifact:
    """Validate a complete shared trial matrix and aggregate it.

    The bootstrap is paired by ``split_id``. Positive ``improvement_mean``
    always favors the candidate, regardless of whether the raw metric is
    minimized or maximized.
    """

    if bootstrap_samples < 1:
        msg = "bootstrap_samples must be >= 1"
        raise ValueError(msg)
    method_by_id = _unique_by_id(methods, "method_id")
    metric_by_name = _unique_by_id(metrics, "name")
    split_by_id = _unique_by_id(protocol.splits, "split_id")
    if reference_method_id not in method_by_id:
        msg = f"unknown reference_method_id: {reference_method_id}"
        raise ValueError(msg)
    _validate_trials(
        trials,
        method_ids=set(method_by_id),
        metric_names=set(metric_by_name),
        split_ids=set(split_by_id),
    )

    summaries = {
        method_id: _method_summary(
            method_id,
            trials,
            metrics,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for method_id in method_by_id
    }
    _assign_metric_ranks(summaries, metrics)
    paired = [
        _paired_comparison(
            reference_method_id,
            candidate_method_id,
            metric,
            trials,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for candidate_method_id in method_by_id
        if candidate_method_id != reference_method_id
        for metric in metrics
    ]
    return BenchmarkArtifact(
        benchmark_id=benchmark_id,
        title=title,
        protocol=protocol,
        metrics=metrics,
        methods=methods,
        trials=trials,
        method_summaries=summaries,
        paired_comparisons=paired,
        reference_method_id=reference_method_id,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        limitations=limitations,
        provenance=provenance,
    )


def aggregate_benchmark_definition(definition: BenchmarkDefinition) -> BenchmarkArtifact:
    """Aggregate one validated raw benchmark definition."""

    return aggregate_benchmark(
        benchmark_id=definition.benchmark_id,
        title=definition.title,
        protocol=definition.protocol,
        metrics=definition.metrics,
        methods=definition.methods,
        trials=definition.trials,
        reference_method_id=definition.reference_method_id,
        limitations=definition.limitations,
        provenance=definition.provenance,
        bootstrap_samples=definition.bootstrap_samples,
        bootstrap_seed=definition.bootstrap_seed,
    )


def load_benchmark_definition(path: str | Path) -> BenchmarkDefinition:
    """Load and validate a raw benchmark definition."""

    definition_path = Path(path)
    try:
        return BenchmarkDefinition.model_validate(read_mapping(definition_path))
    except Exception as exc:
        raise BenchmarkError(f"invalid benchmark definition {definition_path}: {exc}") from exc


def load_benchmark(path: str | Path) -> BenchmarkArtifact:
    """Load and validate a benchmark artifact."""

    benchmark_path = Path(path)
    try:
        return BenchmarkArtifact.model_validate(read_mapping(benchmark_path))
    except Exception as exc:
        raise BenchmarkError(f"invalid benchmark {benchmark_path}: {exc}") from exc


def benchmark_json_schema() -> dict[str, Any]:
    """Return the benchmark JSON Schema."""

    return BenchmarkArtifact.model_json_schema()


def benchmark_definition_json_schema() -> dict[str, Any]:
    """Return the raw benchmark-definition JSON Schema."""

    return BenchmarkDefinition.model_json_schema()


def render_benchmark_markdown(benchmark: BenchmarkArtifact) -> str:
    """Render a deterministic Markdown summary from a benchmark artifact."""

    lines = [
        f"<!-- Generated from {benchmark.schema_version}; do not edit by hand. -->",
        "",
        f"## {benchmark.title}",
        "",
        render_benchmark_table_markdown(benchmark),
    ]
    lines.extend(
        [
            "",
            (
                f"Shared protocol: `{benchmark.protocol.protocol_id}`; "
                f"{len(benchmark.protocol.splits)} split(s); "
                f"{benchmark.bootstrap_samples} paired bootstrap samples."
            ),
            "",
            "Limitations:",
            "",
            *[f"- {limitation}" for limitation in benchmark.limitations],
            "",
        ]
    )
    return "\n".join(lines)


def render_benchmark_table_markdown(benchmark: BenchmarkArtifact) -> str:
    """Render only the generated Markdown table for embedding in docs."""

    methods = {method.method_id: method for method in benchmark.methods}
    headers = [
        "Method",
        *[
            (
                f"{metric.label} {metric.display_unit or metric.unit} "
                f"{_direction_arrow(metric.direction)}"
            )
            for metric in benchmark.metrics
        ],
        "Failure rate",
        "Runtime s",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---", *(["---:"] * (len(headers) - 1))]) + "|",
    ]
    for method in benchmark.methods:
        summary = benchmark.method_summaries[method.method_id]
        metric_cells = [
            _metric_markdown_cell(
                summary.metrics[metric.name],
                scale=metric.display_scale,
            )
            for metric in benchmark.metrics
        ]
        runtime = _format_number(summary.runtime_seconds.mean)
        lines.append(
            "| "
            + " | ".join(
                [
                    methods[method.method_id].label,
                    *metric_cells,
                    f"{summary.failure_rate:.1%}",
                    runtime,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def update_benchmark_table_in_markdown(
    path: str | Path,
    benchmark: BenchmarkArtifact,
) -> None:
    """Replace one marker-delimited benchmark table in a Markdown file."""

    markdown_path = Path(path)
    start = f"<!-- calibrex-benchmark:{benchmark.benchmark_id}:start -->"
    end = f"<!-- calibrex-benchmark:{benchmark.benchmark_id}:end -->"
    text = markdown_path.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1:
        msg = f"{markdown_path} must contain exactly one {start!r} and {end!r}"
        raise BenchmarkError(msg)
    prefix, remainder = text.split(start, maxsplit=1)
    _, suffix = remainder.split(end, maxsplit=1)
    replacement = f"{start}\n{render_benchmark_table_markdown(benchmark)}\n{end}"
    markdown_path.write_text(prefix + replacement + suffix, encoding="utf-8")


def _direction_arrow(direction: MetricDirection) -> str:
    return "↓" if direction == "lower" else "↑"


def _metric_markdown_cell(
    summary: BenchmarkMetricSummary,
    *,
    scale: float,
) -> str:
    distribution = summary.distribution
    if distribution.mean is None:
        return "—"
    value = _format_number(distribution.mean * scale)
    if (
        distribution.count > 1
        and distribution.mean_ci95_low is not None
        and distribution.mean_ci95_high is not None
    ):
        value += (
            f" [{_format_number(distribution.mean_ci95_low * scale)}, "
            f"{_format_number(distribution.mean_ci95_high * scale)}]"
        )
    return f"**{value}**" if summary.rank == 1 else value


def _format_number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6g}"


def _unique_by_id(items: list[Any], field: str) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    for item in items:
        item_id = getattr(item, field)
        if item_id in by_id:
            msg = f"duplicate {field}: {item_id}"
            raise ValueError(msg)
        by_id[item_id] = item
    return by_id


def _validate_trials(
    trials: list[BenchmarkTrial],
    *,
    method_ids: set[str],
    metric_names: set[str],
    split_ids: set[str],
) -> None:
    expected = {(method_id, split_id) for method_id in method_ids for split_id in split_ids}
    observed: set[tuple[str, str]] = set()
    for trial in trials:
        key = (trial.method_id, trial.split_id)
        if key in observed:
            msg = f"duplicate benchmark trial: method={key[0]} split={key[1]}"
            raise ValueError(msg)
        observed.add(key)
        if trial.method_id not in method_ids:
            msg = f"trial references unknown method: {trial.method_id}"
            raise ValueError(msg)
        if trial.split_id not in split_ids:
            msg = f"trial references unknown split: {trial.split_id}"
            raise ValueError(msg)
        unknown_metrics = set(trial.metrics) - metric_names
        if unknown_metrics:
            msg = f"trial contains undeclared metrics: {sorted(unknown_metrics)}"
            raise ValueError(msg)
        if trial.status == "success" and set(trial.metrics) != metric_names:
            missing_metrics = sorted(metric_names - set(trial.metrics))
            msg = f"successful trial is missing metrics: {missing_metrics}"
            raise ValueError(msg)
        if trial.status == "failed" and not trial.failure_reason:
            msg = "failed trial requires failure_reason"
            raise ValueError(msg)
    if observed != expected:
        missing_trials = sorted(expected - observed)
        extra_trials = sorted(observed - expected)
        msg = (
            "incomplete shared trial matrix; "
            f"missing={missing_trials}, extra={extra_trials}"
        )
        raise ValueError(msg)


def _method_summary(
    method_id: str,
    trials: list[BenchmarkTrial],
    metrics: list[BenchmarkMetricDefinition],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> BenchmarkMethodSummary:
    selected = [trial for trial in trials if trial.method_id == method_id]
    successes = [trial for trial in selected if trial.status == "success"]
    failures = len(selected) - len(successes)
    return BenchmarkMethodSummary(
        method_id=method_id,
        trial_count=len(selected),
        success_count=len(successes),
        failure_count=failures,
        failure_rate=failures / len(selected),
        runtime_seconds=_distribution(
            [trial.runtime_seconds for trial in selected if trial.runtime_seconds is not None],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        peak_memory_mb=_distribution(
            [trial.peak_memory_mb for trial in selected if trial.peak_memory_mb is not None],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        metrics={
            metric.name: BenchmarkMetricSummary(
                metric=metric.name,
                distribution=_distribution(
                    [trial.metrics[metric.name] for trial in successes],
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
            )
            for metric in metrics
        },
    )


def _distribution(
    values: list[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> DistributionSummary:
    if not values:
        return DistributionSummary(count=0)
    ci_low, ci_high = _bootstrap_mean_interval(
        values,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return DistributionSummary(
        count=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        standard_deviation=statistics.pstdev(values),
        minimum=min(values),
        maximum=max(values),
        mean_ci95_low=ci_low,
        mean_ci95_high=ci_high,
    )


def _bootstrap_mean_interval(
    values: list[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(bootstrap_seed)
    means = sorted(
        statistics.fmean(generator.choices(values, k=len(values))) for _ in range(bootstrap_samples)
    )
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _assign_metric_ranks(
    summaries: dict[str, BenchmarkMethodSummary],
    metrics: list[BenchmarkMetricDefinition],
) -> None:
    for metric in metrics:
        available: list[tuple[str, float]] = []
        for method_id, summary in summaries.items():
            mean = summary.metrics[metric.name].distribution.mean
            if mean is not None:
                available.append((method_id, mean))
        ordered = sorted(
            available,
            key=lambda item: item[1] if metric.direction == "lower" else -item[1],
        )
        rank = 0
        previous_value: float | None = None
        for position, (method_id, value) in enumerate(ordered, start=1):
            if previous_value is None or value != previous_value:
                rank = position
            summaries[method_id].metrics[metric.name].rank = rank
            previous_value = value


def _paired_comparison(
    reference_method_id: str,
    candidate_method_id: str,
    metric: BenchmarkMetricDefinition,
    trials: list[BenchmarkTrial],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> PairedBootstrapComparison:
    by_key = {
        (trial.method_id, trial.split_id): trial for trial in trials if trial.status == "success"
    }
    split_ids = sorted(
        split_id
        for method_id, split_id in by_key
        if method_id == reference_method_id and (candidate_method_id, split_id) in by_key
    )
    raw_deltas = [
        by_key[(candidate_method_id, split_id)].metrics[metric.name]
        - by_key[(reference_method_id, split_id)].metrics[metric.name]
        for split_id in split_ids
    ]
    sign = -1.0 if metric.direction == "lower" else 1.0
    improvements = [sign * delta for delta in raw_deltas]
    if not improvements:
        return PairedBootstrapComparison(
            reference_method_id=reference_method_id,
            candidate_method_id=candidate_method_id,
            metric=metric.name,
            direction=metric.direction,
            paired_count=0,
        )
    ci_low, ci_high = _bootstrap_mean_interval(
        improvements,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return PairedBootstrapComparison(
        reference_method_id=reference_method_id,
        candidate_method_id=candidate_method_id,
        metric=metric.name,
        direction=metric.direction,
        paired_count=len(improvements),
        raw_delta_mean=statistics.fmean(raw_deltas),
        improvement_mean=statistics.fmean(improvements),
        improvement_ci95_low=ci_low,
        improvement_ci95_high=ci_high,
    )
