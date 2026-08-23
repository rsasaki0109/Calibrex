"""Human-readable reports for repeated solid-state LiDAR benchmark evidence."""

from __future__ import annotations

from html import escape
from pathlib import Path

from calibrex.core.io import write_text
from calibrex.core.solid_state_cross_dataset_benchmark import (
    SolidStateCrossDatasetBenchmarkDataset,
    SolidStateCrossDatasetBenchmarkManifest,
)


def _number(value: float | None, *, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(dataset: SolidStateCrossDatasetBenchmarkDataset) -> str:
    interval = dataset.holdout_improvement_ci
    if interval is None:
        return "n/a"
    return f"[{interval.lower:.2f}, {interval.upper:.2f}]%"


def render_solid_state_cross_dataset_benchmark_markdown(
    manifest: SolidStateCrossDatasetBenchmarkManifest,
) -> str:
    """Render a reproducibility-oriented Markdown report."""

    aggregate = manifest.aggregate
    median_improvement = _number(aggregate.median_holdout_improvement_percent, digits=2)
    lines = [
        "# Solid-state LiDAR benchmark",
        "",
        f"- Schema: `{manifest.schema_version}`",
        f"- Tool: `{manifest.tool}` ({manifest.tool_version})",
        f"- Protocol: `{manifest.protocol.name}`",
        f"- Spec SHA-256: `{manifest.spec_sha256}`",
        f"- Provenance sources: `{len(manifest.provenance.source_paths)}`",
        "",
        "## Summary",
        "",
        f"{aggregate.conclusion}",
        "",
        f"- Scored datasets: **{aggregate.scored_dataset_count}/{aggregate.dataset_count}**",
        f"- Scored replicates: **{aggregate.scored_replicate_count}/"
        f"{aggregate.total_replicate_count}**",
        f"- Adaptive win rate: **{_number(aggregate.adaptive_win_rate, digits=3)}**",
        f"- Mean improvement: **{_number(aggregate.mean_holdout_improvement_percent, digits=2)}%**",
        f"- Median improvement: **{median_improvement}%**",
        f"- 95% bootstrap CI: **{_ci_from_interval(aggregate.holdout_improvement_ci)}**",
        "",
        "| Dataset | Status | Replicates | Adaptive win rate | Mean improvement | "
        "95% CI | Winner | Reference mode |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for dataset in manifest.datasets:
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset.id,
                    dataset.status,
                    f"{dataset.scored_replicate_count}/{dataset.replicate_count}",
                    _number(dataset.adaptive_win_rate, digits=3),
                    f"{_number(dataset.holdout_improvement_mean_percent, digits=2)}%",
                    _ci(dataset),
                    dataset.winner,
                    dataset.reference_mode,
                ]
            )
            + " |"
        )

    lines.extend(["", "## Replicate details", ""])
    lines.extend(
        [
            "| Dataset | Split | Seed | Status | Adaptive RMSE (m) | Uniform RMSE (m) | "
            "Improvement | Winner | Failures |",
            "|---|---|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for dataset in manifest.datasets:
        for replicate in dataset.replicates:
            adaptive = next(
                (item for item in replicate.variants if item.id == "adaptive_mad"),
                None,
            )
            uniform = next(
                (item for item in replicate.variants if item.id == "uniform_none"),
                None,
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset.id,
                        replicate.split_id,
                        str(replicate.seed),
                        replicate.status,
                        _number(adaptive.final_holdout_rmse_m if adaptive else None),
                        _number(uniform.final_holdout_rmse_m if uniform else None),
                        f"{_number(replicate.holdout_improvement_percent, digits=2)}%",
                        replicate.winner,
                        ", ".join(replicate.failure_categories) or "none",
                    ]
                )
                + " |"
            )

    failure_categories = sorted(
        {category for dataset in manifest.datasets for category in dataset.failure_categories}
    )
    lines.extend(
        [
            "",
            "## Failure classification",
            "",
            "The benchmark retains non-converged evidence but exposes it explicitly.",
            "",
            f"- Observed categories: `{', '.join(failure_categories) or 'none'}`",
            "- `trajectory_only` and `identity_control` results do not provide "
            "absolute extrinsic ground truth.",
            "- A positive temporal-holdout improvement is evidence for this protocol, "
            "not a universal SOTA claim.",
            "",
            "## Reproduction",
            "",
            "Use the materialized spec recorded in `spec_path` above; the output "
            "filename is a local choice.",
            "",
            "```powershell",
            "python tools/run_solid_state_cross_dataset_benchmark.py "
            "<materialized-spec.yaml> `",
            "  --output outputs/solid_state_cross_dataset_benchmark.yaml `",
            "  --markdown-output outputs/solid_state_cross_dataset_benchmark.md `",
            "  --html-output outputs/solid_state_cross_dataset_benchmark.html",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _ci_from_interval(interval: object) -> str:
    if interval is None:
        return "n/a"
    lower = getattr(interval, "lower", None)
    upper = getattr(interval, "upper", None)
    if lower is None or upper is None:
        return "n/a"
    return f"[{lower:.2f}, {upper:.2f}]%"


def render_solid_state_cross_dataset_benchmark_html(
    manifest: SolidStateCrossDatasetBenchmarkManifest,
) -> str:
    """Render a self-contained HTML report without external assets."""

    aggregate = manifest.aggregate
    rows: list[str] = []
    for dataset in manifest.datasets:
        dataset_improvement = _number(dataset.holdout_improvement_mean_percent, digits=2)
        rows.append(
            "<tr>"
            + "".join(
                [
                    f"<td>{escape(dataset.id)}</td>",
                    f"<td>{escape(dataset.status)}</td>",
                    f"<td>{dataset.scored_replicate_count}/{dataset.replicate_count}</td>",
                    f"<td>{escape(_number(dataset.adaptive_win_rate, digits=3))}</td>",
                    f"<td>{escape(dataset_improvement)}%</td>",
                    f"<td>{escape(_ci(dataset))}</td>",
                    f"<td>{escape(dataset.winner)}</td>",
                    f"<td>{escape(dataset.reference_mode)}</td>",
                ]
            )
            + "</tr>"
        )
    replicate_rows: list[str] = []
    for dataset in manifest.datasets:
        for replicate in dataset.replicates:
            adaptive = next(
                (item for item in replicate.variants if item.id == "adaptive_mad"),
                None,
            )
            uniform = next(
                (item for item in replicate.variants if item.id == "uniform_none"),
                None,
            )
            values = [
                dataset.id,
                replicate.split_id,
                str(replicate.seed),
                replicate.status,
                _number(adaptive.final_holdout_rmse_m if adaptive else None),
                _number(uniform.final_holdout_rmse_m if uniform else None),
                f"{_number(replicate.holdout_improvement_percent, digits=2)}%",
                replicate.winner,
                ", ".join(replicate.failure_categories) or "none",
            ]
            replicate_rows.append(
                "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>"
            )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solid-state LiDAR benchmark</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
line-height: 1.45; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #d7dbe0; padding: .45rem .6rem; text-align: left; }}
th {{ background: #f1f3f4; }}
code {{ background: #f1f3f4; padding: .1rem .25rem; }}
.metric {{ display: inline-block; margin: .25rem 1.2rem .25rem 0; }}
.note {{ background: #fff8e1; border-left: 4px solid #f9ab00; padding: .7rem 1rem; }}
</style>
</head>
<body>
<h1>Solid-state LiDAR benchmark</h1>
<p>{escape(aggregate.conclusion)}</p>
<p>
<span class="metric"><strong>Datasets:</strong>
{aggregate.scored_dataset_count}/{aggregate.dataset_count}</span>
<span class="metric"><strong>Replicates:</strong>
{aggregate.scored_replicate_count}/{aggregate.total_replicate_count}</span>
<span class="metric"><strong>Adaptive win rate:</strong>
{escape(_number(aggregate.adaptive_win_rate, digits=3))}</span>
<span class="metric"><strong>Mean improvement:</strong>
{escape(_number(aggregate.mean_holdout_improvement_percent, digits=2))}%</span>
<span class="metric"><strong>95% CI:</strong>
{escape(_ci_from_interval(aggregate.holdout_improvement_ci))}</span>
</p>
<div class="note">Reference modes and failure categories are retained per dataset.
Temporal-holdout evidence is not an absolute-GT or universal-SOTA claim.</div>
<h2>Dataset summary</h2>
<table><thead><tr><th>Dataset</th><th>Status</th><th>Replicates</th>
<th>Adaptive win rate</th><th>Mean improvement</th><th>95% CI</th><th>Winner</th>
<th>Reference</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Replicate details</h2>
<table><thead><tr><th>Dataset</th><th>Split</th><th>Seed</th><th>Status</th>
<th>Adaptive RMSE (m)</th><th>Uniform RMSE (m)</th><th>Improvement</th>
<th>Winner</th><th>Failures</th></tr></thead>
<tbody>{''.join(replicate_rows)}</tbody></table>
<h2>Provenance</h2>
<p>Protocol <code>{escape(manifest.protocol.name)}</code>; schema
<code>{escape(manifest.schema_version)}</code>; spec SHA-256
<code>{escape(manifest.spec_sha256)}</code>; source artifacts
{len(manifest.provenance.source_paths)}.</p>
</body>
</html>
"""


def write_solid_state_cross_dataset_benchmark_reports(
    manifest: SolidStateCrossDatasetBenchmarkManifest,
    *,
    markdown_path: Path | None = None,
    html_path: Path | None = None,
) -> tuple[Path, ...]:
    """Write selected Markdown/HTML reports and return their paths."""

    written: list[Path] = []
    if markdown_path is not None:
        write_text(markdown_path, render_solid_state_cross_dataset_benchmark_markdown(manifest))
        written.append(markdown_path)
    if html_path is not None:
        write_text(html_path, render_solid_state_cross_dataset_benchmark_html(manifest))
        written.append(html_path)
    return tuple(written)
