"""Human-readable reports for solid-state counterexample diagnostics."""

from __future__ import annotations

from html import escape
from pathlib import Path

from calibrex.core.io import write_text
from calibrex.core.solid_state_failure_analysis import (
    SolidStateFailureAnalysisManifest,
    SolidStateFailureAnalysisVariant,
)


def _number(value: float | None, *, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _percent(value: float | None, *, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100.0:.{digits}f}%"


def _variant_row(variant: SolidStateFailureAnalysisVariant | None) -> list[str]:
    if variant is None:
        return ["n/a"] * 12
    return [
        variant.id,
        variant.optimization_status or "n/a",
        _number(variant.final_train_rmse_m),
        _number(variant.final_holdout_rmse_m),
        _number(variant.train_holdout_gap_m),
        _number(variant.holdout_train_ratio, digits=3),
        str(variant.train_correspondence_count),
        str(variant.holdout_correspondence_count),
        _percent(variant.outlier_rejection_rate),
        str(variant.observability_rank) if variant.observability_rank is not None else "n/a",
        _number(variant.condition_number, digits=2),
        _number(variant.estimated_time_offset_sec, digits=3),
    ]


def render_solid_state_failure_analysis_markdown(
    analysis: SolidStateFailureAnalysisManifest,
) -> str:
    """Render a compact diagnostic report with explicit causal caveats."""

    summary = analysis.summary
    lines = [
        f"# Solid-state counterexample analysis: {analysis.dataset_id}",
        "",
        f"- Schema: `{analysis.schema_version}`",
        f"- Benchmark manifest SHA-256: `{analysis.benchmark_manifest_sha256}`",
        f"- Replicates: `{summary.scored_replicate_count}/{summary.replicate_count}` scored",
        "",
        "## Conclusion",
        "",
        analysis.conclusion,
        "",
        "## Summary",
        "",
        f"- Adaptive wins: **{summary.adaptive_wins}/{summary.scored_replicate_count}**",
        f"- Uniform wins: **{summary.uniform_wins}/{summary.scored_replicate_count}**",
        "- Mean holdout improvement: **"
        f"{_number(summary.mean_holdout_improvement_percent, digits=2)}%**",
        "- Mean adaptive train-minus-uniform RMSE: **"
        f"{_number(summary.mean_adaptive_train_minus_uniform_m)} m**",
        "- Mean adaptive generalization-gap delta: **"
        f"{_number(summary.mean_adaptive_generalization_gap_minus_uniform_m)} m**",
        "- Mean adaptive rejection rate: **"
        f"{_percent(summary.mean_adaptive_outlier_rejection_rate)}**",
        f"- Rank-6 evidence: **adaptive {summary.adaptive_rank_six_count}/"
        f"{summary.scored_replicate_count}, uniform {summary.uniform_rank_six_count}/"
        f"{summary.scored_replicate_count}**",
        "",
        "## Findings",
        "",
        "| Severity | Confidence | Finding | Evidence | Caveat |",
        "|---|---|---|---|---|",
    ]
    for finding in analysis.findings:
        lines.append(
            "| "
            + " | ".join(
                [
                    finding.severity,
                    finding.confidence,
                    finding.title,
                    finding.evidence,
                    finding.caveat,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Replicate comparison",
            "",
            "Positive holdout delta means adaptive has higher (worse) RMSE. "
            "Negative train delta means adaptive has lower train RMSE.",
            "",
            "| Replicate | Split | Seed | Winner | Diagnosis | Holdout delta (m) | "
            "Train delta (m) | Gap delta (m) | Offset delta (s) | Signals |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for replicate in analysis.replicates:
        lines.append(
            "| "
            + " | ".join(
                [
                    replicate.id,
                    replicate.split_id,
                    str(replicate.seed),
                    replicate.winner,
                    replicate.diagnosis,
                    _number(replicate.adaptive_holdout_minus_uniform_m),
                    _number(replicate.adaptive_train_minus_uniform_m),
                    _number(replicate.adaptive_generalization_gap_minus_uniform_m),
                    _number(replicate.adaptive_offset_minus_uniform_sec, digits=3),
                    ", ".join(replicate.signals) or "none",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Variant diagnostics",
            "",
            "RMSE and correspondence counts are copied from the revalidated v0.1 "
            "continuous-time result artifacts. Rejection rate uses retained plus "
            "rejected correspondences as its denominator.",
            "",
            "| Replicate | Variant | Status | Train RMSE | Holdout RMSE | Train→holdout gap | "
            "Holdout/train | Train corr. | Holdout corr. | Rejected | Rank | Condition | Offset |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for replicate in analysis.replicates:
        for variant in (replicate.adaptive, replicate.uniform):
            row = _variant_row(variant)
            lines.append("| " + " | ".join([replicate.id, *row]) + " |")
    lines.extend(
        [
            "",
            "## Provenance and limitations",
            "",
            f"- Source files/digests recorded: **{len(analysis.provenance.source_sha256)}**",
            "- The v0.1 result artifact does not contain residual histograms, per-iteration "
            "correspondence counts, or adaptive voxel-cell support distributions.",
            "- Findings labelled `candidate_cause` are hypotheses for the next controlled "
            "ablation, not causal conclusions.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python tools/run_solid_state_failure_analysis.py "
            f"{analysis.benchmark_manifest_path} "
            "--dataset-id agrob_modular_e "
            "--output outputs/agrob_solid_state_failure_analysis.yaml "
            "--markdown-output outputs/agrob_solid_state_failure_analysis.md "
            "--html-output outputs/agrob_solid_state_failure_analysis.html",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _table_row(values: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>"


def render_solid_state_failure_analysis_html(
    analysis: SolidStateFailureAnalysisManifest,
) -> str:
    """Render a self-contained HTML diagnostic report."""

    summary = analysis.summary
    finding_rows = "".join(
        _table_row(
            [
                finding.severity,
                finding.confidence,
                finding.title,
                finding.evidence,
                finding.caveat,
            ]
        )
        for finding in analysis.findings
    )
    replicate_rows = "".join(
        _table_row(
            [
                replicate.id,
                replicate.split_id,
                str(replicate.seed),
                replicate.winner,
                replicate.diagnosis,
                _number(replicate.adaptive_holdout_minus_uniform_m),
                _number(replicate.adaptive_train_minus_uniform_m),
                _number(replicate.adaptive_generalization_gap_minus_uniform_m),
                ", ".join(replicate.signals) or "none",
            ]
        )
        for replicate in analysis.replicates
    )
    variant_rows = "".join(
        _table_row([replicate.id, *(_variant_row(variant))])
        for replicate in analysis.replicates
        for variant in (replicate.adaptive, replicate.uniform)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Solid-state counterexample analysis: {escape(analysis.dataset_id)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1500px; margin: 2rem auto;
padding: 0 1rem; line-height: 1.45; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: .9rem; }}
th, td {{ border: 1px solid #d7dbe0; padding: .4rem .5rem; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f4; }}
code {{ background: #f1f3f4; padding: .1rem .25rem; }}
.metric {{ display: inline-block; margin: .25rem 1.2rem .25rem 0; }}
.note {{ background: #fff8e1; border-left: 4px solid #f9ab00; padding: .7rem 1rem; }}
</style>
</head>
<body>
<h1>Solid-state counterexample analysis: {escape(analysis.dataset_id)}</h1>
<p>{escape(analysis.conclusion)}</p>
<p>
<span class="metric"><strong>Scored:</strong>
{summary.scored_replicate_count}/{summary.replicate_count}</span>
<span class="metric"><strong>Adaptive wins:</strong> {summary.adaptive_wins}</span>
<span class="metric"><strong>Uniform wins:</strong> {summary.uniform_wins}</span>
<span class="metric"><strong>Mean improvement:</strong>
{escape(_number(summary.mean_holdout_improvement_percent, digits=2))}%</span>
</p>
<div class="note">Candidate-cause findings are hypotheses for controlled follow-up
ablations. They are not causal conclusions.</div>
<h2>Findings</h2>
<table><thead><tr><th>Severity</th><th>Confidence</th><th>Finding</th><th>Evidence</th><th>Caveat</th></tr></thead>
<tbody>{finding_rows}</tbody></table>
<h2>Replicate comparison</h2>
<table><thead><tr><th>Replicate</th><th>Split</th><th>Seed</th><th>Winner</th><th>Diagnosis</th>
<th>Holdout delta (m)</th><th>Train delta (m)</th><th>Gap delta (m)</th>
<th>Signals</th></tr></thead>
<tbody>{replicate_rows}</tbody></table>
<h2>Variant diagnostics</h2>
<table><thead><tr><th>Replicate</th><th>Variant</th><th>Status</th>
<th>Train RMSE</th><th>Holdout RMSE</th>
<th>Train→holdout gap</th><th>Holdout/train</th><th>Train corr.</th>
<th>Holdout corr.</th><th>Rejected</th>
<th>Rank</th><th>Condition</th><th>Offset</th></tr></thead>
<tbody>{variant_rows}</tbody></table>
<h2>Provenance</h2>
<p>Schema <code>{escape(analysis.schema_version)}</code>; benchmark SHA-256
<code>{escape(analysis.benchmark_manifest_sha256)}</code>; recorded source digests
{len(analysis.provenance.source_sha256)}.</p>
</body>
</html>
"""


def write_solid_state_failure_analysis_reports(
    analysis: SolidStateFailureAnalysisManifest,
    *,
    markdown_path: Path | None = None,
    html_path: Path | None = None,
) -> tuple[Path, ...]:
    """Write selected Markdown/HTML reports and return their paths."""

    written: list[Path] = []
    if markdown_path is not None:
        write_text(markdown_path, render_solid_state_failure_analysis_markdown(analysis))
        written.append(markdown_path)
    if html_path is not None:
        write_text(html_path, render_solid_state_failure_analysis_html(analysis))
        written.append(html_path)
    return tuple(written)
