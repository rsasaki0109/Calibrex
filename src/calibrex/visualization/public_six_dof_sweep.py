"""Offline overview linking every prespecified public six-DoF case."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from calibrex.core.benchmark import BenchmarkArtifact

if TYPE_CHECKING:
    from calibrex.pipelines.public_camera_lidar_demo import PublicSixDofDemoResult


def write_public_six_dof_sweep_report(
    output: Path,
    *,
    benchmark: BenchmarkArtifact,
    demos: dict[str, PublicSixDofDemoResult],
    cases: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
    workers: int,
    elapsed_seconds: float,
) -> Path:
    """Show per-case comparisons without substituting a cross-case speed ratio."""

    rows = []
    labels = {method.method_id: method.label for method in benchmark.methods}
    for case_id, rotation, translation in cases:
        demo = demos[case_id]
        for method_id, summary in demo.benchmark.method_summaries.items():
            metrics = summary.metrics
            rows.append(
                f'<tr><th><a href="{html.escape(case_id)}/report.html">{html.escape(case_id)}</a>'
                f'<br><small>R={rotation} deg<br>T={translation} m</small></th>'
                f'<td>{html.escape(labels[method_id])}</td>'
                f'<td>{summary.success_count} / {summary.trial_count}</td>'
                f'<td>{"PASS" if demo.numerical_parity else "FAIL"}</td>'
                f'<td>{_number(metrics["solve_seconds"].distribution.median)}</td>'
                f'<td>{_number(metrics["rotation_difference_deg"].distribution.median)}</td>'
                f'<td>{_number(metrics["translation_difference_m"].distribution.median)}</td>'
                f'<td>{_number(metrics["holdout_mi"].distribution.median)}</td></tr>'
            )
    limitations = "".join(f"<li>{html.escape(note)}</li>" for note in benchmark.limitations)
    success_count = sum(trial.status == "success" for trial in benchmark.trials)
    status = (
        "Complete comparison" if all(demo.successful for demo in demos.values()) else "Incomplete"
    )
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Calibrex public six-DoF sweep</title><style>
body {{font:16px/1.5 system-ui,sans-serif;color:#203247;background:#f2f6fa;margin:0;padding:24px}}
main {{max-width:1300px;margin:auto}} .table-wrap {{overflow-x:auto}}
table {{border-collapse:collapse;width:100%;background:white}} th,td {{padding:12px;text-align:left;
border-bottom:1px solid #dbe4ed}} small {{font-size:12px}} a {{color:#176787}}
code {{overflow-wrap:anywhere}} .status {{font-weight:700}} li {{margin:8px 0}}
</style></head><body><main><h1>KITTI raw: signed-axis CPU/candidate sweep</h1>
<p class="status">{status}: {success_count} / {len(benchmark.trials)} successful attempts</p>
<p>Six prespecified positive/negative x, y, z initializations. Each uses the same verified
25 frames, 20 fit / 5 holdout split, provider and solver settings.
Rotation is local Euler; translation is in the camera frame. Dataset calibration supplies
a controlled initial perturbation and reference differences, never the optimizer target.</p>
<p>CPU worker process limit: {workers}. Elapsed through all case reports: {elapsed_seconds:.3f} s.
CPU-only comparisons may run cases in separate processes; CUDA comparisons require one worker.
This elapsed time includes shared verification, worker startup, case input verification/loading,
optimization, holdout, serialization and individual HTML reports.
Suite summary writing is excluded.</p>
<h2>Per-case successful-run medians</h2><div class="table-wrap"><table><thead><tr>
<th>Case / initial offsets</th><th>Method</th><th>Successful / attempted</th><th>Exact parity</th>
<th>Solve (s)</th><th>Rotation difference (deg)</th><th>Translation difference (m)</th>
<th>Holdout MI (nat)</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p>Each case links to every attempt, failure reason, full traces, per-invocation timings and
holdout overlays. Failed/partial trials remain in those reports and the suite benchmark;
they are excluded from medians above. Parity compares backends and repetitions within a case.
Different cases may converge to different poses. No cross-case speed ratio or accuracy
acceptance threshold is inferred. Quality remains a warning with unknown
observability/covariance.</p>
<h2>Artifacts</h2><ul>
<li><a href="sweep.plan.definition.json">Frozen plan (unexecuted entries)</a></li>
<li><a href="sweep.source.problem.json">Verified source problem</a></li>
<li><a href="sweep.definition.json">Suite definition</a></li>
<li><a href="sweep.benchmark.json">Suite benchmark</a></li></ul>
<details><summary>Limitations</summary><ul>{limitations}</ul></details>
</main></body></html>'''
    report = output / "report.html"
    report.write_text(page, encoding="utf-8")
    return report


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.5f}"
