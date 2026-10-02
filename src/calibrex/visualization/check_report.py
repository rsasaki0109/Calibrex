"""Self-contained HTML report for ``slac.calibration_check`` artifacts.

One static page, no scripts and no external assets: the overall verdict, a
table of the judged pairs (per-axis error against tolerance, unchecked axes
with their reasons, detectable error, evidence link and digest), the skipped
pairs, the closure loops, the candidate frame tree and the provenance.
Light and dark themes follow ``prefers-color-scheme``.
"""

from __future__ import annotations

import math
import os
import re
from html import escape
from pathlib import Path

from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckAxisJudgement,
    CheckClosure,
    CheckFocalComponent,
    CheckFrameEdge,
    CheckPairRecord,
)

ARROW = " \u2192 "
_ABSOLUTE_PATH = re.compile(r"(?<![\w.|])/(?:[^\s'\"<>/]+/)*[^\s'\"<>/]+")

_CSS = """
:root {
  --bg: #fbfbfa; --fg: #1b2127; --muted: #5b6770; --line: #d8dde1; --panel: #f1f3f4;
  --pass-bg: #d9f0e3; --pass-fg: #0f5a33; --warn-bg: #fff0c2; --warn-fg: #6a4a00;
  --fail-bg: #ffd9d9; --fail-fg: #7a1616; --inc-bg: #e4e7ea; --inc-fg: #3d474f;
  --bar: #7a8791; --accent: #2b6cb0;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14181c; --fg: #e4e8eb; --muted: #9aa6af; --line: #333c43; --panel: #1d2329;
    --pass-bg: #163a28; --pass-fg: #8fe0b2; --warn-bg: #43380f; --warn-fg: #f2d37a;
    --fail-bg: #4a1c1c; --fail-fg: #ffb4b4; --inc-bg: #2a3239; --inc-fg: #c1cad1;
    --bar: #8b99a3; --accent: #7db2ee;
  }
}
:root[data-theme="dark"] {
  --bg: #14181c; --fg: #e4e8eb; --muted: #9aa6af; --line: #333c43; --panel: #1d2329;
  --pass-bg: #163a28; --pass-fg: #8fe0b2; --warn-bg: #43380f; --warn-fg: #f2d37a;
  --fail-bg: #4a1c1c; --fail-fg: #ffb4b4; --inc-bg: #2a3239; --inc-fg: #c1cad1;
  --bar: #8b99a3; --accent: #7db2ee;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 32px 0 8px; padding-top: 8px; border-top: 1px solid var(--line); }
p, li { max-width: 80ch; }
.sub { color: var(--muted); margin: 0 0 16px; overflow-wrap: anywhere; }
code { background: var(--panel); padding: 1px 5px; border-radius: 4px;
  font: 0.88em ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; }
a { color: var(--accent); }
.banner { border-radius: 8px; padding: 14px 16px; margin: 12px 0 8px; font-size: 1.05rem; }
.banner strong { font-size: 1.3rem; text-transform: uppercase; letter-spacing: 0.04em; }
.banner ul { margin: 6px 0 0; padding-left: 20px; font-size: 0.92rem; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-weight: 700;
  font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em; white-space: nowrap; }
.pass { background: var(--pass-bg); color: var(--pass-fg); }
.warn { background: var(--warn-bg); color: var(--warn-fg); }
.fail { background: var(--fail-bg); color: var(--fail-fg); }
.inconclusive, .skipped, .planned { background: var(--inc-bg); color: var(--inc-fg); }
.tablewrap { overflow-x: auto; margin: 8px 0; }
table { border-collapse: collapse; width: 100%; min-width: 640px; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left;
  vertical-align: top; }
th { background: var(--panel); font-size: 0.8rem; text-transform: uppercase;
  letter-spacing: 0.03em; color: var(--muted); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.axis { display: grid; grid-template-columns: 4.2em 1fr; gap: 2px 8px; align-items: center;
  margin: 2px 0; min-width: 230px; }
.axis .name { color: var(--muted); }
.meter { position: relative; height: 8px; background: var(--panel); border-radius: 4px;
  border: 1px solid var(--line); overflow: hidden; }
.meter i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--bar); }
.meter i.pass { background: var(--pass-fg); } .meter i.warn { background: var(--warn-fg); }
.meter i.fail { background: var(--fail-fg); }
.meter b { position: absolute; left: 50%; top: -1px; bottom: -1px; width: 1px;
  background: var(--fg); opacity: 0.45; }
.axis .val { grid-column: 2; font-size: 0.82rem; color: var(--muted);
  font-variant-numeric: tabular-nums; }
.note { color: var(--muted); font-size: 0.88rem; }
.unchecked { color: var(--muted); font-size: 0.88rem; margin: 2px 0; }
.legend { color: var(--muted); font-size: 0.85rem; }
ul.tree, ul.tree ul { list-style: none; margin: 0; padding-left: 20px;
  border-left: 1px solid var(--line); }
ul.tree { border-left: 0; padding-left: 0; }
ul.tree li { margin: 2px 0; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 8px 0; }
dt { color: var(--muted); } dd { margin: 0; overflow-wrap: anywhere; }
details { margin: 8px 0; } summary { cursor: pointer; }
"""


def _redact(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    return _ABSOLUTE_PATH.sub(lambda match: ".../" + match.group(0).rsplit("/", 1)[-1], text)


def _badge(status: str) -> str:
    return f'<span class="badge {escape(status)}">{escape(status)}</span>'


def _number(value: float, digits: int = 3) -> str:
    if value == 0.0 or not math.isfinite(value):
        return f"{value:g}"
    return f"{value:.{digits}g}"


def _axis_row(axis: CheckAxisJudgement, *, signed: bool = False) -> str:
    fill = min(axis.ratio / 2.0, 1.0) * 100.0
    error = _number(axis.candidate_error) if signed else _number(abs(axis.candidate_error))
    sigma = f", 1&sigma; {_number(axis.estimate_std)}" if axis.estimate_std else ""
    return (
        '<div class="axis">'
        f'<span class="name">{escape(axis.name)}</span>'
        f'<span class="meter" title="{axis.ratio:.2f} x tolerance"><i class="{axis.status}" '
        f'style="width:{fill:.0f}%"></i><b></b></span>'
        f'<span class="val">{error} / {_number(axis.tolerance)} {axis.unit}'
        f"{sigma} ({_number(axis.ratio, 2)}x)"
        f"{'' if axis.status == 'pass' else ' ' + _badge(axis.status)}</span>"
        "</div>"
    )


def _focal_row(item: CheckFocalComponent) -> str:
    fill = min(item.ratio / 2.0, 1.0) * 100.0
    sigma = f", 1&sigma; {_number(item.scale_std * 100.0)} %" if item.scale_std else ""
    return (
        '<div class="axis">'
        f'<span class="name">{escape(item.name)}</span>'
        f'<span class="meter" title="{item.ratio:.2f} x tolerance"><i class="{item.status}" '
        f'style="width:{fill:.0f}%"></i><b></b></span>'
        f'<span class="val">scale {item.scale:.4f}: {_number(abs(item.scale_error) * 100.0)} / '
        f"{_number(item.tolerance * 100.0)} %{sigma} ({_number(item.ratio, 2)}x), "
        f"{_number(item.estimate_px, 4)} px against {_number(item.candidate_px, 4)} px deployed"
        f"{'' if item.status == 'pass' else ' ' + _badge(item.status)}</span>"
        "</div>"
    )


def _evidence_cell(pair: CheckPairRecord, artifact_dir: Path | None, html_dir: Path | None) -> str:
    if not pair.evidence:
        return '<span class="note">none</span>'
    items = []
    for ref in pair.evidence:
        label = escape(ref.path.rsplit("/", 1)[-1])
        if artifact_dir is not None and html_dir is not None:
            target = Path(os.path.relpath((artifact_dir / ref.path).resolve(), html_dir.resolve()))
            label = f'<a href="{escape(target.as_posix())}">{label}</a>'
        cached = " (cache)" if ref.from_cache else ""
        items.append(
            f"<div>{label}{cached}<br><code>sha256 {ref.sha256[:12]}</code> "
            f'<span class="note">{escape(ref.schema_version)}</span></div>'
        )
    return "".join(items)


def _pair_rows(
    artifact: CalibrationCheckArtifact, artifact_dir: Path | None, html_dir: Path | None
) -> str:
    rows = []
    for pair in artifact.pairs:
        if pair.status == "skipped":
            continue
        verdict = _badge(pair.status)
        if pair.coverage == "partial" and (
            pair.axes or (pair.focal_scale is not None and pair.focal_scale.components)
        ):
            verdict += '<div class="note">partial coverage</div>'
        axes = "".join(_axis_row(axis) for axis in pair.axes) or '<span class="note">none</span>'
        unchecked = "".join(
            f'<div class="unchecked"><b>{escape(item.name)}</b> unchecked: '
            f"{escape(item.reason)}</div>"
            for item in pair.unchecked_axes
        )
        detect = "<br>".join(
            f"{escape(axis.name)} {_number(axis.detectable_error)} {axis.unit}"
            for axis in pair.axes
        )
        if pair.focal_scale is not None:
            focal = pair.focal_scale
            axes = "".join(_focal_row(item) for item in focal.components) or (
                '<span class="note">none</span>'
            )
            unchecked = "".join(
                f'<div class="unchecked"><b>{escape(item.name)}</b> unchecked: '
                f"{escape(item.reason)}</div>"
                for item in focal.unchecked
            )
            if focal.optical_axis_ratio is not None:
                unchecked += (
                    '<div class="unchecked">optical-axis control ratio '
                    f"{focal.optical_axis_ratio:.4f} (must be 1)</div>"
                )
            detect = "<br>".join(
                f"{escape(item.name)} {_number(item.detectable_error * 100.0)} %"
                for item in focal.components
            )
        reason = f'<div class="note">{escape(pair.reason)}</div>' if pair.reason else ""
        rows.append(
            "<tr>"
            f'<td><b>{escape(pair.pair)}</b><div class="note">{escape(" / ".join(pair.sensors))}'
            f"</div></td><td>{verdict}</td>"
            f"<td>{axes}{unchecked}{reason}</td>"
            f"<td>{detect or '-'}</td>"
            f"<td>{_evidence_cell(pair, artifact_dir, html_dir)}</td></tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="tablewrap"><table><thead><tr><th>pair</th><th>verdict</th>'
        "<th>|candidate error| / tolerance</th><th>detectable error</th><th>evidence</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _skipped_rows(artifact: CalibrationCheckArtifact, redact: bool) -> str:
    rows = [
        f"<tr><td><b>{escape(pair.pair)}</b></td><td>{escape(pair.reason_code or '')}</td>"
        f"<td>{escape(_redact(pair.reason or '', redact))}</td></tr>"
        for pair in artifact.pairs
        if pair.status == "skipped"
    ]
    return (
        '<div class="tablewrap"><table><thead><tr><th>pair</th><th>reason</th><th>detail</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        if rows
        else ""
    )


def _closure_block(loop: CheckClosure) -> str:
    label = " + ".join(
        escape(m.pair) + ("" if m.direction == "forward" else " (inverse)") for m in loop.members
    )
    axes = "".join(_axis_row(axis, signed=True) for axis in loop.axes)
    unchecked = "".join(
        f'<div class="unchecked"><b>{escape(item.name)}</b> unchecked: {escape(item.reason)}</div>'
        for item in loop.unchecked_axes
    )
    notes = "".join(f'<div class="note">{escape(note)}</div>' for note in loop.notes)
    return (
        f'<tr><td>{escape(loop.kind)}<div class="note">{escape(ARROW.join(loop.frames))}'
        f"</div></td><td>{label}</td><td>{_badge(loop.verdict)}</td>"
        f"<td>{axes}{unchecked}{notes}</td></tr>"
    )


def _closure_section(artifact: CalibrationCheckArtifact) -> str:
    report = artifact.closures
    if report is None:
        return ""
    body = [
        "<h2>Closure of the estimates</h2>",
        '<p class="note">Independent estimates judged against each other instead of against '
        "the candidate. A failing loop means the estimators disagree, so their verdicts on the "
        "candidate are less trustworthy; it raises the overall verdict to warn at most.</p>",
    ]
    if report.loops:
        body.append(
            '<div class="tablewrap"><table><thead><tr><th>loop</th><th>members</th>'
            "<th>verdict</th><th>closure error / tolerance (signed)</th></tr></thead><tbody>"
            + "".join(_closure_block(loop) for loop in report.loops)
            + "</tbody></table></div>"
        )
    else:
        body.append(
            '<p class="note">No loop: no two independent estimates share a frame pair, and '
            "no three form a cycle.</p>"
        )
    excluded = [edge for edge in report.edges if edge.role != "independent"]
    if excluded:
        body.append(
            "<ul>"
            + "".join(
                f"<li><b>{escape(edge.pair)}</b>: {escape(edge.role)}"
                + (f" ({escape(edge.reason)})" if edge.reason else "")
                + "</li>"
                for edge in excluded
            )
            + "</ul>"
        )
    body.append(
        "<details><summary>assumptions</summary><ul>"
        + "".join(f"<li>{escape(item)}</li>" for item in report.assumptions)
        + "</ul></details>"
    )
    return "".join(body)


def _frame_tree(artifact: CalibrationCheckArtifact) -> str:
    tree = artifact.frame_tree
    children: dict[str, list[CheckFrameEdge]] = {}
    for edge in tree.edges:
        children.setdefault(edge.transform.parent_frame, []).append(edge)

    def render(frame: str, edge: CheckFrameEdge | None, seen: frozenset[str]) -> str:
        detail = ""
        if edge is not None:
            t = edge.transform.translation_m
            q = edge.transform.rotation_quat_xyzw
            detail = (
                f' <span class="note">t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m, '
                f"q=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}); "
                f"{escape(edge.source)}</span>"
            )
        inner = ""
        if frame not in seen:
            kids = children.get(frame, [])
            if kids:
                inner = (
                    "<ul>"
                    + "".join(
                        render(kid.transform.child_frame, kid, seen | {frame}) for kid in kids
                    )
                    + "</ul>"
                )
        return f"<li><code>{escape(frame)}</code>{detail}{inner}</li>"

    roots = tree.roots or sorted(
        {e.transform.parent_frame for e in tree.edges}
        - {e.transform.child_frame for e in tree.edges}
    )
    if not roots:
        return '<p class="note">No static transforms.</p>'
    return (
        '<ul class="tree">' + "".join(render(root, None, frozenset()) for root in roots) + "</ul>"
    )


def _banner(artifact: CalibrationCheckArtifact) -> str:
    if artifact.plan_only:
        return (
            f'<div class="banner inconclusive">{_badge("planned")} Plan only: no estimator was '
            "run, so nothing was judged.</div>"
        )
    verdict = artifact.overall_verdict or "inconclusive"
    counts = artifact.summary.status_counts
    judged = [
        f"{counts[name]} {name}"
        for name in ("pass", "warn", "fail", "inconclusive")
        if name in counts
    ]
    lines = [f"{artifact.summary.runnable_count} pair(s) ran: {', '.join(judged) or 'none judged'}"]
    skipped = counts.get("skipped", 0)
    if skipped:
        lines.append(f"{skipped} pair(s) skipped (see below)")
    if artifact.summary.partial_pairs:
        lines.append(
            f"{artifact.summary.partial_pairs} pair(s) have partial coverage: unchecked axes "
            "were not judged, so a pass covers only the judged axes"
        )
    report = artifact.closures
    if report is not None and report.loops:
        lines.append(f"closure of the estimates: {report.verdict}")
    worst_axes = [
        axis
        for pair in artifact.pairs
        if pair.status in {"warn", "fail"}
        for axis in (*pair.axes, *(pair.focal_scale.components if pair.focal_scale else ()))
        if axis.status != "pass"
    ]
    if worst_axes:
        lines.append(f"{len(worst_axes)} axis judgement(s) outside the tolerance")
    return (
        f'<div class="banner {escape(verdict)}"><strong>{escape(verdict)}</strong> '
        "overall verdict<ul>"
        + "".join(f"<li>{escape(line)}</li>" for line in lines)
        + "</ul></div>"
    )


def render_check_html(
    artifact: CalibrationCheckArtifact,
    *,
    artifact_dir: Path | None = None,
    html_dir: Path | None = None,
    redact_paths: bool = False,
) -> str:
    """Render a calibration check artifact as one self-contained HTML page.

    ``artifact_dir`` (where the evidence paths are relative to) and
    ``html_dir`` make the evidence file names links. With ``redact_paths``
    absolute paths in free text and in the provenance are shortened to their
    final component, so a report can be published without local paths.
    """

    options = artifact.options
    provenance = artifact.provenance
    command = " ".join(_redact(item, redact_paths) for item in provenance.command)
    bag_path = _redact(artifact.bag.path, redact_paths)
    legend = (
        '<p class="legend">Bars show |candidate error| as a fraction of twice the tolerance; the '
        "centre line is the tolerance (pass up to it, fail beyond twice it). tolerance = "
        + (
            f"max({_number(options.sigma_k)} &sigma;, floor), floors "
            f"{_number(options.rotation_floor_deg)} deg / "
            f"{_number(options.translation_floor_m)} m. "
            if options
            else "max(k &sigma;, floor). "
        )
        + "Detectable error: a deliberate error larger than this on the axis would be flagged."
        "</p>"
    )
    sources = (
        "".join(
            f"<li><code>{escape(source.kind)}</code> "
            f"{escape(_redact(source.path or '-', redact_paths))}"
            f" ({source.frame_count} frames)"
            + (f" <code>sha256 {source.sha256[:12]}</code>" if source.sha256 else "")
            + "</li>"
            for source in artifact.candidate_sources
        )
        or "<li>none</li>"
    )
    topics = "".join(
        f"<tr><td><code>{escape(t.topic)}</code></td><td>{escape(t.role or '-')}"
        f"{'/' + escape(t.odometry_kind) if t.odometry_kind else ''}</td>"
        f"<td>{escape(t.header_frame_id or '-')}</td>"
        f"<td>{escape(t.mapped_frame or '-')}"
        f"{' (' + escape(t.frame_source) + ')' if t.frame_source else ''}</td></tr>"
        for t in artifact.topics
    )
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>calibrex check - {escape(Path(bag_path).name)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>calibrex check</h1>",
        f'<p class="sub">Is the deployed calibration consistent with the data? Bag '
        f"<code>{escape(bag_path)}</code>, {artifact.bag.topic_count} topics"
        + (
            f", vehicle frame <code>{escape(artifact.vehicle_frame)}</code>"
            if artifact.vehicle_frame
            else ""
        )
        + f". Schema <code>{escape(artifact.schema_version)}</code>.</p>",
        _banner(artifact),
        "<h2>Pairs</h2>",
        legend,
        _pair_rows(artifact, artifact_dir, html_dir) or '<p class="note">No pair ran.</p>',
    ]
    skipped = _skipped_rows(artifact, redact_paths)
    if skipped:
        parts += ["<h2>Skipped pairs</h2>", skipped]
    parts.append(_closure_section(artifact))
    parts += [
        "<h2>Candidate frame tree</h2>",
        _frame_tree(artifact),
        "<h2>Candidate sources</h2>",
        f"<ul>{sources}</ul>",
        "<details><summary>Topics and frame mapping</summary>"
        '<div class="tablewrap"><table><thead><tr><th>topic</th><th>role</th>'
        "<th>header frame</th><th>tree frame</th></tr></thead><tbody>"
        + topics
        + "</tbody></table></div></details>",
        "<h2>Provenance</h2>",
        "<dl>"
        f"<dt>generator</dt><dd>{escape(provenance.generator)} "
        f"{escape(provenance.generator_version)}"
        + (
            f" (git <code>{escape(provenance.git_commit[:12])}</code>)"
            if provenance.git_commit
            else ""
        )
        + "</dd>"
        f"<dt>created</dt><dd>{escape(provenance.created_at)}</dd>"
        f"<dt>command</dt><dd><code>{escape(command) or '-'}</code></dd>"
        f"<dt>bag digest</dt><dd><code>{provenance.input_sha256}</code><br>"
        f'<span class="note">scope: {escape(provenance.input_digest_scope)}</span></dd>'
        + (
            f"<dt>evidence dir</dt><dd><code>{escape(artifact.evidence_dir)}</code></dd>"
            if artifact.evidence_dir
            else ""
        )
        + "</dl>",
        "<ul>"
        + "".join(f"<li>{escape(_redact(n, redact_paths))}</li>" for n in provenance.notes)
        + "</ul>",
        "</main></body></html>",
    ]
    return "\n".join(parts) + "\n"


def write_check_html(
    artifact: CalibrationCheckArtifact,
    path: Path,
    *,
    artifact_dir: Path | None = None,
    redact_paths: bool = False,
) -> Path:
    """Write :func:`render_check_html` to ``path`` (parent directories are created)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_check_html(
            artifact, artifact_dir=artifact_dir, html_dir=path.parent, redact_paths=redact_paths
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path
