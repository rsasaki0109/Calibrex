"""Self-contained HTML report for ``slac.bag_estimate`` artifacts.

One static page that reuses the style of the ``calibrex check`` report (and its
light and dark themes): the run summary, each pair's six axes as value, 1 sigma
and observability status (an axis the data did not observe is greyed and says
where its value came from), the exported and omitted frames, the exported
files with their digests, copyable commands for the next step, and the
provenance. The only script is the copy-button handler; without it the commands
are still plain selectable text.
"""

from __future__ import annotations

import math
import os
from html import escape
from pathlib import Path

from calibrex.core.bag_estimate import (
    BagEstimateArtifact,
    EstimateAxis,
    EstimateFrameEntry,
    EstimatePairRecord,
)
from calibrex.visualization.check_report import _CSS as _CHECK_CSS
from calibrex.visualization.check_report import _number, _redact

_AXIS_ORDER = ("roll", "pitch", "yaw", "x", "y", "z")
_STATUS_TEXT = {
    "observed": "observed",
    "unobservable": "unobservable",
    "control_not_detected": "control not detected",
    "no_estimate": "no estimate",
    "not_estimated": "not estimated",
}
_PAIR_CLASS = {"estimated": "pass", "partial": "warn", "failed": "fail", "skipped": "skipped"}
_POLICY_CLASS = {"pass": "pass", "warn": "warn", "fail": "fail"}
# 1 sigma bars are drawn on a log scale: (low, high, reference) per unit. The reference mark is
# the default tolerance floor of ``calibrex check`` (0.5 deg, 0.02 m).
_SCALE = {"deg": (1e-3, 10.0, 0.5), "m": (1e-4, 1.0, 0.02)}

_CSS = (
    _CHECK_CSS
    + """
.observed { background: var(--pass-bg); color: var(--pass-fg); }
.unobservable, .control_not_detected { background: var(--warn-bg); color: var(--warn-fg); }
.no_estimate { background: var(--fail-bg); color: var(--fail-fg); }
.not_estimated { background: var(--inc-bg); color: var(--inc-fg); }
h3 { font-size: 1rem; margin: 20px 0 4px; }
.pair { border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; margin: 14px 0; }
.pair h3 { margin: 0 0 2px; font-size: 1.02rem; }
.pair.skip { padding: 6px 14px; margin: 8px 0; }
.axes { display: grid; grid-template-columns: 3.4em minmax(120px, 1fr) minmax(0, 2.2fr);
  gap: 5px 10px; align-items: center; margin: 10px 0 6px; }
.axes .name { font-weight: 600; }
.axes .txt { font-size: 0.88rem; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.axes .txt .note { display: block; }
.axes .off { color: var(--muted); }
@media (max-width: 640px) {
  .axes { grid-template-columns: 3.4em 1fr; }
  .axes .txt { grid-column: 1 / -1; margin-bottom: 6px; }
}
svg.sigma { width: 100%; height: 14px; display: block; }
svg.sigma .track { fill: var(--panel); stroke: var(--line); }
svg.sigma .bar { fill: var(--pass-fg); }
svg.sigma .bar.off { fill: var(--bar); fill-opacity: 0.45; }
svg.sigma .ref { stroke: var(--fg); opacity: 0.5; stroke-width: 1; }
.notmeasured { color: var(--warn-fg); background: var(--warn-bg); border-radius: 4px;
  padding: 0 5px; font-weight: 600; font-size: 0.8rem; }
.cmd { position: relative; margin: 6px 0; }
.cmd pre { margin: 0; padding: 8px 76px 8px 10px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 6px; overflow-x: auto; white-space: pre;
  font: 0.84rem ui-monospace, SFMono-Regular, Menlo, monospace; }
.cmd button, button.copy { cursor: pointer; background: var(--bg); color: var(--fg);
  border: 1px solid var(--line); border-radius: 5px; font-size: 0.78rem; padding: 2px 8px; }
.cmd button { position: absolute; top: 6px; right: 6px; }
td code.sha { word-break: break-all; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
"""
)

_SCRIPT = """
document.addEventListener('click', function (event) {
  var button = event.target.closest('button[data-copy]');
  if (!button) { return; }
  var text = button.getAttribute('data-copy');
  var done = function () {
    var old = button.textContent; button.textContent = 'copied';
    setTimeout(function () { button.textContent = old; }, 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, function () {});
  }
});
"""


def _badge(label: str, css_class: str) -> str:
    return f'<span class="badge {escape(css_class)}">{escape(label)}</span>'


def _shell_quote(text: str) -> str:
    if text and all(c.isalnum() or c in "/._-:=@+,%" for c in text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


def _copy_block(text: str) -> str:
    return (
        f'<div class="cmd"><pre>{escape(text)}</pre>'
        f'<button type="button" data-copy="{escape(text, quote=True)}">copy</button></div>'
    )


def _sigma_svg(axis: EstimateAxis, *, off: bool) -> str:
    low, high, ref = _SCALE[axis.unit]
    span = math.log10(high / low)

    def at(value: float) -> float:
        return max(0.0, min(1.0, math.log10(max(value, low) / low) / span)) * 199.0

    parts = [
        '<svg class="sigma" viewBox="0 0 200 14" preserveAspectRatio="none" role="img" '
        f'aria-label="1 sigma on a log scale from {_number(low)} to {_number(high)} {axis.unit}">'
        '<rect class="track" x="0.5" y="2.5" width="199" height="9" rx="2"/>'
    ]
    if axis.std is not None and axis.std > 0.0:
        parts.append(
            f'<rect class="bar{" off" if off else ""}" x="0.5" y="2.5" '
            f'width="{max(at(axis.std), 2.0):.1f}" height="9" rx="2"/>'
        )
    parts.append(f'<line class="ref" x1="{at(ref):.1f}" x2="{at(ref):.1f}" y1="0" y2="14"/></svg>')
    return "".join(parts)


def _axis_rows(pair: EstimatePairRecord, redact: bool) -> str:
    rows = []
    axes: dict[str, EstimateAxis] = {axis.name: axis for axis in pair.axes}
    for name in _AXIS_ORDER:
        axis = axes.get(name)
        if axis is None:
            continue
        off = axis.status != "observed"
        value = _number(axis.value) if axis.value is not None else None
        sigma = _number(axis.std) if axis.std is not None else None
        if not off:
            text = "observed" if value is None else f"{value}"
            if value is not None and sigma:
                text += f" &plusmn; {sigma}"
            if value is not None:
                text += f" {axis.unit}"
        else:
            if axis.status == "not_estimated":
                headline = "not attempted by this estimator"
            elif pair.fill == "prior":
                headline = '<span class="notmeasured">from prior &mdash; NOT MEASURED</span>'
            else:
                headline = '<span class="notmeasured">NOT MEASURED</span>'
            bits = [headline]
            if value is not None:
                bits.append(f"{value} {axis.unit} (not trusted)")
            if sigma:
                bits.append(f"1&sigma; {sigma} {axis.unit}")
            text = " ".join(bits)
        reason = (
            f'<span class="note">{escape(_redact(axis.reason, redact))}</span>'
            if axis.reason and axis.status != "not_estimated"
            else ""
        )
        label = _STATUS_TEXT.get(axis.status, axis.status)
        rows.append(
            f'<span class="name{" off" if off else ""}">{escape(name)}</span>'
            f"{_sigma_svg(axis, off=off)}"
            f'<span class="txt{" off" if off else ""}">{_badge(label, axis.status)} {text}'
            f"{reason}</span>"
        )
    return '<div class="axes">' + "".join(rows) + "</div>"


def _evidence(pair: EstimatePairRecord, artifact_dir: Path | None, html_dir: Path | None) -> str:
    items = []
    for ref in pair.evidence:
        label = escape(ref.path.rsplit("/", 1)[-1])
        if artifact_dir is not None and html_dir is not None:
            target = Path(os.path.relpath((artifact_dir / ref.path).resolve(), html_dir.resolve()))
            label = f'<a href="{escape(target.as_posix())}">{label}</a>'
        cached = " (cache)" if ref.from_cache else ""
        items.append(
            f"<li>{label}{cached} <code>sha256 {ref.sha256[:12]}</code> "
            f'<span class="note">{escape(ref.schema_version)}'
            + (f", policy {escape(ref.policy_status)}" if ref.policy_status else "")
            + "</span></li>"
        )
    return f"<ul>{''.join(items)}</ul>" if items else ""


def _pair_block(
    pair: EstimatePairRecord,
    artifact_dir: Path | None,
    html_dir: Path | None,
    redact: bool,
) -> str:
    title = f"<b>{escape(pair.pair)}</b> " + _badge(pair.status, _PAIR_CLASS[pair.status])
    where = " / ".join(pair.sensors or pair.frames)
    if pair.status == "skipped":
        reason = (f" <code>{escape(pair.reason_code)}</code>" if pair.reason_code else "") + (
            f" {escape(_redact(pair.reason or '', redact))}"
        )
        return (
            f'<div class="pair skip"><h3>{title}</h3><div class="note">{escape(where)}'
            f"{reason}</div></div>"
        )
    parts = [f'<div class="pair"><h3>{title}</h3><div class="note">{escape(where)}</div>']
    if pair.reason:
        parts.append(f'<p class="note">{escape(_redact(pair.reason, redact))}</p>')
    if pair.transform is not None:
        parts.append(
            f'<div class="note">T_{escape(pair.transform.parent_frame)}_'
            f"{escape(pair.transform.child_frame)}; rotation vector in degrees about the "
            "parent frame's axes, translation in metres</div>"
        )
    parts.append(_axis_rows(pair, redact))
    facts = []
    if pair.estimator:
        facts.append(f"estimator <code>{escape(pair.estimator)}</code>")
    if pair.estimator_policy_status:
        status = pair.estimator_policy_status
        facts.append(f"policy {_badge(status, _POLICY_CLASS.get(status, 'inconclusive'))}")
    if pair.deskew:
        facts.append(f"deskew <code>{escape(pair.deskew)}</code>")
    if pair.time_offset is not None:
        offset = pair.time_offset
        if offset.status == "estimated":
            std = f" &plusmn; {_number(offset.std_s * 1e3)}" if offset.std_s is not None else ""
            facts.append(f"time offset {_number(offset.estimate_s * 1e3)}{std} ms")
        else:
            facts.append("time offset unobservable")
    if pair.fill:
        facts.append(
            "non-observed axes filled from "
            + ("the <code>--tf</code> prior" if pair.fill == "prior" else "an identity placeholder")
        )
    if pair.runtime_s is not None:
        facts.append(f"runtime {pair.runtime_s:.1f} s")
    if facts:
        parts.append(f'<div class="note">{" &middot; ".join(facts)}</div>')
    if pair.estimator_policy_reasons:
        parts.append(
            '<ul class="note">'
            + "".join(
                f"<li>{escape(_redact(item, redact))}</li>"
                for item in pair.estimator_policy_reasons
            )
            + "</ul>"
        )
    if pair.notes:
        parts.append(
            "<details><summary>notes</summary><ul>"
            + "".join(f"<li>{escape(_redact(n, redact))}</li>" for n in pair.notes)
            + "</ul></details>"
        )
    if pair.evidence:
        parts.append(
            '<details><summary>evidence</summary><p class="note">Estimator artifacts; their '
            "reference fields compare against the placeholder or prior and are not "
            "meaningful.</p>" + _evidence(pair, artifact_dir, html_dir) + "</details>"
        )
    parts.append("</div>")
    return "".join(parts)


def _frame_tree(artifact: BagEstimateArtifact) -> str:
    frames = artifact.frames
    if not frames.entries:
        return '<p class="note">No frame was exported.</p>'
    children: dict[str, list[EstimateFrameEntry]] = {}
    for entry in frames.entries:
        children.setdefault(entry.parent, []).append(entry)

    def render(frame: str, entry: EstimateFrameEntry | None, seen: frozenset[str]) -> str:
        detail = ""
        if entry is not None:
            t = entry.transform.translation_m
            q = entry.transform.rotation_quat_xyzw
            detail = (
                f' <span class="note">t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m, '
                f"q=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}); {escape(entry.pair)}</span> "
            )
            if entry.axes_from_prior:
                detail += (
                    '<span class="notmeasured">NOT MEASURED: '
                    f"{escape(', '.join(entry.axes_from_prior))} from prior</span>"
                )
            else:
                detail += _badge("all six observed", "observed")
        kids = children.get(frame, []) if frame not in seen else []
        inner = (
            "<ul>" + "".join(render(k.frame, k, seen | {frame}) for k in kids) + "</ul>"
            if kids
            else ""
        )
        return f"<li><code>{escape(frame)}</code>{detail}{inner}</li>"

    roots = (
        [frames.root] if frames.root else sorted(set(children) - {e.frame for e in frames.entries})
    )
    return '<ul class="tree">' + "".join(render(r, None, frozenset()) for r in roots) + "</ul>"


def _omitted(artifact: BagEstimateArtifact, redact: bool) -> str:
    rows = [
        f"<tr><td><code>{escape(o.frame)}</code></td><td>{escape(o.pair or '-')}</td>"
        f"<td>{escape(_redact(o.reason, redact))}</td></tr>"
        for o in artifact.frames.omitted
    ]
    if not rows:
        return ""
    return (
        "<h3>Omitted frames</h3>"
        '<div class="tablewrap"><table><thead><tr><th>frame</th><th>pair</th><th>why</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _exports(
    artifact: BagEstimateArtifact, artifact_dir: Path | None, html_dir: Path | None
) -> str:
    if not artifact.exports:
        return '<p class="note">Nothing was exported (see the omitted frames and next steps).</p>'
    rows = []
    for item in artifact.exports:
        label = escape(item.path)
        if artifact_dir is not None and html_dir is not None:
            target = Path(os.path.relpath((artifact_dir / item.path).resolve(), html_dir.resolve()))
            label = f'<a href="{escape(target.as_posix())}">{label}</a>'
        rows.append(
            f"<tr><td>{label}</td><td>{escape(item.kind)}</td>"
            f'<td><code class="sha">{item.sha256}</code></td>'
            f'<td><button type="button" class="copy" data-copy="{item.sha256}">copy</button></td>'
            "</tr>"
        )
    return (
        '<div class="tablewrap"><table><thead><tr><th>file</th><th>kind</th><th>sha256</th>'
        "<th></th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _commands(artifact: BagEstimateArtifact, directory: str) -> str:
    out = []
    kinds = {item.kind: item.path for item in artifact.exports}
    if "frames_yaml" in kinds:
        frames_path = f"{directory.rstrip('/')}/{kinds['frames_yaml']}"
        out.append(
            "<h3>Verify on a different recording</h3>"
            '<p class="note">The exported tree is a candidate for '
            "<code>calibrex check</code>; judging it against the bag it was estimated from "
            "proves nothing.</p>"
            + _copy_block(f"calibrex check <other-bag> --tf {_shell_quote(frames_path)}")
        )
    if artifact.frames.entries:
        lines = []
        for entry in artifact.frames.entries:
            t = entry.transform.translation_m
            q = entry.transform.rotation_quat_xyzw
            if entry.axes_from_prior:
                lines.append(
                    f"# {entry.frame}: NOT MEASURED axes {', '.join(entry.axes_from_prior)} "
                    "come from your --tf prior"
                )
            lines.append(
                "ros2 run tf2_ros static_transform_publisher "
                f"--x {t[0]:.9g} --y {t[1]:.9g} --z {t[2]:.9g} "
                f"--qx {q[0]:.9g} --qy {q[1]:.9g} --qz {q[2]:.9g} --qw {q[3]:.9g} "
                f"--frame-id {_shell_quote(entry.parent)} "
                f"--child-frame-id {_shell_quote(entry.frame)}"
            )
        out.append(
            "<h3>Publish the transforms (ROS 2)</h3>"
            '<p class="note">Each command publishes <code>T_parent_child</code> on '
            "<code>/tf_static</code>.</p>" + _copy_block("\n".join(lines))
        )
    return "".join(out)


def _summary(artifact: BagEstimateArtifact) -> str:
    counts = artifact.summary.status_counts
    chips = "".join(
        _badge(f"{counts[name]} {name}", _PAIR_CLASS[name])
        for name in ("estimated", "partial", "failed", "skipped")
        if counts.get(name)
    )
    lines = [
        f"{artifact.summary.frames_written} frame(s) exported, "
        f"{artifact.summary.frames_omitted} omitted"
    ]
    skipped = ", ".join(f"{n} {k}" for k, n in sorted(artifact.summary.skipped_by_reason.items()))
    if skipped:
        lines.append(f"skipped: {skipped}")
    lines.append(
        "an axis is a measurement only when the data observed it and the estimator's known-bad "
        "control detected it; every other value is a prior or a placeholder"
    )
    return (
        f'<div class="chips">{chips or "<span class=note>no pair</span>"}</div><ul>'
        + "".join(f"<li>{escape(line)}</li>" for line in lines)
        + "</ul>"
    )


def render_estimate_html(
    artifact: BagEstimateArtifact,
    *,
    artifact_dir: Path | None = None,
    html_dir: Path | None = None,
    redact_paths: bool = False,
) -> str:
    """Render a bag estimate artifact as one self-contained HTML page.

    ``artifact_dir`` (where the exported and evidence paths are relative to) and
    ``html_dir`` make the file names links. With ``redact_paths`` absolute paths
    are shortened to their final component.
    """

    provenance = artifact.provenance
    bag_path = _redact(artifact.bag.path, redact_paths)
    command = " ".join(_redact(item, redact_paths) for item in provenance.command)
    directory = _redact(str(artifact_dir), redact_paths) if artifact_dir is not None else "DIR"
    ran = [p for p in artifact.pairs if p.status != "skipped"]
    skipped = [p for p in artifact.pairs if p.status == "skipped"]
    sources = (
        "".join(
            f"<li><code>{escape(s.kind)}</code> {escape(_redact(s.path or '-', redact_paths))}"
            f" ({s.frame_count} frames)"
            + (f" <code>sha256 {s.sha256[:12]}</code>" if s.sha256 else "")
            + "</li>"
            for s in artifact.prior_sources
        )
        or "<li>none: every non-observed axis is an identity placeholder</li>"
    )
    legend = (
        '<p class="legend">Bars show 1&sigma; on a log scale (0.001 to 10 deg, 0.1 mm to 1 m); '
        "the vertical mark is the default tolerance floor of <code>calibrex check</code> "
        "(0.5 deg, 0.02 m): a bar well left of it can judge a calibration to that tolerance. "
        "Greyed rows are not measurements.</p>"
    )
    if artifact.next_steps:
        steps = (
            "<ul>"
            + "".join(f"<li>{escape(_redact(s, redact_paths))}</li>" for s in artifact.next_steps)
            + "</ul>"
        )
    else:
        steps = '<p class="note">None.</p>'
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>calibrex estimate - {escape(Path(bag_path).name)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>calibrex estimate</h1>",
        '<p class="sub">What the bag\'s own data say about its extrinsics. Bag '
        f"<code>{escape(bag_path)}</code>, {artifact.bag.topic_count} topics"
        + (
            f", vehicle frame <code>{escape(artifact.vehicle_frame)}</code>"
            if artifact.vehicle_frame
            else ""
        )
        + f". Schema <code>{escape(artifact.schema_version)}</code>.</p>",
        _summary(artifact),
        "<h2>Pairs</h2>",
        legend,
        "".join(_pair_block(p, artifact_dir, html_dir, redact_paths) for p in ran)
        or '<p class="note">No pair ran.</p>',
    ]
    if skipped:
        parts.append(
            f"<details><summary>{len(skipped)} skipped pair(s)</summary>"
            + "".join(_pair_block(p, artifact_dir, html_dir, redact_paths) for p in skipped)
            + "</details>"
        )
    parts += [
        "<h2>Exported frame tree</h2>",
        _frame_tree(artifact),
        _omitted(artifact, redact_paths),
        "<h2>Exported files</h2>",
        _exports(artifact, artifact_dir, html_dir),
        _commands(artifact, directory),
        "<h2>Next steps</h2>",
        steps,
        "<h2>Prior sources</h2>",
        f"<ul>{sources}</ul>",
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
        f"</main><script>{_SCRIPT}</script></body></html>",
    ]
    return "\n".join(parts) + "\n"


def write_estimate_html(
    artifact: BagEstimateArtifact,
    path: Path,
    *,
    artifact_dir: Path | None = None,
    redact_paths: bool = False,
) -> Path:
    """Write :func:`render_estimate_html` to ``path`` (parent directories are created)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_estimate_html(
            artifact, artifact_dir=artifact_dir, html_dir=path.parent, redact_paths=redact_paths
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path
