"""Self-contained HTML report of a ``slac.calibration_drift`` artifact.

One card per pair; per axis an error-bar chart of each bag's estimate (value
+- std) with the minimum detectable change drawn as a band around the weighted
mean, so a reader sees at a glance whether the bags agree and how big a change
could have been seen at all.
"""

from __future__ import annotations

import html
from pathlib import Path

from calibrex.core.calibration_drift import (
    CalibrationDriftArtifact,
    DriftAxisRecord,
    DriftPairRecord,
)

_STYLE = """
:root{--bg:#fff;--fg:#1b1f24;--muted:#5b6570;--card:#f6f8fa;--line:#d0d7de;
--stable:#1a7f37;--drift:#cf222e;--inconclusive:#9a6700;--dot:#0969da;--band:#0969da22}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d1117;--fg:#e6edf3;
--muted:#8b949e;--card:#161b22;--line:#30363d;--stable:#3fb950;--drift:#ff7b72;
--inconclusive:#d29922;--dot:#58a6ff;--band:#58a6ff33}}
:root[data-theme="dark"]{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--card:#161b22;--line:#30363d;
--stable:#3fb950;--drift:#ff7b72;--inconclusive:#d29922;--dot:#58a6ff;--band:#58a6ff33}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;
max-width:960px;padding:24px 16px}
h1{font-size:1.4rem;margin:0 0 4px}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;
margin:16px 0}
.badge{font-weight:600}.stable{color:var(--stable)}.drift{color:var(--drift)}
.inconclusive{color:var(--inconclusive)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
td,th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:right}
td:first-child,th:first-child{text-align:left}
svg{max-width:100%;height:auto}
"""


def _axis_svg(axis: DriftAxisRecord, bag_names: list[str]) -> str:
    used = [o for o in axis.observations if o.used and o.value is not None and o.std is not None]
    if len(used) < 2:
        return ""
    scale = 1.0 if axis.unit == "deg" else 100.0
    unit = "deg" if axis.unit == "deg" else "cm"
    lows = [(o.value or 0.0) * scale - (o.std or 0.0) * scale for o in used]
    highs = [(o.value or 0.0) * scale + (o.std or 0.0) * scale for o in used]
    mean = (axis.weighted_mean or 0.0) * scale
    half = (axis.minimum_detectable_change or axis.floor) * scale / 2.0
    lo = min(min(lows), mean - half)
    hi = max(max(highs), mean + half)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    width, row = 560.0, 22.0
    height = row * len(bag_names) + 24.0

    def x(value: float) -> float:
        return 110.0 + (value - lo) / (hi - lo) * (width - 130.0)

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{html.escape(axis.name)} estimate per bag">',
        f'<rect x="{x(mean - half):.1f}" y="2" width="{x(mean + half) - x(mean - half):.1f}" '
        f'height="{row * len(bag_names):.0f}" fill="var(--band)"/>',
    ]
    for i, name in enumerate(bag_names):
        y = 2 + row * i + row / 2
        parts.append(
            f'<text x="104" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="currentColor">{html.escape(name[:16])}</text>'
        )
        obs = next((o for o in used if o.bag == name), None)
        if obs is None:
            continue
        v, s = (obs.value or 0.0) * scale, (obs.std or 0.0) * scale
        parts.append(
            f'<line x1="{x(v - s):.1f}" x2="{x(v + s):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="var(--dot)" stroke-width="2"/>'
            f'<circle cx="{x(v):.1f}" cy="{y:.1f}" r="4" fill="var(--dot)"/>'
        )
    parts.append(
        f'<text x="110" y="{height - 4:.0f}" font-size="10" fill="currentColor">{lo:.3g}</text>'
        f'<text x="{width - 20:.0f}" y="{height - 4:.0f}" text-anchor="end" font-size="10" '
        f'fill="currentColor">{hi:.3g} {unit}; band = minimum detectable change around the '
        "weighted mean</text></svg>"
    )
    return "".join(parts)


def _pair_card(pair: DriftPairRecord, bag_names: list[str]) -> str:
    rows = []
    charts = []
    for axis in pair.axes:
        scale = 1.0 if axis.unit == "deg" else 100.0
        unit = "deg" if axis.unit == "deg" else "cm"
        diff = "-" if axis.max_abs_difference is None else f"{axis.max_abs_difference * scale:.3f}"
        floor = (
            "-"
            if axis.minimum_detectable_change is None
            else f"{axis.minimum_detectable_change * scale:.3f}"
        )
        p = "-" if axis.p_value is None else f"{axis.p_value:.2g}"
        rows.append(
            f"<tr><td>{axis.name} ({unit})</td><td>{axis.bags_observed}</td><td>{diff}</td>"
            f"<td>{floor}</td><td>{p}</td>"
            f'<td class="badge {axis.status}">{axis.status}</td></tr>'
        )
        charts.append(_axis_svg(axis, bag_names))
    detail = ""
    if pair.verdict == "drift":
        who = ", ".join(pair.deviating_bags) or "cannot be attributed to one bag"
        detail = f"<p>Deviating: {html.escape(who)}</p>"
        for change in pair.changes:
            angle = (
                f", rotation {change.rotation_delta_deg:.2f} deg"
                if change.rotation_delta_deg is not None
                else ""
            )
            detail += f"<p class='muted'>{html.escape(change.bag)}{angle}</p>"
    frames = (
        f" <span class='muted'>T_{html.escape(pair.parent_frame or '')}_"
        f"{html.escape(pair.child_frame or '')}</span>"
        if pair.parent_frame
        else ""
    )
    return (
        f'<div class="card"><h2>{html.escape(pair.pair)}: '
        f'<span class="badge {pair.verdict}">{pair.verdict}</span>{frames}</h2>{detail}'
        "<table><tr><th>axis</th><th>bags</th><th>max |diff|</th><th>min detectable</th>"
        "<th>chi2 p</th><th>status</th></tr>"
        + "".join(rows)
        + "</table>"
        + "".join(charts)
        + "</div>"
    )


def render_drift_html(artifact: CalibrationDriftArtifact) -> str:
    """The report as one HTML string."""

    names = [b.name for b in artifact.bags]
    steps = "".join(f"<li>{html.escape(s)}</li>" for s in artifact.next_steps)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Calibration drift</title><style>{_STYLE}</style></head><body>"
        f"<h1>Calibration drift: <span class='badge {artifact.overall_verdict}'>"
        f"{artifact.overall_verdict}</span></h1>"
        f"<p class='muted'>{len(names)} bags: {html.escape(', '.join(names))}. "
        f"A change is flagged beyond max({artifact.thresholds.sigma_k:g} sigma, floor) with "
        f"chi-square p &lt; {artifact.thresholds.chi2_alpha:g}. "
        f"{html.escape(artifact.schema_version)}, calibrex "
        f"{html.escape(artifact.provenance.generator_version)}.</p>"
        + "".join(_pair_card(p, names) for p in artifact.pairs)
        + (f"<h3>Next steps</h3><ul>{steps}</ul>" if steps else "")
        + "</body></html>"
    )


def write_drift_html(artifact: CalibrationDriftArtifact, path: Path) -> None:
    """Write the report to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_drift_html(artifact), encoding="utf-8")
