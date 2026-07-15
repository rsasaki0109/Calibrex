"""SVG comparison tables derived from validated Calibrex results."""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

from calibrex.core.result import CalibrationResult, TransformResult
from calibrex.evaluation.compare import MetricComparison, ResultComparison, compare_results

_GRADE_COLOR = {"pass": "#22c55e", "warn": "#f59e0b", "fail": "#ef4444"}
_METRIC_LABELS = {
    "lidar_point_to_plane_rmse_m": "LiDAR holdout RMSE",
    "reprojection_rmse_px": "Reprojection holdout RMSE",
    "lidar_pair_holdout_point_to_plane_rmse_m": "LiDAR pair holdout RMSE",
    "lidar_pair_known_bad_detectable_fraction": "Known-bad rejection",
}


def render_comparison_table(
    left: CalibrationResult,
    right: CalibrationResult,
    *,
    left_label: str = "baseline",
    right_label: str = "candidate",
    left_source: dict[str, str | None] | None = None,
    right_source: dict[str, str | None] | None = None,
) -> str:
    """Render a deterministic SVG table from two validated results."""

    comparison = compare_results(left, right)
    left_color = _GRADE_COLOR[left.quality.grade]
    right_color = _GRADE_COLOR[right.quality.grade]
    protocol = comparison.protocol_compatibility.status.replace("_", " ").upper()
    protocol_color = {
        "compatible": "#22c55e",
        "warning": "#f59e0b",
        "not_comparable": "#f59e0b",
    }[comparison.protocol_compatibility.status]
    metrics = list(comparison.metrics.values())[:4]
    row_svgs = [_metric_row(index, metric) for index, metric in enumerate(metrics)]
    if len(row_svgs) < 4:
        row_svgs.append(
            _summary_row(
                len(row_svgs),
                "Overall verdict",
                left.quality.grade.upper(),
                right.quality.grade.upper(),
                "KNOWN-BAD REJECTED" if right.quality.grade == "fail" else "REVIEW",
                right_color,
            )
        )
    if len(row_svgs) < 4:
        left_rank = _rank_text(left)
        right_rank = _rank_text(right)
        row_svgs.append(
            _summary_row(
                len(row_svgs),
                "Observability rank",
                left_rank,
                right_rank,
                "UNCHANGED" if left_rank == right_rank else "CHANGED",
                "#91a4c2",
            )
        )
    rows = "\n".join(row_svgs)
    metadata = _comparison_metadata(
        left,
        right,
        comparison,
        left_label,
        right_label,
        left_source or {"label": "left-result.yaml", "sha256": None},
        right_source or {"label": "right-result.yaml", "sha256": None},
    )
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    left_digest = _short_digest(metadata["sources"]["left"]["sha256"])
    right_digest = _short_digest(metadata["sources"]["right"]["sha256"])
    transform_delta = _transform_delta_summary(comparison)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630"
  viewBox="0 0 1200 630" role="img" aria-labelledby="title description">
  <title id="title">Calibrex evidence comparison</title>
  <desc id="description">Schema-valid comparison of {escape(left_label)} and
    {escape(right_label)}, including metrics, transform delta, and provenance.</desc>
  <metadata id="calibrex-provenance">{escape(metadata_json)}</metadata>
  <defs>
    <linearGradient id="background" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#07111f"/>
      <stop offset="0.55" stop-color="#0b1730"/>
      <stop offset="1" stop-color="#111a35"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#38bdf8"/>
      <stop offset="1" stop-color="#818cf8"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="14" stdDeviation="22" flood-color="#020617"
        flood-opacity="0.5"/>
    </filter>
    <style>
      .sans {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif; }}
      .caps {{ font-size: 12px; font-weight: 750; letter-spacing: 1.7px; }}
      .muted {{ fill: #91a4c2; }}
      .bright {{ fill: #f8fafc; }}
    </style>
  </defs>
  <rect width="1200" height="630" rx="28" fill="url(#background)"/>
  <circle cx="1110" cy="35" r="190" fill="#38bdf8" opacity="0.055"/>
  <circle cx="40" cy="620" r="180" fill="#818cf8" opacity="0.045"/>
  <rect x="48" y="42" width="1104" height="516" rx="24" fill="#0f1b31"
    stroke="#263957" filter="url(#shadow)"/>
  <g class="sans">
    <g transform="translate(76 70)">
      <rect width="46" height="46" rx="13" fill="url(#accent)"/>
      <path d="M13 23c0-7 4-11 11-11h9v7h-8c-3 0-5 2-5 5s2 5 5 5h8v7h-9
        c-7 0-11-5-11-13z" fill="#07111f"/>
    </g>
    <text x="138" y="91" class="bright" font-size="27" font-weight="780">Evidence comparison</text>
    <text x="138" y="115" class="muted" font-size="14">Same metrics. Same contract.
      Visible provenance.</text>
    <rect x="902" y="72" width="210" height="40" rx="20" fill="{protocol_color}"
      opacity="0.14"/>
    <text x="1007" y="97" text-anchor="middle" fill="{protocol_color}"
      font-size="12" font-weight="800">PROTOCOL · {protocol}</text>

    <line x1="76" y1="142" x2="1124" y2="142" stroke="#263957"/>
    <text x="78" y="178" class="caps muted">METRIC</text>
    <text x="648" y="178" text-anchor="end" class="caps"
      fill="{left_color}">{escape(left_label.upper())}</text>
    <text x="850" y="178" text-anchor="end" class="caps"
      fill="{right_color}">{escape(right_label.upper())}</text>
    <text x="1080" y="178" text-anchor="end" class="caps muted">COMPARISON</text>
    {rows}

    <g transform="translate(78 410)">
      <rect width="1044" height="62" rx="15" fill="#0a1427" stroke="#263957"/>
      <text x="20" y="25" class="caps muted">TRANSFORM DELTA</text>
      <text x="20" y="47" class="bright" font-size="15"
        font-weight="700">{escape(transform_delta)}</text>
      <text x="1022" y="37" text-anchor="end" class="muted"
        font-size="13">{comparison.summary.transform_comparison_count} shared transforms</text>
    </g>

    <line x1="76" y1="492" x2="1124" y2="492" stroke="#263957"/>
    <text x="78" y="521" class="muted" font-size="12">{escape(left_label)} · {left_digest}</text>
    <text x="78" y="541" class="muted" font-size="12">{escape(right_label)} · {right_digest}</text>
    <text x="1122" y="533" text-anchor="end" fill="#38bdf8" font-size="13"
      font-weight="700">SCHEMA-VALID SOURCES</text>
  </g>
</svg>
"""


def write_comparison_table(
    left: CalibrationResult,
    right: CalibrationResult,
    output_path: str | Path,
    *,
    left_path: str | Path,
    right_path: str | Path,
    left_label: str = "baseline",
    right_label: str = "candidate",
) -> Path:
    """Write an SVG comparison table bound to both source result digests."""

    output = Path(output_path)
    left_source_path = Path(left_path)
    right_source_path = Path(right_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_comparison_table(
            left,
            right,
            left_label=left_label,
            right_label=right_label,
            left_source=_source_metadata(left_source_path),
            right_source=_source_metadata(right_source_path),
        ),
        encoding="utf-8",
    )
    return output


def _metric_row(index: int, metric: MetricComparison) -> str:
    y = 194 + index * 51
    left_value = _metric_value(metric.left_preferred, metric.unit)
    right_value = _metric_value(metric.right_preferred, metric.unit)
    winner = {
        "left": "BASELINE BETTER",
        "right": "CANDIDATE BETTER",
        "tie": "TIE",
        "not_comparable": "NOT COMPARABLE",
    }[metric.winner]
    winner_color = {
        "left": "#22c55e",
        "right": "#38bdf8",
        "tie": "#91a4c2",
        "not_comparable": "#f59e0b",
    }[metric.winner]
    label = _METRIC_LABELS.get(metric.name, metric.name.replace("_", " ").title())
    return f"""<g transform="translate(78 {y})">
      <rect width="1044" height="42" rx="11" fill="#0a1427" stroke="#223552"/>
      <text x="18" y="27" class="bright" font-size="14">{escape(label)}</text>
      <text x="570" y="27" text-anchor="end" class="bright" font-size="15"
        font-weight="750">{escape(left_value)}</text>
      <text x="772" y="27" text-anchor="end" class="bright" font-size="15"
        font-weight="750">{escape(right_value)}</text>
      <rect x="830" y="8" width="196" height="26" rx="13" fill="{winner_color}"
        opacity="0.14"/>
      <text x="928" y="26" text-anchor="middle" fill="{winner_color}" font-size="11"
        font-weight="800">{winner}</text>
    </g>"""


def _metric_value(value: float | None, unit: str | None) -> str:
    if value is None:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{value:.4g}{suffix}"


def _summary_row(
    index: int,
    label: str,
    left_value: str,
    right_value: str,
    comparison: str,
    color: str,
) -> str:
    y = 194 + index * 51
    return f"""<g transform="translate(78 {y})">
      <rect width="1044" height="42" rx="11" fill="#0a1427" stroke="#223552"/>
      <text x="18" y="27" class="bright" font-size="14">{escape(label)}</text>
      <text x="570" y="27" text-anchor="end" class="bright" font-size="15"
        font-weight="750">{escape(left_value)}</text>
      <text x="772" y="27" text-anchor="end" class="bright" font-size="15"
        font-weight="750">{escape(right_value)}</text>
      <rect x="830" y="8" width="196" height="26" rx="13" fill="{color}"
        opacity="0.14"/>
      <text x="928" y="26" text-anchor="middle" fill="{color}" font-size="11"
        font-weight="800">{escape(comparison)}</text>
    </g>"""


def _rank_text(result: CalibrationResult) -> str:
    rank = result.observability.rank
    return "n/a" if rank is None else str(rank)


def _transform_delta_summary(comparison: ResultComparison) -> str:
    translation = comparison.summary.max_translation_delta_m
    rotation = comparison.summary.max_rotation_delta_deg
    if translation is None and rotation is None:
        return "No shared transform delta"
    translation_text = "n/a" if translation is None else f"{translation:.4g} m"
    rotation_text = "n/a" if rotation is None else f"{rotation:.4g}°"
    return f"max translation {translation_text}  ·  max rotation {rotation_text}"


def _comparison_metadata(
    left: CalibrationResult,
    right: CalibrationResult,
    comparison: ResultComparison,
    left_label: str,
    right_label: str,
    left_source: dict[str, str | None],
    right_source: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "artifact": "calibrex.evidence-comparison-table/v0.1",
        "comparison_schema": comparison.schema_version,
        "protocol_compatibility": comparison.protocol_compatibility.status,
        "sources": {
            "left": {**left_source, "label": left_label, "run_id": left.run.id},
            "right": {**right_source, "label": right_label, "run_id": right.run.id},
        },
        "estimate_producers": {
            "left": _producers(left),
            "right": _producers(right),
        },
    }


def _producers(result: CalibrationResult) -> list[str]:
    transforms: list[TransformResult] = (
        list(result.transforms.values())
        + list(result.candidate_extrinsics.values())
        + list(result.reference_extrinsics.values())
    )
    return sorted({transform.provenance.producer for transform in transforms})


def _source_metadata(path: Path) -> dict[str, str | None]:
    return {
        "label": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _short_digest(value: object) -> str:
    if not isinstance(value, str):
        return "sha256:unavailable"
    return f"sha256:{value[:12]}"
