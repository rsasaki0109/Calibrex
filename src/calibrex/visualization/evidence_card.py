"""Portable SVG evidence cards for README and release surfaces."""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

from calibrex.core.result import CalibrationResult, MetricResult

_GRADE_COLOR = {
    "pass": "#22c55e",
    "warn": "#f59e0b",
    "fail": "#ef4444",
}

_METRIC_LABELS = {
    "lidar_pair_holdout_point_to_plane_rmse_m": "LiDAR holdout RMSE",
    "lidar_world_map_point_to_plane_rmse_m": "World-map holdout RMSE",
    "lidar_point_to_plane_rmse_m": "LiDAR holdout RMSE",
    "reprojection_rmse_px": "Reprojection holdout RMSE",
    "lidar_camera_edge_alignment_score": "Camera-LiDAR edge alignment",
    "lidar_pair_known_bad_detectable_fraction": "Known-bad rejection",
    "lidar_world_map_perturbation_detectable_fraction": "Perturbation detection",
    "radar_lidar_velocity_consistency": "Radar velocity consistency",
    "trajectory_cross_segment_translation_rmse_m": "Trajectory drift proxy",
}

_METRIC_PRIORITY = tuple(_METRIC_LABELS)


def render_evidence_card(
    result: CalibrationResult,
    *,
    source_label: str = "result.yaml",
    source_sha256: str | None = None,
) -> str:
    """Render a deterministic, self-contained SVG summary for a validated result."""

    grade = result.quality.grade
    status = grade.upper()
    status_color = _GRADE_COLOR[grade]
    metric_rows = _select_metric_rows(result)
    frame_summary = _frame_summary(result)
    metadata = _provenance_metadata(result, source_label, source_sha256)
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    rows = "\n".join(
        _metric_row_svg(index, label, value, metric_grade)
        for index, (label, value, metric_grade) in enumerate(metric_rows)
    )
    source_digest = f"sha256:{source_sha256[:12]}" if source_sha256 else "sha256:unavailable"
    metrics_origin = str(result.run.provenance.get("metrics_origin", "declared"))
    data_verified = result.run.provenance.get("data_verified")
    verification = "verified raw data" if data_verified is True else "validation recorded"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630"
  viewBox="0 0 1200 630" role="img" aria-labelledby="title description">
  <title id="title">Calibrex evidence card for {escape(result.run.id)}</title>
  <desc id="description">Calibration run status {status}, with schema-validated metrics
    and provenance.</desc>
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
      <feDropShadow dx="0" dy="14" stdDeviation="22" flood-color="#020617" flood-opacity="0.5"/>
    </filter>
    <style>
      .sans {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif; }}
      .caps {{ font-size: 13px; font-weight: 700; letter-spacing: 2px; }}
      .muted {{ fill: #91a4c2; }}
      .bright {{ fill: #f8fafc; }}
    </style>
  </defs>
  <rect width="1200" height="630" rx="28" fill="url(#background)"/>
  <circle cx="1090" cy="55" r="180" fill="#38bdf8" opacity="0.055"/>
  <circle cx="80" cy="610" r="180" fill="#818cf8" opacity="0.045"/>
  <rect x="48" y="42" width="1104" height="516" rx="24" fill="#0f1b31"
    stroke="#263957" filter="url(#shadow)"/>

  <g class="sans">
    <g transform="translate(76 70)">
      <rect width="46" height="46" rx="13" fill="url(#accent)"/>
      <path d="M13 23c0-7 4-11 11-11h9v7h-8c-3 0-5 2-5 5s2 5 5 5h8v7h-9
        c-7 0-11-5-11-13z" fill="#07111f"/>
    </g>
    <text x="138" y="91" class="bright" font-size="27" font-weight="780">Calibrex</text>
    <text x="138" y="115" class="muted" font-size="14">Calibration evidence,
      not just a matrix.</text>
    <rect x="964" y="70" width="148" height="44" rx="22" fill="{status_color}" opacity="0.14"/>
    <circle cx="990" cy="92" r="6" fill="{status_color}"/>
    <text x="1010" y="98" fill="{status_color}" font-size="16" font-weight="800">{status}</text>

    <line x1="76" y1="142" x2="1124" y2="142" stroke="#263957"/>

    <text x="78" y="179" class="caps muted">CALIBRATION RUN</text>
    <text x="78" y="218" class="bright" font-size="25"
      font-weight="730">{escape(result.run.id)}</text>
    <text x="78" y="248" class="muted" font-size="15">{escape(frame_summary)}</text>

    <g transform="translate(78 286)">
      <rect width="350" height="154" rx="18" fill="#0a1427" stroke="#263957"/>
      <text x="22" y="31" class="caps muted">EVIDENCE CONTRACT</text>
      {_contract_line_svg(60, "Schema", result.schema_version, "#38bdf8")}
      {_contract_line_svg(91, "Metrics", metrics_origin, "#818cf8")}
      {_contract_line_svg(122, "Data", verification, status_color)}
    </g>

    <text x="478" y="179" class="caps muted">EVIDENCE SNAPSHOT</text>
    {rows}

    <line x1="76" y1="477" x2="1124" y2="477" stroke="#263957"/>
    <text x="78" y="511" class="muted" font-size="13">Source
      {escape(source_label)}  ·  {source_digest}</text>
    <text x="78" y="535" class="muted" font-size="13">Run provenance  Calibrex
      {escape(result.run.slac_version)}  ·  {escape(metrics_origin)}</text>
    <text x="1122" y="526" text-anchor="end" fill="#38bdf8" font-size="13"
      font-weight="700">SCHEMA VALIDATED</text>
  </g>
</svg>
"""


def write_evidence_card(
    result: CalibrationResult,
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
    source_label: str | None = None,
) -> Path:
    """Write an SVG evidence card and bind it to its source result digest."""

    output = Path(output_path)
    source = Path(source_path) if source_path is not None else None
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest() if source else None
    label = source_label or (source.name if source else "result.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_evidence_card(
            result,
            source_label=label,
            source_sha256=source_sha256,
        ),
        encoding="utf-8",
    )
    return output


def _select_metric_rows(result: CalibrationResult) -> list[tuple[str, str, str]]:
    ordered_names = [name for name in _METRIC_PRIORITY if name in result.metrics]
    ordered_names.extend(name for name in sorted(result.metrics) if name not in ordered_names)
    rows: list[tuple[str, str, str]] = []
    for name in ordered_names[:4]:
        metric = result.metrics[name]
        rows.append((_metric_label(name), _metric_value(metric), metric.grade))
    if len(rows) < 4:
        rows.append(
            ("Observability rank", _observability_value(result), result.observability.grade)
        )
    if len(rows) < 4:
        rows.append(("Degeneracy check", "No blocking signal", result.degeneracy.grade))
    if len(rows) < 4:
        rows.append(("Result contract", result.schema_version, result.quality.grade))
    return rows[:4]


def _metric_label(name: str) -> str:
    if name in _METRIC_LABELS:
        return _METRIC_LABELS[name]
    return name.replace("_", " ").title()


def _metric_value(metric: MetricResult) -> str:
    value = metric.holdout
    if value is None:
        value = metric.value
    if value is None:
        value = metric.train
    if value is None:
        return metric.reason or "Recorded"
    suffix = f" {metric.unit}" if metric.unit else ""
    return f"{value:.4g}{suffix}"


def _observability_value(result: CalibrationResult) -> str:
    rank = result.observability.rank
    if rank is None:
        return "Not reported"
    weak_count = len(result.observability.weak_directions)
    return f"{rank} rank · {weak_count} weak directions"


def _frame_summary(result: CalibrationResult) -> str:
    children = sorted(name for name in result.frame_graph.frames if name != result.frame_graph.root)
    if not children:
        return result.frame_graph.root
    displayed = children[:3]
    suffix = f" +{len(children) - 3}" if len(children) > 3 else ""
    return f"{result.frame_graph.root} → {' · '.join(displayed)}{suffix}"


def _provenance_metadata(
    result: CalibrationResult,
    source_label: str,
    source_sha256: str | None,
) -> dict[str, Any]:
    transforms = (
        list(result.transforms.values())
        + list(result.candidate_extrinsics.values())
        + list(result.reference_extrinsics.values())
    )
    producers = sorted({transform.provenance.producer for transform in transforms})
    run_fields = {
        key: result.run.provenance[key]
        for key in ("pipeline", "solver", "metrics_origin", "data_verified")
        if key in result.run.provenance
    }
    return {
        "artifact": "calibrex.evidence-card/v0.1",
        "result_schema": result.schema_version,
        "run_id": result.run.id,
        "calibrex_version": result.run.slac_version,
        "git_commit": result.run.git_commit,
        "config_sha256": result.run.config_sha256,
        "dataset_sha256": result.run.dataset_sha256,
        "source": {"label": source_label, "sha256": source_sha256},
        "estimate_producers": producers,
        "run_provenance": run_fields,
    }


def _metric_row_svg(index: int, label: str, value: str, grade: str) -> str:
    y = 206 + index * 62
    color = _GRADE_COLOR[grade]
    status = grade.upper()
    return f"""<g transform="translate(478 {y})">
      <rect width="646" height="50" rx="13" fill="#0a1427" stroke="#223552"/>
      <rect width="5" height="50" rx="2.5" fill="{color}"/>
      <text x="22" y="22" class="muted" font-size="13">{escape(label)}</text>
      <text x="22" y="41" class="bright" font-size="16" font-weight="700">{escape(value)}</text>
      <rect x="546" y="12" width="82" height="26" rx="13" fill="{color}" opacity="0.14"/>
      <text x="587" y="30" text-anchor="middle" fill="{color}" font-size="11"
        font-weight="800">{status}</text>
    </g>"""


def _contract_line_svg(y: int, label: str, value: str, color: str) -> str:
    return f"""<circle cx="24" cy="{y - 5}" r="4" fill="{color}"/>
      <text x="40" y="{y}" class="muted" font-size="13">{escape(label)}</text>
      <text x="328" y="{y}" text-anchor="end" class="bright" font-size="13"
        font-weight="650">{escape(value)}</text>"""
