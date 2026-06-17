"""HTML report rendering."""

from __future__ import annotations

import math
from collections.abc import Iterable
from html import escape
from pathlib import Path
from typing import Any

from calibrex.core.geometry import normalize_quaternion_xyzw
from calibrex.core.io import write_mapping
from calibrex.core.result import CalibrationResult, MetricResult, TransformResult

_WORLD_MAP_SUMMARY_METRICS = (
    "lidar_world_map_point_to_plane_rmse_m",
    "lidar_world_map_point_to_plane_median_holdout_m",
    "lidar_world_map_point_to_plane_p95_holdout_m",
    "lidar_world_map_perturbation_detectable_fraction",
    "lidar_world_map_weak_dof_count",
    "lidar_world_map_min_dof_sensitivity_m",
)

_WORLD_MAP_DOF_METRICS = (
    ("roll_lidar0", "lidar_world_map_sensitivity_roll_m"),
    ("pitch_lidar0", "lidar_world_map_sensitivity_pitch_m"),
    ("yaw_lidar0", "lidar_world_map_sensitivity_yaw_m"),
    ("x_lidar0", "lidar_world_map_sensitivity_x_m"),
    ("y_lidar0", "lidar_world_map_sensitivity_y_m"),
    ("z_lidar0", "lidar_world_map_sensitivity_z_m"),
)


def render_html_report(result: CalibrationResult) -> str:
    """Render a portable HTML calibration report."""

    metric_rows = "\n".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{escape(metric.grade.upper())}</td>"
        f"<td>{_fmt(metric.train)}</td>"
        f"<td>{_fmt(metric.holdout)}</td>"
        f"<td>{_fmt(metric.value)}</td>"
        f"<td>{escape(metric.unit or '')}</td>"
        f"<td>{escape(metric.reason or '')}</td>"
        "</tr>"
        for name, metric in sorted(result.metrics.items())
    )
    transform_rows = "\n".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{escape(transform.parent)}</td>"
        f"<td>{escape(transform.child)}</td>"
        f"<td>{escape(str(transform.translation_m))}</td>"
        f"<td>{escape(str(transform.rotation_quat_xyzw))}</td>"
        f"<td>{escape(transform.quality.grade.upper())}</td>"
        "</tr>"
        for name, transform in sorted(result.transforms.items())
    )
    comparison_rows = _candidate_reference_rows(result)
    candidate_rows = _transform_rows(result.candidate_extrinsics)
    reference_rows = _transform_rows(result.reference_extrinsics)
    warnings = "".join(f"<li>{escape(item)}</li>" for item in result.quality.warnings)
    failures = "".join(f"<li>{escape(item)}</li>" for item in result.quality.blocking_failures)
    recommendations = "".join(f"<li>{escape(item)}</li>" for item in result.quality.recommendation)
    artifact_rows = _artifact_rows(result)
    observability_rows = _observability_rows(result)
    lidar_world_map_section = _lidar_world_map_section(result)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Calibrex Report - {escape(result.run.id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #182026; }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    .grade {{
      display: inline-block;
      padding: 0.25rem 0.5rem;
      border-radius: 4px;
      font-weight: 700;
    }}
    .pass {{ background: #d9f0e3; color: #125c35; }}
    .warn {{ background: #fff0c2; color: #6a4a00; }}
    .fail {{ background: #ffd9d9; color: #7a1616; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d6dce1; padding: 0.5rem; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f4; }}
    .metric-warn {{ background: #fffaf0; }}
    .metric-fail {{ background: #fff1f1; }}
    code {{ background: #eef2f4; padding: 0.1rem 0.25rem; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>Calibrex Calibration Report</h1>
  <p>Run <code>{escape(result.run.id)}</code> finished with
    <span class="grade {escape(result.quality.grade)}">
      {escape(result.quality.grade.upper())}
    </span>.
  </p>
  <h2>Provenance</h2>
  <table>
    <tr><th>Calibrex Version</th><td>{escape(result.run.calibrex_version)}</td></tr>
    <tr><th>Domain</th><td>{escape(result.run.domain)}</td></tr>
    <tr><th>Git Commit</th><td>{escape(result.run.git_commit or 'unknown')}</td></tr>
    <tr><th>Config SHA256</th><td>{escape(result.run.config_sha256 or 'unknown')}</td></tr>
    <tr><th>Dataset SHA256</th><td>{escape(result.run.dataset_sha256 or 'unknown')}</td></tr>
  </table>
  <h2>Quality</h2>
  <h3>Failures</h3>
  <ul>{failures or '<li>None</li>'}</ul>
  <h3>Warnings</h3>
  <ul>{warnings or '<li>None</li>'}</ul>
  <h3>Recommendations</h3>
  <ul>{recommendations or '<li>None</li>'}</ul>
  <h2>Observability</h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    {observability_rows}
  </table>
  {lidar_world_map_section}
  <h2>Artifacts</h2>
  <table>
    <tr><th>Name</th><th>Path</th></tr>
    {artifact_rows}
  </table>
  <h2>Metrics</h2>
  <table>
    <tr><th>Name</th><th>Grade</th><th>Train</th><th>Holdout</th><th>Value</th><th>Unit</th><th>Reason</th></tr>
    {metric_rows}
  </table>
  <h2>Transforms</h2>
  <table>
    <tr>
      <th>Name</th><th>Parent</th><th>Child</th>
      <th>Translation m</th><th>Quaternion xyzw</th><th>Grade</th>
    </tr>
    {transform_rows}
  </table>
  <h2>Candidate vs Reference Extrinsics</h2>
  <table>
    <tr>
      <th>Candidate</th><th>Reference</th><th>Edge</th>
      <th>Translation Delta m</th><th>Rotation Delta deg</th><th>Grade</th>
    </tr>
    {comparison_rows}
  </table>
  <h2>Candidate Extrinsics</h2>
  <table>
    <tr>
      <th>Name</th><th>Parent</th><th>Child</th>
      <th>Translation m</th><th>Quaternion xyzw</th><th>Grade</th>
    </tr>
    {candidate_rows}
  </table>
  <h2>Reference Extrinsics</h2>
  <table>
    <tr>
      <th>Name</th><th>Parent</th><th>Child</th>
      <th>Translation m</th><th>Quaternion xyzw</th><th>Grade</th>
    </tr>
    {reference_rows}
  </table>
</main>
</body>
</html>
"""


def write_report_artifacts(
    result: CalibrationResult,
    output_dir: str | Path,
    *,
    html_filename: str | Path = "report.html",
    include_html: bool = True,
) -> dict[str, str]:
    """Write the HTML report and machine-readable report sidecars."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    if include_html:
        report_path = _resolve_output_path(output_path, html_filename)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        result.artifacts.html_report = str(report_path)
        report_path.write_text(render_html_report(result), encoding="utf-8")
        written["html_report"] = str(report_path)

    for name, payload in _report_sidecar_payloads(result).items():
        sidecar_path = output_path / name
        write_mapping(sidecar_path, payload)
        written[name.removesuffix(".json")] = str(sidecar_path)

    return written


def _resolve_output_path(output_dir: Path, filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return output_dir / path


def _report_sidecar_payloads(result: CalibrationResult) -> dict[str, dict[str, Any]]:
    return {
        "summary.json": _summary_payload(result),
        "metrics.json": _metrics_payload(result),
        "observability.json": _observability_payload(result),
        "degeneracy.json": _degeneracy_payload(result),
    }


def _summary_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "schema_version": "calibrex.report.summary/v0.1",
        "run": _run_payload(result),
        "quality": result.quality.model_dump(mode="json", exclude_none=True),
        "metric_counts": _grade_counts(metric.grade for metric in result.metrics.values()),
        "transform_counts": _grade_counts(
            transform.quality.grade for transform in result.transforms.values()
        ),
        "candidate_extrinsic_count": len(result.candidate_extrinsics),
        "reference_extrinsic_count": len(result.reference_extrinsics),
        "matched_candidate_reference_count": _matched_candidate_reference_count(result),
        "weak_direction_count": len(result.observability.weak_directions),
        "artifact_paths": result.artifacts.model_dump(mode="json", exclude_none=True),
    }


def _metrics_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "schema_version": "calibrex.report.metrics/v0.1",
        "run": _run_payload(result),
        "metrics": {
            name: metric.model_dump(mode="json", exclude_none=True)
            for name, metric in sorted(result.metrics.items())
        },
    }


def _observability_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "schema_version": "calibrex.report.observability/v0.1",
        "run": _run_payload(result),
        "observability": result.observability.model_dump(mode="json", exclude_none=True),
        "weak_directions": list(result.observability.weak_directions),
        "metrics": {
            name: metric.model_dump(mode="json", exclude_none=True)
            for name, metric in sorted(result.metrics.items())
            if "observability" in name or "sensitivity" in name or "weak_dof" in name
        },
    }


def _degeneracy_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "schema_version": "calibrex.report.degeneracy/v0.1",
        "run": _run_payload(result),
        "degeneracy": result.degeneracy.model_dump(mode="json", exclude_none=True),
        "quality_warnings": list(result.quality.warnings),
        "recommendations": list(result.quality.recommendation),
        "metrics": {
            name: metric.model_dump(mode="json", exclude_none=True)
            for name, metric in sorted(result.metrics.items())
            if "degeneracy" in name
            or "excitation" in name
            or "timestamp" in name
            or "weak_dof" in name
        },
    }


def _run_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "id": result.run.id,
        "status": result.run.status,
        "domain": result.run.domain,
        "calibrex_version": result.run.calibrex_version,
        "git_commit": result.run.git_commit,
        "created_at": result.run.created_at,
        "dataset_type": result.run.provenance.get("dataset_type"),
        "dataset_path": result.run.provenance.get("dataset_path"),
    }


def _grade_counts(grades: Iterable[str]) -> dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for grade in grades:
        if grade in counts:
            counts[grade] += 1
    return counts


def _matched_candidate_reference_count(result: CalibrationResult) -> int:
    reference_edges = {
        (transform.parent, transform.child)
        for transform in result.reference_extrinsics.values()
    }
    return sum(
        1
        for transform in result.candidate_extrinsics.values()
        if (transform.parent, transform.child) in reference_edges
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _transform_rows(transforms: dict[str, TransformResult]) -> str:
    if not transforms:
        return '<tr><td colspan="6">None</td></tr>'
    return "\n".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{escape(transform.parent)}</td>"
        f"<td>{escape(transform.child)}</td>"
        f"<td>{escape(str(transform.translation_m))}</td>"
        f"<td>{escape(str(transform.rotation_quat_xyzw))}</td>"
        f"<td>{escape(transform.quality.grade.upper())}</td>"
        "</tr>"
        for name, transform in sorted(transforms.items())
    )


def _candidate_reference_rows(result: CalibrationResult) -> str:
    if not result.candidate_extrinsics or not result.reference_extrinsics:
        return '<tr><td colspan="6">None</td></tr>'

    rows: list[str] = []
    references_by_edge = {
        (transform.parent, transform.child): (name, transform)
        for name, transform in sorted(result.reference_extrinsics.items())
    }
    for candidate_name, candidate in sorted(result.candidate_extrinsics.items()):
        reference_item = references_by_edge.get((candidate.parent, candidate.child))
        if reference_item is None:
            continue
        reference_name, reference = reference_item
        translation_delta_m = _translation_delta_m(candidate, reference)
        rotation_delta_deg = _rotation_delta_deg(candidate, reference)
        grade = (
            "pass"
            if translation_delta_m <= 0.05 and rotation_delta_deg <= 1.0
            else "warn"
        )
        rows.append(
            f'<tr class="{escape(_grade_css_class(grade))}">'
            f"<td>{escape(candidate_name)}</td>"
            f"<td>{escape(reference_name)}</td>"
            f"<td>{escape(candidate.parent)} -> {escape(candidate.child)}</td>"
            f"<td>{_fmt(translation_delta_m)}</td>"
            f"<td>{_fmt(rotation_delta_deg)}</td>"
            f"<td>{escape(grade.upper())}</td>"
            "</tr>"
        )
    if not rows:
        return (
            '<tr><td colspan="6">'
            "No candidate extrinsics matched reference parent/child pairs"
            "</td></tr>"
        )
    return "\n".join(rows)


def _translation_delta_m(left: TransformResult, right: TransformResult) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.translation_m,
                right.translation_m,
                strict=True,
            )
        )
    )


def _rotation_delta_deg(left: TransformResult, right: TransformResult) -> float:
    left_quat = normalize_quaternion_xyzw(left.rotation_quat_xyzw)
    right_quat = normalize_quaternion_xyzw(right.rotation_quat_xyzw)
    dot = abs(
        sum(
            left_value * right_value
            for left_value, right_value in zip(
                left_quat,
                right_quat,
                strict=True,
            )
        )
    )
    clamped = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(clamped))


def _observability_rows(result: CalibrationResult) -> str:
    weak_directions = (
        ", ".join(result.observability.weak_directions)
        if result.observability.weak_directions
        else "None"
    )
    rows = [
        ("Grade", result.observability.grade.upper()),
        ("Rank", _fmt_int(result.observability.rank)),
        ("Condition Number", _fmt(result.observability.condition_number)),
        ("Weak Directions", weak_directions),
        ("Degeneracy Grade", result.degeneracy.grade.upper()),
        ("Degeneracy Reason", result.degeneracy.reason or "None"),
    ]
    return "\n".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{escape(value)}</td>"
        "</tr>"
        for name, value in rows
    )


def _lidar_world_map_section(result: CalibrationResult) -> str:
    summary_rows = _selected_metric_rows(result, _WORLD_MAP_SUMMARY_METRICS)
    dof_rows = _world_map_dof_rows(result)
    if not summary_rows and not dof_rows:
        return ""
    return f"""
  <h2>LiDAR World-Map Diagnostics</h2>
  <p>
    OXTS-projected train/holdout point-to-plane metrics and known extrinsic
    perturbation sensitivity for fixed-vehicle LiDAR validation.
  </p>
  <h3>Summary</h3>
  <table>
    <tr><th>Metric</th><th>Grade</th><th>Train</th><th>Holdout</th><th>Value</th><th>Unit</th><th>Reason</th></tr>
    {summary_rows or '<tr><td colspan="7">None</td></tr>'}
  </table>
  <h3>DoF Sensitivity</h3>
  <table>
    <tr><th>DoF</th><th>Metric</th><th>Grade</th><th>Sensitivity</th><th>Unit</th><th>Reason</th></tr>
    {dof_rows or '<tr><td colspan="6">None</td></tr>'}
  </table>
"""


def _selected_metric_rows(result: CalibrationResult, metric_names: tuple[str, ...]) -> str:
    rows: list[str] = []
    for name in metric_names:
        metric = result.metrics.get(name)
        if metric is None:
            continue
        rows.append(_metric_row(name, metric))
    return "\n".join(rows)


def _world_map_dof_rows(result: CalibrationResult) -> str:
    rows: list[str] = []
    for direction, metric_name in _WORLD_MAP_DOF_METRICS:
        metric = result.metrics.get(metric_name)
        if metric is None:
            continue
        css_class = _metric_css_class(metric)
        rows.append(
            f'<tr class="{css_class}">'
            f"<td>{escape(direction)}</td>"
            f"<td>{escape(metric_name)}</td>"
            f"<td>{escape(metric.grade.upper())}</td>"
            f"<td>{_fmt(metric.value)}</td>"
            f"<td>{escape(metric.unit or '')}</td>"
            f"<td>{escape(metric.reason or '')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _metric_row(name: str, metric: MetricResult) -> str:
    css_class = _metric_css_class(metric)
    return (
        f'<tr class="{css_class}">'
        f"<td>{escape(name)}</td>"
        f"<td>{escape(metric.grade.upper())}</td>"
        f"<td>{_fmt(metric.train)}</td>"
        f"<td>{_fmt(metric.holdout)}</td>"
        f"<td>{_fmt(metric.value)}</td>"
        f"<td>{escape(metric.unit or '')}</td>"
        f"<td>{escape(metric.reason or '')}</td>"
        "</tr>"
    )


def _metric_css_class(metric: MetricResult) -> str:
    return _grade_css_class(metric.grade)


def _grade_css_class(grade: str) -> str:
    if grade == "fail":
        return "metric-fail"
    if grade == "warn":
        return "metric-warn"
    return ""


def _fmt_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _artifact_rows(result: CalibrationResult) -> str:
    rows: list[str] = []
    artifacts = {
        "html_report": result.artifacts.html_report,
        "camera_lidar_overlay": result.artifacts.camera_lidar_overlay,
        "trajectory_plot": result.artifacts.trajectory_plot,
        "residual_histogram": result.artifacts.residual_histogram,
    }
    for name, path in artifacts.items():
        if path:
            rows.append(
                "<tr>"
                f"<td>{escape(name)}</td>"
                f"<td><code>{escape(path)}</code></td>"
                "</tr>"
            )
    return "\n".join(rows) or '<tr><td colspan="2">None</td></tr>'
