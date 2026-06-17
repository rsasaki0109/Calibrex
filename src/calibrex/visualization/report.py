"""HTML report rendering."""

from __future__ import annotations

from html import escape

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
    if metric.grade == "fail":
        return "metric-fail"
    if metric.grade == "warn":
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
