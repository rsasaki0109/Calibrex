"""HTML report rendering."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any

from calibrex.core.geometry import normalize_quaternion_xyzw
from calibrex.core.io import write_mapping
from calibrex.core.report_artifacts import (
    REPORT_DEGENERACY_SCHEMA_VERSION,
    REPORT_EVIDENCE_SCHEMA_VERSION,
    REPORT_METRICS_SCHEMA_VERSION,
    REPORT_OBSERVABILITY_SCHEMA_VERSION,
    REPORT_SUMMARY_SCHEMA_VERSION,
    EvidenceCaseItem,
    EvidenceSummaryItem,
    validate_report_sidecar_payload,
)
from calibrex.core.result import CalibrationResult, MetricResult, TransformResult
from calibrex.evaluation.evidence_summary import (
    evidence_cases_from_result,
    evidence_summaries_from_result,
)
from calibrex.evaluation.metric_families import (
    grade_counts,
    metric_family_payloads,
)

_WORLD_MAP_SUMMARY_METRICS = (
    "lidar_world_map_point_to_plane_rmse_m",
    "lidar_world_map_point_to_plane_median_holdout_m",
    "lidar_world_map_point_to_plane_p95_holdout_m",
    "lidar_world_map_perturbation_detectable_fraction",
    "lidar_world_map_weak_dof_count",
    "lidar_world_map_min_dof_sensitivity_m",
)

_LIDAR_PAIR_SUMMARY_METRICS = (
    "lidar_pair_shared_voxel_count",
    "lidar_pair_source_voxel_recall_in_target",
    "lidar_pair_target_voxel_recall_in_source",
    "lidar_pair_shared_voxel_centroid_rmse_m",
    "lidar_pair_holdout_point_to_plane_median_abs_m",
    "lidar_pair_holdout_point_to_plane_p90_abs_m",
    "lidar_pair_holdout_point_to_plane_rmse_m",
    "lidar_pair_holdout_point_to_plane_unmatched_fraction",
    "lidar_pair_known_bad_detectable_fraction",
    "lidar_pair_known_bad_centroid_rmse_delta_max_m",
    "lidar_pair_known_bad_point_to_plane_p90_delta_max_m",
)

_WORLD_MAP_DOF_METRICS = (
    ("roll_lidar0", "lidar_world_map_sensitivity_roll_m"),
    ("pitch_lidar0", "lidar_world_map_sensitivity_pitch_m"),
    ("yaw_lidar0", "lidar_world_map_sensitivity_yaw_m"),
    ("x_lidar0", "lidar_world_map_sensitivity_x_m"),
    ("y_lidar0", "lidar_world_map_sensitivity_y_m"),
    ("z_lidar0", "lidar_world_map_sensitivity_z_m"),
)

_REPORT_SIDECAR_KINDS = {
    "summary.json": "report-summary",
    "metrics.json": "report-metrics",
    "observability.json": "report-observability",
    "degeneracy.json": "report-degeneracy",
    "evidence.json": "report-evidence",
}


def report_artifact_paths(
    output_dir: str | Path,
    *,
    html_filename: str | Path = "report.html",
    include_html: bool = True,
) -> dict[str, str]:
    """Return the artifact paths produced by `write_report_artifacts`."""

    output_path = Path(output_dir)
    paths: dict[str, str] = {}
    if include_html:
        paths["html_report"] = str(_resolve_output_path(output_path, html_filename))
    for filename in _REPORT_SIDECAR_KINDS:
        paths[filename.removesuffix(".json")] = str(output_path / filename)
    return paths


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
    scoreboard_section = _scoreboard_section(result)
    artifact_rows = _artifact_rows(result)
    observability_rows = _observability_rows(result)
    lidar_world_map_section = _lidar_world_map_section(result)
    lidar_pair_section = _lidar_pair_section(result)

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
    .banner {{
      border: 1px solid #f0c36d;
      border-radius: 6px;
      background: #fff8e6;
      color: #5d4100;
      padding: 0.75rem;
      margin: 1rem 0 1.5rem;
    }}
    .scoreboard {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.75rem;
      margin: 1rem 0 2rem;
    }}
    .scorecard {{
      border: 1px solid #d6dce1;
      border-radius: 6px;
      padding: 0.75rem;
      background: #f8fafb;
    }}
    .scorecard .label {{
      color: #53616c;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .scorecard .value {{
      display: block;
      margin-top: 0.35rem;
      font-size: 1.35rem;
      font-weight: 800;
    }}
    .scorecard .detail {{
      color: #53616c;
      display: block;
      margin-top: 0.25rem;
      font-size: 0.85rem;
    }}
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
  {_cached_evidence_banner(result)}
  {scoreboard_section}
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
  {lidar_pair_section}
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

    written = report_artifact_paths(
        output_path,
        html_filename=html_filename,
        include_html=include_html,
    )
    if include_html:
        report_path = Path(written["html_report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        result.artifacts.html_report = str(report_path)
        report_path.write_text(render_html_report(result), encoding="utf-8")

    for name, payload in _report_sidecar_payloads(result).items():
        sidecar_path = output_path / name
        write_mapping(sidecar_path, payload)

    return written


def _resolve_output_path(output_dir: Path, filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return output_dir / path


def _report_sidecar_payloads(result: CalibrationResult) -> dict[str, dict[str, Any]]:
    payloads = {
        "summary.json": _summary_payload(result),
        "metrics.json": _metrics_payload(result),
        "observability.json": _observability_payload(result),
        "degeneracy.json": _degeneracy_payload(result),
        "evidence.json": _evidence_payload(result),
    }
    return {
        filename: validate_report_sidecar_payload(_REPORT_SIDECAR_KINDS[filename], payload)
        for filename, payload in payloads.items()
    }


def _summary_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SUMMARY_SCHEMA_VERSION,
        "run": _run_payload(result),
        "materialization": _evidence_materialization_payload(result),
        "quality": result.quality.model_dump(mode="json", exclude_none=True),
        "metric_counts": grade_counts(metric.grade for metric in result.metrics.values()),
        "transform_counts": grade_counts(
            transform.quality.grade for transform in result.transforms.values()
        ),
        "candidate_extrinsic_count": len(result.candidate_extrinsics),
        "reference_extrinsic_count": len(result.reference_extrinsics),
        "matched_candidate_reference_count": _matched_candidate_reference_count(result),
        "weak_direction_count": len(result.observability.weak_directions),
        "artifact_paths": result.artifacts.model_dump(mode="json", exclude_none=True),
        "evidence_summaries": [
            item.model_dump(mode="json") for item in evidence_summaries_from_result(result)
        ],
    }


def _metrics_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "schema_version": REPORT_METRICS_SCHEMA_VERSION,
        "run": _run_payload(result),
        "metrics": {
            name: metric.model_dump(mode="json", exclude_none=True)
            for name, metric in sorted(result.metrics.items())
        },
        "metric_families": metric_family_payloads(result.metrics),
    }


def _observability_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "schema_version": REPORT_OBSERVABILITY_SCHEMA_VERSION,
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
        "schema_version": REPORT_DEGENERACY_SCHEMA_VERSION,
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


def _evidence_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "schema_version": REPORT_EVIDENCE_SCHEMA_VERSION,
        "run": _run_payload(result),
        "materialization": _evidence_materialization_payload(result),
        "protocols": _evidence_protocol_payloads(result),
        "summaries": [
            item.model_dump(mode="json") for item in evidence_summaries_from_result(result)
        ],
        "cases": [item.model_dump(mode="json") for item in evidence_cases_from_result(result)],
    }


def _evidence_materialization_payload(result: CalibrationResult) -> dict[str, Any]:
    provenance = result.run.provenance
    metrics_origin = str(provenance.get("metrics_origin", "recomputed"))
    if metrics_origin not in {"recomputed", "cached", "unknown"}:
        metrics_origin = "unknown"
    data_verified = provenance.get("data_verified")
    return {
        "metrics_origin": metrics_origin,
        "data_verified": data_verified if isinstance(data_verified, bool) else None,
        "computed_at": _provenance_text(result, "computed_at"),
        "report_generated_at": result.run.created_at,
    }


def _evidence_protocol_payloads(result: CalibrationResult) -> list[dict[str, Any]]:
    protocols: list[dict[str, Any]] = []
    livox_pair = result.run.provenance.get("livox_pair_evidence")
    if isinstance(livox_pair, dict):
        protocols.append(_livox_pair_protocol_payload(result, livox_pair))
    return protocols


def _livox_pair_protocol_payload(
    result: CalibrationResult,
    livox_pair: dict[str, Any],
) -> dict[str, Any]:
    holdout_geometry = livox_pair.get("holdout_geometry")
    parameters: dict[str, Any] = {}
    limitations: list[str] = []
    status: str | None = None
    split_policy: str | None = None
    independent_holdout: bool | None = None
    if isinstance(holdout_geometry, dict):
        status = _str_or_none(holdout_geometry.get("status"))
        split_policy = _str_or_none(holdout_geometry.get("split_policy"))
        independent_holdout = _bool_or_none(holdout_geometry.get("independent_holdout"))
        parameters = {
            key: value
            for key, value in holdout_geometry.items()
            if key
            in {
                "map_voxel_count",
                "matched_point_count",
                "unmatched_point_count",
                "unmatched_fraction",
                "voxel_size_m",
                "correspondence_gate_m",
                "inlier_threshold_m",
                "plane_normal_source",
            }
        }
        reason = _str_or_none(holdout_geometry.get("reason"))
        if reason:
            limitations.append(reason)
    return {
        "family": "lidar_pair",
        "protocol_id": "livox_pair_single_pair_holdout_point_to_plane/v0.1",
        "status": status,
        "split_policy": split_policy,
        "independent_holdout": independent_holdout,
        "candidate_transform": _str_or_none(livox_pair.get("candidate_transform")),
        "transform_convention": _str_or_none(livox_pair.get("transform_convention")),
        "known_bad_perturbation": _str_or_none(livox_pair.get("known_bad_perturbation")),
        "known_bad_case_count": _int_or_none(livox_pair.get("known_bad_case_count")),
        "metric_ids": _evidence_metric_ids(result, family="lidar_pair"),
        "parameters": parameters,
        "limitations": limitations,
    }


def _run_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "id": result.run.id,
        "status": result.run.status,
        "domain": result.run.domain,
        "calibrex_version": result.run.calibrex_version,
        "git_commit": result.run.git_commit,
        "created_at": result.run.created_at,
        "dataset_type": _provenance_text(result, "dataset_type"),
        "dataset_path": _provenance_text(result, "dataset_path"),
    }


def _provenance_text(result: CalibrationResult, key: str) -> str | None:
    value = result.run.provenance.get(key)
    if value is None:
        return None
    return str(value)


def _scoreboard_section(result: CalibrationResult) -> str:
    metric_counts = grade_counts(metric.grade for metric in result.metrics.values())
    transform_counts = grade_counts(
        transform.quality.grade for transform in result.transforms.values()
    )
    matched_count = _matched_candidate_reference_count(result)
    candidate_count = len(result.candidate_extrinsics)
    weak_direction_count = len(result.observability.weak_directions)
    artifact_count = _artifact_count(result)
    return f"""
  <h2>Calibration Scoreboard</h2>
  <section class="scoreboard" aria-label="Calibration scoreboard">
    {_scorecard(
        "Overall",
        result.quality.grade.upper(),
        _quality_detail(result),
        grade=result.quality.grade,
    )}
    {_scorecard("Metric Grades", _grade_count_text(metric_counts), f"{len(result.metrics)} total")}
    {_scorecard(
        "Transform Grades",
        _grade_count_text(transform_counts),
        f"{len(result.transforms)} optimized",
    )}
    {_scorecard(
        "Candidate Matches",
        f"{matched_count} / {candidate_count}",
        "candidate-reference edges",
    )}
    {_scorecard(
        "Weak DoF",
        str(weak_direction_count),
        _weak_direction_detail(result),
        grade=result.observability.grade,
    )}
    {_scorecard(
        "Degeneracy",
        result.degeneracy.grade.upper(),
        result.degeneracy.reason or "No degeneracy reason reported",
        grade=result.degeneracy.grade,
    )}
    {_scorecard("Artifacts", str(artifact_count), "linked report outputs")}
  </section>
"""


def _scorecard(label: str, value: str, detail: str, *, grade: str | None = None) -> str:
    value_html = (
        f'<span class="value grade {escape(grade)}">{escape(value)}</span>'
        if grade is not None
        else f'<span class="value">{escape(value)}</span>'
    )
    return (
        '<div class="scorecard">'
        f'<span class="label">{escape(label)}</span>'
        f"{value_html}"
        f'<span class="detail">{escape(detail)}</span>'
        "</div>"
    )


def _grade_count_text(counts: dict[str, int]) -> str:
    return (
        f"PASS {counts['pass']} / "
        f"WARN {counts['warn']} / "
        f"FAIL {counts['fail']}"
    )


def _quality_detail(result: CalibrationResult) -> str:
    warning_count = len(result.quality.warnings)
    failure_count = len(result.quality.blocking_failures)
    if warning_count == 0 and failure_count == 0:
        return "no warnings or blocking failures"
    return f"{warning_count} warnings, {failure_count} blocking failures"


def _weak_direction_detail(result: CalibrationResult) -> str:
    if not result.observability.weak_directions:
        return "no weak directions reported"
    return ", ".join(result.observability.weak_directions)


def _artifact_count(result: CalibrationResult) -> int:
    return len(result.artifacts.model_dump(mode="json", exclude_none=True))


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


def _cached_evidence_banner(result: CalibrationResult) -> str:
    provenance = result.run.provenance
    if (
        provenance.get("metrics_origin") != "cached"
        and provenance.get("data_verified") is not False
    ):
        return ""
    metrics_origin = escape(str(provenance.get("metrics_origin", "unknown")))
    data_verified = escape(str(provenance.get("data_verified", "unknown")).lower())
    computed_at = provenance.get("computed_at")
    computed = f" Computed at: <code>{escape(str(computed_at))}</code>." if computed_at else ""
    return (
        '<div class="banner">'
        "<strong>CACHED EVIDENCE - RAW DATA NOT READ OR RECOMPUTED.</strong> "
        f"metrics_origin=<code>{metrics_origin}</code>, "
        f"data_verified=<code>{data_verified}</code>."
        f"{computed}"
        "</div>"
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


def _lidar_pair_section(result: CalibrationResult) -> str:
    summary_rows = _selected_metric_rows(result, _LIDAR_PAIR_SUMMARY_METRICS)
    if not summary_rows:
        return ""
    materialization_rows = _evidence_materialization_rows(result)
    protocol_rows = _evidence_protocol_rows(
        [
            protocol
            for protocol in _evidence_protocol_payloads(result)
            if protocol.get("family") == "lidar_pair"
        ]
    )
    protocol_section = (
        f"""
  <h3>Evidence Protocol</h3>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    {materialization_rows}
  </table>
  <table>
    <tr>
      <th>Protocol</th><th>Status</th><th>Split</th><th>Independent Holdout</th>
      <th>Known-Bad Cases</th><th>Parameters</th><th>Limitations</th>
    </tr>
    {protocol_rows}
  </table>
"""
        if protocol_rows
        else ""
    )
    evidence_rows = _evidence_summary_rows(
        [
            item
            for item in evidence_summaries_from_result(result)
            if item.family == "lidar_pair"
        ]
    )
    evidence_case_rows = _evidence_case_rows(
        [
            item
            for item in evidence_cases_from_result(result)
            if item.family == "lidar_pair"
        ]
    )
    evidence_case_section = (
        f"""
  <h3>Known-Bad Case Details</h3>
  <table>
    <tr>
      <th>Case</th><th>Status</th><th>DoF</th><th>Amount</th>
      <th>Source Recall Delta</th><th>Centroid RMSE Delta m</th>
      <th>P90 Point-to-Plane Delta m</th><th>Point-to-Plane RMSE Delta m</th>
      <th>P90 Point-to-Plane m</th><th>Unmatched Fraction</th>
    </tr>
    {evidence_case_rows}
  </table>
"""
        if evidence_case_rows
        else ""
    )
    return f"""
  <h2>LiDAR Pair Evidence</h2>
  <p>
    Fixed-LiDAR overlap and holdout geometry metrics for comparing solid-state
    or fixed-rig LiDAR extrinsic candidates. These are evidence metrics, not
    absolute ground truth.
  </p>
  {protocol_section}
  <h3>Evidence Summary</h3>
  <table>
    <tr><th>Check</th><th>Status</th><th>Evidence</th><th>Interpretation</th></tr>
    {evidence_rows}
  </table>
  {evidence_case_section}
  <h3>Metric Details</h3>
  <table>
    <tr><th>Metric</th><th>Grade</th><th>Train</th><th>Holdout</th><th>Value</th><th>Unit</th><th>Reason</th></tr>
    {summary_rows}
  </table>
"""


def _evidence_summary_rows(items: list[EvidenceSummaryItem]) -> str:
    return "\n".join(_evidence_summary_row(item) for item in items)


def _evidence_summary_row(item: EvidenceSummaryItem) -> str:
    status = item.status
    return (
        f'<tr class="{escape(_grade_css_class(status))}">'
        f"<td>{escape(item.check)}</td>"
        f"<td>{escape(status.upper())}</td>"
        f"<td>{escape(item.evidence)}</td>"
        f"<td>{escape(item.interpretation)}</td>"
        "</tr>"
    )


def _evidence_materialization_rows(result: CalibrationResult) -> str:
    materialization = _evidence_materialization_payload(result)
    rows = [
        ("Metrics Origin", _payload_value(materialization.get("metrics_origin"))),
        ("Data Verified", _payload_value(materialization.get("data_verified"))),
        ("Computed At", _payload_value(materialization.get("computed_at"))),
        ("Report Generated At", _payload_value(materialization.get("report_generated_at"))),
    ]
    return "\n".join(
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{escape(value)}</td>"
        "</tr>"
        for label, value in rows
        if value
    )


def _evidence_protocol_rows(protocols: list[dict[str, Any]]) -> str:
    return "\n".join(_evidence_protocol_row(protocol) for protocol in protocols)


def _evidence_protocol_row(protocol: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{escape(_payload_value(protocol.get('protocol_id')))}</td>"
        f"<td>{escape(_payload_value(protocol.get('status')))}</td>"
        f"<td>{escape(_payload_value(protocol.get('split_policy')))}</td>"
        f"<td>{escape(_payload_value(protocol.get('independent_holdout')))}</td>"
        f"<td>{escape(_payload_value(protocol.get('known_bad_case_count')))}</td>"
        f"<td>{escape(_protocol_parameters_text(protocol))}</td>"
        f"<td>{escape(_protocol_limitations_text(protocol))}</td>"
        "</tr>"
    )


def _protocol_parameters_text(protocol: dict[str, Any]) -> str:
    parameters = protocol.get("parameters")
    if not isinstance(parameters, dict):
        return ""
    return ", ".join(
        f"{key}={_payload_value(value)}"
        for key, value in sorted(parameters.items())
    )


def _protocol_limitations_text(protocol: dict[str, Any]) -> str:
    limitations = protocol.get("limitations")
    if not isinstance(limitations, list):
        return ""
    return "; ".join(str(item) for item in limitations)


def _payload_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _evidence_case_rows(items: list[EvidenceCaseItem]) -> str:
    return "\n".join(_evidence_case_row(item) for item in items)


def _evidence_case_row(item: EvidenceCaseItem) -> str:
    status = item.status
    amount = _case_amount(item)
    return (
        f'<tr class="{escape(_grade_css_class(status))}">'
        f"<td>{escape(item.case_id)}</td>"
        f"<td>{escape(status.upper())}</td>"
        f"<td>{escape(item.dof or '')}</td>"
        f"<td>{escape(amount)}</td>"
        f"<td>{_fmt(_case_delta(item, 'source_recall_delta'))}</td>"
        f"<td>{_fmt(_case_delta(item, 'centroid_rmse_delta_m'))}</td>"
        f"<td>{_fmt(_case_delta(item, 'point_to_plane_p90_delta_m'))}</td>"
        f"<td>{_fmt(_case_delta(item, 'point_to_plane_rmse_delta_m'))}</td>"
        f"<td>{_fmt(_case_metric(item, 'lidar_pair_holdout_point_to_plane_p90_abs_m'))}</td>"
        "<td>"
        f"{_fmt(_case_metric(item, 'lidar_pair_holdout_point_to_plane_unmatched_fraction'))}"
        "</td>"
        "</tr>"
    )


def _case_amount(item: EvidenceCaseItem) -> str:
    if item.amount is None:
        return ""
    return f"{item.amount:g}{item.unit or ''}"


def _case_delta(item: EvidenceCaseItem, name: str) -> float | None:
    return item.delta_values.get(name)


def _case_metric(item: EvidenceCaseItem, name: str) -> float | None:
    return item.metric_values.get(name)


def _evidence_metric_ids(result: CalibrationResult, *, family: str) -> list[str]:
    metric_ids: list[str] = []
    seen: set[str] = set()
    for item in evidence_summaries_from_result(result):
        if item.family != family:
            continue
        for metric_id in item.metric_ids:
            if metric_id in seen:
                continue
            metric_ids.append(metric_id)
            seen.add(metric_id)
    return metric_ids


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


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
        "rig_3d_viewer": result.artifacts.rig_3d_viewer,
        "camera_lidar_overlay": result.artifacts.camera_lidar_overlay,
        "trajectory_plot": result.artifacts.trajectory_plot,
        "residual_histogram": result.artifacts.residual_histogram,
    }
    for name, path in artifacts.items():
        if path:
            rows.append(_artifact_row(name, path))
    rows.extend(_report_sidecar_artifact_rows(result))
    return "\n".join(rows) or '<tr><td colspan="2">None</td></tr>'


def _report_sidecar_artifact_rows(result: CalibrationResult) -> list[str]:
    base_dir = _report_artifact_base_dir(result)
    sidecars = {
        filename.removesuffix(".json"): str(base_dir / filename)
        for filename in _REPORT_SIDECAR_KINDS
    }
    return [_artifact_row(name, path) for name, path in sidecars.items()]


def _report_artifact_base_dir(result: CalibrationResult) -> Path:
    html_report = result.artifacts.html_report
    if not html_report:
        return Path(".")
    html_path = Path(html_report)
    parent = html_path.parent
    return parent if str(parent) else Path(".")


def _artifact_row(name: str, path: str) -> str:
    return (
        "<tr>"
        f"<td>{escape(name)}</td>"
        f'<td><a href="{escape(path)}"><code>{escape(path)}</code></a></td>'
        "</tr>"
    )
