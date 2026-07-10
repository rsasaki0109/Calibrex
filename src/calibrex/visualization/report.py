"""HTML report rendering."""

from __future__ import annotations

import hashlib
import math
from html import escape
from pathlib import Path
from typing import Any

from calibrex.core.assessment import (
    AssessmentArtifact,
    assess_report_evidence,
    write_assessment_from_evidence,
)
from calibrex.core.evidence_bundle import (
    BundleArtifactKind,
    verify_evidence_bundle,
    write_evidence_bundle,
    write_evidence_bundle_verification,
)
from calibrex.core.evidence_contract import (
    ProtocolArtifact,
    policy_artifact_from_assessment,
)
from calibrex.core.geometry import normalize_quaternion_xyzw
from calibrex.core.io import write_mapping
from calibrex.core.online_timeline import (
    OnlineBatchSnapshot,
    OnlineCalibrationTimelineArtifact,
    OnlineGateStatus,
)
from calibrex.core.report_artifacts import (
    REPORT_DEGENERACY_SCHEMA_VERSION,
    REPORT_EVIDENCE_SCHEMA_VERSION,
    REPORT_METRICS_SCHEMA_VERSION,
    REPORT_OBSERVABILITY_SCHEMA_VERSION,
    REPORT_SUMMARY_SCHEMA_VERSION,
    EvidenceCaseItem,
    EvidenceProtocolItem,
    EvidenceSummaryItem,
    ReportEvidenceArtifact,
    SourceEvidenceReference,
    validate_report_sidecar_payload,
)
from calibrex.core.result import CalibrationResult, MetricResult, TransformResult
from calibrex.core.transform_artifacts import transform_artifact_from_result
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
    "lidar_pair_holdout_point_to_plane_eligible_point_count",
    "lidar_pair_holdout_point_to_plane_accepted_correspondence_count",
    "lidar_pair_holdout_point_to_plane_support_ratio",
    "lidar_pair_holdout_point_to_plane_median_abs_m",
    "lidar_pair_holdout_point_to_plane_p90_abs_m",
    "lidar_pair_holdout_point_to_plane_rmse_m",
    "lidar_pair_holdout_point_to_plane_unmatched_fraction",
    "lidar_pair_known_bad_detectable_fraction",
    "lidar_pair_known_bad_centroid_rmse_delta_max_m",
    "lidar_pair_known_bad_point_to_plane_p90_delta_max_m",
)

_LIDAR_CAMERA_SUMMARY_METRICS = (
    "lidar_camera_projected_points",
    "lidar_camera_projection_ratio",
    "lidar_camera_projection_horizontal_coverage",
    "lidar_camera_projection_vertical_coverage",
    "lidar_camera_edge_alignment_score",
    "lidar_camera_depth_discontinuity_points",
    "lidar_camera_depth_edge_alignment_score",
    "lidar_camera_perturbation_detectable_fraction",
    "lidar_camera_perturbation_mandatory_detectable_count",
    "lidar_camera_perturbation_edge_delta_mean",
    "lidar_camera_perturbation_depth_edge_delta_mean",
    "lidar_camera_perturbation_projection_ratio_delta_mean",
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

_REPORT_BUNDLE_FILENAME = "bundle.json"
_ASSESSMENT_FILENAME = "assessment.json"
_POLICY_FILENAME = "policy.json"
_PROTOCOL_FILENAME = "protocol.json"
_TRANSFORMS_FILENAME = "transforms.json"
_VERIFICATION_FILENAME = "verification.json"


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
    paths["assessment"] = str(output_path / _ASSESSMENT_FILENAME)
    paths["policy"] = str(output_path / _POLICY_FILENAME)
    paths["protocol"] = str(output_path / _PROTOCOL_FILENAME)
    paths["transforms"] = str(output_path / _TRANSFORMS_FILENAME)
    paths["bundle"] = str(output_path / _REPORT_BUNDLE_FILENAME)
    paths["verification"] = str(output_path / _VERIFICATION_FILENAME)
    return paths


def render_html_report(
    result: CalibrationResult,
    *,
    timeline: OnlineCalibrationTimelineArtifact | None = None,
) -> str:
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
    assessment_section = _assessment_section(result)
    lidar_world_map_section = _lidar_world_map_section(result)
    lidar_pair_section = _lidar_pair_section(result)
    lidar_camera_section = _lidar_camera_section(result)
    online_timeline_section = _online_timeline_section(result, timeline)

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
    .gate-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 2px;
      margin: 0.75rem 0 1.25rem;
    }}
    .gate-cell {{
      width: 14px;
      height: 28px;
      border-radius: 2px;
      border: 1px solid rgba(24, 32, 38, 0.12);
      cursor: default;
    }}
    .gate-pass {{ background: #d9f0e3; }}
    .gate-fail {{ background: #ffd9d9; }}
    .gate-inconclusive {{ background: #fff0c2; }}
    .online-chart {{
      display: block;
      width: 100%;
      max-width: 900px;
      height: auto;
      margin: 0.75rem 0 1.5rem;
      border: 1px solid #d6dce1;
      border-radius: 6px;
      background: #fcfdfe;
    }}
    .chart-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin: 0.25rem 0 1rem;
      font-size: 0.85rem;
      color: #53616c;
    }}
    .legend-swatch {{
      display: inline-block;
      width: 1rem;
      height: 0.2rem;
      margin-right: 0.35rem;
      vertical-align: middle;
      border-radius: 1px;
    }}
    .rank-lifted {{ background: #e8f4fc; }}
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
    <tr><th>Legacy Wire Producer Version</th><td>{escape(result.run.slac_version)}</td></tr>
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
  {assessment_section}
  {online_timeline_section}
  {lidar_world_map_section}
  {lidar_pair_section}
  {lidar_camera_section}
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
    timeline: OnlineCalibrationTimelineArtifact | None = None,
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
        report_path.write_text(
            render_html_report(result, timeline=timeline),
            encoding="utf-8",
        )

    evidence_payload = validate_report_sidecar_payload("report-evidence", _evidence_payload(result))
    evidence_path = output_path / "evidence.json"
    write_mapping(evidence_path, evidence_payload)
    evidence_sha256 = _sha256_file(evidence_path)
    source_evidence = SourceEvidenceReference(
        path="evidence.json",
        sha256=evidence_sha256,
        schema_version=REPORT_EVIDENCE_SCHEMA_VERSION,
        run_id=result.run.id,
    )

    for name, payload in _report_sidecar_payloads(
        result,
        source_evidence=source_evidence,
    ).items():
        sidecar_path = output_path / name
        write_mapping(sidecar_path, payload)
    evidence_model = ReportEvidenceArtifact.model_validate(evidence_payload)
    assessment = write_assessment_from_evidence(
        output_path / "evidence.json",
        output_path / _ASSESSMENT_FILENAME,
    )
    write_mapping(
        output_path / _PROTOCOL_FILENAME,
        _protocol_artifact(result).model_dump(mode="json", exclude_none=True),
    )
    write_mapping(
        output_path / _POLICY_FILENAME,
        policy_artifact_from_assessment(
            assessment,
            run=evidence_model.run,
        ).model_dump(mode="json", exclude_none=True),
    )
    write_mapping(
        output_path / _TRANSFORMS_FILENAME,
        transform_artifact_from_result(
            result,
            run=evidence_model.run,
        ).model_dump(mode="json", exclude_none=True),
    )
    write_evidence_bundle(
        output_path / _REPORT_BUNDLE_FILENAME,
        run_id=result.run.id,
        primary_evidence_path=output_path / "evidence.json",
        artifacts=_bundle_artifacts(output_path, written, include_html=include_html),
    )
    verification = verify_evidence_bundle(output_path / _REPORT_BUNDLE_FILENAME)
    write_evidence_bundle_verification(
        output_path / _VERIFICATION_FILENAME,
        verification,
        source_bundle_path=output_path / _REPORT_BUNDLE_FILENAME,
    )

    return written


def evidence_artifact_from_result(result: CalibrationResult) -> ReportEvidenceArtifact:
    """Build the standalone evidence artifact represented by a result."""

    payload = validate_report_sidecar_payload("report-evidence", _evidence_payload(result))
    return ReportEvidenceArtifact.model_validate(payload)


def write_evidence_artifact(
    result: CalibrationResult,
    output_path: str | Path,
) -> ReportEvidenceArtifact:
    """Write only `evidence.json` for an existing result."""

    evidence = evidence_artifact_from_result(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_mapping(path, evidence.model_dump(mode="json", exclude_none=True))
    return evidence


def write_evidence_contract_artifacts(
    result: CalibrationResult,
    *,
    evidence_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write evidence and its audit sidecars without rendering HTML."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evidence_file = Path(evidence_path)
    evidence = write_evidence_artifact(result, evidence_file)

    assessment = write_assessment_from_evidence(
        evidence_file,
        output_path / _ASSESSMENT_FILENAME,
    )
    write_mapping(
        output_path / _PROTOCOL_FILENAME,
        ProtocolArtifact(run=evidence.run, protocols=list(evidence.protocols)).model_dump(
            mode="json",
            exclude_none=True,
        ),
    )
    write_mapping(
        output_path / _POLICY_FILENAME,
        policy_artifact_from_assessment(
            assessment,
            run=evidence.run,
        ).model_dump(mode="json", exclude_none=True),
    )
    write_mapping(
        output_path / _TRANSFORMS_FILENAME,
        transform_artifact_from_result(
            result,
            run=evidence.run,
        ).model_dump(mode="json", exclude_none=True),
    )
    bundle_path = output_path / _REPORT_BUNDLE_FILENAME
    write_evidence_bundle(
        bundle_path,
        run_id=result.run.id,
        primary_evidence_path=evidence_file,
        artifacts=[
            (evidence_file, "report-evidence"),
            (output_path / _ASSESSMENT_FILENAME, "assessment"),
            (output_path / _PROTOCOL_FILENAME, "protocol"),
            (output_path / _POLICY_FILENAME, "policy"),
            (output_path / _TRANSFORMS_FILENAME, "transforms"),
        ],
    )
    verification = verify_evidence_bundle(bundle_path)
    write_evidence_bundle_verification(
        output_path / _VERIFICATION_FILENAME,
        verification,
        source_bundle_path=bundle_path,
    )
    return {
        "evidence": str(evidence_file),
        "assessment": str(output_path / _ASSESSMENT_FILENAME),
        "protocol": str(output_path / _PROTOCOL_FILENAME),
        "policy": str(output_path / _POLICY_FILENAME),
        "transforms": str(output_path / _TRANSFORMS_FILENAME),
        "bundle": str(bundle_path),
        "verification": str(output_path / _VERIFICATION_FILENAME),
    }


def _bundle_artifacts(
    output_path: Path,
    written: dict[str, str],
    *,
    include_html: bool,
) -> list[tuple[Path, BundleArtifactKind]]:
    artifacts: list[tuple[Path, BundleArtifactKind]] = []
    if include_html:
        artifacts.append((Path(written["html_report"]), "report-html"))
    artifacts.extend(
        [
            (output_path / "summary.json", "report-summary"),
            (output_path / "metrics.json", "report-metrics"),
            (output_path / "observability.json", "report-observability"),
            (output_path / "degeneracy.json", "report-degeneracy"),
            (output_path / "evidence.json", "report-evidence"),
            (output_path / _ASSESSMENT_FILENAME, "assessment"),
            (output_path / _PROTOCOL_FILENAME, "protocol"),
            (output_path / _POLICY_FILENAME, "policy"),
            (output_path / _TRANSFORMS_FILENAME, "transforms"),
        ]
    )
    return artifacts


def _resolve_output_path(output_dir: Path, filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return output_dir / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_sidecar_payloads(
    result: CalibrationResult,
    *,
    source_evidence: SourceEvidenceReference,
) -> dict[str, dict[str, Any]]:
    payloads = {
        "summary.json": _summary_payload(result, source_evidence=source_evidence),
        "metrics.json": _metrics_payload(result, source_evidence=source_evidence),
        "observability.json": _observability_payload(
            result,
            source_evidence=source_evidence,
        ),
        "degeneracy.json": _degeneracy_payload(result, source_evidence=source_evidence),
    }
    return {
        filename: validate_report_sidecar_payload(_REPORT_SIDECAR_KINDS[filename], payload)
        for filename, payload in payloads.items()
    }


def _summary_payload(
    result: CalibrationResult,
    *,
    source_evidence: SourceEvidenceReference,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SUMMARY_SCHEMA_VERSION,
        "run": _run_payload(result),
        "source_evidence": source_evidence.model_dump(mode="json", exclude_none=True),
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


def _metrics_payload(
    result: CalibrationResult,
    *,
    source_evidence: SourceEvidenceReference,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_METRICS_SCHEMA_VERSION,
        "run": _run_payload(result),
        "source_evidence": source_evidence.model_dump(mode="json", exclude_none=True),
        "metrics": {
            name: metric.model_dump(mode="json", exclude_none=True)
            for name, metric in sorted(result.metrics.items())
        },
        "metric_families": metric_family_payloads(result.metrics),
    }


def _observability_payload(
    result: CalibrationResult,
    *,
    source_evidence: SourceEvidenceReference,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_OBSERVABILITY_SCHEMA_VERSION,
        "run": _run_payload(result),
        "source_evidence": source_evidence.model_dump(mode="json", exclude_none=True),
        "observability": result.observability.model_dump(mode="json", exclude_none=True),
        "weak_directions": list(result.observability.weak_directions),
        "metrics": {
            name: metric.model_dump(mode="json", exclude_none=True)
            for name, metric in sorted(result.metrics.items())
            if "observability" in name or "sensitivity" in name or "weak_dof" in name
        },
    }


def _degeneracy_payload(
    result: CalibrationResult,
    *,
    source_evidence: SourceEvidenceReference,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_DEGENERACY_SCHEMA_VERSION,
        "run": _run_payload(result),
        "source_evidence": source_evidence.model_dump(mode="json", exclude_none=True),
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
        "input_files": _evidence_input_files_payload(result),
        "summaries": [
            item.model_dump(mode="json") for item in evidence_summaries_from_result(result)
        ],
        "cases": [item.model_dump(mode="json") for item in evidence_cases_from_result(result)],
    }


def _protocol_artifact(result: CalibrationResult) -> ProtocolArtifact:
    return ProtocolArtifact(
        run=ReportEvidenceArtifact.model_validate(_evidence_payload(result)).run,
        protocols=[
            EvidenceProtocolItem.model_validate(protocol)
            for protocol in _evidence_protocol_payloads(result)
        ],
    )


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


def _evidence_input_files_payload(result: CalibrationResult) -> list[dict[str, Any]]:
    raw_files = result.run.provenance.get("raw_input_files")
    if not isinstance(raw_files, list):
        return []
    files: list[dict[str, Any]] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            continue
        path = _str_or_none(raw_file.get("path"))
        if path is None:
            continue
        size_bytes = _int_or_none(raw_file.get("size_bytes"))
        files.append(
            {
                "path": path,
                "role": _str_or_none(raw_file.get("role")),
                "sha256": _str_or_none(raw_file.get("sha256")),
                "size_bytes": size_bytes,
                "source_url": _str_or_none(raw_file.get("source_url")),
            }
        )
    return files


def _evidence_protocol_payloads(result: CalibrationResult) -> list[dict[str, Any]]:
    protocols: list[dict[str, Any]] = []
    livox_pair = result.run.provenance.get("livox_pair_evidence")
    if isinstance(livox_pair, dict):
        protocols.append(_livox_pair_protocol_payload(result, livox_pair))
    lidar_camera = result.run.provenance.get("lidar_camera_evidence")
    if isinstance(lidar_camera, dict):
        protocols.append(_lidar_camera_protocol_payload(result, lidar_camera))
    radar_lidar = result.run.provenance.get("radar_velocity_consistency")
    if isinstance(radar_lidar, dict):
        protocols.append(_radar_lidar_protocol_payload(result, radar_lidar))
    return protocols


def _radar_lidar_protocol_payload(
    result: CalibrationResult,
    radar: dict[str, Any],
) -> dict[str, Any]:
    split = radar.get("split")
    challenge = radar.get("known_bad_rotation_probes")
    assessment = radar.get("assessment")
    policy = radar.get("policy")
    holdout_ids = split.get("holdout_frame_ids") if isinstance(split, dict) else None
    limitations = ["yaw-only Doppler evidence; translation is unobservable"]
    if isinstance(assessment, dict) and assessment.get("status") != "pass":
        reason = _str_or_none(assessment.get("reason"))
        if reason:
            limitations.append(reason)
    parameters: dict[str, Any] = {
        "split": split,
        "residual_summary": radar.get("residual_summary"),
        "static_return_summary": radar.get("static_return_summary"),
        "holdout_yaw_observability": radar.get("holdout_yaw_observability"),
        "known_bad_rotation_probes": challenge,
        "policy": policy,
        "assessment": assessment,
        "sensor_evaluations": radar.get("sensor_evaluations"),
    }
    sensor_evaluations = radar.get("sensor_evaluations")
    if isinstance(sensor_evaluations, dict):
        independent_holdout = bool(sensor_evaluations) and all(
            isinstance(item, dict)
            and isinstance(item.get("split"), dict)
            and bool(item["split"].get("holdout_frame_ids"))
            for item in sensor_evaluations.values()
        )
    else:
        independent_holdout = bool(holdout_ids) if isinstance(holdout_ids, list) else False
    return {
        "family": "radar_lidar",
        "protocol_id": "radar_lidar_seeded_holdout_doppler_yaw/v0.1",
        "status": _str_or_none(assessment.get("status"))
        if isinstance(assessment, dict)
        else "inconclusive",
        "split_policy": _str_or_none(split.get("policy"))
        if isinstance(split, dict)
        else None,
        "independent_holdout": independent_holdout,
        "candidate_transform": "configured T_ego_radar transforms",
        "transform_convention": "T_ego_radar maps radar-frame LOS into ego frame",
        "known_bad_perturbation": "left-composed yaw rotation about ego +Z",
        "known_bad_case_count": _int_or_none(challenge.get("required_count"))
        if isinstance(challenge, dict)
        else None,
        "metric_ids": _evidence_metric_ids(result, family="radar_lidar"),
        "parameters": parameters,
        "limitations": limitations,
    }


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
                "support_population_id",
                "support_definition",
                "eligible_point_count",
                "considered_point_count",
                "accepted_correspondence_count",
                "support_ratio",
                "exclusion_counts",
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
    known_bad_challenge = livox_pair.get("known_bad_challenge")
    if isinstance(known_bad_challenge, dict):
        parameters.update(
            {
                key: value
                for key, value in known_bad_challenge.items()
                if key
                in {
                    "challenge_id",
                    "composition",
                    "tangent_frame",
                    "mandatory_rotation_deg",
                    "mandatory_translation_m",
                    "mandatory_case_count",
                    "mandatory_supported_detection_count",
                    "mandatory_supported_detection_fraction",
                    "mandatory_detection_by_dof",
                    "mandatory_support_collapse_count",
                    "min_support_ratio",
                    "min_accepted_correspondence_count",
                    "target_supported_detection_count",
                }
            }
        )
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


def _lidar_camera_protocol_payload(
    result: CalibrationResult,
    lidar_camera: dict[str, Any],
) -> dict[str, Any]:
    parameters = {
        key: value
        for key, value in lidar_camera.items()
        if key
        in {
            "protocol_id",
            "split_policy",
            "projection_frame_count",
            "use_frame_graph_candidate",
            "evidence_gate_min_edge_alignment_holdout",
            "evidence_gate_min_depth_edge_alignment_holdout",
            "evidence_gate_min_perturbation_detectable_fraction",
            "evidence_gate_min_mandatory_detectable_count",
        }
        or key.startswith("evidence_gate_")
    }
    challenge = lidar_camera.get("known_bad_challenge")
    if isinstance(challenge, dict):
        parameters.update(
            {
                key: value
                for key, value in challenge.items()
                if key
                in {
                    "challenge_id",
                    "mandatory_rotation_deg",
                    "mandatory_translation_m",
                    "mandatory_case_count",
                    "target_mandatory_detectable_count",
                }
            }
        )
    limitations = [
        str(item)
        for item in lidar_camera.get("limitations", [])
        if isinstance(item, str)
    ]
    return {
        "family": "lidar_camera",
        "protocol_id": _str_or_none(lidar_camera.get("protocol_id"))
        or "kitti_lidar_camera_projection_edge_holdout/v0.1",
        "status": "scored",
        "split_policy": _str_or_none(lidar_camera.get("split_policy")),
        "independent_holdout": _bool_or_none(lidar_camera.get("independent_holdout")),
        "candidate_transform": _str_or_none(lidar_camera.get("candidate_transform")),
        "transform_convention": _str_or_none(lidar_camera.get("transform_convention")),
        "known_bad_perturbation": _str_or_none(lidar_camera.get("known_bad_perturbation")),
        "known_bad_case_count": _int_or_none(lidar_camera.get("known_bad_case_count")),
        "metric_ids": _evidence_metric_ids(result, family="lidar_camera"),
        "parameters": parameters,
        "limitations": limitations,
    }


def _run_payload(result: CalibrationResult) -> dict[str, Any]:
    return {
        "id": result.run.id,
        "status": result.run.status,
        "domain": result.run.domain,
        "slac_version": result.run.slac_version,
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
    metrics_origin_raw = provenance.get("metrics_origin")
    data_verified_raw = provenance.get("data_verified")
    if metrics_origin_raw == "recomputed" and data_verified_raw is True:
        return ""
    metrics_origin = escape(str(metrics_origin_raw or "unknown"))
    data_verified_value = data_verified_raw if data_verified_raw is not None else "unknown"
    data_verified = escape(str(data_verified_value).lower())
    computed_at = provenance.get("computed_at")
    computed = f" Computed at: <code>{escape(str(computed_at))}</code>." if computed_at else ""
    return (
        '<div class="banner">'
        "<strong>EVIDENCE INPUTS NOT VERIFIED AS RAW RECOMPUTATION.</strong> "
        f"metrics_origin=<code>{metrics_origin}</code>, "
        f"data_verified=<code>{data_verified}</code>."
        f"{computed}"
        "</div>"
    )


def _assessment_section(result: CalibrationResult) -> str:
    assessment = _assessment_from_result(result)
    if assessment is None:
        return ""
    rule_rows = "\n".join(_assessment_rule_row(rule) for rule in assessment.rules)
    return f"""
  <h2>Falsification Assessment</h2>
  <p>
    Status:
    <span class="grade {_assessment_css_class(assessment.status)}">
      {escape(assessment.status.upper())}
    </span>
    {escape(assessment.reason)}
  </p>
  <table>
    <tr><th>Rule</th><th>Status</th><th>Reason</th><th>Observed</th><th>Thresholds</th></tr>
    {rule_rows}
  </table>
"""


def _assessment_from_result(result: CalibrationResult) -> AssessmentArtifact | None:
    try:
        evidence = ReportEvidenceArtifact.model_validate(_evidence_payload(result))
    except Exception:
        return None
    return assess_report_evidence(evidence)


def _assessment_rule_row(rule: Any) -> str:
    return (
        f'<tr class="{escape(_assessment_css_class(rule.status))}">'
        f"<td>{escape(rule.rule_id)}</td>"
        f"<td>{escape(rule.status.upper())}</td>"
        f"<td>{escape(rule.reason)}</td>"
        f"<td>{escape(_mapping_text(rule.observed))}</td>"
        f"<td>{escape(_mapping_text(rule.thresholds))}</td>"
        "</tr>"
    )


def _assessment_css_class(status: str) -> str:
    if status == "fail":
        return "fail"
    if status == "inconclusive":
        return "warn"
    return "pass"


def _mapping_text(values: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_payload_value(value)}" for key, value in sorted(values.items()))


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
    input_file_rows = _evidence_input_file_rows(_evidence_input_files_payload(result))
    input_files_section = (
        f"""
  <h3>Raw Input Files</h3>
  <table>
    <tr><th>Role</th><th>Path</th><th>SHA-256</th><th>Size bytes</th><th>Source URL</th></tr>
    {input_file_rows}
  </table>
"""
        if input_file_rows
        else ""
    )
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
      <th>Support Ratio Delta</th><th>P90 Point-to-Plane m</th>
      <th>Support Ratio</th><th>Unmatched Fraction</th>
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
  {input_files_section}
  {evidence_case_section}
  <h3>Metric Details</h3>
  <table>
    <tr><th>Metric</th><th>Grade</th><th>Train</th><th>Holdout</th><th>Value</th><th>Unit</th><th>Reason</th></tr>
    {summary_rows}
  </table>
"""


def _lidar_camera_section(result: CalibrationResult) -> str:
    summary_rows = _selected_metric_rows(result, _LIDAR_CAMERA_SUMMARY_METRICS)
    if not summary_rows:
        return ""
    protocol_rows = _evidence_protocol_rows(
        [
            protocol
            for protocol in _evidence_protocol_payloads(result)
            if protocol.get("family") == "lidar_camera"
        ]
    )
    protocol_section = (
        f"""
  <h3>Evidence Protocol</h3>
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
            if item.family == "lidar_camera"
        ]
    )
    evidence_case_rows = _lidar_camera_evidence_case_rows(
        [
            item
            for item in evidence_cases_from_result(result)
            if item.family == "lidar_camera"
        ]
    )
    evidence_case_section = (
        f"""
  <h3>Known-Bad Case Details</h3>
  <table>
    <tr>
      <th>Case</th><th>Status</th><th>DoF</th><th>Amount</th>
      <th>Edge Delta</th><th>Depth-Edge Delta</th><th>Projection-Ratio Delta</th>
      <th>Edge Score</th><th>Depth-Edge Score</th><th>Projection Ratio</th>
    </tr>
    {evidence_case_rows}
  </table>
"""
        if evidence_case_rows
        else ""
    )
    return f"""
  <h2>Camera-LiDAR Evidence</h2>
  <p>
    KITTI projection and edge-alignment evidence for evaluating camera-LiDAR
    extrinsic candidates. This is an evaluation layer, not a standalone camera
    calibration algorithm.
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


def _lidar_camera_evidence_case_rows(items: list[EvidenceCaseItem]) -> str:
    return "\n".join(_lidar_camera_evidence_case_row(item) for item in items)


def _lidar_camera_evidence_case_row(item: EvidenceCaseItem) -> str:
    status = item.status
    amount = _case_amount(item)
    return (
        f'<tr class="{escape(_grade_css_class(status))}">'
        f"<td>{escape(item.case_id)}</td>"
        f"<td>{escape(status.upper())}</td>"
        f"<td>{escape(item.dof or '')}</td>"
        f"<td>{escape(amount)}</td>"
        f"<td>{_fmt(_case_delta(item, 'edge_alignment_delta'))}</td>"
        f"<td>{_fmt(_case_delta(item, 'depth_edge_alignment_delta'))}</td>"
        f"<td>{_fmt(_case_delta(item, 'projection_ratio_delta'))}</td>"
        f"<td>{_fmt(_case_metric(item, 'lidar_camera_edge_alignment_score'))}</td>"
        f"<td>{_fmt(_case_metric(item, 'lidar_camera_depth_edge_alignment_score'))}</td>"
        f"<td>{_fmt(_case_metric(item, 'lidar_camera_projection_ratio'))}</td>"
        "</tr>"
    )


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


def _evidence_input_file_rows(input_files: list[dict[str, Any]]) -> str:
    return "\n".join(_evidence_input_file_row(input_file) for input_file in input_files)


def _evidence_input_file_row(input_file: dict[str, Any]) -> str:
    sha256 = _payload_value(input_file.get("sha256"))
    short_sha = sha256[:16] + "..." if len(sha256) > 16 else sha256
    return (
        "<tr>"
        f"<td>{escape(_payload_value(input_file.get('role')))}</td>"
        f"<td>{escape(_payload_value(input_file.get('path')))}</td>"
        f"<td><code>{escape(short_sha)}</code></td>"
        f"<td>{escape(_payload_value(input_file.get('size_bytes')))}</td>"
        f"<td>{escape(_payload_value(input_file.get('source_url')))}</td>"
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
        f"<td>{_fmt(_case_delta(item, 'support_ratio_delta'))}</td>"
        f"<td>{_fmt(_case_metric(item, 'lidar_pair_holdout_point_to_plane_p90_abs_m'))}</td>"
        f"<td>{_fmt(_case_metric(item, 'lidar_pair_holdout_point_to_plane_support_ratio'))}</td>"
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
    sidecars["assessment"] = str(base_dir / _ASSESSMENT_FILENAME)
    sidecars["protocol"] = str(base_dir / _PROTOCOL_FILENAME)
    sidecars["policy"] = str(base_dir / _POLICY_FILENAME)
    sidecars["transforms"] = str(base_dir / _TRANSFORMS_FILENAME)
    sidecars["bundle"] = str(base_dir / _REPORT_BUNDLE_FILENAME)
    sidecars["verification"] = str(base_dir / _VERIFICATION_FILENAME)
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


def _online_timeline_section(
    result: CalibrationResult,
    timeline: OnlineCalibrationTimelineArtifact | None,
) -> str:
    if timeline is None or not timeline.batches:
        return ""

    thresholds = _online_gate_thresholds(result)
    gate_strip = _online_gate_verdict_strip(timeline.batches)
    rmse_chart = _online_rmse_chart(timeline.batches, thresholds)
    observability_track = _online_observability_track(timeline)
    summary_table = _online_timeline_summary_table(result, timeline, thresholds)
    return f"""
  <h2>Online Timeline</h2>
  <p>
    Per-batch gate verdicts, holdout and rolling RMSE, and observability for
    <code>{escape(timeline.variable)}</code>
    ({escape(timeline.source_sensor)} → {escape(timeline.target_sensor)}).
    Data from <code>timeline.json</code>
    ({escape(timeline.schema_version)}).
  </p>
  <h3>Gate Verdicts</h3>
  <p class="detail">Hover a cell for batch index and gate reason.</p>
  {gate_strip}
  <h3>Holdout and Rolling RMSE</h3>
  {rmse_chart}
  <h3>Observability</h3>
  {observability_track}
  <h3>Session Summary</h3>
  {summary_table}
"""


def _online_gate_thresholds(result: CalibrationResult) -> dict[str, float | int]:
    provenance = result.run.provenance
    return {
        "min_rank": int(provenance.get("online_gate_min_rank", 6)),
        "max_holdout_rmse_m": float(provenance.get("online_gate_max_holdout_rmse_m", 0.05)),
        "max_rolling_regression_m": float(
            provenance.get("online_gate_max_rolling_regression_m", 0.02)
        ),
    }


def _online_gate_verdict_strip(batches: list[OnlineBatchSnapshot]) -> str:
    cells = []
    for batch in batches:
        css = _online_gate_css_class(batch.gate_status)
        title = f"batch {batch.batch_index}: {batch.gate_status} — {batch.gate_reason}"
        cells.append(
            f'<span class="gate-cell {css}" title="{escape(title)}"></span>'
        )
    return f'<div class="gate-strip" aria-label="Per-batch gate verdicts">{"".join(cells)}</div>'


def _online_gate_css_class(status: OnlineGateStatus) -> str:
    if status == "pass":
        return "gate-pass"
    if status == "fail":
        return "gate-fail"
    return "gate-inconclusive"


def _online_rmse_chart(
    batches: list[OnlineBatchSnapshot],
    thresholds: dict[str, float | int],
) -> str:
    width = 860
    height = 240
    pad_left = 56
    pad_right = 20
    pad_top = 20
    pad_bottom = 44
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    holdout_values = [
        batch.batch_holdout_rmse_m
        for batch in batches
        if batch.batch_holdout_rmse_m is not None
    ]
    rolling_values = [
        batch.rolling_rmse_m for batch in batches if batch.rolling_rmse_m is not None
    ]
    regression_values = [
        batch.rolling_rmse_m + float(thresholds["max_rolling_regression_m"])
        for batch in batches
        if batch.rolling_rmse_m is not None
    ]
    max_holdout = float(thresholds["max_holdout_rmse_m"])
    y_values = [
        *holdout_values,
        *rolling_values,
        *regression_values,
        max_holdout,
    ]
    if not y_values:
        return "<p>No RMSE values recorded in the timeline.</p>"

    y_min = 0.0
    y_max = max(y_values) * 1.08
    if y_max <= y_min:
        y_max = y_min + 0.01

    def x_coord(batch_index: int) -> float:
        if len(batches) <= 1:
            return pad_left + plot_width / 2
        return pad_left + (batch_index / (len(batches) - 1)) * plot_width

    def y_coord(value: float) -> float:
        ratio = (value - y_min) / (y_max - y_min)
        return pad_top + (1.0 - ratio) * plot_height

    holdout_points = " ".join(
        f"{x_coord(batch.batch_index):.2f},{y_coord(batch.batch_holdout_rmse_m):.2f}"
        for batch in batches
        if batch.batch_holdout_rmse_m is not None
    )
    rolling_points = " ".join(
        f"{x_coord(batch.batch_index):.2f},{y_coord(batch.rolling_rmse_m):.2f}"
        for batch in batches
        if batch.rolling_rmse_m is not None
    )
    regression_points = " ".join(
        f"{x_coord(batch.batch_index):.2f},"
        f"{y_coord(batch.rolling_rmse_m + float(thresholds['max_rolling_regression_m'])):.2f}"
        for batch in batches
        if batch.rolling_rmse_m is not None
    )
    threshold_y = y_coord(max_holdout)
    y_ticks = _chart_y_ticks(y_min, y_max)
    y_axis = "\n".join(
        f'<text x="{pad_left - 8}" y="{y_coord(tick) + 4:.2f}" '
        f'text-anchor="end" font-size="11" fill="#53616c">{_fmt(tick)}</text>'
        f'<line x1="{pad_left}" y1="{y_coord(tick):.2f}" '
        f'x2="{width - pad_right}" y2="{y_coord(tick):.2f}" '
        f'stroke="#e5eaee" stroke-width="1"/>'
        for tick in y_ticks
    )
    x_label_positions = _chart_x_label_positions(len(batches))
    x_axis = "\n".join(
        f'<text x="{x_coord(index):.2f}" y="{height - 12}" '
        f'text-anchor="middle" font-size="11" fill="#53616c">{index}</text>'
        for index in x_label_positions
    )
    polylines = []
    if holdout_points:
        polylines.append(
            f'<polyline fill="none" stroke="#2563eb" stroke-width="2" '
            f'points="{holdout_points}"/>'
        )
    if rolling_points:
        polylines.append(
            f'<polyline fill="none" stroke="#059669" stroke-width="2" '
            f'points="{rolling_points}"/>'
        )
    if regression_points:
        polylines.append(
            f'<polyline fill="none" stroke="#d97706" stroke-width="1.5" '
            f'stroke-dasharray="5 4" points="{regression_points}"/>'
        )
    polylines.append(
        f'<line x1="{pad_left}" y1="{threshold_y:.2f}" '
        f'x2="{width - pad_right}" y2="{threshold_y:.2f}" '
        f'stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    legend = f"""
  <div class="chart-legend" aria-hidden="true">
    <span><span class="legend-swatch" style="background:#2563eb"></span>Holdout RMSE (m)</span>
    <span><span class="legend-swatch" style="background:#059669"></span>Rolling RMSE (m)</span>
    <span><span class="legend-swatch" style="background:#d97706"></span>
      Rolling + max regression Δ ({_fmt(float(thresholds['max_rolling_regression_m']))} m)</span>
    <span><span class="legend-swatch" style="background:#dc2626"></span>
      Max holdout RMSE ({_fmt(max_holdout)} m)</span>
  </div>
"""
    return (
        legend
        + f"""
  <svg class="online-chart" viewBox="0 0 {width} {height}" role="img"
       aria-label="Holdout and rolling RMSE per batch">
    <rect x="{pad_left}" y="{pad_top}" width="{plot_width}" height="{plot_height}"
          fill="#ffffff" stroke="#d6dce1"/>
    {y_axis}
    {x_axis}
    <text x="{pad_left + plot_width / 2:.2f}" y="{height - 2}" text-anchor="middle"
          font-size="12" fill="#182026">Batch index</text>
    <text x="14" y="{pad_top + plot_height / 2:.2f}" text-anchor="middle"
          font-size="12" fill="#182026"
          transform="rotate(-90 14 {pad_top + plot_height / 2:.2f})">RMSE (m)</text>
    {"".join(polylines)}
  </svg>
"""
    )


def _chart_y_ticks(y_min: float, y_max: float) -> list[float]:
    span = y_max - y_min
    if span <= 0:
        return [y_min, y_max]
    step = 10 ** math.floor(math.log10(span))
    if span / step < 2:
        step /= 5
    elif span / step < 5:
        step /= 2
    start = math.ceil(y_min / step) * step
    ticks = [y_min]
    value = start
    while value < y_max:
        if value > y_min:
            ticks.append(value)
        value += step
    if y_max not in ticks:
        ticks.append(y_max)
    return ticks


def _chart_x_label_positions(batch_count: int) -> list[int]:
    if batch_count <= 1:
        return [0]
    if batch_count <= 8:
        return list(range(batch_count))
    step = max(1, batch_count // 8)
    positions = list(range(0, batch_count, step))
    if positions[-1] != batch_count - 1:
        positions.append(batch_count - 1)
    return positions


def _online_observability_track(timeline: OnlineCalibrationTimelineArtifact) -> str:
    batches = timeline.batches
    has_batch_observability = any(batch.batch_observability is not None for batch in batches)
    rows: list[str] = []
    for batch in batches:
        accumulated_rank = batch.observability.rank
        batch_rank = (
            batch.batch_observability.rank
            if batch.batch_observability is not None
            else accumulated_rank
        )
        rank_lifted = (
            has_batch_observability
            and batch.batch_observability is not None
            and accumulated_rank is not None
            and batch_rank is not None
            and accumulated_rank > batch_rank
        )
        row_class = ' class="rank-lifted"' if rank_lifted else ""
        weak = (
            ", ".join(batch.observability.weak_directions)
            if batch.observability.weak_directions
            else "None"
        )
        rows.append(
            f"<tr{row_class}>"
            f"<td>{batch.batch_index}</td>"
            f"<td>{escape(batch.gate_status.upper())}</td>"
            f"<td>{_fmt_int(batch_rank)}</td>"
            f"<td>{_fmt_int(accumulated_rank)}</td>"
            f"<td>{_fmt(batch.observability.condition_number)}</td>"
            f"<td>{escape(weak)}</td>"
            f"<td>{'yes' if batch.retained_for_accumulation else 'no'}</td>"
            "</tr>"
        )
    batch_rank_header = (
        "<th>Batch-only rank</th><th>Accumulated rank</th>"
        if has_batch_observability
        else "<th>Rank</th><th>Accumulated rank</th>"
    )
    return f"""
  <table>
    <tr>
      <th>Batch</th><th>Gate</th>{batch_rank_header}
      <th>Condition number</th><th>Weak DoF</th><th>Retained</th>
    </tr>
    {"".join(rows)}
  </table>
"""


def _online_timeline_summary_table(
    result: CalibrationResult,
    timeline: OnlineCalibrationTimelineArtifact,
    thresholds: dict[str, float | int],
) -> str:
    retained_count = sum(1 for batch in timeline.batches if batch.retained_for_accumulation)
    final_estimate = _online_final_estimate(result, timeline)
    translation = _fmt_vector(final_estimate.translation_m)
    quaternion = _fmt_vector(final_estimate.rotation_quat_xyzw)
    return f"""
  <table>
    <tr><th>Total batches</th><td>{len(timeline.batches)}</td></tr>
    <tr><th>Pass / fail / inconclusive</th>
        <td>{timeline.accepted_batch_count} / {timeline.rejected_batch_count} /
            {timeline.inconclusive_batch_count}</td></tr>
    <tr><th>Retained for accumulation</th><td>{retained_count}</td></tr>
    <tr><th>Final gate status</th>
        <td><span class="grade {_online_gate_css_class(timeline.final_gate_status)}">
          {escape(timeline.final_gate_status.upper())}</span></td></tr>
    <tr><th>Final estimate translation (m)</th><td>{escape(translation)}</td></tr>
    <tr><th>Final estimate quaternion xyzw</th><td>{escape(quaternion)}</td></tr>
    <tr><th>Gate thresholds</th>
        <td>min_rank={thresholds['min_rank']},
            max_holdout_rmse_m={_fmt(float(thresholds['max_holdout_rmse_m']))} m,
            max_rolling_regression_m=
            {_fmt(float(thresholds['max_rolling_regression_m']))} m</td></tr>
  </table>
"""


def _online_final_estimate(
    result: CalibrationResult,
    timeline: OnlineCalibrationTimelineArtifact,
) -> TransformResult:
    transform = result.transforms.get(timeline.variable)
    if transform is not None:
        return transform
    for batch in reversed(timeline.batches):
        if batch.estimate_accepted:
            return batch.estimate
    return timeline.batches[-1].estimate


def _fmt_vector(values: list[float]) -> str:
    return "[" + ", ".join(_fmt(value) for value in values) + "]"
