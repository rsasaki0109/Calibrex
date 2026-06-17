"""Camera-LiDAR overlay visualization artifact scaffolding."""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any

from calibrex.core.result import CalibrationResult, MetricResult
from calibrex.data.kitti import (
    KITTILidarCameraProjectedPoint,
    KITTILidarCameraProjection,
    project_velodyne_to_camera,
)


def write_camera_lidar_overlay_artifact(
    result: CalibrationResult,
    artifacts_dir: str | Path,
) -> Path | None:
    """Write a portable camera-LiDAR overlay HTML scaffold when metrics exist."""

    if not _has_lidar_camera_metrics(result):
        return None
    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "camera_lidar_overlay.html"
    output_path.write_text(render_camera_lidar_overlay_artifact(result), encoding="utf-8")
    result.artifacts.camera_lidar_overlay = str(output_path)
    return output_path


def render_camera_lidar_overlay_artifact(result: CalibrationResult) -> str:
    """Render a self-contained HTML artifact for camera-LiDAR overlay diagnostics."""

    readiness = result.metrics.get("lidar_camera_overlay_readiness")
    score = result.metrics.get("lidar_camera_overlay_score")
    pairs = result.metrics.get("lidar_camera_transform_pairs")
    mutual_information = result.metrics.get("lidar_camera_mutual_information_score")
    frame_pairs = _camera_lidar_frame_pairs(result)
    rows = "\n".join(
        _metric_row(name, metric)
        for name, metric in [
            ("lidar_camera_transform_pairs", pairs),
            ("lidar_camera_overlay_readiness", readiness),
            ("lidar_camera_overlay_score", score),
            ("lidar_camera_mutual_information_score", mutual_information),
        ]
        if metric is not None
    )
    pair_rows = "\n".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{escape(transform.parent)}</td>"
        f"<td>{escape(transform.child)}</td>"
        f"<td>{escape(_fmt_vector(transform.translation_m))}</td>"
        f"<td>{escape(_fmt_vector(transform.rotation_quat_xyzw))}</td>"
        "</tr>"
        for name, transform in sorted(result.transforms.items())
        if _is_camera_lidar_related(transform.parent, transform.child)
    )
    frame_pair_rows = "\n".join(_frame_pair_row(pair) for pair in frame_pairs)
    empty_frame_pair_rows = (
        '<tr><td colspan="6">No concrete camera-LiDAR frame pairs were recorded.</td></tr>'
    )
    projection = _kitti_projection(result, frame_pairs)
    projection_html = _projection_overlay_html(projection)
    score_value = _metric_value(score)
    readiness_grade = readiness.grade if readiness is not None else "warn"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera-LiDAR Overlay - {escape(result.run.id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #182026; }}
    main {{ max-width: 980px; margin: 0 auto; }}
    .grade {{
      display: inline-block;
      padding: 0.2rem 0.45rem;
      border-radius: 4px;
      font-weight: 700;
    }}
    .pass {{ background: #d9f0e3; color: #125c35; }}
    .warn {{ background: #fff0c2; color: #6a4a00; }}
    .fail {{ background: #ffd9d9; color: #7a1616; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d6dce1; padding: 0.5rem; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f4; }}
    td.path {{ word-break: break-all; }}
    svg {{ width: 100%; height: auto; border: 1px solid #d6dce1; background: #f7f9fa; }}
    .overlay-frame {{
      position: relative;
      width: 100%;
      border: 1px solid #d6dce1;
      background: #111820;
    }}
    .overlay-frame img {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: fill;
    }}
    .overlay-frame svg {{ position: relative; display: block; border: 0; background: transparent; }}
    .caption {{ color: #52616b; }}
  </style>
</head>
<body>
<main>
  <h1>Camera-LiDAR Overlay</h1>
  <p>
    Overlay readiness:
    <span class="grade {escape(readiness_grade)}">{escape(readiness_grade.upper())}</span>
    &nbsp; score: <strong>{escape(_fmt_optional(score_value))}</strong>
  </p>
  {projection_html or _diagnostic_svg(score_value, readiness_grade)}
  <p class="caption">
    This alpha artifact projects a sampled KITTI Velodyne frame into the camera
    image when calibration files are present. If projection inputs are missing,
    it falls back to the schematic readiness view.
  </p>
  <h2>Metrics</h2>
  <table>
    <tr><th>Name</th><th>Grade</th><th>Value</th><th>Unit</th><th>Reason</th></tr>
    {rows or '<tr><td colspan="5">No camera-LiDAR metrics were recorded.</td></tr>'}
  </table>
  <h2>Frame Pairs</h2>
  <table>
    <tr>
      <th>Camera Index</th><th>LiDAR Index</th><th>Delta ms</th>
      <th>LiDAR Points</th><th>Camera Path</th><th>LiDAR Path</th>
    </tr>
    {frame_pair_rows or empty_frame_pair_rows}
  </table>
  <h2>Related Transforms</h2>
  <table>
    <tr>
      <th>Name</th><th>Parent</th><th>Child</th>
      <th>Translation m</th><th>Quaternion xyzw</th>
    </tr>
    {pair_rows or '<tr><td colspan="5">No camera-LiDAR related transforms were found.</td></tr>'}
  </table>
</main>
</body>
</html>
"""


def _metric_row(name: str, metric: MetricResult | None) -> str:
    if metric is None:
        return ""
    value = _metric_value(metric)
    return (
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{escape(metric.grade.upper())}</td>"
        f"<td>{escape(_fmt_optional(value))}</td>"
        f"<td>{escape(metric.unit or '')}</td>"
        f"<td>{escape(metric.reason or '')}</td>"
        "</tr>"
    )


def _frame_pair_row(pair: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{escape(_string_value(pair.get('camera_index')))}</td>"
        f"<td>{escape(_string_value(pair.get('lidar_index')))}</td>"
        f"<td>{escape(_float_value(pair.get('delta_ms')))}</td>"
        f"<td>{escape(_string_value(pair.get('lidar_point_count')))}</td>"
        f"<td class=\"path\">{escape(_string_value(pair.get('camera_path')))}</td>"
        f"<td class=\"path\">{escape(_string_value(pair.get('lidar_path')))}</td>"
        "</tr>"
    )


def _kitti_projection(
    result: CalibrationResult,
    frame_pairs: list[dict[str, Any]],
) -> KITTILidarCameraProjection | None:
    if not frame_pairs:
        return None
    dataset_type = result.run.provenance.get("dataset_type")
    if dataset_type != "kitti_raw":
        return None
    dataset_path = _dataset_path(result)
    if dataset_path is None:
        return None
    pair = frame_pairs[0]
    camera_path = pair.get("camera_path")
    lidar_path = pair.get("lidar_path")
    if not isinstance(camera_path, str) or not isinstance(lidar_path, str):
        return None
    return project_velodyne_to_camera(
        dataset_path,
        camera_path=camera_path,
        lidar_path=lidar_path,
        max_points=800,
    )


def _dataset_path(result: CalibrationResult) -> str | None:
    path = result.run.provenance.get("dataset_path")
    if isinstance(path, str):
        return path
    inspection = result.run.provenance.get("dataset_inspection")
    if not isinstance(inspection, dict):
        return None
    inspection_path = inspection.get("path")
    return inspection_path if isinstance(inspection_path, str) else None


def _projection_overlay_html(projection: KITTILidarCameraProjection | None) -> str:
    if projection is None or projection.status != "projected" or projection.image_size_px is None:
        return ""
    width, height = projection.image_size_px
    image_uri = _image_data_uri(Path(projection.camera_path))
    image_tag = (
        f'<img src="{escape(image_uri)}" alt="KITTI camera frame">'
        if image_uri is not None
        else ""
    )
    points = "\n".join(_projected_point_circle(point) for point in projection.points)
    background = "" if image_uri is not None else f'<rect width="{width}" height="{height}" />'
    return f"""
  <h2>Projected Overlay</h2>
  <div class="overlay-frame" style="aspect-ratio: {width} / {height};">
    {image_tag}
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Projected KITTI LiDAR points">
      {background}
      {points}
    </svg>
  </div>
  <p class="caption">
    Projected {projection.projected_point_count} of {projection.sampled_point_count}
    sampled Velodyne points from {escape(Path(projection.lidar_path).name)}.
  </p>
"""


def _projected_point_circle(point: KITTILidarCameraProjectedPoint) -> str:
    radius = max(1.5, min(3.5, 5.0 / (1.0 + (point.depth_m / 20.0))))
    opacity = _intensity_opacity(point.intensity)
    return (
        f'<circle cx="{point.u_px:.3f}" cy="{point.v_px:.3f}" r="{radius:.3f}" '
        f'fill="{_depth_color(point.depth_m)}" opacity="{opacity:.3f}" />'
    )


def _depth_color(depth_m: float) -> str:
    if depth_m < 10.0:
        return "#fff05a"
    if depth_m < 25.0:
        return "#35d0ba"
    return "#4f8cff"


def _intensity_opacity(intensity: float) -> float:
    return max(0.45, min(0.95, 0.45 + (float(intensity) * 0.5)))


def _image_data_uri(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > 2_500_000 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _diagnostic_svg(score: float | None, readiness_grade: str) -> str:
    normalized = 0.0 if score is None else max(0.0, min(1.0, score))
    x_position = 80.0 + (normalized * 640.0)
    color = {"pass": "#1c8c50", "warn": "#b97a00", "fail": "#b83232"}.get(
        readiness_grade,
        "#52616b",
    )
    return f"""<svg viewBox="0 0 800 360" role="img" aria-label="Camera LiDAR overlay schematic">
  <rect x="60" y="40" width="680" height="250" fill="#ffffff" stroke="#8fa1ad" />
  <line x1="60" y1="165" x2="740" y2="165" stroke="#d6dce1" />
  <line x1="400" y1="40" x2="400" y2="290" stroke="#d6dce1" />
  <polyline points="100,250 250,170 390,190 560,110 705,130"
    fill="none" stroke="#52616b" stroke-width="3" />
  <circle cx="135" cy="235" r="5" fill="#0f7fbf" />
  <circle cx="235" cy="180" r="5" fill="#0f7fbf" />
  <circle cx="345" cy="188" r="5" fill="#0f7fbf" />
  <circle cx="505" cy="135" r="5" fill="#0f7fbf" />
  <circle cx="670" cy="130" r="5" fill="#0f7fbf" />
  <line x1="80" y1="320" x2="720" y2="320"
    stroke="#8fa1ad" stroke-width="8" stroke-linecap="round" />
  <circle cx="{x_position:.1f}" cy="320" r="13" fill="{color}" />
  <text x="80" y="345" fill="#52616b" font-size="16">low</text>
  <text x="686" y="345" fill="#52616b" font-size="16">high</text>
</svg>"""


def _has_lidar_camera_metrics(result: CalibrationResult) -> bool:
    return any(name.startswith("lidar_camera_") for name in result.metrics)


def _camera_lidar_frame_pairs(result: CalibrationResult) -> list[dict[str, Any]]:
    inspection = result.run.provenance.get("dataset_inspection")
    if not isinstance(inspection, dict):
        return []
    diagnostics = inspection.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    pairs = diagnostics.get("camera_lidar_pairs")
    if not isinstance(pairs, list):
        return []
    return [pair for pair in pairs if isinstance(pair, dict)]


def _metric_value(metric: MetricResult | None) -> float | None:
    if metric is None:
        return None
    if metric.holdout is not None:
        return metric.holdout
    if metric.value is not None:
        return metric.value
    return metric.train


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _float_value(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.6g}"
    return _string_value(value)


def _string_value(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _fmt_vector(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _is_camera_lidar_related(parent: str, child: str) -> bool:
    return (_is_camera(parent) and _is_lidar(child)) or (
        _is_lidar(parent) and _is_camera(child)
    ) or _is_camera(child) or _is_lidar(child)


def _is_camera(name: str) -> bool:
    return name.lower().startswith("camera")


def _is_lidar(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("lidar") or lowered.startswith("velodyne")
