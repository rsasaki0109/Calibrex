"""Deterministic three-plane Bull's Eye plots for rotation recovery."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.camera_lidar_artifacts import (
    BullseyePlotArtifact,
    BullseyeTrialPoint,
    CameraLidarArtifactProvenance,
    load_calibration_candidate_trace,
    load_camera_lidar_benchmark_protocol,
    load_camera_lidar_problem,
)
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path

FloatArray: TypeAlias = NDArray[np.float64]
BULLSEYE_GENERATOR_VERSION = "calibrex.bullseye_plot/v0.1"


def write_bullseye_plot(
    problem_path: str | Path,
    protocol_path: str | Path,
    trace_directory: str | Path,
    *,
    svg_path: str | Path,
    artifact_path: str | Path,
    command: tuple[str, ...] = (),
) -> BullseyePlotArtifact:
    """Render all retained trials and write a provenance-complete sidecar."""

    problem_file = Path(problem_path)
    protocol_file = Path(protocol_path)
    trace_root = Path(trace_directory)
    problem = load_camera_lidar_problem(problem_file)
    protocol = load_camera_lidar_benchmark_protocol(protocol_file)
    problem_digest = _required_digest(problem_file)
    protocol_digest = _required_digest(protocol_file)
    if problem_digest != protocol.problem_sha256:
        raise ValueError("Bull's Eye protocol does not match problem digest")
    reference = problem.reference_transform_camera_lidar.as_se3()
    perturbation_by_trial = {
        item.trial_id: item for item in protocol.perturbations
    }
    points = []
    trace_digests = {}
    for trace_file in sorted(trace_root.glob("*.trace.yaml")):
        trace = load_calibration_candidate_trace(trace_file)
        if trace.protocol_sha256 != protocol_digest:
            raise ValueError(f"trace protocol digest mismatch: {trace_file}")
        try:
            perturbation = perturbation_by_trial[trace.trial_id]
        except KeyError as exc:
            raise ValueError(f"trace trial is absent from protocol: {trace.trial_id}") from exc
        trace_digests[trace.trial_id] = _required_digest(trace_file)
        points.append(
            BullseyeTrialPoint(
                trial_id=trace.trial_id,
                initial_rotation_deg_xyz=perturbation.rotation_deg_xyz,
                output_rotation_error_deg_xyz=list(
                    relative_rotation_vector_deg(
                        trace.output_transform_camera_lidar.as_se3(),
                        reference,
                    )
                ),
                hit=trace.outcome.hit,
            )
        )
    if len(points) != protocol.perturbation_count:
        raise ValueError(
            "Bull's Eye requires one trace per protocol perturbation: "
            f"expected {protocol.perturbation_count}, observed {len(points)}"
        )
    svg_file = Path(svg_path)
    svg_file.parent.mkdir(parents=True, exist_ok=True)
    svg_file.write_text(
        _render_svg(
            points,
            maximum_deg=max(
                protocol.rotation_magnitude_deg,
                max(
                    _vector_norm(item.output_rotation_error_deg_xyz)
                    for item in points
                ),
            ),
            hit_radius_deg=protocol.hit.rotation_error_max_deg,
            protocol_id=protocol.protocol_id,
        ),
        encoding="utf-8",
    )
    svg_digest = _required_digest(svg_file)
    artifact = BullseyePlotArtifact(
        plot_id=f"{protocol.protocol_id}:bullseye",
        problem_sha256=problem_digest,
        protocol_sha256=protocol_digest,
        trace_sha256=trace_digests,
        svg_path=str(svg_file),
        svg_sha256=svg_digest,
        panels=["roll_pitch", "roll_yaw", "pitch_yaw"],
        hit_radius_deg=protocol.hit.rotation_error_max_deg,
        points=points,
        provenance=CameraLidarArtifactProvenance(
            generator="calibrex.visualization.bullseye",
            generator_version=BULLSEYE_GENERATOR_VERSION,
            git_commit=git_commit(),
            command=list(command),
            config_sha256=protocol_digest,
            source_sha256=problem_digest,
        ),
    )
    artifact.save(artifact_path)
    return artifact


def relative_rotation_vector_deg(
    candidate: SE3,
    reference: SE3,
) -> tuple[float, float, float]:
    """Return candidate-reference geodesic rotation as an axis-angle vector."""

    cx, cy, cz, cw = candidate.rotation_quat_xyzw
    rx, ry, rz, rw = reference.rotation_quat_xyzw
    inverse_reference = (-rx, -ry, -rz, rw)
    x, y, z, w = _quaternion_product(
        (cx, cy, cz, cw),
        inverse_reference,
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1.0e-15:
        return (0.0, 0.0, 0.0)
    angle_deg = math.degrees(2.0 * math.atan2(vector_norm, w))
    scale = angle_deg / vector_norm
    return (x * scale, y * scale, z * scale)


def _render_svg(
    points: list[BullseyeTrialPoint],
    *,
    maximum_deg: float,
    hit_radius_deg: float,
    protocol_id: str,
) -> str:
    width = 1080
    height = 390
    panel_size = 320
    radius = 135.0
    margin_x = 25.0
    center_y = 205.0
    limit = max(maximum_deg * 1.1, hit_radius_deg * 2.0, 1.0e-6)
    scale = radius / limit
    panels = [
        ("roll / pitch", 0, 1),
        ("roll / yaw", 0, 2),
        ("pitch / yaw", 1, 2),
    ]
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<metadata>"
        + html.escape(
            f"generator={BULLSEYE_GENERATOR_VERSION}; protocol={protocol_id}"
        )
        + "</metadata>",
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        (
            '<text x="24" y="28" font-family="sans-serif" font-size="18" '
            'font-weight="700" fill="#14212b">Rotation recovery Bull&#39;s Eye</text>'
        ),
    ]
    for panel_index, (label, axis_x, axis_y) in enumerate(panels):
        center_x = margin_x + panel_size * panel_index + panel_size / 2.0
        elements.extend(
            [
                (
                    f'<circle cx="{center_x:.3f}" cy="{center_y:.3f}" r="{radius:.3f}" '
                    'fill="#ffffff" stroke="#9aa8b4" stroke-width="1"/>'
                ),
                (
                    f'<circle cx="{center_x:.3f}" cy="{center_y:.3f}" '
                    f'r="{hit_radius_deg * scale:.3f}" fill="#e8f7ed" '
                    'stroke="#2f855a" stroke-width="1.5"/>'
                ),
                (
                    f'<line x1="{center_x - radius:.3f}" y1="{center_y:.3f}" '
                    f'x2="{center_x + radius:.3f}" y2="{center_y:.3f}" '
                    'stroke="#d4dce2"/>'
                ),
                (
                    f'<line x1="{center_x:.3f}" y1="{center_y - radius:.3f}" '
                    f'x2="{center_x:.3f}" y2="{center_y + radius:.3f}" '
                    'stroke="#d4dce2"/>'
                ),
                (
                    f'<text x="{center_x:.3f}" y="365" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="14" fill="#263845">{label}</text>'
                ),
            ]
        )
        for point in points:
            initial_x = center_x + point.initial_rotation_deg_xyz[axis_x] * scale
            initial_y = center_y - point.initial_rotation_deg_xyz[axis_y] * scale
            output_x = (
                center_x + point.output_rotation_error_deg_xyz[axis_x] * scale
            )
            output_y = (
                center_y - point.output_rotation_error_deg_xyz[axis_y] * scale
            )
            color = "#238636" if point.hit else "#cf222e"
            elements.extend(
                [
                    (
                        f'<line x1="{initial_x:.3f}" y1="{initial_y:.3f}" '
                        f'x2="{output_x:.3f}" y2="{output_y:.3f}" '
                        'stroke="#7b8790" stroke-width="0.7" opacity="0.55"/>'
                    ),
                    (
                        f'<path d="M {initial_x - 2.5:.3f} {initial_y - 2.5:.3f} '
                        f'L {initial_x + 2.5:.3f} {initial_y + 2.5:.3f} '
                        f'M {initial_x + 2.5:.3f} {initial_y - 2.5:.3f} '
                        f'L {initial_x - 2.5:.3f} {initial_y + 2.5:.3f}" '
                        'stroke="#4b5560" stroke-width="0.8"/>'
                    ),
                    (
                        f'<circle cx="{output_x:.3f}" cy="{output_y:.3f}" r="2.3" '
                        f'fill="{color}"><title>{html.escape(point.trial_id)}</title></circle>'
                    ),
                ]
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _quaternion_product(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _vector_norm(values: list[float]) -> float:
    array: FloatArray = np.asarray(values, dtype=float)
    return float(np.linalg.norm(array))


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"cannot digest Bull's Eye source: {path}")
    return digest
