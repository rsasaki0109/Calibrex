"""Offline HTML comparison and deterministic held-out image overlays."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import numpy as np

from calibrex.core.benchmark import BenchmarkArtifact
from calibrex.core.geometry import SE3
from calibrex.core.result import CalibrationResult
from calibrex.evaluation.borer_rotation_benchmark import LoadedCameraLidarProblem
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthObservation,
    project_depth_pairs,
)


def write_public_six_dof_report(
    output: Path,
    *,
    loaded: LoadedCameraLidarProblem,
    provider_base: Path,
    benchmark: BenchmarkArtifact,
    results: dict[str, CalibrationResult],
    numerical_parity: bool,
    preview_ids: list[str],
) -> Path:
    """Write an offline report bound to benchmark provenance and verified PNGs.

    SVGs sample at most 1,200 finite, visible depth pairs for display. Solving
    uses the entire point cloud. The NumPy projection defines all overlays,
    independently of the backend being compared.
    """

    labels = {method.method_id: method.label for method in benchmark.methods}
    rows = []
    for trial in benchmark.trials:
        result = results.get(trial.method_id)
        solver = result.run.provenance["solver_result"] if result else {}
        final_mi = result.metrics["final_d2d_mutual_information"] if result else None
        initial_mi = result.metrics["initial_d2d_mutual_information"] if result else None
        status_note = trial.failure_reason or (
            "quality: " + result.quality.grade if result else "no result"
        )
        rows.append(
            f"<tr><th>{html.escape(labels[trial.method_id])}</th>"
            f"<td>{html.escape(str(solver.get('status', 'failed')))}"
            f"<br><small>{html.escape(status_note)}</small></td>"
            f"<td>{solver.get('evaluation_count', '—')}</td>"
            f"<td>{_number(trial.metrics.get('initial_rotation_difference_deg'))} → "
            f"{_number(trial.metrics.get('rotation_difference_deg'))}°</td>"
            f"<td>{_number(trial.metrics.get('initial_translation_difference_m'))} → "
            f"{_number(trial.metrics.get('translation_difference_m'))} m</td>"
            f"<td>{_number(initial_mi.train if initial_mi else None)} → "
            f"{_number(final_mi.train if final_mi else None)}</td>"
            f"<td>{_number(initial_mi.holdout if initial_mi else None)} → "
            f"{_number(final_mi.holdout if final_mi else None)}</td></tr>"
        )
    timing_rows = []
    for method_id, label in labels.items():
        result = results.get(method_id)
        timing = result.run.provenance["timing_seconds"] if result else {}
        timing_rows.append(
            f"<tr><th>{html.escape(label)}</th>"
            + "".join(
                f"<td>{_number(timing.get(name))}</td>"
                for name in (
                    "input_verification_and_loading",
                    "backend_preparation",
                    "solve",
                    "holdout_evaluation",
                )
            )
            + f"<td>{_number(sum(timing.values()) if timing else None)}</td></tr>"
        )
    successful = numerical_parity and all(trial.status == "success" for trial in benchmark.trials)
    ratio = ""
    if successful:
        cpu, candidate = benchmark.trials
        if candidate.metrics["solve_seconds"] > 0:
            solve_ratio = cpu.metrics["solve_seconds"] / candidate.metrics["solve_seconds"]
            ratio = f"Solve-time ratio (CPU / candidate): {solve_ratio:.2f}x. "
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    depth_by_id = {item.frame_id: item for item in loaded.depth_provider.observations}
    observations = {item.frame_id: item for item in loaded.observations}
    initial = loaded.problem.initial_transform_camera_lidar.as_se3()
    previews = []
    for frame_id in preview_ids:
        depth = depth_by_id[frame_id]
        image_bytes = (provider_base / depth.image.path).read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != depth.image.sha256:
            raise ValueError("source image digest changed before overlay rendering")
        image_name = f"assets/frame-{frame_id}.png"
        (output / image_name).write_bytes(image_bytes)
        panels = [_panel("Declared initial pose", image_name, observations[frame_id], initial)]
        for method_id, label in labels.items():
            result = results.get(method_id)
            pose = (
                next(iter(result.transforms.values())).as_se3()
                if result and result.transforms
                else None
            )
            panels.append(_panel(label, image_name, observations[frame_id], pose))
        previews.append(
            f"<section><h3>Holdout frame {html.escape(frame_id)}</h3>"
            f"<div class='panels'>{''.join(panels)}</div></section>"
        )
    links = [
        "source.problem.json",
        "demo.problem.json",
        "comparison.definition.json",
        "comparison.benchmark.json",
        *(f"{method_id}.result.json" for method_id in results),
    ]
    limitations = "".join(f"<li>{html.escape(note)}</li>" for note in benchmark.limitations)
    artifact_links = "".join(
        f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in links
    )
    completion = "Complete comparison" if successful else "Incomplete comparison — see status below"
    parity_label = "PASS" if numerical_parity else "FAIL / unavailable"
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KITTI raw · CPU/candidate six-DoF comparison</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f1f5f9;color:#18283d;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1450px;margin:auto;padding:32px 24px}}
h1{{font-size:30px;line-height:1.25}}h2{{margin:28px 0 12px}}
section,.card{{background:white;border:1px solid #d8e1ea;border-radius:12px;
padding:18px;margin:20px 0}}
.badge{{display:inline-block;padding:4px 12px;border-radius:24px;
background:#dceef5;color:#18566d;font-weight:650}}
.warning{{background:#fff0d6;color:#774709}}.table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
td,th{{text-align:left;padding:12px;border-bottom:1px solid #e4eaf0;vertical-align:top}}
small,figcaption{{color:#53667b;font-size:12px}}
.panels{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}
figure{{margin:0}}h4{{margin:0 0 8px}}
.image{{position:relative;background:#101a27;line-height:0}}.image img{{width:100%;height:auto}}
.image svg{{position:absolute;inset:0;width:100%;height:100%}}
.empty{{padding:35px;background:#e9eef4;line-height:1.6}}a{{color:#12678f}}code{{overflow-wrap:anywhere}}
@media(max-width:800px){{.panels{{grid-template-columns:1fr}}
main{{padding:20px 14px}}h1{{font-size:25px}}}}
</style></head><body><main>
<p>CALIBREX · PUBLIC DATA DEMO</p><h1>KITTI raw: CPU / candidate six-DoF calibration</h1>
<p>{html.escape(loaded.problem.sequence_id)} · 25 frames · 20 fit / 5 holdout</p>
<span class="badge{" warning" if not successful else ""}">{completion}</span>
<span class="badge{" warning" if not numerical_parity else ""}">
Exact numerical parity: {parity_label}</span>
<div class="card"><p>{html.escape(benchmark.protocol.initial_estimate_policy)}</p>
<p>Reference differences below compare the final estimate to dataset-provided calibration.
Convergence does not prove accuracy; quality remains a warning.</p></div>
<h2>Calibration and holdout evidence</h2><div class="table-wrap"><table><thead><tr>
<th>Method</th><th>Status</th><th>Evaluations</th><th>Rotation difference (initial → final)</th>
<th>Translation difference (initial → final)</th><th>Fit MI (nat)</th><th>Holdout MI (nat)</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<h2>Timing per invocation (seconds)</h2><div class="table-wrap"><table><thead><tr>
<th>Method</th><th>Load / verify</th><th>Prepare</th><th>Solve</th><th>Holdout</th><th>Total</th>
</tr></thead><tbody>{"".join(timing_rows)}</tbody></table></div>
<p>{ratio}These are single invocations; CUDA solve time includes lazy compilation.
This is not a repeated latency benchmark.</p>
<h2>Held-out image overlays</h2>
<p>Points are finite, visible camera/LiDAR depth pairs after z-buffering.
Display samples at most 1,200 pairs; optimization uses full scans.
Colour is fixed LiDAR range: blue (0 m) → red (80 m).</p>
{"".join(previews)}
<details><summary>Limitations and provenance</summary><ul>{limitations}</ul>
<p>Provider SHA256: <code>{loaded.depth_provider_sha256}</code></p>
<p>Problem SHA256: <code>{loaded.problem_sha256}</code></p>
<p>{html.escape(benchmark.provenance.generator_version)}</p></details>
<h2>Input and result artifacts</h2><ul>{artifact_links}</ul>
</main></body></html>"""
    path = output / "report.html"
    path.write_text(body, encoding="utf-8")
    return path


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.5f}"


def _panel(title: str, image: str, observation: DepthToDepthObservation, pose: SE3 | None) -> str:
    heading = f"<figure><h4>{html.escape(title)}</h4>"
    if pose is None:
        return heading + "<div class='empty'>No output pose available.</div></figure>"
    pairs = project_depth_pairs(observation, pose)
    count = pairs.pixel_u.size
    selected = np.linspace(0, count - 1, min(count, 1200), dtype=np.int64) if count else []
    circles = []
    for index in selected:
        hue = 240 * (1 - min(float(pairs.lidar_range_m[index]) / 80, 1))
        circles.append(
            f"<circle cx='{int(pairs.pixel_u[index])}' cy='{int(pairs.pixel_v[index])}' "
            f"r='1.5' fill='hsl({hue:.1f},95%,60%)' fill-opacity='.8'/>"
        )
    width, height = observation.camera.width, observation.camera.height
    return (
        heading + f"<div class='image'><img src='{html.escape(image)}' "
        f"alt='KITTI held-out image {html.escape(observation.frame_id)}' "
        f"width='{width}' height='{height}'>"
        f"<svg viewBox='0 0 {width} {height}' aria-label='Projected depth pairs'>"
        f"{''.join(circles)}</svg></div><figcaption>Showing {len(circles):,} of {count:,} "
        "visible pairs</figcaption></figure>"
    )
