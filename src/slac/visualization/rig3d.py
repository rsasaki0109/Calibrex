"""3D rig visualization for reference, online, and candidate calibration results."""

from __future__ import annotations

import json
from html import escape
from math import acos, degrees, sqrt
from pathlib import Path
from typing import Any, Literal

from slac.core.geometry import normalize_quaternion_xyzw
from slac.core.result import CalibrationResult, TransformResult

LayerName = Literal["reference", "online", "candidate"]

_LAYER_META: dict[LayerName, dict[str, str]] = {
    "reference": {"label": "Reference", "color": "#2563eb"},
    "online": {"label": "Online / Estimated", "color": "#16a34a"},
    "candidate": {"label": "Candidate", "color": "#f59e0b"},
}


def write_rig_3d_artifact(
    result: CalibrationResult,
    artifacts_dir: str | Path,
) -> Path | None:
    """Write a portable 3D rig viewer when extrinsics are available."""

    if not _has_any_extrinsics(result):
        return None
    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rig_3d.html"
    output_path.write_text(render_rig_3d_artifact(result), encoding="utf-8")
    result.artifacts.rig_3d_viewer = str(output_path)
    return output_path


def render_rig_3d_artifact(result: CalibrationResult) -> str:
    """Render a self-contained HTML artifact for rig extrinsic comparison."""

    payload = _rig_payload(result)
    fallback_svg = _fallback_svg(payload)
    delta_rows = _delta_rows(payload["deltas"])
    layer_rows = _layer_rows(payload["layers"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>slac 3D Rig View - {escape(result.run.id)}</title>
  <style>
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      font-family: system-ui, sans-serif;
      color: #172026;
      background: #f7f9fb;
    }}
    #scene {{
      position: relative;
      height: 72vh;
      min-height: 520px;
      background: #eef3f7;
      overflow: hidden;
    }}
    #scene canvas {{ display: block; width: 100%; height: 100%; }}
    .hud {{
      position: absolute;
      left: 1rem;
      top: 1rem;
      max-width: min(420px, calc(100vw - 2rem));
      padding: 0.75rem;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid #ccd6dd;
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(10, 20, 30, 0.10);
    }}
    .hud h1 {{ margin: 0 0 0.35rem; font-size: 1rem; }}
    .hud p {{ margin: 0.2rem 0; color: #53616c; font-size: 0.86rem; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.6rem; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.82rem; }}
    .swatch {{ width: 0.75rem; height: 0.75rem; border-radius: 50%; display: inline-block; }}
    .controls {{ display: flex; gap: 0.35rem; margin-top: 0.7rem; }}
    .controls button {{
      border: 1px solid #b8c5ce;
      background: #ffffff;
      color: #172026;
      border-radius: 4px;
      padding: 0.3rem 0.55rem;
      font-weight: 700;
      cursor: pointer;
    }}
    .controls button.active {{ background: #172026; color: #ffffff; }}
    #fallback {{
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 1rem;
    }}
    #fallback svg {{
      width: min(920px, 100%);
      height: auto;
      background: #f7f9fb;
      border: 1px solid #ccd6dd;
    }}
    main {{ max-width: 1120px; margin: 1.5rem auto 2.5rem; padding: 0 1rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; background: #ffffff; }}
    th, td {{ border: 1px solid #d6dce1; padding: 0.5rem; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f4; }}
    code {{ background: #eef2f4; padding: 0.1rem 0.25rem; border-radius: 3px; }}
  </style>
</head>
<body>
<section id="scene" aria-label="3D rig extrinsic comparison">
  <div class="hud">
    <h1>3D Calibration Rig View</h1>
    <p>Reference, online estimate, and candidate extrinsics are rendered in
       the same parent frame.</p>
    <p>Delta gain only exaggerates differences from the matched reference for visibility.</p>
    <div class="legend">
      <span><i class="swatch" style="background:#2563eb"></i>Reference</span>
      <span><i class="swatch" style="background:#16a34a"></i>Online / Estimated</span>
      <span><i class="swatch" style="background:#f59e0b"></i>Candidate</span>
    </div>
    <div class="controls" aria-label="delta gain">
      <button data-gain="1">x1</button>
      <button class="active" data-gain="10">x10</button>
      <button data-gain="50">x50</button>
    </div>
  </div>
  <div id="fallback">{fallback_svg}</div>
</section>
<main>
  <h2>Matched Deltas</h2>
  <table>
    <tr>
      <th>Layer</th><th>Transform</th><th>Reference</th><th>Edge</th>
      <th>Translation Delta m</th><th>Rotation Delta deg</th>
    </tr>
    {delta_rows}
  </table>
  <h2>Rendered Transforms</h2>
  <table>
    <tr><th>Layer</th><th>Name</th><th>Parent</th><th>Child</th><th>Translation m</th></tr>
    {layer_rows}
  </table>
</main>
<script id="rig-data" type="application/json">{_json_script(payload)}</script>
<script type="importmap">
  {{
    "imports": {{
      "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"
    }}
  }}
</script>
<script type="module">
  import * as THREE from 'three';
  import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

  const data = JSON.parse(document.getElementById('rig-data').textContent);
  const sceneEl = document.getElementById('scene');
  const fallback = document.getElementById('fallback');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef3f7);

  const camera = new THREE.PerspectiveCamera(
    50,
    sceneEl.clientWidth / sceneEl.clientHeight,
    0.01,
    1000
  );
  camera.position.set(4.0, -6.0, 3.5);
  const renderer = new THREE.WebGLRenderer({{ antialias: true, preserveDrawingBuffer: true }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
  sceneEl.appendChild(renderer.domElement);
  fallback.style.display = 'none';

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.6);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb6c0ca, 2.2));
  const grid = new THREE.GridHelper(8, 16, 0x8fa0ac, 0xc6d1d8);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
  scene.add(new THREE.AxesHelper(1.0));

  const root = new THREE.Group();
  scene.add(root);
  let deltaGain = 10;
  const referenceByEdge = new Map();
  for (const layer of data.layers) {{
    if (layer.name !== 'reference') continue;
    for (const transform of layer.transforms) {{
      referenceByEdge.set(`${{transform.parent}}->${{transform.child}}`, transform);
    }}
  }}

  function clearRig() {{
    while (root.children.length) root.remove(root.children[0]);
  }}

  function vectorFor(transform, layerName) {{
    const base = transform.translation_m;
    if (layerName === 'reference') return new THREE.Vector3(base[0], base[1], base[2]);
    const reference = referenceByEdge.get(`${{transform.parent}}->${{transform.child}}`);
    if (!reference) return new THREE.Vector3(base[0], base[1], base[2]);
    return new THREE.Vector3(
      reference.translation_m[0] + (base[0] - reference.translation_m[0]) * deltaGain,
      reference.translation_m[1] + (base[1] - reference.translation_m[1]) * deltaGain,
      reference.translation_m[2] + (base[2] - reference.translation_m[2]) * deltaGain
    );
  }}

  function drawAxisFrame(parent, transform, color, layerName) {{
    const group = new THREE.Group();
    group.position.copy(vectorFor(transform, layerName));
    const q = transform.rotation_quat_xyzw;
    group.quaternion.set(q[0], q[1], q[2], q[3]);
    parent.add(group);

    group.add(
      new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.45, 0xef4444, 0.10, 0.06
      )
    );
    group.add(
      new THREE.ArrowHelper(
        new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.45, 0x22c55e, 0.10, 0.06
      )
    );
    group.add(
      new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.45, 0x3b82f6, 0.10, 0.06
      )
    );

    const material = new THREE.MeshStandardMaterial({{ color, roughness: 0.55, metalness: 0.05 }});
    const geometry = transform.sensor_kind === 'camera'
      ? new THREE.ConeGeometry(0.16, 0.32, 4)
      : transform.sensor_kind === 'lidar'
        ? new THREE.CylinderGeometry(0.18, 0.18, 0.10, 32)
        : new THREE.BoxGeometry(0.22, 0.16, 0.12);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 0, 0);
    group.add(mesh);
  }}

  function renderRig() {{
    clearRig();
    for (const layer of data.layers) {{
      const color = Number.parseInt(layer.color.replace('#', '0x'));
      for (const transform of layer.transforms) {{
        drawAxisFrame(root, transform, color, layer.name);
        const reference = referenceByEdge.get(`${{transform.parent}}->${{transform.child}}`);
        if (layer.name !== 'reference' && reference) {{
          const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(
              reference.translation_m[0],
              reference.translation_m[1],
              reference.translation_m[2]
            ),
            vectorFor(transform, layer.name)
          ]);
          const line = new THREE.Line(
            lineGeometry,
            new THREE.LineBasicMaterial({{ color, linewidth: 2 }})
          );
          root.add(line);
        }}
      }}
    }}
  }}

  function animate() {{
    controls.update();
    renderer.render(scene, camera);
    window.requestAnimationFrame(animate);
  }}

  renderRig();
  animate();

  for (const button of document.querySelectorAll('[data-gain]')) {{
    button.addEventListener('click', () => {{
      deltaGain = Number(button.dataset.gain);
      for (const item of document.querySelectorAll('[data-gain]')) item.classList.remove('active');
      button.classList.add('active');
      renderRig();
    }});
  }}

  window.addEventListener('resize', () => {{
    camera.aspect = sceneEl.clientWidth / sceneEl.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(sceneEl.clientWidth, sceneEl.clientHeight);
  }});
</script>
</body>
</html>
"""


def _has_any_extrinsics(result: CalibrationResult) -> bool:
    return bool(result.transforms or result.reference_extrinsics or result.candidate_extrinsics)


def _rig_payload(result: CalibrationResult) -> dict[str, Any]:
    layers = [
        _layer_payload("reference", result.reference_extrinsics),
        _layer_payload("online", result.transforms),
        _layer_payload("candidate", result.candidate_extrinsics),
    ]
    layers = [layer for layer in layers if layer["transforms"]]
    return {
        "run_id": result.run.id,
        "layers": layers,
        "deltas": _matched_deltas(result),
    }


def _layer_payload(
    layer: LayerName,
    transforms: dict[str, TransformResult],
) -> dict[str, Any]:
    meta = _LAYER_META[layer]
    return {
        "name": layer,
        "label": meta["label"],
        "color": meta["color"],
        "transforms": [
            {
                "name": name,
                "parent": transform.parent,
                "child": transform.child,
                "translation_m": list(transform.translation_m),
                "rotation_quat_xyzw": list(transform.rotation_quat_xyzw),
                "sensor_kind": _sensor_kind(transform.child),
            }
            for name, transform in sorted(transforms.items())
        ],
    }


def _matched_deltas(result: CalibrationResult) -> list[dict[str, Any]]:
    references_by_edge = {
        (transform.parent, transform.child): (name, transform)
        for name, transform in sorted(result.reference_extrinsics.items())
    }
    deltas: list[dict[str, Any]] = []
    layers: list[tuple[LayerName, dict[str, TransformResult]]] = [
        ("online", result.transforms),
        ("candidate", result.candidate_extrinsics),
    ]
    for layer, transforms in layers:
        for name, transform in sorted(transforms.items()):
            reference = references_by_edge.get((transform.parent, transform.child))
            if reference is None:
                continue
            reference_name, reference_transform = reference
            deltas.append(
                {
                    "layer": layer,
                    "layer_label": _LAYER_META[layer]["label"],
                    "transform": name,
                    "reference": reference_name,
                    "edge": f"{transform.parent} -> {transform.child}",
                    "translation_delta_m": _translation_delta_m(transform, reference_transform),
                    "rotation_delta_deg": _rotation_delta_deg(transform, reference_transform),
                }
            )
    return deltas


def _delta_rows(deltas: list[dict[str, Any]]) -> str:
    if not deltas:
        return (
            '<tr><td colspan="6">'
            "No online or candidate transform matched a reference edge."
            "</td></tr>"
        )
    rows = []
    for delta in deltas:
        rows.append(
            "<tr>"
            f"<td>{escape(str(delta['layer_label']))}</td>"
            f"<td><code>{escape(str(delta['transform']))}</code></td>"
            f"<td><code>{escape(str(delta['reference']))}</code></td>"
            f"<td>{escape(str(delta['edge']))}</td>"
            f"<td>{float(delta['translation_delta_m']):.6g}</td>"
            f"<td>{float(delta['rotation_delta_deg']):.6g}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _layer_rows(layers: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for layer in layers:
        for transform in layer["transforms"]:
            rows.append(
                "<tr>"
                f"<td>{escape(str(layer['label']))}</td>"
                f"<td><code>{escape(str(transform['name']))}</code></td>"
                f"<td>{escape(str(transform['parent']))}</td>"
                f"<td>{escape(str(transform['child']))}</td>"
                f"<td>{escape(str(transform['translation_m']))}</td>"
                "</tr>"
            )
    return "\n".join(rows) or '<tr><td colspan="5">No transforms were recorded.</td></tr>'


def _fallback_svg(payload: dict[str, Any]) -> str:
    circles: list[str] = []
    labels: list[str] = []
    for layer in payload["layers"]:
        color = str(layer["color"])
        for transform in layer["transforms"]:
            x, y = _project_iso(transform["translation_m"])
            circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{escape(color)}" />')
            labels.append(
                f'<text x="{x + 10:.1f}" y="{y - 8:.1f}" font-size="12" fill="#172026">'
                f"{escape(str(transform['child']))}</text>"
            )
    return f"""<svg viewBox="0 0 960 520" role="img" aria-label="Fallback 3D rig view">
      <rect width="960" height="520" fill="#f7f9fb" />
      <line x1="120" y1="390" x2="820" y2="390" stroke="#c6d1d8" />
      <line x1="480" y1="455" x2="480" y2="80" stroke="#c6d1d8" />
      <circle cx="480" cy="360" r="10" fill="#172026" />
      <text x="492" y="352" font-size="12" fill="#172026">root</text>
      {''.join(circles)}
      {''.join(labels)}
    </svg>"""


def _project_iso(translation: list[float]) -> tuple[float, float]:
    x, y, z = (float(translation[0]), float(translation[1]), float(translation[2]))
    return (480.0 + (x * 120.0) - (y * 55.0), 360.0 - (z * 120.0) + (y * 38.0))


def _json_script(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True).replace("<", "\\u003c")


def _sensor_kind(child: str) -> str:
    lowered = child.lower()
    if "cam" in lowered:
        return "camera"
    if "lidar" in lowered or "velo" in lowered:
        return "lidar"
    if "radar" in lowered:
        return "radar"
    if "imu" in lowered:
        return "imu"
    return "sensor"


def _translation_delta_m(left: TransformResult, right: TransformResult) -> float:
    return sqrt(
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
            for left_value, right_value in zip(left_quat, right_quat, strict=True)
        )
    )
    return degrees(2.0 * acos(max(-1.0, min(1.0, dot))))
