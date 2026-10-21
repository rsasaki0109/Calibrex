"""Small, reproducible development inputs; no downloads and no accuracy claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.evaluation.camera_lidar_edge import (
    REP103_OPTICAL_FROM_BODY,
    EdgeCamera,
    PreparedFrame,
)


def prepare(output: Path, a2d2_root: Path | None = None) -> None:
    """Write new pickle-free fixtures with input/source provenance embedded in each NPZ."""

    if output.exists():
        raise ValueError("output already exists; preserve earlier fixtures")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "src/calibrex/evaluation/camera_lidar_edge.py",
        root / "src/calibrex/solvers/native_levinson_thrun_online_solver.py",
        root / "src/calibrex/solvers/levinson_online_preparation.py",
        root / "src/calibrex/solvers/levinson_thrun_online_solver.py",
    ]
    provenance: dict[str, Any] = {
        "schema_version": "calibrex.browser_webgpu_edge_fixture/v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip(),
        "generator": "tools.prepare_browser_webgpu_edge_fixtures",
        "sources": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "accuracy_claim": False,
    }

    def save(
        name: str,
        frames: list[PreparedFrame],
        camera: EdgeCamera,
        start: np.ndarray,
        origin: dict[str, Any],
    ) -> None:
        if not frames or not sum(len(frame.points) for frame in frames):
            raise ValueError("cannot benchmark an empty prepared objective")
        record = {**provenance, "origin": origin}
        payload: dict[str, Any] = {
            "frames": len(frames),
            "intrinsics": [camera.fx, camera.fy, camera.cx, camera.cy],
            "distortion_model": camera.distortion_model,
            "distortion": camera.distortion,
            "start": start,
            "provenance_json": json.dumps(record, sort_keys=True),
        }
        for index, frame in enumerate(frames):
            for key in ("edge", "points", "weights", "scale"):
                payload[f"{key}_{index}"] = getattr(frame, key)
        np.savez_compressed(output / (name + ".npz"), **payload)

    if a2d2_root is not None:
        from calibrex.solvers.native_levinson_thrun_online_solver import _load_a2d2_frames

        source_frames, inputs, preparation = _load_a2d2_frames(a2d2_root.resolve(), {})
        source_camera = source_frames[0].camera
        camera = EdgeCamera(
            source_camera.fx,
            source_camera.fy,
            source_camera.cx,
            source_camera.cy,
            "equidistant",
            source_camera.distortion,
        )
        frames = [
            PreparedFrame(
                time_s=float(index),
                gap_s=0,
                edge=frame.edge_response.astype(np.float32),
                scale=1,
                points=(frame.lidar_points @ REP103_OPTICAL_FROM_BODY.T).astype(np.float32),
                weights=frame.discontinuity_weights.astype(np.float32),
                rings=0,
            )
            for index, frame in enumerate(source_frames)
        ]
        save(
            "a2d2-spent",
            frames,
            camera,
            np.eye(4),
            {
                "kind": "spent public A2D2 camera-registered points; objective timing only",
                "dataset_license": "CC-BY-ND-4.0",
                "preparation": preparation,
                "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "conversion": (
                    "all prepared points, weights and images cast to float32; body to optical axes"
                ),
                "reference_use": (
                    "registered camera-view cloud; no independent extrinsic accuracy claim"
                ),
            },
        )
    for name, count, points_per_frame, model, distortion in (
        ("synthetic-48-frames", 48, 4000, "none", ()),
        (
            "synthetic-radtan",
            2,
            37,
            "radtan",
            (0.08, -0.02, 0.002, -0.003, 0.001, 0.001, -0.0002, 0.00001),
        ),
        ("synthetic-equidistant", 2, 37, "equidistant", (0.02, -0.001, 0.0001, -0.00001)),
    ):
        rng = np.random.default_rng(20261021)
        synthetic = []
        for index in range(count):
            height, width = (240 + index % 3, 320 + index % 2) if count > 2 else (17, 19)
            yy, xx = np.mgrid[:height, :width]
            edge = (np.sin(xx * 0.1 + index) + np.cos(yy * 0.15 - index) * 0.7).astype(np.float32)
            points = rng.uniform(-8, 8, size=(points_per_frame, 3)).astype(np.float32)
            points[:, 2] = rng.uniform(2, 40, size=points_per_frame)
            points[0] = [0, 0, 0]
            points[1] = [0, 0, -1]
            points[2] = [0, 0, 10]
            synthetic.append(
                PreparedFrame(
                    time_s=float(index),
                    gap_s=0,
                    edge=edge,
                    scale=1 + index % 2,
                    points=points,
                    weights=rng.uniform(0.01, 4, size=points_per_frame).astype(np.float32),
                    rings=0,
                )
            )
        synthetic_camera = EdgeCamera(
            *(280, 285, 320, 240) if count > 2 else (12, 14, 9, 8),
            distortion_model=model,
            distortion=distortion,
        )
        start = np.eye(4)
        start[:3, :3] = Rotation.from_euler("xyz", [1, -2, 3], degrees=True).as_matrix()
        start[:3, 3] = [0.1, -0.15, 0.2]
        save(
            name,
            synthetic,
            synthetic_camera,
            start,
            {
                "kind": "synthetic arithmetic/stress fixture; not calibration accuracy evidence",
                "seed": 20261021,
                "frames": count,
                "points_per_frame": points_per_frame,
            },
        )
    print(json.dumps({"fixtures": sorted(p.name for p in output.glob("*.npz"))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--a2d2-root", type=Path)
    args = parser.parse_args()
    prepare(args.out, args.a2d2_root)
