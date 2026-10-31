"""Reproducible synthetic default-shape smoke; zero model calls or real frames."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tools.cofii2p_input_boundary import Camera, FileRef, Initializer, Request, Strict, prepare
from tools.cofii2p_model_bridge import (
    BridgeOptions,
    build_bridge,
    decode_fine_pixels,
    digest,
)


class SmokeResult(Strict):
    scope: Literal["synthetic geometry only; no official parity or accuracy claim"]
    source_commit: str
    smoke_driver: FileRef
    bridge_manifests: list[FileRef]
    stage_counts: list[int]
    unique_voxels_per_stage: list[int]
    checks_passed: list[str]
    legacy_different_patch_positions: int
    real_frames: Literal[0]
    model_invocations: Literal[0]


def run_smoke(root: Path) -> SmokeResult:
    root.mkdir(parents=True, exist_ok=False)
    source = root / "inputs"
    source.mkdir()
    pixels = np.zeros((375, 1242, 3), np.uint8)
    pixels[:, :, 0] = 128
    assert cv2.imwrite(str(source / "image.png"), pixels)
    grid = np.indices((16, 16, 16)).reshape(3, -1).T
    points = grid * 0.3 + [-2, -2, 5]
    ids = np.arange(len(points)) * 3 + 10000
    np.savez_compressed(
        source / "cloud.npz",
        point_lidar_m=points,
        intensity=np.linspace(0.1, 0.9, len(points))[:, None],
        normal_lidar=np.tile([1.0, 0, 0], (len(points), 1)),
        point_id=ids,
    )
    transform = np.array([[0.0, -1, 0, 0.2], [1, 0, 0, -0.1], [0, 0, 1, 0.3], [0, 0, 0, 1]])
    (source / "initializer.json").write_text(
        Initializer(
            schema_version="calibrex.explicit_initial_transform/v0.1",
            transform_camera_lidar=transform.tolist(),
            provenance="synthetic declared transform",
        ).model_dump_json(indent=2)
    )
    (source / "camera.json").write_text(
        Camera(
            width=1242,
            height=375,
            intrinsic_matrix=[[720, 0, 620], [0, 720, 180], [0, 0, 1]],
        ).model_dump_json(indent=2)
    )
    refs = {
        name: FileRef(path=str(source / file), sha256=digest(source / file))
        for name, file in [
            ("image", "image.png"),
            ("cloud", "cloud.npz"),
            ("initializer", "initializer.json"),
            ("camera", "camera.json"),
        ]
    }
    request = Request(
        schema_version="calibrex.cofii2p_input/v0.1",
        frame_id="synthetic-grid4096",
        source_crop_top_rows=50,
        **refs,
    )
    (source / "request.json").write_text(request.model_dump_json(indent=2))
    prepare(source / "request.json", root / "prepared")
    options = BridgeOptions(seed=20261030)
    a = build_bridge(root / "prepared/manifest.json", root / "bridge_a", options)
    b = build_bridge(root / "prepared/manifest.json", root / "bridge_b", options)
    unique = []
    counts = []
    with np.load(a.artifact.path) as x, np.load(b.artifact.path) as y:
        for name in x.files:
            np.testing.assert_array_equal(x[name], y[name])
        np.testing.assert_array_equal(x["source_point_id"], ids)
        rows = x["voxel_member_source_rows"]
        np.testing.assert_array_equal(np.sort(rows), np.arange(len(points)))
        assert np.all(np.diff(x["voxel_member_offsets"]) == 1)
        np.testing.assert_allclose(x["voxel_point_lidar_m"], points[rows])
        np.testing.assert_allclose(
            x["voxel_point_initial_camera_m"], points[rows] @ transform[:3, :3].T + transform[:3, 3]
        )
        for level in range(5):
            n = options.count // 2**level
            p = x[f"points_{level}"]
            assert p.shape == (n, 3)
            counts.append(n)
            unique.append(len(np.unique(x[f"voxel_index_{level}"])))
            np.testing.assert_allclose(
                p, x["voxel_point_initial_camera_m"][x[f"voxel_index_{level}"]], atol=1e-6
            )
            pairs = [("neighbors", p, p)]
            if level < 4:
                sub = x[f"points_{level + 1}"]
                pairs += [("subsampling", p, sub), ("upsampling", sub, p)]
            for label, support, query in pairs:
                indices = x[f"{label}_{level}"]
                assert indices.shape == (len(query), 128)
                assert indices.min() >= 0 and indices.max() < len(support)
                for row in [0, len(query) // 2, len(query) - 1]:
                    all_dist = np.linalg.norm(support.astype(float) - query[row], axis=1)
                    np.testing.assert_allclose(
                        np.sort(all_dist[indices[row]]), np.sort(all_dist)[:128], atol=1e-10
                    )
            if level:
                np.testing.assert_array_equal(
                    x[f"voxel_index_{level}"], x[f"voxel_index_{level - 1}"][x[f"parent_{level}"]]
                )
    centers, indices = np.tile([12, 20], (16, 1)), np.arange(16)
    corrected = decode_fine_pixels(centers, indices, layout="row_major")
    legacy = decode_fine_pixels(centers, indices, layout="official_eval_legacy")
    different = int(np.count_nonzero(np.any(corrected != legacy, axis=1)))
    assert different == 12
    result = SmokeResult(
        scope="synthetic geometry only; no official parity or accuracy claim",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        smoke_driver=FileRef(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__))),
        bridge_manifests=[
            FileRef(
                path=str(root / folder / "manifest.json"),
                sha256=digest(root / folder / "manifest.json"),
            )
            for folder in ["bridge_a", "bridge_b"]
        ],
        stage_counts=counts,
        unique_voxels_per_stage=unique,
        checks_passed=[
            "repeat arrays exact",
            "all original rows/IDs retained",
            "centroid transform",
            "all level shapes",
            "level-to-source lineage",
            "all KNN index ranges and shapes",
            "39 KNN query distances match exhaustive search",
            "explicit patch decoding difference",
        ],
        legacy_different_patch_positions=different,
        real_frames=0,
        model_invocations=0,
    )
    (root / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    (root / "result.schema.json").write_text(
        json.dumps(SmokeResult.model_json_schema(), indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(run_smoke(args.output).model_dump_json(indent=2))
