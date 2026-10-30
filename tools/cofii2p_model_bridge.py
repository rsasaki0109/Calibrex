"""CPU geometry bridge with lineage; no learned inference or Open3D parity claim."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import scipy
from pydantic import Field
from scipy.spatial import cKDTree
from tools.cofii2p_input_boundary import FileRef, Prepared, Strict


class BridgeOptions(Strict):
    seed: int = Field(ge=0, le=2**32 - 1)
    voxel_size_m: float = Field(default=0.1, gt=0)
    count: int = Field(default=20480, gt=0)
    stages: int = Field(default=5, ge=1, le=10)
    neighbors: int = Field(default=128, gt=0)


class BridgeManifest(Strict):
    scope: Literal["CPU geometry only; no model inference or official numerical parity"]
    source_commit: str
    driver_sha256: str
    prepared_manifest: FileRef
    prepared_artifact: FileRef
    options: BridgeOptions
    artifact: FileRef
    array_shapes: dict[str, list[int]]
    conventions: dict[str, str]
    runtime: dict[str, str]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def voxelize_with_lineage(
    points: np.ndarray, features: np.ndarray, size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean points/features in a min-bound grid; retain all source row indices."""
    p, f = np.asarray(points, float), np.asarray(features, float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or len(p) == 0
        or f.ndim != 2
        or len(f) != len(p)
        or not np.isfinite(p).all()
        or not np.isfinite(f).all()
        or not np.isfinite(size)
        or size <= 0
    ):
        raise ValueError("finite nonempty Nx3 points, NxF features and positive size required")
    origin = p.min(axis=0) - size / 2
    coordinates = np.floor((p - origin) / size)
    if not np.isfinite(coordinates).all() or np.max(np.abs(coordinates)) >= 2**63:
        raise ValueError("voxel coordinates overflow")
    _, inverse = np.unique(coordinates.astype(np.int64), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.zeros((len(counts), 3))
    averaged = np.zeros((len(counts), f.shape[1]))
    np.add.at(means, inverse, p)
    np.add.at(averaged, inverse, f)
    means /= counts[:, None]
    averaged /= counts[:, None]
    rows = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    return means, averaged, offsets, rows, inverse


def count_indices(n: int, target: int, rng: np.random.RandomState) -> np.ndarray:
    """Explicit duplicates for scarce input, matching the loader's count rule."""
    if n <= 0 or target <= 0:
        raise ValueError("positive source/target counts required")
    if n >= target:
        return rng.choice(n, target, replace=False)
    repeats = (target - 1) // n
    return np.r_[np.tile(np.arange(n), repeats), rng.choice(n, target - repeats * n, replace=False)]


def neighbors(support: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Query-to-support KNN indices; equal-distance ordering is backend-specific."""
    if not 1 <= k <= len(support):
        raise ValueError("KNN requires at least k support points")
    _, indices = cKDTree(support).query(query, k=k, workers=1)
    return np.asarray(indices, dtype=np.int64).reshape(len(query), k)


def decode_fine_pixels(
    centers_xy: np.ndarray,
    indices: np.ndarray,
    *,
    layout: Literal["row_major", "official_eval_legacy"],
) -> np.ndarray:
    """Return Nx2 fine-grid (u,v), requiring an explicit decoding convention."""
    c, i = np.asarray(centers_xy, float), np.asarray(indices)
    if (
        c.ndim != 2
        or c.shape[1] != 2
        or not np.isfinite(c).all()
        or not np.equal(c, np.floor(c)).all()
        or i.shape != (len(c),)
        or i.dtype.kind not in "iu"
        or np.any(i < 0)
        or np.any(i >= 16)
    ):
        raise ValueError("integer Nx2 centers and N patch indices in [0,15] required")
    top_left = c - 2
    if np.any(top_left < 0) or np.any(top_left + 3 >= [256, 80]):
        raise ValueError("4x4 patch exceeds 256x80 fine grid")
    if layout == "row_major":
        offset = np.column_stack([i % 4, i // 4])
    elif layout == "official_eval_legacy":
        offset = np.column_stack([i // 4, i % 4])
    else:
        raise ValueError("explicit supported pixel layout required")
    return top_left + offset


def fine_to_source_pixels(fine_xy: np.ndarray, source_to_input: np.ndarray) -> np.ndarray:
    """Use the declared geometry boundary's K_fine=diag(.5,.5,1)@K_input."""
    from tools.cofii2p_input_boundary import map_input_pixels_to_source

    return map_input_pixels_to_source(np.asarray(fine_xy) * 2, source_to_input)


def build_bridge(manifest_path: Path, output: Path, options: BridgeOptions) -> BridgeManifest:
    """Build model-shaped arrays and complete centroid/source lineage on CPU."""
    if output.exists():
        raise ValueError("refuse overwrite")
    if options.count // (2 ** (options.stages - 1)) < options.neighbors:
        raise ValueError("smallest level must have at least neighbors points")
    prepared = Prepared.model_validate_json(manifest_path.read_text())
    artifact_path = Path(prepared.artifact.path)
    if digest(artifact_path) != prepared.artifact.sha256:
        raise ValueError("prepared artifact digest mismatch")
    with np.load(artifact_path, allow_pickle=False) as archive:
        p = archive["point_initial_camera_m"]
        lidar = archive["point_lidar_m"]
        features = np.column_stack([archive["intensity"], archive["normal_initial_camera"]])
        ids = archive["point_id"]
        image = archive["image_bgr_chw"]
        k = archive["input_intrinsics"]
    means, feats, offsets, rows, inverse = voxelize_with_lineage(p, features, options.voxel_size_m)
    lidar_means = np.zeros_like(means)
    np.add.at(lidar_means, inverse, lidar)
    lidar_means /= np.diff(offsets)[:, None]
    rng = np.random.RandomState(options.seed)
    selection = count_indices(len(means), options.count, rng)
    arrays = {
        "voxel_point_initial_camera_m": means,
        "voxel_point_lidar_m": lidar_means,
        "voxel_features": feats,
        "voxel_member_offsets": offsets,
        "voxel_member_source_rows": rows,
        "source_point_id": ids,
        "image_bgr_chw": image,
        "source_to_input_pixel": np.asarray(prepared.source_to_input_pixel),
        "initial_camera_lidar": np.asarray(prepared.initial_camera_lidar),
        "fine_intrinsics": np.diag([0.5, 0.5, 1]) @ k,
        "coarse_intrinsics": np.diag([0.125, 0.125, 1]) @ k,
        "feats": feats[selection].astype(np.float32),
    }
    for level in range(options.stages):
        if level:
            parent = rng.choice(len(selection), len(selection) // 2, replace=True)
            arrays[f"parent_{level}"] = parent
            selection = selection[parent]
        else:
            arrays["count_selection"] = selection.copy()
        arrays[f"voxel_index_{level}"] = selection.copy()
        arrays[f"points_{level}"] = means[selection].astype(np.float32)
    for level in range(options.stages):
        points = arrays[f"points_{level}"]
        arrays[f"neighbors_{level}"] = neighbors(points, points, options.neighbors)
        if level + 1 < options.stages:
            sub = arrays[f"points_{level + 1}"]
            arrays[f"subsampling_{level}"] = neighbors(points, sub, options.neighbors)
            arrays[f"upsampling_{level}"] = neighbors(sub, points, options.neighbors)
    output.mkdir(parents=True)
    artifact = output / "bridge.npz"
    np.savez_compressed(artifact, **arrays)
    report = BridgeManifest(
        scope="CPU geometry only; no model inference or official numerical parity",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        driver_sha256=digest(Path(__file__)),
        prepared_manifest=FileRef(path=str(manifest_path), sha256=digest(manifest_path)),
        prepared_artifact=prepared.artifact,
        options=options,
        artifact=FileRef(path=str(artifact), sha256=digest(artifact)),
        array_shapes={name: list(value.shape) for name, value in arrays.items()},
        conventions={
            "voxel_order": "lexicographic grid keys; differs from Open3D hash iteration",
            "voxel_origin": "min(initial-camera points) - voxel_size/2",
            "features": "mean intensity and mean rotated normals; no renormalization",
            "pyramid": "half sampling WITH replacement; duplicates retained",
            "lineage": "level voxel indices -> CSR member source rows -> original IDs",
            "neighbors": "query-to-support indices; SciPy ties not Open3D parity",
            "pixels": "fine K=.5*input K, coarse K=.125*input K; decoder choice required",
        },
        runtime={"numpy": np.__version__, "scipy": scipy.__version__},
    )
    (output / "manifest.json").write_text(report.model_dump_json(indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_manifest", type=Path)
    parser.add_argument("options", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_bridge(
        args.prepared_manifest,
        args.output,
        BridgeOptions.model_validate_json(args.options.read_text()),
    )
