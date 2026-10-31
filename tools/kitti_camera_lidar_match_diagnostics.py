"""Explain saved pitch-control failures without fitting or changing any pose."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field
from scipy import ndimage
from scipy.spatial.transform import Rotation
from tools import kitti_camera_lidar_nearest_edge as nearest
from tools import kitti_camera_lidar_profile_probe as probe
from tools import kitti_camera_lidar_v3_replay as replay
from tools.kitti_camera_lidar_initialization_compare import sha

from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel
from calibrex.evaluation import camera_lidar_edge as edge


class Diagnostics(StrictModel):
    schema_version: Literal["calibrex.pitch_match_diagnostics/v0.1"] = (
        "calibrex.pitch_match_diagnostics/v0.1"
    )
    provenance: dict[str, Any]
    folds: list[dict[str, Any]]
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Post hoc attribution on spent data; no fitting or new accuracy claim.",
            "No cross-frame identity tracking; semantic labels require visual inspection.",
            "Nearest-ridge lines use rounded pixels; scores use exact bilinear interpolation.",
            "Examples select the largest control gain per fold; every frame is retained in JSON.",
        ]
    )


def contributions(
    objective: edge.EdgeObjective, theta: np.ndarray, *, verify: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected pixels and per-point contributions with the fixed denominator."""
    rotation = edge._rotation_from_vector(np.radians(theta[:3])).astype(np.float32)
    moved = objective._points @ rotation.T
    moved += objective._start_translation + theta[3:].astype(np.float32)
    u, v, valid = objective.camera.project(moved)
    u = (u - objective._shift) * objective._inverse_scale
    v = (v - objective._shift) * objective._inverse_scale
    inside = valid & (u >= 0) & (v >= 0) & (u < objective._u_limit) & (v < objective._v_limit)
    result = np.zeros(len(u))
    x, y = u[inside], v[inside]
    left, top = x.astype(np.int64), y.astype(np.int64)
    dx, dy = x - left, y - top
    stride = objective._stride[inside]
    base = objective._offset[inside] + top * stride + left
    flat = objective._flat
    response = (1 - dx) * ((1 - dy) * flat[base] + dy * flat[base + stride]) + dx * (
        (1 - dy) * flat[base + 1] + dy * flat[base + stride + 1]
    )
    result[inside] = response * objective._weights[inside] / objective._total_weight
    if verify:
        np.testing.assert_allclose(result.sum(), objective.score(theta), atol=1e-12, rtol=0)
    return np.column_stack([u, v]), result


def run(snapshot: Path, receipt: Path, output: Path) -> Diagnostics:
    """Replay all six folds and retain every held frame, selecting one example per fold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta, frames, camera, candidate, options = replay.load_snapshot(snapshot)
    saved = probe.ProfileReport.model_validate_json(receipt.read_bytes())
    if (meta.drive, meta.run) != ("2011_09_28_0002", "inject_y+3"):
        raise ValueError("this diagnostic requires the fixed targeted pitch input")
    if (
        saved.status != "complete"
        or not saved.provenance.get("nearest_edge")
        or saved.provenance["arrays_sha256"] != meta.arrays_sha256
        or saved.provenance["snapshot_sha256"] != sha(snapshot / "snapshot.json")
        or saved.provenance["nearest_edge_source_sha256"] != sha(Path(nearest.__file__))
    ):
        raise ValueError("receipt does not match frozen nearest-edge input and source")
    output.mkdir(parents=True, exist_ok=False)
    prepared, _ = edge.prepare_frames(frames, camera, candidate, options)
    images = {f.time_s: f.image for f in frames}
    prepared = [
        replace(f, edge=nearest.prepare_nearest_edge(images[f.time_s], options)[0])
        for f in prepared
    ]
    report = Diagnostics(
        provenance={
            "git_commit": git_commit(),
            "generator_sha256": sha(Path(__file__)),
            "receipt_sha256": sha(receipt),
            "snapshot_sha256": sha(snapshot / "snapshot.json"),
            "arrays_sha256": meta.arrays_sha256,
            "nearest_source_sha256": sha(Path(nearest.__file__)),
            "estimator_sha256": sha(Path(edge.__file__)),
            "new_evaluation_started": False,
            "new_accuracy_claim": False,
            "reference_theta": [0, -3, 0, 0, 0, 0],
        },
        folds=[],
    )
    reference = np.array([0.0, -3, 0, 0, 0, 0])
    ref_rotation = edge._transform(candidate, reference)[:3, :3]
    for fold_index, (indices, fold) in enumerate(
        zip(edge._block_slices(len(prepared), options.blocks), saved.axes[1].folds, strict=True)
    ):
        poses = [reference, np.array(fold.base_theta), *map(np.array, fold.profiled_theta)]
        held = edge.EdgeObjective([prepared[i] for i in indices], camera, candidate)
        scores = [held.score(p) for p in poses]
        np.testing.assert_allclose(
            [scores[1] - x for x in scores[2:]], fold.profiled_drops, atol=1e-9, rtol=0
        )
        errors = [
            np.degrees(
                Rotation.from_matrix(
                    edge._transform(candidate, p)[:3, :3] @ ref_rotation.T
                ).as_rotvec()
            ).tolist()
            for p in poses
        ]
        row: dict[str, Any] = {
            "fold": fold_index,
            "poses": [p.tolist() for p in poses],
            "reference_rotation_error_vectors_deg": errors,
            "rotation_error_norms_deg": [float(np.linalg.norm(e)) for e in errors],
            "held_scores": scores,
            "frames": [],
        }
        candidates = []
        for index in indices:
            frame = prepared[index]
            objective = edge.EdgeObjective([frame], camera, candidate)
            projected = [contributions(objective, p) for p in poses]
            values = [float(c.sum()) for _, c in projected]
            controls = []
            for sign in range(2):
                pixels, control = projected[sign + 2]
                gain = control - projected[1][1]
                positive = np.maximum(gain, 0)
                tile_gain = np.zeros((3, 3))
                h, w = frame.edge.shape
                for k in np.flatnonzero(positive > 0):
                    tx, ty = np.clip((pixels[k] / [w, h] * 3).astype(int), 0, 2)
                    tile_gain[ty, tx] += positive[k]
                order = np.argsort(-positive, kind="stable")
                top = order[: max(1, len(order) // 10)]
                controls.append(
                    {
                        "gain": float(gain.sum()),
                        "positive_gain": float(positive.sum()),
                        "negative_gain": float(np.minimum(gain, 0).sum()),
                        "top_decile_positive_gain": float(positive[top].sum()),
                        "positive_gain_by_image_tile_3x3": tile_gain.tolist(),
                    }
                )
                candidates.append((float(gain.sum()), index, sign, projected, order))
            row["frames"].append(
                {
                    "prepared_index": index,
                    "time_s": frame.time_s,
                    "point_count": len(frame.points),
                    "scores": values,
                    "controls": controls,
                }
            )
        gain, index, sign, projected, order = max(candidates, key=lambda x: x[0])
        row["example"] = {
            "prepared_index": index,
            "control_sign_index": sign,
            "gain": gain,
            "selection": "largest per-frame control-minus-baseline score in fold",
        }
        frame = prepared[index]
        gray = images[frame.time_s].astype(float)
        f = frame.scale
        gray = gray[: gray.shape[0] // f * f, : gray.shape[1] // f * f]
        gray = gray.reshape(gray.shape[0] // f, f, gray.shape[1] // f, f).mean(axis=(1, 3))
        ridges = frame.edge == 1
        _, nearest_indices = ndimage.distance_transform_edt(~ridges, return_indices=True)
        figure, axes = plt.subplots(3, 1, figsize=(14, 10))
        figure.subplots_adjust(left=0.05, right=0.99, bottom=0.04, top=0.91, hspace=0.4)
        for axis, pose_index, title in zip(
            axes,
            [0, 1, sign + 2],
            ["Dataset reference", "Fitted baseline", "Refitted control"],
            strict=True,
        ):
            axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
            pixels, _ = projected[pose_index]
            axis.scatter(pixels[:, 0], pixels[:, 1], s=3, c="cyan", alpha=0.35)
            for k in order[:40]:
                x, y = pixels[k]
                if not (0 <= x < gray.shape[1] - 1 and 0 <= y < gray.shape[0] - 1):
                    continue
                yy, xx = nearest_indices[:, round(y), round(x)]
                axis.plot([x, xx], [y, yy], color="orange", linewidth=0.65)
                axis.plot(x, y, ".", color="orange", markersize=4)
            axis.set(
                xlim=(0, gray.shape[1]),
                ylim=(gray.shape[0], 0),
                title=f"{title}: score={projected[pose_index][1].sum():.4f}, "
                f"rotation error={row['rotation_error_norms_deg'][pose_index]:.3f} deg",
            )
        figure.suptitle(
            f"Fold {fold_index}, frame {index}: cyan=all points; "
            "orange=same 40 largest control-gain points"
        )
        figure.savefig(output / f"fold-{fold_index}.png", dpi=140)
        plt.close(figure)
        report.folds.append(row)
    (output / "report.json").write_text(report.model_dump_json(indent=2))
    (output / "report.schema.json").write_text(json.dumps(report.model_json_schema(), indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.snapshot, args.receipt, args.output)


if __name__ == "__main__":
    main()
