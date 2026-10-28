"""Fixed-pose screen of neighboring depth points matching one image contour."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field
from scipy import ndimage
from scipy.spatial import cKDTree
from tools import kitti_camera_lidar_match_diagnostics as diagnostics
from tools import kitti_camera_lidar_nearest_edge as nearest
from tools import kitti_camera_lidar_v3_replay as replay
from tools.kitti_camera_lidar_initialization_compare import sha

from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel
from calibrex.evaluation import camera_lidar_edge as edge

SETTINGS = {
    "neighbors": 4,
    "radius_m": 0.75,
    "mutual_neighbors": True,
    "image_connectivity": 8,
    "maximum_match_distance_px": 6.0,
}


class Screen(StrictModel):
    schema_version: Literal["calibrex.contour_coherence_screen/v0.1"] = (
        "calibrex.contour_coherence_screen/v0.1"
    )
    provenance: dict[str, Any]
    support: list[dict[str, Any]]
    folds: list[dict[str, Any]]
    summary: dict[str, Any]
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Post hoc fixed-pose falsification screen; no optimization or new accuracy claim.",
            "Image connected components are not semantic boundary identities.",
            "Three-dimensional neighbors can belong to different physical boundaries.",
            "Labels use rounded pixels; reward retains the existing bilinear distance field.",
            "Unknown labels penalize linked points; isolated points retain the old score.",
        ]
    )


def neighbor_links(points: np.ndarray) -> np.ndarray:
    """Build a fixed undirected graph using mutual radius-bounded nearest neighbors."""
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.int64)
    distances, indices = cKDTree(points).query(
        points, k=SETTINGS["neighbors"] + 1, distance_upper_bound=SETTINGS["radius_m"]
    )
    directed = {
        (i, int(j))
        for i, (ds, js) in enumerate(zip(distances, indices, strict=True))
        for d, j in zip(ds, js, strict=True)
        if np.isfinite(d) and i != j
    }
    return np.array(
        sorted((i, j) for i, j in directed if i < j and (j, i) in directed), dtype=np.int64
    ).reshape(-1, 2)


def compatibility(labels: np.ndarray, links: np.ndarray) -> np.ndarray:
    """Fraction of fixed neighbors on the same nonzero contour; isolated points are neutral."""
    degree = np.bincount(links.ravel(), minlength=len(labels))
    agree = (labels[links[:, 0]] > 0) & (labels[links[:, 0]] == labels[links[:, 1]])
    same = np.bincount(links[agree].ravel(), minlength=len(labels))
    result = np.ones(len(labels))
    np.divide(same, degree, out=result, where=degree > 0)
    return result


def contour_map(response: np.ndarray) -> tuple[np.ndarray, int]:
    """Propagate eight-connected ridge identity no farther than the fixed distance limit."""
    ridges = response == 1
    if not ridges.any():
        raise ValueError("contour input has no ridge pixels")
    labels, count = ndimage.label(ridges, structure=np.ones((3, 3)))
    distances, indices = ndimage.distance_transform_edt(~ridges, return_indices=True)
    propagated = labels[tuple(indices)]
    propagated[distances > SETTINGS["maximum_match_distance_px"]] = 0
    return propagated, count


def score_frame(
    objective: edge.EdgeObjective, theta: np.ndarray, label_map: np.ndarray, links: np.ndarray
) -> tuple[float, float, dict[str, Any]]:
    """Attenuate the saved per-point reward without changing support or denominator."""
    pixels, values = diagnostics.contributions(objective, theta)
    h, w = label_map.shape
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 0] < w - 1)
        & (pixels[:, 1] < h - 1)
    )
    labels = np.zeros(len(pixels), dtype=np.int64)
    rounded = np.rint(pixels[inside]).astype(int)
    labels[inside] = label_map[rounded[:, 1], rounded[:, 0]]
    factors = compatibility(labels, links)
    return (
        float(values.sum()),
        float(values @ factors),
        {
            "matched_points": int(np.count_nonzero(labels)),
            "matched_components": len(np.unique(labels[labels > 0])),
            "same_contour_links": int(
                np.count_nonzero(
                    (labels[links[:, 0]] > 0) & (labels[links[:, 0]] == labels[links[:, 1]])
                )
            ),
        },
    )


def run(snapshot: Path, receipt: Path, output: Path, *, geometry: bool = False) -> Screen:
    """Use all saved poses and held frames from the previous targeted diagnostic."""
    meta, frames, camera, candidate, options = replay.load_snapshot(snapshot)
    saved = diagnostics.Diagnostics.model_validate_json(receipt.read_bytes())
    if (
        (meta.drive, meta.run) != ("2011_09_28_0002", "inject_y+3")
        or saved.provenance["arrays_sha256"] != meta.arrays_sha256
        or saved.provenance["snapshot_sha256"] != sha(snapshot / "snapshot.json")
        or saved.provenance["nearest_source_sha256"] != sha(Path(nearest.__file__))
        or saved.provenance["estimator_sha256"] != sha(Path(edge.__file__))
    ):
        raise ValueError("screen requires the matching fixed-pose diagnostic and sources")
    output.mkdir(parents=True, exist_ok=False)
    prepared, _ = edge.prepare_frames(frames, camera, candidate, options)
    images = {f.time_s: f.image for f in frames}
    prepared = [
        replace(f, edge=nearest.prepare_nearest_edge(images[f.time_s], options)[0])
        for f in prepared
    ]
    graphs = [neighbor_links(f.points) for f in prepared]
    if geometry:
        from tools import kitti_camera_lidar_match_geometry as geometric

        maps = [geometric.geometry_map(f.edge) for f in prepared]
    else:
        maps = [contour_map(f.edge) for f in prepared]
    report = Screen(
        provenance={
            "git_commit": git_commit(),
            "generator_sha256": sha(Path(__file__)),
            "diagnostic_source_sha256": sha(Path(diagnostics.__file__)),
            "snapshot_sha256": sha(snapshot / "snapshot.json"),
            "arrays_sha256": meta.arrays_sha256,
            "receipt_sha256": sha(receipt),
            "settings": SETTINGS,
            "geometry": geometry,
            "geometry_settings": geometric.SETTINGS if geometry else None,
            "geometry_source_sha256": sha(Path(geometric.__file__)) if geometry else None,
            "new_evaluation_started": False,
            "new_accuracy_claim": False,
            "decision_rule": "do not integrate if a saved wrong control still beats the reference "
            "in at least five of six folds; passing is not adoption evidence",
        },
        support=[
            {
                "points": len(f.points),
                "links": len(g),
                "linked_points": len(np.unique(g)),
                "ridge_components": m[1],
            }
            for f, g, m in zip(prepared, graphs, maps, strict=True)
        ],
        folds=[],
        summary={},
    )
    for fold in saved.folds:
        weighted_old = np.zeros(4)
        weighted_new = np.zeros(4)
        weight = 0.0
        rows = []
        for previous in fold["frames"]:
            i = previous["prepared_index"]
            frame = prepared[i]
            if frame.time_s != previous["time_s"]:
                raise ValueError("prepared frame identity differs")
            objective = edge.EdgeObjective([frame], camera, candidate)
            scores = [
                (geometric.score_geometry(objective, np.array(p), maps[i][0], graphs[i], maps[i][2])
                 if geometry else score_frame(objective, np.array(p), maps[i][0], graphs[i]))
                for p in fold["poses"]
            ]
            old, new = [s[0] for s in scores], [s[1] for s in scores]
            np.testing.assert_allclose(old, previous["scores"], atol=1e-9, rtol=0)
            weighted_old += np.array(old) * objective._total_weight
            weighted_new += np.array(new) * objective._total_weight
            weight += objective._total_weight
            rows.append(
                {
                    "prepared_index": i,
                    "time_s": frame.time_s,
                    "original_scores": old,
                    "coherent_scores": new,
                    "match_support_by_pose": [s[2] for s in scores],
                }
            )
        old, new = (weighted_old / weight).tolist(), (weighted_new / weight).tolist()
        np.testing.assert_allclose(old, fold["held_scores"], atol=1e-9, rtol=0)
        report.folds.append(
            {
                "fold": fold["fold"],
                "poses": fold["poses"],
                "rotation_error_norms_deg": fold["rotation_error_norms_deg"],
                "original_scores": old,
                "coherent_scores": new,
                "frames": rows,
            }
        )
    for name in ["original_scores", "coherent_scores"]:
        report.summary[name] = {
            "folds_control_beats_reference": sum(
                max(f[name][2:]) > f[name][0] for f in report.folds
            ),
            "folds_control_beats_baseline": sum(
                max(f[name][2:]) > f[name][1] for f in report.folds
            ),
            "frames_control_beats_reference": sum(
                max(f[name][2:]) > f[name][0] for fold in report.folds for f in fold["frames"]
            ),
        }
    report.summary["decision"] = (
        "reject_integration"
        if report.summary["coherent_scores"]["folds_control_beats_reference"] >= 5
        else "requires_full_paired_development_experiment"
    )
    (output / "report.json").write_text(report.model_dump_json(indent=2))
    (output / "report.schema.json").write_text(json.dumps(report.model_json_schema(), indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    report = run(args.snapshot, args.receipt, args.output, geometry=args.geometry)
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    main()
