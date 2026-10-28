"""Cross-score saved fits to separate search failures from objective ambiguity."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation
from tools import kitti_camera_lidar_contour_coherence as contour
from tools import kitti_camera_lidar_match_geometry as geometry
from tools import kitti_camera_lidar_nearest_edge as nearest
from tools import kitti_camera_lidar_profile_probe as probe
from tools import kitti_camera_lidar_v3_replay as replay
from tools.kitti_camera_lidar_initialization_compare import sha

from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel
from calibrex.evaluation import camera_lidar_edge as edge


class Attribution(StrictModel):
    schema_version: Literal["calibrex.geometry_failure_attribution/v0.1"] = (
        "calibrex.geometry_failure_attribution/v0.1"
    )
    provenance: dict[str, Any]
    cases: list[dict[str, Any]]
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Post hoc selection: two lost controls and the previously targeted pitch control.",
            "Saved poses are cross-scored without reoptimization; this is not a new estimator.",
            "The dataset reference is a diagnostic pose, not an available initialization.",
            "Training score dominance establishes a missed candidate, not global optimality.",
        ]
    )


def run(captures: Path, receipts: Path, output: Path) -> Attribution:
    """Replay all six folds of each selected case and both objectives on both saved fits."""
    output.mkdir(parents=True, exist_ok=False)
    report = Attribution(
        provenance={
            "git_commit": git_commit(),
            "generator_sha256": sha(Path(__file__)),
            "geometry_source_sha256": sha(Path(geometry.__file__)),
            "graph_source_sha256": sha(Path(contour.__file__)),
            "nearest_source_sha256": sha(Path(nearest.__file__)),
            "estimator_sha256": sha(Path(edge.__file__)),
            "selection": {"2011_09_26_0005": [0, 2], "2011_09_28_0002": [1]},
            "new_evaluation_started": False,
            "new_accuracy_claim": False,
        },
        cases=[],
    )
    for drive, axes in [("2011_09_26_0005", [0, 2]), ("2011_09_28_0002", [1])]:
        snapshot = captures / drive / "inject_y+3"
        meta, frames, camera, candidate, options = replay.load_snapshot(snapshot)
        prepared, dropped = edge.prepare_frames(frames, camera, candidate, options)
        images = {f.time_s: f.image for f in frames}
        changed = [
            replace(f, edge=nearest.prepare_nearest_edge(images[f.time_s], options)[0])
            for f in prepared
        ]
        maps = [geometry.geometry_map(f.edge) for f in changed]
        graphs = [contour.neighbor_links(f.points) for f in prepared]
        saved = {}
        hashes = {}
        for arm in ["original", "match_geometry"]:
            path = receipts / drive / meta.run / arm / "report.json"
            r = probe.ProfileReport.model_validate_json(path.read_bytes())
            if (
                r.status != "complete"
                or r.provenance["snapshot_sha256"] != sha(snapshot / "snapshot.json")
                or r.provenance["arrays_sha256"] != meta.arrays_sha256
                or r.provenance["estimator_sha256"] != sha(Path(edge.__file__))
            ):
                raise ValueError("saved report does not match input/source")
            if arm == "match_geometry" and r.provenance["geometry_source_sha256"] != sha(
                Path(geometry.__file__)
            ):
                raise ValueError("geometry source differs")
            saved[arm], hashes[arm] = r, sha(path)
        reference = np.array([0.0, -3, 0, 0, 0, 0])
        reference_rotation = edge._transform(candidate, reference)[:3, :3]
        for axis in axes:
            case: dict[str, Any] = {
                "drive": drive,
                "run": meta.run,
                "axis": axis,
                "snapshot_sha256": sha(snapshot / "snapshot.json"),
                "arrays_sha256": meta.arrays_sha256,
                "receipt_sha256": hashes,
                "prepared_frames": len(prepared),
                "dropped": dropped,
                "folds": [],
            }
            for number, block in enumerate(edge._block_slices(len(prepared), options.blocks)):
                subsets = {
                    "held": list(block),
                    "training": [i for i in range(len(prepared)) if i not in block],
                }
                objectives = {}
                for subset, indices in subsets.items():
                    objectives[subset] = {
                        "original": edge.EdgeObjective(
                            [prepared[i] for i in indices], camera, candidate
                        ),
                        "match_geometry": geometry.GeometryObjective(
                            [changed[i] for i in indices],
                            camera,
                            candidate,
                            [maps[i] for i in indices],
                            [graphs[i] for i in indices],
                        ),
                    }
                fold: dict[str, Any] = {"fold": number, "fits": {}}
                for arm, r in saved.items():
                    saved_fold = r.axes[axis].folds[number]
                    poses = [
                        reference,
                        np.array(saved_fold.base_theta),
                        *map(np.array, saved_fold.profiled_theta),
                    ]
                    errors = [
                        np.degrees(
                            Rotation.from_matrix(
                                edge._transform(candidate, p)[:3, :3] @ reference_rotation.T
                            ).as_rotvec()
                        )
                        for p in poses
                    ]
                    scores = {
                        subset: {
                            objective: [o.score(p) for p in poses] for objective, o in opts.items()
                        }
                        for subset, opts in objectives.items()
                    }
                    own = scores["held"][arm]
                    np.testing.assert_allclose(
                        [own[1] - v for v in own[2:]], saved_fold.profiled_drops, atol=1e-9, rtol=0
                    )
                    fold["fits"][arm] = {
                        "pose_order": ["reference", "baseline", "minus", "plus"],
                        "poses": [p.tolist() for p in poses],
                        "full_rotation_errors_deg": [float(np.linalg.norm(e)) for e in errors],
                        "scores": scores,
                    }
                case["folds"].append(fold)
            report.cases.append(case)
    (output / "report.json").write_text(report.model_dump_json(indent=2))
    (output / "report.schema.json").write_text(json.dumps(report.model_json_schema(), indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.captures, args.receipts, args.output)


if __name__ == "__main__":
    main()
