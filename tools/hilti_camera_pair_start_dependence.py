#!/usr/bin/env python3
"""Does a Hilti camera pair's reading depend on the candidate it starts from?

An evaluation harness, not part of the Calibrex package. It reads the frames where all
five alphasense cameras fired together (as ``hilti_camera_pair_reproducibility.py``)
and estimates each requested pair from the camchain candidate turned by each of
``--starts`` degrees about the first camera's y axis (pitch). It does this with the
check's re-centring and, for comparison, with a single measurement about the start
(``max_recentres=0``, no coarse re-centring). The start turn is removed from the
reading, so a reading that does not depend on the start comes out the same each time.

::

    python tools/hilti_camera_pair_start_dependence.py exp01_head.bag \\
        --camchain calib_3_cam0-1-camchain-imucam.yaml \\
        --camchain calib_3_cam2-camchain-imucam.yaml \\
        --camchain calib_3_cam3-camchain-imucam.yaml \\
        --camchain calib_3_cam4-camchain-imucam.yaml \\
        --pairs 0-3,1-3,0-4,1-4 --output exp01_start_dependence.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hilti_camera_pair_reproducibility import TOPICS, read_synchronised

from calibrex.check.estimators import transform_matrix
from calibrex.check.tf_sources import camchain_entries, load_tf_file, merge_sources
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.stereo_rotation import StereoRotationProvenance
from calibrex.evaluation.stereo_rotation import (
    StereoRotationOptions,
    estimate_stereo_rotation,
    stereo_camera,
)
from calibrex.evaluation.visual_rotation import CameraModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--camchain", type=Path, action="append", required=True)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--spacing-s", type=float, default=1.0)
    parser.add_argument("--pairs", default="0-3,1-3,0-4,1-4")
    parser.add_argument("--starts", default="-0.5,0,0.5", help="pitch turns of the start, deg")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    commit = git_commit()
    tree, _ = merge_sources(None, [load_tf_file(path) for path in args.camchain])
    models = {
        frame: CameraModel.from_kalibr(entry)
        for path in args.camchain
        for _, frame, entry in camchain_entries(path)
    }
    started = time.monotonic()
    groups, _ = read_synchronised(args.bag, TOPICS, args.spacing_s, args.frames)
    print(f"read {len(groups)} synchronised sets", flush=True)
    height, width = groups[0][TOPICS[0]].shape
    starts = [float(item) for item in args.starts.split(",")]
    results: dict[str, list[dict[str, object]]] = {}
    for item in args.pairs.split(","):
        i, j = (int(value) for value in item.split("-"))
        left_frame, right_frame = TOPICS[i].lstrip("/"), TOPICS[j].lstrip("/")
        relation = tree.lookup(left_frame, right_frame)
        if relation is None:
            continue
        candidate = transform_matrix(relation.translation_m, relation.rotation_quat_xyzw)
        left = stereo_camera(models[left_frame], width, height)
        right = stereo_camera(models[right_frame], width, height)
        pairs = [
            (group[TOPICS[i]].astype(np.float64), group[TOPICS[j]].astype(np.float64))
            for group in groups
        ]
        provenance = StereoRotationProvenance(
            generator="tools/hilti_camera_pair_start_dependence.py",
            generator_version="0.1",
            git_commit=commit,
            left_topic=TOPICS[i],
            right_topic=TOPICS[j],
            left_frame=left_frame,
            right_frame=right_frame,
            intrinsics_source=", ".join(path.name for path in args.camchain),
        )
        rows: list[dict[str, object]] = []
        for start in starts:
            moved = candidate.copy()
            turn = Rotation.from_rotvec(np.radians(start) * np.eye(3)[1]).as_matrix()
            moved[:3, :3] = turn @ candidate[:3, :3]
            for method, options in (
                ("re-centred", StereoRotationOptions()),
                ("single", StereoRotationOptions(max_recentres=0, recentre_deg=1.0e9)),
            ):
                artifact = estimate_stereo_rotation(
                    pairs, left, right, moved, provenance=provenance, options=options
                )
                # the reading against the camchain: the start's own turn removed
                error = [axis.candidate_error_deg for axis in artifact.axes]
                if error:
                    error[1] -= start
                rows.append(
                    {
                        "start_pitch_deg": start,
                        "method": method,
                        "policy_status": artifact.policy_status,
                        "error_deg": [round(value, 4) for value in error],
                        "std_deg": [round(axis.std_deg, 4) for axis in artifact.axes],
                        "control_response": [round(a.control_response, 3) for a in artifact.axes],
                        "observable": [axis.observable for axis in artifact.axes],
                        "residual_std_px": artifact.residual_std_px,
                        "recentred_by_deg": artifact.recentred_by_deg,
                    }
                )
                print(f"cam{i}-cam{j}", json.dumps(rows[-1]), flush=True)
        results[f"cam{i}-cam{j}"] = rows
    output = {
        "frames": len(groups),
        "pairs": results,
        "provenance": {
            "git_commit": commit,
            "bag": {"name": args.bag.name, "size_bytes": args.bag.stat().st_size},
            "camchain_sha256": {path.name: sha256_path(path) for path in args.camchain},
            "elapsed_s": round(time.monotonic() - started, 1),
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
