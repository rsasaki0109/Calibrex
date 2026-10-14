#!/usr/bin/env python3
"""Does a Hilti camera pair's rotation reading depend on where in the images it is read?

An evaluation harness, not part of the Calibrex package. ``camera-camera`` fits one
rotation to the row offsets of every matched patch. If one camera's lens model is off
in part of its view (Kalibr's equidistant polynomial is extrapolated where the target
never was), the patches there read a different rotation. This tool:

1. estimates each requested pair as the check does (re-centred), on frames where all
   five alphasense cameras fired together;
2. rectifies with that estimate, matches patches again, and traces each one back to its
   raw pixel in both cameras, with the ray's angle from each optical axis (``theta``)
   and its direction (towards which other camera);
3. refits the residual rotation on subsets of the patches: by ``theta`` in the first
   camera, and by the side of the first camera's view a patch lies on (``left``,
   ``centre``, ``right`` thirds of its columns, in raw pixels);
4. refits all patches with terms the rotation model leaves out: a row offset growing
   with the row (a vertical scale mismatch between the rectified images) and with row
   times disparity (a baseline direction off along the rectified optical axis).

A lens model that is right everywhere gives the same reading (near zero) on every
subset. A reading that moves with one camera's ``theta`` or side points at that camera.

::

    python tools/hilti_camera_edge_bias.py exp01_head.bag \\
        --camchain calib_3_cam0-1-camchain-imucam.yaml \\
        --camchain calib_3_cam2-camchain-imucam.yaml \\
        --camchain calib_3_cam3-camchain-imucam.yaml \\
        --camchain calib_3_cam4-camchain-imucam.yaml \\
        --pairs 0-1,0-3,1-3,0-4,1-4 --output exp01_edge_bias.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hilti_camera_pair_reproducibility import TOPICS, read_synchronised

from calibrex.check.estimators import transform_matrix
from calibrex.check.tf_sources import camchain_entries, load_tf_file, merge_sources
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.stereo_rotation import StereoRotationProvenance
from calibrex.evaluation.fisheye_stereo import (
    VERTICAL_OFFSET_COLUMNS,
    rectify_fisheye_pair,
    vertical_offset_samples,
)
from calibrex.evaluation.stereo_rotation import (
    StereoRotationOptions,
    estimate_stereo_rotation,
    stereo_camera,
)
from calibrex.evaluation.visual_rotation import CameraModel

FloatArray = NDArray[np.float64]
AXES = ("roll", "pitch", "yaw")


def _inverse(transform: FloatArray) -> FloatArray:
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _sample_map(field: FloatArray, rows: FloatArray, cols: FloatArray) -> FloatArray:
    return np.stack(
        [ndimage.map_coordinates(field[k], [rows, cols], order=1, mode="nearest") for k in range(2)]
    )


def _theta_deg(camera: object, raw: FloatArray) -> FloatArray:
    rays = camera.unproject(raw[1], raw[0])  # type: ignore[attr-defined]
    return np.degrees(np.arccos(np.clip(rays[..., 2], -1.0, 1.0)))


def _fit(
    design: FloatArray, values: FloatArray, frames: NDArray[np.int64], blocks: int = 6
) -> tuple[FloatArray, FloatArray]:
    """Trimmed least squares with a block jackknife over frames: every coefficient and its
    std (the first three, the rotation, in degrees)."""

    def solve(rows: NDArray[np.bool_]) -> FloatArray:
        keep = rows.copy()
        coefficients = np.zeros(design.shape[1])
        for _ in range(4):
            coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
            error = values - design @ coefficients
            spread = 1.4826 * float(np.median(np.abs(error[keep])))
            keep = rows & (np.abs(error) <= 3.0 * max(spread, 1.0e-6))
        return coefficients

    every = np.ones(len(values), dtype=bool)
    estimate = solve(every)
    groups = np.array_split(np.unique(frames), blocks)
    jack = np.array([solve(~np.isin(frames, group)) for group in groups])
    std = np.sqrt((blocks - 1) / blocks * ((jack - jack.mean(axis=0)) ** 2).sum(axis=0))
    scale = np.ones(len(estimate))
    scale[:3] = np.degrees(1.0)
    return estimate * scale, std * scale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--camchain", type=Path, action="append", required=True)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--spacing-s", type=float, default=1.0)
    parser.add_argument("--pairs", default="0-1,0-3,1-3,0-4,1-4")
    parser.add_argument("--min-patches", type=int, default=300)
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
    results: dict[str, object] = {}
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
            generator="tools/hilti_camera_edge_bias.py",
            generator_version="0.1",
            git_commit=commit,
            left_topic=TOPICS[i],
            right_topic=TOPICS[j],
            left_frame=left_frame,
            right_frame=right_frame,
            intrinsics_source=", ".join(path.name for path in args.camchain),
        )
        artifact = estimate_stereo_rotation(
            pairs, left, right, candidate, provenance=provenance, options=StereoRotationOptions()
        )
        if not artifact.axes:
            results[f"cam{i}-cam{j}"] = {"status": artifact.policy_status}
            continue
        error = np.array([axis.candidate_error_deg for axis in artifact.axes])
        estimate = candidate.copy()
        estimate[:3, :3] = Rotation.from_rotvec(-np.radians(error)).as_matrix() @ candidate[:3, :3]
        rect = rectify_fisheye_pair(left, right, _inverse(estimate))
        rectified = [(rect.rectify_left(a), rect.rectify_right(b)) for a, b in pairs]
        samples = vertical_offset_samples(rectified, patches_per_pair=300, max_disparity_px=60)
        column = {name: samples[:, k] for k, name in enumerate(VERTICAL_OFFSET_COLUMNS)}
        known = column["vertical_texture"] > 0.5
        row, col = column["row_px"][known], column["column_px"][known]
        offset, disparity = column["offset_px"][known], column["disparity_px"][known]
        frame = column["pair"][known].astype(np.int64)
        raw_left = _sample_map(rect.left_map, row, col)
        raw_right = _sample_map(rect.right_map, row + offset, col - disparity)
        theta_left = _theta_deg(left, raw_left)
        theta_right = _theta_deg(right, raw_right)
        f = rect.fx
        x, y = col - rect.cx, row - rect.cy
        rows_model = f * np.c_[-(1.0 + (y / f) ** 2), x * y / f**2, x / f]
        design = np.hstack([rows_model @ rect.rectified_from_left, disparity[:, None]])

        def subset(
            mask: NDArray[np.bool_],
            design: FloatArray = design,
            offset: FloatArray = offset,
            frame: NDArray[np.int64] = frame,
        ) -> dict[str, object]:
            if mask.sum() < args.min_patches:
                return {"patches": int(mask.sum())}
            value, std = _fit(design[mask], offset[mask], frame[mask])
            return {
                "patches": int(mask.sum()),
                "residual_deg": {
                    a: round(float(v), 3) for a, v in zip(AXES, value[:3], strict=True)
                },
                "std_deg": {a: round(float(e), 3) for a, e in zip(AXES, std[:3], strict=True)},
                # the disparity term (px per px) and any added terms (px per unit), with stds
                "other_terms": [
                    [round(float(v), 5), round(float(e), 5)]
                    for v, e in zip(value[3:], std[3:], strict=True)
                ],
            }

        # terms the rotation model leaves out: a row offset growing with the row (a vertical
        # scale mismatch between the two rectified images) and with row times disparity (a
        # baseline direction off along the rectified optical axis)
        every = np.ones(len(offset), dtype=bool)
        nuisance = {
            "row": (y / f)[:, None],
            "row_x_disparity": (y * disparity / f)[:, None],
        }
        extended: dict[str, object] = {}
        for label, columns in (
            ("row", [nuisance["row"]]),
            ("row_x_disparity", [nuisance["row_x_disparity"]]),
            ("both", [nuisance["row"], nuisance["row_x_disparity"]]),
        ):
            extended[label] = subset(every, np.hstack([design, *columns]))
        cuts = np.quantile(theta_left, [1 / 3, 2 / 3])
        thirds = np.digitize(raw_left[1], [width / 3, 2 * width / 3])
        entry: dict[str, object] = {
            "check_error_deg": [round(float(v), 3) for v in error],
            "patches": int(known.sum()),
            "theta_left_deg": [
                round(float(q), 1) for q in np.quantile(theta_left, [0.1, 0.5, 0.9])
            ],
            "theta_right_deg": [
                round(float(q), 1) for q in np.quantile(theta_right, [0.1, 0.5, 0.9])
            ],
            "all": subset(np.ones(len(offset), dtype=bool)),
            "with_nuisance_terms": extended,
            "by_theta_left": {
                f"<{cuts[0]:.0f}": subset(theta_left < cuts[0]),
                f"{cuts[0]:.0f}-{cuts[1]:.0f}": subset(
                    (theta_left >= cuts[0]) & (theta_left < cuts[1])
                ),
                f">={cuts[1]:.0f}": subset(theta_left >= cuts[1]),
            },
            "by_left_column_third": {
                name: subset(thirds == k) for k, name in enumerate(("left", "centre", "right"))
            },
        }
        results[f"cam{i}-cam{j}"] = entry
        print(f"cam{i}-cam{j}", json.dumps(entry), flush=True)
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
