#!/usr/bin/env python3
"""Pilot: does a disparity scale term belong in the stereo depth estimate?

An evaluation harness, not part of the Calibrex package. It measures the stereo depth
residuals (:func:`calibrex.evaluation.stereo_lidar_depth.frame_residuals`) at the
declared reference and at the reference moved 0.1 and 0.25 m along the optical axis, and
fits each with three models:

* **current**: ``residual = delta + offset * base / Z**2``;
* **free scale**: the same plus ``scale * base / Z``;
* **fixed scale**: the current model after removing a given scale (the joint stereo
  fit's) from the residuals.

Each reports the reference's offset, its block-jackknife std (6 blocks over frames), the
0.1 m control response and what the 0.25 m move reads as. No edge alignment is run.

::

    python tools/stereo_depth_scale_pilot.py kitti DRIVE_DIR spread --fixed-scale -0.0019
    python tools/stereo_depth_scale_pilot.py hilti exp21_head.bag \\
        --camchain CAMCHAIN --lidar-calibration LIDAR_CALIBRATION \\
        --stamp-pairing --no-motion --fixed-scale -0.0267
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np

from calibrex.data.kitti import read_velodyne_to_camera_transform
from calibrex.evaluation.kitti_edge_falsification import (
    kitti_edge_candidates,
    kitti_edge_frames,
    kitti_stereo,
    kitti_stereo_frames,
    select_kitti_pairs,
)
from calibrex.evaluation.stereo_lidar_depth import (
    RectifiedStereo,
    StereoDepthOptions,
    frame_residuals,
)

SHIFTS_M = (0.0, 0.1, 0.25)
BLOCKS = 6
Residuals = list[tuple[np.ndarray, np.ndarray]]


def fit(depth: np.ndarray, residual: np.ndarray, base: float, scale: bool) -> np.ndarray:
    """``[delta, offset(, scale)]``, three passes trimming beyond 3 robust std."""

    columns = [np.ones_like(depth), base / depth**2] + ([base / depth] if scale else [])
    design = np.stack(columns, axis=1)
    keep = np.ones(len(depth), dtype=bool)
    coefficients = np.zeros(design.shape[1])
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[keep], residual[keep], rcond=None)
        error = residual - design @ coefficients
        spread = 1.4826 * float(np.median(np.abs(error[keep])))
        keep = np.abs(error) <= 3.0 * max(spread, 1.0e-6)
    return coefficients


def estimate(per_frame: Residuals, base: float, scale: bool) -> tuple[np.ndarray, float]:
    """The fit on every frame and the block-jackknife std of its offset."""

    def stacked(frames: Sequence[int]) -> np.ndarray:
        depth = np.concatenate([per_frame[i][0] for i in frames])
        residual = np.concatenate([per_frame[i][1] for i in frames])
        return fit(depth, residual, base, scale)

    everything = range(len(per_frame))
    blocks = np.array_split(np.arange(len(per_frame)), BLOCKS)
    offsets = np.array(
        [stacked([i for i in everything if i not in set(b.tolist())])[1] for b in blocks]
    )
    std = float(np.sqrt((BLOCKS - 1) / BLOCKS * ((offsets - offsets.mean()) ** 2).sum()))
    return stacked(list(everything)), std


def report(name: str, residuals: dict[float, Residuals], base: float) -> None:
    results = {shift: estimate(items, base, scale=False) for shift, items in residuals.items()}
    reference, std = results[0.0]
    print(
        f"{name}: offset {reference[1]:+.3f} m, std {std:.3f} m, control "
        f"{(results[0.1][0][1] - reference[1]) / 0.1:.2f}, +0.25 m reads "
        f"{results[0.25][0][1]:+.3f} m"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="dataset", required=True)
    kitti = sub.add_parser("kitti")
    kitti.add_argument("drive", type=Path)
    kitti.add_argument("selection", choices=["spread", "first"])
    kitti.add_argument("--frames", type=int, default=50)
    hilti = sub.add_parser("hilti")
    tool = _hilti_tool()
    tool.add_input_arguments(hilti)
    for item in (kitti, hilti):
        item.add_argument("--fixed-scale", type=float, required=True)
    args = parser.parse_args(argv)

    stereo: RectifiedStereo
    if args.dataset == "kitti":
        pairs = select_kitti_pairs(args.drive, args.frames, args.selection)
        edge_frames = kitti_edge_frames(pairs)
        reference_se3 = read_velodyne_to_camera_transform(args.drive)
        if reference_se3 is None:
            parser.error("calib_velo_to_cam.txt could not be parsed")
        _, transforms = kitti_edge_candidates(args.drive, {"dataset_reference": reference_se3})
        maybe_stereo = kitti_stereo(args.drive)
        frames = kitti_stereo_frames(pairs, edge_frames) if maybe_stereo else None
        if maybe_stereo is None or frames is None:
            parser.error("the drive has no camera 3 images or projection")
        stereo, reference = maybe_stereo, transforms["dataset_reference"]
    else:
        setup = tool.load_setup(args)
        loaded = tool.load_candidate_frames(args, setup, setup.reference)
        vertical = tool.measure_vertical_offset(setup.rectified, loaded)
        rectified = (
            setup.rectified
            if vertical is None
            else tool.with_right_row_shift(setup.rectified, vertical)
        )
        frames = tool.stereo_frames(rectified, loaded)
        stereo, reference = rectified.stereo, setup.stereo_from_camera @ setup.reference

    options = StereoDepthOptions()
    residuals: dict[float, Residuals] = {}
    for shift in SHIFTS_M:
        moved = reference.copy()
        moved[2, 3] += shift
        residuals[shift] = [frame_residuals(frame, stereo, moved, options) for frame in frames]
    base = stereo.base_px_m
    depth = np.concatenate([item[0] for item in residuals[0.0]])
    print(
        f"{len(frames)} frames, {len(depth)} points, depth p10/p50/p90 "
        f"{np.round(np.percentile(depth, [10, 50, 90]), 1).tolist()} m"
    )
    report("current", residuals, base)
    free = {shift: estimate(items, base, scale=True) for shift, items in residuals.items()}
    reference_fit, std = free[0.0]
    print(
        f"free scale: offset {reference_fit[1]:+.3f} m, std {std:.3f} m, scale "
        f"{reference_fit[2] * 100:+.2f}%, control "
        f"{(free[0.1][0][1] - reference_fit[1]) / 0.1:.2f}, +0.25 m reads "
        f"{free[0.25][0][1]:+.3f} m"
    )
    fixed = {
        shift: [(d, r - args.fixed_scale * base / d) for d, r in items]
        for shift, items in residuals.items()
    }
    report(f"fixed scale {args.fixed_scale * 100:+.2f}%", fixed, base)
    return 0


def _hilti_tool() -> ModuleType:
    path = Path(__file__).with_name("hilti_stereo_falsification.py")
    spec = importlib.util.spec_from_file_location("hilti_stereo_falsification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    sys.exit(main())
