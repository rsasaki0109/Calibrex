#!/usr/bin/env python3
"""Where a Hilti 2022 recording's rectified vertical offset comes from.

An evaluation harness, not part of the Calibrex package. The stereo decision
(``tools/hilti_stereo_falsification.py``) removes one row offset per recording,
measured from the rectified images. This tool samples every frame's matched patches
(:func:`calibrex.evaluation.fisheye_stereo.vertical_offset_samples`) and separates the
explanations:

* **a relative pitch of the cameras** shifts every row by the same amount (constant);
* **a vertical baseline component** shifts rows in proportion to disparity, so near
  scenes read more than far ones;
* **a relative roll** shifts rows in proportion to the column;
* **a time offset between the cameras** shifts rows with the rig's angular rate about
  the baseline, frame by frame.

It fits ``offset = a + b * disparity + c * (u - cx)/100 + d * (v - cy)/100`` on the
patches whose texture constrains rows, with a block jackknife over frames (6 blocks),
and correlates each frame's median offset with the gyro's rate in the rectified frame.

::

    python tools/hilti_vertical_offset.py exp21_head.bag \\
        --camchain calib_3_cam0-1-camchain-imucam.yaml \\
        --lidar-calibration lidar_calibration.yaml --output exp21_vertical.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np

from calibrex.core.provenance import git_commit, sha256_path
from calibrex.evaluation.fisheye_stereo import VERTICAL_OFFSET_COLUMNS, vertical_offset_samples

BLOCKS = 6
TERMS = ("constant_px", "per_disparity_px", "per_100_columns_px", "per_100_rows_px")


def robust_fit(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Least squares, four passes trimming beyond 3 robust standard deviations."""

    keep = np.ones(len(values), dtype=bool)
    coefficients = np.zeros(design.shape[1])
    for _ in range(4):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        error = values - design @ coefficients
        spread = 1.4826 * float(np.median(np.abs(error[keep])))
        keep = np.abs(error) <= 3.0 * max(spread, 1.0e-6)
    return coefficients


def main(argv: list[str] | None = None) -> int:
    tool = _hilti_tool()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    tool.add_input_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    commit = git_commit()
    setup = tool.load_setup(args)
    # the images and their stamps are all this needs: no candidate moves them
    frames = tool.load_candidate_frames(
        args, tool.HiltiSetup(**{**setup.__dict__, "motion": None}), setup.reference
    )
    rectified = setup.rectified
    pairs = [
        (
            rectified.rectify_left(frame.image.astype(np.float64)),
            rectified.rectify_right(right.astype(np.float64)),
        )
        for frame, right in zip(frames.frames, frames.right_images, strict=True)
    ]
    samples = vertical_offset_samples(pairs)
    column = {name: samples[:, i] for i, name in enumerate(VERTICAL_OFFSET_COLUMNS)}
    rows_known = column["vertical_texture"] > 0.5
    design = np.c_[
        np.ones(len(samples)),
        column["disparity_px"],
        (column["column_px"] - rectified.cx) / 100.0,
        (column["row_px"] - rectified.cy) / 100.0,
    ][rows_known]
    offsets = column["offset_px"][rows_known]
    pair = column["pair"][rows_known]
    fit = robust_fit(design, offsets)
    blocks = np.array_split(np.arange(len(pairs)), BLOCKS)
    jackknife = np.array(
        [robust_fit(design[~np.isin(pair, b)], offsets[~np.isin(pair, b)]) for b in blocks]
    )
    std = np.sqrt((BLOCKS - 1) / BLOCKS * ((jackknife - jackknife.mean(axis=0)) ** 2).sum(axis=0))

    per_frame = np.array(
        [
            np.median(offsets[pair == i]) if (pair == i).sum() >= 10 else np.nan
            for i in range(len(pairs))
        ]
    )
    used = np.isfinite(per_frame)
    half = len(pairs) // 2
    rate: dict[str, object] = {}
    if setup.motion is not None:
        to_rectified = rectified.rectified_from_left @ setup.reference[:3, :3]
        omega = setup.motion.angular_rad_s @ to_rectified.T
        times = np.array([frame.image_time_s for frame in frames.frames])
        about_baseline = np.interp(times, setup.motion.times_s, omega[:, 0])
        slope = np.polyfit(about_baseline[used], per_frame[used], 1)[0]
        rate = {
            "max_abs_rate_deg_s": float(np.degrees(np.abs(about_baseline[used]).max())),
            "correlation": float(np.corrcoef(about_baseline[used], per_frame[used])[0, 1]),
            "implied_time_offset_ms": float(slope / rectified.fx * 1.0e3),
        }
    start = datetime.fromtimestamp(frames.frames[0].image_time_s, timezone.utc)
    result = {
        "recording_start_utc": start.isoformat(),
        "frames": len(pairs),
        "patches": len(samples),
        "patches_constraining_rows": int(rows_known.sum()),
        "median_offset_px": float(np.median(column["offset_px"])) if len(samples) else None,
        "fit": {name: float(value) for name, value in zip(TERMS, fit, strict=True)},
        "fit_jackknife_std": {name: float(value) for name, value in zip(TERMS, std, strict=True)},
        "per_frame_median_std_px": float(np.nanstd(per_frame)),
        "first_half_median_px": float(np.nanmedian(per_frame[:half])),
        "second_half_median_px": float(np.nanmedian(per_frame[half:])),
        "rate_about_baseline": rate,
        "provenance": {
            "git_commit": commit,
            "bag": {"name": args.bag.name, "size_bytes": args.bag.stat().st_size},
            "camchain_sha256": sha256_path(args.camchain),
            "lidar_calibration_sha256": sha256_path(args.lidar_calibration),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))
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
