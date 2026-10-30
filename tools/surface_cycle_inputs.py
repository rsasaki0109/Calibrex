"""Reproduce the frozen1px reverse test on archived three-frame SGBM inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tools.compare_stereo_cycle_filter import mask_depth
from tools.diagnose_stereo_cycle import round_trip_error
from tools.run_stereo_translation_refinement import digest
from tools.stereo_rerectification import restore_left_depth

from calibrex.data.kitti import read_calibration_file


def prepare(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    meta = json.loads(Path(frame["source_metadata"]).read_text())
    images = [Path(p) for p in meta["source_sha256"] if p.endswith(".png")]
    cam_path = next(Path(p) for p in meta["source_sha256"] if p.endswith("calib_cam_to_cam.txt"))
    cam = read_calibration_file(cam_path)
    projections = [np.array(cam["P_rect_" + c]).reshape(3, 4) for c in ["02", "03"]]
    g = frame["geometry"]
    if cv2.__version__ != g["opencv_version"]:
        raise ValueError("OpenCV version changed")
    rotations = [np.array(g[k]) for k in ["r1", "r2"]]
    new_projections = [np.array(g[k]) for k in ["p1", "p2"]]
    ims = []
    masks = []
    for path, p, r, new in zip(images, projections, rotations, new_projections, strict=True):
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        x, y = cv2.initUndistortRectifyMap(
            p[:, :3], np.zeros(5), r, new[:, :3], (w, h), cv2.CV_32FC1
        )
        ims.append(cv2.remap(im, x, y, cv2.INTER_LINEAR))
        masks.append((x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1))
    forward = cv2.StereoSGBM_create(**frame["settings"]).compute(*ims).astype(float) / 16
    # Horizontal flip converts right-to-left negative disparity to the same positive range.
    reverse = (
        cv2.StereoSGBM_create(**frame["settings"])
        .compute(np.ascontiguousarray(ims[1][:, ::-1]), np.ascontiguousarray(ims[0][:, ::-1]))
        .astype(float)[:, ::-1]
        / 16
    )
    v, u = np.indices(forward.shape)
    ur = np.rint(u - forward).astype(int)
    valid = (forward > 0) & masks[0] & (ur >= 0) & (ur < w)
    valid &= masks[1][v, np.clip(ur, 0, w - 1)]
    base = new_projections[0][0, 3] - new_projections[1][0, 3]
    depth = np.full(forward.shape, np.nan)
    depth[valid] = base / forward[valid]
    depth[(depth < 0.1) | (depth > 80)] = np.nan
    restored = restore_left_depth(
        depth, rotations[0], projections[0][:, :3], new_projections[0][:, :3]
    )
    saved_path = next(Path(p) for p in frame["depth_sha256"] if p.endswith("-sgbm.npy"))
    if digest(saved_path) != frame["depth_sha256"][str(saved_path)]:
        raise ValueError("saved depth changed")
    np.testing.assert_array_equal(restored, np.load(saved_path, allow_pickle=False))
    # Reject reverse endpoints outside the image remap support too.
    ul = np.rint(u + reverse).astype(int)
    reverse[~(masks[1] & (ul >= 0) & (ul < w) & masks[0][v, np.clip(ul, 0, w - 1)])] = np.nan
    cycle = round_trip_error(forward, reverse)

    filtered = restore_left_depth(
        mask_depth(depth, cycle), rotations[0], projections[0][:, :3], new_projections[0][:, :3]
    )
    keep = np.isfinite(filtered)
    np.testing.assert_array_equal(filtered[keep], restored[keep])
    return restored, filtered
