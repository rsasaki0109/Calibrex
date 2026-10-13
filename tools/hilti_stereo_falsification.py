#!/usr/bin/env python3
"""Data-only camera-LiDAR falsification on a Hilti 2022 bag with its fisheye stereo pair.

An evaluation harness, not part of the Calibrex package. It repeats the KITTI data-only
decision (:mod:`calibrex.evaluation.kitti_edge_falsification`) on a ROS bag whose camera
pair is an equidistant fisheye (Hilti 2022's alphasense cam0 and cam1, Kalibr camchain):

* **candidates**: the reference ``T_cam0_lidar`` composed from the camchain (cam0 to IMU)
  and ``lidar_calibration.yaml`` (IMU to PandarXT-32), and a known-bad declared here,
  before scoring: the reference turned 5 deg about the camera's y axis (pitch) and moved
  0.25 m along its optical axis, the same perturbation in camera terms as KITTI's;
* **frames**: the bag-level check's own selection of cam0 images and scans, with the
  cam1 image taken at the same stamp (the pair is hardware-synchronised);
* **rotation** from edge alignment on the raw fisheye cam0 images (equidistant model);
* **translation along the optical axis** from stereo depth on the rectified pair
  (:mod:`calibrex.evaluation.fisheye_stereo`), after removing the pair's vertical offset
  as the rectified images measure it.

::

    python tools/hilti_stereo_falsification.py exp21_head.bag \\
        --camchain calib_3_cam0-1-camchain-imucam.yaml \\
        --lidar-calibration lidar_calibration.yaml --output hilti_exp21.json

The scans are not moved to the image time (no motion compensation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.check.estimators import transform_matrix
from calibrex.check.tf_sources import load_tf_file, merge_sources
from calibrex.core.camera_lidar_edge import CameraLidarEdgeProvenance
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data import ros_cdr
from calibrex.data.ros2_camera_imu import to_gray
from calibrex.data.ros2_camera_lidar import load_camera_lidar_frames
from calibrex.data.rosbag2 import iter_messages
from calibrex.evaluation.camera_lidar_edge import EdgeAlignmentOptions, EdgeCamera
from calibrex.evaluation.fisheye_stereo import (
    estimate_vertical_offset,
    read_kalibr_fisheye_pair,
    rectify_fisheye_pair,
    with_right_row_shift,
)
from calibrex.evaluation.kitti_edge_falsification import (
    _decision,
    judge_candidate,
    perturbed_axes,
)
from calibrex.evaluation.stereo_lidar_depth import StereoFrame

KNOWN_BAD_PITCH_DEG = 5.0
KNOWN_BAD_DEPTH_M = 0.25


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--camchain", type=Path, required=True)
    parser.add_argument("--lidar-calibration", type=Path, required=True)
    parser.add_argument("--left-topic", default="/alphasense/cam0/image_raw")
    parser.add_argument("--right-topic", default="/alphasense/cam1/image_raw")
    parser.add_argument("--lidar-topic", default="/hesai/pandar")
    parser.add_argument("--lidar-frame", default="PandarXT-32")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    commit = git_commit()
    left, right, right_from_left = read_kalibr_fisheye_pair(args.camchain)
    rectified = rectify_fisheye_pair(left, right, right_from_left)
    tree, _ = merge_sources(
        None, [load_tf_file(args.camchain), load_tf_file(args.lidar_calibration)]
    )
    camera_frame = args.left_topic.lstrip("/")
    relation = tree.lookup(camera_frame, args.lidar_frame)
    if relation is None:
        parser.error(f"{camera_frame} and {args.lidar_frame} are not connected")
    reference = transform_matrix(relation.translation_m, relation.rotation_quat_xyzw)
    known_bad = reference.copy()
    known_bad[:3, :3] = (
        Rotation.from_rotvec([0.0, np.radians(KNOWN_BAD_PITCH_DEG), 0.0]).as_matrix()
        @ reference[:3, :3]
    )
    known_bad[2, 3] += KNOWN_BAD_DEPTH_M
    candidates = {"dataset_reference": reference, "known_bad": known_bad}

    options = EdgeAlignmentOptions(max_seconds=args.max_seconds)
    selection: dict[str, object] = {}
    frames = load_camera_lidar_frames(
        args.bag,
        args.left_topic,
        args.lidar_topic,
        spacing_s=options.effective_spacing_s,
        short_recording_spacing_s=options.short_recording_spacing_s,
        short_recording_min_view_change=options.short_recording_min_view_change,
        report=selection,
        max_frames=options.max_frames,
        max_seconds=options.max_seconds,
        max_gap_s=options.max_gap_s,
    )
    wanted = {round(frame.image_time_s * 1.0e9): index for index, frame in enumerate(frames)}
    right_images: dict[int, np.ndarray] = {}
    for _, stamp, payload in iter_messages(args.bag, topics={args.right_topic}):
        header = ros_cdr.decode_ros2_image(args.right_topic, stamp, payload, include_data=False)
        match = next((k for k in wanted if abs(k - header.timestamp_ns) < 2_000_000), None)
        if match is None:
            continue
        image = ros_cdr.decode_ros2_image(args.right_topic, stamp, payload)
        assert image.data is not None
        right_images[wanted[match]] = to_gray(
            bytes(image.data),
            height=image.height,
            width=image.width,
            step=image.step,
            encoding=image.encoding,
        )
        if len(right_images) == len(frames):
            break
    keep = sorted(right_images)
    frames = [frames[index] for index in keep]
    pairs = [
        (
            rectified.rectify_left(frame.image.astype(np.float64)),
            rectified.rectify_right(right_images[index].astype(np.float64)),
        )
        for frame, index in zip(frames, keep, strict=True)
    ]
    vertical = estimate_vertical_offset(pairs[:: max(1, len(pairs) // 8)])
    if vertical is not None:
        rectified = with_right_row_shift(rectified, vertical)
    stereo_frames = [
        StereoFrame(
            left=rectified.rectify_left(frame.image.astype(np.float64)),
            right=rectified.rectify_right(right_images[index].astype(np.float64)),
            points=np.asarray(frame.points, dtype=np.float64),
        )
        for frame, index in zip(frames, keep, strict=True)
    ]
    stereo_from_camera = np.eye(4)
    stereo_from_camera[:3, :3] = rectified.rectified_from_left
    camera = EdgeCamera(left.fx, left.fy, left.cx, left.cy, "equidistant", left.distortion)
    provenance = CameraLidarEdgeProvenance(
        generator="tools/hilti_stereo_falsification.py",
        generator_version="0.1",
        dataset_family="hilti_2022",
        sequence_ids=[args.bag.stem],
        image_topic=args.left_topic,
        lidar_topic=args.lidar_topic,
        input_sha256=sha256_path(args.camchain) or "0" * 64,
        input_digest_scope="the Kalibr camchain (the bag is named and sized below)",
        dataset_license="CC BY-NC-SA 3.0",
    )
    trials = [
        judge_candidate(
            frames,
            camera,
            candidate_id,  # type: ignore[arg-type]
            transform,
            provenance=provenance,
            parent_frame=camera_frame,
            child_frame=args.lidar_frame,
            stereo=rectified.stereo,
            stereo_frames=stereo_frames,
            stereo_from_camera=stereo_from_camera,
            options=options,
        )
        for candidate_id, transform in candidates.items()
    ]
    declared = perturbed_axes(reference, known_bad)
    status, reason = _decision(trials, declared)
    result = {
        "status": status,
        "reason": reason,
        "known_bad_axes": declared,
        "known_bad": f"reference turned {KNOWN_BAD_PITCH_DEG:g} deg about the camera's y axis "
        f"and moved {KNOWN_BAD_DEPTH_M:g} m along its optical axis, declared before scoring",
        "frames": len(frames),
        "frame_selection": selection.get("selection"),
        "vertical_offset_px": vertical,
        "baseline_m": rectified.baseline_m,
        "trials": [trial.model_dump(mode="json") for trial in trials],
        "provenance": {
            "git_commit": commit,
            "bag": {"name": args.bag.name, "size_bytes": args.bag.stat().st_size},
            "camchain_sha256": sha256_path(args.camchain),
            "lidar_calibration_sha256": sha256_path(args.lidar_calibration),
            "motion_compensation": "none",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{status}: {reason}; wrote {args.output}")
    for trial in trials:
        depth = trial.depth_axis
        print(
            f"  {trial.candidate_id}: {trial.verdict}; "
            + ", ".join(
                f"{a.name} {a.candidate_error_deg:+.2f}/{a.tolerance_deg:.2f} {a.status}"
                for a in trial.axes
            )
            + (
                f"; z {depth.candidate_error_m:+.3f}/{depth.tolerance_m:.3f} m {depth.status}"
                if depth
                else "; z unchecked"
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
