#!/usr/bin/env python3
"""Inject known timing errors into a synchronised stereo bag and measure them back.

An evaluation harness, not part of the Calibrex package. The camera-camera check fits a
stereo pair's rotation error together with the cameras' clock offset when a gyro gives
the rig's rotation (:class:`calibrex.evaluation.stereo_rotation.StereoTiming`). A
hardware-synchronised bag (Hilti 2022's alphasense) has no timing error of its own, so
this tool makes them, with the candidate from the Kalibr camchain:

* ``synchronised``: each left image with the right image of the same stamp;
* ``one frame late``: each left image with the *next* right image, stamped as it was
  taken, so a known gap the fit must turn out with the gyro;
* ``stamp offset``: the synchronised pairs with the right stamps moved by
  ``--stamp-offset-ms``, so a clock offset of minus that the fit must find.

Each scenario is estimated with and without the gyro ("blind").

::

    python tools/stereo_timing_injection.py exp21_head.bag \\
        --camchain calib_3_cam0-1-camchain-imucam.yaml --output exp21_timing.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from calibrex.check.estimators import transform_matrix
from calibrex.check.tf_sources import camchain_entries, load_tf_file, merge_sources
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.stereo_rotation import StereoRotationArtifact, StereoRotationProvenance
from calibrex.data import ros_cdr
from calibrex.data.rosbag2 import iter_messages
from calibrex.evaluation.camera_lidar_motion import IMU_TYPE, read_lidar_motion
from calibrex.evaluation.stereo_rotation import (
    StereoRotationOptions,
    StereoTiming,
    _gray,
    estimate_stereo_rotation,
    stereo_camera,
)
from calibrex.evaluation.visual_rotation import CameraModel


def read_frames(
    bag: Path, left_topic: str, right_topic: str, spacing_s: float, count: int, shift: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """``count`` left images ``spacing_s`` apart, each with the right image ``shift`` later."""

    stamps: dict[str, list[int]] = {left_topic: [], right_topic: []}
    for connection, stamp, payload in iter_messages(bag, topics={left_topic, right_topic}):
        header = ros_cdr.decode_ros2_image(connection.topic, stamp, payload, include_data=False)
        stamps[connection.topic].append(header.timestamp_ns)
        if len(stamps[right_topic]) > (count + 2) * spacing_s * 60:
            break
    rights = np.asarray(stamps[right_topic])
    wanted: dict[int, int] = {}  # left stamp -> right stamp
    last = None
    for left in stamps[left_topic]:
        if last is not None and left - last < spacing_s * 1.0e9:
            continue
        same = int(np.argmin(np.abs(rights - left)))
        if abs(int(rights[same]) - left) > 2_000_000 or same + shift >= len(rights):
            continue
        wanted[left] = int(rights[same + shift])
        last = left
        if len(wanted) == count:
            break
    needed = {(left_topic, a) for a in wanted} | {(right_topic, b) for b in wanted.values()}
    images: dict[tuple[str, int], np.ndarray] = {}
    for connection, stamp, payload in iter_messages(bag, topics={left_topic, right_topic}):
        header = ros_cdr.decode_ros2_image(connection.topic, stamp, payload, include_data=False)
        key = (connection.topic, header.timestamp_ns)
        if key in needed and key not in images:
            images[key] = _gray(connection.topic, stamp, payload)
            if len(images) == len(needed):
                break
    pairs = [(images[(left_topic, a)], images[(right_topic, b)]) for a, b in wanted.items()]
    lefts = np.array(list(wanted), dtype=np.float64) * 1.0e-9
    rights_s = np.array(list(wanted.values()), dtype=np.float64) * 1.0e-9
    return pairs, lefts, rights_s


def summary(artifact: StereoRotationArtifact) -> dict[str, object]:
    return {
        "policy_status": artifact.policy_status,
        "axes": {
            axis.name: {
                "error_deg": round(axis.candidate_error_deg, 4),
                "std_deg": round(axis.std_deg, 4),
                "observable": axis.observable,
            }
            for axis in artifact.axes
        },
        "time_offset_ms": None
        if artifact.time_offset_s is None
        else round(artifact.time_offset_s * 1e3, 3),
        "time_offset_std_ms": None
        if artifact.time_offset_std_s is None
        else round(artifact.time_offset_std_s * 1e3, 3),
        "time_offset_status": artifact.time_offset_status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--camchain", type=Path, required=True)
    parser.add_argument("--left-topic", default="/alphasense/cam0/image_raw")
    parser.add_argument("--right-topic", default="/alphasense/cam1/image_raw")
    parser.add_argument("--imu-topic", default="/alphasense/imu")
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--spacing-s", type=float, default=1.0)
    parser.add_argument("--stamp-offset-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    commit = git_commit()
    tree, _ = merge_sources(None, [load_tf_file(args.camchain)])
    left_frame, right_frame = args.left_topic.lstrip("/"), args.right_topic.lstrip("/")
    relation = tree.lookup(left_frame, right_frame)
    imu = tree.lookup("imu", left_frame)
    if relation is None or imu is None:
        parser.error("the camchain does not connect the two cameras and the IMU")
    candidate = transform_matrix(relation.translation_m, relation.rotation_quat_xyzw)
    models = {
        frame: CameraModel.from_kalibr(entry) for _, frame, entry in camchain_entries(args.camchain)
    }
    motion = read_lidar_motion(
        args.bag,
        args.imu_topic,
        IMU_TYPE,
        transform_matrix(imu.translation_m, imu.rotation_quat_xyzw),
    )
    if motion is None:
        parser.error(f"{args.imu_topic} has no samples")
    options = StereoRotationOptions(max_pairs=args.pairs)
    provenance = StereoRotationProvenance(
        generator="tools/stereo_timing_injection.py",
        generator_version="0.1",
        git_commit=commit,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        left_frame=left_frame,
        right_frame=right_frame,
        intrinsics_source=args.camchain.name,
    )
    scenarios: dict[str, dict[str, object]] = {}
    for name, shift, offset_s in (
        ("synchronised", 0, 0.0),
        ("one frame late", 1, 0.0),
        ("stamp offset", 0, args.stamp_offset_ms * 1e-3),
    ):
        pairs, lefts, rights = read_frames(
            args.bag, args.left_topic, args.right_topic, args.spacing_s, args.pairs, shift
        )
        height, width = pairs[0][0].shape
        left = stereo_camera(models[left_frame], width, height)
        right = stereo_camera(models[right_frame], width, height)
        sampled = [motion.velocity_at(float(time)) for time in lefts]
        keep = [index for index, rate in enumerate(sampled) if rate is not None]
        timing = StereoTiming(
            gaps_s=(rights - lefts + offset_s)[keep],
            rates_rad_s=np.array([rate[0] for rate in sampled if rate is not None]),
            source=f"gyro {args.imu_topic}",
        )
        kept = [pairs[index] for index in keep]
        timed = estimate_stereo_rotation(
            kept, left, right, candidate, provenance=provenance, options=options, timing=timing
        )
        blind = estimate_stereo_rotation(
            kept, left, right, candidate, provenance=provenance, options=options
        )
        rates = np.degrees(np.linalg.norm(timing.rates_rad_s, axis=1))
        scenarios[name] = {
            "pairs": len(kept),
            "stamped_gap_ms": round(float(np.median(timing.gaps_s)) * 1e3, 3),
            "injected_time_offset_ms": -args.stamp_offset_ms if offset_s else 0.0,
            "rate_deg_s": {
                "median": round(float(np.median(rates)), 2),
                "max": round(float(rates.max()), 2),
            },
            "with_gyro": summary(timed),
            "blind": summary(blind),
        }
        print(name, json.dumps(scenarios[name]))
    result = {
        "scenarios": scenarios,
        "provenance": {
            "git_commit": commit,
            "bag": {"name": args.bag.name, "size_bytes": args.bag.stat().st_size},
            "camchain_sha256": sha256_path(args.camchain),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
