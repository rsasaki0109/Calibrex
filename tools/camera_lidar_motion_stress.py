#!/usr/bin/env python3
"""Camera-LiDAR check on an emulated unsynchronized camera, with and without rig motion.

An evaluation harness, not part of the Calibrex package. KITTI's camera is triggered by
its LiDAR, so a scan and its image are 10 ms apart and the rig hardly moves in between.
An unsynchronized camera can fire up to half a frame away. This tool pairs every scan
with the image ``--image-offset`` frames later (or earlier), so each pair is a real
image and a real scan recorded about 0.1 s apart on the moving car. It then runs the
camera-lidar edge alignment twice on the same frames: on the scans as recorded, and on
the scans moved to the image time with the bag's velocity topic
(:mod:`calibrex.evaluation.camera_lidar_motion`). Each run is judged with the check's
per-axis rule::

    python tools/camera_lidar_motion_stress.py data/public/kitti_raw/bags/kitti_0005 \\
        --image-offset 1 --rotation-deg 0,0,3

The candidate is the bag's ``/tf_static`` (plus ``--tf`` files) between the image and
LiDAR frames, optionally rotated by ``--rotation-deg`` (roll, pitch, yaw about the camera
axes, applied on the left, as in the check benchmark).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.check.estimators import (
    camera_lidar_axis_estimates,
    rotation_errors_deg,
    transform_matrix,
)
from calibrex.check.tf_sources import load_bag_tf_static, load_tf_file, merge_sources
from calibrex.check.verdict import VerdictOptions, judge_pair
from calibrex.data import ros_cdr
from calibrex.data.ros2_camera_imu import to_gray
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.data.rosbag2 import iter_messages
from calibrex.evaluation.camera_lidar_edge import (
    EdgeAlignmentOptions,
    EdgeCamera,
    default_provenance,
    estimate_from_frames,
)
from calibrex.evaluation.camera_lidar_motion import (
    move_frames_to_image_time,
    read_lidar_motion,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--image-topic", default="/camera/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera_info")
    parser.add_argument("--lidar-topic", default="/velodyne_points")
    parser.add_argument("--motion-topic", default="/oxts/odometry")
    parser.add_argument("--tf", type=Path, action="append", default=[])
    parser.add_argument("--image-offset", type=int, default=1)
    parser.add_argument("--rotation-deg", default="0,0,0", help="roll,pitch,yaw perturbation")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--output", type=Path, help="write both runs as JSON")
    args = parser.parse_args(argv)

    images: list[tuple[int, bytes]] = []
    scans: list[tuple[int, str, np.ndarray]] = []
    info = None
    types: dict[str, str] = {}
    topics = {args.image_topic, args.lidar_topic, args.camera_info_topic, args.motion_topic}
    for connection, timestamp_ns, payload in iter_messages(args.bag, topics=topics):
        types[connection.topic] = connection.message_type
        if connection.topic == args.image_topic:
            header = ros_cdr.decode_ros2_image(
                connection.topic, timestamp_ns, payload, include_data=False
            )
            images.append((header.timestamp_ns, payload))
        elif connection.topic == args.lidar_topic:
            cloud = ros_cdr.decode_ros2_pointcloud2(connection.topic, timestamp_ns, payload)
            scans.append((cloud.timestamp_ns, cloud.frame_id, np.asarray(cloud.xyz, np.float32)))
        elif connection.topic == args.camera_info_topic and info is None:
            info = ros_cdr.decode_ros2_camera_info(connection.topic, timestamp_ns, payload)
    if info is None or not images or not scans:
        parser.error("the bag lacks images, camera info or scans on the given topics")
    images.sort(key=lambda item: item[0])
    scans.sort(key=lambda item: item[0])
    image_times = np.asarray([stamp for stamp, _ in images])

    frames: list[CameraLidarFrame] = []
    for index in np.unique(np.round(np.linspace(0, len(scans) - 1, args.frames)).astype(int)):
        stamp, cloud_frame, points = scans[index]
        paired = int(np.argmin(np.abs(image_times - stamp))) + args.image_offset
        if not 0 <= paired < len(images):
            continue
        image = ros_cdr.decode_ros2_image(args.image_topic, images[paired][0], images[paired][1])
        assert image.data is not None
        frames.append(
            CameraLidarFrame(
                time_s=stamp * 1.0e-9,
                image_time_s=images[paired][0] * 1.0e-9,
                image=to_gray(
                    bytes(image.data),
                    height=image.height,
                    width=image.width,
                    step=image.step,
                    encoding=image.encoding,
                ),
                points=points,
                cloud_frame_id=cloud_frame,
                image_frame_id=image.frame_id,
            )
        )
    camera_frame, lidar_frame = frames[0].image_frame_id, frames[0].cloud_frame_id

    tree, _ = merge_sources(load_bag_tf_static(args.bag), [load_tf_file(path) for path in args.tf])
    reference_relation = tree.lookup(camera_frame, lidar_frame)
    if reference_relation is None:
        parser.error(f"{camera_frame} and {lidar_frame} are not connected in the frame tree")
    reference = transform_matrix(
        reference_relation.translation_m, reference_relation.rotation_quat_xyzw
    )
    rotation = [float(value) for value in args.rotation_deg.split(",")]
    candidate = reference.copy()
    candidate[:3, :3] = Rotation.from_rotvec(np.radians(rotation)).as_matrix() @ reference[:3, :3]

    motion_frame = _motion_frame(args.bag, args.motion_topic, types[args.motion_topic])
    body_relation = tree.lookup(motion_frame, lidar_frame)
    if body_relation is None:
        parser.error(f"{motion_frame} and {lidar_frame} are not connected in the frame tree")
    motion = read_lidar_motion(
        args.bag,
        args.motion_topic,
        types[args.motion_topic],
        transform_matrix(body_relation.translation_m, body_relation.rotation_quat_xyzw),
    )
    if motion is None:
        parser.error(f"{args.motion_topic} has no messages")

    k = info.k
    camera = EdgeCamera(k[0], k[4], k[2], k[5])
    provenance = default_provenance(
        bag=args.bag,
        image_topic=args.image_topic,
        lidar_topic=args.lidar_topic,
        dataset_family="camera-lidar motion stress",
        dataset_license="see the dataset",
        command=("tools/camera_lidar_motion_stress.py", *(argv or sys.argv[1:])),
    )
    gaps = np.asarray([frame.gap_s for frame in frames]) * 1.0e3
    print(
        f"{len(frames)} pairs, image offset {args.image_offset:+d} frame(s): image minus scan "
        f"time median {np.median(gaps):.1f} ms (range {gaps.min():.1f} to {gaps.max():.1f}); "
        f"candidate rotated by {rotation} deg"
    )
    runs: dict[str, object] = {}
    moved, summary = move_frames_to_image_time(frames, motion)
    for label, used, motion_summary in (
        ("as recorded", frames, None),
        (f"moved with {args.motion_topic}", moved, summary),
    ):
        artifact = estimate_from_frames(
            used,
            camera,
            candidate,
            options=EdgeAlignmentOptions(),
            provenance=provenance,
            parent_frame=camera_frame,
            child_frame=lidar_frame,
            motion_compensation=motion_summary,
        )
        judgement = judge_pair(camera_lidar_axis_estimates(artifact, candidate), VerdictOptions())
        estimate = (
            transform_matrix(
                artifact.transform.translation_m, artifact.transform.rotation_quat_xyzw
            )
            if artifact.transform is not None
            else candidate
        )
        from_reference = rotation_errors_deg(estimate[:3, :3], reference[:3, :3])
        axes = "; ".join(
            f"{axis.name} {abs(axis.candidate_error):.2f}/{axis.tolerance:.2f}"
            for axis in judgement.axes
        )
        unchecked = ", ".join(
            item.name for item in judgement.unchecked if item.name in {"roll", "pitch", "yaw"}
        )
        print(
            f"  {label}: {judgement.verdict} ({artifact.samples.frames} frames) | {axes or '-'}"
            f" | unchecked rotation: {unchecked or '-'} | estimate minus reference "
            f"{np.round(from_reference, 2).tolist()} deg"
        )
        runs[label] = {
            "verdict": judgement.verdict,
            "frames": artifact.samples.frames,
            "axes": {
                axis.name: {"error": axis.candidate_error, "tolerance": axis.tolerance}
                for axis in judgement.axes
            },
            "unchecked": [item.name for item in judgement.unchecked],
            "estimate_minus_reference_deg": list(from_reference),
            "motion_compensation": summary.model_dump() if motion_summary is not None else None,
        }
    if args.output is not None:
        args.output.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return 0


def _motion_frame(bag: Path, topic: str, message_type: str) -> str:
    """The frame a velocity topic's twist is expressed in (child frame for odometry)."""

    for _connection, timestamp_ns, payload in iter_messages(bag, topics={topic}):
        if message_type == "nav_msgs/msg/Odometry":
            return ros_cdr.decode_ros2_odometry(topic, timestamp_ns, payload).child_frame_id
        return ros_cdr.decode_ros2_header_frame_id(payload)
    raise SystemExit(f"{topic} has no messages")


if __name__ == "__main__":
    sys.exit(main())
