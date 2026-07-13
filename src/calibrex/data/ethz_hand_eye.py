"""Reader and leakage-safe motion builder for ETHZ ASL hand-eye datasets."""

from __future__ import annotations

import bisect
import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from calibrex.core.geometry import SE3

ETHZ_HAND_EYE_COMMIT = "966cd92518f24aa7dfdacc8ba9c5fa4a441270cd"
ETHZ_ROBOT_ARM_REAL_ARCHIVE = "robot_arm_w_color_camera_real.zip"
ETHZ_ROBOT_ARM_REAL_SHA256 = (
    "2454578f731e656a940ddf51017b56e8d535c58d16a50629b85326005dedd6c3"
)
ETHZ_ROBOT_ARM_REAL_URL = (
    "https://raw.githubusercontent.com/ethz-asl/hand_eye_calibration/"
    f"{ETHZ_HAND_EYE_COMMIT}/datasets/{ETHZ_ROBOT_ARM_REAL_ARCHIVE}"
)
ETHZ_HAND_MEMBER = (
    "robot_arm/robot_arm_complete_bag_color_and_ir_base_link_sr300_hinge.csv"
)
ETHZ_EYE_MEMBER = "robot_arm/robot_arm_complete_bag_color_and_ir_target_ir.csv"


@dataclass(frozen=True)
class TimestampedPose:
    timestamp_sec: float
    transform: SE3
    source_index: int


@dataclass(frozen=True)
class RelativeMotionPair:
    pair_id: str
    motion_a: SE3
    motion_b: SE3


@dataclass(frozen=True)
class AlignedHandEyePosePair:
    """One-to-one absolute poses satisfying ``pose_a X = Y pose_b``."""

    pair_id: str
    pose_a: SE3
    pose_b: SE3
    alignment_delta_sec: float
    hand_source_index: int
    eye_source_index: int


@dataclass(frozen=True)
class HandEyeMotionDataset:
    motions: tuple[RelativeMotionPair, ...]
    absolute_pose_pairs: tuple[AlignedHandEyePosePair, ...]
    hand_pose_count: int
    eye_pose_count: int
    aligned_pose_count: int
    maximum_alignment_delta_sec: float | None
    motion_stride: int
    absolute_pose_reuse_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "motion_count": len(self.motions),
            "robot_world_pose_pair_count": len(self.absolute_pose_pairs),
            "hand_pose_count": self.hand_pose_count,
            "eye_pose_count": self.eye_pose_count,
            "aligned_pose_count": self.aligned_pose_count,
            "maximum_alignment_delta_sec": self.maximum_alignment_delta_sec,
            "motion_stride": self.motion_stride,
            "absolute_pose_reuse_count": self.absolute_pose_reuse_count,
            "pairing_policy": "disjoint absolute-pose blocks",
            "robot_world_pairing_policy": (
                "closest timestamp alignment with one unique eye sample per hand pose"
            ),
        }


def read_ethz_robot_arm_hand_eye_motions(
    archive_path: str | Path,
    *,
    maximum_time_delta_sec: float = 0.011,
    motion_stride: int = 60,
) -> HandEyeMotionDataset:
    """Build disjoint ``A X = X B`` pairs from the public ETHZ robot-arm zip."""

    if motion_stride < 1:
        raise ValueError("motion_stride must be positive")
    path = Path(archive_path)
    with zipfile.ZipFile(path) as archive:
        hand = _read_pose_member(archive, ETHZ_HAND_MEMBER)
        eye = _read_pose_member(archive, ETHZ_EYE_MEMBER)
    aligned = _nearest_aligned_poses(hand, eye, maximum_time_delta_sec)
    absolute_pose_pairs = tuple(
        AlignedHandEyePosePair(
            pair_id=f"ethz-absolute-{hand_pose.source_index:05d}-{eye_pose.source_index:05d}",
            pose_a=hand_pose.transform,
            pose_b=eye_pose.transform,
            alignment_delta_sec=delta,
            hand_source_index=hand_pose.source_index,
            eye_source_index=eye_pose.source_index,
        )
        for hand_pose, eye_pose, delta in _unique_hand_alignments(aligned)
    )
    motions: list[RelativeMotionPair] = []
    used_absolute_ids: list[tuple[str, int]] = []
    for start in range(0, len(aligned) - motion_stride, 2 * motion_stride):
        hand_first, eye_first, _delta_first = aligned[start]
        hand_second, eye_second, _delta_second = aligned[start + motion_stride]
        motion_a = hand_first.transform.inverse().compose(hand_second.transform)
        motion_b = eye_first.transform.inverse().compose(eye_second.transform)
        motions.append(
            RelativeMotionPair(
                pair_id=(
                    f"ethz-{eye_first.source_index:05d}-{eye_second.source_index:05d}"
                ),
                motion_a=motion_a,
                motion_b=motion_b,
            )
        )
        used_absolute_ids.extend(
            (
                ("hand", hand_first.source_index),
                ("hand", hand_second.source_index),
                ("eye", eye_first.source_index),
                ("eye", eye_second.source_index),
            )
        )
    reuse_count = len(used_absolute_ids) - len(set(used_absolute_ids))
    return HandEyeMotionDataset(
        motions=tuple(motions),
        absolute_pose_pairs=absolute_pose_pairs,
        hand_pose_count=len(hand),
        eye_pose_count=len(eye),
        aligned_pose_count=len(aligned),
        maximum_alignment_delta_sec=(
            max(item[2] for item in aligned) if aligned else None
        ),
        motion_stride=motion_stride,
        absolute_pose_reuse_count=reuse_count,
    )


def _unique_hand_alignments(
    aligned: list[tuple[TimestampedPose, TimestampedPose, float]],
) -> list[tuple[TimestampedPose, TimestampedPose, float]]:
    """Retain the closest eye sample for each hand pose without pose leakage."""

    closest_by_hand: dict[int, tuple[TimestampedPose, TimestampedPose, float]] = {}
    for item in aligned:
        hand_pose, eye_pose, delta = item
        previous = closest_by_hand.get(hand_pose.source_index)
        if previous is None or (delta, eye_pose.source_index) < (
            previous[2],
            previous[1].source_index,
        ):
            closest_by_hand[hand_pose.source_index] = item
    return sorted(
        closest_by_hand.values(),
        key=lambda item: (item[1].timestamp_sec, item[0].source_index),
    )


def _read_pose_member(
    archive: zipfile.ZipFile, member: str
) -> list[TimestampedPose]:
    try:
        payload = archive.read(member).decode("utf-8")
    except KeyError as exc:
        raise ValueError(f"ETHZ archive is missing {member}") from exc
    poses: list[TimestampedPose] = []
    for index, row in enumerate(csv.reader(io.StringIO(payload))):
        if len(row) != 8:
            raise ValueError(f"{member}:{index + 1}: expected eight CSV values")
        try:
            values = tuple(float(value) for value in row)
        except ValueError as exc:
            raise ValueError(f"{member}:{index + 1}: non-numeric CSV value") from exc
        poses.append(
            TimestampedPose(
                timestamp_sec=values[0],
                transform=SE3(
                    (values[1], values[2], values[3]),
                    (values[4], values[5], values[6], values[7]),
                ),
                source_index=index,
            )
        )
    return poses


def _nearest_aligned_poses(
    hand: list[TimestampedPose],
    eye: list[TimestampedPose],
    maximum_delta_sec: float,
) -> list[tuple[TimestampedPose, TimestampedPose, float]]:
    hand_times = [pose.timestamp_sec for pose in hand]
    aligned: list[tuple[TimestampedPose, TimestampedPose, float]] = []
    for eye_pose in eye:
        insertion = bisect.bisect_left(hand_times, eye_pose.timestamp_sec)
        indices = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(hand)
        ]
        if not indices:
            continue
        closest = min(
            indices,
            key=lambda index: abs(hand[index].timestamp_sec - eye_pose.timestamp_sec),
        )
        delta = abs(hand[closest].timestamp_sec - eye_pose.timestamp_sec)
        if delta <= maximum_delta_sec:
            aligned.append((hand[closest], eye_pose, delta))
    return aligned
