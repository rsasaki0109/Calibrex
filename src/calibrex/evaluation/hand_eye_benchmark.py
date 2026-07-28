"""Leakage-safe shared protocols for absolute-pose hand-eye benchmarks."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass

from calibrex.solvers.park_martin_hand_eye_solver import HandEyeMotionPair
from calibrex.solvers.shah_robot_world_hand_eye_solver import (
    RobotWorldHandEyePosePair,
)


@dataclass(frozen=True)
class HandEyeAbsolutePoseSplit:
    """Bounded absolute-pose split shared by native and adapter methods."""

    split_id: str
    seed: int
    fit_poses: tuple[RobotWorldHandEyePosePair, ...]
    holdout_poses: tuple[RobotWorldHandEyePosePair, ...]

    @property
    def fit_ids_sha256(self) -> str:
        """Return a stable digest of the ordered fit pose IDs."""

        return _ids_sha256(self.fit_poses)

    @property
    def holdout_ids_sha256(self) -> str:
        """Return a stable digest of the ordered holdout pose IDs."""

        return _ids_sha256(self.holdout_poses)


def split_absolute_hand_eye_poses(
    poses: Sequence[RobotWorldHandEyePosePair],
    *,
    seed: int,
    fit_count: int,
    holdout_count: int,
    split_id: str | None = None,
) -> HandEyeAbsolutePoseSplit:
    """Select disjoint bounded fit and holdout absolute poses.

    Sampling happens at the absolute-pose level before relative motions are
    constructed. This prevents the same source pose from leaking into both
    populations and keeps OpenCV's quadratic pair construction bounded.
    """

    if fit_count < 3 or holdout_count < 2:
        raise ValueError("hand-eye splits require at least 3 fit and 2 holdout poses")
    ordered = tuple(sorted(poses, key=lambda item: item.pair_id))
    if len({pose.pair_id for pose in ordered}) != len(ordered):
        raise ValueError("absolute hand-eye pose IDs must be unique")
    required = fit_count + holdout_count
    if required > len(ordered):
        raise ValueError(f"requested {required} absolute poses from a population of {len(ordered)}")
    indices = list(range(len(ordered)))
    random.Random(seed).shuffle(indices)
    holdout_indices = sorted(indices[:holdout_count])
    fit_indices = sorted(indices[holdout_count:required])
    fit = tuple(ordered[index] for index in fit_indices)
    holdout = tuple(ordered[index] for index in holdout_indices)
    return HandEyeAbsolutePoseSplit(
        split_id=split_id or f"seed-{seed:03d}",
        seed=seed,
        fit_poses=fit,
        holdout_poses=holdout,
    )


def opencv_compatible_hand_eye_motions(
    poses: Sequence[RobotWorldHandEyePosePair],
) -> tuple[HandEyeMotionPair, ...]:
    """Build the exact pairwise ``AX=XB`` motions used by OpenCV 4.

    For absolute pairs satisfying ``A_i X = Y B_i``, OpenCV constructs
    ``inv(A_j) A_i`` and, because the adapter supplies ``inv(B_i)`` as
    ``target2cam``, ``inv(B_j) B_i``. These motions satisfy the native
    Calibrex convention ``motion_a X = X motion_b``.
    """

    ordered = tuple(sorted(poses, key=lambda item: item.pair_id))
    if len({pose.pair_id for pose in ordered}) != len(ordered):
        raise ValueError("absolute hand-eye pose IDs must be unique")
    motions: list[HandEyeMotionPair] = []
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            motions.append(
                HandEyeMotionPair(
                    pair_id=f"{second.pair_id}__to__{first.pair_id}",
                    motion_a=second.pose_a.inverse().compose(first.pose_a),
                    motion_b=second.pose_b.inverse().compose(first.pose_b),
                    weight=first.weight * second.weight,
                )
            )
    return tuple(motions)


def _ids_sha256(poses: Sequence[RobotWorldHandEyePosePair]) -> str:
    payload = "\n".join(pose.pair_id for pose in poses).encode()
    return hashlib.sha256(payload).hexdigest()
