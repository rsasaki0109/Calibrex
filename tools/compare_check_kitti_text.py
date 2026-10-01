"""Compare ``calibrex check`` vehicle-pair evidence on a converted KITTI bag with the text CLIs.

The bag was made by ``calibrex convert kitti-raw``; the text artifacts by
``calibrex lidar-vehicle kitti``, ``imu-vehicle kitti``, ``ins-lidar kitti`` and
``lidar-wheel kitti`` on the same drive. Prints, per pair, the rotation
difference of the two estimates and the largest difference of any DoF value,
reported std and held-out control statistic, plus the policy statuses::

    python tools/compare_check_kitti_text.py EVIDENCE_DIR TEXT_PREFIX

``TEXT_PREFIX`` is the path prefix of ``<prefix>_lidar_vehicle.yaml`` and so on.
With ``--body-frame-wheel DRIVE`` the KITTI-text wheel proxy is re-run with the
body-frame forward speed and yaw rate the bag carries (the text path uses the
level-frame ``vf`` and ``wu``), which isolates that declared frame change.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.core.ins_lidar_hand_eye import load_ins_lidar_hand_eye
from calibrex.core.lidar_wheel_odometry import load_lidar_wheel_odometry
from calibrex.core.vehicle_frame_rotation import load_vehicle_frame_rotation

PAIRS = (
    ("lidar-vehicle", "lidar_vehicle", "lidar-vehicle_*.yaml", load_vehicle_frame_rotation),
    ("imu-vehicle", "imu_vehicle", "imu-vehicle_*.yaml", load_vehicle_frame_rotation),
    ("ins-lidar", "ins_lidar", "ins-lidar_*.yaml", load_ins_lidar_hand_eye),
    ("lidar-wheel", "lidar_wheel", "lidar-wheel_odometry_*.yaml", load_lidar_wheel_odometry),
)


def _records(artifact: Any) -> list[Any]:
    return list(getattr(artifact, "dofs", None) or getattr(artifact, "parameters", []))


def _rotation(artifact: Any) -> Rotation | None:
    quat = getattr(artifact, "rotation_quat_xyzw", None)
    if quat is None and getattr(artifact, "transform", None) is not None:
        quat = artifact.transform.rotation_quat_xyzw
    return None if quat is None else Rotation.from_quat(quat)


def compare(bag_artifact: Any, text_artifact: Any) -> dict[str, float | str]:
    """Rotation difference (deg), largest DoF/std/control differences, policy statuses."""

    a, b = _rotation(bag_artifact), _rotation(text_artifact)
    rotation_deg = math.nan if a is None or b is None else math.degrees((a * b.inv()).magnitude())
    value = std = control = 0.0
    text_by_name = {record.name: record for record in _records(text_artifact)}
    for record in _records(bag_artifact):
        other = text_by_name[record.name]
        value = max(value, abs(record.value - other.value))
        std = max(std, abs(record.std_reported - other.std_reported))
        if record.known_bad_control and other.known_bad_control:
            control = max(
                control,
                abs(
                    record.known_bad_control.holdout_delta_chi2
                    - other.known_bad_control.holdout_delta_chi2
                ),
            )
    return {
        "rotation_diff_deg": rotation_deg,
        "max_dof_value_diff": value,
        "max_std_diff": std,
        "max_control_dchi2_diff": control,
        "policy_bag": str(bag_artifact.policy_status),
        "policy_text": str(text_artifact.policy_status),
    }


def _body_frame_wheel_run(drive: Path, output: Path) -> Any:
    from calibrex.evaluation import lidar_wheel
    from calibrex.evaluation.vehicle_frame import oxts_body_frame_motion
    from calibrex.solvers.lidar_wheel_solver import WheelOdometry

    def proxy(drive_dir: str | Path) -> WheelOdometry:
        from calibrex.data.kitti import read_timestamps

        root = Path(drive_dir)
        times = np.array(
            [
                stamp.timestamp_ns * 1.0e-9
                for stamp in read_timestamps(root / "oxts" / "timestamps.txt")
            ]
        )
        rows = [
            oxts_body_frame_motion(np.loadtxt(p))
            for p in sorted((root / "oxts" / "data").glob("*.txt"))
        ]
        order = np.argsort(times, kind="stable")
        speed = np.array([row[0][0] for row in rows])[order]
        yaw = np.array([row[1][2] for row in rows])[order]
        return WheelOdometry(times[order], speed, yaw)

    lidar_wheel.kitti_oxts_wheel_proxy = proxy  # type: ignore[assignment]
    artifact = lidar_wheel.run_kitti_lidar_wheel([drive])
    artifact.save(output)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("text_prefix", type=str)
    parser.add_argument("--body-frame-wheel", type=Path, metavar="DRIVE")
    args = parser.parse_args(argv)
    print(
        f"{'pair':<14}{'rot diff deg':>14}{'max dof':>12}{'max std':>12}{'max dchi2':>12}"
        "  policy bag/text"
    )
    for name, text_suffix, glob, loader in PAIRS:
        bag_files = sorted(args.evidence_dir.glob(glob))
        text_path = Path(f"{args.text_prefix}_{text_suffix}.yaml")
        if not bag_files or not text_path.is_file():
            print(f"{name:<14}missing ({'bag' if not bag_files else 'text'})")
            continue
        bag = loader(bag_files[0])
        text = loader(text_path)
        row = compare(bag, text)
        print(
            f"{name:<14}{row['rotation_diff_deg']:>14.2e}{row['max_dof_value_diff']:>12.2e}"
            f"{row['max_std_diff']:>12.2e}{row['max_control_dchi2_diff']:>12.2e}  "
            f"{row['policy_bag']}/{row['policy_text']}"
        )
        if name == "lidar-wheel" and args.body_frame_wheel is not None:
            body_path = Path(f"{args.text_prefix}_lidar_wheel_bodyframe.yaml")
            body = _body_frame_wheel_run(args.body_frame_wheel, body_path)
            row = compare(bag, body)
            print(
                f"{'  vs body-frame':<14}{row['rotation_diff_deg']:>14.2e}"
                f"{row['max_dof_value_diff']:>12.2e}{row['max_std_diff']:>12.2e}"
                f"{row['max_control_dchi2_diff']:>12.2e}  {row['policy_bag']}/{row['policy_text']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
