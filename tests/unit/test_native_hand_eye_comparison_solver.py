import csv
import io
import math
import zipfile
from pathlib import Path

import numpy as np

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.data.ethz_hand_eye import (
    ETHZ_EYE_MEMBER,
    ETHZ_HAND_MEMBER,
    read_ethz_robot_arm_hand_eye_motions,
)
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.native_hand_eye_comparison_solver import (
    NativeHandEyeComparisonSolver,
)


def _write_pose_csv(poses: list[tuple[float, SE3]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    for timestamp, transform in poses:
        writer.writerow(
            (
                timestamp,
                *transform.translation_m,
                *transform.rotation_quat_xyzw,
            )
        )
    return output.getvalue()


def _write_synthetic_archive(path: Path, count: int = 40) -> SE3:
    rng = np.random.default_rng(222)
    truth = SE3((0.24, -0.16, 0.11), (0.08, -0.12, 0.15, 0.978))
    world = SE3((-0.3, 0.5, 0.2), (-0.03, 0.07, 0.09, 0.993))
    hand_poses: list[tuple[float, SE3]] = []
    eye_poses: list[tuple[float, SE3]] = []
    for index in range(count):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        half = math.radians(float(rng.uniform(-80.0, 80.0))) / 2.0
        hand = SE3(
            tuple(rng.uniform(-0.5, 0.5, size=3)),
            (
                axis[0] * math.sin(half),
                axis[1] * math.sin(half),
                axis[2] * math.sin(half),
                math.cos(half),
            ),
        )
        eye = world.inverse().compose(hand).compose(truth)
        timestamp = 1000.0 + 0.02 * index
        hand_poses.append((timestamp, hand))
        eye_poses.append((timestamp + 0.001, eye))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(ETHZ_HAND_MEMBER, _write_pose_csv(hand_poses))
        archive.writestr(ETHZ_EYE_MEMBER, _write_pose_csv(eye_poses))
    return truth


def _config(dataset_path: Path) -> CalibrationConfig:
    return CalibrationConfig.model_validate(
        {
            "dataset": {"type": "filesystem", "path": str(dataset_path)},
            "sensors": {"eye": {"type": "camera"}},
            "frames": {
                "hand": {"root": True},
                "eye": {"parent": "hand", "transform": {"estimate": True}},
            },
            "pipeline": {
                "factors": {
                    "hand_eye_motion_closure": {
                        "enabled": True,
                        "options": {
                            "archive_file": "robot_arm_w_color_camera_real.zip",
                            "motion_stride": 1,
                            "maximum_time_delta_sec": 0.005,
                        },
                    }
                }
            },
            "solver": {"backend": "native_hand_eye_comparison", "seed": 3},
        }
    )


def test_ethz_reader_builds_disjoint_absolute_pose_pairs(tmp_path: Path) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    _write_synthetic_archive(archive)

    dataset = read_ethz_robot_arm_hand_eye_motions(
        archive, maximum_time_delta_sec=0.005, motion_stride=1
    )

    assert len(dataset.motions) == 20
    assert dataset.aligned_pose_count == 40
    assert dataset.absolute_pose_reuse_count == 0
    assert dataset.maximum_alignment_delta_sec is not None
    assert dataset.maximum_alignment_delta_sec < 0.002


def test_native_comparison_recovers_truth_with_common_metrics(tmp_path: Path) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    truth = _write_synthetic_archive(archive)
    config = _config(tmp_path)

    result = NativeHandEyeComparisonSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    assert result.status == "pass"
    assert np.linalg.norm(
        np.asarray(result.transforms["T_hand_eye"].translation_m) - truth.translation_m
    ) < 1.0e-9
    assert result.metrics["hand_eye_absolute_pose_reuse_count"].value == 0.0
    for method in ("park_martin", "tsai_lenz", "daniilidis"):
        assert result.metrics[f"hand_eye_{method}_holdout_rotation_rmse_deg"].grade == "pass"
        assert (
            result.metrics[f"hand_eye_{method}_known_bad_detectable_fraction"].value
            == 1.0
        )
    assert result.provenance["metrics_origin"] == "recomputed"
    assert result.provenance["data_verified"] is False
