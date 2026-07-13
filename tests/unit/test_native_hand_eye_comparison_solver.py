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
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration
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


def _config(
    dataset_path: Path,
    *,
    min_horaud_gap: float = 1.0e-3,
    min_andreff_width: float = 1.0e-4,
    min_shah_gap: float = 1.0e-3,
    max_li_projection: float = 0.05,
    min_dornaika_gap: float = 1.0e-3,
    min_dornaika_sign_fraction: float = 0.99,
) -> CalibrationConfig:
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
                            "min_horaud_quaternion_normalized_eigengap": (min_horaud_gap),
                            "min_andreff_rotation_width": min_andreff_width,
                            "min_shah_rotation_normalized_gap": min_shah_gap,
                            "max_li_so3_projection_correction_frobenius": (max_li_projection),
                            "min_dornaika_horaud_rotation_normalized_gap": (min_dornaika_gap),
                            "min_dornaika_horaud_sign_synchronization_fraction": (
                                min_dornaika_sign_fraction
                            ),
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
    assert len(dataset.absolute_pose_pairs) == 40
    assert len({pair.hand_source_index for pair in dataset.absolute_pose_pairs}) == 40
    assert len({pair.eye_source_index for pair in dataset.absolute_pose_pairs}) == 40
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
    assert (
        np.linalg.norm(
            np.asarray(result.transforms["T_hand_eye"].translation_m) - truth.translation_m
        )
        < 1.0e-9
    )
    assert result.metrics["hand_eye_absolute_pose_reuse_count"].value == 0.0
    assert result.metrics["hand_eye_horaud_dornaika_rotation_axis_rank"].grade == "pass"
    assert result.metrics["hand_eye_horaud_dornaika_quaternion_normalized_eigengap"].grade == "pass"
    assert result.metrics["hand_eye_common_split_consistent"].value == 1.0
    assert result.metrics["hand_eye_andreff_rotation_observable_rank"].grade == "pass"
    assert result.metrics["hand_eye_andreff_rotation_minimum_width"].grade == "pass"
    assert result.metrics["hand_eye_andreff_rotation_nullspace_ratio"].grade == "pass"
    assert result.metrics["robot_world_hand_eye_pose_pair_count"].value == 40.0
    assert (
        result.metrics["robot_world_hand_eye_shah_rotation_dominant_multiplicity"].grade == "pass"
    )
    assert result.metrics["robot_world_hand_eye_shah_rotation_normalized_gap"].grade == "pass"
    assert result.metrics["robot_world_hand_eye_shah_translation_rank"].value == 6.0
    assert result.metrics["robot_world_hand_eye_shah_translation_condition_number"].grade == "pass"
    assert result.metrics["robot_world_hand_eye_shah_holdout_rotation_rmse_deg"].grade == "pass"
    assert result.metrics["robot_world_hand_eye_shah_holdout_translation_rmse_m"].grade == "pass"
    assert result.metrics["robot_world_hand_eye_shah_known_bad_detectable_fraction"].value == 1.0
    assert result.metrics["robot_world_hand_eye_li_linear_rank"].value == 24.0
    assert result.metrics["robot_world_hand_eye_li_linear_condition_number"].grade == "pass"
    assert (
        result.metrics["robot_world_hand_eye_li_so3_projection_correction_frobenius_max"].grade
        == "pass"
    )
    assert result.metrics["robot_world_hand_eye_li_shah_common_split_consistent"].value == 1.0
    assert result.metrics["robot_world_hand_eye_li_holdout_rotation_rmse_deg"].grade == "pass"
    assert result.metrics["robot_world_hand_eye_li_holdout_translation_rmse_m"].grade == "pass"
    assert result.metrics["robot_world_hand_eye_li_known_bad_detectable_fraction"].value == 1.0
    assert result.metrics["robot_world_hand_eye_li_shah_x_rotation_delta_deg"].value is not None
    assert result.metrics["robot_world_hand_eye_li_shah_z_translation_delta_m"].value is not None
    assert result.metrics["robot_world_hand_eye_absolute_common_split_consistent"].value == 1.0
    assert (
        result.metrics["robot_world_hand_eye_dornaika_horaud_rotation_dominant_multiplicity"].value
        == 1.0
    )
    assert (
        result.metrics["robot_world_hand_eye_dornaika_horaud_rotation_normalized_gap"].grade
        == "pass"
    )
    assert (
        result.metrics["robot_world_hand_eye_dornaika_horaud_quaternion_unit_error_max"].grade
        == "pass"
    )
    assert (
        result.metrics["robot_world_hand_eye_dornaika_horaud_sign_synchronization_fraction"].value
        == 1.0
    )
    assert result.metrics["robot_world_hand_eye_dornaika_horaud_translation_rank"].value == 6.0
    assert (
        result.metrics["robot_world_hand_eye_dornaika_horaud_holdout_rotation_rmse_deg"].grade
        == "pass"
    )
    assert (
        result.metrics["robot_world_hand_eye_dornaika_horaud_known_bad_detectable_fraction"].value
        == 1.0
    )
    assert "T_robot_world" in result.transforms
    for method in (
        "park_martin",
        "tsai_lenz",
        "daniilidis",
        "horaud_dornaika",
        "andreff",
    ):
        assert result.metrics[f"hand_eye_{method}_holdout_rotation_rmse_deg"].grade == "pass"
        assert result.metrics[f"hand_eye_{method}_known_bad_detectable_fraction"].value == 1.0
    assert result.provenance["metrics_origin"] == "recomputed"
    assert result.provenance["data_verified"] is False
    comparison = result.provenance["native_hand_eye_comparison"]
    assert isinstance(comparison, dict)
    results = comparison["results"]
    assert isinstance(results, dict)
    assert results["horaud_dornaika"]["paper_doi"] == "10.1177/027836499501400301"
    assert results["andreff"]["paper_doi"] == "10.1109/IM.1999.805374"
    assert results["shah_robot_world_hand_eye"]["paper"]["doi"] == "10.1115/1.4024473"
    assert results["li_robot_world_hand_eye"]["paper"]["doi"] == "10.5897/IJPS.9000501"
    assert results["dornaika_horaud_robot_world_hand_eye"]["paper"]["doi"] == "10.1109/70.704233"


def test_pipeline_replaces_generic_slac_degeneracy_with_hand_eye_evidence(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    _write_synthetic_archive(archive)
    config_path = tmp_path / "config.json"
    config_path.write_text(_config(tmp_path).model_dump_json(), encoding="utf-8")

    result = run_calibration(config_path, CalibrationRunOptions(output_dir=tmp_path / "output"))

    assert result is not None
    assert result.degeneracy.grade == "pass"
    assert result.degeneracy.reason is None
    assert "hand_eye_horaud_dornaika_quaternion_normalized_eigengap" in result.metrics


def test_declared_minimum_width_gate_cannot_be_weakened_by_convergence(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    _write_synthetic_archive(archive)
    config = _config(tmp_path, min_horaud_gap=1.0)

    result = NativeHandEyeComparisonSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    metric = result.metrics["hand_eye_horaud_dornaika_quaternion_normalized_eigengap"]
    assert metric.grade == "fail"
    assert result.status == "inconclusive"
    assert result.observability is not None
    assert result.observability.grade == "fail"
    assert "horaud_quaternion_minimum_width" in result.observability.weak_directions


def test_andreff_width_gate_cannot_be_weakened_by_convergence(tmp_path: Path) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    _write_synthetic_archive(archive)
    config = _config(tmp_path, min_andreff_width=1.0)

    result = NativeHandEyeComparisonSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    assert result.metrics["hand_eye_andreff_rotation_minimum_width"].grade == "fail"
    assert result.status == "inconclusive"
    assert result.observability is not None
    assert result.observability.grade == "fail"
    assert "andreff_kronecker_observability" in result.observability.weak_directions


def test_shah_gap_gate_cannot_be_weakened_by_convergence(tmp_path: Path) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    _write_synthetic_archive(archive)
    config = _config(tmp_path, min_shah_gap=1.0)

    result = NativeHandEyeComparisonSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    assert result.metrics["robot_world_hand_eye_shah_rotation_normalized_gap"].grade == "fail"
    assert result.status == "inconclusive"
    assert result.observability is not None
    assert result.observability.grade == "fail"
    assert "shah_robot_world_hand_eye_observability" in result.observability.weak_directions


def test_li_projection_gate_cannot_be_weakened_by_convergence(tmp_path: Path) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    _write_synthetic_archive(archive)
    config = _config(tmp_path, max_li_projection=1.0e-16)

    result = NativeHandEyeComparisonSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    metric = result.metrics["robot_world_hand_eye_li_so3_projection_correction_frobenius_max"]
    assert metric.grade == "fail"
    assert result.status == "inconclusive"
    assert result.observability is not None
    assert result.observability.grade == "fail"
    assert "li_robot_world_hand_eye_observability" in result.observability.weak_directions


def test_dornaika_horaud_gap_gate_cannot_be_weakened_by_convergence(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    _write_synthetic_archive(archive)
    config = _config(tmp_path, min_dornaika_gap=1.0)

    result = NativeHandEyeComparisonSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    metric = result.metrics["robot_world_hand_eye_dornaika_horaud_rotation_normalized_gap"]
    assert metric.grade == "fail"
    assert result.status == "inconclusive"
    assert result.observability is not None
    assert result.observability.grade == "fail"
    assert (
        "dornaika_horaud_robot_world_hand_eye_observability" in result.observability.weak_directions
    )


def test_dornaika_horaud_sign_gate_cannot_be_weakened_by_convergence(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "robot_arm_w_color_camera_real.zip"
    _write_synthetic_archive(archive)
    config = _config(tmp_path, min_dornaika_sign_fraction=1.01)

    result = NativeHandEyeComparisonSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("filesystem", str(tmp_path), True),
    )

    metric = result.metrics["robot_world_hand_eye_dornaika_horaud_sign_synchronization_fraction"]
    assert metric.value == 1.0
    assert metric.grade == "fail"
    assert result.status == "inconclusive"
    assert result.observability is not None
    assert result.observability.grade == "fail"
