from __future__ import annotations

import math
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.camera_lidar_artifacts import (
    CameraLidarArtifactProvenance,
    CameraLidarCalibrationProblem,
    CameraLidarObservationBinding,
)
from calibrex.core.camera_lidar_initializer_calibration import (
    CameraLidarInitializerRecoveryGate,
    load_camera_lidar_initializer_calibration,
)
from calibrex.core.geometry import SE3
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    ProbabilisticCorrespondenceArtifact,
    ProbabilisticCorrespondenceFrame,
    ProbabilisticCorrespondenceProvenance,
    ProbabilisticImageCorrespondence,
    load_probabilistic_pnp_result,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import TransformResult
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics, DepthFileReference
from calibrex.evaluation.camera_lidar_initializer_calibration import (
    calibrate_camera_lidar_initializer,
    initializer_options_from_camera_lidar_calibration,
)

pytest.importorskip("cv2")


def test_initializer_calibration_locks_deterministic_highest_confidence(
    tmp_path: Path,
) -> None:
    correspondence_path, problem_path = _write_development_inputs(tmp_path)
    output = tmp_path / "initializer-calibration.yaml"

    calibration = calibrate_camera_lidar_initializer(
        correspondence_path,
        problem_path,
        confidence_thresholds=(0.2, 0.9),
        ransac_reprojection_thresholds_px=(2.0, 8.0),
        random_seeds=(0, 1),
        evaluation_dataset_ids_excluded=("evaluation-a", "evaluation-b"),
    )
    calibration.save(output)

    assert calibration.status == "locked"
    assert len(calibration.candidates) == 4
    assert calibration.selected_candidate_id == "confidence=0.9;ransac_px=2"
    locked = calibration.locked_initializer_options
    assert locked is not None
    assert locked.minimum_confidence == pytest.approx(0.9)
    assert locked.ransac_reprojection_threshold_px == pytest.approx(2.0)
    selected = next(
        item
        for item in calibration.candidates
        if item.candidate_id == calibration.selected_candidate_id
    )
    assert selected.gate_pass
    assert selected.maximum_rotation_error_deg is not None
    assert selected.maximum_rotation_error_deg < 1.0e-4
    assert selected.maximum_translation_error_m is not None
    assert selected.maximum_translation_error_m < 1.0e-4
    assert selected.maximum_pairwise_seed_rotation_delta_deg == pytest.approx(0.0)
    assert selected.maximum_pairwise_seed_translation_delta_m == pytest.approx(0.0)
    assert validate_file(output, "camera-lidar-initializer-calibration").valid
    assert (
        load_camera_lidar_initializer_calibration(output).calibration_id
        == calibration.calibration_id
    )
    options = initializer_options_from_camera_lidar_calibration(
        calibration,
        random_seed=1,
    )
    assert options.minimum_confidence == pytest.approx(0.9)
    pnp_output = tmp_path / "locked-aggregate-pnp.yaml"
    assert (
        main(
            [
                "camera-lidar",
                "refine-probabilistic-pnp-aggregate",
                str(correspondence_path),
                "--output",
                str(pnp_output),
                "--initializer-calibration",
                str(output),
                "--random-seed",
                "1",
                "--json",
            ]
        )
        == 0
    )
    pnp = load_probabilistic_pnp_result(pnp_output)
    assert pnp.options["minimum_confidence"] == pytest.approx(0.9)
    assert pnp.provenance.initializer_calibration_id == calibration.calibration_id
    assert pnp.provenance.initializer_calibration_sha256 == sha256_path(output)
    with pytest.raises(ValueError, match="absent from the initializer"):
        initializer_options_from_camera_lidar_calibration(
            calibration,
            random_seed=7,
        )


def test_initializer_calibration_rejects_grid_without_recovery(
    tmp_path: Path,
) -> None:
    correspondence_path, problem_path = _write_development_inputs(tmp_path)
    calibration = calibrate_camera_lidar_initializer(
        correspondence_path,
        problem_path,
        confidence_thresholds=(0.2, 0.9),
        ransac_reprojection_thresholds_px=(2.0, 8.0),
        random_seeds=(0,),
        recovery_gate=CameraLidarInitializerRecoveryGate(
            rotation_error_max_deg=0.5,
            translation_error_max_m=0.2,
            minimum_selected_frame_count=4,
            minimum_selected_correspondence_count=16,
            minimum_ransac_inlier_count=1000,
            maximum_ransac_inlier_reprojection_rmse_px=8.0,
            maximum_seed_rotation_delta_deg=0.01,
            maximum_seed_translation_delta_m=0.005,
        ),
        evaluation_dataset_ids_excluded=("evaluation-a",),
    )

    assert calibration.status == "rejected"
    assert calibration.selected_candidate_id is None
    assert calibration.locked_initializer_options is None
    assert all(not item.gate_pass for item in calibration.candidates)
    assert all(
        "minimum_ransac_inlier_count_failed" in item.gate_reasons
        for item in calibration.candidates
    )
    with pytest.raises(ValueError, match="rejected"):
        initializer_options_from_camera_lidar_calibration(
            calibration,
            random_seed=0,
        )


def test_initializer_calibration_cli_writes_schema_valid_lock(
    tmp_path: Path,
) -> None:
    correspondence_path, problem_path = _write_development_inputs(tmp_path)
    output = tmp_path / "initializer-calibration-cli.yaml"

    exit_code = main(
        [
            "camera-lidar",
            "calibrate-probabilistic-pnp-initializer",
            str(correspondence_path),
            str(problem_path),
            "--output",
            str(output),
            "--confidence-thresholds",
            "0.2,0.9",
            "--ransac-reprojection-thresholds-px",
            "2,8",
            "--random-seeds",
            "0,1",
            "--excluded-evaluation-dataset-ids",
            "evaluation-a,evaluation-b",
            "--json",
        ]
    )

    assert exit_code == 0
    assert validate_file(output, "auto").valid


def _write_development_inputs(tmp_path: Path) -> tuple[Path, Path]:
    truth = SE3(
        translation_m=(0.2, -0.1, 0.3),
        rotation_quat_xyzw=(
            0.0,
            math.sin(math.radians(4.0) / 2.0),
            0.0,
            math.cos(math.radians(4.0) / 2.0),
        ),
    )
    frames = _frames(truth)
    correspondence = ProbabilisticCorrespondenceArtifact(
        artifact_id="synthetic-development-correspondence",
        dataset_id="synthetic-development",
        split_id="development",
        sequence_id="development-sequence",
        dataset_license_spdx="CC0-1.0",
        provider=CorrespondenceProviderIdentity(
            provider="fixture",
            model="exact",
            version="1",
            source_repository="https://example.test/provider",
            source_commit="a" * 40,
            license_spdx="Apache-2.0",
            checkpoint_redistribution="not applicable",
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="pytest",
            generator_version="1",
            input_sha256={"fixture": "b" * 64},
        ),
    )
    reference = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=list(truth.translation_m),
        rotation_quat_xyzw=list(truth.rotation_quat_xyzw),
    )
    initial = TransformResult(
        parent="camera",
        child="lidar",
        translation_m=[0.0, 0.0, 0.0],
        rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    problem = CameraLidarCalibrationProblem(
        problem_id="synthetic-development-problem",
        dataset_id="synthetic-development",
        dataset_family="synthetic",
        sequence_id="development-sequence",
        depth_provider_path=str(tmp_path / "depth-provider.yaml"),
        depth_provider_sha256="c" * 64,
        observations=[
            CameraLidarObservationBinding(
                frame_id=frame.frame_id,
                split_id="development",
                depth_observation_frame_id=frame.frame_id,
                lidar=DepthFileReference(
                    path=str(tmp_path / f"{frame.frame_id}.npy"),
                    sha256="d" * 64,
                    size_bytes=1,
                    encoding="npy_float64_xyz",
                ),
                lidar_capture_time_ns=frame.capture_time_ns,
            )
            for frame in frames
        ],
        reference_transform_camera_lidar=reference,
        initial_transform_camera_lidar=initial,
        time_convention="synchronized synthetic observations",
        rotation_bound_deg=2.0,
        translation_bound_m=0.25,
        provenance=CameraLidarArtifactProvenance(
            generator="pytest",
            generator_version="1",
            source_sha256="e" * 64,
        ),
    )
    correspondence_path = tmp_path / "development-correspondence.yaml"
    problem_path = tmp_path / "development-problem.yaml"
    correspondence.save(correspondence_path)
    problem.save(problem_path)
    return correspondence_path, problem_path


def _frames(truth: SE3) -> list[ProbabilisticCorrespondenceFrame]:
    intrinsics = DepthCameraIntrinsics(
        width=1280,
        height=720,
        fx=600.0,
        fy=600.0,
        cx=640.0,
        cy=360.0,
    )
    points = (
        (-2.0, -1.0, 8.0),
        (-1.0, 1.0, 7.0),
        (0.5, -1.5, 9.0),
        (2.0, 1.0, 10.0),
        (-2.5, 0.5, 12.0),
        (1.5, -0.5, 6.0),
        (0.2, 1.8, 11.0),
        (2.8, -1.2, 13.0),
    )
    frames: list[ProbabilisticCorrespondenceFrame] = []
    for frame_index in range(4):
        correspondences = []
        for point_index, point in enumerate(points):
            shifted = (
                point[0] + 0.03 * frame_index,
                point[1] - 0.02 * frame_index,
                point[2] + 0.01 * frame_index,
            )
            camera_point = truth.transform_point(shifted)
            correspondences.append(
                ProbabilisticImageCorrespondence(
                    correspondence_id=str(point_index),
                    point_lidar_m=list(shifted),
                    image_mean_px=[
                        intrinsics.fx * camera_point[0] / camera_point[2]
                        + intrinsics.cx,
                        intrinsics.fy * camera_point[1] / camera_point[2]
                        + intrinsics.cy,
                    ],
                    image_covariance_px2=[0.25, 0.0, 0.0, 0.25],
                    outlier_probability=0.01,
                    reliability=0.99,
                )
            )
        frames.append(
            ProbabilisticCorrespondenceFrame(
                frame_id=f"frame-{frame_index:03d}",
                capture_time_ns=frame_index,
                camera_frame="camera",
                lidar_frame="lidar",
                intrinsics=intrinsics,
                correspondences=correspondences,
            )
        )
    return frames
