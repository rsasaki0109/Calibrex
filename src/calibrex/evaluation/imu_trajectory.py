"""IMU-to-sensor calibration from an IMU CSV and a sensor trajectory (no bags, no ROS).

This runs the same evidence as the Livox pipeline - rotation, clock offset,
and gyro bias from angular rates, then the lever arm from the accelerometer -
but takes the sensor motion from a trajectory the user already has, for
example a LiDAR-inertial odometry or SLAM output in TUM format.  It needs only
NumPy, SciPy, and Pydantic, so it also runs in the browser under Pyodide.

The artifacts call the trajectory's sensor frame "lidar": it is the frame
whose poses the trajectory lists, whatever the sensor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.imu_lidar_rotation import (
    ImuLidarRotationArtifact,
    ImuLidarRotationProvenance,
    ImuLidarWindowSummary,
)
from calibrex.core.imu_lidar_translation import (
    ImuLidarRotationInput,
    ImuLidarTranslationArtifact,
    ImuLidarTranslationProvenance,
)
from calibrex.data.imu_trajectory import (
    parse_imu_csv,
    parse_tum_trajectory,
    split_trajectory_windows,
)
from calibrex.data.livox_ros2 import MID360_T_LIDAR_IMU
from calibrex.evaluation.imu_lidar_rotation import (
    ImuLidarRunOptions,
    evaluate_imu_lidar_rotation,
    rotation_artifact_from_evaluation,
)
from calibrex.evaluation.imu_lidar_translation import (
    ImuLidarTranslationOptions,
    evaluate_imu_lidar_translation,
    translation_artifact_from_evaluation,
)
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries
from calibrex.solvers.imu_lidar_translation_solver import ImuPreintegrator

TRAJECTORY_LIMITATIONS: tuple[str, ...] = (
    "The sensor trajectory is supplied by the user; its accuracy, deskewing, and any "
    "IMU dependence (for example a LiDAR-inertial odometry) are not controlled here.",
    "The artifacts call the trajectory's sensor frame 'lidar'.",
    "Accelerometer scale and axis misalignment are not estimated.",
)


@dataclass(frozen=True)
class ImuTrajectoryCalibration:
    """Rotation artifact and, when the rotation was solved, the lever-arm artifact."""

    rotation: ImuLidarRotationArtifact
    translation: ImuLidarTranslationArtifact | None


def run_imu_trajectory_calibration(
    imu_text: str,
    trajectory_text: str,
    *,
    acceleration_unit: Literal["mps2", "g"] = "mps2",
    reference: Literal["none", "mid360"] = "none",
    estimate_translation: bool = True,
    sequence_id: str = "trajectory",
    dataset_family: str = "user",
    dataset_license: str = "user-provided",
    rotation_options: ImuLidarRunOptions | None = None,
    translation_options: ImuLidarTranslationOptions | None = None,
    command: list[str] | None = None,
) -> ImuTrajectoryCalibration:
    """Calibrate the IMU against a sensor trajectory and return schema-valid artifacts."""

    rotation_opts = rotation_options or ImuLidarRunOptions()
    translation_opts = translation_options or ImuLidarTranslationOptions()
    imu = parse_imu_csv(imu_text, acceleration_unit=acceleration_unit)
    times, poses = parse_tum_trajectory(trajectory_text)
    windows = split_trajectory_windows(
        times,
        poses,
        window_duration_s=rotation_opts.windowing.window_duration_s,
        min_window_poses=rotation_opts.windowing.min_window_scans,
    )
    if len(windows) < 3:
        raise ValueError(
            f"the trajectory yields {len(windows)} windows of "
            f"{rotation_opts.windowing.window_duration_s:g} s; at least 3 are needed"
        )
    gyro = GyroSeries(imu.times_s, imu.gyro_rps)
    use_mid360 = reference == "mid360"
    digest = hashlib.sha256()
    for part in (imu_text, "\0", trajectory_text):
        digest.update(part.encode("utf-8"))
    common: dict[str, Any] = {
        "generator": __name__,
        "generator_version": __version__,
        "command": command or [],
        "dataset_family": dataset_family,
        "sequence_ids": [sequence_id],
        "stream_profile": f"imu-csv-{acceleration_unit}+tum-trajectory",
        "input_sha256": digest.hexdigest(),
        "input_digest_scope": "IMU CSV and trajectory text in full",
        "dataset_license": dataset_license,
    }

    evaluation = evaluate_imu_lidar_rotation(
        gyro,
        windows,
        rotation_opts,
        reference_rotation=MID360_T_LIDAR_IMU[:3, :3] if use_mid360 else None,
    )
    rotation_artifact = rotation_artifact_from_evaluation(
        evaluation,
        rotation_opts,
        windows=ImuLidarWindowSummary(
            scans_read=len(times),
            odometry_segments=len({window.block for window in windows}),
            unreliable_registrations=0,
            windows=len(windows),
            rate_intervals=evaluation.interval_count,
            imu_samples=len(imu.times_s),
        ),
        provenance=ImuLidarRotationProvenance(**common),
        reference="Livox MID360 manual: IMU axes aligned with the LiDAR frame"
        if use_mid360
        else None,
        limitations=list(TRAJECTORY_LIMITATIONS),
    )
    result = evaluation.result
    if not estimate_translation or result.rotation is None:
        return ImuTrajectoryCalibration(rotation_artifact, None)

    preintegrator = ImuPreintegrator(
        imu.times_s, imu.gyro_rps, imu.accel_mps2, result.gyro_bias_rps
    )
    translation_evaluation = evaluate_imu_lidar_translation(
        preintegrator,
        windows,
        result.rotation,
        result.time_offset_s,
        translation_opts,
        reference_translation=MID360_T_LIDAR_IMU[:3, 3] if use_mid360 else None,
    )
    rotation_digest = hashlib.sha256(
        rotation_artifact.model_dump_json(exclude_none=True).encode("utf-8")
    ).hexdigest()
    translation_artifact = translation_artifact_from_evaluation(
        translation_evaluation,
        translation_opts,
        rotation_input=ImuLidarRotationInput(
            artifact_sha256=rotation_digest,
            policy_status=rotation_artifact.policy_status,
            rotation_quat_xyzw=[
                float(value) for value in Rotation.from_matrix(result.rotation).as_quat()
            ],
            time_offset_s=result.time_offset_s,
            gyro_bias_rps=[float(value) for value in result.gyro_bias_rps],
        ),
        windows=len(windows),
        imu_samples=len(imu.times_s),
        reference="Livox MID360 manual: IMU at (11.0, 23.29, -44.12) mm in the LiDAR frame"
        if use_mid360
        else None,
        limitations=list(TRAJECTORY_LIMITATIONS),
        provenance=ImuLidarTranslationProvenance(**common),
    )
    return ImuTrajectoryCalibration(rotation_artifact, translation_artifact)


def calibration_summary(calibration: ImuTrajectoryCalibration) -> dict[str, Any]:
    """A compact, JSON-ready summary for a report or a web page."""

    rotation = calibration.rotation
    summary: dict[str, Any] = {
        "rotation": {
            "policy_status": rotation.policy_status,
            "policy_reasons": rotation.policy_reasons,
            "dofs": [
                {
                    "name": dof.name,
                    "unit": dof.unit,
                    "value": dof.value,
                    "std": dof.std_reported,
                    "status": dof.status,
                    "control_detected": None
                    if dof.known_bad_control is None
                    else dof.known_bad_control.detected,
                }
                for dof in rotation.dofs
            ],
            "rotation_quat_xyzw": rotation.rotation_quat_xyzw,
            "holdout_median_rate_residual_rps": rotation.holdout_median_rate_residual_rps,
        }
    }
    translation = calibration.translation
    if translation is not None:
        summary["translation"] = {
            "policy_status": translation.policy_status,
            "policy_reasons": translation.policy_reasons,
            "translation_m": translation.translation_m,
            "axes": [
                {
                    "name": axis.name,
                    "value_m": axis.value,
                    "std_m": axis.std_reported,
                    "status": axis.status,
                    "control_detected": None
                    if axis.known_bad_control is None
                    else axis.known_bad_control.detected,
                }
                for axis in translation.axes
            ],
            "gravity_norm_median_mps2": translation.gravity_norm_median_mps2,
        }
    if calibration.rotation.rotation_quat_xyzw is not None:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(calibration.rotation.rotation_quat_xyzw).as_matrix()
        if translation is not None and translation.translation_m is not None:
            matrix[:3, 3] = translation.translation_m
        summary["T_lidar_imu"] = matrix.tolist()
        summary["time_offset_s"] = calibration.rotation.time_offset_s
    return summary
