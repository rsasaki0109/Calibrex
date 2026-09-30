"""Compose a GNSS-IMU lever arm from GNSS-LiDAR and IMU-LiDAR artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.gnss_imu_lever_arm import (
    GnssImuAxisRecord,
    GnssImuInput,
    GnssImuLeverArmArtifact,
)
from calibrex.core.gnss_lidar_lever_arm import GnssLidarLeverArmArtifact
from calibrex.core.imu_lidar_rotation import ImuLidarRotationArtifact
from calibrex.core.imu_lidar_translation import ImuLidarTranslationArtifact
from calibrex.core.io import read_mapping
from calibrex.core.provenance import sha256_path

LIMITATIONS: tuple[str, ...] = (
    "A composition, not a direct GNSS-IMU fit: its accuracy is that of the two inputs, "
    "and the std assumes their errors are independent.",
    "The rotation's uncertainty is neglected; at 0.1 deg and a 0.1 m lever it adds about 0.2 mm.",
    "The inputs should come from the same rig; this is not checked.",
)


def compose_gnss_imu_lever_arm(
    gnss_lidar_path: str | Path,
    rotation_path: str | Path,
    translation_path: str | Path,
    *,
    reference_m: Sequence[float] | None = None,
    reference: str | None = None,
    observable_std_m: float = 0.02,
    command: list[str] | None = None,
) -> GnssImuLeverArmArtifact:
    """``p_imu = R_lidar_imu^T (p_lidar - t_lidar_imu)`` with propagated std."""

    gnss = GnssLidarLeverArmArtifact.model_validate(read_mapping(Path(gnss_lidar_path)))
    rotation_artifact = ImuLidarRotationArtifact.model_validate(read_mapping(Path(rotation_path)))
    translation = ImuLidarTranslationArtifact.model_validate(read_mapping(Path(translation_path)))
    if gnss.lever_arm_m is None or rotation_artifact.rotation_quat_xyzw is None:
        raise ValueError("the GNSS-LiDAR or IMU-LiDAR rotation artifact has no estimate")
    if translation.translation_m is None:
        raise ValueError("the IMU-LiDAR translation artifact has no estimate")
    rotation = Rotation.from_quat(rotation_artifact.rotation_quat_xyzw).as_matrix()
    p_lidar = np.asarray(gnss.lever_arm_m, dtype=np.float64)
    t_lidar_imu = np.asarray(translation.translation_m, dtype=np.float64)
    lever = rotation.T @ (p_lidar - t_lidar_imu)
    gnss_std = {item.name: item.std_reported for item in gnss.dofs}
    translation_std = {item.name: item.std_reported for item in translation.axes}
    variance_lidar = np.diag(
        [gnss_std[name] ** 2 + translation_std[name] ** 2 for name in ("x", "y", "z")]
    )
    std = np.sqrt(np.diag(rotation.T @ variance_lidar @ rotation))
    # An output axis inherits "unobservable" from any input axis it depends on
    # (rotation weight above 0.1), even when the propagated std looks small.
    unobservable_inputs = {item.name for item in gnss.dofs if item.status != "estimated"} | {
        item.name for item in translation.axes if item.status != "estimated"
    }
    axes = []
    for index, name in enumerate(("x", "y", "z")):
        ref = None if reference_m is None else float(reference_m[index])
        depends_on = {
            source
            for position, source in enumerate(("x", "y", "z"))
            if abs(rotation[position, index]) > 0.1
        }
        observable = std[index] <= observable_std_m and not (depends_on & unobservable_inputs)
        axes.append(
            GnssImuAxisRecord(
                name=name,  # type: ignore[arg-type]
                value_m=float(lever[index]),
                std_m=float(std[index]),
                status="estimated" if observable else "unobservable",
                reference_m=ref,
                error_to_reference_m=None if ref is None else float(lever[index] - ref),
            )
        )
    estimated = [item.name for item in axes if item.status == "estimated"]
    unobservable = [item.name for item in axes if item.status == "unobservable"]
    reasons = (
        ["every axis's propagated std is within the bound"]
        if not unobservable
        else [
            "not calibrated (propagated std over the bound, or an input axis it depends on "
            "is unobservable): " + ", ".join(unobservable)
        ]
    )
    inputs = []
    for role, path, status in (
        ("gnss_lidar_lever_arm", gnss_lidar_path, gnss.policy_status),
        ("imu_lidar_rotation", rotation_path, rotation_artifact.policy_status),
        ("imu_lidar_translation", translation_path, translation.policy_status),
    ):
        digest = sha256_path(Path(path))
        assert digest is not None
        inputs.append(
            GnssImuInput(role=role, path=Path(path).name, sha256=digest, policy_status=status)  # type: ignore[arg-type]
        )
    return GnssImuLeverArmArtifact(
        policy_status="pass" if not unobservable else "inconclusive",
        policy_reasons=reasons,
        calibrated_dofs=estimated,
        lever_arm_m=[float(value) for value in lever],
        axes=axes,
        inputs=inputs,
        reference=reference,
        observable_std_m=observable_std_m,
        limitations=list(LIMITATIONS),
        generator=__name__,
        generator_version=__version__,
        command=command or [],
    )
