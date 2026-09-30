"""Separate a rig-level IMU-frame offset from per-camera errors in camera-IMU results.

Given several cameras' rotation artifacts (``calibrex camera-imu rotation``)
and a reference ``T_cam_imu`` for each (Kalibr camchains), this prints:

* the difference of every estimate from its reference, expressed about the IMU
  axes;
* the camera-to-camera rotation error, in which any common IMU-frame offset
  cancels; and
* the mean IMU-frame offset and each camera's residual after removing it.

A large common offset with small camera-to-camera errors points at the IMU
frame (for example the gyro frame versus the reference's IMU frame) rather
than at the camera pipeline.

Usage::

    python tools/analyze_camera_imu_consistency.py \\
        cam0=cam0_exp21.yaml:calib_3_cam0-1-camchain-imucam.yaml:cam0 \\
        cam3=cam3_exp21.yaml:calib_3_cam3-camchain-imucam.yaml:cam0 ...
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
from calibrex.core.io import read_mapping


def main() -> None:
    estimates: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for argument in sys.argv[1:]:
        name, spec = argument.split("=", 1)
        artifact_path, camchain_path, key = spec.split(":")
        artifact = load_imu_lidar_rotation(artifact_path)
        if artifact.rotation_quat_xyzw is None:
            raise SystemExit(f"{artifact_path} has no rotation")
        estimates[name] = Rotation.from_quat(artifact.rotation_quat_xyzw).as_matrix()
        entry = read_mapping(Path(camchain_path))[key]
        references[name] = np.asarray(entry["T_cam_imu"], dtype=np.float64)[:3, :3]
    errors = {
        name: references[name].T
        @ Rotation.from_matrix(estimates[name] @ references[name].T).as_rotvec()
        for name in estimates
    }
    print("difference to reference about IMU x, y, z (deg)")
    for name, error in errors.items():
        print(f"  {name}: {np.round(np.degrees(error), 3).tolist()}")
    print("camera-to-camera rotation error (deg); a common IMU-frame offset cancels here")
    for first, second in itertools.combinations(estimates, 2):
        relative = (estimates[first] @ estimates[second].T) @ (
            references[first] @ references[second].T
        ).T
        print(f"  {first}-{second}: {np.degrees(Rotation.from_matrix(relative).magnitude()):.3f}")
    stacked = np.array(list(errors.values()))
    common = stacked.mean(axis=0)
    print(
        "common IMU-frame offset (deg):",
        np.round(np.degrees(common), 3).tolist(),
        "spread:",
        np.round(np.degrees(stacked.std(axis=0)), 3).tolist(),
    )
    offset = Rotation.from_rotvec(common).as_matrix()
    for name in estimates:
        residual = estimates[name] @ (references[name] @ offset).T
        angle = np.degrees(Rotation.from_matrix(residual).magnitude())
        print(f"  {name} after removing the common offset: {angle:.3f} deg")


if __name__ == "__main__":
    main()
