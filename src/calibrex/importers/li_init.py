"""Import LI-Init (hku-mars/LiDAR_IMU_Init) results as external-run artifacts.

LI-Init is GPL-2.0 and runs only inside a container (see
``tools/external/li_init``); this module parses the plain-text
``Initialization_result.txt`` it writes and never links against it.

Conventions, taken from LI-Init's ``laserMapping.cpp``:

* ``offset_R_L_I`` / ``offset_T_L_I`` map LiDAR points into the IMU frame
  (``p_imu = R p_lidar + T``), i.e. ``T_imu_lidar``.  The file prints that
  4x4 matrix as "Homogeneous Transformation Matrix from LiDAR to IMU".  The
  artifact records it as ``T_imu_lidar`` and also as its inverse
  ``T_lidar_imu``, which is what Calibrex's IMU-LiDAR evaluation uses.
* "Time Lag IMU to LiDAR" is subtracted from IMU stamps, so
  ``t_imu = t_lidar + lag`` (Calibrex's ``dt``).

When refinement ran, the file holds an "Initialization result" section
followed by a "Refinement result" section; the refined values are imported.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.external_run import (
    ExternalArtifactDigest,
    ExternalCalibrationRunArtifact,
    ExternalDataIsolation,
    ExternalExecution,
    ExternalParsedOutputs,
    ExternalRunProvenance,
    ExternalRunStatus,
    ExternalToolIdentity,
    ExternalTransformOutput,
)
from calibrex.core.provenance import git_commit, sha256_path

LI_INIT_ADAPTER_VERSION = "li_init_result/v0.1"
LI_INIT_SOURCE_REPOSITORY = "https://github.com/hku-mars/LiDAR_IMU_Init"
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_li_init_result(text: str) -> dict[str, object]:
    """Parse the last (refined when present) section of ``Initialization_result.txt``."""

    sections = re.split(r"^(Initialization result:|Refinement result:)\s*$", text, flags=re.M)
    labelled = [
        (sections[index].rstrip(":"), sections[index + 1]) for index in range(1, len(sections), 2)
    ]
    if not labelled:
        labelled = [("Initialization result", text)]
    stage, body = labelled[-1]

    def vector(label: str) -> list[float]:
        match = re.search(re.escape(label) + r"[^=]*=\s*(.+)", body)
        if match is None:
            raise ValueError(f"LI-Init result lacks '{label}'")
        values = [float(value) for value in re.findall(_NUMBER, match.group(1))]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"LI-Init result has no finite values for '{label}'")
        return values

    matrix_match = re.search(
        r"Homogeneous Transformation Matrix from LiDAR to IMU:\s*\n((?:\s*"
        + _NUMBER
        + r".*\n?){4})",
        body,
    )
    if matrix_match is None:
        raise ValueError("LI-Init result lacks the LiDAR-to-IMU homogeneous matrix")
    rows = [
        [float(value) for value in re.findall(_NUMBER, line)]
        for line in matrix_match.group(1).strip().splitlines()
    ]
    matrix = np.array(rows, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("LI-Init homogeneous matrix is not a finite 4x4 matrix")
    return {
        "stage": stage,
        "T_imu_lidar": matrix,
        "time_lag_imu_to_lidar_s": vector("Time Lag IMU to LiDAR")[0],
        "gyro_bias_rps": vector("Bias of Gyroscope"),
        "accel_bias_mps2": vector("Bias of Accelerometer"),
        "gravity_mps2": vector("Gravity in World Frame"),
    }


def import_li_init_result(
    path: str | Path,
    *,
    input_artifacts: tuple[str | Path, ...] = (),
    container_digest: str | None = None,
    source_commit: str | None = None,
    command: list[str] | None = None,
    duration_seconds: float | None = None,
) -> ExternalCalibrationRunArtifact:
    """Convert an LI-Init result file into a generic external-run artifact."""

    source = Path(path)
    source_sha256 = sha256_path(source)
    artifacts = [
        artifact
        for input_path in input_artifacts
        if (artifact := _digest("input", Path(input_path))) is not None
    ]
    output = _digest("output", source)
    if output is not None:
        artifacts.append(output)
    warnings: list[str] = []
    if source_commit is None:
        warnings.append("LI-Init source commit is not declared")
    if not any(item.role == "input" for item in artifacts):
        warnings.append("LI-Init import has no digest-bound input bag")

    status: ExternalRunStatus
    outputs = ExternalParsedOutputs()
    if source_sha256 is None:
        status = "unavailable"
        warnings.append(f"LI-Init result file does not exist: {source}")
    else:
        try:
            parsed = parse_li_init_result(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            status = "invalid_output"
            warnings.append(f"LI-Init result could not be parsed: {exc}")
        else:
            status = "success"
            outputs = _outputs(parsed)
            if parsed["stage"] != "Refinement result":
                warnings.append("LI-Init result has no refinement section; initial values used")
    mode: Literal["container", "imported"] = "container" if container_digest else "imported"
    return ExternalCalibrationRunArtifact(
        run_id=f"li_init:{source.parent.name}",
        adapter_name="li_init_result",
        adapter_version=LI_INIT_ADAPTER_VERSION,
        tool=ExternalToolIdentity(
            name="li_init",
            source_repository=LI_INIT_SOURCE_REPOSITORY,
            source_commit=source_commit,
            license_spdx="GPL-2.0-only",
            license_boundary=mode,
        ),
        execution=ExternalExecution(
            mode=mode,
            command=command or [],
            container_digest=container_digest,
            attempted=container_digest is not None,
            duration_seconds=duration_seconds,
        ),
        artifacts=artifacts,
        frame_convention=(
            "T_imu_lidar maps LiDAR points into the IMU frame (LI-Init offset_R_L_I, "
            "offset_T_L_I); T_lidar_imu is its inverse."
        ),
        time_convention="t_imu = t_lidar + time_offset (LI-Init 'Time Lag IMU to LiDAR')",
        train_data_isolation=ExternalDataIsolation(declared=False),
        status=status,
        warnings=warnings,
        parsed_outputs=outputs,
        provenance=ExternalRunProvenance(
            calibrex_version=__version__,
            git_commit=git_commit(),
            source_artifact=str(source),
            source_artifact_sha256=source_sha256,
        ),
    )


def _outputs(parsed: dict[str, object]) -> ExternalParsedOutputs:
    imu_lidar = np.asarray(parsed["T_imu_lidar"], dtype=np.float64)
    lidar_imu = np.linalg.inv(imu_lidar)
    transforms = {}
    for key, parent, child, matrix in (
        ("T_imu_lidar", "imu", "lidar", imu_lidar),
        ("T_lidar_imu", "lidar", "imu", lidar_imu),
    ):
        transforms[key] = ExternalTransformOutput(
            parent=parent,
            child=child,
            translation_m=[float(value) for value in matrix[:3, 3]],
            rotation_quat_xyzw=[
                float(value) for value in Rotation.from_matrix(matrix[:3, :3]).as_quat()
            ],
        )
    gyro = parsed["gyro_bias_rps"]
    assert isinstance(gyro, list)
    return ExternalParsedOutputs(
        transforms=transforms,
        time_offsets_seconds={"imu_minus_lidar_s": float(parsed["time_lag_imu_to_lidar_s"])},  # type: ignore[arg-type]
        external_metrics={
            "stage": str(parsed["stage"]),
            "gyro_bias_x_rps": float(gyro[0]),
            "gyro_bias_y_rps": float(gyro[1]),
            "gyro_bias_z_rps": float(gyro[2]),
        },
    )


def _digest(role: Literal["input", "output"], path: Path) -> ExternalArtifactDigest | None:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        return None
    return ExternalArtifactDigest(
        role=role,
        path=str(path),
        sha256=digest,
        size_bytes=path.stat().st_size,
        media_type="text/plain" if path.suffix == ".txt" else "application/octet-stream",
    )
