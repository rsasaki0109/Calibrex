"""A2D2 pre-registered Camera-LiDAR D2D execution problem."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from calibrex import __version__
from calibrex.core.camera_lidar_artifacts import (
    CameraLidarArtifactProvenance,
    CameraLidarCalibrationProblem,
    CameraLidarObservationBinding,
)
from calibrex.core.io import write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult
from calibrex.data.depth import (
    DepthCameraIntrinsics,
    depth_file_reference,
    load_depth_provider,
)
from calibrex.data.manifest import DatasetManifest, StreamManifest

A2D2_SOURCE_URL = "https://www.a2d2.audi/a2d2/en.html"
A2D2_LICENSE = "CC-BY-ND-4.0"


def build_a2d2_camera_lidar_problem(
    data_directory: str | Path,
    depth_provider_path: str | Path,
    *,
    output_lidar_directory: str | Path,
    generated_manifest_path: str | Path,
    command: list[str] | None = None,
    problem_id: str = "a2d2-frontleft-preregistered-d2d",
    rotation_bound_deg: float = 20.0,
) -> CameraLidarCalibrationProblem:
    """Back-project official A2D2 pixels/depth and freeze a D2D smoke problem.

    A2D2's public NPZ points and pixel coordinates are already registered into
    the camera view.  The resulting identity transform is therefore useful for
    cross-family execution and capture-range diagnostics, but not independent
    extrinsic-accuracy evidence.
    """

    if rotation_bound_deg <= 0.0:
        raise ValueError("rotation_bound_deg must be positive")
    data_root = Path(data_directory).resolve()
    provider_file = Path(depth_provider_path).resolve()
    output_root = Path(output_lidar_directory).resolve()
    manifest_file = Path(generated_manifest_path).resolve()
    provider = load_depth_provider(provider_file)
    provider_digest = _required_digest(provider_file)
    output_root.mkdir(parents=True, exist_ok=True)

    input_paths = [provider_file]
    output_paths = []
    bindings = []
    point_counts: dict[str, int] = {}
    for observation in provider.observations:
        matches = sorted(
            data_root.glob(
                f"*_lidar_frontleft_{observation.frame_id}.npz"
            )
        )
        if len(matches) != 1:
            raise ValueError(
                "expected one A2D2 front-left LiDAR NPZ for frame "
                f"{observation.frame_id}, found {len(matches)}"
            )
        source = matches[0]
        input_paths.append(source)
        points = _camera_points_from_a2d2(source, observation.intrinsics)
        target = output_root / f"{observation.frame_id}.npy"
        np.save(target, points, allow_pickle=False)
        output_paths.append(target)
        point_counts[observation.frame_id] = len(points)
        bindings.append(
            CameraLidarObservationBinding(
                frame_id=observation.frame_id,
                split_id="evaluation",
                depth_observation_frame_id=observation.frame_id,
                lidar=depth_file_reference(
                    target, encoding="npy_xyz_float64"
                ),
                lidar_capture_time_ns=observation.capture_time_ns,
            )
        )
    manifest = DatasetManifest(
        name="a2d2-frontleft-preregistered-camera-points",
        description=(
            "Official A2D2 view-filtered LiDAR rows/columns/depth "
            "back-projected into a z-forward pinhole camera frame"
        ),
        time_base="paired A2D2 public frame identifier",
        streams={
            "camera_registered_lidar": StreamManifest(
                kind="pointcloud",
                count=len(output_paths),
                path=str(output_root),
                sensor="A2D2 lidar_frontleft",
                frame_id="camera_frontleft",
                fields=["x_camera", "y_camera", "z_camera"],
            )
        },
        provenance={
            "generator": "calibrex.data.a2d2_camera_lidar_problem",
            "generator_version": __version__,
            "git_commit": git_commit() or "unknown",
            "command": " ".join(command or []),
            "dataset_source": A2D2_SOURCE_URL,
            "dataset_license": A2D2_LICENSE,
            "input_sha256": _paths_digest(input_paths),
            "output_sha256": _paths_digest(output_paths),
            "point_counts": ",".join(
                f"{frame_id}:{count}"
                for frame_id, count in sorted(point_counts.items())
            ),
            "accuracy_limitation": (
                "source points are already registered into the camera view"
            ),
        },
    )
    write_mapping(manifest_file, manifest.model_dump(mode="json"))
    identity = TransformResult(
        parent="camera_frontleft",
        child="camera_registered_lidar",
        translation_m=[0.0, 0.0, 0.0],
        rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    return CameraLidarCalibrationProblem(
        problem_id=problem_id,
        dataset_id="A2D2-20180810-150607-frames-60-61",
        dataset_family="A2D2",
        sequence_id="20180810_150607",
        depth_provider_path=str(provider_file),
        depth_provider_sha256=provider_digest,
        observations=bindings,
        reference_transform_camera_lidar=identity,
        initial_transform_camera_lidar=identity,
        time_convention=(
            "image and view-filtered LiDAR are paired by official frame ID; "
            "numeric timestamps are unavailable in the small fixed sample"
        ),
        rotation_bound_deg=rotation_bound_deg,
        translation_bound_m=0.0,
        provenance=CameraLidarArtifactProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            config_sha256=_required_digest(manifest_file),
            source_sha256=_paths_digest(
                [provider_file, manifest_file, *input_paths[1:], *output_paths]
            ),
        ),
    )


def _camera_points_from_a2d2(
    path: Path,
    intrinsics: DepthCameraIntrinsics,
) -> NDArray[np.float64]:
    with np.load(path, allow_pickle=False) as payload:
        valid = np.asarray(payload["pcloud_attr.valid"], dtype=bool)
        row = np.asarray(payload["pcloud_attr.row"], dtype=float)
        col = np.asarray(payload["pcloud_attr.col"], dtype=float)
        depth = np.asarray(payload["pcloud_attr.depth"], dtype=float)
    width = intrinsics.width
    height = intrinsics.height
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.cx
    cy = intrinsics.cy
    selected = (
        valid
        & np.isfinite(row)
        & np.isfinite(col)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (row >= 0.0)
        & (row < height)
        & (col >= 0.0)
        & (col < width)
    )
    z = depth[selected]
    x = (col[selected] - cx) * z / fx
    y = (row[selected] - cy) * z / fy
    points: NDArray[np.float64] = np.column_stack((x, y, z)).astype(
        np.float64, copy=False
    )
    if len(points) < 64:
        raise ValueError(f"too few valid A2D2 camera-view points: {path}")
    return points


def _paths_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(_required_digest(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required file is unreadable: {path}")
    return digest
