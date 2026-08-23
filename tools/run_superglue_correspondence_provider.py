#!/usr/bin/env python3
"""Export external SuperGlue 2D--3D matches for KITTI-family captures.

This adapter intentionally keeps the external matching runtime outside the
core package.  It follows the virtual LiDAR-image convention used by
``direct_visual_lidar_calibration`` and never reads the dataset reference
extrinsic.  The resulting NPZ files are validated by the provider-export
boundary before they are used by a Calibrex solver.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from calibrex.core.camera_lidar_correspondence_export import (
    CameraLidarCorrespondenceExportFrame,
    CameraLidarCorrespondenceExportManifest,
)
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    CorrespondenceTrainingDeclaration,
    ProbabilisticCorrespondenceProvenance,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.depth import (
    DepthFileReference,
    depth_file_reference,
    load_depth_provider,
)
from calibrex.data.kitti import read_timestamps
from calibrex.data.kitti360_lidar_window_integration import (
    load_kitti360_lidar_window_integration,
    verify_kitti360_lidar_window_integration_files,
)
from calibrex.data.kitti_raw_lidar_window_integration import (
    load_kitti_raw_lidar_window_integration,
    verify_kitti_raw_lidar_window_integration_files,
)

KOIDE_REPOSITORY = "https://github.com/koide3/direct_visual_lidar_calibration"
KOIDE_COMMIT = "02a0dc039f5509708f384be4ff3228e0ae09352d"
SUPERGLUE_REPOSITORY = (
    "https://github.com/magicleap/SuperGluePretrainedNetwork"
)
SUPERGLUE_COMMIT = "ddcf11f42e7e0732a0c4607648f9448ea8d73590"
SUPERGLUE_LICENSE = "LicenseRef-SuperGlue-NonCommercial"
SUPERGLUE_ADAPTER_VERSION = "calibrex.superglue_correspondence_adapter/v0.5"

FloatArray: TypeAlias = NDArray[np.float64]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export license-pinned SuperGlue LiDAR-camera correspondences"
    )
    parser.add_argument("sequence_path", type=Path)
    parser.add_argument("--depth-provider", type=Path, required=True)
    parser.add_argument("--superglue-repository", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--lidar-directory", type=Path)
    parser.add_argument("--lidar-integration-manifest", type=Path)
    parser.add_argument(
        "--dataset-family",
        choices=("kitti_raw", "kitti_360"),
        required=True,
    )
    parser.add_argument("--split-id", default="evaluation")
    parser.add_argument("--camera-stream", default="image_02")
    parser.add_argument("--superglue-weights", choices=("indoor", "outdoor"), default="outdoor")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-keypoints", type=int, default=-1)
    parser.add_argument("--keypoint-threshold", type=float, default=0.05)
    parser.add_argument("--match-threshold", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument(
        "--equalize-camera-histogram",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lidar-intensity-normalization",
        choices=("rank-histogram", "percentile-linear"),
        default="rank-histogram",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate NPZ exports and a schema-valid correspondence manifest."""

    args = _parser().parse_args(argv)
    if args.max_keypoints != -1 and args.max_keypoints < 4:
        raise ValueError("--max-keypoints must be -1 or at least four")
    if not 0.0 <= args.keypoint_threshold <= 1.0:
        raise ValueError("--keypoint-threshold must be in [0, 1]")
    if not 0.0 <= args.match_threshold <= 1.0:
        raise ValueError("--match-threshold must be in [0, 1]")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be in [0, 1]")
    if args.top_k < 4:
        raise ValueError("--top-k must be at least four")
    if args.lidar_integration_manifest is not None and args.lidar_directory is None:
        raise ValueError(
            "--lidar-integration-manifest requires an explicit --lidar-directory"
        )

    sequence = args.sequence_path.resolve()
    provider_path = args.depth_provider.resolve()
    superglue = args.superglue_repository.resolve()
    output_directory = args.output_directory.resolve()
    manifest_path = args.manifest_output.resolve()
    _verify_superglue_repository(superglue)

    provider = load_depth_provider(provider_path)
    observations = {item.frame_id: item for item in provider.observations}
    if not observations:
        raise ValueError("depth provider has no observations")
    lidar_directory = (
        args.lidar_directory.resolve()
        if args.lidar_directory is not None
        else sequence / "velodyne_points" / "data"
    )
    if not lidar_directory.is_dir():
        raise ValueError(f"LiDAR directory does not exist: {lidar_directory}")

    timestamp_path = sequence / "velodyne_points" / "timestamps.txt"
    timestamps = {item.index: item.timestamp_ns for item in read_timestamps(timestamp_path)}
    output_directory.mkdir(parents=True, exist_ok=True)

    import torch

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    sys.path.insert(0, str(superglue))
    from models.matching import Matching  # type: ignore[import-not-found]
    from models.utils import frame2tensor  # type: ignore[import-not-found]

    config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": args.keypoint_threshold,
            "max_keypoints": args.max_keypoints,
        },
        "superglue": {
            "weights": args.superglue_weights,
            "sinkhorn_iterations": 20,
            "match_threshold": args.match_threshold,
        },
    }
    device = torch.device(args.device)
    matching = Matching(config).eval().to(device)

    checkpoint = superglue / "models" / "weights" / (
        f"superglue_{args.superglue_weights}.pth"
    )
    superpoint_checkpoint = superglue / "models" / "weights" / "superpoint_v1.pth"
    input_digests = {
        "depth_provider": _required_digest(provider_path),
        "superglue_license": _required_digest(superglue / "LICENSE"),
        "superglue_checkpoint": _required_digest(checkpoint),
        "superpoint_checkpoint": _required_digest(superpoint_checkpoint),
    }
    if args.lidar_integration_manifest is not None:
        integration_path = args.lidar_integration_manifest.resolve()
        _verify_lidar_integration_binding(
            integration_path,
            lidar_directory=lidar_directory,
            frame_ids=list(observations),
            sequence_id=sequence.name,
            dataset_family=args.dataset_family,
        )
        input_digests["lidar_integration_manifest"] = _required_digest(
            integration_path
        )
    frames: list[CameraLidarCorrespondenceExportFrame] = []
    match_counts: list[int] = []

    with torch.inference_mode():
        for position, observation in enumerate(provider.observations, start=1):
            frame_id = observation.frame_id
            frame_index = int(frame_id)
            if frame_index not in timestamps:
                raise ValueError(f"timestamps do not cover frame {frame_id}")
            camera_path = _reference_path(provider_path.parent, observation.image)
            lidar_path = lidar_directory / f"{frame_id}.bin"
            if not camera_path.is_file():
                raise ValueError(f"camera image does not exist: {camera_path}")
            if not lidar_path.is_file():
                raise ValueError(f"LiDAR scan does not exist: {lidar_path}")
            camera_digest = _required_digest(camera_path)
            if camera_digest != observation.image.sha256:
                raise ValueError(
                    f"provider image digest mismatch for frame {frame_id}: "
                    f"expected {observation.image.sha256}, observed {camera_digest}"
                )
            input_digests[f"image:{frame_id}"] = camera_digest
            input_digests[f"lidar:{frame_id}"] = _required_digest(lidar_path)

            image = cv2.imread(str(camera_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"cannot read camera image: {camera_path}")
            if args.equalize_camera_histogram:
                image = cv2.equalizeHist(image)
            points = np.fromfile(lidar_path, dtype=np.float32)
            if points.size % 4:
                raise ValueError(f"invalid KITTI point shape: {lidar_path}")
            point_cloud = points.reshape(-1, 4).astype(np.float64, copy=False)
            lidar_image, index_image = _virtual_lidar_image(
                point_cloud,
                intensity_normalization=args.lidar_intensity_normalization,
            )
            camera_tensor = frame2tensor(image, device)
            lidar_tensor = frame2tensor(lidar_image, device)
            prediction = matching(
                {"image0": camera_tensor, "image1": lidar_tensor}
            )
            keypoints_camera = prediction["keypoints0"][0].cpu().numpy()
            keypoints_lidar = prediction["keypoints1"][0].cpu().numpy()
            matches = prediction["matches0"][0].cpu().numpy()
            confidence = prediction["matching_scores0"][0].cpu().numpy()
            arrays = _matched_arrays(
                point_cloud,
                index_image,
                keypoints_camera,
                keypoints_lidar,
                matches,
                confidence,
                min_confidence=args.min_confidence,
                top_k=args.top_k,
            )
            count = int(arrays["point_lidar_m"].shape[0])
            if count < 4:
                raise ValueError(
                    f"SuperGlue produced only {count} valid 2D-3D matches "
                    f"for frame {frame_id}"
                )
            match_counts.append(count)
            export_path = output_directory / f"{frame_id}.npz"
            _save_npz(export_path, arrays)
            export_reference = depth_file_reference(
                export_path,
                encoding="npz_probabilistic_camera_lidar_v0.1",
            )
            frames.append(
                CameraLidarCorrespondenceExportFrame(
                    frame_id=frame_id,
                    capture_time_ns=observation.capture_time_ns,
                    camera_frame=observation.source_frame,
                    lidar_frame="lidar",
                    intrinsics=observation.intrinsics,
                    export=export_reference,
                )
            )
            print(
                f"[{position}/{len(provider.observations)}] {frame_id} "
                f"matches={count}",
                flush=True,
            )

    input_digests["lidar_directory"] = _digest_path_list(
        [lidar_directory / f"{item.frame_id}.bin" for item in provider.observations]
    )
    input_digests["camera_directory"] = _digest_path_list(
        [_reference_path(provider_path.parent, item.image) for item in provider.observations]
    )
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    dataset_stem = sequence.name.removesuffix("_sync")
    dataset_id = (
        f"kitti_raw_{dataset_stem}"
        if args.dataset_family == "kitti_raw"
        else f"kitti360_{dataset_stem}"
    )
    manifest = CameraLidarCorrespondenceExportManifest(
        artifact_id=(
            f"koide-superglue-{args.dataset_family}-{sequence.name}-"
            f"{args.camera_stream}-{len(frames)}"
        ),
        dataset_id=dataset_id,
        dataset_family=args.dataset_family,
        sequence_id=sequence.name,
        split_id=args.split_id,
        dataset_license_spdx=(
            "CC-BY-NC-SA-3.0"
            if args.dataset_family == "kitti_360"
            else "LicenseRef-KITTI"
        ),
        provider=CorrespondenceProviderIdentity(
            provider="Koide virtual-LiDAR SuperGlue adapter",
            model=f"SuperPoint+SuperGlue/{args.superglue_weights}",
            version=(
                f"koide-{KOIDE_COMMIT[:12]}+superglue-{SUPERGLUE_COMMIT[:12]}"
                "+calibrex-adapter-v0.5"
            ),
            source_repository=(
                f"{KOIDE_REPOSITORY};{SUPERGLUE_REPOSITORY}"
            ),
            source_commit=f"koide:{KOIDE_COMMIT};superglue:{SUPERGLUE_COMMIT}",
            license_spdx=SUPERGLUE_LICENSE,
            checkpoint=depth_file_reference(
                checkpoint,
                encoding="pytorch_state_dict",
            ),
            checkpoint_redistribution=(
                "local-only; SuperGlue license permits noncommercial internal "
                "research use and does not permit redistribution"
            ),
            training=CorrespondenceTrainingDeclaration(
                training_datasets=["SuperGlue official training mixture"],
                evaluation_overlap="unknown",
                test_data_used_for_online_refinement=False,
                evidence=(
                    "official SuperGlue outdoor checkpoint; overlap with the "
                    "KITTI-family evaluation captures was not independently verified"
                ),
            ),
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="tools.run_superglue_correspondence_provider",
            generator_version=SUPERGLUE_ADAPTER_VERSION,
            git_commit=git_commit(),
            command=command,
            input_sha256=input_digests,
        ),
        warnings=[
            "SuperGlue is restricted to noncommercial internal research use",
            (
                "Koide virtual LiDAR-image convention is MIT; its optional "
                "SuperGlue dependency is not MIT"
            ),
            "reference extrinsics were not read by this provider",
            (
                "LiDAR image uses Koide's fixed 1920x960 equirectangular "
                "projection and nearest-point index lookup"
            ),
            (
                "raw SuperGlue matching_scores0 values are represented once as "
                "reliability and are never normalized per frame"
            ),
            (
                "the raw SuperGlue score is represented once as reliability; "
                "no independent outlier probability is available, so it is zero"
            ),
            (
                "Koide-faithful preprocessing: camera histogram equalization="
                f"{args.equalize_camera_histogram}, LiDAR intensity normalization="
                f"{args.lidar_intensity_normalization}, SuperPoint keypoint threshold="
                f"{args.keypoint_threshold:g}, max keypoints={args.max_keypoints}, "
                f"SuperGlue match threshold={args.match_threshold:g}"
            ),
            (
                "the depth-provider artifact supplies digest-pinned camera frames and "
                "intrinsics; its predicted depth arrays are not consumed by this adapter"
            ),
            (
                "LiDAR integration manifest was verified and digest-pinned"
                if args.lidar_integration_manifest is not None
                else "no LiDAR integration manifest was supplied"
            ),
            f"per-frame valid correspondence count range={min(match_counts)}..{max(match_counts)}",
            (
                "KITTI raw dataset license is recorded as LicenseRef-KITTI; "
                "verify local terms before redistribution"
            ),
        ],
    )
    manifest.save(manifest_path)
    print(f"manifest={manifest_path}", flush=True)
    return 0


def _virtual_lidar_image(
    points: FloatArray,
    *,
    intensity_normalization: str = "rank-histogram",
) -> tuple[NDArray[np.uint8], NDArray[np.int32]]:
    """Render Koide's fixed equirectangular LiDAR intensity/index image."""

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must have shape Nx4")
    xyz = points[:, :3]
    finite = np.isfinite(points).all(axis=1)
    norms = np.linalg.norm(xyz, axis=1)
    finite &= norms > 1.0e-6
    angle = -math.pi / 2.0
    cosine, sine = math.cos(angle), math.sin(angle)
    lidar_camera = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )
    camera_xyz = xyz @ lidar_camera
    unit = np.linalg.norm(camera_xyz, axis=1)
    finite &= unit > 1.0e-6
    directions = np.zeros_like(camera_xyz)
    directions[finite] = camera_xyz[finite] / unit[finite, None]
    u = 1920.0 * (
        0.5 + np.arctan2(directions[:, 0], directions[:, 2]) / (2.0 * math.pi)
    )
    v = 960.0 * (
        0.5 + np.arcsin(np.clip(directions[:, 1], -1.0, 1.0)) / math.pi
    )
    pixels = np.floor(np.column_stack((u, v))).astype(np.int32)
    valid = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < 1920)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < 960)
    )
    image: NDArray[np.float32] = np.zeros((960, 1920), dtype=np.float32)
    index_image: NDArray[np.int32] = np.full(
        (960, 1920),
        -1,
        dtype=np.int32,
    )
    finite = np.isfinite(points).all(axis=1)
    finite_intensity = points[finite, 3]
    if finite_intensity.size == 0:
        raise ValueError("LiDAR scan has no finite intensity values")
    normalized = np.zeros(points.shape[0], dtype=np.float64)
    finite_indices = np.flatnonzero(finite)
    if intensity_normalization == "rank-histogram":
        order = np.argsort(finite_intensity, kind="stable")
        ranks = np.floor(
            256.0 * np.arange(len(order), dtype=np.float64) / len(order)
        ) / 256.0
        normalized[finite_indices[order]] = ranks
    elif intensity_normalization == "percentile-linear":
        low, high = np.percentile(finite_intensity, [1.0, 99.0])
        scale = max(float(high - low), 1.0e-6)
        normalized[finite_indices] = np.clip(
            (finite_intensity - low) / scale, 0.0, 1.0
        )
    else:
        raise ValueError(
            "unsupported LiDAR intensity normalization: "
            f"{intensity_normalization}"
        )
    valid_indices = np.flatnonzero(valid)
    flat_pixels = (
        pixels[valid_indices, 1] * image.shape[1] + pixels[valid_indices, 0]
    )
    order = np.lexsort(
        (
            valid_indices,
            norms[valid_indices],
            flat_pixels,
        )
    )
    ordered_pixels = flat_pixels[order]
    first_per_pixel: NDArray[np.bool_] = np.ones(len(order), dtype=bool)
    first_per_pixel[1:] = ordered_pixels[1:] != ordered_pixels[:-1]
    selected = valid_indices[order[first_per_pixel]]
    selected_x = pixels[selected, 0]
    selected_y = pixels[selected, 1]
    image[selected_y, selected_x] = normalized[selected]
    index_image[selected_y, selected_x] = selected.astype(np.int32)
    return np.rint(image * 255.0).astype(np.uint8), index_image


def _matched_arrays(
    points: FloatArray,
    index_image: NDArray[np.int32],
    keypoints_camera: FloatArray,
    keypoints_lidar: FloatArray,
    matches: NDArray[Any],
    confidence: NDArray[Any],
    *,
    min_confidence: float,
    top_k: int,
) -> dict[str, NDArray[Any]]:
    """Map virtual-image matches to unique 3D points and calibrated pixels."""

    candidates: list[tuple[float, int, float, float, float, float]] = []
    seen_points: set[int] = set()
    for camera_index, lidar_index in enumerate(matches):
        point_match = int(lidar_index)
        if point_match < 0 or camera_index >= len(confidence):
            continue
        score = float(confidence[camera_index])
        if not math.isfinite(score) or score < min_confidence:
            continue
        if point_match >= len(keypoints_lidar):
            continue
        point_index = _nearest_index(
            index_image,
            float(keypoints_lidar[point_match, 0]),
            float(keypoints_lidar[point_match, 1]),
        )
        if point_index is None or point_index in seen_points:
            continue
        seen_points.add(point_index)
        candidates.append(
            (
                score,
                point_index,
                float(keypoints_camera[camera_index, 0]),
                float(keypoints_camera[camera_index, 1]),
                float(keypoints_lidar[point_match, 0]),
                float(keypoints_lidar[point_match, 1]),
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    candidates = candidates[:top_k]
    count = len(candidates)
    point_values = np.asarray([points[item[1], :3] for item in candidates], dtype=np.float64)
    image_values = np.asarray(
        [[item[2], item[3]] for item in candidates],
        dtype=np.float64,
    )
    virtual_values = np.asarray(
        [[item[4], item[5]] for item in candidates],
        dtype=np.float64,
    )
    scores = np.asarray([item[0] for item in candidates], dtype=np.float64)
    sigma = 1.0 + 5.0 * np.clip(1.0 - scores, 0.0, 1.0)
    covariance = np.column_stack(
        (sigma * sigma, np.zeros(count), np.zeros(count), sigma * sigma)
    )
    return {
        "point_lidar_m": point_values,
        "image_mean_px": image_values,
        "image_covariance_px2": covariance,
        "outlier_probability": np.zeros(count, dtype=np.float64),
        "reliability": np.clip(scores, 0.0, 1.0),
        "provider_confidence": scores,
        "virtual_lidar_keypoint_px": virtual_values,
        "correspondence_id": np.asarray(
            [f"superglue-{index:05d}" for index in range(count)],
            dtype=str,
        ),
    }


def _nearest_index(
    index_image: NDArray[np.int32],
    x: float,
    y: float,
    radius: int = 4,
) -> int | None:
    """Find the closest rendered LiDAR point around a SuperPoint keypoint."""

    center_x = round(x)
    center_y = round(y)
    left = max(0, center_x - radius)
    right = min(index_image.shape[1], center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(index_image.shape[0], center_y + radius + 1)
    if left >= right or top >= bottom:
        return None
    window = index_image[top:bottom, left:right]
    candidates = np.argwhere(window >= 0)
    if candidates.size == 0:
        return None
    distances = (
        (candidates[:, 1] + left - x) ** 2
        + (candidates[:, 0] + top - y) ** 2
    )
    return int(window[tuple(candidates[int(np.argmin(distances))])])


def _save_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    """Write a non-pickle NPZ atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _verify_superglue_repository(repository: Path) -> None:
    if not (repository / ".git" / "HEAD").is_file():
        raise ValueError(f"SuperGlue repository is not a Git checkout: {repository}")
    observed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != SUPERGLUE_COMMIT:
        raise ValueError(
            f"SuperGlue commit mismatch: expected {SUPERGLUE_COMMIT}, got {observed}"
        )


def _verify_lidar_integration_binding(
    manifest_path: Path,
    *,
    lidar_directory: Path,
    frame_ids: list[str],
    sequence_id: str,
    dataset_family: str,
) -> None:
    if dataset_family == "kitti_360":
        artifact = load_kitti360_lidar_window_integration(manifest_path)
        issues = verify_kitti360_lidar_window_integration_files(artifact)
    elif dataset_family == "kitti_raw":
        artifact = load_kitti_raw_lidar_window_integration(manifest_path)
        issues = verify_kitti_raw_lidar_window_integration_files(artifact)
    else:
        raise ValueError(f"unsupported integration dataset family: {dataset_family}")
    if issues:
        raise ValueError(
            "LiDAR integration artifact file verification failed: "
            + ", ".join(issues)
        )
    if artifact.sequence_id != sequence_id:
        raise ValueError(
            "LiDAR integration sequence differs from the provider sequence: "
            f"{artifact.sequence_id} != {sequence_id}"
        )
    normalized_ids = sorted(set(frame_ids), key=int)
    if artifact.center_frame_ids != normalized_ids:
        raise ValueError(
            "LiDAR integration centers differ from depth-provider observations"
        )
    outputs = {window.center_frame_id: window.output for window in artifact.windows}
    for frame_id in normalized_ids:
        expected = (lidar_directory / f"{frame_id}.bin").resolve()
        observed = Path(outputs[frame_id].path).resolve()
        if observed != expected:
            raise ValueError(
                f"LiDAR integration output path mismatch for {frame_id}: "
                f"{observed} != {expected}"
            )


def _reference_path(base: Path, reference: DepthFileReference) -> Path:
    path = Path(reference.path)
    return path if path.is_absolute() else (base / path).resolve()


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        raise ValueError(f"required file is missing or unreadable: {path}")
    return digest


def _digest_path_list(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(_required_digest(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
