#!/usr/bin/env python3
"""Export I2PNet soft point-to-pixel correspondences.

I2PNet and PyTorch remain external to the Calibrex package.  This adapter
loads a digest-pinned official checkpoint, captures the learned coarse global
cost-volume logits, converts the model's own soft point-to-pixel weights, and
exports only provider-neutral NumPy arrays plus a schema-valid provenance
manifest.

The adapter requires an explicit initial ``T_camera_lidar`` artifact.  It
never accepts or loads a CameraLidarCalibrationProblem, which prevents it from
silently consuming that problem's reference transform.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from calibrex.core.camera_lidar_correspondence_export import (
    CameraLidarCorrespondenceExportFrame,
    CameraLidarCorrespondenceExportManifest,
)
from calibrex.core.geometry import (
    SE3,
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
    CorrespondenceTrainingDeclaration,
    ProbabilisticCorrespondenceProvenance,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import (
    TransformEstimateProvenance,
    TransformQuality,
    TransformResult,
)
from calibrex.data.depth import (
    DepthCameraIntrinsics,
    DepthFileReference,
    depth_file_reference,
    load_depth_provider,
)

I2PNET_REPOSITORY = "https://github.com/IRMVLab/I2PNet"
I2PNET_COMMIT = "7675f7f362be5c5ccbaab6ccea63a37ca377c633"
I2PNET_CHECKPOINT_SHA256 = "1c56e7a362075db53c073aa4e8d38492811a8f52983740b3627d3085c80f422a"
I2PNET_ADAPTER_VERSION = "calibrex.i2pnet_correspondence_adapter/v0.2"
I2PNET_CORRESPONDENCE_RELIABILITY_GATE = 0.01

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class _ImageMapping:
    """Mapping between a source camera image and I2PNet's input grid."""

    source_width: int
    source_height: int
    crop_top_rows: int
    scale: float
    crop_x_scaled_px: int
    crop_y_scaled_px: int
    output_width: int
    output_height: int

    def feature_pixels_to_source(
        self,
        indices: NDArray[np.int64],
        *,
        feature_height: int,
        feature_width: int,
    ) -> FloatArray:
        """Map flattened feature-grid coordinates to source-image pixels."""

        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError("feature indices must have shape NxK")
        if feature_height <= 0 or feature_width <= 0:
            raise ValueError("feature dimensions must be positive")
        if np.any(values < 0) or np.any(values >= feature_height * feature_width):
            raise ValueError("feature indices are outside the feature grid")
        feature_u = values % feature_width
        feature_v = values // feature_width
        processed_u = feature_u.astype(np.float64) * (self.output_width / feature_width)
        processed_v = feature_v.astype(np.float64) * (self.output_height / feature_height)
        source_u = (processed_u + self.crop_x_scaled_px) / self.scale
        source_v = (processed_v + self.crop_y_scaled_px) / self.scale + self.crop_top_rows
        return np.stack((source_u, source_v), axis=-1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export I2PNet soft LiDAR-camera correspondences")
    parser.add_argument("sequence_path", type=Path)
    parser.add_argument("--depth-provider", type=Path, required=True)
    parser.add_argument("--i2pnet-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path)
    parser.add_argument("--initial-transform", type=Path, required=True)
    parser.add_argument("--provider-patch", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--pose-output",
        type=Path,
        help="optional aggregate I2PNet pose-initializer TransformResult",
    )
    parser.add_argument("--lidar-directory", type=Path)
    parser.add_argument(
        "--dataset-family",
        choices=("kitti_raw", "kitti_360"),
        required=True,
    )
    parser.add_argument("--split-id", default="development")
    parser.add_argument("--camera-stream", default="image_00")
    parser.add_argument(
        "--frame-ids",
        help="comma-separated depth-provider frame IDs to evaluate",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--neighbor-backend",
        choices=("torch", "cuda_extension"),
        default="torch",
    )
    parser.add_argument("--crop-top-rows", type=int, default=50)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--input-height", type=int, default=160)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--model-point-count", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-reliability", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--covariance-floor-px2", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate digest-pinned NPZ exports and their provider manifest."""

    args = _parser().parse_args(argv)
    _validate_arguments(args)
    sequence = args.sequence_path.resolve()
    provider_path = args.depth_provider.resolve()
    repository = args.i2pnet_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    initial_path = args.initial_transform.resolve()
    patch_path = args.provider_patch.resolve() if args.provider_patch is not None else None
    archive_path = (
        args.checkpoint_archive.resolve() if args.checkpoint_archive is not None else None
    )
    output_directory = args.output_directory.resolve()
    manifest_path = args.manifest_output.resolve()
    pose_path = args.pose_output.resolve() if args.pose_output is not None else None
    lidar_directory = (
        args.lidar_directory.resolve()
        if args.lidar_directory is not None
        else sequence / "velodyne_points" / "data"
    )
    if not sequence.is_dir():
        raise ValueError(f"sequence directory does not exist: {sequence}")
    if not lidar_directory.is_dir():
        raise ValueError(f"LiDAR directory does not exist: {lidar_directory}")

    _verify_i2pnet_repository(repository, patch_path=patch_path)
    checkpoint_digest = _required_digest(checkpoint)
    if checkpoint_digest != I2PNET_CHECKPOINT_SHA256:
        raise ValueError(
            "I2PNet checkpoint digest mismatch: expected "
            f"{I2PNET_CHECKPOINT_SHA256}, observed {checkpoint_digest}"
        )
    provider = load_depth_provider(provider_path)
    initial = _load_initial_transform(initial_path)
    _verify_initial_transform(initial)
    if not provider.observations:
        raise ValueError("depth provider has no observations")
    observations = _select_observations(provider.observations, args.frame_ids)
    _verify_observation_frames(
        observations,
        initial=initial,
        camera_stream=args.camera_stream,
    )

    torch, model, config = _load_i2pnet_runtime(
        repository,
        checkpoint=checkpoint,
        device=args.device,
        neighbor_backend=args.neighbor_backend,
    )
    device = torch.device(args.device)
    output_directory.mkdir(parents=True, exist_ok=True)
    initial_matrix = _transform_matrix(initial)
    input_digests = _base_input_digests(
        provider_path=provider_path,
        repository=repository,
        checkpoint=checkpoint,
        initial_path=initial_path,
        patch_path=patch_path,
        archive_path=archive_path,
    )
    frames: list[CameraLidarCorrespondenceExportFrame] = []
    counts: list[int] = []
    reliabilities: list[float] = []
    pose_corrections: list[FloatArray] = []

    for position, observation in enumerate(observations, start=1):
        image_path = _reference_path(provider_path.parent, observation.image)
        lidar_path = lidar_directory / f"{observation.frame_id}.bin"
        _verify_reference(image_path, observation.image, label="camera image")
        lidar_digest = _required_digest(lidar_path)
        input_digests[f"image:{observation.frame_id}"] = observation.image.sha256
        input_digests[f"lidar:{observation.frame_id}"] = lidar_digest
        rgb, intrinsic, mapping = _preprocess_image(
            image_path,
            observation.intrinsics,
            crop_top_rows=args.crop_top_rows,
            scale=args.image_scale,
            output_height=args.input_height,
            output_width=args.input_width,
        )
        frame_seed = _frame_seed(args.seed, observation.frame_id)
        raw_points = _load_lidar_points(
            lidar_path,
            point_count=args.model_point_count,
            seed=frame_seed,
        )
        _seed_torch_runtime(torch, frame_seed)
        arrays = _run_i2pnet_frame(
            torch=torch,
            model=model,
            config=config,
            device=device,
            rgb_chw=rgb,
            raw_points=raw_points,
            initial_matrix=initial_matrix,
            intrinsic=intrinsic,
            mapping=mapping,
            min_reliability=args.min_reliability,
            top_k=args.top_k,
            covariance_floor_px2=args.covariance_floor_px2,
        )
        count = int(arrays["point_lidar_m"].shape[0])
        if count < 4:
            raise ValueError(
                f"I2PNet produced only {count} valid soft correspondences "
                f"for frame {observation.frame_id}"
            )
        export_path = output_directory / f"{observation.frame_id}.npz"
        _save_npz(export_path, arrays)
        frames.append(
            CameraLidarCorrespondenceExportFrame(
                frame_id=observation.frame_id,
                capture_time_ns=observation.capture_time_ns,
                camera_frame=observation.source_frame,
                lidar_frame=initial.child,
                intrinsics=observation.intrinsics,
                export=depth_file_reference(
                    export_path,
                    encoding="npz_i2pnet_soft_correspondence_v0.2",
                ),
            )
        )
        counts.append(count)
        reliabilities.extend(arrays["reliability"].tolist())
        pose_corrections.append(arrays["i2pnet_pose_correction_wxyz_t"].copy())
        print(
            f"[{position}/{len(observations)}] "
            f"{observation.frame_id} correspondences={count} "
            f"mean_reliability={float(np.mean(arrays['reliability'])):.6f}",
            flush=True,
        )

    input_digests["camera_files"] = _digest_path_list(
        [_reference_path(provider_path.parent, item.image) for item in observations]
    )
    input_digests["lidar_files"] = _digest_path_list(
        [lidar_directory / f"{item.frame_id}.bin" for item in observations]
    )
    _release_i2pnet_runtime(torch=torch, model=model, device=device)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    dataset_stem = sequence.name.removesuffix("_sync")
    dataset_id = (
        f"kitti_raw_{dataset_stem}"
        if args.dataset_family == "kitti_raw"
        else f"kitti360_{dataset_stem}"
    )
    mean_reliability = float(np.mean(reliabilities))
    correspondence_rejected = mean_reliability < I2PNET_CORRESPONDENCE_RELIABILITY_GATE
    manifest = CameraLidarCorrespondenceExportManifest(
        artifact_id=(
            f"i2pnet-{args.dataset_family}-{sequence.name}-{args.camera_stream}-{len(frames)}"
        ),
        dataset_id=dataset_id,
        dataset_family=args.dataset_family,
        sequence_id=sequence.name,
        split_id=args.split_id,
        dataset_license_spdx=(
            "CC-BY-NC-SA-3.0" if args.dataset_family == "kitti_360" else "LicenseRef-KITTI"
        ),
        provider=CorrespondenceProviderIdentity(
            provider="I2PNet diagnostic cost-volume adapter",
            model="I2PNet/KITTI-large",
            version=(
                f"i2pnet-{I2PNET_COMMIT[:12]}+"
                f"checkpoint-{I2PNET_CHECKPOINT_SHA256[:12]}+adapter-v0.2"
            ),
            source_repository=I2PNET_REPOSITORY,
            source_commit=I2PNET_COMMIT,
            license_spdx="MIT",
            checkpoint=depth_file_reference(
                checkpoint,
                encoding="pytorch_checkpoint_i2pnet_kitti_large",
            ),
            checkpoint_redistribution=(
                "not redistributed by Calibrex; user-supplied official Google "
                "Drive asset; no separate checkpoint redistribution grant was "
                "identified"
            ),
            training=CorrespondenceTrainingDeclaration(
                training_datasets=["KITTI Odometry"],
                evaluation_overlap="unknown",
                test_data_used_for_online_refinement=False,
                evidence=(
                    "official I2PNet README declares KITTI Odometry training; "
                    "exact raw-capture overlap was not independently audited"
                ),
            ),
        ),
        frames=frames,
        provenance=ProbabilisticCorrespondenceProvenance(
            generator="tools.run_i2pnet_correspondence_provider",
            generator_version=I2PNET_ADAPTER_VERSION,
            git_commit=git_commit(),
            command=command,
            input_sha256=input_digests,
        ),
        warnings=[
            (
                "development correspondence verdict=rejected; mean reliability "
                f"{mean_reliability:.9f} is below the prespecified diagnostic gate "
                f"{I2PNET_CORRESPONDENCE_RELIABILITY_GATE:g}; do not use these "
                "correspondences for calibration"
                if correspondence_rejected
                else "development correspondence verdict=eligible for further holdout testing"
            ),
            "reference_transform_camera_lidar is not an accepted adapter input",
            (
                "explicit initial transform is required and digest-pinned: "
                f"estimate_id={initial.estimate_id}"
            ),
            (f"I2PNet source remains external to src/calibrex; repository commit={I2PNET_COMMIT}"),
            (
                "projection-neighbor backend="
                f"{args.neighbor_backend}; torch reproduces local-grid selection "
                "without invoking the unstable Windows fused-kernel path"
            ),
            (
                "Windows int64 portability patch is digest-pinned"
                if patch_path is not None
                else "no external I2PNet source patch was declared"
            ),
            (
                "official KITTI-large preprocessing: crop top rows="
                f"{args.crop_top_rows}, scale={args.image_scale:g}, center crop="
                f"{args.input_width}x{args.input_height}, RGB values remain 0..255"
            ),
            (
                "learned coarse global cost-volume channel weights are averaged "
                "into one point-to-pixel distribution; reliability combines mean "
                "channel sharpness with Jensen-Shannon channel agreement"
            ),
            (
                "provider confidence is represented once as reliability; no "
                "independent outlier probability is available, so it is zero"
            ),
            (
                f"aggregate pose initializer={pose_path}"
                if pose_path is not None
                else "aggregate pose initializer was not requested"
            ),
            (
                f"covariance diagonal floor={args.covariance_floor_px2:g} px^2; "
                f"valid correspondence count range={min(counts)}..{max(counts)}"
            ),
            (f"reliability range={min(reliabilities):.9f}..{max(reliabilities):.9f}"),
            (
                "runtime="
                f"python-{platform.python_version()},numpy-{np.__version__},"
                f"opencv-{cv2.__version__},torch-{torch.__version__},"
                f"cuda-{torch.version.cuda},device-{device}"
            ),
            (
                "depth-provider predicted depth is not consumed; the artifact "
                "supplies digest-pinned camera images, intrinsics, frame IDs, and "
                "capture times"
            ),
            (
                f"sampling seed={args.seed}; NumPy and Torch per-frame seeds are "
                "SHA-256-derived from the frame ID"
            ),
            "PyTorch deterministic algorithms and cuBLAS workspace configuration are enforced",
            "selected frame IDs=" + ",".join(item.frame_id for item in observations),
        ],
    )
    manifest.save(manifest_path)
    print(f"manifest={manifest_path}", flush=True)
    if pose_path is not None:
        pose_result = _aggregate_pose_result(
            initial=initial,
            corrections=pose_corrections,
            frame_ids=[str(item.frame_id) for item in observations],
            checkpoint=checkpoint,
            checkpoint_digest=checkpoint_digest,
            initial_digest=input_digests["initial_transform"],
            command=command,
            source_digests={
                "correspondence_manifest": _required_digest(manifest_path),
                **input_digests,
            },
        )
        write_mapping(
            pose_path,
            pose_result.model_dump(mode="json", exclude_none=True),
        )
        print(f"pose_result={pose_path}", flush=True)
    return 0


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.crop_top_rows < 0:
        raise ValueError("--crop-top-rows must be non-negative")
    if not math.isfinite(args.image_scale) or args.image_scale <= 0.0:
        raise ValueError("--image-scale must be positive")
    if args.input_height < 32 or args.input_width < 32:
        raise ValueError("I2PNet input dimensions must be at least 32 pixels")
    if args.model_point_count < 256:
        raise ValueError("--model-point-count must be at least 256")
    if not 0.0 <= args.min_reliability <= 1.0:
        raise ValueError("--min-reliability must be in [0, 1]")
    if args.top_k < 4:
        raise ValueError("--top-k must be at least four")
    if not math.isfinite(args.covariance_floor_px2) or args.covariance_floor_px2 <= 0.0:
        raise ValueError("--covariance-floor-px2 must be positive")


def _load_initial_transform(path: Path) -> TransformResult:
    try:
        return TransformResult.model_validate(read_mapping(path))
    except Exception as exc:
        raise ValueError(f"invalid initial transform artifact {path}: {exc}") from exc


def _verify_initial_transform(transform: TransformResult) -> None:
    provenance = transform.provenance
    if provenance.role_in_comparison != "initial":
        raise ValueError("initial transform provenance role must be 'initial'")
    if provenance.execution_mode == "dataset_reference":
        raise ValueError("dataset_reference transforms are forbidden as provider inputs")
    if provenance.producer == "unknown" or provenance.evidence_level == "unknown":
        raise ValueError("initial transform producer and evidence level must be declared")
    if not transform.estimate_id:
        raise ValueError("initial transform estimate_id must be declared")
    if not provenance.source:
        raise ValueError("initial transform provenance source must be declared")
    transform.as_se3()


def _verify_observation_frames(
    observations: list[Any],
    *,
    initial: TransformResult,
    camera_stream: str,
) -> None:
    frame_ids = [str(item.frame_id) for item in observations]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("depth-provider frame IDs must be unique")
    camera_frames = {str(item.source_frame) for item in observations}
    if camera_frames != {initial.parent}:
        raise ValueError(
            "initial transform parent differs from observation camera frames: "
            f"{initial.parent!r} != {sorted(camera_frames)!r}"
        )
    expected_frames = {
        "image_00": "camera_0",
        "image_01": "camera_1",
        "image_02": "camera_2",
        "image_03": "camera_3",
    }
    expected_frame = expected_frames.get(camera_stream)
    if expected_frame is None:
        raise ValueError(f"unsupported KITTI camera stream: {camera_stream}")
    if camera_frames != {expected_frame}:
        raise ValueError(
            "camera stream differs from observation camera frames: "
            f"{camera_stream!r} implies {expected_frame!r}, observed "
            f"{sorted(camera_frames)!r}"
        )


def _select_observations(observations: list[Any], raw: str | None) -> list[Any]:
    if raw is None:
        return list(observations)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError("--frame-ids must contain at least one frame ID")
    if len(requested) != len(set(requested)):
        raise ValueError("--frame-ids must not contain duplicates")
    by_id = {str(item.frame_id): item for item in observations}
    missing = [frame_id for frame_id in requested if frame_id not in by_id]
    if missing:
        raise ValueError(
            "requested frame IDs are absent from the depth provider: " + ", ".join(missing)
        )
    return [by_id[frame_id] for frame_id in requested]


def _verify_i2pnet_repository(
    repository: Path,
    *,
    patch_path: Path | None,
) -> None:
    if not (repository / ".git" / "HEAD").is_file():
        raise ValueError(f"I2PNet repository is not a Git checkout: {repository}")
    observed = _git_output(repository, "rev-parse", "HEAD")
    if observed != I2PNET_COMMIT:
        raise ValueError(f"I2PNet commit mismatch: expected {I2PNET_COMMIT}, got {observed}")
    license_path = repository / "LICENSE"
    if not license_path.is_file() or "MIT License" not in license_path.read_text(encoding="utf-8"):
        raise ValueError("I2PNet checkout is missing its MIT LICENSE")
    changed = {
        line.strip()
        for line in _git_output(repository, "diff", "--name-only").splitlines()
        if line.strip()
    }
    if patch_path is None:
        if changed:
            raise ValueError(
                "I2PNet repository has undocumented tracked changes: " + ", ".join(sorted(changed))
            )
        return
    _required_digest(patch_path)
    expected = {
        "src/projectPN/fused_conv_select/fused_conv_g.cpp",
        "src/projectPN/fused_conv_select/fused_conv_go.cu",
        "src/projectPN/fused_conv_select/fused_conv_gpu.h",
    }
    if changed != expected:
        raise ValueError(
            "I2PNet tracked changes differ from the declared portability patch: "
            f"expected={sorted(expected)}, observed={sorted(changed)}"
        )
    checked = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "apply",
            "--reverse",
            "--check",
            "--unidiff-zero",
            "--ignore-whitespace",
            str(patch_path),
        ],
        capture_output=True,
        text=True,
    )
    if checked.returncode != 0:
        raise ValueError(
            "declared I2PNet portability patch is not applied: "
            + (checked.stderr.strip() or checked.stdout.strip())
        )


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _torch_neighbor_copy(
    xyz1_proj: Any,
    xyz2_proj: Any,
    idx_n2: Any,
    kernel_shape: Any,
    knn_points: int,
    stride_h: int = 1,
    stride_w: int = 1,
    distance: float = 10,
) -> tuple[Any, Any, Any, Any]:
    """Reproduce I2PNet's circular local-grid KNN with nearest padding."""

    return _torch_local_grid_neighbors(
        xyz1_proj,
        xyz2_proj,
        idx_n2,
        kernel_shape,
        knn_points,
        stride_h=stride_h,
        stride_w=stride_w,
        distance=distance,
        copy_nearest=True,
    )


def _torch_neighbor_att(
    xyz1_proj: Any,
    xyz2_proj: Any,
    idx_n2: Any,
    kernel_shape: Any,
    knn_points: int,
    stride_h: int = 1,
    stride_w: int = 1,
    distance: float = 10,
) -> tuple[Any, Any, Any, Any]:
    """Reproduce I2PNet's circular local-grid KNN without padding."""

    return _torch_local_grid_neighbors(
        xyz1_proj,
        xyz2_proj,
        idx_n2,
        kernel_shape,
        knn_points,
        stride_h=stride_h,
        stride_w=stride_w,
        distance=distance,
        copy_nearest=False,
    )


def _torch_local_grid_neighbors(
    xyz1_proj: Any,
    xyz2_proj: Any,
    idx_n2: Any,
    kernel_shape: Any,
    knn_points: int,
    *,
    stride_h: int,
    stride_w: int,
    distance: float,
    copy_nearest: bool,
) -> tuple[Any, Any, Any, Any]:
    """Pure-Torch equivalent of I2PNet's fused neighbor-selection kernel."""

    import torch

    if xyz1_proj.ndim != 4 or xyz2_proj.ndim != 4:
        raise ValueError("projected point tensors must have shape BxHxWxC")
    if xyz1_proj.shape[0] != xyz2_proj.shape[0]:
        raise ValueError("projected point tensors must have the same batch size")
    if xyz1_proj.shape[-1] < 3 or xyz2_proj.shape[-1] < 3:
        raise ValueError("projected point tensors must contain XYZ channels")
    if idx_n2.ndim != 3 or idx_n2.shape[0] != xyz1_proj.shape[0] or idx_n2.shape[2] != 2:
        raise ValueError("query indices must have shape BxNx2")
    if xyz1_proj.device != xyz2_proj.device or xyz1_proj.device != idx_n2.device:
        raise ValueError("projected points and query indices must share a device")
    kernel_height, kernel_width = (int(value) for value in kernel_shape)
    candidate_count = kernel_height * kernel_width
    if kernel_height <= 0 or kernel_width <= 0:
        raise ValueError("neighbor kernel dimensions must be positive")
    if knn_points <= 0 or knn_points > candidate_count:
        raise ValueError("knn_points must be within the local-grid candidate count")
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("neighbor strides must be positive")
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("neighbor distance must be positive")

    batch, source_height, source_width, _ = xyz1_proj.shape
    target_height, target_width = xyz2_proj.shape[1:3]
    indices = idx_n2.to(dtype=torch.long)
    center_h = indices[:, :, 0]
    center_w = indices[:, :, 1]
    center_in_bounds = (
        (center_h >= 0) & (center_h < source_height) & (center_w >= 0) & (center_w < source_width)
    )
    safe_center_h = center_h.clamp(0, source_height - 1)
    safe_center_w = center_w.clamp(0, source_width - 1)
    batch_indices = torch.arange(batch, device=xyz1_proj.device).view(batch, 1)
    center_xyz = xyz1_proj[
        batch_indices,
        safe_center_h,
        safe_center_w,
        :3,
    ]

    offsets = torch.arange(candidate_count, device=xyz1_proj.device)
    offset_h = offsets.div(kernel_width, rounding_mode="floor") - kernel_height // 2
    offset_w = offsets.remainder(kernel_width) - kernel_width // 2
    candidate_h = center_h[:, :, None].div(stride_h, rounding_mode="floor") + offset_h
    candidate_w = center_w[:, :, None].div(stride_w, rounding_mode="floor") + offset_w
    candidate_in_bounds = (candidate_h >= 0) & (candidate_h < target_height)
    safe_candidate_h = candidate_h.clamp(0, target_height - 1)
    wrapped_candidate_w = candidate_w.remainder(target_width)
    candidate_xyz = xyz2_proj[
        batch_indices[:, :, None],
        safe_candidate_h,
        wrapped_candidate_w,
        :3,
    ]

    center_valid = center_in_bounds & ((center_xyz * center_xyz).sum(dim=-1) > 1.0e-10)
    candidate_valid = (candidate_xyz * candidate_xyz).sum(dim=-1) > 1.0e-10
    squared_distance = ((candidate_xyz - center_xyz[:, :, None, :]) ** 2).sum(dim=-1)
    valid = (
        center_valid[:, :, None]
        & candidate_in_bounds
        & candidate_valid
        & (squared_distance <= float(distance) ** 2)
    )
    ranked_distance = torch.where(
        valid,
        squared_distance,
        torch.full_like(squared_distance, torch.inf),
    )
    selected_distance, selection = torch.topk(
        ranked_distance,
        knn_points,
        dim=2,
        largest=False,
        sorted=True,
    )
    selected_h = torch.gather(safe_candidate_h, 2, selection)
    selected_w = torch.gather(wrapped_candidate_w, 2, selection)
    selected_valid = torch.isfinite(selected_distance)

    if copy_nearest:
        has_neighbor = selected_valid[:, :, :1]
        selected_h = torch.where(selected_valid, selected_h, selected_h[:, :, :1])
        selected_w = torch.where(selected_valid, selected_w, selected_w[:, :, :1])
        selected_valid = has_neighbor.expand_as(selected_valid)

    zeros = torch.zeros_like(selected_h)
    selected_h = torch.where(selected_valid, selected_h, zeros).to(dtype=torch.long)
    selected_w = torch.where(selected_valid, selected_w, zeros).to(dtype=torch.long)
    selected_b = batch_indices[:, :, None].expand_as(selected_h)
    selected_b = torch.where(selected_valid, selected_b, zeros).to(dtype=torch.long)
    mask = selected_valid.unsqueeze(-1).to(dtype=xyz1_proj.dtype)
    return (
        selected_b.contiguous(),
        selected_h.contiguous(),
        selected_w.contiguous(),
        mask.contiguous(),
    )


def _load_i2pnet_runtime(
    repository: Path,
    *,
    checkpoint: Path,
    device: str,
    neighbor_backend: str,
) -> tuple[Any, Any, Any]:
    paths = (
        repository,
        repository / "pointnet2",
        repository / "src" / "projectPN" / "fused_conv_select",
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {device}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from src.config_proj_lidarcenter import I2PNetConfig
    from src.modellearn_proj_center import RegNet_v2
    from src.projectPN import PPBackbone_center
    from src.projectPN import utils as project_utils

    if neighbor_backend == "torch":
        project_utils.get_neighbor_copy = _torch_neighbor_copy
        project_utils.get_neighbor_att = _torch_neighbor_att
        PPBackbone_center.get_neighbor_copy = _torch_neighbor_copy
        PPBackbone_center.get_neighbor_att = _torch_neighbor_att
    elif neighbor_backend != "cuda_extension":
        raise ValueError(f"unsupported neighbor backend: {neighbor_backend}")

    config = I2PNetConfig()
    model = RegNet_v2(eval_info=True, cfg=config)
    allowed_globals = [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(allowed_globals):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("I2PNet checkpoint has no model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval().to(torch.device(device))
    torch.set_grad_enabled(False)
    return torch, model, config


def _release_i2pnet_runtime(*, torch: Any, model: Any, device: Any) -> None:
    """Release CUDA state before Windows unloads custom extension DLLs."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.to(torch.device("cpu"))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _preprocess_image(
    path: Path,
    intrinsics: DepthCameraIntrinsics,
    *,
    crop_top_rows: int,
    scale: float,
    output_height: int,
    output_width: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], _ImageMapping]:
    """Apply the official KITTI-large test-time crop/scale/center crop."""

    if intrinsics.projection != "pinhole":
        raise ValueError("I2PNet KITTI-large requires pinhole intrinsics")
    if intrinsics.distortion_model != "none" and any(
        abs(value) > 1.0e-12 for value in intrinsics.distortion
    ):
        raise ValueError("I2PNet adapter requires an already rectified image")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read camera image: {path}")
    source_height, source_width = bgr.shape[:2]
    if (source_width, source_height) != (intrinsics.width, intrinsics.height):
        raise ValueError(
            f"image/intrinsics size mismatch for {path}: "
            f"{source_width}x{source_height} != "
            f"{intrinsics.width}x{intrinsics.height}"
        )
    if crop_top_rows >= source_height:
        raise ValueError("top crop removes the complete source image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[crop_top_rows:, :, :]
    scaled_width = round(source_width * scale)
    scaled_height = round((source_height - crop_top_rows) * scale)
    if scaled_width < output_width or scaled_height < output_height:
        raise ValueError(
            "scaled image is smaller than the requested I2PNet center crop: "
            f"{scaled_width}x{scaled_height} < {output_width}x{output_height}"
        )
    rgb = cv2.resize(
        rgb,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LINEAR,
    )
    crop_x = (scaled_width - output_width) // 2
    crop_y = (scaled_height - output_height) // 2
    rgb = rgb[crop_y : crop_y + output_height, crop_x : crop_x + output_width]
    matrix = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix[1, 2] -= crop_top_rows
    matrix[:2, :] *= scale
    matrix[0, 2] -= crop_x
    matrix[1, 2] -= crop_y
    mapping = _ImageMapping(
        source_width=source_width,
        source_height=source_height,
        crop_top_rows=crop_top_rows,
        scale=scale,
        crop_x_scaled_px=crop_x,
        crop_y_scaled_px=crop_y,
        output_width=output_width,
        output_height=output_height,
    )
    return (
        np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32),
        matrix.astype(np.float32),
        mapping,
    )


def _load_lidar_points(
    path: Path,
    *,
    point_count: int,
    seed: int,
) -> NDArray[np.float32]:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        raise ValueError(f"invalid KITTI LiDAR scan: {path}")
    points = values.reshape(-1, 4)[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    points = points[np.linalg.norm(points, axis=1) > 1.0e-6]
    if points.shape[0] < 256:
        raise ValueError(f"LiDAR scan has too few finite points: {path}")
    order = np.random.default_rng(seed).permutation(points.shape[0])
    points = points[order[:point_count]]
    if points.shape[0] < point_count:
        points = np.pad(points, ((0, point_count - points.shape[0]), (0, 0)))
    return np.ascontiguousarray(points, dtype=np.float32)


def _run_i2pnet_frame(
    *,
    torch: Any,
    model: Any,
    config: Any,
    device: Any,
    rgb_chw: NDArray[np.float32],
    raw_points: NDArray[np.float32],
    initial_matrix: FloatArray,
    intrinsic: NDArray[np.float32],
    mapping: _ImageMapping,
    min_reliability: float,
    top_k: int,
    covariance_floor_px2: float,
) -> dict[str, NDArray[Any]]:
    homogeneous = np.column_stack((raw_points.astype(np.float64), np.ones(raw_points.shape[0])))
    camera_points = (initial_matrix @ homogeneous.T).T[:, :3].astype(np.float32)
    captured_cost: list[tuple[Any, ...]] = []
    captured_logits: list[Any] = []
    feature_shapes: list[tuple[int, int]] = []

    def capture_cost(_module: Any, inputs: tuple[Any, ...]) -> None:
        captured_cost.append(inputs)

    def capture_logits(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        captured_logits.append(output)

    def capture_feature(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        feature_shapes.append((int(output.shape[-2]), int(output.shape[-1])))

    cost_handle = model.cost_volume1.register_forward_pre_hook(capture_cost)
    logits_handle = model.cost_volume1.mlp2_convs[-1].register_forward_hook(capture_logits)
    feature_handle = model.RGB_net3.register_forward_hook(capture_feature)
    try:
        with torch.inference_mode():
            outputs = model(
                torch.from_numpy(rgb_chw[None]).to(device),
                torch.from_numpy(camera_points[None]).to(device),
                torch.from_numpy(raw_points[None]).to(device),
                torch.from_numpy(initial_matrix[:3][None].astype(np.float32)).to(device),
                torch.from_numpy(intrinsic[None]).to(device),
                torch.tensor([[mapping.scale, mapping.scale]], device=device),
                lidar_feature=torch.from_numpy(camera_points[None]).to(device),
                cfg=config,
            )
    finally:
        cost_handle.remove()
        logits_handle.remove()
        feature_handle.remove()
    if len(captured_cost) != 1 or len(captured_logits) != 1 or len(feature_shapes) != 1:
        raise ValueError("I2PNet coarse cost-volume capture was not unique")
    raw_sampled, candidate_indices, channel_weights = _coarse_soft_weights(
        torch=torch,
        cost_volume=model.cost_volume1,
        inputs=captured_cost[0],
        logits=captured_logits[0],
    )
    arrays = _soft_correspondence_arrays(
        raw_sampled,
        candidate_indices,
        channel_weights,
        feature_height=feature_shapes[0][0],
        feature_width=feature_shapes[0][1],
        mapping=mapping,
        min_reliability=min_reliability,
        top_k=top_k,
        covariance_floor_px2=covariance_floor_px2,
    )
    pose = outputs[0][0].detach().cpu().numpy().astype(np.float64)
    arrays["i2pnet_pose_correction_wxyz_t"] = pose
    return arrays


def _coarse_soft_weights(
    *,
    torch: Any,
    cost_volume: Any,
    inputs: tuple[Any, ...],
    logits: Any,
) -> tuple[FloatArray, NDArray[np.int64], FloatArray]:
    """Extract I2PNet's learned coarse global cost-volume WQ tensor."""

    if len(inputs) < 7:
        raise ValueError("unexpected I2PNet coarse cost-volume signature")
    xyz_raw, _, _, _, image_xyz, _, _ = inputs[:7]
    if cost_volume.nsample_q > 0 or not cost_volume.backward_validation:
        raise ValueError("unexpected I2PNet KITTI-large coarse cost-volume settings")
    if xyz_raw.shape[0] != 1 or image_xyz.shape[0] != 1 or logits.shape[0] != 1:
        raise ValueError("I2PNet correspondence export currently requires batch size one")
    point_count = int(xyz_raw.shape[1] * xyz_raw.shape[2])
    candidate_count = int(image_xyz.shape[1])
    expected = (1, point_count, candidate_count)
    if tuple(logits.shape[:3]) != expected:
        raise ValueError(
            "unexpected I2PNet coarse cost-volume dimensions: "
            f"expected prefix {expected}, observed {tuple(logits.shape[:3])}"
        )
    with torch.inference_mode():
        weights = torch.nn.functional.softmax(logits, dim=2)
        point_indices = (
            torch.arange(candidate_count, device=logits.device, dtype=torch.long)
            .view(1, 1, candidate_count)
            .expand(1, point_count, candidate_count)
        )
    return (
        xyz_raw.reshape(1, point_count, 3)[0].detach().cpu().numpy().astype(np.float64),
        point_indices[0].detach().cpu().numpy().astype(np.int64),
        weights[0].detach().cpu().numpy().astype(np.float64),
    )


def _soft_correspondence_arrays(
    raw_points: FloatArray,
    candidate_indices: NDArray[np.int64],
    channel_weights: FloatArray,
    *,
    feature_height: int,
    feature_width: int,
    mapping: _ImageMapping,
    min_reliability: float,
    top_k: int,
    covariance_floor_px2: float,
) -> dict[str, NDArray[Any]]:
    """Convert channel-wise WQ values to provider-neutral distributions."""

    points = np.asarray(raw_points, dtype=np.float64)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    weights = np.asarray(channel_weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("raw sampled points must have shape Nx3")
    if indices.ndim != 2 or indices.shape[0] != points.shape[0]:
        raise ValueError("candidate indices must have shape NxK")
    if weights.ndim != 3 or weights.shape[:2] != indices.shape:
        raise ValueError("channel weights must have shape NxKxC")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("soft correspondence weights must be finite and non-negative")
    channel_totals = weights.sum(axis=1, keepdims=True)
    if np.any(channel_totals <= 0.0):
        raise ValueError("each soft correspondence channel must have positive mass")
    channel_probabilities = weights / channel_totals
    probabilities = channel_probabilities.mean(axis=2)
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("soft correspondence rows must have positive mass")
    probabilities /= totals
    candidate_pixels = mapping.feature_pixels_to_source(
        indices,
        feature_height=feature_height,
        feature_width=feature_width,
    )
    means = np.sum(probabilities[:, :, None] * candidate_pixels, axis=1)
    offsets = candidate_pixels - means[:, None, :]
    covariance = np.einsum(
        "nk,nki,nkj->nij",
        probabilities,
        offsets,
        offsets,
    )
    covariance[:, 0, 0] += covariance_floor_px2
    covariance[:, 1, 1] += covariance_floor_px2
    candidate_count = probabilities.shape[1]
    mixture_entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1.0e-15, 1.0)),
        axis=1,
    )
    channel_entropy = -np.sum(
        channel_probabilities * np.log(np.clip(channel_probabilities, 1.0e-15, 1.0)),
        axis=1,
    )
    mean_channel_entropy = channel_entropy.mean(axis=1)
    if candidate_count == 1:
        channel_sharpness = np.ones(points.shape[0], dtype=np.float64)
        channel_agreement = np.ones(points.shape[0], dtype=np.float64)
    else:
        maximum_entropy = math.log(candidate_count)
        channel_sharpness = np.clip(
            1.0 - mean_channel_entropy / maximum_entropy,
            0.0,
            1.0,
        )
        normalized_divergence = np.clip(
            (mixture_entropy - mean_channel_entropy) / maximum_entropy,
            0.0,
            1.0,
        )
        channel_agreement = 1.0 - normalized_divergence
    reliability = channel_sharpness * channel_agreement
    valid = (
        np.isfinite(points).all(axis=1)
        & (np.linalg.norm(points, axis=1) > 1.0e-6)
        & np.isfinite(means).all(axis=1)
        & (means[:, 0] >= 0.0)
        & (means[:, 0] < mapping.source_width)
        & (means[:, 1] >= 0.0)
        & (means[:, 1] < mapping.source_height)
        & (reliability >= min_reliability)
    )
    source_indices = np.flatnonzero(valid)
    order = np.lexsort((source_indices, -reliability[source_indices]))
    selected = source_indices[order[:top_k]]
    count = len(selected)
    return {
        "point_lidar_m": points[selected],
        "image_mean_px": means[selected],
        "image_covariance_px2": covariance[selected].reshape(count, 4),
        "outlier_probability": np.zeros(count, dtype=np.float64),
        "reliability": reliability[selected],
        "provider_confidence": reliability[selected],
        "soft_correspondence_entropy": mixture_entropy[selected],
        "effective_candidate_count": np.exp(mixture_entropy[selected]),
        "channel_mean_entropy": mean_channel_entropy[selected],
        "channel_sharpness": channel_sharpness[selected],
        "channel_agreement": channel_agreement[selected],
        "i2pnet_p3_index": selected.astype(np.int64),
        "correspondence_id": np.asarray(
            [f"i2pnet-p3-{index:05d}" for index in selected],
            dtype=str,
        ),
    }


def _transform_matrix(transform: TransformResult) -> FloatArray:
    se3 = transform.as_se3()
    x, y, z, w = se3.rotation_quat_xyzw
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = se3.translation_m
    return matrix


def _aggregate_pose_result(
    *,
    initial: TransformResult,
    corrections: list[FloatArray],
    frame_ids: list[str],
    checkpoint: Path,
    checkpoint_digest: str,
    initial_digest: str,
    command: list[str],
    source_digests: dict[str, str] | None = None,
) -> TransformResult:
    """Compose and aggregate per-frame I2PNet corrections as a typed result."""

    if not corrections or len(corrections) != len(frame_ids):
        raise ValueError("pose corrections and frame IDs must be non-empty and aligned")
    initial_se3 = initial.as_se3()
    corrected = [_correction_se3(value).compose(initial_se3) for value in corrections]
    translations = np.asarray([item.translation_m for item in corrected], dtype=np.float64)
    quaternions = np.asarray([item.rotation_quat_xyzw for item in corrected], dtype=np.float64)
    mean_quaternion = _mean_quaternion_xyzw(quaternions)
    mean_transform = SE3(
        tuple(float(value) for value in translations.mean(axis=0)),
        mean_quaternion,
    )
    if len(corrected) > 1:
        translation_std = np.std(translations, axis=0, ddof=1)
        rotation_vectors = np.asarray(
            [
                _relative_rotation_vector_deg(
                    tuple(float(value) for value in quaternion),
                    mean_quaternion,
                )
                for quaternion in quaternions
            ],
            dtype=np.float64,
        )
        rotation_std = np.std(rotation_vectors, axis=0, ddof=1)
    else:
        translation_std = np.zeros(3, dtype=np.float64)
        rotation_std = np.zeros(3, dtype=np.float64)
    adapter_digest = _required_digest(Path(__file__).resolve())
    digest_notes = [
        f"source sha256 {name}={digest}" for name, digest in sorted((source_digests or {}).items())
    ]
    return TransformResult(
        parent=initial.parent,
        child=initial.child,
        translation_m=list(mean_transform.translation_m),
        rotation_quat_xyzw=list(mean_transform.rotation_quat_xyzw),
        estimate_id=(
            f"i2pnet-{len(frame_ids)}frame-aggregate-"
            f"{hashlib.sha256(','.join(frame_ids).encode()).hexdigest()[:12]}"
        ),
        quality=TransformQuality(
            grade="warn",
            std_translation_m=translation_std.tolist(),
            std_rotation_deg=rotation_std.tolist(),
        ),
        provenance=TransformEstimateProvenance(
            producer="external_tool",
            execution_mode="offline_batch",
            role_in_comparison="output",
            evidence_level="algorithmically_refined",
            source=I2PNET_REPOSITORY,
            source_path=str(checkpoint),
            tool_name="I2PNet/KITTI-large",
            tool_version=(f"commit-{I2PNET_COMMIT[:12]}+checkpoint-{checkpoint_digest[:12]}"),
            source_commit=I2PNET_COMMIT,
            license_spdx="MIT",
            adapter_version=I2PNET_ADAPTER_VERSION,
            command=subprocess.list2cmdline(command),
            notes=[
                "composition follows official training convention: "
                "T_camera_lidar_output = T_correction @ T_camera_lidar_initial",
                "frame IDs=" + ",".join(frame_ids),
                f"initial estimate_id={initial.estimate_id}",
                f"initial transform sha256={initial_digest}",
                f"checkpoint sha256={checkpoint_digest}",
                f"adapter sha256={adapter_digest}",
                "translation and quaternion means aggregate per-frame outputs",
                "reported standard deviations are frame-to-frame sample spread, "
                "not calibrated predictive uncertainty",
                *digest_notes,
            ],
        ),
    )


def _correction_se3(values: FloatArray) -> SE3:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("I2PNet pose correction must contain finite wxyz+t values")
    w, x, y, z, tx, ty, tz = (float(value) for value in pose)
    return SE3((tx, ty, tz), (x, y, z, w))


def _mean_quaternion_xyzw(values: FloatArray) -> tuple[float, float, float, float]:
    quaternions = np.asarray(values, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4 or not len(quaternions):
        raise ValueError("quaternions must have shape Nx4")
    normalized = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(normalized.T @ normalized)
    mean = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(mean @ normalized[0]) < 0.0:
        mean = -mean
    mean /= np.linalg.norm(mean)
    return tuple(float(value) for value in mean)


def _relative_rotation_vector_deg(
    quaternion: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    relative = quaternion_multiply_xyzw(
        quaternion,
        quaternion_conjugate_xyzw(reference),
    )
    vector = np.asarray(relative[:3], dtype=np.float64)
    scalar = float(relative[3])
    if scalar < 0.0:
        vector = -vector
        scalar = -scalar
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        return (0.0, 0.0, 0.0)
    angle_deg = math.degrees(2.0 * math.atan2(norm, scalar))
    result = vector * (angle_deg / norm)
    return tuple(float(value) for value in result)


def _base_input_digests(
    *,
    provider_path: Path,
    repository: Path,
    checkpoint: Path,
    initial_path: Path,
    patch_path: Path | None,
    archive_path: Path | None,
) -> dict[str, str]:
    values = {
        "depth_provider": _required_digest(provider_path),
        "initial_transform": _required_digest(initial_path),
        "i2pnet_checkpoint": _required_digest(checkpoint),
        "i2pnet_license": _required_digest(repository / "LICENSE"),
        "i2pnet_model_source": _required_digest(repository / "src" / "modellearn_proj_center.py"),
        "i2pnet_config_source": _required_digest(repository / "src" / "config_proj_lidarcenter.py"),
        "adapter_source": _required_digest(Path(__file__).resolve()),
    }
    if patch_path is not None:
        values["i2pnet_provider_patch"] = _required_digest(patch_path)
        for name in (
            "fused_conv_g.cpp",
            "fused_conv_go.cu",
            "fused_conv_gpu.h",
        ):
            values[f"patched_source:{name}"] = _required_digest(
                repository / "src" / "projectPN" / "fused_conv_select" / name
            )
    if archive_path is not None:
        values["i2pnet_checkpoint_archive"] = _required_digest(archive_path)
    return values


def _verify_reference(
    path: Path,
    reference: DepthFileReference,
    *,
    label: str,
) -> None:
    observed = _required_digest(path)
    if observed != reference.sha256:
        raise ValueError(f"{label} digest mismatch: expected {reference.sha256}, got {observed}")
    if path.stat().st_size != reference.size_bytes:
        raise ValueError(
            f"{label} size mismatch: expected {reference.size_bytes}, got {path.stat().st_size}"
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


def _frame_seed(seed: int, frame_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{frame_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _seed_torch_runtime(torch: Any, seed: int) -> None:
    torch_seed = seed % ((1 << 63) - 1)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _save_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
