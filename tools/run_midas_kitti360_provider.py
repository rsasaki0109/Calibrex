#!/usr/bin/env python3
"""Run pinned MiDaS v3.1 inference for KITTI-360 outside the core."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from calibrex import __version__
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.depth import (
    DepthImageTransform,
    DepthMapObservation,
    DepthPreprocessingStep,
    DepthProviderArtifact,
    DepthProviderEnvironment,
    DepthProviderIdentity,
    DepthProviderProvenance,
    DepthProviderTrainingDeclaration,
    depth_file_reference,
)
from calibrex.data.kitti import read_timestamps
from calibrex.data.kitti360_camera_lidar_problem import (
    read_kitti360_fisheye_intrinsics,
)
from calibrex.data.kitti_camera_lidar_problem import uniform_kitti_frame_ids

MIDAS_REPOSITORY = "https://github.com/isl-org/MiDaS"
MIDAS_COMMIT = "1645b7e1675301fdfac03640738fe5a6531e17d6"
MIDAS_TAG = "v3_1"
MODEL_TYPE = "dpt_beit_large_512"
CHECKPOINT_URL = (
    "https://github.com/isl-org/MiDaS/releases/download/v3_1/"
    "dpt_beit_large_512.pt"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate frozen KITTI-360 MiDaS maps and provenance"
    )
    parser.add_argument("sequence_path", type=Path)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--midas-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--camera-stream", default="image_03")
    parser.add_argument("--frame-count", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run MiDaS and write a schema-valid depth-provider artifact."""

    args = _parser().parse_args(argv)
    if args.frame_count < 2:
        raise ValueError("--frame-count must be at least two")
    sequence = args.sequence_path.resolve()
    calibration = args.calibration_root.resolve()
    repository = args.midas_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    output_directory = args.output_directory.resolve()
    artifact_output = args.artifact_output.resolve()
    _verify_repository(repository)
    checkpoint_digest = _required_digest(checkpoint)

    import torch
    import torch.nn.functional as functional

    sys.path.insert(0, str(repository))
    from midas.model_loader import load_model

    device = torch.device(args.device)
    model, transform, net_width, net_height = load_model(
        device,
        str(checkpoint),
        MODEL_TYPE,
        optimize=False,
        height=None,
        square=False,
    )
    image_directory = sequence / args.camera_stream / "data_rgb"
    timestamps = read_timestamps(sequence / args.camera_stream / "timestamps.txt")
    frame_ids = uniform_kitti_frame_ids(len(timestamps), args.frame_count)
    intrinsics = read_kitti360_fisheye_intrinsics(
        calibration,
        camera_stream=args.camera_stream,
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    observations = []
    input_references = [
        depth_file_reference(checkpoint, encoding="pytorch_checkpoint")
    ]
    output_references = []
    with torch.inference_mode():
        for position, frame_id in enumerate(frame_ids, start=1):
            frame_index = int(frame_id)
            image_path = image_directory / f"{frame_id}.png"
            image_reference = depth_file_reference(image_path, encoding="png_rgb8")
            input_references.append(image_reference)
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            source_height, source_width = rgb.shape[:2]
            transformed: dict[str, Any] = transform({"image": rgb})
            sample = torch.from_numpy(transformed["image"]).to(device).unsqueeze(0)
            prediction = model.forward(sample)
            prediction = functional.interpolate(
                prediction.unsqueeze(1),
                size=(source_height, source_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_array = prediction.cpu().numpy().astype(np.float32)
            depth_path = output_directory / f"{frame_id}.npy"
            np.save(depth_path, depth_array, allow_pickle=False)
            depth_reference = depth_file_reference(
                depth_path,
                encoding="npy_float32",
            )
            output_references.append(depth_reference)
            observations.append(
                DepthMapObservation(
                    frame_id=frame_id,
                    capture_time_ns=timestamps[frame_index].timestamp_ns,
                    source_frame=f"camera_{args.camera_stream[-1]}",
                    image=image_reference,
                    depth=depth_reference,
                    intrinsics=intrinsics,
                    image_transform=DepthImageTransform(
                        source_width=source_width,
                        source_height=source_height,
                        output_width=source_width,
                        output_height=source_height,
                        scale_x=1.0,
                        scale_y=1.0,
                        interpolation=(
                            "official aspect-preserving minimal resize to a "
                            "multiple of 32 then bicubic to source"
                        ),
                    ),
                    scale_convention="relative_depth",
                    invalid_values=[],
                    valid_fraction=float(np.isfinite(depth_array).mean()),
                )
            )
            print(f"[{position}/{len(frame_ids)}] {frame_id}", flush=True)

    artifact = DepthProviderArtifact(
        artifact_id=(
            f"midas-{MODEL_TYPE}-{sequence.name}-{args.camera_stream}-"
            f"uniform-{len(frame_ids)}"
        ),
        provider=DepthProviderIdentity(
            provider="MiDaS",
            model=MODEL_TYPE,
            version=MIDAS_TAG,
            source_repository=MIDAS_REPOSITORY,
            source_commit=MIDAS_COMMIT,
            license_spdx="MIT",
            checkpoint=depth_file_reference(
                checkpoint,
                encoding="pytorch_checkpoint",
            ),
            checkpoint_redistribution=(
                f"not redistributed; official release asset {CHECKPOINT_URL}"
            ),
            training=DepthProviderTrainingDeclaration(
                training_dataset="MiDaS 3.1 multi-dataset mixture",
                training_split="official model training mixture",
                evaluation_overlap="possible",
                test_data_used_for_online_refinement=False,
                evidence=(
                    "official checkpoint is used without online refinement; "
                    "KITTI-family training data prevents excluding overlap"
                ),
            ),
        ),
        environment=DepthProviderEnvironment(
            execution_mode="subprocess",
            command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            python_version=platform.python_version(),
            dependencies={
                "numpy": np.__version__,
                "pillow": Image.__version__,
                "torch": torch.__version__,
                "timm": (
                    "0.6.12 with Python 3.12 dataclass default_factory "
                    "compatibility patch"
                ),
            },
            device=str(device),
        ),
        scale_convention="relative_depth",
        preprocessing=[
            DepthPreprocessingStep(
                name="uniform_frame_selection",
                parameters={
                    "count": len(frame_ids),
                    "sequence_frame_count": len(timestamps),
                    "rounding": "nearest_integer_half_up",
                    "frame_ids": frame_ids,
                },
            ),
            DepthPreprocessingStep(
                name="midas_official_inference",
                parameters={
                    "model_type": MODEL_TYPE,
                    "nominal_encoder_width": net_width,
                    "nominal_encoder_height": net_height,
                    "keep_aspect_ratio": True,
                    "ensure_multiple_of": 32,
                    "resize_method": "minimal",
                    "input_interpolation": "opencv_cubic",
                    "output_interpolation": "torch_bicubic_align_corners_false",
                    "saved_feature": "raw_nonnegative_relative_inverse_depth",
                },
            ),
        ],
        observations=observations,
        warnings=[
            "training/evaluation sequence overlap cannot be excluded",
            "raw MiDaS relative inverse depth is retained for D2D histograms",
            "KITTI-360 official MEI intrinsics are used; the cited Borer "
            "experiment describes an unpublished double-sphere fit",
            f"checkpoint sha256={checkpoint_digest}",
        ],
        provenance=DepthProviderProvenance(
            generator="tools.run_midas_kitti360_provider",
            generator_version=__version__,
            git_commit=git_commit(),
            input_manifest_sha256=_reference_digest(input_references),
            output_manifest_sha256=_reference_digest(output_references),
        ),
    )
    artifact.save(artifact_output)
    print(f"artifact={artifact_output}", flush=True)
    return 0


def _verify_repository(repository: Path) -> None:
    if not (repository / ".git" / "HEAD").is_file():
        raise ValueError(f"MiDaS repository is not a Git checkout: {repository}")
    observed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != MIDAS_COMMIT:
        raise ValueError(
            f"MiDaS commit mismatch: expected {MIDAS_COMMIT}, got {observed}"
        )


def _reference_digest(references: list[Any]) -> str:
    digest = hashlib.sha256()
    for reference in sorted(references, key=lambda item: item.path):
        digest.update(reference.path.encode())
        digest.update(b"\0")
        digest.update(reference.sha256.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required file is missing or unreadable: {path}")
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
