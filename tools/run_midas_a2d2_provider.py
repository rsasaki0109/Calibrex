#!/usr/bin/env python3
"""Run pinned MiDaS v3.1 on the public A2D2 Camera-LiDAR pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from calibrex import __version__
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.depth import (
    DepthCameraIntrinsics,
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
        description="Generate frozen A2D2 MiDaS maps and provenance"
    )
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--midas-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run MiDaS and write an A2D2 depth-provider artifact."""

    args = _parser().parse_args(argv)
    data_directory = args.data_directory.resolve()
    calibration_path = args.calibration.resolve()
    repository = args.midas_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    output_directory = args.output_directory.resolve()
    artifact_output = args.artifact_output.resolve()
    _verify_repository(repository)
    checkpoint_digest = _required_digest(checkpoint)
    intrinsics = _read_intrinsics(calibration_path)
    images = sorted(data_directory.glob("*_camera_frontleft_*.png"))
    if len(images) < 2:
        raise ValueError("A2D2 provider requires at least two front-left images")

    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as functional  # type: ignore[import-not-found]

    sys.path.insert(0, str(repository))
    from midas.model_loader import load_model  # type: ignore[import-not-found]

    device = torch.device(args.device)
    model, transform, net_width, net_height = load_model(
        device,
        str(checkpoint),
        MODEL_TYPE,
        optimize=False,
        height=None,
        square=False,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    observations = []
    input_references = [
        depth_file_reference(checkpoint, encoding="pytorch_checkpoint"),
        depth_file_reference(calibration_path, encoding="json"),
    ]
    output_references = []
    with torch.inference_mode():
        for position, image_path in enumerate(images, start=1):
            frame_id = _frame_id(image_path)
            image_reference = depth_file_reference(
                image_path, encoding="png_rgb8"
            )
            input_references.append(image_reference)
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            source_height, source_width = rgb.shape[:2]
            transformed: dict[str, Any] = transform({"image": rgb})
            sample = (
                torch.from_numpy(transformed["image"]).to(device).unsqueeze(0)
            )
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
                depth_path, encoding="npy_float32"
            )
            output_references.append(depth_reference)
            observations.append(
                DepthMapObservation(
                    frame_id=frame_id,
                    capture_time_ns=int(frame_id),
                    source_frame="camera_frontleft",
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
            print(f"[{position}/{len(images)}] {frame_id}", flush=True)

    artifact = DepthProviderArtifact(
        artifact_id=f"midas-{MODEL_TYPE}-a2d2-frontleft-{len(images)}",
        provider=DepthProviderIdentity(
            provider="MiDaS",
            model=MODEL_TYPE,
            version=MIDAS_TAG,
            source_repository=MIDAS_REPOSITORY,
            source_commit=MIDAS_COMMIT,
            license_spdx="MIT",
            checkpoint=depth_file_reference(
                checkpoint, encoding="pytorch_checkpoint"
            ),
            checkpoint_redistribution=(
                f"not redistributed; official release asset {CHECKPOINT_URL}"
            ),
            training=DepthProviderTrainingDeclaration(
                training_dataset="MiDaS 3.1 multi-dataset mixture",
                training_split="official model training mixture",
                evaluation_overlap="unknown",
                test_data_used_for_online_refinement=False,
                evidence=(
                    "official checkpoint is used without online refinement; "
                    "A2D2 overlap is not documented"
                ),
            ),
        ),
        environment=DepthProviderEnvironment(
            execution_mode="subprocess",
            command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            python_version=platform.python_version(),
            dependencies={
                "numpy": np.__version__,
                "pillow": version("pillow"),
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
                name="a2d2_fixed_public_pair",
                parameters={
                    "camera": "front_left",
                    "frame_ids": [item.frame_id for item in observations],
                    "image_state": "official undistorted public PNG",
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
                    "output_interpolation": "torch_bicubic_align_corners_false",
                    "saved_feature": "raw_nonnegative_relative_inverse_depth",
                },
            ),
        ],
        observations=observations,
        warnings=[
            (
                "A2D2 NPZ points are already registered into the camera view; "
                "this is execution evidence, not independent extrinsic truth"
            ),
            "only two public frames are present in this fixed small sample",
            "MiDaS training overlap with A2D2 is unknown",
            f"checkpoint sha256={checkpoint_digest}",
        ],
        provenance=DepthProviderProvenance(
            generator="tools.run_midas_a2d2_provider",
            generator_version=__version__,
            git_commit=git_commit(),
            input_manifest_sha256=_reference_digest(input_references),
            output_manifest_sha256=_reference_digest(output_references),
        ),
    )
    artifact.save(artifact_output)
    print(f"artifact={artifact_output}", flush=True)
    return 0


def _read_intrinsics(path: Path) -> DepthCameraIntrinsics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera = payload["cameras"]["front_left"]
    matrix = camera["CamMatrix"]
    width, height = camera["Resolution"]
    return DepthCameraIntrinsics(
        width=int(width),
        height=int(height),
        fx=float(matrix[0][0]),
        fy=float(matrix[1][1]),
        cx=float(matrix[0][2]),
        cy=float(matrix[1][2]),
        projection="pinhole",
        distortion_model="none",
        distortion=[],
    )


def _frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


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
