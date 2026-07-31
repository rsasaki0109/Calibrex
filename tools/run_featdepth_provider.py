#!/usr/bin/env python3
"""Run pinned FeatDepth inference outside the Calibrex core environment.

This adapter deliberately imports PyTorch and the external FeatDepth checkout
only from this executable. It emits NumPy depth maps plus a schema-valid
Calibrex depth-provider artifact; no FeatDepth code or checkpoint is vendored.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
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
from calibrex.data.kitti import read_calibration_file, read_timestamps
from calibrex.data.kitti_camera_lidar_problem import uniform_kitti_frame_ids

FEATDEPTH_REPOSITORY = "https://github.com/sconlyshootery/FeatDepth"
FEATDEPTH_COMMIT = "550420b3fb51a027549716b74c6fbce41651d3a5"
FEATDEPTH_CHECKPOINT_FILE_ID = "1HlAubfuja5nBKpfNU3fQs-3m3Zaiu9RI"
INPUT_HEIGHT = 320
INPUT_WIDTH = 1024
MIN_DEPTH_M = 1.0e-3
MAX_DEPTH_M = 80.0
STEREO_SCALE = 36.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate frozen KITTI FeatDepth maps and provenance"
    )
    parser.add_argument("sequence_path", type=Path)
    parser.add_argument("--featdepth-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--camera-stream", default="image_02")
    parser.add_argument("--frame-count", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run FeatDepth and write a depth-provider artifact."""

    args = _parser().parse_args(argv)
    if args.frame_count < 2:
        raise ValueError("--frame-count must be at least two")
    sequence = args.sequence_path.resolve()
    repository = args.featdepth_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    output_directory = args.output_directory.resolve()
    artifact_output = args.artifact_output.resolve()
    _verify_repository(repository)
    checkpoint_digest = _required_digest(checkpoint)

    import torch
    import torch.nn.functional as functional

    sys.path.insert(0, str(repository))
    from mono.model.mono_fm.depth_decoder import DepthDecoder
    from mono.model.mono_fm.depth_encoder import DepthEncoder

    device = torch.device(args.device)
    encoder = DepthEncoder(50, None)
    decoder = DepthDecoder(encoder.num_ch_enc)
    payload: dict[str, Any] = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state = payload["state_dict"]
    encoder.load_state_dict(
        {
            key.removeprefix("DepthEncoder."): value
            for key, value in state.items()
            if key.startswith("DepthEncoder.")
        },
        strict=True,
    )
    decoder.load_state_dict(
        {
            key.removeprefix("DepthDecoder."): value
            for key, value in state.items()
            if key.startswith("DepthDecoder.")
        },
        strict=True,
    )
    del payload, state
    encoder.to(device).eval()
    decoder.to(device).eval()

    image_directory = sequence / args.camera_stream / "data"
    timestamps = read_timestamps(sequence / args.camera_stream / "timestamps.txt")
    frame_ids = uniform_kitti_frame_ids(len(timestamps), args.frame_count)
    calibration = read_calibration_file(sequence.parent / "calib_cam_to_cam.txt")
    suffix = _camera_suffix(args.camera_stream)
    projection = calibration.get(f"P_rect_{suffix}")
    if projection is None or len(projection) != 12:
        raise ValueError(f"missing 3x4 P_rect_{suffix} in calib_cam_to_cam.txt")
    output_directory.mkdir(parents=True, exist_ok=True)

    observations = []
    input_references = [depth_file_reference(checkpoint, encoding="pytorch_checkpoint")]
    output_references = []
    with torch.inference_mode():
        for position, frame_id in enumerate(frame_ids, start=1):
            frame_index = int(frame_id)
            image_path = image_directory / f"{frame_id}.png"
            image_reference = depth_file_reference(image_path, encoding="png_rgb8")
            input_references.append(image_reference)
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
            source_height, source_width = rgb.shape[:2]
            tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
            tensor = functional.interpolate(
                tensor,
                size=(INPUT_HEIGHT, INPUT_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
            disparity = decoder(encoder(tensor))[("disp", 0, 0)]
            disparity = functional.interpolate(
                disparity,
                size=(source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            depth_array = disparity.detach().cpu().numpy().astype(np.float32)
            depth_path = output_directory / f"{frame_id}.npy"
            np.save(depth_path, depth_array, allow_pickle=False)
            depth_reference = depth_file_reference(
                depth_path,
                encoding="npy_float32",
            )
            output_references.append(depth_reference)
            scale_x = source_width / source_width
            scale_y = source_height / source_height
            observations.append(
                DepthMapObservation(
                    frame_id=frame_id,
                    capture_time_ns=timestamps[frame_index].timestamp_ns,
                    source_frame=f"camera_{suffix[-1]}",
                    image=image_reference,
                    depth=depth_reference,
                    intrinsics=DepthCameraIntrinsics(
                        width=source_width,
                        height=source_height,
                        fx=float(projection[0]) * scale_x,
                        fy=float(projection[5]) * scale_y,
                        cx=float(projection[2]) * scale_x,
                        cy=float(projection[6]) * scale_y,
                    ),
                    image_transform=DepthImageTransform(
                        source_width=source_width,
                        source_height=source_height,
                        output_width=source_width,
                        output_height=source_height,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        interpolation="network resize 1024x320 then bilinear to source",
                    ),
                    scale_convention="disparity",
                    invalid_values=[],
                    valid_fraction=float(np.isfinite(depth_array).mean()),
                )
            )
            print(f"[{position}/{len(frame_ids)}] {frame_id}", flush=True)

    artifact = DepthProviderArtifact(
        artifact_id=(f"featdepth-{sequence.name}-{args.camera_stream}-uniform-{len(frame_ids)}"),
        provider=DepthProviderIdentity(
            provider="FeatDepth",
            model="FeatDepth trained on KITTI raw",
            version="official-2020-kitti-raw-checkpoint",
            source_repository=FEATDEPTH_REPOSITORY,
            source_commit=FEATDEPTH_COMMIT,
            license_spdx="MIT",
            checkpoint=depth_file_reference(
                checkpoint,
                encoding="pytorch_checkpoint",
            ),
            checkpoint_redistribution=(
                f"not redistributed; official Google Drive file {FEATDEPTH_CHECKPOINT_FILE_ID}"
            ),
            training=DepthProviderTrainingDeclaration(
                training_dataset="KITTI raw",
                training_split="repository-described KITTI raw training data",
                evaluation_overlap="possible",
                test_data_used_for_online_refinement=False,
                evidence=(
                    "official non-online-refined checkpoint selected; repository "
                    "does not publish enough training IDs to exclude drive overlap"
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
            },
            device=str(device),
        ),
        scale_convention="disparity",
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
                name="featdepth_official_inference",
                parameters={
                    "input_height": INPUT_HEIGHT,
                    "input_width": INPUT_WIDTH,
                    "resize_mode": "bilinear_align_corners_false",
                    "min_depth_m": MIN_DEPTH_M,
                    "max_depth_m": MAX_DEPTH_M,
                    "stereo_scale": STEREO_SCALE,
                    "saved_feature": "raw_network_inverse_depth_disparity",
                },
            ),
        ],
        observations=observations,
        warnings=[
            "training/evaluation drive overlap cannot be excluded",
            "raw FeatDepth inverse-depth/disparity is retained for D2D histograms",
            "metric depth conversion parameters are recorded but not applied",
            f"checkpoint sha256={checkpoint_digest}",
        ],
        provenance=DepthProviderProvenance(
            generator="tools.run_featdepth_provider",
            generator_version=__version__,
            git_commit=git_commit(),
            input_manifest_sha256=_reference_digest(input_references),
            output_manifest_sha256=_reference_digest(output_references),
        ),
    )
    artifact.save(artifact_output)
    print(f"artifact={artifact_output}", flush=True)
    return 0


def _camera_suffix(camera_stream: str) -> str:
    if (
        not camera_stream.startswith("image_")
        or len(camera_stream) != 8
        or not camera_stream[-2:].isdigit()
    ):
        raise ValueError("camera stream must use image_XX form")
    return camera_stream[-2:]


def _verify_repository(repository: Path) -> None:
    head_path = repository / ".git" / "HEAD"
    if not head_path.is_file():
        raise ValueError(f"FeatDepth repository is not a Git checkout: {repository}")
    import subprocess

    observed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != FEATDEPTH_COMMIT:
        raise ValueError(f"FeatDepth commit mismatch: expected {FEATDEPTH_COMMIT}, got {observed}")


def _reference_digest(references: list[Any]) -> str:
    digest = hashlib.sha256()
    for reference in sorted(references, key=lambda item: item.path):
        digest.update(reference.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(reference.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required file is missing or unreadable: {path}")
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
