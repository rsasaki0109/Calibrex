#!/usr/bin/env python3
"""Run pinned FeatDepth inference outside the Calibrex core environment.

This adapter deliberately imports PyTorch and the external FeatDepth checkout
only from this executable. It emits NumPy depth maps plus a schema-valid
Calibrex depth-provider artifact; no FeatDepth code or checkpoint is vendored.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import sys
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from calibrex import __version__
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.depth import (
    DepthCameraIntrinsics,
    DepthFileReference,
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
FEATDEPTH_ADAPTER_VERSION = "calibrex.featdepth_provider/v0.2"


@dataclass(frozen=True)
class _PreparedImage:
    frame_id: str
    rgb: Any
    reference: DepthFileReference


def _read_image(image_directory: Path, frame_id: str) -> _PreparedImage:
    path = image_directory / f"{frame_id}.png"
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    reference = DepthFileReference(
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        encoding="png_rgb8",
    )
    return _PreparedImage(frame_id, rgb, reference)


def _load_checkpoint(checkpoint: Path, torch: Any) -> tuple[dict[str, Any], DepthFileReference]:
    # Bind the digest to the bytes actually decoded. Reuse this reference in
    # both manifests rather than hashing a large checkpoint three times.
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source, io.BytesIO() as data:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            data.write(chunk)
        reference = DepthFileReference(
            path=str(checkpoint),
            sha256=digest.hexdigest(),
            size_bytes=data.tell(),
            encoding="pytorch_checkpoint",
        )
        data.seek(0)
        payload = torch.load(data, map_location="cpu", weights_only=False)
    return payload, reference


def _prepared_images(
    directory: Path,
    frame_ids: list[str],
    *,
    workers: int,
    batch_size: int,
) -> Iterator[_PreparedImage]:
    if not workers:
        for frame_id in frame_ids:
            yield _read_image(directory, frame_id)
        return
    # Executor.map queues the entire sequence on Python 3.10-3.13. Keep only
    # this explicit window plus the caller's current batch, independent of N.
    pending_ids = iter(frame_ids)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="featdepth-image") as pool:
        pending = deque(
            pool.submit(_read_image, directory, frame_id)
            for frame_id in islice(pending_ids, batch_size + 2 * workers)
        )
        while pending:
            image = pending.popleft().result()
            next_id = next(pending_ids, None)
            if next_id is not None:
                pending.append(pool.submit(_read_image, directory, next_id))
            yield image


def _predict_batch(
    images: list[_PreparedImage],
    encoder: Any,
    decoder: Any,
    torch: Any,
    functional: Any,
    device: Any,
) -> list[Any]:
    tensors = []
    for image in images:
        tensor = torch.from_numpy(image.rgb).to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
        tensors.append(
            functional.interpolate(
                tensor,
                size=(INPUT_HEIGHT, INPUT_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
        )
    tensor = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
    disparities = decoder(encoder(tensor))[("disp", 0, 0)]
    disparities = disparities.float()
    arrays = []
    for index, image in enumerate(images):
        # Source sizes may differ; return each prediction to its own grid.
        disparity = functional.interpolate(
            disparities[index : index + 1],
            size=image.rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        arrays.append(disparity.detach().float().cpu().numpy().astype(np.float32))
    return arrays


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--preprocess-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--timing-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run FeatDepth and write a depth-provider artifact."""

    started = time.perf_counter()
    arguments = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(arguments)
    if args.frame_count < 2:
        raise ValueError("--frame-count must be at least two")
    if args.batch_size < 1 or args.preprocess_workers < 0:
        raise ValueError("--batch-size must be positive and --preprocess-workers non-negative")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    sequence = args.sequence_path.resolve()
    repository = args.featdepth_repository.resolve()
    checkpoint = args.checkpoint.resolve()
    output_directory = args.output_directory.resolve()
    artifact_output = args.artifact_output.resolve()
    if args.timing_output is not None and args.timing_output.resolve() == artifact_output:
        raise ValueError("--timing-output must be different from --artifact-output")
    import torch
    import torch.nn.functional as functional

    device = torch.device(args.device)
    if args.cpu_threads is not None:
        if device.type != "cpu":
            raise ValueError("--cpu-threads applies only to --device cpu")
        torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requires a CUDA-enabled PyTorch and NVIDIA GPU")
        torch.cuda.reset_peak_memory_stats(device)
    _verify_repository(repository)
    sys.path.insert(0, str(repository))
    from mono.model.mono_fm.depth_decoder import DepthDecoder
    from mono.model.mono_fm.depth_encoder import DepthEncoder

    encoder = DepthEncoder(50, None)
    decoder = DepthDecoder(encoder.num_ch_enc)
    payload, checkpoint_reference = _load_checkpoint(checkpoint, torch)
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
    input_references = [checkpoint_reference]
    output_references = []
    prepared_at = time.perf_counter()
    prediction_seconds = 0.0
    batch_count = position = 0
    with (
        torch.inference_mode(),
        closing(
            _prepared_images(
                image_directory,
                frame_ids,
                workers=args.preprocess_workers,
                batch_size=args.batch_size,
            )
        ) as images,
    ):
        while batch := list(islice(images, args.batch_size)):
            prediction_started = time.perf_counter()
            predictions = _predict_batch(
                batch,
                encoder,
                decoder,
                torch,
                functional,
                device,
            )
            prediction_seconds += time.perf_counter() - prediction_started
            batch_count += 1
            for image, depth_array in zip(batch, predictions, strict=True):
                if not np.isfinite(depth_array).all():
                    raise ValueError(f"non-finite FeatDepth prediction for frame {image.frame_id}")
                position += 1
                frame_id = image.frame_id
                frame_index = int(frame_id)
                image_reference = image.reference
                input_references.append(image_reference)
                source_height, source_width = image.rgb.shape[:2]
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

    predicted_at = time.perf_counter()
    execution = {
        "adapter_version": FEATDEPTH_ADAPTER_VERSION,
        "adapter_sha256": _required_digest(Path(__file__)),
        "batch_size": args.batch_size,
        "batch_count": batch_count,
        "preprocess_workers": args.preprocess_workers,
        "precision": "float32",
        "torch_cpu_threads": torch.get_num_threads(),
        "image_queue_limit": args.batch_size + 2 * args.preprocess_workers
        if args.preprocess_workers
        else 0,
        "timing_seconds": {
            "preparation": prepared_at - started,
            "inference_and_export": predicted_at - prepared_at,
            "prediction_batches": prediction_seconds,
        },
        "prediction_timing_scope": (
            "tensor upload/resize, model, output resize/download; subset of inference_and_export"
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if device.type == "cuda":
        execution.update(
            {
                "gpu": torch.cuda.get_device_name(device),
                "torch_cuda": torch.version.cuda,
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )

    artifact = DepthProviderArtifact(
        artifact_id=(f"featdepth-{sequence.name}-{args.camera_stream}-uniform-{len(frame_ids)}"),
        provider=DepthProviderIdentity(
            provider="FeatDepth",
            model="FeatDepth trained on KITTI raw",
            version="official-2020-kitti-raw-checkpoint",
            source_repository=FEATDEPTH_REPOSITORY,
            source_commit=FEATDEPTH_COMMIT,
            license_spdx="MIT",
            checkpoint=checkpoint_reference,
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
            command=[sys.executable, str(Path(__file__).resolve()), *arguments],
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
                    "execution": execution,
                },
            ),
        ],
        observations=observations,
        warnings=[
            "training/evaluation drive overlap cannot be excluded",
            "raw FeatDepth inverse-depth/disparity is retained for D2D histograms",
            "metric depth conversion parameters are recorded but not applied",
            f"checkpoint sha256={checkpoint_reference.sha256}",
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
    saved_at = time.perf_counter()
    if args.timing_output is not None:
        args.timing_output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generator": FEATDEPTH_ADAPTER_VERSION,
            "source_commit": git_commit(),
            "adapter_sha256": execution["adapter_sha256"],
            "provider_sha256": _required_digest(artifact_output),
            "command": artifact.environment.command,
            "execution": execution,
            "artifact_save_seconds": saved_at - predicted_at,
            "run_seconds": saved_at - started,
            "timing_scope": (
                "from main entry through provider save; "
                "excludes interpreter/import startup and timing report write"
            ),
        }
        timing_artifact = artifact.model_copy(deep=True)
        timing_artifact.artifact_id += "-timing"
        timing_artifact.preprocessing.append(
            DepthPreprocessingStep(name="provider_execution_timing", parameters=report)
        )
        timing_artifact.save(args.timing_output)
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
