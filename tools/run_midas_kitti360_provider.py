#!/usr/bin/env python3
"""Run pinned MiDaS v3.1 inference for KITTI-360 outside the core."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
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
from calibrex.data.remote_archive_selection import (
    RemoteArchiveSelectionArtifact,
    load_remote_archive_selection,
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
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="schema-valid remote archive selection whose frame IDs must be used",
    )
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

    device = torch.device(args.device)
    (
        model,
        transform,
        net_width,
        net_height,
        ignored_checkpoint_keys,
        drop_path_alias_count,
        timm_version,
    ) = (
        _load_compatible_midas_model(torch, device, checkpoint)
    )
    image_directory = sequence / args.camera_stream / "data_rgb"
    timestamps = read_timestamps(sequence / args.camera_stream / "timestamps.txt")
    selection_manifest = (
        args.selection_manifest.resolve() if args.selection_manifest is not None else None
    )
    if selection_manifest is None:
        frame_ids = uniform_kitti_frame_ids(len(timestamps), args.frame_count)
        frame_selection_label = "uniform"
        frame_selection_step = DepthPreprocessingStep(
            name="uniform_frame_selection",
            parameters={
                "count": len(frame_ids),
                "sequence_frame_count": len(timestamps),
                "rounding": "nearest_integer_half_up",
                "frame_ids": frame_ids,
            },
        )
    else:
        frame_ids, selection = _frame_ids_from_selection(
            selection_manifest,
            sequence=sequence,
            camera_stream=args.camera_stream,
            frame_count=args.frame_count,
            timestamp_count=len(timestamps),
        )
        selection_digest = _required_digest(selection_manifest)
        frame_selection_label = "archive-selected"
        frame_selection_step = DepthPreprocessingStep(
            name="remote_archive_frame_selection",
            parameters={
                "count": len(frame_ids),
                "sequence_frame_count": len(timestamps),
                "frame_ids": frame_ids,
                "selection_manifest": str(selection_manifest),
                "selection_manifest_sha256": selection_digest,
                "selection_policy": selection.selection_policy,
            },
        )
    intrinsics = read_kitti360_fisheye_intrinsics(
        calibration,
        camera_stream=args.camera_stream,
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    observations = []
    input_references = [
        depth_file_reference(checkpoint, encoding="pytorch_checkpoint")
    ]
    if selection_manifest is not None:
        input_references.append(
            depth_file_reference(selection_manifest, encoding="yaml")
        )
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
            depth_array: NDArray[np.float32] = prediction.cpu().numpy().astype(
                np.float32
            )
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
            f"{frame_selection_label}-{len(frame_ids)}"
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
                "pillow": version("Pillow"),
                "torch": torch.__version__,
                "timm": timm_version,
                "checkpoint_loader": (
                    "strict state-dict load after filtering legacy "
                    "relative_position_index buffers"
                ),
            },
            device=str(device),
        ),
        scale_convention="relative_depth",
        preprocessing=[
            frame_selection_step,
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
            (
                "MiDaS checkpoint compatibility filtered "
                f"{len(ignored_checkpoint_keys)} legacy relative_position_index "
                "buffer keys; all remaining state-dict keys loaded strictly"
            ),
            (
                "MiDaS checkpoint compatibility aliased "
                f"drop_path1 to drop_path for {drop_path_alias_count} timm "
                "BEiT blocks"
            ),
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


def _frame_ids_from_selection(
    manifest_path: Path,
    *,
    sequence: Path,
    camera_stream: str,
    frame_count: int,
    timestamp_count: int,
) -> tuple[list[str], RemoteArchiveSelectionArtifact]:
    selection = load_remote_archive_selection(manifest_path)
    if selection.sequence_id != sequence.name:
        raise ValueError(
            "selection manifest sequence differs from sequence path: "
            f"{selection.sequence_id} != {sequence.name}"
        )
    if Path(selection.output_root).resolve() != sequence.resolve():
        raise ValueError("selection manifest output_root differs from sequence path")
    if len(selection.frame_ids) != frame_count:
        raise ValueError(
            f"selection has {len(selection.frame_ids)} frames, expected {frame_count}"
        )
    invalid_timestamp_ids = [
        frame_id for frame_id in selection.frame_ids if int(frame_id) >= timestamp_count
    ]
    if invalid_timestamp_ids:
        raise ValueError(
            "selection frame IDs exceed timestamp coverage: "
            f"{invalid_timestamp_ids}"
        )
    image_paths = {
        member.frame_id: Path(member.local_path).resolve()
        for member in selection.members
        if member.role == "image"
    }
    expected_paths = {
        frame_id: (sequence / camera_stream / "data_rgb" / f"{frame_id}.png").resolve()
        for frame_id in selection.frame_ids
    }
    if image_paths != expected_paths:
        raise ValueError(
            "selection image member paths do not match the requested camera stream"
        )
    return list(selection.frame_ids), selection


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


def _load_compatible_midas_model(
    torch_module: Any,
    device: Any,
    checkpoint: Path,
) -> tuple[Any, Any, int, int, list[str], int, str]:
    """Load the pinned DPT model across the current timm buffer layout.

    The official v3.1 checkpoint contains legacy ``relative_position_index``
    buffers that are not registered by the pinned MiDaS model when it is
    imported with the current runtime timm.  Only those known non-parameter
    buffers are filtered; every trainable/model state key must still match.
    The pinned MiDaS forward patch also refers to the older ``drop_path``
    name, so the current timm ``drop_path1`` module is aliased explicitly.
    """

    import cv2
    import timm
    from midas.dpt_depth import DPTDepthModel  # type: ignore[import-not-found]
    from midas.transforms import (  # type: ignore[import-not-found]
        NormalizeImage,
        PrepareForNet,
        Resize,
    )
    from torchvision.transforms import Compose  # type: ignore[import-untyped]

    model = DPTDepthModel(
        path=None,
        backbone="beitl16_512",
        non_negative=True,
    )
    parameters = torch_module.load(
        checkpoint,
        map_location=torch_module.device("cpu"),
    )
    if "optimizer" in parameters:
        parameters = parameters["model"]
    ignored_keys = sorted(
        key
        for key in parameters
        if key.endswith("relative_position_index")
    )
    filtered_parameters = {
        key: value for key, value in parameters.items() if key not in ignored_keys
    }
    incompatibilities = model.load_state_dict(
        filtered_parameters,
        strict=False,
    )
    if incompatibilities.missing_keys or incompatibilities.unexpected_keys:
        raise RuntimeError(
            "MiDaS checkpoint state-dict mismatch after compatibility filter: "
            f"missing={incompatibilities.missing_keys}, "
            f"unexpected={incompatibilities.unexpected_keys}"
        )

    drop_path_alias_count = 0
    for block in model.pretrained.model.blocks:
        if not hasattr(block, "drop_path"):
            if not hasattr(block, "drop_path1"):
                raise RuntimeError(
                    "MiDaS BEiT block has neither drop_path nor drop_path1"
                )
            block.drop_path = block.drop_path1
            drop_path_alias_count += 1

    transform = Compose(
        [
            Resize(
                512,
                512,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )
    model.eval()
    model.to(device)
    return (
        model,
        transform,
        512,
        512,
        ignored_keys,
        drop_path_alias_count,
        str(timm.__version__),
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
