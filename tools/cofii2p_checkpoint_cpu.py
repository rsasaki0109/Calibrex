"""Digest-pinned CPU checkpoint runner and raw correspondence export; no pose scoring."""

from __future__ import annotations

import argparse
import hashlib
import random
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
from pydantic import Field
from tools.cofii2p_cpu_probe import UPSTREAM, prepare_source
from tools.cofii2p_input_boundary import FileRef, Prepared, Strict
from tools.cofii2p_model_bridge import (
    BridgeManifest,
    decode_fine_pixels,
    digest,
    fine_to_source_pixels,
)


class CheckpointRequest(Strict):
    schema_version: Literal["calibrex.cofii2p_checkpoint_cpu/v0.1"]
    source_checkout: str
    bridge_manifest: FileRef
    checkpoint: FileRef
    checkpoint_kind: Literal["declared_learned", "synthetic_fixture"]
    declared_checkpoint_source: str = Field(min_length=1)
    declared_training_provenance: str = Field(min_length=1)
    declared_terms: str = Field(min_length=1)
    image_policy: Literal["prepared_bgr", "prepared_bgr_to_rgb"]
    pixel_layout: Literal["row_major", "official_eval_legacy"]
    seed: int = Field(ge=0, le=2**32 - 1)
    threads: int = Field(default=2, ge=1, le=8)


class CheckpointResult(Strict):
    scope: Literal["CPU checkpoint execution and raw pairs; no accuracy or official parity claim"]
    source_commit: str
    upstream_commit: str
    driver: FileRef
    request: FileRef
    config: CheckpointRequest
    runtime: dict[str, str]
    bridge_artifact: FileRef
    prepared_manifest: FileRef
    prepared_artifact: FileRef
    patched_source_sha256: dict[str, str]
    status: Literal["completed", "failed"]
    checkpoint_loaded: bool
    error: str | None
    init_load_seconds: float
    forward_seconds: float
    peak_rss_kib: int
    loaded_state_sha256: str | None
    repeat_arrays_exact: bool | None
    pair_count: int
    raw_pairs: FileRef | None
    warnings: list[str]


def verify(ref: FileRef) -> Path:
    path = Path(ref.path)
    if digest(path) != ref.sha256:
        raise ValueError("artifact digest mismatch: " + str(path))
    return path


def load_strict_state(model, checkpoint: Path) -> str:
    """Load tensor-only raw state on CPU, with exact keys/shapes/dtypes and finiteness."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("checkpoint must be raw state_dict with exact model keys")
    signature = hashlib.sha256()
    for name, wanted in expected.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != wanted.shape
            or value.dtype != wanted.dtype
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError("checkpoint tensor shape/dtype/finiteness mismatch: " + name)
        signature.update(name.encode())
        signature.update(value.contiguous().numpy().tobytes())
    model.load_state_dict(state, strict=True)
    return signature.hexdigest()


def export_pairs(
    centers_xy: np.ndarray,
    patch_indices: np.ndarray,
    selected_coarse_rows: np.ndarray,
    bridge: dict[str, np.ndarray],
    *,
    layout: Literal["row_major", "official_eval_legacy"],
    frame_id: str,
) -> dict[str, np.ndarray]:
    """Restore source pixels and LiDAR centroids, preserving every contributor ID."""
    selected = np.asarray(selected_coarse_rows)
    if (
        selected.shape != (len(centers_xy),)
        or selected.dtype.kind not in "iu"
        or np.any(selected < 0)
        or np.any(selected >= len(bridge["voxel_index_4"]))
    ):
        raise ValueError("invalid selected coarse rows")
    fine = decode_fine_pixels(centers_xy, patch_indices, layout=layout)
    pixels = fine_to_source_pixels(fine, bridge["source_to_input_pixel"])
    voxel = bridge["voxel_index_4"][selected]
    offsets = bridge["voxel_member_offsets"]
    groups = [bridge["voxel_member_source_rows"][offsets[v] : offsets[v + 1]] for v in voxel]
    rows = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    return {
        "point_lidar_m": bridge["voxel_point_lidar_m"][voxel],
        "image_mean_px": pixels,
        "correspondence_id": np.array([f"{frame_id}:cofii-cpu:{i}" for i in range(len(voxel))]),
        "selected_coarse_row": selected,
        "voxel_index": voxel,
        "image_fine_px": fine,
        "patch_index": patch_indices,
        "source_member_offsets": np.r_[0, np.cumsum([len(g) for g in groups])],
        "source_member_point_id": bridge["source_point_id"][rows],
    }


def initialize_model(source: Path, output: Path, seed: int, threads: int):
    import torch

    if any(name == "model" or name.startswith("model.") for name in sys.modules):
        raise ValueError("run in a fresh Python process")
    _, patched = prepare_source(source, output / "source")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str((output / "source").resolve()))
    from model import network

    model = network.CoFiI2P(
        SimpleNamespace(img_H=160, img_W=512, img_fine_resolution_scale=32, norm="gn")
    ).eval()
    return model, network, patched


def run_checkpoint(
    request_path: Path, output: Path, *, allow_fixture: bool = False
) -> CheckpointResult:
    import torch

    request = CheckpointRequest.model_validate_json(request_path.read_text())
    if request.checkpoint_kind == "synthetic_fixture" and not allow_fixture:
        raise ValueError("synthetic fixture requires explicit --allow-fixture")
    checkpoint = verify(request.checkpoint)
    bridge_manifest = BridgeManifest.model_validate_json(
        verify(request.bridge_manifest).read_text()
    )
    bridge_path = verify(bridge_manifest.artifact)
    prepared = Prepared.model_validate_json(verify(bridge_manifest.prepared_manifest).read_text())
    verify(bridge_manifest.prepared_artifact)
    if prepared.artifact != bridge_manifest.prepared_artifact:
        raise ValueError("prepared provenance mismatch")
    with np.load(bridge_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with np.load(prepared.artifact.path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(arrays["image_bgr_chw"], archive["image_bgr_chw"])
        np.testing.assert_array_equal(
            arrays["source_to_input_pixel"], prepared.source_to_input_pixel
        )
    if bridge_manifest.options.stages != 5:
        raise ValueError("model requires five point stages")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    init_seconds = forward_seconds = 0.0
    patched, signature, error, repeat, pair_ref = {}, None, None, None, None
    loaded, count, status = False, 0, "failed"
    try:
        model, network, patched = initialize_model(
            Path(request.source_checkout), output, request.seed, request.threads
        )
        signature = load_strict_state(model, checkpoint)
        loaded = True
        init_seconds = time.perf_counter() - start
        pc = {
            name: [
                torch.from_numpy(arrays[f"{name}_{i}"])
                for i in range(5 if name in ["points", "neighbors"] else 4)
            ]
            for name in ["points", "neighbors", "subsampling", "upsampling"]
        }
        pc["feats"] = torch.from_numpy(arrays["feats"])
        pc["lengths"] = [len(p) for p in pc["points"]]
        image = arrays["image_bgr_chw"]
        if request.image_policy == "prepared_bgr_to_rgb":
            image = image[::-1].copy()
        image = torch.from_numpy(image).unsqueeze(0)
        selected = None
        original = network.fine_process

        def observe_selection(*args, **kwargs):
            nonlocal selected
            xy, rows = original(*args, **kwargs)
            selected = rows.detach().cpu().numpy().copy()
            return xy, rows

        network.fine_process = observe_selection
        start = time.perf_counter()
        with torch.inference_mode():
            values = model(pc, image, None, None, None, "test")
            first_selected = selected.copy()
            forward_seconds = time.perf_counter() - start
            again = model(pc, image, None, None, None, "test")
        for a, b in zip(values, again, strict=True):
            if a.device.type != "cpu" or not torch.isfinite(a).all() or not torch.equal(a, b):
                raise ValueError("nonfinite/non-CPU/nonrepeatable model output")
        np.testing.assert_array_equal(first_selected, selected)
        repeat = True
        np.testing.assert_array_equal(values[7].numpy(), arrays["points_4"][selected])
        similarity = torch.nn.functional.cosine_similarity(
            values[4], values[5].unsqueeze(-1), dim=1
        )
        patch_index = similarity.argmax(dim=1).numpy()
        pairs = export_pairs(
            values[6].numpy().T,
            patch_index,
            selected,
            arrays,
            layout=request.pixel_layout,
            frame_id=prepared.frame_id,
        )
        count = len(pairs["point_lidar_m"])
        for name in ["point_lidar_m", "image_mean_px"]:
            if not np.isfinite(pairs[name]).all():
                raise ValueError("nonfinite raw correspondence")
        path = output / "raw_pairs.npz"
        np.savez_compressed(path, **pairs)
        pair_ref = FileRef(path=str(path), sha256=digest(path))
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if not loaded:
            init_seconds = time.perf_counter() - start
        elif not forward_seconds:
            forward_seconds = time.perf_counter() - start
    result = CheckpointResult(
        scope="CPU checkpoint execution and raw pairs; no accuracy or official parity claim",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        upstream_commit=UPSTREAM,
        driver=FileRef(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__))),
        request=FileRef(path=str(request_path), sha256=digest(request_path)),
        config=request,
        runtime={"torch": torch.__version__, "numpy": np.__version__},
        bridge_artifact=bridge_manifest.artifact,
        prepared_manifest=bridge_manifest.prepared_manifest,
        prepared_artifact=bridge_manifest.prepared_artifact,
        patched_source_sha256=patched,
        status=status,
        checkpoint_loaded=loaded,
        error=error,
        init_load_seconds=init_seconds,
        forward_seconds=forward_seconds,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        loaded_state_sha256=signature,
        repeat_arrays_exact=repeat,
        pair_count=count,
        raw_pairs=pair_ref,
        warnings=[
            "Source/training/terms declarations do not verify checkpoint identity",
            "Local image policy/SciPy geometry not official preprocessing parity",
            "Raw pairs have no calibrated uncertainty, acceptance or pose accuracy verdict",
        ]
        + (
            ["SYNTHETIC FIXTURE: not learned weights or model accuracy evidence"]
            if request.checkpoint_kind == "synthetic_fixture"
            else []
        ),
    )
    (output / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    result = run_checkpoint(args.request, args.output, allow_fixture=args.allow_fixture)
    print(result.model_dump_json(indent=2))
    if result.status != "completed":
        raise SystemExit(2)
