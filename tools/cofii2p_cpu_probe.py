"""Bounded CPU forward probe on an isolated pinned checkout, with random weights."""

from __future__ import annotations

import argparse
import hashlib
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
from tools.cofii2p_input_boundary import FileRef, Strict
from tools.cofii2p_model_bridge import digest

UPSTREAM = "ed90edf51cca540ee76152d91133c7e79b63256c"


class ProbeResult(Strict):
    scope: Literal["random-weight CPU execution only; no checkpoint or accuracy/parity claim"]
    source_commit: str
    upstream_commit: str
    driver: FileRef
    bridge: FileRef
    patched_source_sha256: dict[str, str]
    original_source_sha256: dict[str, str]
    seed: int
    threads: int
    runtime: dict[str, str]
    status: Literal["completed", "failed"]
    error: str | None
    progress: dict
    init_seconds: float
    forward_seconds: float
    peak_rss_kib: int
    random_state_sha256: str | None
    output_shapes: dict[str, list[int]]
    outputs: FileRef | None
    repeat_arrays_exact: bool | None
    checkpoint_loaded: Literal[False]
    real_frames: Literal[0]


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("upstream patch anchor absent or ambiguous")
    return text.replace(old, new, 1)


def prepare_source(source: Path, target: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Patch only a new external copy; refuse a changed upstream revision/tree."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != UPSTREAM:
        raise ValueError("unreviewed upstream revision")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=source, check=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    paths = ["model/network.py", "model/kpconv/kernel_points.py"]
    original = {p: digest(target / p) for p in paths}
    path = target / paths[0]
    text = replace_once(path.read_text(), "import open3d as o3d\n", "")
    lines, count, in_fine = [], 0, False
    for line in text.splitlines(keepends=True):
        if line.startswith("def fine_process("):
            in_fine = True
        if ".cuda()" in line and not line.lstrip().startswith("#"):
            device = "coarse_img_feature.device" if in_fine else "img.device"
            count += line.count(".cuda()")
            line = line.replace(".cuda()", f".to({device})")
        lines.append(line)
    if count != 4:
        raise ValueError("unexpected CUDA call count")
    text = "".join(lines)
    text = replace_once(
        text,
        "                thrs -= 0.02\n",
        "                thrs -= 0.02\n"
        "                if thrs < -0.02 and pc_inline_index.numel() < 4:\n"
        "                    raise RuntimeError('bounded probe: fewer than four pairs')\n",
    )
    path.write_text(text)
    path = target / paths[1]
    text = replace_once(path.read_text(), "import open3d as o3d\n", "")
    text = replace_once(text, "import matplotlib.pyplot as plt\n", "")
    text = replace_once(text, "if not exists(kernel_file):", "if not exists(kernel_file + '.npy'):")
    text = replace_once(
        text,
        "        pcd = o3d.geometry.PointCloud()\n"
        "        pcd.points = o3d.utility.Vector3dVector(kernel_points)\n"
        "        o3d.io.write_point_cloud(kernel_file, pcd)\n",
        "        np.save(kernel_file + '.npy', kernel_points)\n",
    )
    text = replace_once(
        text,
        "        pcd = o3d.io.read_point_cloud(kernel_file)\n"
        "        kernel_points = np.array(pcd.points).astype(np.float32)\n",
        "        kernel_points = np.load(kernel_file + '.npy', "
        "allow_pickle=False).astype(np.float32)\n",
    )
    path.write_text(text)
    return original, {p: digest(target / p) for p in paths}


def probe(source: Path, bridge: Path, output: Path, seed: int, threads: int) -> ProbeResult:
    """Run the real test-mode network on synthetic geometry; never load a checkpoint."""
    import einops
    import torch

    if not 0 <= seed < 2**32 or not 1 <= threads <= 8:
        raise ValueError("seed/threads outside bounded probe range")
    if any(name == "model" or name.startswith("model.") for name in sys.modules):
        raise ValueError("run probe in a fresh Python process")
    output.mkdir(parents=True, exist_ok=False)
    original, patched = prepare_source(source, output / "source")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str((output / "source").resolve()))
    progress = {}
    init_seconds = forward_seconds = 0.0
    state_sha = error = artifact = repeat = None
    shapes = {}
    status = "failed"
    try:
        from model.network import CoFiI2P

        with np.load(bridge, allow_pickle=False) as data:
            pc = {
                name: [
                    torch.from_numpy(data[f"{name}_{i}"])
                    for i in range(5 if name in ["points", "neighbors"] else 4)
                ]
                for name in ["points", "neighbors", "subsampling", "upsampling"]
            }
            pc["feats"] = torch.from_numpy(data["feats"])
            pc["lengths"] = [len(p) for p in pc["points"]]
        image = torch.rand(1, 3, 160, 512)
        progress["synthetic_image_sha256"] = hashlib.sha256(image.numpy().tobytes()).hexdigest()
        progress["point_counts"] = pc["lengths"]
        start = time.perf_counter()
        net = CoFiI2P(
            SimpleNamespace(img_H=160, img_W=512, img_fine_resolution_scale=32, norm="gn")
        ).eval()
        init_seconds = time.perf_counter() - start
        state = hashlib.sha256()
        for name, value in net.state_dict().items():
            state.update(name.encode())
            state.update(value.detach().cpu().contiguous().numpy().tobytes())
        state_sha = state.hexdigest()

        def score_hook(module, args, value):
            progress["coarse_point_score"] = {
                "shape": list(value.shape),
                "finite": bool(torch.isfinite(value).all()),
                "device": str(value.device),
            }

        net.pc_score_layer.register_forward_hook(score_hook)
        start = time.perf_counter()
        with torch.inference_mode():
            values = net(pc, image, None, None, None, "test")
            forward_seconds = time.perf_counter() - start
            again = net(pc, image, None, None, None, "test")
        arrays = {}
        for i, (a, b) in enumerate(zip(values, again, strict=True)):
            if a.device.type != "cpu" or not torch.isfinite(a).all():
                raise ValueError("nonfinite or non-CPU model output")
            if not torch.equal(a, b):
                raise ValueError("repeat forward not exact")
            arrays[f"output_{i}"] = a.numpy()
        repeat = True
        shapes = {k: list(v.shape) for k, v in arrays.items()}
        path = output / "outputs.npz"
        np.savez_compressed(path, **arrays)
        artifact = FileRef(path=str(path), sha256=digest(path))
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if "start" in locals() and forward_seconds == 0:
            forward_seconds = time.perf_counter() - start
    result = ProbeResult(
        scope="random-weight CPU execution only; no checkpoint or accuracy/parity claim",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        upstream_commit=UPSTREAM,
        driver=FileRef(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__))),
        bridge=FileRef(path=str(bridge), sha256=digest(bridge)),
        patched_source_sha256=patched,
        original_source_sha256=original,
        seed=seed,
        threads=threads,
        runtime={"torch": torch.__version__, "numpy": np.__version__, "einops": einops.__version__},
        status=status,
        error=error,
        progress=progress,
        init_seconds=init_seconds,
        forward_seconds=forward_seconds,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        random_state_sha256=state_sha,
        output_shapes=shapes,
        outputs=artifact,
        repeat_arrays_exact=repeat,
        checkpoint_loaded=False,
        real_frames=0,
    )
    (output / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("bridge", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(
        probe(args.source, args.bridge, args.output, args.seed, args.threads).model_dump_json(
            indent=2
        )
    )
