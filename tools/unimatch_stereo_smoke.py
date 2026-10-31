"""Pinned external UniMatch CPU smoke; no calibration/reference pose input."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tools.stereo_rerectification import rectification

from calibrex.data.kitti import read_calibration_file

COMMIT = "943a99006950ab6c84ea72a72beb147fdd03c19a"
CHECKPOINT_SHA256 = "2dd12e97b60bb7b91a0a45845a59260f9fdfe1ae9449dc491353c8842080ea6d"


def digest(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def padding(height: int, width: int) -> tuple[int, int, int, int]:
    """Official symmetric replicate-padding convention, without resizing."""
    dh, dw = -height % 32, -width % 32
    return dw // 2, dw - dw // 2, dh // 2, dh - dh // 2


def run(source: Path, checkpoint: Path, metadata: Path, output: Path) -> None:
    import cv2
    import torch
    import torch.nn.functional as functional

    if output.exists():
        raise ValueError("refuse to overwrite smoke evidence")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if revision != COMMIT or dirty or digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("external code/checkpoint differs from pinned model")
    sys.path.insert(0, str(source.resolve()))
    from unimatch.unimatch import UniMatch

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    meta = json.loads(metadata.read_text())
    paths = [Path(p) for p in meta["source_sha256"] if p.endswith(".png")]
    camera = next(Path(p) for p in meta["source_sha256"] if p.endswith("calib_cam_to_cam.txt"))
    if len(paths) != 2:
        raise ValueError("exactly two stereo images required")
    for path in [*paths, camera]:
        if digest(path) != meta["source_sha256"][str(path)]:
            raise ValueError("changed source input")
    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
    h, w = images[0].shape[:2]
    if images[0].shape != images[1].shape:
        raise ValueError("image sizes differ")
    calibration = read_calibration_file(camera)
    pl, pr = [np.array(calibration["P_rect_" + c]).reshape(3, 4) for c in ["02", "03"]]
    r1, r2, p1, p2 = rectification(pl, pr, (w, h))
    tensors = []
    pad = padding(h, w)
    for im, p, r, new in [(images[0], pl, r1, p1), (images[1], pr, r2, p2)]:
        x, y = cv2.initUndistortRectifyMap(
            p[:, :3], np.zeros(5), r, new[:, :3], (w, h), cv2.CV_32FC1
        )
        rgb = cv2.remap(im, x, y, cv2.INTER_LINEAR)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
        tensors.append(functional.pad(tensor, pad, mode="replicate"))
    model = UniMatch(num_scales=2, upsample_factor=4, reg_refine=True, task="stereo")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights["model"], strict=True)
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model(
            *tensors,
            attn_type="self_swin2d_cross_swin1d",
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=3,
            task="stereo",
        )["flow_preds"][-1]
    elapsed = time.perf_counter() - start
    disparity = prediction[0, pad[2] : pad[2] + h, pad[0] : pad[0] + w].numpy()
    if disparity.shape != (h, w) or not np.isfinite(disparity).all():
        raise ValueError("invalid disparity output")
    output.mkdir(parents=True)
    np.save(output / "disparity.npy", disparity)
    report = {
        "scope": "spent-data runtime smoke only; no calibration accuracy claim",
        "external_commit": revision,
        "checkpoint_sha256": digest(checkpoint),
        "adapter_sha256": digest(Path(__file__)),
        "metadata_sha256": digest(metadata),
        "input_sha256": {str(p): digest(p) for p in [*paths, camera]},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
        "input_shape": [h, w],
        "padded_shape": list(tensors[0].shape[-2:]),
        "padding_lrtb": list(pad),
        "resize": False,
        "color": "RGB [0,255]",
        "forward_seconds_cold": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "disparity_sha256": digest(output / "disparity.npy"),
        "disparity_quantiles": np.quantile(disparity, [0, 0.5, 0.9, 1]).tolist(),
        "finite_pixels": int(np.isfinite(disparity).sum()),
        "positive_pixels": int((disparity > 0).sum()),
        "rectification": {
            "r1": r1.tolist(),
            "r2": r2.tolist(),
            "p1": p1.tolist(),
            "p2": p2.tolist(),
        },
    }
    (output / "runtime.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "checkpoint", "metadata", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
