"""Standalone fixed-scale scientific maps of the full eighteen-frame diagnosis."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(report: Path, destination: Path) -> None:
    destination.mkdir(exist_ok=False)
    cases = json.loads(report.read_text())["cases"]
    for c in cases:
        a = np.load(c["map_path"])
        mask = a["mask"]
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.2), layout="constrained")
        axes[0].imshow(plt.imread(c["image"]))
        axes[0].set_title("Camera image")
        axes[1].imshow(a["boundary"], cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(np.where(mask, 1.0, np.nan), cmap="Greens", vmin=0, vmax=1, alpha=0.6)
        axes[1].set_title("Selected (green); boundary band (white)")
        for ax, key, title, maximum in [
            (axes[2], "distance", "Nearest LiDAR distance (m)", 2.0),
            (axes[3], "angle", "Unsigned normal disagreement (deg)", 90.0),
        ]:
            im = ax.imshow(a[key], cmap="magma", vmin=0, vmax=maximum)
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(c["drive"] + " / " + c["frame"] + " — offline reference-pose diagnosis")
        fig.savefig(destination / (c["drive"] + "-" + c["frame"] + ".png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    render(a.report, a.destination)
