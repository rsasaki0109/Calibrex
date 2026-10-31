"""Fixed image-support audit on acquired ±1 stereo neighbors; no metric claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from tools.audit_temporal_surface_inputs import image_pair
from tools.known_geometry_joint import Report

from calibrex.data.kitti import read_timestamps


def run(plan_path: Path, output: Path) -> None:
    import cv2

    if output.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    report = Report(
        scope="36 short-baseline directional-support pairs; no metric camera-motion validation",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    workspace = Path(plan["equalized_workspace"])
    workspace.mkdir(parents=True, exist_ok=False)
    for drive in plan["drives"]:
        timestamps = read_timestamps(Path(drive["camera_timestamps"]))
        for f in drive["frames"]:
            raw = cv2.imread(f["center_raw"], cv2.IMREAD_GRAYSCALE)
            old = cv2.imread(f["image"], cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(cv2.equalizeHist(raw), old)
            for neighbor in f["neighbors"]:
                neighbor_raw = cv2.imread(neighbor["path"], cv2.IMREAD_GRAYSCALE)
                equalized = workspace / (drive["drive"] + f"-{neighbor['index']:010d}.png")
                if not cv2.imwrite(str(equalized), cv2.equalizeHist(neighbor_raw)):
                    raise ValueError("failed to save equalized image")
                neighbor_path = str(equalized)
                first, second = (
                    (neighbor_path, f["image"])
                    if neighbor["index"] < f["index"]
                    else (f["image"], neighbor_path)
                )
                a, b = sorted([neighbor["index"], f["index"]])
                result = image_pair(Path(first), Path(second), drive["intrinsics"])
                result["directional_support_screen"] = (
                    result["cheirality_inliers"] >= 30 and min(result["cells"]) >= 6
                )
                report.cases.append(
                    {
                        "drive": drive["drive"],
                        "center": f["frame"],
                        "first_index": a,
                        "second_index": b,
                        "interval_s": (timestamps[b].timestamp_ns - timestamps[a].timestamp_ns)
                        * 1e-9,
                        "center_equalization_matches_archive": True,
                        "derived_neighbor_path": neighbor_path,
                        "derived_neighbor_sha256": hashlib.sha256(
                            equalized.read_bytes()
                        ).hexdigest(),
                        "image_support": result,
                        "independent_metric_motion_validated": False,
                    }
                )
                output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.plan, a.output)
