"""Conditional1px cycle quality after the frozen predictive surface selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from tools.known_geometry_joint import Report
from tools.map_surface_errors import summarize
from tools.surface_cycle_inputs import prepare


def run(plan_path: Path, workspace: Path, output: Path) -> None:
    import cv2

    cv2.setNumThreads(1)
    if output.exists() or workspace.exists():
        raise ValueError("refuse overwrite")
    plan = json.loads(plan_path.read_text())
    for group in ["source_sha256", "input_sha256"]:
        for p, sha in plan[group].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("changed " + p)
    workspace.mkdir(parents=True)
    report = Report(
        scope="18 conditional cycle diagnostics on predictive surfaces; no fits",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    for f in plan["frames"]:
        depth, filtered = prepare(f["stereo"])
        np.testing.assert_array_equal(depth, np.load(f["depth"], allow_pickle=False))
        if f["old_filtered"]:
            np.testing.assert_array_equal(filtered, np.load(f["old_filtered"], allow_pickle=False))
        maps = np.load(f["map_path"], allow_pickle=False)
        mask = maps["mask"]
        passed = np.isfinite(filtered)[10:-10, 10:-10]
        rows = []
        for label, pop in [
            ("all_selected", mask),
            ("cycle_pass", mask & passed),
            ("cycle_rejected_or_missing", mask & ~passed),
        ]:
            rows.append(
                {
                    "population": label,
                    **summarize(pop, maps["supported"], maps["distance"], maps["angle"]),
                    "retention_of_domain": float(pop.mean()),
                }
            )
        artifact = workspace / (f["drive"] + "-" + f["frame"] + ".npy")
        np.save(artifact, filtered)
        report.cases.append(
            {
                "drive": f["drive"],
                "frame": f["frame"],
                "populations": rows,
                "archived_cycle_reproduced": bool(f["old_filtered"]),
                "filtered_depth_path": str(artifact),
                "filtered_depth_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        )
        output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Report.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["plan", "workspace", "output"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.plan, a.workspace, a.output)
