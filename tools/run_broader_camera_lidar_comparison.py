"""Compare existing six-DoF depth and mutual-information models on fixed spent drives."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tools.run_camera_lidar_development_matrix import (
    DevelopmentMatrixPlan,
    MatrixResult,
    digest,
    pose_metrics,
)


def run(plan_path: Path, source: Path, output: Path) -> None:
    plan = DevelopmentMatrixPlan.model_validate_json(plan_path.read_text())
    for path, sha in plan.source_sha256.items():
        if digest(Path(path)) != sha:
            raise ValueError(f"frozen source changed: {path}")
    if plan.expected_cases != len(plan.drives) * len(plan.perturbations):
        raise ValueError("case inventory mismatch")
    output.mkdir(parents=True, exist_ok=False)
    records = [
        {"case_id": d.id + "-" + p.id, "drive": d.id, "seed": p.id, "methods": {}}
        for d in plan.drives
        for p in plan.perturbations
    ]
    report = MatrixResult(
        plan_sha256=digest(plan_path), driver_sha256=digest(Path(__file__)), records=records
    )

    def save() -> None:
        report.records = records
        (output / "matrix.json").write_text(report.model_dump_json(indent=2) + "\n")
        (output / "matrix.schema.json").write_text(
            json.dumps(MatrixResult.model_json_schema(), indent=2) + "\n"
        )

    save()
    try:
        for record in records:
            drive = next(d for d in plan.drives if d.id == record["drive"])
            case = output / record["case_id"]
            case.mkdir()
            names = []
            mapping = {}
            camera = None
            seed = None
            for frame in drive.frames:
                shared = source / drive.id / f"frame-{frame}"
                meta = json.loads((shared / "frontend.json").read_text())
                config = json.loads((shared / "calib.json").read_text())
                if camera is not None and camera != config["camera"]:
                    raise ValueError("inconsistent camera model")
                camera = config["camera"]
                name = f"frame-{frame}"
                names.append(name)
                for filename, suffix in [
                    ("aligned.png", ".png"),
                    ("aligned.ply", ".ply"),
                    ("depth.npy", ".depth.npy"),
                ]:
                    src = shared / filename
                    if digest(src) != meta["output_sha256"][filename]:
                        raise ValueError("changed prepared input")
                    dest = case / (name + suffix)
                    shutil.copyfile(src, dest)
                    mapping[dest.name] = {"source": str(src), "sha256": digest(dest)}
                current = json.loads(
                    (source / f"{drive.id}-{frame:010d}-{record['seed']}" / "seed.json").read_text()
                )
                if seed is not None and current != seed:
                    raise ValueError("inconsistent seed")
                seed = current
            config = {"camera": camera, "meta": {"bag_names": names}}
            (case / "calib.json").write_text(json.dumps(config, indent=2) + "\n")
            (case / "frontend.json").write_text(json.dumps(mapping, indent=2) + "\n")
            (case / "seed.json").write_text(json.dumps(seed, indent=2) + "\n")
            initial = np.array(seed["initial_T_camera_lidar"])
            inverse = np.linalg.inv(initial)
            config["results"] = {
                "init_T_lidar_camera": [
                    *inverse[:3, 3],
                    *Rotation.from_matrix(inverse[:3, :3]).as_quat(),
                ]
            }
            (case / "calib-seeded.json").write_text(json.dumps(config, indent=2) + "\n")
            for method, tool in [
                ("d2d", "tools/run_joint_stereo_d2d.py"),
                ("pandey", "tools/koide_shared_bundle_pandey.py"),
            ]:
                destination = case / (method + ".json")
                command = [
                    "taskset",
                    "-c",
                    ",".join(map(str, sorted(os.sched_getaffinity(0))[:4])),
                    sys.executable,
                    tool,
                    str(case),
                    str(case / "seed.json"),
                    str(destination),
                ]
                with (case / (method + ".log")).open("w") as log:
                    try:
                        code = subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "4"},
                            timeout=plan.timeout_seconds,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                # Only scoring uses the reference after the solver subprocess.
                reference = np.array(seed["reference_T_camera_lidar"])
                fit = json.loads(destination.read_text()) if code == 0 else None
                matrix = (
                    np.array(fit["output_T_camera_lidar"])
                    if fit and fit["output_T_camera_lidar"] is not None
                    else None
                )
                status = fit["result"]["status"] if fit else "execution_failed"
                record["initial"] = pose_metrics(initial, reference, True)
                record["methods"][method] = {
                    "returncode": code,
                    "status": status,
                    "metrics": pose_metrics(matrix, reference, code == 0 and status == "converged"),
                    "raw_metrics": pose_metrics(matrix, reference, code == 0),
                    "reference_file_sha256": digest(case / "seed.json"),
                }
                save()
                print(
                    record["case_id"],
                    method,
                    status,
                    record["methods"][method]["metrics"],
                    flush=True,
                )
    finally:
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ["plan", "source", "output"]:
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    run(a.plan, a.source, a.output)
