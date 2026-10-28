"""Run the existing native six-DoF Pandey solver on an official Koide bundle.

This is an optimizer-only development comparison: both consume Koide's integrated,
equalized image/cloud. It is not an independent end-to-end method comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation

from calibrex.core.geometry import SE3
from calibrex.data.kitti import read_png_luminance_array
from calibrex.solvers.pandey_mutual_information_solver import (
    MutualInformationCameraModel,
    MutualInformationObservation,
    PandeyMutualInformationOptions,
    PandeyMutualInformationSolver,
)


class SharedBundleRun(BaseModel):
    """Provenance and raw output; no independent accuracy claim."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str
    source_commit: str
    driver_sha256: str
    input_sha256: dict[str, str]
    options: dict[str, Any]
    initial_T_camera_lidar: list[list[float]]
    output_T_camera_lidar: list[list[float]] | None
    elapsed_seconds: float
    result: dict[str, Any]


def read_koide_ply(path: Path) -> np.ndarray:
    """Read the exact four-float binary layout emitted by official preprocessing."""
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line or len(header) > 30:
                raise ValueError("invalid or unsupported PLY header")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        properties = [v for v in header if v.startswith("property ")]
        if "format binary_little_endian 1.0" not in header or properties != [
            "property float " + v for v in ("x", "y", "z", "intensity")
        ]:
            raise ValueError("unsupported PLY layout")
        counts = [int(v.split()[-1]) for v in header if v.startswith("element vertex ")]
        if len(counts) != 1:
            raise ValueError("expected one vertex element")
        data = np.frombuffer(stream.read(), dtype="<f4")
        if data.size != counts[0] * 4:
            raise ValueError("PLY length mismatch")
        return data.reshape(-1, 4).astype(float)


def run(bundle: Path, seed_path: Path, output: Path) -> SharedBundleRun:
    """Use all shared points, the same supplied seed, and unchanged solver defaults."""
    calib_path = bundle / "calib-seeded.json"
    calib = json.loads(calib_path.read_text())
    seed = json.loads(seed_path.read_text())
    initial = np.array(seed["initial_T_camera_lidar"])
    transform = SE3(
        translation_m=tuple(initial[:3, 3]),
        rotation_quat_xyzw=tuple(Rotation.from_matrix(initial[:3, :3]).as_quat()),
    )
    native_seed = calib["results"]["init_T_lidar_camera"]
    koide_initial = np.eye(4)
    koide_initial[:3, :3] = Rotation.from_quat(native_seed[3:]).as_matrix()
    koide_initial[:3, 3] = native_seed[:3]
    np.testing.assert_allclose(initial, np.linalg.inv(koide_initial), atol=1e-6, rtol=0)
    observations = []
    files = [calib_path, seed_path]
    for name in calib["meta"]["bag_names"]:
        png, ply = bundle / (name + ".png"), bundle / (name + ".ply")
        image = read_png_luminance_array(png)
        if image is None:
            raise ValueError("unsupported PNG")
        points = read_koide_ply(ply)
        camera = MutualInformationCameraModel(
            width=image.shape[1],
            height=image.shape[0],
            **dict(zip(("fx", "fy", "cx", "cy"), calib["camera"]["intrinsics"], strict=True)),
        )
        observations.append(
            MutualInformationObservation(name, image, points[:, :3], points[:, 3], camera)
        )
        files.extend([png, ply])
    options = PandeyMutualInformationOptions(min_train_observations=1, holdout_ratio=0)
    started = time.perf_counter()
    result = PandeyMutualInformationSolver().solve(observations, transform, options)
    elapsed = time.perf_counter() - started
    matrix = None
    if result.transform_camera_lidar is not None:
        estimate = result.transform_camera_lidar
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(estimate.rotation_quat_xyzw).as_matrix()
        matrix[:3, 3] = estimate.translation_m
    report = SharedBundleRun(
        scope="spent-data shared-bundle optimizer comparison; no held-out evaluation",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        options=asdict(options),
        initial_T_camera_lidar=initial.tolist(),
        output_T_camera_lidar=None if matrix is None else matrix.tolist(),
        elapsed_seconds=elapsed,
        result=result.as_dict(),
    )
    output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(SharedBundleRun.model_json_schema(), indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("seed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.bundle, args.seed, args.output)
    print(result.result["status"], result.elapsed_seconds, flush=True)


if __name__ == "__main__":
    main()
