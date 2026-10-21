"""Run inside Pyodide, on a frozen, pickle-free prepared-frame NPZ fixture.

This measures the objective only, not a complete calibration. The caller records
input, source, wheel and tool hashes, and retains failed browser attempts.
"""

from __future__ import annotations

import itertools
import json
import time
from typing import Any

import numpy as np

from calibrex.evaluation.camera_lidar_edge import EdgeCamera, EdgeObjective, PreparedFrame
from calibrex.solvers.browser_webgpu_edge_adapter import BrowserWebGpuEdgeObjective


async def measure(path: str, runtime: Any, repetitions: int = 3) -> dict[str, Any]:
    """Evaluate all 729 candidate poses, including CPU/GPU parity and full transfer cost."""

    with np.load(path, allow_pickle=False) as fixture:
        fixture_provenance = json.loads(str(fixture["provenance_json"]))
        camera = EdgeCamera(
            *fixture["intrinsics"].tolist(),
            distortion_model=str(fixture["distortion_model"]),
            distortion=tuple(fixture["distortion"].tolist()),
        )
        start = fixture["start"].copy()
        frames = [
            PreparedFrame(
                time_s=float(index),
                gap_s=0.0,
                edge=fixture[f"edge_{index}"].copy(),
                points=fixture[f"points_{index}"].copy(),
                weights=fixture[f"weights_{index}"].copy(),
                scale=int(fixture[f"scale_{index}"]),
                rings=0,
            )
            for index in range(int(fixture["frames"]))
        ]
    theta = np.asarray(list(itertools.product((-1.0, 0.0, 1.0), repeat=6)))
    theta[:, 3:] *= 0.05
    began = time.perf_counter()
    cpu = EdgeObjective(frames, camera, start)
    cpu_setup = time.perf_counter() - began
    began = time.perf_counter()
    cpu_first = np.asarray([cpu.score(value) for value in theta])
    cpu_first_s = time.perf_counter() - began
    began = time.perf_counter()
    gpu = await BrowserWebGpuEdgeObjective.create(frames, camera, start, runtime=runtime)
    gpu_setup = time.perf_counter() - began
    attempts: list[dict[str, Any]] = []
    try:
        began = time.perf_counter()
        gpu_first = await gpu.score_many(theta)
        gpu_first_s = time.perf_counter() - began
        # Thresholds fixed before the first browser measurement. These bounds
        # validate sampled objective values, never an optimizer's decisions.
        absolute_tolerance = 2e-6
        relative_tolerance = 2e-5

        def compare(values: np.ndarray) -> dict[str, Any]:
            error = np.abs(values - cpu_first)
            return {
                "parity": bool(
                    np.all(error <= absolute_tolerance + relative_tolerance * np.abs(cpu_first))
                ),
                "max_absolute_error": float(error.max()),
                "cpu_argmax": int(cpu_first.argmax()),
                "gpu_argmax": int(values.argmax()),
                "cpu_top_gap": float(np.diff(np.sort(cpu_first)[-2:])[0]),
            }

        first_comparison = compare(gpu_first)
        for repetition in range(repetitions):
            print(f"edge batch repetition {repetition + 1}/{repetitions}", flush=True)
            row: dict[str, Any] = {"repetition": repetition, "order": []}
            # Alternate order, retain every run. Both paths execute every pose.
            for backend in ("cpu", "gpu") if repetition % 2 == 0 else ("gpu", "cpu"):
                began = time.perf_counter()
                values = (
                    np.asarray([cpu.score(value) for value in theta])
                    if backend == "cpu"
                    else await gpu.score_many(theta)
                )
                row[backend + "_seconds"] = time.perf_counter() - began
                row[backend + "_comparison"] = compare(values)
                row["order"].append(backend)
            row["ratio"] = row["cpu_seconds"] / row["gpu_seconds"]
            attempts.append(row)
        return {
            "fixture_provenance": fixture_provenance,
            "status": "PASS"
            if first_comparison["parity"]
            and all(
                row[backend + "_comparison"]["parity"]
                for row in attempts
                for backend in ("cpu", "gpu")
            )
            else "FAIL",
            "frames": len(frames),
            "points": sum(len(f.points) for f in frames),
            "poses": len(theta),
            "image_pixels": sum(f.edge.size for f in frames),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "cpu_setup_seconds": cpu_setup,
            "gpu_setup_seconds": gpu_setup,
            "cpu_first_seconds": cpu_first_s,
            "gpu_first_seconds": gpu_first_s,
            "first_comparison": first_comparison,
            "attempts": attempts,
            "cold_batch_ratio": (cpu_setup + cpu_first_s) / (gpu_setup + gpu_first_s),
            "resident_median_ratio": float(np.median([row["ratio"] for row in attempts])),
            "allocated_gpu_bytes": int(gpu._session.allocatedBytes),
            "fallback_adapter": gpu._session.fallbackAdapter,
            "scope": (
                "objective batches only; no solver, file decoding, preparation or accuracy claim"
            ),
            "cpu_first_scores": cpu_first.tolist(),
            "gpu_first_scores": gpu_first.tolist(),
        }
    finally:
        gpu.close()
