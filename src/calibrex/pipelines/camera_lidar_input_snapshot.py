"""Caller-owned, verified input snapshots for repeated six-DoF estimation."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from calibrex.core.provenance import sha256_path
from calibrex.evaluation.borer_rotation_benchmark import (
    LoadedCameraLidarProblem,
    load_borer_problem,
)
from calibrex.solvers.borer_depth_to_depth_solver import FloatArray

INPUT_SNAPSHOT_VERSION = "calibrex.camera_lidar_input_snapshot/v0.1"


class CameraLidarSixDofInputSnapshot:
    """Verify once and retain immutable arrays until the caller releases this object.

    A snapshot deliberately keeps the original input even if external files
    subsequently change or disappear. Each estimate checks the problem artifact
    by SHA-256 again, but does not re-read its provider, checkpoint, images or
    arrays. Create a new snapshot to use changed external data. No global or
    persistent cache is involved; backends and solver state are never retained.
    """

    def __init__(self, problem_path: str | Path) -> None:
        started = time.perf_counter()
        self._path = Path(problem_path).resolve()
        loaded = load_borer_problem(self._path)
        observations = []
        for item in loaded.observations:
            frozen = replace(
                item,
                depth_map=_immutable_array(item.depth_map),
                lidar_points=_immutable_array(item.lidar_points),
            )
            # Preserve the already verified range values exactly, including
            # their floating-point rounding; all three arrays are immutable.
            object.__setattr__(frozen, "lidar_range_m", _immutable_array(item.lidar_range_m))
            observations.append(frozen)
        self._loaded = replace(loaded, observations=tuple(observations))
        self._preparation_seconds = time.perf_counter() - started
        self._array_bytes = sum(
            item.depth_map.nbytes + item.lidar_points.nbytes + item.lidar_range_m.nbytes
            for item in observations
        )
        self._digest = hashlib.sha256(
            f"{INPUT_SNAPSHOT_VERSION}\n{loaded.problem_sha256}\n"
            f"{loaded.depth_provider_sha256}".encode()
        ).hexdigest()

    @property
    def preparation_seconds(self) -> float:
        """One-time verification, loading and immutable capture duration."""

        return self._preparation_seconds

    def provenance(self) -> dict[str, Any]:
        """Return an independent record of snapshot scope and preparation cost."""

        return {
            "version": INPUT_SNAPSHOT_VERSION,
            "mode": "verified_snapshot",
            "snapshot_sha256": self._digest,
            "problem_sha256": self._loaded.problem_sha256,
            "depth_provider_sha256": self._loaded.depth_provider_sha256,
            "preparation_seconds": self._preparation_seconds,
            "array_bytes": self._array_bytes,
            "problem_artifact_reverified": True,
            "external_files_reverified": False,
            "preparation_in_run_timing": False,
        }

    def _loaded_for_estimate(self, problem_path: str | Path) -> LoadedCameraLidarProblem:
        path = Path(problem_path).resolve()
        if path != self._path:
            raise ValueError("input snapshot is bound to a different problem path")
        if sha256_path(path) != self._loaded.problem_sha256:
            raise ValueError("input snapshot problem digest mismatch; create a new snapshot")
        # Mutable artifact metadata belongs to each invocation. Arrays can be
        # shared safely even with caller-owned concurrent CPU estimates.
        return replace(
            self._loaded,
            problem=self._loaded.problem.model_copy(deep=True),
            depth_provider=self._loaded.depth_provider.model_copy(deep=True),
        )


def _immutable_array(array: FloatArray) -> FloatArray:
    # A bytes owner prevents setflags(write=True), unlike a read-only owning
    # ndarray. Preserve C-order values without sharing writable source storage.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
