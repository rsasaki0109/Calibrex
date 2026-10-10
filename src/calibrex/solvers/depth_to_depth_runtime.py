"""Lazy selection of optional D2D projection implementations."""

from __future__ import annotations

from typing import Literal

import numpy as np

from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthPairProjector,
    DepthToDepthObservation,
    project_depth_pairs,
)

ProjectionBackend = Literal["numpy", "numba_cpu", "cuda"]


def create_depth_pair_projector(
    backend: ProjectionBackend,
    observations: tuple[DepthToDepthObservation, ...] = (),
) -> tuple[DepthPairProjector, str]:
    """Prepare a projector and its reproducible identity; never fall back silently."""

    if backend == "numpy":
        return (
            project_depth_pairs,
            f"calibrex.numpy_depth_pair_projector/v0.2;numpy={np.__version__}",
        )
    if backend == "numba_cpu":
        try:
            from calibrex.solvers.numba_depth_to_depth_adapter import (
                numba_depth_pair_projector_identity,
                project_depth_pairs_numba,
            )
        except ImportError as exc:
            raise ValueError(
                "numba_cpu projection requires the optional calibrex[numba] dependency"
            ) from exc
        return project_depth_pairs_numba, numba_depth_pair_projector_identity()
    if backend == "cuda":
        try:
            from calibrex.solvers.cuda_depth_to_depth_adapter import CudaDepthPairProjector
        except ImportError as exc:
            raise ValueError(
                "cuda projection requires the optional calibrex[cuda] dependency"
            ) from exc
        projector = CudaDepthPairProjector(observations)
        return projector, projector.identity()
    raise ValueError(f"unsupported D2D projection backend: {backend}")
