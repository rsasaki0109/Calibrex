"""Lazy selection of optional D2D projection implementations."""

from __future__ import annotations

from typing import Final, Literal, Protocol, runtime_checkable

import numpy as np

from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthPairProjector,
    DepthToDepthObservation,
    DepthToDepthOptions,
    project_depth_pairs,
)

ProjectionBackend = Literal["numpy", "numba_cpu", "cuda"]
DEFAULT_CUDA_CACHE_BYTES: Final = 256 * 1024 * 1024


def validate_frame_workers(backend: ProjectionBackend, workers: int) -> None:
    """Validate explicit CPU frame concurrency before loading inputs."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("frame_workers must be a positive integer")
    if backend == "cuda" and workers != 1:
        raise ValueError("CUDA projection requires frame_workers=1")


def validate_cuda_cache_bytes(backend: ProjectionBackend, cache_bytes: int | None) -> None:
    """Reject invalid or inapplicable explicit CUDA cache settings."""

    if cache_bytes is not None:
        if cache_bytes < 0:
            raise ValueError("cuda_cache_bytes must be non-negative")
        if backend != "cuda":
            raise ValueError("CUDA cache settings require --projection-backend cuda")


@runtime_checkable
class DepthPairTransferStatistics(Protocol):
    """Optional observations of large input uploads, independent of CuPy imports."""

    def transfer_statistics(self) -> dict[str, int]:
        """Return a snapshot of cache capacity, residency and input transfers."""
        ...


def depth_pair_transfer_statistics(projector: DepthPairProjector) -> dict[str, int]:
    """Return accelerator counters when supported, otherwise an empty mapping."""

    if isinstance(projector, DepthPairTransferStatistics):
        return projector.transfer_statistics()
    return {}


def create_depth_pair_projector(
    backend: ProjectionBackend,
    observations: tuple[DepthToDepthObservation, ...] = (),
    *,
    cuda_cache_bytes: int | None = None,
    histogram_options: DepthToDepthOptions | None = None,
) -> tuple[DepthPairProjector, str]:
    """Prepare a projector and its reproducible identity; never fall back silently."""

    validate_cuda_cache_bytes(backend, cuda_cache_bytes)
    if backend == "numpy":
        return (
            project_depth_pairs,
            f"calibrex.numpy_depth_pair_projector/v0.2;numpy={np.__version__}",
        )
    if backend == "numba_cpu":
        try:
            from calibrex.solvers.numba_depth_to_depth_adapter import (
                NumbaDepthPairProjector,
            )
        except ImportError as exc:
            raise ValueError(
                "numba_cpu projection requires the optional calibrex[numba] dependency"
            ) from exc
        cpu_projector = NumbaDepthPairProjector(observations, histogram_options=histogram_options)
        return cpu_projector, cpu_projector.identity()
    if backend == "cuda":
        try:
            from calibrex.solvers.cuda_depth_to_depth_adapter import CudaDepthPairProjector
        except ImportError as exc:
            raise ValueError(
                "cuda projection requires the optional calibrex[cuda] dependency"
            ) from exc
        projector = CudaDepthPairProjector(
            observations,
            cache_bytes=DEFAULT_CUDA_CACHE_BYTES if cuda_cache_bytes is None else cuda_cache_bytes,
        )
        return projector, projector.identity()
    raise ValueError(f"unsupported D2D projection backend: {backend}")
