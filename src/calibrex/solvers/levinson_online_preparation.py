"""Ordered CPU input preparation overlapping optional CUDA image initialization."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from calibrex.solvers.levinson_thrun_online_solver import (
    FloatArray,
    LevinsonCameraModel,
    LevinsonOnlineFrame,
    OnlineEdgeBackend,
    _validate_edge_parameters,
    levinson_edge_response,
)

if TYPE_CHECKING:
    from calibrex.solvers.cuda_online_edge_adapter import (
        CudaEdgeImageRuntime as _CudaEdgeImageRuntime,
    )

_Item = TypeVar("_Item")


@dataclass(frozen=True)
class LevinsonOnlineImage:
    """A loader's raw luminance and synchronized LiDAR discontinuities."""

    frame_id: str
    luminance: FloatArray
    lidar_points: FloatArray
    discontinuity_weights: FloatArray
    camera: LevinsonCameraModel


def prepare_levinson_online_frames(
    items: Sequence[_Item],
    loader: Callable[[_Item], LevinsonOnlineImage],
    *,
    workers: int = 1,
    evaluation_backend: OnlineEdgeBackend = "numpy",
    alpha: float = 1.0 / 3.0,
    gamma: float = 0.98,
    max_radius: int = 64,
) -> list[LevinsonOnlineFrame]:
    """Load independent inputs concurrently and preserve their original order.

    CPU edge computation runs in the loading workers. CUDA initialization runs
    on the caller while those workers load inputs; subsequent CUDA operations
    stay on that caller. A loader must support independent concurrent calls.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("preparation workers must be an integer between 1 and 32")
    if evaluation_backend not in {"numpy", "cuda"}:
        raise ValueError("online edge evaluation backend must be numpy or cuda")
    _validate_edge_parameters(alpha, gamma, max_radius)

    def load(item: _Item) -> tuple[LevinsonOnlineImage, FloatArray | None]:
        data = loader(item)
        edge = (
            levinson_edge_response(data.luminance, alpha=alpha, gamma=gamma, max_radius=max_radius)
            if evaluation_backend == "numpy"
            else None
        )
        return data, edge

    # Keep the sequential path free of executor imports and thread creation.
    pending: Iterator[tuple[LevinsonOnlineImage, FloatArray | None]]
    if workers == 1:
        executor = None
    else:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="calibrex-edge-input")
    try:
        pending = map(load, items) if executor is None else executor.map(load, items)
        runtime: _CudaEdgeImageRuntime | None = None
        if evaluation_backend == "cuda":
            try:
                from calibrex.solvers.cuda_online_edge_adapter import CudaEdgeImageRuntime
            except ImportError as exc:
                raise ValueError("CUDA online edge evaluation requires calibrex[cuda]") from exc
            runtime = CudaEdgeImageRuntime()
        frames = []
        for data, edge in pending:
            if edge is None:
                assert runtime is not None
                edge = runtime.evaluate(
                    data.luminance, alpha=alpha, gamma=gamma, max_radius=max_radius
                )
            frames.append(
                LevinsonOnlineFrame(
                    data.frame_id, edge, data.lidar_points, data.discontinuity_weights, data.camera
                )
            )
        return frames
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
