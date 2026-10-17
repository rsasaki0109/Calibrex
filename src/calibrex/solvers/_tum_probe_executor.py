"""Bounded spawned CPU workers for native, independent TUM probe evaluations."""

from __future__ import annotations

import os
import pickle
import threading
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context, parent_process
from multiprocessing.process import BaseProcess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from calibrex.data.livox import VoxelPlaneMap, VoxelPlaneQueryContext
from calibrex.graph.joint_optimization import ParameterValues
from calibrex.graph.joint_reassociation import JointReassociationCallback, JointReassociationState
from calibrex.solvers import native_tum_joint_slac_solver as a
from calibrex.solvers._tum_window_transport import _write_native

_CALLBACKS: ContextVar[OrderedDict[str, JointReassociationCallback] | None] = ContextVar(
    "native_tum_probe_callbacks", default=None
)


@dataclass(frozen=True)
class _ProbeJob:
    kind: Literal["pair", "spatial"]
    subject: a._TUMXYZPairProblem | a._TUMSpatialWindowRun
    options: a.TUMJointSlacOptions
    planes: VoxelPlaneMap | None


def _effective_probe_workers(options: a.TUMJointSlacOptions) -> int:
    windows = min(3, options.window_workers, len(options.replication_start_indices))
    return min(4, options.probe_workers, max(1, (os.cpu_count() or 1) // max(1, windows)))


def _watch_parent(parent: BaseProcess) -> None:
    while parent.is_alive():
        parent.join(timeout=1.0)
    # Window cancellation can terminate a worker's parent without unwinding its
    # executor. Only this owned probe process exits when that parent is gone.
    os._exit(1)


def _probe_worker_init() -> None:
    parent = parent_process()
    if parent is not None:
        threading.Thread(target=_watch_parent, args=(parent,), daemon=True).start()


def _load_callback(path: Path) -> JointReassociationCallback:
    # Written by the calling pool into its private TemporaryDirectory; never an
    # externally supplied calibration file or arbitrary callback pickle.
    with path.open("rb") as stream:
        job = pickle.load(stream)
    if type(job) is not _ProbeJob:
        raise ValueError("invalid native TUM probe job")
    if job.kind == "pair" and type(job.subject) is a._TUMXYZPairProblem:
        return partial(
            a._xyz_pair_reassociation_state, job.subject, options=job.options,
            context=a._TUMXYZPairReassociationContext(job.subject, job.options),
        )
    if (
        job.kind == "spatial" and type(job.subject) is a._TUMSpatialWindowRun
        and job.planes is not None
    ):
        return partial(
            a._reassociated_spatial_state, job.subject, job.planes, job.options,
            points=a._spatial_point_batch(job.subject.source_problem,
                                          job.subject.source_problem.measurements),
            query=VoxelPlaneQueryContext(job.planes), factor_cache=OrderedDict(),
        )
    raise ValueError("native TUM probe job kind and subject differ")


def _probe_callback(values: ParameterValues, *, job_path: str) -> JointReassociationState:
    callbacks = _CALLBACKS.get()
    if callbacks is None:
        callbacks = OrderedDict()
        _CALLBACKS.set(callbacks)
    callback = callbacks.get(job_path)
    if callback is None:
        callback = _load_callback(Path(job_path))
        callbacks[job_path] = callback
        while len(callbacks) > 2:
            callbacks.popitem(last=False)
    callbacks.move_to_end(job_path)
    return callback(values)


class _TUMProbePool:
    """One shared pool per window; bootstrap each native problem once per worker."""

    def __init__(self, workers: int) -> None:
        if not 1 < workers <= 4:
            raise ValueError("native probe pool requires two to four workers")
        self.temporary = TemporaryDirectory(prefix="calibrex-tum-probes-")
        self.folder = Path(self.temporary.name)
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"), initializer=_probe_worker_init,
        )
        self.jobs = 0

    def __enter__(self) -> _TUMProbePool:
        return self

    def __exit__(self, *_error: object) -> None:
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self.temporary.cleanup()

    @contextmanager
    def callback(
        self, kind: Literal["pair", "spatial"],
        subject: a._TUMXYZPairProblem | a._TUMSpatialWindowRun,
        options: a.TUMJointSlacOptions, callback: JointReassociationCallback,
        planes: VoxelPlaneMap | None = None,
    ) -> Iterator[JointReassociationCallback]:
        self.jobs += 1
        path = self.folder / f"{self.jobs}-{kind}.pkl"
        _write_native(path, _ProbeJob(kind, subject, options, planes))
        local = OrderedDict(_CALLBACKS.get() or ())
        local[str(path)] = callback
        token = _CALLBACKS.set(local)
        try:
            yield partial(_probe_callback, job_path=str(path))
        finally:
            _CALLBACKS.reset(token)
