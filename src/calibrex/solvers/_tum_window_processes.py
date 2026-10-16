"""Bounded native TUM window processes with deterministic parent aggregation."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.data.tum_rgbd import TUMDepthIntrinsics, TUMDepthPoseEntry
from calibrex.graph.joint_optimization import JointOptimizerResult
from calibrex.solvers import native_tum_joint_slac_solver as a
from calibrex.solvers._tum_window_transport import _write_native


@dataclass(frozen=True)
class _WindowRequest:
    root: Path
    paired: list[TUMDepthPoseEntry]
    intrinsics: TUMDepthIntrinsics
    config: CalibrationConfig
    options: a.TUMJointSlacOptions
    start_index: int


@dataclass(frozen=True)
class _WindowResponse:
    products: a._TUMWindowProducts
    pid: int
    parent_pid: int


def _worker_command(request: Path, response: Path) -> list[str]:
    return [
        sys.executable, "-m", "calibrex.solvers._tum_window_worker", str(request), str(response)
    ]


def _direct_python_command(command: list[str], env: dict[str, str]) -> list[str]:
    # Match CPython's Windows multiprocessing launcher: a venv redirector is
    # another process, so terminating its PID need not terminate the interpreter.
    # __PYVENV_LAUNCHER__ keeps this interpreter's venv and installed dependencies.
    base = str(getattr(sys, "_base_executable", sys.executable))
    if (
        os.name == "nt"
        and os.path.normcase(base) != os.path.normcase(sys.executable)
        and os.path.normcase(command[0]) == os.path.normcase(sys.executable)
    ):
        env["__PYVENV_LAUNCHER__"] = sys.executable
        return [base, *command[1:]]
    return command


def _execute_request(request: Path, response: Path) -> None:
    # Both paths belong to the current call's private TemporaryDirectory.
    with request.open("rb") as stream:
        job = pickle.load(stream)
    if type(job) is not _WindowRequest:
        raise ValueError("invalid native TUM window request")
    options = replace(job.options, frame_start_index=job.start_index)
    selected = a._select_frames(job.paired, options)
    products = a._TUMWindowProducts((), (), (), (), (), (), ())
    if selected is not None:
        try:
            problem = a._build_problem(job.root, selected, job.intrinsics, options)
        except (OSError, ValueError):
            problem = None
        if (
            problem is not None
            and len(problem.query_frames) >= a._MIN_QUERY_FRAMES
            and len(problem.measurements) >= a._MIN_TOTAL_CORRESPONDENCES
        ):
            run = a._TUMWindowRun(
                job.start_index, problem, a._optimize_problem(job.config, problem, options)
            )
            # The global primary controls which window receives multistart.
            products = a._evaluate_window_runs(
                job.root, job.intrinsics, job.config, job.options, (run,)
            )
    _write_native(response, _WindowResponse(products, os.getpid(), os.getppid()))


class _WindowChildren:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.lock = threading.Lock()
        self.closed = False
        self.processes: set[subprocess.Popen[bytes]] = set()

    def run(self, job: _WindowRequest) -> a._TUMWindowProducts:
        request = self.folder / f"{job.start_index}-request.pkl"
        response = self.folder / f"{job.start_index}-response.pkl"
        stdout = self.folder / f"{job.start_index}-stdout.log"
        stderr = self.folder / f"{job.start_index}-stderr.log"
        _write_native(request, job)
        env = dict(os.environ)
        source = str(Path(a.__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH", ""))))
        # Each process already evaluates a complete window. Do not multiply
        # the worker count by a default BLAS/OpenMP pool within each child.
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            env.setdefault(name, "1")
        with stdout.open("wb") as out, stderr.open("wb") as err:
            with self.lock:
                if self.closed:
                    raise RuntimeError("native TUM window execution was cancelled")
                command = _direct_python_command(_worker_command(request, response), env)
                process = subprocess.Popen(
                    command, env=env, stdout=out, stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self.processes.add(process)
            try:
                code = process.wait()
            finally:
                with self.lock:
                    self.processes.discard(process)
        if code:
            detail = stderr.read_text(encoding="utf-8", errors="replace")[-8192:]
            raise RuntimeError(f"TUM window {job.start_index} worker exited {code}: {detail}")
        with response.open("rb") as stream:
            reply = pickle.load(stream)
        if (
            type(reply) is not _WindowResponse
            or reply.pid != process.pid
            or reply.parent_pid != os.getpid()
        ):
            raise ValueError(f"TUM window {job.start_index} worker process identity differs")
        products = reply.products
        if type(products) is not a._TUMWindowProducts or any(
            run.start_index != job.start_index
            for field in fields(products) for run in getattr(products, field.name)
        ):
            raise ValueError(f"TUM window {job.start_index} response identity differs")
        return products

    def close(self) -> None:
        with self.lock:
            self.closed = True
            processes = tuple(self.processes)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _merge_windows(products: list[a._TUMWindowProducts]) -> a._TUMWindowProducts:
    return a._TUMWindowProducts(*(
        tuple(run for product in products for run in getattr(product, field.name))
        for field in fields(a._TUMWindowProducts)
    ))


def _evaluate_windows(
    root: Path,
    paired: list[TUMDepthPoseEntry],
    intrinsics: TUMDepthIntrinsics,
    config: CalibrationConfig,
    options: a.TUMJointSlacOptions,
    problem: a._TUMJointProblem,
) -> tuple[JointOptimizerResult, a._TUMWindowProducts, dict[str, Any]]:
    starts = options.replication_start_indices
    limit = min(3, options.window_workers, os.cpu_count() or 1, len(starts))
    execution: dict[str, Any] = {
        "method": "native_tum_window_execution/v0.1",
        "backend": "process" if limit > 1 else "serial",
        "requested_workers": options.window_workers,
        "effective_workers": max(1, limit),
        "window_start_indices": list(starts),
        "aggregation": "configured replication order; primary problem is reused",
    }
    if limit <= 1:
        result = a._optimize_problem(config, problem, options)
        primary = a._TUMWindowRun(options.frame_start_index, problem, result)
        runs = a._replicate_windows(root, paired, intrinsics, config, options, primary)
        return result, a._evaluate_window_runs(root, intrinsics, config, options, runs), execution

    # Reserve one execution slot for the caller, which owns the primary graph.
    with TemporaryDirectory(prefix="calibrex-tum-windows-") as temporary:
        children = _WindowChildren(Path(temporary))
        pool = ThreadPoolExecutor(max_workers=limit - 1, thread_name_prefix="tum-window-launch")
        futures = {}
        try:
            for start in starts:
                if start != options.frame_start_index:
                    futures[start] = pool.submit(
                        children.run,
                        _WindowRequest(root, paired, intrinsics, config, options, start),
                    )
            result = a._optimize_problem(config, problem, options)
            primary = a._TUMWindowRun(options.frame_start_index, problem, result)
            local = a._evaluate_window_runs(
                root, intrinsics, config, options,
                (primary,) if options.frame_start_index in starts else (),
            )
            products = [
                local if start == options.frame_start_index else futures[start].result()
                for start in starts
            ]
        finally:
            children.close()
            pool.shutdown(wait=True, cancel_futures=True)
        return result, _merge_windows(products), execution
