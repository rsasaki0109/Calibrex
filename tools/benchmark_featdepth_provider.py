"""Compare CPU and CUDA FeatDepth generation on identical public input frames.

Each sample starts a fresh interpreter and includes startup and artifact writes.
Repetitions describe timing variation on these frames, not independent accuracy
trials. External images, checkpoints and generated arrays are not redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from calibrex.core.benchmark import (
    BenchmarkDefinition,
    BenchmarkMethodDefinition,
    BenchmarkMetricDefinition,
    BenchmarkProtocol,
    BenchmarkProvenance,
    BenchmarkSplit,
    BenchmarkTrial,
    BenchmarkTrialProvenance,
    aggregate_benchmark_definition,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.data.depth import (
    DepthProviderArtifact,
    load_depth_provider,
    verify_depth_provider_files,
)


def _digest(path: Path) -> str:
    value = sha256_path(path)
    if value is None:
        raise ValueError(f"unreadable benchmark input: {path}")
    return value


def _difference(
    reference: DepthProviderArtifact, candidate: DepthProviderArtifact
) -> dict[str, float]:
    if reference.provider != candidate.provider:
        raise ValueError("provider identity changed")
    maximum = relative = squared = 0.0
    pixels = 0
    for left, right in zip(reference.observations, candidate.observations, strict=True):
        if (
            left.frame_id != right.frame_id
            or left.image != right.image
            or left.capture_time_ns != right.capture_time_ns
            or left.intrinsics != right.intrinsics
            or left.image_transform != right.image_transform
            or left.scale_convention != right.scale_convention
        ):
            raise ValueError("observation identity/grid changed")
        expected = np.load(left.depth.path, allow_pickle=False).astype(np.float64)
        measured = np.load(right.depth.path, allow_pickle=False).astype(np.float64)
        if expected.shape != measured.shape or not (
            np.isfinite(expected).all() and np.isfinite(measured).all()
        ):
            raise ValueError("depth grid mismatch or non-finite prediction")
        delta = np.abs(measured - expected)
        maximum = max(maximum, float(delta.max()))
        relative = max(relative, float((delta / np.maximum(np.abs(expected), 1e-12)).max()))
        squared += float(np.sum(delta * delta))
        pixels += delta.size
    return {"max_absolute": maximum, "max_relative": relative, "rmse": (squared / pixels) ** 0.5}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_path", type=Path)
    parser.add_argument("--featdepth-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--gpu-batch-size", type=int, default=2)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    if (
        args.frame_count < 2
        or args.repetitions < 1
        or args.gpu_batch_size < 1
        or args.preprocess_workers < 0
        or (args.cpu_threads is not None and args.cpu_threads < 1)
    ):
        raise ValueError("invalid frame, repetition, batch, worker or CPU thread count")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("--output-dir must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_featdepth_provider.py").resolve()
    runner_digest = _digest(runner)
    source_commit = git_commit()
    methods = {
        "cpu": ["--device", "cpu"]
        + (["--cpu-threads", str(args.cpu_threads)] if args.cpu_threads else []),
        "cuda_serial": ["--device", "cuda"],
        "cuda_batch": [
            "--device",
            "cuda",
            "--batch-size",
            str(args.gpu_batch_size),
            "--preprocess-workers",
            str(args.preprocess_workers),
        ],
    }
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    rows: list[dict[str, Any]] = []
    sources = {str(runner): runner_digest, str(Path(__file__).resolve()): _digest(Path(__file__))}
    anchor = None
    for repetition in range(1, args.repetitions + 1):
        order = list(methods)
        if repetition % 2 == 0:
            order.reverse()
        repeated: list[dict[str, Any]] = []
        providers = {}
        for method in order:
            directory = args.output_dir / f"{method}.repeat-{repetition:02d}"
            directory.mkdir()
            command = [
                sys.executable,
                str(runner),
                str(args.sequence_path.resolve()),
                "--featdepth-repository",
                str(args.featdepth_repository.resolve()),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--frame-count",
                str(args.frame_count),
                "--output-directory",
                str((directory / "depth").resolve()),
                "--artifact-output",
                str((directory / "provider.json").resolve()),
                "--timing-output",
                str((directory / "timing.json").resolve()),
                *methods[method],
            ]
            print(f"START {repetition} {method}", flush=True)
            started = time.perf_counter()
            with (directory / "run.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            elapsed = time.perf_counter() - started
            provider = load_depth_provider(directory / "provider.json")
            issues = verify_depth_provider_files(provider)
            if issues:
                raise ValueError(f"provider verification failed: {issues}")
            timing = load_depth_provider(directory / "timing.json").preprocessing[-1].parameters
            if (
                timing["adapter_sha256"] != runner_digest
                or _digest(runner) != runner_digest
                or timing["source_commit"] != source_commit
                or timing["provider_sha256"] != _digest(directory / "provider.json")
                or len(provider.observations) != args.frame_count
                or any(o.valid_fraction != 1 for o in provider.observations)
            ):
                raise ValueError("source/provider binding mismatch or incomplete prediction")
            for name in ("provider.json", "timing.json", "run.log"):
                sources[str((directory / name).resolve())] = _digest(directory / name)
            providers[method] = provider
            repeated.append(
                {
                    "method": method,
                    "repeat": repetition,
                    "order": order,
                    "command": command,
                    "provider_path": str((directory / "provider.json").resolve()),
                    "provider_sha256": timing["provider_sha256"],
                    "subprocess_wall_seconds": elapsed,
                    "timing": timing,
                }
            )
            print(f"DONE {repetition} {method}: {elapsed:.6f}s", flush=True)
        reference = providers["cpu"]
        if anchor is None:
            anchor = reference
        if reference.provenance.input_manifest_sha256 != anchor.provenance.input_manifest_sha256:
            raise ValueError("inputs changed across repetitions")
        for row in repeated:
            row["depth_difference"] = _difference(reference, providers[row["method"]])
        rows.extend(repeated)
    assert anchor is not None
    frames = [o.frame_id for o in anchor.observations]
    frame_digest = hashlib.sha256(json.dumps(frames).encode()).hexdigest()
    trials = []
    for row in rows:
        execution = row["timing"]["execution"]
        metrics = {
            "subprocess_wall_seconds": row["subprocess_wall_seconds"],
            **execution["timing_seconds"],
            **{f"depth_{k}": v for k, v in row["depth_difference"].items()},
        }
        trials.append(
            BenchmarkTrial(
                method_id=row["method"],
                split_id=f"repeat-{row['repeat']:02d}",
                status="success",
                metrics=metrics,
                runtime_seconds=row["subprocess_wall_seconds"],
                provenance=BenchmarkTrialProvenance(
                    command=subprocess.list2cmdline(row["command"]),
                    config_sha256=runner_digest,
                    input_sha256=anchor.provenance.input_manifest_sha256,
                    output_sha256=row["provider_sha256"],
                    execution_host=platform.platform(),
                    notes=[json.dumps(row, sort_keys=True)],
                ),
            )
        )
    definition = BenchmarkDefinition(
        benchmark_id="featdepth-cpu-cuda-inference",
        title="Public FeatDepth inference execution comparison",
        reference_method_id="cpu",
        bootstrap_samples=1,
        protocol=BenchmarkProtocol(
            protocol_id="featdepth-identical-images-v1",
            dataset_id=args.sequence_path.name,
            dataset_source_sha256=anchor.provenance.input_manifest_sha256,
            data_license="KITTI raw dataset terms; external data and weights are not redistributed",
            split_policy=(
                "Same uniformly selected images; holdout is a CPU prediction anchor, "
                "not independent accuracy data"
            ),
            splits=[
                BenchmarkSplit(
                    split_id=f"repeat-{i:02d}",
                    seed=i,
                    fit_count=len(frames),
                    holdout_count=len(frames),
                    fit_ids_sha256=frame_digest,
                    holdout_ids_sha256=frame_digest,
                )
                for i in range(1, args.repetitions + 1)
            ],
            initial_estimate_policy="No optimizer; same pinned checkpoint and input images",
            tuning_policy=(
                f"FP32, CPU threads={args.cpu_threads or 'PyTorch default'}, "
                f"GPU batch={args.gpu_batch_size}, workers={args.preprocess_workers}; "
                "reverse order every other repetition"
            ),
            failure_policy=(
                "Abort without an aggregate on subprocess, provenance, finite-value or "
                "verification failure; preserve per-run evidence"
            ),
        ),
        methods=[
            BenchmarkMethodDefinition(
                method_id=m,
                label=m,
                implementation="subprocess",
                tool_name="FeatDepth",
                tool_version=anchor.provider.source_commit,
                adapter_version="calibrex.featdepth_provider/v0.2",
                source_commit=source_commit,
            )
            for m in methods
        ],
        metrics=[
            BenchmarkMetricDefinition(
                name=k,
                label=k,
                unit="disparity"
                if k in ("depth_max_absolute", "depth_rmse")
                else "ratio"
                if k == "depth_max_relative"
                else "s",
                direction="lower",
                primary=k == "subprocess_wall_seconds",
                interpretation=(
                    "Difference from same-repetition CPU FP32 maps; "
                    "relative denominator floored at 1e-12"
                )
                if k.startswith("depth_")
                else (
                    "Prediction batches are a subset of inference_and_export; "
                    "wall includes fresh interpreter and artifact writes"
                ),
            )
            for k in trials[0].metrics
        ],
        trials=trials,
        limitations=[
            "Repeated identical frames on a shared host describe observed timing variation, "
            "not statistical confidence, recovery or portable performance.",
            "No separate warmup. Every subprocess is fresh; OS file caches and concurrent "
            "user work may vary. Parent verification and comparison are excluded from wall.",
            "Only FP32. Torch GPU counters exclude host memory and other device users. "
            "Loading holds one checkpoint byte buffer alongside deserialized weights.",
            "Raw inverse-depth/disparity is retained. FeatDepth training overlap with "
            "KITTI raw is possible; no holdout-based model refinement.",
            "Depth differences alone do not prove calibration equality; evaluate six-DoF "
            "search and held-out observations separately.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.benchmark_featdepth_provider",
            generator_version="v0.1",
            git_commit=source_commit,
            command=subprocess.list2cmdline([sys.executable, __file__, *arguments]),
            data_verified=True,
            source_artifacts=sources,
        ),
    )
    definition.save(args.output_dir / "inference.definition.json")
    aggregate_benchmark_definition(definition).save(args.output_dir / "inference.benchmark.json")


if __name__ == "__main__":
    main()
