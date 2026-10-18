"""Frozen-source, complete-artifact comparison of repeated probabilistic refinement.

The public KITTI fixture builder is in benchmark_probabilistic_refinement.py.
Projected image means are execution controls, not independent accuracy truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_bspline_lidar_imu import _freeze, _verify_sources  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write(path: Path, value: object) -> None:
    import jsonschema
    schema = ROOT / "tools/schemas/probabilistic_ablation_benchmark.schema.json"
    jsonschema.validate(value, json.loads(schema.read_bytes()))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def comparable_result(value: dict) -> dict:
    """Remove only declared execution lineage and run/source timestamps."""
    value = json.loads(json.dumps(value))
    for field in ("created_at", "git_commit", "execution"):
        value["provenance"].pop(field, None)
    return value


def comparable_benchmark(value: dict, *, cuda: bool) -> dict:
    """Keep all quality, failures, splits, confidence intervals and options."""
    value = json.loads(json.dumps(value))
    for field in ("generated_at", "git_commit"):
        value["provenance"].pop(field)
    for method in value["methods"]:
        method.pop("source_commit")
    for trial in value["trials"]:
        trial.pop("runtime_seconds")
        trial["provenance"].pop("output_sha256")
        if cuda:
            assert trial["provenance"]["notes"].pop() == "pose_evaluation_backend=cuda"
    for summary in value.get("method_summaries", {}).values():
        summary.pop("runtime_seconds")
    return value


def worker(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    sys.path.insert(0, str(source / "src"))
    expected = _verify_sources(source)
    protocol = json.loads((args.output.parent / "protocol.json").read_bytes())
    assert sha(Path(__file__)) == protocol["launcher_sha256"]
    assert sha(ROOT / "tools/benchmark_bspline_lidar_imu.py") == protocol["freezer_sha256"]
    assert all(sha(getattr(args, name)) == digest for name, digest in protocol["inputs"].items())
    from calibrex.core.io import read_mapping
    from calibrex.core.validation import validate_file
    from calibrex.evaluation import probabilistic_camera_lidar_run as runner
    from calibrex.evaluation import probabilistic_refinement_ablation as ablation
    from calibrex.solvers.probabilistic_camera_lidar_refiner import (
        ProbabilisticCameraLidarRefinementOptions,
    )

    # Keep the real Git lookup cost; bind the saved lineage to executed frozen bytes.
    native_git = runner.git_commit

    def frozen_git() -> str:
        native_git()
        return args.commit

    runner.git_commit = frozen_git
    ablation.git_commit = frozen_git
    args.output.mkdir()
    kwargs = (
        {}
        if args.variant == "prior"
        else {
            "evaluation_backend": "cuda" if args.variant == "cuda" else "numpy",
        }
    )
    start = time.perf_counter()
    definition, benchmark = ablation.run_probabilistic_refinement_ablation(
        args.correspondence,
        args.problem,
        result_directory=args.output / "results",
        command="complete-paired-probabilistic-refinement-comparison",
        split_seeds=tuple(protocol["split_seeds"]),
        base_options=ProbabilisticCameraLidarRefinementOptions(max_evaluations=400),
        bootstrap_samples=2000,
        **kwargs,
    )
    definition.save(args.output / "definition.json")
    benchmark.save(args.output / "benchmark.json")
    api = time.perf_counter() - start
    results, execution = {}, []
    for trial in definition.trials:
        seed = int(trial.split_id.removeprefix("seed-"))
        path = args.output / "results" / f"{trial.method_id}.seed-{seed:04d}.yaml"
        assert sha(path) == trial.provenance.output_sha256
        assert validate_file(path, "probabilistic-refinement-result").valid
        value = read_mapping(path)
        assert value["provenance"]["git_commit"] == args.commit
        assert (
            value["provenance"]["correspondence_artifact_sha256"]
            == protocol["inputs"]["correspondence"]
        )
        assert value["provenance"]["initialization_problem_sha256"] == protocol["inputs"]["problem"]
        if args.variant == "cuda":
            info = value["provenance"]["execution"]
            assert info["backend"] == "cuda" and info["cuda_calls"] > 0
            assert info["numpy_fallback_calls"] == 0
            assert info["math_policy"] == "fp64_no_fmad_numpy_reductions"
            execution.append(info)
        else:
            assert "execution" not in value["provenance"]
        results[path.name] = canonical(comparable_result(value))
    for name in ("definition", "benchmark"):
        assert validate_file(args.output / (name + ".json")).valid
    for name, mod in list(sys.modules.items()):
        if (name == "calibrex" or name.startswith("calibrex.")) and getattr(mod, "__file__", None):
            path = Path(mod.__file__).resolve()
            assert path.is_relative_to(source / "src")
            assert sha(path) == expected[path.relative_to(source / "src").as_posix()]
    _verify_sources(source)
    assert all(sha(getattr(args, name)) == digest for name, digest in protocol["inputs"].items())
    assert sha(Path(__file__)) == protocol["launcher_sha256"]
    write(
        args.output / "measurement.json",
        {
            "variant": args.variant,
            "api_seconds": api,
            "complete_results": results,
            "definition_sha256": canonical(
                comparable_benchmark(
                    definition.model_dump(mode="json"), cuda=args.variant == "cuda"
                )
            ),
            "benchmark_sha256": canonical(
                comparable_benchmark(benchmark.model_dump(mode="json"), cuda=args.variant == "cuda")
            ),
            "execution": execution,
            "provenance": {
                "source_commit": args.commit,
                "source_manifest_sha256": sha(source / "manifest.json"),
                "launcher_sha256": protocol["launcher_sha256"],
                "protocol_sha256": sha(args.output.parent / "protocol.json"),
                "python": sys.version,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", default="88d5602f47bb906c0f0cb6419a2c24afbd29ecfe")
    parser.add_argument("--current", default="HEAD")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--variant", choices=("prior", "numpy", "cuda"))
    parser.add_argument("--split-seeds", default="0,1")
    args = parser.parse_args()
    if args.variant:
        worker(args)
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.correspondence, args.problem = args.correspondence.resolve(), args.problem.resolve()
    seeds = [int(seed) for seed in args.split_seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("split seeds must be nonempty and unique")
    commits = {
        name: subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
        for name, ref in (("prior", args.prior), ("current", args.current))
    }
    for name, commit in commits.items():
        _freeze(args.output / ("source-" + name), commit)
    protocol = {
        "scope": (
            "Complete artifact-backed uncertainty ablation; public geometry with generated "
            "image means; performance/parity only"
        ),
        "order": ["prior", "numpy", "cuda", "cuda", "numpy", "prior"],
        "split_seeds": seeds,
        "uncertainty_variants": 4,
        "max_evaluations": 400,
        "bootstrap_samples": 2000,
        "gates": {
            "complete_results": (
                "exact except separately validated execution lineage and timestamps"
            ),
            "prior_cuda_whole_ratio_at_least": 2.0,
            "numpy_cuda_whole_ratio_at_least": 2.0,
        },
        "commits": commits,
        "source_manifests": {
            name: sha(args.output / ("source-" + name) / "manifest.json") for name in commits
        },
        "inputs": {name: sha(getattr(args, name)) for name in ("correspondence", "problem")},
        "launcher_sha256": sha(Path(__file__)),
        "freezer_sha256": sha(ROOT / "tools/benchmark_bspline_lidar_imu.py"),
        "provenance_binding": (
            "Real Git lookup timed; saved commit is the verified executed source capsule commit"
        ),
        "provenance": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "host_unreserved": True,
            "blas_threads": 1,
            "fresh_worker_and_cupy_cache": True,
            "cuda_driver_cache": False,
        },
    }
    write(args.output / "protocol.json", protocol)
    attempts = []
    for index, variant in enumerate(protocol["order"]):
        source_name = "prior" if variant == "prior" else "current"
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            CUPY_CACHE_DIR=str(args.output / f"cache-{index:02d}"),
            CUDA_CACHE_DISABLE="1",
        )
        assert not Path(env["CUPY_CACHE_DIR"]).exists()
        target = args.output / f"{index:02d}-{variant}"
        cmd = [
            sys.executable,
            str(Path(__file__)),
            "--variant",
            variant,
            "--source",
            str(args.output / ("source-" + source_name)),
            "--commit",
            commits[source_name],
            "--output",
            str(target),
            "--correspondence",
            str(args.correspondence),
            "--problem",
            str(args.problem),
        ]
        start = time.perf_counter()
        with (
            (args.output / f"{index:02d}-{variant}.stdout.log").open("w") as stdout,
            (args.output / f"{index:02d}-{variant}.stderr.log").open("w") as stderr,
        ):
            process = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr)
        record = {
            "index": index,
            "variant": variant,
            "whole_seconds": time.perf_counter() - start,
            "exit_code": process.returncode,
        }
        attempts.append(record)
        write(args.output / "attempts.json", attempts)
        print(json.dumps(record), flush=True)
        if process.returncode:
            raise RuntimeError("worker failed; all attempts and stderr retained")
    values = [
        json.loads(
            (args.output / f"{a['index']:02d}-{a['variant']}" / "measurement.json").read_bytes()
        )
        for a in attempts
    ]
    exact = all(
        len({canonical(v[key]) for v in values}) == 1
        for key in ("complete_results", "definition_sha256", "benchmark_sha256")
    )
    medians = {
        variant: statistics.median(a["whole_seconds"] for a in attempts if a["variant"] == variant)
        for variant in ("prior", "numpy", "cuda")
    }
    summary = {
        "exact_complete_results": exact,
        "median_whole_seconds": medians,
        "prior_cuda_whole_ratio": medians["prior"] / medians["cuda"],
        "numpy_cuda_whole_ratio": medians["numpy"] / medians["cuda"],
        "provenance": {
            "protocol_sha256": sha(args.output / "protocol.json"),
            "attempts_sha256": sha(args.output / "attempts.json"),
        },
    }
    summary["passes_gate"] = exact and all(
        summary[name] >= 2.0 for name in ("prior_cuda_whole_ratio", "numpy_cuda_whole_ratio")
    )
    write(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
