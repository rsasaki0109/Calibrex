"""Independently audit the explicit native CPU/CUDA RGB-D comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_radar_doppler_batches import sources  # noqa: E402

COMMITS = {
    "numpy": "e3f39bbfc7fb07df730d9f7bebf41e3fa70c14c0",
    "cuda": "e3f39bbfc7fb07df730d9f7bebf41e3fa70c14c0",
}
ORDER = ["cuda", "numpy", "numpy", "cuda", "cuda", "numpy"]


def _load(path: Path) -> dict[str, Any]:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def audit(root: Path) -> dict[str, Any]:
    """Check every file, imported source, input pin, numeric trace and reported time."""
    import jsonschema
    import yaml

    index = _load(root / "index.json")
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual_names != set(index["files"]) | {"index.json"}:
        raise ValueError("archive files are incomplete or unexpected")
    for name, expected in index["files"].items():
        if (
            _sha(root / name) != expected["sha256"]
            or (root / name).stat().st_size != expected["size_bytes"]
        ):
            raise ValueError("archive file bytes differ: " + name)
    schemas = _load(root / "record-schemas.json")
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
    protocol = _load(root / "provenance.json")
    jsonschema.validate(protocol, schemas["provenance"])
    if _sha(root / "sources-current-manifest.json") != (
        "5b656a584599617feedaaca7db2447bbe445e660d426b0be019d06466a4d4f0d"
    ) or _sha(root / "inputs.json") != (
        "f1ec8837db3c70ddb173869a4362dd54a5a9da3dc8e6f9cc7ac59a7b14e2a108"
    ):
        raise ValueError("frozen runtime manifest or original public input pins differ")
    if (
        protocol["commits"] != COMMITS
        or protocol["blas_threads"] != 1
        or protocol["launcher_sha256"] != _sha(root / "launcher.py.txt")
        or protocol["config_sha256"] != {v: _sha(root / ("config-" + v + ".yaml")) for v in COMMITS}
    ):
        raise ValueError("measured runtime, launcher or configuration differs")
    current = sources(root, "current")
    screen = sources(root, "screen")
    intermediate = sources(root, "intermediate")
    if (
        len(screen) != 389
        or len(current) != 392
        or len(intermediate) != 392
        or set(screen) - set(current)
        or set(current) - set(screen)
        != {
            "calibrex/graph/_rgbd_execution.py",
            "calibrex/solvers/cuda_rgbd_joint_adapter.py",
            "calibrex/solvers/_cuda_rgbd_joint_kernels.py",
        }
        or {name for name in screen if screen[name] != current[name]}
        != {
            "calibrex/graph/_point_plane_batches.py",
            "calibrex/graph/joint_optimization.py",
            "calibrex/solvers/native_tum_joint_slac_solver.py",
        }
        or set(intermediate) != set(current)
        or {name for name in current if current[name] != intermediate[name]}
        != {"calibrex/graph/_rgbd_execution.py"}
        or protocol["source_manifest_sha256"] != _sha(root / "sources-current-manifest.json")
        or protocol["parent_commit"] != "5a77afdf2b9a4714bdb75a4d1eb44950be518e66"
    ):
        raise ValueError("unexpected source capsules or runtime changes")
    manifests = dict.fromkeys(COMMITS, current)
    config = yaml.safe_load((root / "config-numpy.yaml").read_bytes())
    candidate = yaml.safe_load((root / "config-cuda.yaml").read_bytes())
    candidate_options = candidate["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"]
    if candidate_options.pop("joint_evaluation_backend") != "cuda" or candidate != config:
        raise ValueError("configurations differ beyond the explicit execution backend")
    options = config["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"]
    if (
        config["solver"]["max_iterations"] != 20
        or options["query_points_per_frame"] != 20000
        or options["map_points_per_frame"] != 50000
        or options["max_correspondences_per_frame"] != 4000
        or options["query_frame_count"] != 8
    ):
        raise ValueError("dense graph work limits differ")
    inputs = {item["name"]: item for item in _load_inputs(root / "inputs.json")}
    attempts_record = _load(root / "attempts.json")
    jsonschema.validate(attempts_record, schemas["attempts"])
    attempts = attempts_record["attempts"]
    if (
        attempts_record["provenance_sha256"] != _sha(root / "provenance.json")
        or [item["variant"] for item in attempts] != ORDER
        or any(item["exit_code"] for item in attempts)
    ):
        raise ValueError("incomplete, failed or reordered measurement attempts")
    reference_result = reference_trace = reference_inputs = reference_frontend = None
    pids: set[int] = set()
    for ordinal, attempt in enumerate(attempts, 1):
        variant = attempt["variant"]
        if attempt["folder"] != f"{ordinal:02d}-{variant}":
            raise ValueError("measurement folder differs")
        folder = root / attempt["folder"]
        measurement = _load(folder / "measurement.json")
        result = _load(folder / "result.json.gz")
        trace = _load(folder / "traces.json.gz")
        frontend = _load(folder / "frontend.json.gz")
        for name, value in (
            ("measurement", measurement),
            ("result", result),
            ("traces", trace),
            ("frontend", frontend),
        ):
            jsonschema.validate(value, schemas[name])
        execution_record = _load(folder / "execution.json")
        jsonschema.validate(execution_record, schemas["execution_record"])
        if (
            execution_record["provenance"]
            != {
                "run_provenance_sha256": _sha(root / "provenance.json"),
                "source_manifest_sha256": _sha(root / "sources-current-manifest.json"),
                "launcher_sha256": protocol["launcher_sha256"],
            }
            or execution_record["execution"] != measurement["execution"]
        ):
            raise ValueError("execution telemetry binding differs")
        execution = measurement["execution"]
        if variant == "numpy":
            if execution is not None:
                raise ValueError("NumPy worker unexpectedly used a CUDA adapter")
        elif (
            execution is None
            or execution["linearizations"] != 21
            or execution["evaluations"] != 606
            or execution["kernel_launches"] != 1090
            or execution["budget_fallbacks"] != 0
            or execution["cache_budget_bytes"] != 67108864
            or not 0 < execution["peak_owned_bytes_bound"] <= execution["cache_budget_bytes"]
            or not 0 < execution["max_column_elements"] <= 131072
            or "fp64;fmad=false" not in execution["implementation"]
            or not execution["implementation"].endswith(
                "kernel_sha256=452e7fff2430c475e6e7cbe3f9efa36cb8990f28cdb2cb6408246e33e549742f"
            )
        ):
            raise ValueError("GPU math, population, work counters or memory bounds differ")
        provenance = measurement["provenance"]
        if (
            result["provenance"] != provenance
            or trace["provenance"] != provenance
            or frontend["provenance"] != provenance
            or provenance["variant"] != variant
            or provenance["git_commit"] != COMMITS[variant]
            or provenance["source_manifest_sha256"] != _sha(root / "sources-current-manifest.json")
            or provenance["launcher_sha256"] != protocol["launcher_sha256"]
            or provenance["config_sha256"] != protocol["config_sha256"][variant]
            or provenance["requested_backend"] != variant
            or provenance["input_inventory_sha256"] != _sha(root / "inputs.json")
            or provenance["python"] != protocol["python"]
        ):
            raise ValueError("worker provenance differs")
        manifest = manifests[variant]
        if (
            any(
                path != name.replace(".", "/") + ".py"
                and path != name.replace(".", "/") + "/__init__.py"
                for name, path in provenance["imports"].items()
            )
            or any(path not in manifest for path in provenance["imports"].values())
            or ("calibrex.solvers.cuda_rgbd_joint_adapter" in provenance["imports"])
            != (variant == "cuda")
            or provenance["pid"] in pids
        ):
            raise ValueError("worker imports or process identity differ")
        pids.add(provenance["pid"])
        verified = provenance["verified_inputs"]
        if len(verified) != 13 or any(inputs.get(name) != pin for name, pin in verified.items()):
            raise ValueError("public input bytes differ")
        if (
            measurement["result_sha256"] != _sha(folder / "result.json.gz")
            or measurement["frontend_sha256"] != _sha(folder / "frontend.json.gz")
            or measurement["traces_sha256"] != _sha(folder / "traces.json.gz")
            or measurement["data_factors"] != 27544
            or measurement["all_factors"] != 27554
            or measurement["iterations"] != 20
            or measurement["status"] != "max_iterations"
            or result["payload"]["status"] != "max_iterations"
            or len(result["payload"]["history"]) != 20
            or len(result["payload"]["known_bad_probes"]) != 60
            or measurement["trace_events"] != 102
            or Counter(event["kind"] for event in trace["payload"])
            != {"_linearize": 21, "_factor_residuals": 81}
        ):
            raise ValueError("complete optimizer work or result binding differs")
        fingerprint, history = _canonical(result["payload"]), _canonical(trace["payload"])
        front = _canonical(frontend["payload"])
        _frontend(frontend["payload"], verified)
        if reference_result is None:
            reference_result, reference_trace, reference_inputs, reference_frontend = (
                fingerprint,
                history,
                verified,
                front,
            )
        elif (fingerprint, history, verified, front) != (
            reference_result,
            reference_trace,
            reference_inputs,
            reference_frontend,
        ):
            raise ValueError("complete values, numeric array histories or inputs differ")
        if not (
            0 < measurement["seconds"] == attempt["optimizer_seconds"] < attempt["whole_seconds"]
            and 0
            < measurement["frontend_seconds"]
            == attempt["frontend_seconds"]
            < attempt["whole_seconds"]
            and attempt["data_factors"] == 27544
        ):
            raise ValueError("reported timing or population differs")
    medians = {
        key: {
            variant: statistics.median(
                attempt[key] for attempt in attempts if attempt["variant"] == variant
            )
            for variant in COMMITS
        }
        for key in ("frontend_seconds", "optimizer_seconds", "whole_seconds")
    }
    report = {
        "frontend_seconds_median": medians["frontend_seconds"],
        "frontend_speedup": medians["frontend_seconds"]["numpy"]
        / medians["frontend_seconds"]["cuda"],
        "optimizer_seconds_median": medians["optimizer_seconds"],
        "optimizer_speedup": medians["optimizer_seconds"]["numpy"]
        / medians["optimizer_seconds"]["cuda"],
        "whole_seconds_median": medians["whole_seconds"],
        "whole_speedup": medians["whole_seconds"]["numpy"] / medians["whole_seconds"]["cuda"],
        "complete_result_and_histories_exact": True,
        "complete_frontend_exact": True,
        "attempts": attempts,
        "provenance_sha256": _sha(root / "provenance.json"),
    }
    if report != _load(root / "summary.json"):
        raise ValueError("published summary differs from the complete measurements")
    parent = _load(root / "parent-result.json.gz")
    if _canonical(parent["payload"]) != reference_result:
        raise ValueError("complete result differs from the previous published runtime")
    _suite(root, {"prior": current, "current": current})
    return report


def _frontend(events: list[dict[str, Any]], verified: dict[str, Any]) -> None:
    if Counter(event["kind"] for event in events) != {
        "image": 11,
        "samples": 11,
        "map": 1,
        "initial_problem": 1,
    }:
        raise ValueError("incomplete frontend trace")
    images = [event for event in events if event["kind"] == "image"]
    samples = [event for event in events if event["kind"] == "samples"]
    if (
        {event["path"] for event in images} != set(verified) - {"depth.txt", "groundtruth.txt"}
        or [event["path"] for event in images] != [event["path"] for event in samples]
        or any((event["width"], event["height"]) != (640, 480) for event in images)
        or [event["count"] for event in samples]
        != [
            25926,
            26554,
            26375,
            15217,
            15049,
            14562,
            14465,
            15318,
            14963,
            15222,
            14795,
        ]
        or next(event for event in events if event["kind"] == "map")["records"] != 78855
        or next(event for event in events if event["kind"] == "map")["voxels"] != 645
        or events[-1]["kind"] != "initial_problem"
        or events[-1]["data_factors"] != 27544
    ):
        raise ValueError("frontend population, order or correspondence work differs")


SUITE_COMMITS = {
    "prior": "e3f39bbfc7fb07df730d9f7bebf41e3fa70c14c0",
    "current": "e3f39bbfc7fb07df730d9f7bebf41e3fa70c14c0",
}


def _suite(root: Path, manifests: dict[str, dict[str, str]]) -> None:
    from tools.benchmark_tum_window_processes import comparable

    with tarfile.open(root / "full-suite-screen.tar.gz") as bundle:
        files = {
            member.name: bundle.extractfile(member).read()
            for member in bundle.getmembers()
            if member.isfile()
        }

    def sha(name: str) -> str:
        return hashlib.sha256(files[name]).hexdigest()

    def load(name: str) -> Any:
        raw = gzip.decompress(files[name]) if name.endswith(".gz") else files[name]
        return json.loads(raw)

    protocol = load("provenance.json")
    summary = load("summary.json")
    if (
        summary != _load(root / "suite-summary.json")
        or protocol["commits"] != SUITE_COMMITS
        or protocol["blas_threads"] != 1
        or protocol["launcher_sha256"] != sha("launcher.py.txt")
        or protocol["trace_worker_sha256"] != sha("trace-worker.py.txt")
        or protocol["input_inventory_sha256"] != _sha(root / "inputs.json")
        or summary["provenance"]
        != {
            "launcher_sha256": sha("launcher.py.txt"),
            "run_provenance_sha256": sha("provenance.json"),
        }
        or not summary["complete_results_exact"]
        or not summary["all_original_numeric_histories_exact"]
        or summary["fixed_population_probe_events_exact"] != 492
    ):
        raise ValueError("full-suite provenance or summary differs")
    import yaml

    cpu = yaml.safe_load(files["prior/config.yaml"])
    gpu = yaml.safe_load(files["current/config.yaml"])
    if (
        gpu["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"].pop(
            "joint_evaluation_backend"
        )
        != "cuda"
        or gpu != cpu
    ):
        raise ValueError("full-suite configs differ beyond the execution backend")
    for variant in SUITE_COMMITS:
        if (
            load(variant + "/source-manifest.json") != manifests[variant]
            or load(variant + "/protocol.json")["payload"]["commits"]
            != {"current": SUITE_COMMITS[variant]}
            or sha(variant + "/config.yaml") != protocol["config_sha256"][variant]
        ):
            raise ValueError("full-suite source or configuration differs")
    attempts = summary["attempts"]
    if [attempt["variant"] for attempt in attempts] != [
        "current",
        "prior",
        "prior",
        "current",
    ] or load("attempts.json")["attempts"] != [
        {key: value for key, value in attempt.items() if key != "summed_population_seconds"}
        for attempt in attempts
    ]:
        raise ValueError("full-suite attempts are incomplete or reordered")
    reference = None
    for ordinal, attempt in enumerate(attempts, 1):
        variant = attempt["variant"]
        folder = attempt["folder"] + "/"
        measurement = load(folder + "measurement.json")
        provenance = measurement["provenance"]
        payload = measurement["payload"]
        if (
            attempt["folder"] != f"{ordinal:02d}-{variant}"
            or attempt["exit_code"] != 0
            or not 0 < payload["api_seconds"] < attempt["seconds"]
            or provenance["git_commit"] != SUITE_COMMITS[variant]
            or provenance["protocol_sha256"] != sha(variant + "/protocol.json")
            or payload["solve_count"] != 24
            or payload["jacobian_calls"] != 168
            or payload["residual_calls"] != 1624
            or len(payload["processes"]) != 3
            or payload["result_sha256"] != sha(folder + "result.json.gz")
            or payload["history_sha256"] != sha(folder + "history.json.gz")
        ):
            raise ValueError("full-suite work or provenance differs")
        for process in payload["processes"]:
            name = folder + process["path"]
            record = load(name)
            if (
                process["sha256"] != sha(name)
                or record["provenance"] != provenance
                or any(
                    manifests[variant].get(item["path"]) != item["sha256"]
                    for item in record["payload"]["imports"].values()
                )
            ):
                raise ValueError("full-suite process source differs")
        population_names = [
            folder + "population-history.json.gz",
            *sorted(
                name
                for name in files
                if name.startswith(folder + "children/")
                and name.endswith("/population-history.json.gz")
            ),
        ]
        events, seconds = [], 0.0
        for name in population_names:
            record = load(name)
            if (
                record["provenance"]["launcher_sha256"] != protocol["launcher_sha256"]
                or record["provenance"]["protocol_sha256"] != provenance["protocol_sha256"]
            ):
                raise ValueError("full-suite population trace provenance differs")
            for event in record["events"]:
                seconds += event.pop("seconds")
                events.append(event)
        if len(events) != 492 or seconds != attempt["summed_population_seconds"]:
            raise ValueError("full-suite population evaluation work differs")
        events.sort(key=lambda item: item["window"])
        history = load(folder + "history.json.gz")
        if history["provenance"] != provenance:
            raise ValueError("full-suite numeric history provenance differs")
        result = comparable(load(folder + "result.json.gz"))
        run = result["run"]
        if (
            run["config_sha256"] != protocol["config_sha256"][variant]
            or run["provenance"]["config_sha256"] != protocol["config_sha256"][variant]
        ):
            raise ValueError("full-suite result configuration differs")
        run["provenance"]["command"][2] = "<verified identical config bytes>"
        run["config_sha256"] = run["provenance"]["config_sha256"] = (
            "<verified backend-only config difference>"
        )
        backend = run["provenance"]["native_tum_joint_slac_options"].pop(
            "joint_evaluation_backend", "numpy"
        )
        identity = run["provenance"].pop("native_tum_joint_slac_cuda", None)
        if backend != ("cuda" if variant == "current" else "numpy"):
            raise ValueError("full-suite execution backend differs")
        if variant == "current":
            if (
                not isinstance(identity, str)
                or "initial_min_points=4096;fp64;fmad=false" not in identity
            ):
                raise ValueError("full-suite adaptive execution policy differs")
        elif identity is not None:
            raise ValueError("CPU suite unexpectedly recorded CUDA")
        fingerprint = _canonical((history["payload"], events, result))
        if reference is not None and fingerprint != reference:
            raise ValueError("full-suite complete results or evaluation histories differ")
        reference = fingerprint
    medians = {
        variant: statistics.median(
            attempt["seconds"] for attempt in attempts if attempt["variant"] == variant
        )
        for variant in SUITE_COMMITS
    }
    if (
        medians != summary["whole_seconds_median"]
        or medians["prior"] / medians["current"] != summary["whole_speedup"]
    ):
        raise ValueError("full-suite timing summary differs")


def _load_inputs(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(audit(parser.parse_args().archive), indent=2))
