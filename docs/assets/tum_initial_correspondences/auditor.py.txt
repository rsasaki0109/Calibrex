"""Independently audit the native TUM initial correspondence comparison."""

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
    "prior": "2ae6cb775e002cd21d2af6ccbef133600b27a0fd",
    "current": "b654ee4da6de5c9222be8fcdb23c43eb20bb4f2d",
}
ORDER = ["current", "prior", "prior", "current", "current", "prior"]


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
    if (
        protocol["commits"] != COMMITS
        or protocol["blas_threads"] != 1
        or protocol["launcher_sha256"] != _sha(root / "launcher.py.txt")
        or protocol["config_sha256"] != _sha(root / "config.yaml")
    ):
        raise ValueError("measured runtime, launcher or configuration differs")
    manifests = {variant: sources(root, variant) for variant in COMMITS}
    before, after = manifests["prior"], manifests["current"]
    if (
        len(before) != 388
        or len(after) != 389
        or set(before) - set(after)
        or set(after) - set(before) != {"calibrex/solvers/_tum_initial_geometry.py"}
        or {name for name in before if before[name] != after[name]}
        != {"calibrex/solvers/native_tum_joint_slac_solver.py"}
    ):
        raise ValueError("unexpected runtime source changes")
    config = yaml.safe_load((root / "config.yaml").read_bytes())
    options = config["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"]
    if (
        config["solver"]["max_iterations"] != 20
        or options["query_points_per_frame"] != 20000
        or options["map_points_per_frame"] != 50000
        or options["max_correspondences_per_frame"] != 4000
        or options["query_frame_count"] != 8
    ):
        raise ValueError("large graph work limits differ")
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
        provenance = measurement["provenance"]
        if (
            result["provenance"] != provenance
            or trace["provenance"] != provenance
            or frontend["provenance"] != provenance
            or provenance["variant"] != variant
            or provenance["git_commit"] != COMMITS[variant]
            or provenance["source_manifest_sha256"]
            != _sha(root / f"sources-{variant}-manifest.json")
            or provenance["launcher_sha256"] != protocol["launcher_sha256"]
            or provenance["config_sha256"] != protocol["config_sha256"]
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
            or ("calibrex.solvers._tum_initial_geometry" in provenance["imports"])
            != (variant == "current")
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
        "frontend_speedup": medians["frontend_seconds"]["prior"]
        / medians["frontend_seconds"]["current"],
        "optimizer_seconds_median": medians["optimizer_seconds"],
        "optimizer_speedup": medians["optimizer_seconds"]["prior"]
        / medians["optimizer_seconds"]["current"],
        "whole_seconds_median": medians["whole_seconds"],
        "whole_speedup": medians["whole_seconds"]["prior"] / medians["whole_seconds"]["current"],
        "complete_result_and_histories_exact": True,
        "complete_frontend_exact": True,
        "attempts": attempts,
        "provenance_sha256": _sha(root / "provenance.json"),
    }
    if report != _load(root / "summary.json"):
        raise ValueError("published summary differs from the complete measurements")
    _suite(root, manifests)
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
        or protocol["commits"] != COMMITS
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
    for variant in COMMITS:
        if (
            load(variant + "/source-manifest.json") != manifests[variant]
            or load(variant + "/protocol.json")["payload"]["commits"]
            != {"current": COMMITS[variant]}
            or sha(variant + "/config.yaml") != protocol["config_sha256"]
        ):
            raise ValueError("full-suite source or configuration differs")
    attempts = summary["attempts"]
    if [attempt["variant"] for attempt in attempts] != ORDER[:4] or load("attempts.json")[
        "attempts"
    ] != [
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
            or provenance["git_commit"] != COMMITS[variant]
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
            run["config_sha256"] != protocol["config_sha256"]
            or run["provenance"]["config_sha256"] != protocol["config_sha256"]
        ):
            raise ValueError("full-suite result configuration differs")
        run["provenance"]["command"][2] = "<verified identical config bytes>"
        fingerprint = _canonical((history["payload"], events, result))
        if reference is not None and fingerprint != reference:
            raise ValueError("full-suite complete results or evaluation histories differ")
        reference = fingerprint
    medians = {
        variant: statistics.median(
            attempt["seconds"] for attempt in attempts if attempt["variant"] == variant
        )
        for variant in COMMITS
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
