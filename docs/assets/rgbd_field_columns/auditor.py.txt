"""Independently audit the complete large TUM XYZ optimizer comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_radar_doppler_batches import sources  # noqa: E402

COMMITS = {
    "prior": "00fe74e00a72d8f619592627268df9a2408a1496",
    "current": "63b80c3109b7c714fd09129cc653d439e881c138",
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
        len(before) != 387
        or len(after) != 388
        or set(before) - set(after)
        or set(after) - set(before) != {"calibrex/graph/_rgbd_field_columns.py"}
        or {name for name in before if before[name] != after[name]}
        != {"calibrex/graph/_rgbd_batches.py"}
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
    reference_result = reference_trace = reference_inputs = None
    pids: set[int] = set()
    for ordinal, attempt in enumerate(attempts, 1):
        variant = attempt["variant"]
        if attempt["folder"] != f"{ordinal:02d}-{variant}":
            raise ValueError("measurement folder differs")
        folder = root / attempt["folder"]
        measurement = _load(folder / "measurement.json")
        result = _load(folder / "result.json.gz")
        trace = _load(folder / "traces.json.gz")
        for name, value in (("measurement", measurement), ("result", result), ("traces", trace)):
            jsonschema.validate(value, schemas[name])
        provenance = measurement["provenance"]
        if (
            result["provenance"] != provenance
            or trace["provenance"] != provenance
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
            or ("calibrex.graph._rgbd_field_columns" in provenance["imports"])
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
        if reference_result is None:
            reference_result, reference_trace, reference_inputs = fingerprint, history, verified
        elif (fingerprint, history, verified) != (
            reference_result,
            reference_trace,
            reference_inputs,
        ):
            raise ValueError("complete values, numeric array histories or inputs differ")
        if not (
            0 < measurement["seconds"] == attempt["optimizer_seconds"] < attempt["whole_seconds"]
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
        for key in ("optimizer_seconds", "whole_seconds")
    }
    report = {
        "optimizer_seconds_median": medians["optimizer_seconds"],
        "optimizer_speedup": medians["optimizer_seconds"]["prior"]
        / medians["optimizer_seconds"]["current"],
        "whole_seconds_median": medians["whole_seconds"],
        "whole_speedup": medians["whole_seconds"]["prior"] / medians["whole_seconds"]["current"],
        "complete_result_and_histories_exact": True,
        "attempts": attempts,
        "provenance_sha256": _sha(root / "provenance.json"),
    }
    if report != _load(root / "summary.json"):
        raise ValueError("published summary differs from the complete measurements")
    return report


def _load_inputs(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(audit(parser.parse_args().archive), indent=2))
