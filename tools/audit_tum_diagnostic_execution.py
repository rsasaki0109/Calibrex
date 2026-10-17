"""Audit complete production timings separately from diagnostic trace overhead."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import statistics
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

COMMITS = {
    "prior": "dddf26a422ee320736d7cd8e1380733064095c8a",
    "current": "ae943a932e220813650fc5174e3c60a5d5b5f526",
}
MANIFESTS = {
    "prior": "5658118b427355c7f5dbc0f8f7de2d81b882c2398e5d8a15dda86742a101f0e7",
    "current": "2cabf710bad2aad0794336ca7e10bcba0d8722f2fb70b885a8c298697a309593",
}
INPUTS = "f1ec8837db3c70ddb173869a4362dd54a5a9da3dc8e6f9cc7ac59a7b14e2a108"
CONFIG = "e736c92605a536ece26276f19d782ecbac332b8f4c6b05c1bab92504fee543db"
ORDER = ["current", "prior", "prior", "current", "current", "prior"]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(raw: bytes) -> Any:
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _members(raw: bytes) -> dict[str, bytes]:
    result = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as bundle:
        for member in bundle:
            if (not member.isfile() or member.name in result or member.name.startswith("/")
                or ".." in member.name.split("/") or member.size < 0):
                raise ValueError("unsafe, duplicate or non-file archive member")
            total += member.size
            if total > 128 * 1024 * 1024:
                raise ValueError("archive exceeds inspection budget")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("missing archive member")
            result[member.name] = stream.read()
    return result


def _comparable(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(_canonical(payload))
    for key in ("id", "created_at", "git_commit"):
        result["run"].pop(key)
    for key in ("generated_at", "git_commit"):
        result["run"]["provenance"].pop(key)
    if result["solid_state"] is not None:
        result["solid_state"]["provenance"].pop("git_commit")
    # Execution policy and every other result field are retained.
    return result


def audit(root: Path) -> dict[str, Any]:
    """Verify frozen candidates, all attempts, complete results and work traces."""
    import jsonschema

    index = _load((root / "index.json").read_bytes())
    names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if names != set(index["files"]) | {"index.json"}:
        raise ValueError("archive files differ")
    for name, pin in index["files"].items():
        raw = (root / name).read_bytes()
        if _sha(raw) != pin["sha256"] or len(raw) != pin["size_bytes"]:
            raise ValueError("archive bytes differ: " + name)
    schemas = _load((root / "record-schemas.json").read_bytes())
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
    manifests = {}
    for variant, pin in MANIFESTS.items():
        raw = (root / f"sources-{variant}-manifest.json").read_bytes()
        if _sha(raw) != pin:
            raise ValueError("measured source manifest differs")
        manifest = _load(raw)
        members = _members((root / f"sources-{variant}.tar.gz").read_bytes())
        if {name.removeprefix("src/"): _sha(data) for name, data in members.items()} != manifest:
            raise ValueError("source capsule differs")
        manifests[variant] = manifest
    if len(manifests["prior"]) != 392 or len(manifests["current"]) != 394:
        raise ValueError("source population differs")
    protocol = _load((root / "provenance.json").read_bytes())
    jsonschema.validate(protocol, schemas["provenance"])
    if (protocol["commits"] != COMMITS or protocol["blas_threads"] != 1
        or protocol["config_sha256"] != CONFIG
        or _sha((root / "config.yaml").read_bytes()) != CONFIG
        or protocol["source_manifest_sha256"] != MANIFESTS
        or protocol["launcher_sha256"] != _sha((root / "launcher.py.txt").read_bytes())
        or _sha((root / "inputs.json").read_bytes()) != INPUTS):
        raise ValueError("source, configuration or input provenance differs")
    inputs = {p["name"]: p for p in _load((root / "inputs.json").read_bytes())}
    attempts_record = _load((root / "attempts.json").read_bytes())
    attempts = attempts_record["attempts"]
    summary = _load((root / "summary.json").read_bytes())
    jsonschema.validate(attempts_record, schemas["attempts"])
    jsonschema.validate(summary, schemas["summary"])
    protocol_sha = _sha((root / "provenance.json").read_bytes())
    if ([a["variant"] for a in attempts] != ORDER or attempts != summary["attempts"]
        or attempts_record["provenance_sha256"] != protocol_sha
        or summary["provenance_sha256"] != protocol_sha):
        raise ValueError("attempt order or summary differs")
    reference = None
    pids: set[int] = set()
    for ordinal, attempt in enumerate(attempts, 1):
        variant = attempt["variant"]
        folder = root / attempt["folder"]
        result = _load((folder / "result.json.gz").read_bytes())
        measurement = _load((folder / "measurement.json").read_bytes())
        jsonschema.validate(result, schemas["result"])
        jsonschema.validate(measurement, schemas["measurement"])
        jsonschema.validate(result["payload"], schemas["calibration-result"])
        p = measurement["provenance"]
        if (attempt["folder"] != f"{ordinal:02d}-{variant}" or attempt["exit_code"] != 0
            or not 0 < attempt["api_seconds"] <= attempt["seconds"]
            or attempt["api_seconds"] != measurement["api_seconds"]
            or measurement["result_sha256"] != _sha((folder / "result.json.gz").read_bytes())
            or p != result["provenance"] or p["git_commit"] != COMMITS[variant]
            or p["variant"] != variant or p["pid"] in pids
            or p["source_manifest_sha256"] != MANIFESTS[variant]
            or p["run_provenance_sha256"] != protocol_sha
            or p["config_sha256"] != CONFIG or p["input_inventory_sha256"] != INPUTS
            or p["launcher_sha256"] != protocol["launcher_sha256"]):
            raise ValueError("worker provenance, timing or result differs")
        pids.add(p["pid"])
        if (len(p["verified_inputs"]) != 35
            or any(inputs.get(name) != pin for name, pin in p["verified_inputs"].items())):
            raise ValueError("verified public population differs")
        if (any(path not in manifests[variant] for path in p["imports"].values())
            or "calibrex.solvers.cuda_rgbd_joint_adapter" in p["imports"]
            or ("calibrex.data._voxel_geometry" in p["imports"]) != (variant == "current")):
            raise ValueError("worker imports or native capture path differ")
        payload = result["payload"]
        if (payload["run"]["config_sha256"] != CONFIG
            or payload["run"]["provenance"]["config_sha256"] != CONFIG):
            raise ValueError("result configuration binding differs")
        comparable = _canonical(_comparable(payload))
        if reference is not None and comparable != reference:
            raise ValueError("complete production results differ")
        reference = comparable
    medians = {v: statistics.median(a["seconds"] for a in attempts if a["variant"] == v)
               for v in COMMITS}
    if (summary["whole_seconds_median"] != medians
        or summary["whole_speedup"] != medians["prior"] / medians["current"]
        or summary["complete_normalized_results_exact"] is not True):
        raise ValueError("reported whole-worker comparison differs")
    _audit_traces(root, manifests, schemas)
    return {"production_workers": 6, "complete_results_exact": True,
            "solves_per_traced_run": 24, "jacobians_per_traced_run": 168,
            "residuals_per_traced_run": 1624, "fixed_population_events_per_run": 492,
            "whole_seconds_median": medians, "whole_speedup": summary["whole_speedup"],
            "runtime_candidate_adopted": False}


def _audit_traces(
    root: Path, manifests: dict[str, dict[str, str]], schemas: dict[str, Any],
) -> None:
    import jsonschema

    files = _members((root / "supplementary-traces.tar.gz").read_bytes())
    protocol = _load(files["provenance.json"])
    if (protocol["commits"] != COMMITS
        or protocol["launcher_sha256"] != _sha(files["launcher.py.txt"])
        or protocol["trace_worker_sha256"] != _sha(files["trace-worker.py.txt"])
        or protocol["input_inventory_sha256"] != INPUTS):
        raise ValueError("supplementary trace provenance differs")
    histories, populations, results = [], [], []
    for ordinal, variant in enumerate(("current", "prior"), 1):
        prefix = f"{ordinal:02d}-{variant}/"
        config_sha = _sha(files[variant + "/config.yaml"])
        trace_protocol = files[variant + "/protocol.json"]
        manifest = _load(files[variant + "/source-manifest.json"])
        if manifest != manifests[variant]:
            raise ValueError("supplementary runtime differs")
        history = _load(files[prefix + "history.json.gz"])["payload"]
        counts = Counter(event["kind"] for event in history["events"])
        if (len(history["solves"]) != 24 or counts != {"linearize": 168, "residuals": 1624}
            or len(history["frontend"]) != 150 or len(history["diagnostics"]) != 1498):
            raise ValueError("original numerical work population differs")
        histories.append(_canonical(history))
        events = []
        for name, data in sorted(files.items()):
            if not name.startswith(prefix):
                continue
            if name.endswith("population-history.json.gz"):
                record = _load(data)
                if (record["provenance"]["launcher_sha256"] != protocol["launcher_sha256"]
                    or record["provenance"]["protocol_sha256"] != _sha(trace_protocol)):
                    raise ValueError("fixed-population provenance differs")
                for event in record["events"]:
                    event.pop("seconds")
                    events.append(event)
            if name.endswith("process.json"):
                record = _load(data)
                if (record["provenance"]["git_commit"] != COMMITS[variant]
                    or any(manifest.get(item["path"]) != item["sha256"]
                           for item in record["payload"]["imports"].values())):
                    raise ValueError("supplementary process imports differ")
        if len(events) != 492:
            raise ValueError("fixed-population probe count differs")
        populations.append(_canonical(sorted(events, key=lambda e: e["window"])))
        payload = _load(files[prefix + "result.json.gz"])
        jsonschema.validate(payload, schemas["calibration-result"])
        value = _comparable(payload)
        run = value["run"]
        if run["config_sha256"] != config_sha or run["provenance"]["config_sha256"] != config_sha:
            raise ValueError("supplementary result configuration differs")
        run["config_sha256"] = run["provenance"]["config_sha256"] = "<verified identical bytes>"
        run["provenance"]["command"][2] = "<verified trace config>"
        results.append(_canonical(value))
    if histories[0] != histories[1] or populations[0] != populations[1] or results[0] != results[1]:
        raise ValueError("complete supplementary results or numerical traces differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), indent=2))


if __name__ == "__main__":
    main()
