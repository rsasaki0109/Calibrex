"""Audit direct public-example, parallel-control and native arithmetic timings."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_tum_diagnostic_execution import (  # noqa: E402
    _canonical,
    _comparable,
    _load,
    _members,
    _sha,
)

COMMITS = {
    "prior": "c8b048f13986ee9255b0d93ac3ac2913d8b14ece",
    "current": "5e5f10522ab792ed9713b2b5c101589156e60b99",
}
MANIFESTS = {
    "prior": "5658118b427355c7f5dbc0f8f7de2d81b882c2398e5d8a15dda86742a101f0e7",
    "current": "de54a3b9279c66cfed219f5c72aa77f04489d537aa6d4daa0c7cb08e312d1123",
}
CONFIGS = {
    "prior": "baf9517579d768c5a151f6374e3e2c5d247e94068d8243f1eb027e09eab44040",
    "parallel": "2b518c50e8f4a4ecc2044c33845fdbeb25f945e916631d18a09b1d7bc65ba723",
    "current": "2b518c50e8f4a4ecc2044c33845fdbeb25f945e916631d18a09b1d7bc65ba723",
}
EXAMPLES = {
    "prior": "ea677809a17141a9017365f77621af156c53a0de38ca9509afc6aa6c1a03c957",
    "current": "a2813c4d6a604605f03c608969cf166a4601333c22d35cd57d8fa1f965bd0666",
}
VARIANTS = {"prior": "prior", "parallel": "prior", "current": "current"}
ORDER = ["current", "prior", "parallel", "prior", "parallel", "current",
         "parallel", "current", "prior"]
INPUTS = "f1ec8837db3c70ddb173869a4362dd54a5a9da3dc8e6f9cc7ac59a7b14e2a108"
REFERENCE = "b8ab625548a220dc1814661c7a04241406c5f0bbb75ce54679569a86d4d7165c"
REFERENCE_CONFIG = "e736c92605a536ece26276f19d782ecbac332b8f4c6b05c1bab92504fee543db"


def _config(raw: bytes) -> tuple[dict[str, Any], int]:
    import yaml

    value = yaml.safe_load(raw)
    value["project"]["output_dir"] = "<output>"
    options = value["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"]
    workers = options.pop("window_workers", 1)
    return value, workers


def _result(payload: dict[str, Any], raw_config: bytes) -> str:
    import yaml

    value = _comparable(payload)
    cfg = yaml.safe_load(raw_config)
    workers = cfg["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"].get(
        "window_workers", 1,
    )
    run = value["run"]
    p = run["provenance"]
    if (run["config_sha256"] != _sha(raw_config) or p["config_sha256"] != _sha(raw_config)
        or p["native_tum_joint_slac_options"].pop("window_workers", 1) != workers):
        raise ValueError("result configuration or worker policy differs")
    execution = p.pop("native_tum_joint_slac_window_execution", None)
    if workers == 1:
        if execution != {
            "method": "native_tum_window_execution/v0.1", "backend": "serial",
            "requested_workers": 1, "effective_workers": 1,
            "window_start_indices": [60, 180, 300],
            "aggregation": "configured replication order; primary problem is reused",
        }:
            raise ValueError("serial example unexpectedly used window processes")
    elif execution != {
        "method": "native_tum_window_execution/v0.1", "backend": "process",
        "requested_workers": 3, "effective_workers": 3, "window_start_indices": [60, 180, 300],
        "aggregation": "configured replication order; primary problem is reused",
    }:
        raise ValueError("bounded window execution differs")
    command = p["command"]
    if command[:2] != ["calibrex", "calibrate"] or len(command) != 3:
        raise ValueError("calibration command differs")
    run["config_sha256"] = p["config_sha256"] = "<verified equivalent workload>"
    command[2] = "<config>"
    out = cfg["project"]["output_dir"]
    destination = PureWindowsPath(out) if ":" in out or "\\" in out else PurePosixPath(out)
    for name, relative in (("html_report", "report.html"),
                           ("rig_3d_viewer", "artifacts/rig_3d.html")):
        if value["artifacts"][name] != str(destination / relative):
            raise ValueError("artifact destination differs")
        value["artifacts"][name] = "<output>/" + relative
    return _canonical(value)


def audit(root: Path) -> dict[str, Any]:
    """Check all original workers against pinned source, input and prior results."""
    import jsonschema

    index = _load((root / "index.json").read_bytes())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(index["files"]) | {"index.json"}:
        raise ValueError("archive files differ")
    for name, pin in index["files"].items():
        raw = (root / name).read_bytes()
        if _sha(raw) != pin["sha256"] or len(raw) != pin["size_bytes"]:
            raise ValueError("archive bytes differ: " + name)
    schemas = _load((root / "record-schemas.json").read_bytes())
    for schema in schemas.values():
        jsonschema.Draft202012Validator.check_schema(schema)
    manifests = {}
    for variant in COMMITS:
        raw = (root / f"sources-{variant}-manifest.json").read_bytes()
        if _sha(raw) != MANIFESTS[variant]:
            raise ValueError("frozen source differs")
        members = _members((root / f"sources-{variant}.tar.gz").read_bytes())
        manifest = _load(raw)
        if ({n.removeprefix("src/"): _sha(b) for n, b in members.items()} != manifest
            or len(manifest) != (392 if variant == "prior" else 394)):
            raise ValueError("source capsule differs")
        manifests[variant] = manifest
    protocol = _load((root / "provenance.json").read_bytes())
    jsonschema.validate(protocol, schemas["provenance"])
    if (protocol["commits"] != COMMITS or protocol["variant_sources"] != VARIANTS
        or protocol["source_manifest_sha256"] != MANIFESTS or protocol["config_sha256"] != CONFIGS
        or protocol["example_sha256"] != EXAMPLES or protocol["blas_threads"] != 1
        or protocol["input_inventory_sha256"] != INPUTS
        or protocol["launcher_sha256"] != _sha((root / "launcher.py.txt").read_bytes())
        or _sha((root / "inputs.json").read_bytes()) != INPUTS):
        raise ValueError("production protocol differs")
    configurations = {v: (root / f"config-{v}.yaml").read_bytes() for v in VARIANTS}
    for v, raw in configurations.items():
        if _sha(raw) != CONFIGS[v] or _config(raw)[1] != (1 if v == "prior" else 3):
            raise ValueError("calibration configuration differs")
    if len({_canonical(_config(raw)[0]) for raw in configurations.values()}) != 1:
        raise ValueError("workload controls differ beyond window execution")
    import yaml

    examples = {}
    for v in EXAMPLES:
        raw = (root / f"example-{v}.yaml").read_bytes()
        if _sha(raw) != EXAMPLES[v]:
            raise ValueError("checked-in example differs")
        examples[v] = yaml.safe_load(raw)
    if (examples["current"]["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"].pop(
        "window_workers"
    ) != 3 or examples["prior"] != examples["current"]):
        raise ValueError("checked-in example changed more than worker count")
    reference_raw = (root / "reference-result.json.gz").read_bytes()
    reference_cfg = (root / "reference-config.yaml").read_bytes()
    if _sha(reference_raw) != REFERENCE or _sha(reference_cfg) != REFERENCE_CONFIG:
        raise ValueError("accepted preceding result differs")
    if _config(reference_cfg)[0] != _config(configurations["current"])[0]:
        raise ValueError("accepted reference workload differs")
    reference = _result(_load(reference_raw)["payload"], reference_cfg)
    inputs = {p["name"]: p for p in _load((root / "inputs.json").read_bytes())}
    attempts_record = _load((root / "attempts.json").read_bytes())
    summary = _load((root / "summary.json").read_bytes())
    for key, value in (("attempts", attempts_record), ("summary", summary)):
        jsonschema.validate(value, schemas[key])
    attempts = attempts_record["attempts"]
    protocol_sha = _sha((root / "provenance.json").read_bytes())
    if ([a["variant"] for a in attempts] != ORDER or attempts != summary["attempts"]
        or attempts_record["provenance_sha256"] != protocol_sha
        or summary["provenance_sha256"] != protocol_sha):
        raise ValueError("worker order or summary differs")
    pids: set[int] = set()
    for ordinal, attempt in enumerate(attempts, 1):
        variant = attempt["variant"]
        source = VARIANTS[variant]
        folder = root / attempt["folder"]
        result = _load((folder / "result.json.gz").read_bytes())
        measurement = _load((folder / "measurement.json").read_bytes())
        for key, value in (("result", result), ("measurement", measurement)):
            jsonschema.validate(value, schemas[key])
        jsonschema.validate(result["payload"], schemas["calibration-result"])
        p = measurement["provenance"]
        if (attempt["folder"] != f"{ordinal:02d}-{variant}" or attempt["exit_code"]
            or not 0 < attempt["api_seconds"] <= attempt["seconds"]
            or attempt["api_seconds"] != measurement["api_seconds"]
            or measurement["result_sha256"] != _sha((folder / "result.json.gz").read_bytes())
            or result["provenance"] != p or p["variant"] != variant
            or p["source_variant"] != source or p["git_commit"] != COMMITS[source]
            or p["source_manifest_sha256"] != MANIFESTS[source]
            or p["run_provenance_sha256"] != protocol_sha
            or p["config_sha256"] != CONFIGS[variant] or p["input_inventory_sha256"] != INPUTS
            or p["launcher_sha256"] != protocol["launcher_sha256"] or p["pid"] in pids):
            raise ValueError("worker provenance or timing differs")
        pids.add(p["pid"])
        if (len(p["verified_inputs"]) != 35
            or any(inputs.get(n) != pin for n, pin in p["verified_inputs"].items())
            or any(path not in manifests[source] or path not in (
                name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py",
            ) for name, path in p["imports"].items())
            or "calibrex.solvers.cuda_rgbd_joint_adapter" in p["imports"]):
            raise ValueError("input population or CPU import boundary differs")
        git_id = result["payload"]["run"]["git_commit"]
        if (not isinstance(git_id, str) or len(git_id) < 7
            or not COMMITS[source].startswith(git_id)
            or result["payload"]["run"]["provenance"]["git_commit"] != git_id):
            raise ValueError("result Git identity differs from the frozen runtime")
        if _result(result["payload"], configurations[variant]) != reference:
            raise ValueError("complete result differs from accepted reference")
    medians = {v: statistics.median(a["seconds"] for a in attempts if a["variant"] == v)
               for v in VARIANTS}
    ratios = {"direct_example_speedup": medians["prior"] / medians["current"],
              "parallel_only_speedup": medians["prior"] / medians["parallel"],
              "arithmetic_at_same_workers_speedup": medians["parallel"] / medians["current"]}
    if (summary["whole_seconds_median"] != medians
        or any(summary[k] != v for k, v in ratios.items())
        or summary["complete_normalized_results_exact"] is not True):
        raise ValueError("reported direct medians or ratios differ")
    _audit_traces(root, manifests, schemas, reference, configurations["current"])
    return {"production_workers": 9, "complete_results_and_histories_exact": True,
            "whole_seconds_median": medians, **ratios,
            "solves_per_traced_run": 24, "jacobians_per_traced_run": 168,
            "residuals_per_traced_run": 1624, "fixed_population_events_per_run": 492}


def _audit_traces(
    root: Path, manifests: dict[str, dict[str, str]], schemas: dict[str, Any],
    reference: str, production_config: bytes,
) -> None:
    from collections import Counter

    import jsonschema

    files = _members((root / "supplementary-traces.tar.gz").read_bytes())
    protocol = _load(files["provenance.json"])
    if (protocol["commits"] != COMMITS
        or protocol["launcher_sha256"] != _sha(files["launcher.py.txt"])
        or protocol["trace_worker_sha256"] != _sha(files["trace-worker.py.txt"])
        or protocol["input_inventory_sha256"] != INPUTS):
        raise ValueError("supplementary provenance differs")
    histories, populations = [], []
    for ordinal, variant in enumerate(("current", "prior"), 1):
        prefix = f"{ordinal:02d}-{variant}/"
        cfg = files[variant + "/config.yaml"]
        trace_protocol = files[variant + "/protocol.json"]
        if (_load(files[variant + "/source-manifest.json"]) != manifests[variant]
            or _config(cfg)[0] != _config(production_config)[0] or _config(cfg)[1] != 3):
            raise ValueError("supplementary source or workload differs")
        history = _load(files[prefix + "history.json.gz"])["payload"]
        if (len(history["solves"]) != 24
            or Counter(e["kind"] for e in history["events"])
            != {"linearize": 168, "residuals": 1624}
            or len(history["frontend"]) != 150 or len(history["diagnostics"]) != 1498):
            raise ValueError("original numerical work count differs")
        histories.append(_canonical(history))
        events = []
        for name, data in sorted(files.items()):
            if not name.startswith(prefix):
                continue
            if name.endswith("population-history.json.gz"):
                record = _load(data)
                if (record["provenance"]["launcher_sha256"] != protocol["launcher_sha256"]
                    or record["provenance"]["protocol_sha256"] != _sha(trace_protocol)):
                    raise ValueError("population provenance differs")
                for event in record["events"]:
                    event.pop("seconds")
                    events.append(event)
            if name.endswith("process.json"):
                record = _load(data)
                if (record["provenance"]["git_commit"] != COMMITS[variant]
                    or any(manifests[variant].get(v["path"]) != v["sha256"]
                           for v in record["payload"]["imports"].values())):
                    raise ValueError("trace imports differ")
        if len(events) != 492:
            raise ValueError("fixed population scoring count differs")
        populations.append(_canonical(sorted(events, key=lambda e: e["window"])))
        payload = _load(files[prefix + "result.json.gz"])
        jsonschema.validate(payload, schemas["calibration-result"])
        if _result(payload, cfg) != reference:
            raise ValueError("complete supplementary result differs")
    if histories[0] != histories[1] or populations[0] != populations[1]:
        raise ValueError("ordered numerical and scoring histories differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), indent=2))


if __name__ == "__main__":
    main()
