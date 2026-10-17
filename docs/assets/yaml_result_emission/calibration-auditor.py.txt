"""Audit supplementary complete-calibration checks for safe YAML emission."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

import jsonschema
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_radar_doppler_batches import sources
from tools.benchmark_tum_window_processes import comparable
from tools.benchmark_yaml_emission import INPUT, sha


def load(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def _normalize_result(value: dict, config_path: Path, config: dict) -> dict:
    fingerprint = sha(config_path)
    if (value["run"]["config_sha256"] != fingerprint
        or value["run"]["provenance"]["config_sha256"] != fingerprint):
        raise ValueError("calibration config hash differs")
    result = comparable(value)
    base = config["project"]["output_dir"].replace("\\", "/").rstrip("/")
    for name, path in result["artifacts"].items():
        if path is not None:
            path = path.replace("\\", "/")
            if not path.startswith(base + "/"):
                raise ValueError("calibration artifact output path differs")
            result["artifacts"][name] = "<verified output>/" + path[len(base) + 1:]
    result["run"]["config_sha256"] = "<verified config bytes>"
    provenance = result["run"]["provenance"]
    provenance["config_sha256"] = "<verified config bytes>"
    command = provenance["command"]
    benchmark_root = base.rsplit("/", 1)[0]
    variant = config_path.parent.name
    relative = f"{variant}/config.yaml" if variant in ("prior", "current") else "config.yaml"
    if (len(command) != 3 or command[:2] != ["calibrex", "calibrate"]
        or command[2].replace("\\", "/") != benchmark_root + "/" + relative):
        raise ValueError("calibration command differs")
    provenance["command"][2] = "<verified config path>"
    return result


def audit(root: Path) -> dict:
    """Retain full original array work, controls, input pixels and process ownership."""
    folder = root / "calibration"
    expected_commits = load(root / "protocol.json")["payload"]["commits"]
    provenance = load(folder / "provenance.json")
    if (provenance["commits"] != expected_commits
        or provenance["launcher_sha256"] != sha(folder / "launcher.py.txt")
        or provenance["trace_worker_sha256"] != sha(folder / "trace-worker.py.txt")
        or provenance["input_inventory_sha256"] != sha(INPUT.parent.parent / "inputs.json")):
        raise ValueError("calibration launcher or source lineage differs")
    expected_history = load(INPUT.parent / "history.json.gz")["payload"]
    reference_config = yaml.safe_load((INPUT.parent.parent / "config.yaml").read_bytes())
    reference = _normalize_result(
        load(INPUT), INPUT.parent.parent / "config.yaml", reference_config
    )
    manifests = {v: sources(root, v) for v in expected_commits}
    attempts = load(folder / "attempts.json")["attempts"]
    names = ["01-current", "02-prior", "03-prior", "04-current"]
    if [a["folder"] for a in attempts] != names or any(a["exit_code"] for a in attempts):
        raise ValueError("calibration attempts are incomplete, failed or reordered")
    populations, times = [], {v: [] for v in expected_commits}
    validate = jsonschema.Draft202012Validator(load(folder / "worker-schema.json")).validate
    for attempt in attempts:
        variant = attempt["variant"]
        run = folder / attempt["folder"]
        protocol_path = folder / variant / "protocol.json"
        protocol = load(protocol_path)
        if (protocol["payload"]["commits"] != {"current": expected_commits[variant]}
            or protocol["provenance"]["launcher_sha256"] != provenance["launcher_sha256"]
            or protocol["provenance"]["source_manifest_sha256"]
            != sha(root / f"sources-{variant}-manifest.json")):
            raise ValueError("calibration source manifest differs")
        config_path = folder / variant / "config.yaml"
        config = yaml.safe_load(config_path.read_bytes())
        expected_config = yaml.safe_load((INPUT.parent.parent / "config.yaml").read_bytes())
        expected_config["project"]["output_dir"] = config["project"]["output_dir"]
        if config != expected_config:
            raise ValueError("complete calibration scope differs")
        primary = config["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"]
        measurement = load(run / "measurement.json")
        validate(measurement)
        info, lineage = measurement["payload"], measurement["provenance"]
        if (lineage["protocol_sha256"] != sha(protocol_path)
            or lineage["git_commit"] != expected_commits[variant]
            or not expected_commits[variant].startswith(lineage["git_commit_short"])):
            raise ValueError("calibration actual Git identity differs")
        if [p["path"] for p in info["processes"]] != [
            "children/60/process.json", "children/300/process.json", "process.json"
        ]:
            raise ValueError("calibration process inventory differs")
        processes, raw_histories, pop = [], [], []
        for entry in info["processes"]:
            path = run / entry["path"]
            if sha(path) != entry["sha256"]:
                raise ValueError("calibration process fingerprint differs")
            process = load(path)
            validate(process)
            data = process["payload"]
            parent = path.parent == run
            history_path = path.parent / ("parent-history.json.gz" if parent else "history.json.gz")
            history = load(history_path)
            validate(history)
            if (process["provenance"] != lineage or history["provenance"] != lineage
                or sha(history_path) != data["history_sha256"]
                or data["role"] != ("parent" if parent else "child")):
                raise ValueError("calibration raw history lineage differs")
            if any(manifests[variant].get(p["path"]) != p["sha256"]
                   for p in data["imports"].values()):
                raise ValueError("calibration imported an unfrozen module")
            expected_window = primary["frame_start_index"] if parent else int(path.parent.name)
            h = history["payload"]
            if (data["solve_count"] != len(h["solves"])
                or data["rgbd_plan_requests"] != len(h["events"])
                or data["rgbd_plan_accepts"] != data["rgbd_plan_requests"]
                or any(e["window"] != expected_window for events in h.values() for e in events)):
                raise ValueError("calibration raw work or window differs")
            probe = load(path.parent / "population-history.json.gz")
            if (probe["provenance"]["launcher_sha256"] != provenance["launcher_sha256"]
                or probe["provenance"]["protocol_sha256"] != sha(protocol_path)
                or probe["provenance"]["pid"] != data["pid"]
                or probe["provenance"]["parent_pid"] != data["parent_pid"]):
                raise ValueError("calibration fixed-population process lineage differs")
            for event in probe["events"]:
                event = dict(event)
                seconds = event.pop("seconds")
                if not math.isfinite(seconds) or not 0 < seconds <= data["api_seconds"]:
                    raise ValueError("calibration probe time differs")
                pop.append(event)
            processes.append(data)
            raw_histories.append(h)
        parent = processes[-1]
        if any(p["parent_pid"] != parent["pid"] for p in processes[:-1]):
            raise ValueError("calibration child ownership differs")
        combined = {
            key: sorted((e for h in raw_histories for e in h[key]), key=lambda e: e["window"])
            for key in ("events", "solves", "diagnostics", "frontend")
        }
        history_path = run / "history.json.gz"
        if (sha(history_path) != info["history_sha256"]
            or load(history_path)["payload"] != combined or combined != expected_history):
            raise ValueError("calibration original numeric work differs")
        result_path = run / "result.json.gz"
        result = load(result_path)
        execution = result["run"]["provenance"]["native_tum_joint_slac_window_execution"]
        if (sha(result_path) != info["result_sha256"] or execution != {
            "method": "native_tum_window_execution/v0.1", "backend": "process",
            "requested_workers": 3, "effective_workers": 3,
            "window_start_indices": primary["replication_start_indices"],
            "aggregation": "configured replication order; primary problem is reused",
        }):
            raise ValueError("calibration output or execution policy differs")
        if (not expected_commits[variant].startswith(result["run"]["git_commit"])
            or result["run"]["git_commit"] != result["run"]["provenance"]["git_commit"]):
            raise ValueError("calibration result Git identity differs")
        if _normalize_result(result, config_path, config) != reference:
            raise ValueError("complete calibration result differs")
        pop.sort(key=lambda e: e["window"])
        if len(pop) != 492:
            raise ValueError("calibration fixed-population work differs")
        populations.append(pop)
        if not math.isfinite(attempt["seconds"]) or attempt["seconds"] < info["api_seconds"]:
            raise ValueError("calibration whole-worker time differs")
        times[variant].append(attempt["seconds"])
    if any(p != populations[0] for p in populations):
        raise ValueError("calibration fixed-population results differ")
    medians = {v: statistics.median(values) for v, values in times.items()}
    saved = load(folder / "summary.json")
    if saved["whole_seconds_median"] != medians or saved["whole_speedup"] != (
        medians["prior"] / medians["current"]
    ):
        raise ValueError("calibration reported timing differs")
    return {"verified_runs": len(attempts), "complete_results_exact": True,
            "all_original_array_histories_exact": True, "fixed_population_events_exact": 492,
            "whole_seconds_median": medians, "whole_speedup": medians["prior"] / medians["current"],
            "provenance": {"export_protocol_sha256": sha(root / "protocol.json"),
                           "calibration_attempts_sha256": sha(folder / "attempts.json")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
