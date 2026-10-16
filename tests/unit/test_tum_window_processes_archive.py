"""Re-audit complete process traces and reject consistently rehashed tampering."""

import gzip
import json
import shutil
from pathlib import Path

import pytest
import yaml
from tools import benchmark_tum_window_processes as b

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/tum_window_processes"


def load(path):
    return json.loads(path.read_bytes())


def compressed(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def save(path, value):
    raw = (json.dumps(value, allow_nan=False) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def refresh(root):
    """Rebind hashes while preserving the independent semantic checks."""
    for attempt in load(root / "attempts.json")["payload"]["attempts"]:
        folder = root / attempt["folder"]
        measurement = load(folder / "measurement.json")
        for entry in measurement["payload"]["processes"]:
            path = folder / entry["path"]
            process = load(path)
            history = path.parent / (
                "parent-history.json.gz" if path.parent == folder else "history.json.gz"
            )
            process["payload"]["history_sha256"] = b.sha(history)
            save(path, process)
            entry["sha256"] = b.sha(path)
        measurement["payload"]["history_sha256"] = b.sha(folder / "history.json.gz")
        measurement["payload"]["result_sha256"] = b.sha(folder / "result.json.gz")
        save(folder / "measurement.json", measurement)
    value = load(root / "index.json")
    value["files"] = {
        path.relative_to(root).as_posix(): {
            "sha256": b.sha(path), "size_bytes": path.stat().st_size,
        }
        for path in root.rglob("*") if path.is_file() and path.name != "index.json"
    }
    save(root / "index.json", value)


@pytest.fixture
def archive(tmp_path):
    return Path(shutil.copytree(ARCHIVE, tmp_path / "archive"))


def test_published_complete_window_process_comparison_reaudits():
    report = b.audit(ARCHIVE)
    assert report == load(ARCHIVE / "summary.json")
    assert report["payload"]["verified_runs"] == 4
    assert report["payload"]["solve_count"] == 24
    assert report["payload"]["jacobian_calls"] == 168
    assert report["payload"]["residual_calls"] == 1624
    assert report["payload"]["frontend_evaluations_per_run"] == dict.fromkeys(b.FRONTEND, 75)


def test_initial_redirector_protocol_remains_rejected_and_is_preserved():
    with pytest.raises(ValueError, match="child process ownership differs"):
        b.audit(ARCHIVE / "initial-redirector-run")
    assert len(load(ARCHIVE / "initial-redirector-run/attempts.json")["payload"]["attempts"]) == 4


@pytest.mark.parametrize("change", ["iterations", "points", "windows", "workers"])
def test_complete_work_configuration_is_pinned(archive, change):
    path = archive / "config.yaml"
    value = yaml.safe_load(path.read_bytes())
    if change == "iterations":
        value["solver"]["max_iterations"] = 1
    else:
        key, replacement = {
            "points": ("query_points_per_frame", 32),
            "windows": ("replication_start_indices", [180]),
            "workers": ("window_workers", 1),
        }[change]
        value["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"][key] = replacement
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    # Rehash every protocol binding, so rejection tests the pinned recipe.
    protocol = load(archive / "protocol.json")
    protocol["provenance"]["files"]["config.yaml"] = b.sha(path)
    save(archive / "protocol.json", protocol)
    refresh(archive)
    with pytest.raises(ValueError, match="complete pipeline options differ"):
        b.audit(archive)


@pytest.mark.parametrize("change", ["imports", "parent", "window", "count", "timing", "wire"])
def test_child_process_lineage_and_work_are_independently_verified(archive, change):
    folder = archive / "01-current/children/60"
    path = folder / "process.json"
    value = load(path)
    info = value["payload"]
    if change == "imports":
        info["imports"]["calibrex.graph.joint_factors"]["sha256"] = "0" * 64
    elif change == "parent":
        info["parent_pid"] += 1
    elif change == "window":
        history = compressed(folder / "history.json.gz")
        history["payload"]["events"][0]["window"] = 180
        save(folder / "history.json.gz", history)
    elif change == "count":
        info["solve_count"] -= 1
    elif change == "timing":
        info["api_seconds"] = info["optimizer_seconds"] / 2
    else:
        info["response_sha256"] = None
    save(path, value)
    refresh(archive)
    with pytest.raises(
        ValueError, match=r"unfrozen|ownership|window identity|process work|fingerprints"
    ):
        b.audit(archive)


@pytest.mark.parametrize("change", ["missing", "array", "candidate", "pixel", "sample"])
def test_aggregate_history_is_rebuilt_from_complete_raw_process_events(archive, change):
    path = archive / "01-current/history.json.gz"
    value = compressed(path)
    if change == "missing":
        value["payload"]["events"].pop()
    elif change == "array":
        value["payload"]["events"][0]["residual"]["sha256"] = "0" * 64
    elif change == "candidate":
        next(e for e in value["payload"]["diagnostics"] if "candidate_sha256" in e)[
            "candidate_sha256"
        ] = "0" * 64
    else:
        key = "pixels_sha256" if change == "pixel" else "points_sha256"
        next(e for e in value["payload"]["frontend"] if key in e)[key] = "0" * 64
    save(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="canonical history differs from raw process events"):
        b.audit(archive)


@pytest.mark.parametrize("change", ["array", "diagnostic", "frontend"])
def test_rehashed_raw_and_aggregate_numeric_changes_cannot_agree(archive, change):
    paths = [archive / "01-prior/history.json.gz", archive / "01-prior/parent-history.json.gz"]
    for path in paths:
        value = compressed(path)
        if change == "array":
            next(e for e in value["payload"]["events"] if e["window"] == 60)[
                "residual"
            ]["sha256"] = "0" * 64
        elif change == "diagnostic":
            value["payload"]["diagnostics"][0]["state_sha256"] = "0" * 64
        else:
            next(e for e in value["payload"]["frontend"] if e["window"] == 60)[
                "pixels_sha256"
            ] = "0" * 64
        save(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="numeric-array digest differ"):
        b.audit(archive)


@pytest.mark.parametrize("change", ["count", "stage", "optimizer", "inventory", "policy"])
def test_aggregate_process_claims_are_recomputed(archive, change):
    path = archive / "01-current/measurement.json"
    value = load(path)
    info = value["payload"]
    if change == "count":
        info["rgbd_plan_accepts"] -= 1
    elif change == "stage":
        info["stage_seconds"]["_xyz_pair_reassociation_runs"] += 1.0
    elif change == "optimizer":
        info["optimizer_seconds"] += 1.0
    elif change == "inventory":
        info["processes"].pop(0)
    else:
        result_path = archive / "01-current/result.json.gz"
        result = compressed(result_path)
        result["run"]["provenance"]["native_tum_joint_slac_window_execution"].update(
            effective_workers=8
        )
        save(result_path, result)
    save(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match=r"aggregate process|process inventory|execution policy"):
        b.audit(archive)


def test_complete_result_speed_and_file_inventory_are_verified(archive):
    path = archive / "summary.json"
    summary = load(path)
    summary["payload"]["speedup"]["whole"] = 1000.0
    save(path, summary)
    refresh(archive)
    with pytest.raises(ValueError, match="saved summary differs"):
        b.audit(archive)
    (archive / "unexpected.txt").write_text("unindexed", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory differs"):
        b.audit(archive)


def test_rehashed_final_calibration_metric_is_still_compared(archive):
    path = archive / "01-current/result.json.gz"
    value = compressed(path)
    value["metrics"]["tum_joint_point_to_plane_rmse_m"]["train"] += 0.1
    save(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="numeric-array digest differ"):
        b.audit(archive)
