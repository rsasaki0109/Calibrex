"""Archived full-point comparisons must reject rehashed work or identity changes."""

import gzip
import json
import shutil
from pathlib import Path

import pytest
import yaml
from tools import benchmark_joint_point_batches as bench

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/joint_lidar_point_batches"


def load(path):
    return json.loads(path.read_bytes())


def refresh(root):
    """Rehash transport bindings so mutations reach semantic checks."""
    protocol = load(root / "protocol.json")
    protocol["provenance"]["files"] = {
        name: bench.sha(root / name) for name in sorted(bench.PINNED)
    }
    bench.write(root / "protocol.json", protocol)
    digest = bench.sha(root / "protocol.json")
    attempts = load(root / "attempts.json")
    attempts["provenance"]["protocol_sha256"] = digest
    bench.write(root / "attempts.json", attempts)
    for attempt in attempts["payload"]["attempts"]:
        folder = root / attempt["folder"]
        history = load(folder / "history.json")
        history["provenance"]["protocol_sha256"] = digest
        bench.write(folder / "history.json", history)
        measurement = load(folder / "measurement.json")
        measurement["provenance"]["protocol_sha256"] = digest
        measurement["payload"]["result_sha256"] = bench.sha(folder / "result.json.gz")
        measurement["payload"]["history_sha256"] = bench.sha(folder / "history.json")
        bench.write(folder / "measurement.json", measurement)
    index = load(root / "index.json")
    index["provenance"]["protocol_sha256"] = digest
    index["files"] = {
        path.relative_to(root).as_posix(): {
            "sha256": bench.sha(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "index.json"
    }
    bench.write(root / "index.json", index)


@pytest.fixture
def archive(tmp_path):
    return Path(shutil.copytree(ARCHIVE, tmp_path / "archive"))


def test_published_complete_public_comparison_reaudits() -> None:
    result = bench.audit(ARCHIVE)
    assert result == load(ARCHIVE / "summary.json")
    assert result["payload"]["verified_runs"] == 4
    assert result["payload"]["joint_iterations"] == 40


@pytest.mark.parametrize("field,value", [("max_iterations", 1), ("convergence_tolerance", 0.01)])
def test_rehashed_solver_work_reduction_is_rejected(archive, field, value) -> None:
    config = yaml.safe_load((archive / "config.yaml").read_bytes())
    config["solver"][field] = value
    (archive / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    refresh(archive)
    with pytest.raises(ValueError, match="pipeline options differ"):
        bench.audit(archive)


def test_rehashed_point_cap_is_rejected(archive) -> None:
    config = yaml.safe_load((archive / "config.yaml").read_bytes())
    config["pipeline"]["factors"]["lidar_rig_point_to_plane"]["options"]["max_target_points"] = 512
    (archive / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    refresh(archive)
    with pytest.raises(ValueError, match="pipeline options differ"):
        bench.audit(archive)


def test_rehashed_public_point_inventory_is_rejected(archive) -> None:
    value = load(archive / "protocol.json")
    value["payload"]["inputs"][0]["point_count"] -= 1
    bench.write(archive / "protocol.json", value)
    refresh(archive)
    with pytest.raises(ValueError, match="PCD inventory differs"):
        bench.audit(archive)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("missing_event", "evaluation inventory differs"),
        ("jacobian_dimension", "Jacobian dimensions differ"),
        ("factor_ids", "factor IDs or order differ"),
        ("probe_values", "probe values differ"),
        ("array_digest", "numeric-array digest differ"),
    ],
)
def test_rehashed_numeric_history_changes_are_rejected(archive, mutation, pattern) -> None:
    path = archive / "01-prior/history.json"
    value = load(path)
    events = value["payload"]["events"]
    if mutation == "missing_event":
        events.pop()
    elif mutation == "jacobian_dimension":
        events[0]["jacobian"]["shape"][0] -= 1
    elif mutation == "factor_ids":
        events[0]["factor_ids_sha256"] = "0" * 64
    elif mutation == "probe_values":
        next(iter(events[-1]["values"].values()))[0] += 0.25
    else:
        events[0]["jacobian"]["sha256"] = "0" * 64
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match=pattern):
        bench.audit(archive)


def test_rehashed_result_factor_order_is_rejected(archive) -> None:
    path = archive / "01-prior/result.json.gz"
    value = json.loads(gzip.decompress(path.read_bytes()))
    ids = value["run"]["provenance"]["native_joint_slac_solver"]["train_factor_ids"]
    ids[0], ids[1] = ids[1], ids[0]
    path.write_bytes(gzip.compress(json.dumps(value, allow_nan=False).encode(), mtime=0))
    refresh(archive)
    with pytest.raises(ValueError, match="factor IDs or order differ"):
        bench.audit(archive)


@pytest.mark.parametrize(
    "location",
    [
        ("run", "git_commit"),
        ("run", "provenance", "git_commit"),
        ("solid_state", "provenance", "git_commit"),
    ],
)
def test_every_rehashed_pipeline_source_identity_is_verified(archive, location) -> None:
    path = archive / "01-prior/result.json.gz"
    value = json.loads(gzip.decompress(path.read_bytes()))
    item = value
    for part in location[:-1]:
        item = item[part]
    item[location[-1]] = "abcd"
    path.write_bytes(gzip.compress(json.dumps(value, allow_nan=False).encode(), mtime=0))
    refresh(archive)
    with pytest.raises(ValueError, match="pipeline Git identity differs"):
        bench.audit(archive)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("timing", "invalid timing scope"),
        ("import", "unfrozen module"),
        ("git", "Git identity or protocol binding differs"),
    ],
)
def test_rehashed_measurement_changes_are_rejected(archive, mutation, pattern) -> None:
    path = archive / "01-prior/measurement.json"
    value = load(path)
    if mutation == "timing":
        value["payload"]["api_seconds"] = 1e6
    elif mutation == "import":
        value["payload"]["imports"]["calibrex.graph.joint_factors"]["sha256"] = "0" * 64
    else:
        value["provenance"]["git_commit_short"] = "abcd"
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match=pattern):
        bench.audit(archive)


def test_saved_speedup_and_complete_inventory_are_verified(archive) -> None:
    value = load(archive / "summary.json")
    value["payload"]["whole_speedup"] = 1000.0
    bench.write(archive / "summary.json", value)
    refresh(archive)
    with pytest.raises(ValueError, match="saved summary differs"):
        bench.audit(archive)
    (archive / "unindexed.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory differs"):
        bench.audit(archive)
