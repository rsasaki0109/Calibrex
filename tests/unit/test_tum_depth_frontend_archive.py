"""Re-audit the complete TUM workload and reject rehashed evidence changes."""

import gzip
import json
import shutil
from pathlib import Path

import pytest
import yaml
from tools import benchmark_tum_depth_frontend as bench

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/tum_depth_frontend"


def load(path):
    return json.loads(path.read_bytes())


def compressed(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def save_compressed(path, value):
    path.write_bytes(gzip.compress((json.dumps(value, allow_nan=False) + "\n").encode(), mtime=0))


def refresh(root):
    """Refresh transport bindings, retaining the independent semantic checks."""
    protocol = load(root / "protocol.json")
    protocol["provenance"]["files"] = {
        name: bench.sha(root / name) for name in sorted(bench.PINNED)
    }
    protocol["payload"]["inputs_sha256"] = bench.sha(root / "inputs.json")
    bench.write(root / "protocol.json", protocol)
    digest = bench.sha(root / "protocol.json")
    attempts = load(root / "attempts.json")
    attempts["provenance"]["protocol_sha256"] = digest
    bench.write(root / "attempts.json", attempts)
    for attempt in attempts["payload"]["attempts"]:
        folder = root / attempt["folder"]
        history = compressed(folder / "history.json.gz")
        history["provenance"]["protocol_sha256"] = digest
        save_compressed(folder / "history.json.gz", history)
        progress = load(folder / "progress.json")
        progress["provenance"]["protocol_sha256"] = digest
        bench.write(folder / "progress.json", progress)
        measurement = load(folder / "measurement.json")
        measurement["provenance"]["protocol_sha256"] = digest
        measurement["payload"]["result_sha256"] = bench.sha(folder / "result.json.gz")
        measurement["payload"]["history_sha256"] = bench.sha(folder / "history.json.gz")
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


def test_published_complete_tum_comparison_reaudits():
    result = bench.audit(ARCHIVE)
    assert result == load(ARCHIVE / "summary.json")
    assert result["payload"]["verified_runs"] == 4
    assert result["payload"]["solve_count"] == 24


@pytest.mark.parametrize(
    "change", ["iterations", "points", "windows", "local_rotations", "workers"]
)
def test_rehashed_work_reduction_is_rejected(archive, change):
    path = archive / "config.yaml"
    value = yaml.safe_load(path.read_bytes())
    options = value["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"]
    if change == "iterations":
        value["solver"]["max_iterations"] = 1
    elif change == "points":
        options["query_points_per_frame"] = 512
    elif change == "windows":
        options["replication_start_indices"] = [180]
    elif change == "workers":
        options["window_workers"] = 3
    else:
        options["xyz_lattice_local_rotation_updates"] = 0
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    refresh(archive)
    with pytest.raises(ValueError, match="complete pipeline options differ"):
        bench.audit(archive)


def test_rehashed_public_depth_fingerprint_is_rejected(archive):
    path = archive / "inputs.json"
    value = load(path)
    value[5]["sha256"] = "0" * 64
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="public depth input inventory differs"):
        bench.audit(archive)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("missing_event", "array evaluation work differs"),
        ("dimension", "numeric array dimensions differ"),
        ("array", "numeric-array digest differ"),
        ("factor_ids", "numeric-array digest differ"),
        ("values", "numeric-array digest differ"),
        ("primary", "primary optimizer trace differs"),
    ],
)
def test_rehashed_numeric_history_changes_are_rejected(archive, mutation, pattern):
    path = archive / "01-prior/history.json.gz"
    value = compressed(path)
    event = value["payload"]["events"][0]
    if mutation == "missing_event":
        value["payload"]["events"].pop()
    elif mutation == "dimension":
        event["jacobian"]["shape"][0] -= 1
    elif mutation == "array":
        event["jacobian"]["sha256"] = "0" * 64
    elif mutation == "factor_ids":
        event["factor_ids_sha256"] = "0" * 64
    elif mutation == "values":
        event["values_sha256"] = "0" * 64
    else:
        value["payload"]["solves"][0]["result_sha256"] = "0" * 64
    save_compressed(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match=pattern):
        bench.audit(archive)


@pytest.mark.parametrize("location", [("run", "git_commit"), ("run", "provenance", "git_commit")])
def test_every_rehashed_pipeline_source_identity_is_verified(archive, location):
    path = archive / "01-prior/result.json.gz"
    value = compressed(path)
    target = value
    for part in location[:-1]:
        target = target[part]
    target[location[-1]] = "abcdef0"
    save_compressed(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="pipeline Git identity differs"):
        bench.audit(archive)


def test_rehashed_complete_result_metric_is_verified(archive):
    path = archive / "01-prior/result.json.gz"
    value = compressed(path)
    value["metrics"]["tum_joint_point_to_plane_rmse_m"]["train"] += 0.1
    save_compressed(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="complete results or every numeric-array digest differ"):
        bench.audit(archive)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("timing", "invalid timing scope"),
        ("import", "unfrozen module"),
        ("git", "Git identity or protocol binding differs"),
    ],
)
def test_rehashed_measurement_changes_are_rejected(archive, mutation, pattern):
    path = archive / "01-prior/measurement.json"
    value = load(path)
    if mutation == "timing":
        value["payload"]["api_seconds"] = 1e6
    elif mutation == "import":
        value["payload"]["imports"]["calibrex.graph.joint_factors"]["sha256"] = "0" * 64
    else:
        value["provenance"]["git_commit_short"] = "abcdef0"
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match=pattern):
        bench.audit(archive)


def test_rehashed_optimizer_timing_is_verified(archive):
    path = archive / "01-prior/progress.json"
    value = load(path)
    value["payload"]["solve_seconds"][0] += 1.0
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="optimizer timing or work differs"):
        bench.audit(archive)


def test_saved_speedup_and_complete_file_inventory_are_verified(archive):
    path = archive / "summary.json"
    value = load(path)
    value["payload"]["speedup"]["whole"] = 1000.0
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="saved summary differs"):
        bench.audit(archive)
    (archive / "unindexed.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file inventory differs"):
        bench.audit(archive)


@pytest.mark.parametrize("field", ["rgbd_plan_requests", "rgbd_plan_accepts"])
def test_rehashed_array_execution_counts_are_verified(archive, field):
    path = archive / "01-current/measurement.json"
    value = load(path)
    value["payload"][field] = 0
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="array evaluation work differs"):
        bench.audit(archive)


@pytest.mark.parametrize("kind", ["missing", "candidate", "value", "state"])
def test_rehashed_diagnostic_work_and_values_are_verified(archive, kind):
    path = archive / "01-prior/history.json.gz"
    value = compressed(path)
    events = value["payload"]["diagnostics"]
    if kind == "missing":
        events.pop()
    elif kind in ("candidate", "value"):
        event = next(event for event in events if event["kind"] == "_shared_holdout_objective")
        event["candidate_sha256"] = "0" * 64 if kind == "candidate" else event["candidate_sha256"]
        if kind == "value":
            event["value_hex"] = "0x1.0p+0"
    else:
        next(event for event in events if "state_sha256" in event)["state_sha256"] = "0" * 64
    save_compressed(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match=r"diagnostic evaluation work|numeric-array digest differ"):
        bench.audit(archive)


@pytest.mark.parametrize("kind", ["timing", "factory"])
def test_rehashed_diagnostic_timing_and_factory_counts_are_verified(archive, kind):
    path = archive / "01-prior/measurement.json"
    value = load(path)
    if kind == "timing":
        value["payload"]["stage_seconds"]["_iterative_reassociation_runs"] = 1e6
    else:
        value["payload"]["factory_calls"]["_spatial_factor"] += 1
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(
        ValueError, match=r"diagnostic timing differs|repeated factory work differs"
    ):
        bench.audit(archive)


@pytest.mark.parametrize("change", ["exceeds_evaluations", "repeat", "changed_all"])
def test_rehashed_plan_preparation_counts_are_verified(archive, change):
    names = ("01-current", "02-current") if change == "changed_all" else ("01-current",)
    for name in names:
        path = archive / name / "measurement.json"
        value = load(path)
        if change == "exceeds_evaluations":
            value["payload"]["rgbd_plan_builds"] = value["payload"]["rgbd_plan_accepts"] + 1
        elif change == "repeat":
            value["payload"]["rgbd_plan_builds"] += 1
        else:
            value["payload"]["rgbd_plan_builds"] = (
                load(archive / "01-prior/measurement.json")["payload"]["rgbd_plan_builds"] + 1
            )
        bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match=r"preparation work differs|unchanged graph preparation"):
        bench.audit(archive)


@pytest.mark.parametrize("change", ["missing", "pixels", "points", "dimensions", "path"])
def test_every_decoded_pixel_and_sample_population_is_verified(archive, change):
    path = archive / "01-prior/history.json.gz"
    value = compressed(path)
    events = value["payload"]["frontend"]
    if change == "missing":
        events.pop()
    elif change == "pixels":
        events[0]["pixels_sha256"] = "0" * 64
    elif change == "points":
        event = next(e for e in events if e["kind"] == "sample_tum_depth_points")
        event["point_count"] += 1
    elif change == "dimensions":
        events[0]["width"] += 1
    else:
        event = next(e for e in events if e["kind"] == "read_tum_depth_png")
        event["path"] = "depth/changed.png"
    save_compressed(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match=r"frontend .* differ|numeric-array digest differ"):
        bench.audit(archive)


def test_frontend_timing_is_verified(archive):
    path = archive / "01-prior/measurement.json"
    value = load(path)
    value["payload"]["frontend_seconds"]["read_tum_depth_png"] = 1e6
    bench.write(path, value)
    refresh(archive)
    with pytest.raises(ValueError, match="frontend timing differs"):
        bench.audit(archive)
