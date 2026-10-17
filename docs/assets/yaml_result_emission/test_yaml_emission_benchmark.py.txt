"""Complete export evidence and independent full-calibration lineage checks."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from tools.audit_tum_yaml_emission import audit as audit_calibration
from tools.benchmark_yaml_emission import audit

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/assets/yaml_result_emission"


@pytest.fixture
def archive_copy(tmp_path):
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def save(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def reindex(root):
    value = load(root / "index.json")
    value["files"] = {
        p.relative_to(root).as_posix(): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size,
        }
        for p in root.rglob("*") if p.is_file() and p.name != "index.json"
    }
    save(root / "index.json", value)


def test_complete_exports_and_calibration_are_reauditable():
    assert audit(ARCHIVE)["payload"]["full_exports"] == 12
    assert audit_calibration(ARCHIVE)["all_original_array_histories_exact"]
    workers = [load(path) for path in ARCHIVE.glob("0*/worker.json")]
    assert len(workers) == 6
    assert len({method["sha256"] for worker in workers
                for method in worker["payload"]["methods"].values()}) == 1


@pytest.mark.parametrize("change", ["missing", "order", "failed"])
def test_all_export_attempts_are_required_after_reindexing(archive_copy, change):
    path = archive_copy / "attempts.json"
    value = load(path)
    attempts = value["payload"]["attempts"]
    if change == "missing":
        attempts.pop()
    elif change == "order":
        attempts[0], attempts[1] = attempts[1], attempts[0]
    else:
        attempts[0]["exit_code"] = 1
    save(path, value)
    reindex(archive_copy)
    with pytest.raises(ValueError, match="incomplete, failed or reordered"):
        audit(archive_copy)


@pytest.mark.parametrize("change", ["drop", "grade", "order", "float"])
def test_complete_yaml_values_survive_hash_rebinding(archive_copy, change):
    path = archive_copy / "01-current/write.yaml.gz"
    data = yaml.safe_load(gzip.decompress(path.read_bytes()))
    if change == "drop":
        data["metrics"].pop(next(iter(data["metrics"])))
    elif change == "grade":
        data["quality"]["grade"] = "pass" if data["quality"]["grade"] != "pass" else "fail"
    elif change == "order":
        data = dict(reversed(list(data.items())))
    else:
        data["metrics"]["tum_joint_point_to_plane_rmse_m"]["holdout"] += 0.1
    raw = yaml.safe_dump(data, sort_keys=False, allow_unicode=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0))
    worker_path = path.parent / "worker.json"
    worker = load(worker_path)
    worker["payload"]["methods"]["write"].update(
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )
    save(worker_path, worker)
    reindex(archive_copy)
    with pytest.raises(ValueError, match="complete YAML content"):
        audit(archive_copy)


@pytest.mark.parametrize("change", ["import", "backend", "time"])
def test_worker_metadata_is_checked_after_reindexing(archive_copy, change):
    path = archive_copy / "01-current/worker.json"
    value = load(path)
    if change == "import":
        value["payload"]["imports"]["calibrex.core.io"]["sha256"] = "0" * 64
    elif change == "backend":
        value["payload"]["native_emissions"] = 0
    else:
        value["payload"]["methods"]["atomic"]["seconds"] = 1e9
    save(path, value)
    reindex(archive_copy)
    with pytest.raises(
        ValueError, match=r"unfrozen source|backend or environment|invalid writer time"
    ):
        audit(archive_copy)


def test_recomputed_medians_reject_a_reported_speedup(archive_copy):
    path = archive_copy / "summary.json"
    value = load(path)
    value["payload"]["speedup"]["whole"] += 1.0
    save(path, value)
    reindex(archive_copy)
    with pytest.raises(ValueError, match="reported timing differs"):
        audit(archive_copy)


def test_unindexed_file_is_rejected(archive_copy):
    (archive_copy / "extra.txt").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="file inventory"):
        audit(archive_copy)


def test_raw_calibration_arrays_cannot_be_replaced_together(archive_copy):
    folder = archive_copy / "calibration/01-current"
    raw_path = folder / "parent-history.json.gz"
    raw = load(raw_path)
    raw["payload"]["events"][0]["residual"]["sha256"] = "0" * 64
    save(raw_path, raw)
    process_path = folder / "process.json"
    process = load(process_path)
    process["payload"]["history_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    save(process_path, process)
    measurement_path = folder / "measurement.json"
    measurement = load(measurement_path)
    entry = next(p for p in measurement["payload"]["processes"] if p["path"] == "process.json")
    entry["sha256"] = hashlib.sha256(process_path.read_bytes()).hexdigest()
    save(measurement_path, measurement)
    reindex(archive_copy)
    with pytest.raises(ValueError, match="original numeric work"):
        audit_calibration(archive_copy)


def test_calibration_child_process_ownership_is_required(archive_copy):
    folder = archive_copy / "calibration/01-current"
    path = folder / "children/60/process.json"
    value = load(path)
    value["payload"]["parent_pid"] += 1
    save(path, value)
    probe_path = path.parent / "population-history.json.gz"
    probe = load(probe_path)
    probe["provenance"]["parent_pid"] = value["payload"]["parent_pid"]
    save(probe_path, probe)
    measurement_path = folder / "measurement.json"
    measurement = load(measurement_path)
    entry = next(p for p in measurement["payload"]["processes"]
                 if p["path"] == "children/60/process.json")
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(measurement_path, measurement)
    reindex(archive_copy)
    with pytest.raises(ValueError, match="child ownership"):
        audit_calibration(archive_copy)
