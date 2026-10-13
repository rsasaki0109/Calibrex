"""Hashing overlaps computation while changed inputs invalidate the result."""

from pathlib import Path
from threading import Event

import pytest
import yaml

from calibrex.core.exceptions import DatasetError
from calibrex.core.provenance import sha256_path
from calibrex.pipelines import online


def _config(tmp_path):
    dataset = tmp_path / "bag"
    dataset.mkdir()
    (dataset / "recording.db3").write_bytes(b"captured input")
    payload = yaml.safe_load(Path(
        "examples/public_datasets/glim_versatile/online_continuous_time_benchmark_config.yaml"
    ).read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(dataset)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path, dataset


def test_complete_hashing_overlaps_the_evaluation_and_returns_original_digests(
    tmp_path, monkeypatch,
):
    config, dataset = _config(tmp_path)
    started, release = Event(), Event()
    original = online._continuous_time_input_hashes
    observed = []
    sentinel = object()

    def hash_inputs(paths):
        started.set()
        assert release.wait(3)
        return original(paths)

    def compute(path, parsed, bag, hashes):
        assert started.wait(3)
        assert not hashes.done()
        release.set()
        observed.append(hashes.result())
        return sentinel

    monkeypatch.setattr(online, "_continuous_time_input_hashes", hash_inputs)
    monkeypatch.setattr(online, "_evaluate_continuous_time_lidar_pair", compute)
    assert online.evaluate_continuous_time_lidar_pair(config) is sentinel
    assert observed == [{str(config): sha256_path(config), str(dataset): sha256_path(dataset)}]


@pytest.mark.parametrize("change", ["config", "modify", "add", "remove"])
def test_changed_inputs_never_return_a_calibration_result(tmp_path, monkeypatch, change):
    config, dataset = _config(tmp_path)

    def compute(path, parsed, bag, hashes):
        hashes.result()
        if change == "config":
            config.write_text(
                config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8",
            )
        elif change == "modify":
            (dataset / "recording.db3").write_bytes(b"changed longer input")
        elif change == "add":
            (dataset / "extra.db3").write_bytes(b"extra")
        else:
            (dataset / "recording.db3").unlink()
        return object()

    monkeypatch.setattr(online, "_evaluate_continuous_time_lidar_pair", compute)
    with pytest.raises(DatasetError, match="inputs changed"):
        online.evaluate_continuous_time_lidar_pair(config)


def test_config_edit_during_loading_is_rejected_before_hash_worker_starts(tmp_path, monkeypatch):
    config, _dataset = _config(tmp_path)
    original = online.load_config

    def load(path):
        result = original(path)
        config.write_text(config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        return result

    def forbidden_hash(*args):
        raise AssertionError("changed configuration must be rejected before hashing")

    monkeypatch.setattr(online, "load_config", load)
    monkeypatch.setattr(online, "_continuous_time_input_hashes", forbidden_hash)
    with pytest.raises(DatasetError, match="config changed"):
        online.evaluate_continuous_time_lidar_pair(config)


def test_hash_failure_propagates_and_no_result_is_returned(tmp_path, monkeypatch):
    config, _dataset = _config(tmp_path)
    monkeypatch.setattr(online, "sha256_path", lambda path: None)
    monkeypatch.setattr(
        online, "_evaluate_continuous_time_lidar_pair",
        lambda path, parsed, bag, hashes: hashes.result(),
    )
    with pytest.raises(DatasetError, match="could not hash continuous-time input"):
        online.evaluate_continuous_time_lidar_pair(config)
