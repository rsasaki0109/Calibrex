"""Historical benchmark options stay pinned while the public example evolves."""

import hashlib
from pathlib import Path

import pytest
from tools import _tum_benchmark_config as config


def test_original_serial_workload_is_pinned_and_independent_of_current_example(monkeypatch):
    read_bytes = Path.read_bytes

    def reject_current_example(path):
        if path.name == "native_joint_slac_config.yaml":
            raise AssertionError("historical audit read the evolving example")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_current_example)
    raw = config.SERIAL_TEMPLATE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config.SERIAL_TEMPLATE_SHA256
    original = config.load_serial_tum_config()
    original_options = original["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"]
    assert original_options.get("window_workers", 1) == 1
    original_options["query_points_per_frame"] = 1
    assert config.load_serial_tum_config() != original


def test_modified_historical_template_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "changed.yaml"
    path.write_bytes(config.SERIAL_TEMPLATE.read_bytes() + b"\n# modified\n")
    monkeypatch.setattr(config, "SERIAL_TEMPLATE", path)
    with pytest.raises(ValueError, match="historical serial TUM template differs"):
        config.load_serial_tum_config()
