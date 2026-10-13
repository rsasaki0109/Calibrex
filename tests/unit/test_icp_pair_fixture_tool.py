"""Actual-frame direction, immutable fixture bindings and manifest validation."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from tools.benchmark_cuda_point_to_point_icp import _load_pair_fixture
from tools.prepare_kitti_icp_pairs import prepare_pair


def _pair(tmp_path: Path):
    root = tmp_path / "dataset"
    for name in ("velodyne_points/data", "oxts/data"):
        (root / name).mkdir(parents=True)
    for index in (0, 1):
        np.asarray([[index, 2., 3., .1]], dtype=np.float32).tofile(
            root / f"velodyne_points/data/{index:010d}.bin")
        # One metre eastward, with zero latitude and orientation.
        values = [0., index*180/(math.pi*6378137), 0., 0., 0., 0.] + [0.]*24
        (root / f"oxts/data/{index:010d}.txt").write_text(" ".join(map(str, values)))
    calibration = root / "calib_imu_to_velo.txt"
    calibration.write_text("R: 0 -1 0 1 0 0 0 0 1\nT: 1 2 3\n")
    fixture = prepare_pair(root, calibration, 0, 1, 20)
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(fixture))
    return root, path, fixture


def test_navigation_transform_maps_source_into_target_in_velodyne_coordinates(tmp_path):
    root, path, fixture = _pair(tmp_path)
    np.testing.assert_allclose(fixture["reference_transform"]["translation_m"], [0., -1., 0.],
                               atol=1e-12)
    assert fixture["options"]["max_iterations"] == 20
    loaded, files = _load_pair_fixture(path, root)
    assert loaded == fixture
    assert len(files) == 5


def test_modified_raw_scan_is_rejected_before_solver_execution(tmp_path):
    root, path, fixture = _pair(tmp_path)
    (root / fixture["target"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        _load_pair_fixture(path, root)


def test_unbound_pointcloud_is_rejected(tmp_path):
    root, path, fixture = _pair(tmp_path)
    del fixture["provenance"]["input_sha256"][fixture["target"]]
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="digest-bound"):
        _load_pair_fixture(path, root)


def test_input_path_cannot_escape_dataset_root(tmp_path):
    root, path, fixture = _pair(tmp_path)
    fixture["provenance"]["input_sha256"]["../escape.bin"] = "0"*64
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="outside"):
        _load_pair_fixture(path, root)
