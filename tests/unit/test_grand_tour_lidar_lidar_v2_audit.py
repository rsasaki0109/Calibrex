from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tools import build_grand_tour_lidar_lidar_v2_audit as audit
from tools import grand_tour_lidar_lidar_v2_evaluate as runner

from calibrex.data.livox_ros2 import LivoxStreamProfile

ROOT = Path(__file__).resolve().parents[2]


def test_preregistration_matches_code_and_uses_unspent_dates() -> None:
    protocol = yaml.safe_load((ROOT / runner.PREREG).read_text())
    assert protocol["methods"] == runner.MODES
    assert protocol["same_time_fractions"] == runner.FRACTIONS
    assert protocol["emulated_sparse_target"] == runner.SPARSE
    assert protocol["rotation_floor_deg"] == list(audit.ROTATION_FLOOR_DEG)
    assert protocol["requirements"] == [list(gate) for gate in audit.GATES]
    recordings = protocol["recordings"]
    evaluation = recordings["evaluation"]
    dates = [item["mission"][:10] for item in evaluation]
    assert len(set(dates)) == len(dates) == 4
    assert set(dates).isdisjoint({m[:10] for m in recordings["development_only"]})
    for item in evaluation:
        assert set(item["sha256"]) == set(runner.TARS) == set(item["size_bytes"])
        assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in item["sha256"].values())


def test_ratio_preserves_left_tangent_axes_and_peer_uncertainty() -> None:
    matrix = np.eye(4)
    peer = np.eye(4)
    peer[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    matrix[:3, :3] = Rotation.from_euler("x", 0.2, degrees=True).as_matrix() @ peer[:3, :3]
    std = np.full(3, 0.1)
    assert audit.rotation_ratio(matrix, std, [peer, peer], [std, std]) == pytest.approx(
        0.2 / np.sqrt(0.01 + 0.02 / 4)
    )


def test_zarr_replay_matches_absolute_ns_and_sparse_mask(tmp_path: Path) -> None:
    import zarr

    root = zarr.open_group(str(tmp_path), mode="w")
    group = root.create_group("livox_points")
    stamp = 1730565871.0
    group["timestamp"] = [stamp]
    group["valid"] = [[4]]
    group["points"] = np.array([[[0, 0, 0], [1, 0, 0], [21, 0, 0], [2, 0, 0]]], dtype="f4")
    group["time"] = np.array([[(stamp + dt) * 1e9 for dt in (0, 0.01, 0.02, 0.03)]])
    profile = LivoxStreamProfile("lidar", "/livox_points", "", "time", "absolute_ns", "mps2")
    [(time_s, xyz, offsets)] = runner.scans(tmp_path, profile, True, None)
    assert time_s == pytest.approx(stamp)
    assert xyz[:, 0].tolist() == [1, 2]
    assert offsets is not None and offsets == pytest.approx([0.01, 0.03], abs=1e-6)


def test_audit_builds_for_missing_and_present_evidence(tmp_path: Path) -> None:
    protocol = yaml.safe_load((ROOT / runner.PREREG).read_text())
    fixtures = ROOT / "docs/assets/grand_tour_lidar_lidar/evaluation"
    source = "2024-11-03-07-52-45"
    (tmp_path / "runs.json").write_text(
        json.dumps(
            {"preregistration_sha256": audit.sha256_path(audit.PREREG), "finished_at": "test"}
        )
    )
    assert audit.build(tmp_path) == "refuted"
    for item in protocol["recordings"]["evaluation"]:
        for suffix in (
            "calibrex_dense.yaml",
            "calibrex_dense_same_time.json",
            "calibrex_sparse.yaml",
        ):
            (tmp_path / f"{item['mission']}_{suffix}").write_bytes(
                (fixtures / f"{source}_{suffix}").read_bytes()
            )
    assert audit.build(tmp_path) == "supported"
