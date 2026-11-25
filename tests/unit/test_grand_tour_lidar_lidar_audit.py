from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tools import build_grand_tour_lidar_lidar_audit as audit
from tools.grand_tour_lidar_lidar_evaluate import method_command

ROOT = Path(__file__).resolve().parents[2]

PREREGISTRATION = ROOT / "docs/benchmarks/grand_tour_lidar_lidar_preregistration.yaml"
MISSIONS = ["m1", "m2", "m3", "m4"]
BASE = np.eye(4)
BASE[:3, :3] = Rotation.from_euler("xyz", [180.0, 0.0, -90.0], degrees=True).as_matrix()
BASE[:3, 3] = [0.0, 0.0, 0.1]


def prereg() -> dict[str, Any]:
    return yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))


def test_preregistration_matches_the_frozen_scorer() -> None:
    data = prereg()
    rules = {item["id"]: item["rule"] for item in data["requirements"]}
    built = {
        item.requirement_id: item
        for item in audit.requirements(
            {"dense": ("d.yaml", "0" * 64), "sparse": ("s.yaml", "0" * 64)}
        )
    }
    assert set(rules) == set(built)
    for requirement_id, item in built.items():
        if item.threshold:
            number = re.findall(r"[0-9.]+", rules[requirement_id].split("<=")[-1])[0]
            assert float(number) == pytest.approx(item.threshold)
    assert audit.SAME_TIME_THRESHOLD_M == 0.035
    assert audit.SPARSE_ROTATION_THRESHOLD_DEG == 0.25
    assert audit.SPARSE_TRANSLATION_THRESHOLD_M == 0.01
    assert (audit.CONSISTENCY_ROTATION_DEG, audit.CONSISTENCY_TRANSLATION_M) == (0.3, 0.03)
    development = data["thresholds_from_development"]
    assert 1.5 * development["same_time_dense_m"] <= audit.SAME_TIME_THRESHOLD_M
    assert 1.5 * development["sparse_rotation_to_dense_deg_worst"] <= 0.25
    assert 1.5 * development["sparse_translation_to_dense_m_worst"] <= 0.01


def test_preregistration_pins_four_unspent_missions() -> None:
    data = prereg()
    evaluation = data["recordings"]["evaluation"]
    assert [item["mission"][:10] for item in evaluation] == [
        "2024-11-03",
        "2024-11-04",
        "2024-11-11",
        "2024-11-14",
    ]
    assert data["recordings"]["development_only"] == ["2024-11-02-17-43-10"]
    for item in evaluation:
        assert set(item["sha256"]) == {"hesai_points.tar", "livox_points.tar", "tf.tar"}
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in item["sha256"].values())
    assert re.fullmatch(r"[0-9a-f]{40}", data["dataset"]["revision"])


def test_method_commands_follow_the_preregistration(tmp_path: Path) -> None:
    methods = prereg()["methods"]
    sparse = method_command(
        methods["calibrex_sparse"],
        tmp_path / "bag",
        (tmp_path / "a.yaml", tmp_path / "b.yaml"),
        tmp_path / "out.yaml",
    )
    assert sparse[sparse.index("--target-accumulate-scans") + 1] == "10"
    assert sparse[sparse.index("--target-motion") + 1] == "reference_trajectory"
    dense = method_command(
        methods["calibrex_dense"],
        tmp_path / "bag",
        (tmp_path / "a.yaml", tmp_path / "b.yaml"),
        tmp_path / "out.yaml",
    )
    assert "--target-motion" not in dense
    assert dense[dense.index("--target-point-time-encoding") + 1] == "absolute_ns"


def write_artifact(path: Path, matrix: np.ndarray | None, status: str = "pass") -> None:
    payload: dict[str, Any] = {"policy_status": status}
    if matrix is not None:
        payload["transform"] = {
            "rotation_quat_xyzw": Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist(),
            "translation_m": matrix[:3, 3].tolist(),
        }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def write_same_time(path: Path, estimate: float, tf: float = 0.2) -> None:
    rows = [
        {
            "estimate": {"median_point_to_plane_m": estimate},
            "dataset_tf": {"median_point_to_plane_m": tf},
        }
    ] * 5
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def shifted(rotation_deg: float = 0.0, translation_m: float = 0.0) -> np.ndarray:
    moved = BASE.copy()
    moved[:3, :3] = Rotation.from_euler("z", rotation_deg, degrees=True).as_matrix() @ BASE[:3, :3]
    moved[0, 3] += translation_m
    return moved


def fill(directory: Path, overrides: dict[tuple[str, str], tuple[np.ndarray | None, str]]) -> None:
    for mission in MISSIONS:
        for method in ("calibrex_dense", "calibrex_sparse", "sparse_rigid"):
            matrix, status = overrides.get((mission, method), (BASE, "pass"))
            write_artifact(directory / f"{mission}_{method}.yaml", matrix, status)
            write_same_time(directory / f"{mission}_{method}_same_time.json", 0.02)


def test_score_reports_consistency_and_sparse_errors(tmp_path: Path) -> None:
    fill(
        tmp_path,
        {
            ("m2", "calibrex_dense"): (shifted(0.6), "pass"),
            ("m3", "calibrex_sparse"): (shifted(0.0, 0.02), "pass"),
            ("m4", "sparse_rigid"): (None, "fail"),
        },
    )
    scores = audit.score(tmp_path, MISSIONS)["missions"]
    assert scores["m2"]["calibrex_dense"]["metrics"][audit.CONSISTENCY] == pytest.approx(2.0)
    assert scores["m1"]["calibrex_dense"]["metrics"][audit.CONSISTENCY] == pytest.approx(
        (0.6 / 3) / 0.3, rel=1e-3
    )
    sparse = scores["m3"]["calibrex_sparse"]["metrics"]
    assert sparse[audit.TRANSLATION] == pytest.approx(0.02)
    assert sparse[audit.ROTATION] == pytest.approx(0.0, abs=1e-9)
    assert scores["m4"]["sparse_rigid"]["failure"].startswith("no estimate")
    assert scores["m1"]["dataset_tf"]["metrics"][audit.SAME_TIME] == pytest.approx(0.2)


def test_a_non_pass_verdict_or_missing_dense_fails_the_trial(tmp_path: Path) -> None:
    fill(
        tmp_path,
        {
            ("m1", "calibrex_dense"): (None, "fail"),
            ("m2", "calibrex_sparse"): (BASE, "inconclusive"),
        },
    )
    scores = audit.score(tmp_path, MISSIONS)["missions"]
    assert scores["m1"]["calibrex_sparse"]["failure"] == "the mission has no dense estimate"
    assert scores["m2"]["calibrex_sparse"]["failure"] == "verdict inconclusive"
    assert scores["m3"]["calibrex_sparse"]["failure"] is None


def test_chordal_mean_of_identical_rotations_is_that_rotation() -> None:
    mean = audit.chordal_mean([BASE, BASE, BASE])
    assert np.allclose(mean, BASE)
