from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact
from calibrex.evaluation.lidar_lidar_map import (
    LidarLidarMapOptions,
    evaluate_lidar_lidar_map,
    read_transform,
)
from calibrex.solvers.lidar_lidar_map_solver import (
    EXTRINSIC_DOFS,
    MapSample,
    chi_square,
    perturb_extrinsic,
    solve_map_extrinsic,
)

TRUE_X = np.eye(4)
TRUE_X[:3, :3] = Rotation.from_euler("xyz", [90.0, 0.4, 179.0], degrees=True).as_matrix()
TRUE_X[:3, 3] = [-0.5, 0.03, -0.01]


def room(rng: np.random.Generator) -> np.ndarray:
    """Surface samples of a 12 x 8 x 4 m room with two boxes."""

    faces = []
    for axis, value, extent in [
        (0, 0.0, (8, 4)), (0, 12.0, (8, 4)), (1, 0.0, (12, 4)), (1, 8.0, (12, 4)),
        (2, 0.0, (12, 8)), (2, 4.0, (12, 8)),
    ]:  # fmt: skip
        uv = rng.uniform(0, 1, (6000, 2)) * extent
        other = [index for index in range(3) if index != axis]
        points = np.zeros((6000, 3))
        points[:, axis] = value
        points[:, other[0]], points[:, other[1]] = uv[:, 0], uv[:, 1]
        faces.append(points)
    for corner in ([3.0, 2.0, 0.0], [8.0, 5.0, 0.0]):
        box = rng.uniform(0, 1, (4000, 3)) * [1.5, 1.0, 1.2]
        box[np.arange(4000), rng.integers(0, 3, 4000)] = (
            rng.integers(0, 2, 4000) * np.array([1.5, 1.0, 1.2])[rng.integers(0, 3, 4000)]
        )
        faces.append(box + corner)
    return np.concatenate(faces)


def samples(rng: np.random.Generator, count: int = 12) -> list[MapSample]:
    world = room(rng)
    result = []
    for index in range(count):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler(
            "xyz", rng.normal(0, [10, 10, 60]), degrees=True
        ).as_matrix()
        pose[:3, 3] = [rng.uniform(3, 9), rng.uniform(2, 6), rng.uniform(1.2, 2.8)]
        target_pose = pose @ TRUE_X
        visible = world[rng.choice(len(world), 3000, replace=False)]
        local = (visible - target_pose[:3, 3]) @ target_pose[:3, :3]
        result.append(
            MapSample(
                time_s=float(index),
                block=index // 3,
                reference_pose=pose,
                target_points=local + rng.normal(0, 0.01, local.shape),
                map_points=world,
            )
        )
    return result


def test_joint_solve_recovers_a_sideways_extrinsic() -> None:
    rng = np.random.default_rng(0)
    data = samples(rng)
    initial = perturb_extrinsic(perturb_extrinsic(TRUE_X, "yaw", np.radians(2.0)), "x", 0.1)

    result = solve_map_extrinsic(data, initial)

    assert result.transform is not None and result.status == "converged"
    rotation_error = Rotation.from_matrix(result.transform[:3, :3] @ TRUE_X[:3, :3].T).magnitude()
    assert np.degrees(rotation_error) < 0.05
    assert np.allclose(result.transform[:3, 3], TRUE_X[:3, 3], atol=0.005)
    assert result.covariance is not None and np.all(np.diag(result.covariance) > 0)


def test_known_bad_shifts_raise_the_chi_square() -> None:
    rng = np.random.default_rng(1)
    data = samples(rng, count=6)
    base, _, _ = chi_square(data, TRUE_X, 0.01)
    for dof in EXTRINSIC_DOFS:
        amount = np.radians(1.0) if dof in ("roll", "pitch", "yaw") else 0.05
        moved, _, _ = chi_square(data, perturb_extrinsic(TRUE_X, dof, amount), 0.01)
        assert moved - base > 9.0, dof


def test_evaluation_passes_and_reports_the_reference_difference(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    data = samples(rng, count=27)  # 9 blocks of 3
    reference = perturb_extrinsic(TRUE_X, "y", 0.04)

    evaluation = evaluate_lidar_lidar_map(
        data, reference, LidarLidarMapOptions(jackknife_groups=6), reference_transform=reference
    )

    assert evaluation.policy_status == "pass", evaluation.policy_reasons
    records = {record.name: record for record in evaluation.records}
    assert records["y"].error_to_reference == pytest.approx(-0.04, abs=0.005)
    assert abs(records["yaw"].error_to_reference or 0.0) < 0.05
    assert evaluation.reference_delta_chi2 is not None and evaluation.reference_delta_chi2 > 9.0


def test_read_transform_accepts_opencv_yaml_and_plain_matrices(tmp_path: Path) -> None:
    opencv = tmp_path / "lidar.yaml"
    opencv.write_text(
        "%YAML:1.0\n\nT_Body2Lidar: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n"
        "   data: [-1.0, 0.0, 0.0, -0.55, 0.0, 0.0, 1.0, 0.03, 0.0, 1.0, 0.0, 0.05, "
        "0.0, 0.0, 0.0, 1.0]\n",
        encoding="utf-8",
    )
    plain = tmp_path / "plain.yaml"
    plain.write_text("matrix: [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]\n")
    broken = tmp_path / "broken.yaml"
    broken.write_text("matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]\n")

    assert read_transform(opencv)[0, 3] == pytest.approx(-0.55)
    assert read_transform(plain)[:3, 3].tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="4x4"):
        read_transform(broken)


def test_cli_composes_body_transforms_into_the_initial_and_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text("T: [[1, 0, 0, -0.05], [0, 1, 0, 0], [0, 0, 1, 0.055], [0, 0, 0, 1]]\n")
    second.write_text("T: [[-1, 0, 0, -0.55], [0, 0, 1, 0.03], [0, 1, 0, 0.05], [0, 0, 0, 1]]\n")
    captured: dict[str, Any] = {}

    def fake_run(bag: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise ValueError("stop")

    monkeypatch.setattr("calibrex.evaluation.lidar_lidar_map.run_ros2_lidar_lidar_map", fake_run)
    exit_code = main(
        [
            "lidar-lidar", "ros2", "bag", "--reference-topic", "/a", "--target-topic", "/b",
            "--body-transforms", str(first), str(second),
            "--dataset-family", "ntu_viral", "--dataset-license", "CC BY-NC-SA 4.0",
            "--output", str(tmp_path / "out.yaml"),
        ]
    )  # fmt: skip

    assert exit_code != 0
    assert np.allclose(captured["initial"][:3, 3], [-0.5, 0.03, -0.005])
    assert np.allclose(captured["reference_transform"], captured["initial"])


def test_artifact_refuses_a_pass_without_every_dof() -> None:
    record = {
        "name": "x", "unit": "m", "value": 0.0, "std_analytic": 0.0, "std_reported": 0.5,
        "status": "unobservable",
    }  # fmt: skip
    with pytest.raises(ValueError, match="all six"):
        LidarLidarExtrinsicArtifact.model_validate(
            {
                "solver_status": "converged",
                "policy_status": "pass",
                "policy_reasons": ["x"],
                "calibrated_dofs": [],
                "dofs": [record],
                "samples": {
                    "reference_scans": 1, "samples": 1, "train_samples": 1,
                    "holdout_samples": 0, "train_correspondences": 1,
                    "holdout_correspondences": 0,
                },
                "jackknife_fits": 0,
                "options": {},
                "provenance": {
                    "generator": "pytest", "generator_version": "1",
                    "dataset_family": "synthetic", "sequence_ids": ["s"],
                    "reference_topic": "/a", "target_topic": "/b", "input_sha256": "c" * 64,
                    "input_digest_scope": "synthetic", "dataset_license": "synthetic",
                },
            }
        )
