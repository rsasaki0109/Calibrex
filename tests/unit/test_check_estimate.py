"""``calibrex estimate``: stubbed estimators on the synthetic db3 fixture."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.check_fixtures import write_check_bag

from calibrex.check import estimators
from calibrex.check.estimate import (
    ARTIFACT_FILENAME,
    FRAMES_FILENAME,
    build_bag_estimate,
    format_estimate_text,
)
from calibrex.check.estimators import (
    CheckSkipError,
    EstimatorRun,
    PairContext,
    invert_transform,
    transform_matrix,
)
from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.check.tf_sources import load_tf_file
from calibrex.check.verdict import AxisEstimate
from calibrex.cli.main import main
from calibrex.core.bag_estimate import BagEstimateArtifact
from calibrex.core.calibration_check import CheckTimeOffset
from calibrex.core.schema_registry import kind_for_schema_version
from calibrex.core.validation import validate_file

ROTATION_ERROR_DEG = (1.0, -2.0, 3.0)
TRANSLATION_ERROR_M = (0.1, 0.2, -0.3)
SKIP = CheckSkipError("unsupported_sensor", "stub: not in this test")


def _axes(
    names: tuple[str, ...],
    errors: tuple[float, ...],
    unit: str,
    *,
    estimated: bool = True,
) -> list[AxisEstimate]:
    return [
        AxisEstimate(
            name,  # type: ignore[arg-type]
            unit,  # type: ignore[arg-type]
            error,
            0.05,
            estimated,
            None if estimated else "stub: not constrained",
            None if estimated else "unobservable",
        )
        for name, error in zip(names, errors, strict=True)
    ]


def _run(
    compared: np.ndarray[Any, Any],
    *,
    translation: bool,
    rotation_observed: bool = True,
) -> EstimatorRun:
    estimates = _axes(
        ("roll", "pitch", "yaw"), ROTATION_ERROR_DEG, "deg", estimated=rotation_observed
    )
    if translation:
        estimates += _axes(("x", "y", "z"), TRANSLATION_ERROR_M, "m")
    return EstimatorRun(
        estimator="stub/v0",
        artifacts=(),
        solved=True,
        policy_status="pass",
        policy_reasons=("ok",),
        estimates=tuple(estimates),
        compared=compared,
        time_offset=CheckTimeOffset(estimate_s=0.002, std_s=0.001, status="estimated"),
    )


@pytest.fixture
def bag(tmp_path: Path) -> Path:
    return write_check_bag(tmp_path / "bag.db3")


def _install(
    monkeypatch: pytest.MonkeyPatch,
    adapters: dict[str, Callable[[PairContext], EstimatorRun]],
) -> list[PairContext]:
    calls: list[PairContext] = []

    def wrap(
        adapter: Callable[[PairContext], EstimatorRun],
    ) -> Callable[[PairContext], EstimatorRun]:
        def run(ctx: PairContext) -> EstimatorRun:
            calls.append(ctx)
            return adapter(ctx)

        return run

    table: dict[str, Callable[[PairContext], EstimatorRun]] = {
        pair: wrap(adapters[pair]) if pair in adapters else _skip for pair in estimators.ESTIMATORS
    }
    monkeypatch.setattr(estimators, "ESTIMATORS", table)
    return calls


def _skip(_ctx: PairContext) -> EstimatorRun:
    raise SKIP


def _imu_lidar_full(ctx: PairContext) -> EstimatorRun:
    return _run(invert_transform(ctx.candidate), translation=True)


def _estimate(
    bag: Path, out: Path, *, tf_files: tuple[Path, ...] = (), **kwargs: Any
) -> BagEstimateArtifact:
    return build_bag_estimate(
        bag,
        output_dir=out,
        run=CheckRunOptions(evidence_dir=out / "evidence", base_dir=out),
        tf_files=tf_files,
        **kwargs,
    )


def _expected_t_lidar_imu() -> np.ndarray[Any, Any]:
    """The stub's estimate: compared (identity placeholder) corrected by the axis errors."""

    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(np.radians(ROTATION_ERROR_DEG)).as_matrix().T
    result[:3, 3] = -np.asarray(TRANSLATION_ERROR_M)
    return result


def test_estimate_recovers_observed_axes_and_round_trips_through_check(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, {"imu-lidar": _imu_lidar_full})
    out = tmp_path / "est"
    artifact = _estimate(bag, out)

    pair = next(p for p in artifact.pairs if p.pair == "imu-lidar" and p.status != "skipped")
    assert pair.status == "estimated"
    assert pair.fill == "placeholder_identity"
    assert {axis.status for axis in pair.axes} == {"observed"}
    assert pair.transform is not None
    assert (pair.transform.parent_frame, pair.transform.child_frame) == ("lidar_front", "imu_link")
    matrix = transform_matrix(pair.transform.translation_m, pair.transform.rotation_quat_xyzw)
    np.testing.assert_allclose(matrix, _expected_t_lidar_imu(), atol=1e-9)
    assert pair.time_offset is not None

    assert artifact.frames.root == "imu_link"
    assert {(e.frame, e.parent) for e in artifact.frames.entries} == {
        ("lidar_front", "imu_link"),
        ("lidar_rear", "imu_link"),
    }
    assert all(e.axes_from_prior == [] for e in artifact.frames.entries)
    assert {e.kind for e in artifact.exports} == {
        "frames_yaml",
        "static_transform_publisher",
        "launch_yaml",
        "urdf_joints",
    }

    # The frames YAML loads with --tf and carries T_imu_lidar = inverse of the estimate.
    source = load_tf_file(out / FRAMES_FILENAME)
    edge = next(e for e in source.edges if e.child == "lidar_front")
    assert edge.parent == "imu_link"
    expected = invert_transform(_expected_t_lidar_imu())
    np.testing.assert_allclose(
        transform_matrix(edge.transform.translation_m, edge.transform.rotation_quat_xyzw),
        expected,
        atol=1e-8,
    )
    plan = build_calibration_check(bag, tf_files=[out / FRAMES_FILENAME])
    planned = next(
        p
        for p in plan.pairs
        if p.pair == "imu-lidar" and p.status == "planned" and p.frames[1] == "lidar_front"
    )
    assert planned.candidate_transform is not None
    np.testing.assert_allclose(
        planned.candidate_transform.translation_m, expected[:3, 3], atol=1e-8
    )

    text = format_estimate_text(artifact, out)
    assert "imu-lidar" in text and "observed" in text and "next steps" in text
    assert f"--tf {out / FRAMES_FILENAME}" in text


def test_estimate_artifact_is_schema_valid_with_provenance(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, {"imu-lidar": _imu_lidar_full})
    out = tmp_path / "est"
    exit_code = main(["estimate", str(bag), "--output", str(out), "--no-cache"])
    assert exit_code == 0
    path = out / ARTIFACT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "slac.bag_estimate/v0.1"
    assert kind_for_schema_version(payload["schema_version"]) == "bag-estimate"
    assert validate_file(path, "bag-estimate").kind == "bag-estimate"
    assert validate_file(path).kind == "bag-estimate"
    provenance = payload["provenance"]
    assert provenance["generator"] == "calibrex estimate"
    assert len(provenance["input_sha256"]) == 64
    assert provenance["command"][:2] == ["calibrex", "estimate"]
    assert payload["bag"]["input_sha256"] == provenance["input_sha256"]
    assert all(len(item["sha256"]) == 64 for item in payload["exports"])


def test_rotation_only_pair_is_omitted_without_a_prior(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        {"imu-lidar": lambda ctx: _run(invert_transform(ctx.candidate), translation=False)},
    )
    out = tmp_path / "est"
    artifact = _estimate(bag, out)
    pair = next(p for p in artifact.pairs if p.pair == "imu-lidar" and p.status != "skipped")
    assert pair.status == "estimated"
    assert {a.name for a in pair.axes if a.status == "not_estimated"} == {"x", "y", "z"}
    assert artifact.frames.entries == []
    assert any(o.frame == "lidar_front" and "x, y, z" in o.reason for o in artifact.frames.omitted)
    assert not (out / FRAMES_FILENAME).exists()
    assert artifact.exports == []
    assert any("prior" in step for step in artifact.next_steps)


def test_prior_supplies_missing_axes_and_the_yaml_says_so(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        {"imu-lidar": lambda ctx: _run(invert_transform(ctx.candidate), translation=False)},
    )
    prior = tmp_path / "prior.yaml"
    prior.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [
                    {
                        "name": "lidar_front",
                        "parent": "imu_link",
                        "translation_m": [0.5, -0.25, 0.1],
                        "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "est"
    artifact = _estimate(bag, out, tf_files=(prior,))
    (entry,) = artifact.frames.entries
    assert entry.axes_from_prior == ["x", "y", "z"]
    assert entry.transform.translation_m == pytest.approx([0.5, -0.25, 0.1])
    text = (out / FRAMES_FILENAME).read_text(encoding="utf-8")
    assert "NOT MEASURED: x, y, z taken from the --tf prior" in text
    assert "NOT MEASURED" in (out / "static_transforms.sh").read_text(encoding="utf-8")
    # rotation observed: the exported rotation is the estimate, not the prior's identity
    expected_rotation = Rotation.from_matrix(invert_transform(_expected_t_lidar_imu())[:3, :3])
    exported = Rotation.from_quat(entry.transform.rotation_quat_xyzw)
    assert (exported * expected_rotation.inv()).magnitude() == pytest.approx(0.0, abs=1e-8)
    assert load_tf_file(out / FRAMES_FILENAME).edges[0].child == "lidar_front"
    # a sensor frame the prior does not know is still estimated, against the identity placeholder
    rear = next(p for p in artifact.pairs if p.pair == "imu-lidar" and p.frames[1] == "lidar_rear")
    assert rear.status == "estimated" and rear.fill == "placeholder_identity"
    assert [e.frame for e in artifact.frames.entries] == ["lidar_front"]
    assert any(o.frame == "lidar_rear" for o in artifact.frames.omitted)


def test_unobservable_axis_frame_is_not_written_as_measured(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        {
            "imu-lidar": lambda ctx: _run(
                invert_transform(ctx.candidate), translation=True, rotation_observed=False
            )
        },
    )
    out = tmp_path / "est"
    artifact = _estimate(bag, out)
    pair = next(p for p in artifact.pairs if p.pair == "imu-lidar" and p.status != "skipped")
    assert pair.status == "partial"
    assert {a.status for a in pair.axes if a.name in {"roll", "pitch", "yaw"}} == {"unobservable"}
    assert all(a.value is None for a in pair.axes if a.status == "unobservable")
    assert artifact.frames.entries == []
    assert any("roll, pitch, yaw" in o.reason for o in artifact.frames.omitted)


def test_lidar_lidar_needs_a_prior_to_start_from(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install(
        monkeypatch,
        {"lidar-lidar": lambda ctx: _run(ctx.candidate, translation=True)},
    )
    artifact = _estimate(bag, tmp_path / "no_prior")
    skipped = next(p for p in artifact.pairs if p.pair == "lidar-lidar")
    assert skipped.status == "skipped" and skipped.reason_code == "no_candidate_calibration"
    assert not any(ctx.pair.pair == "lidar-lidar" for ctx in calls)
    assert any("lidar-lidar" in step and "--tf" in step for step in artifact.next_steps)

    prior = tmp_path / "prior.yaml"
    prior.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [
                    {"name": "lidar_front", "parent": "base_link", "translation_m": [1, 0, 0.5]},
                    {"name": "lidar_rear", "parent": "base_link", "translation_m": [-1, 0, 0.5]},
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact = _estimate(bag, tmp_path / "prior", tf_files=(prior,))
    pair = next(p for p in artifact.pairs if p.pair == "lidar-lidar" and p.status != "skipped")
    assert pair.fill == "prior" and pair.status == "estimated"
    ctx = next(c for c in calls if c.pair.pair == "lidar-lidar")
    np.testing.assert_allclose(ctx.candidate[:3, 3], [-2.0, 0.0, 0.0], atol=1e-9)
    assert artifact.frames.entries  # observed on all six axes: exported without prior fill


def test_camera_focal_is_not_estimated(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, {})
    artifact = _estimate(bag, tmp_path / "est")
    assert all(p.pair != "camera-focal" for p in artifact.pairs)


def test_estimate_cli_rejects_pairs_without_an_extrinsic(bag: Path, tmp_path: Path) -> None:
    argv = ["estimate", str(bag), "--output", str(tmp_path / "x"), "--pairs", "camera-focal"]
    assert main(argv) != 0
    assert not (tmp_path / "x" / ARTIFACT_FILENAME).exists()


def test_export_is_rooted_at_the_prior_so_the_trees_stay_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A /tf_static with the lidar above the IMU must not get the inverse edge back."""

    _install(monkeypatch, {"imu-lidar": _imu_lidar_full})
    bag = write_check_bag(
        tmp_path / "tf_bag.db3",
        tf_messages=[[("lidar_front", "imu_link", (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))]],
    )
    out = tmp_path / "est"
    artifact = _estimate(bag, out)
    assert artifact.frames.root == "lidar_front"
    assert ("imu_link", "lidar_front") in {(e.frame, e.parent) for e in artifact.frames.entries}
    assert not any("conflicts with the prior" in step for step in artifact.next_steps)
    # next to the bag's /tf_static the exported file is still one tree (it overrides imu_link)
    plan = build_calibration_check(bag, tf_files=[out / FRAMES_FILENAME])
    assert "lidar_front" in plan.frame_tree.roots


# ------------------------------------------- real-data shapes (KITTI vehicle, RTK-SLAM GNSS)


def _pose(rotvec_deg: tuple[float, float, float], xyz: tuple[float, float, float]) -> Any:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def test_topic_of_the_root_frame_is_mapped_in_the_export(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RTK-SLAM round trip: the IMU is the export root, so its topic must be mapped to it.

    The root is not an entry of ``frames.yaml``; without ``/imu -> imu`` in ``topic_frames``
    the checked bag's IMU topic keeps its header frame and ``imu-lidar`` / ``gnss-imu`` are
    skipped as ``frame_not_in_tree``.
    """

    _install(monkeypatch, {"imu-lidar": _imu_lidar_full})
    prior = tmp_path / "prior.yaml"
    prior.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [
                    {
                        "name": "lidar_front",
                        "parent": "imu",
                        "translation_m": [0.5, -0.25, 0.1],
                        "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
                "topic_frames": {"/imu": "imu", "/lidar_front/points": "lidar_front"},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "est"
    artifact = _estimate(bag, out, tf_files=(prior,))
    assert artifact.frames.root == "imu"
    assert "lidar_front" in [e.frame for e in artifact.frames.entries]
    exported = yaml.safe_load((out / FRAMES_FILENAME).read_text(encoding="utf-8"))
    assert exported["topic_frames"]["/imu"] == "imu"
    # the exported file alone lets check find the IMU in the tree
    plan = build_calibration_check(bag, tf_files=[out / FRAMES_FILENAME])
    imu_pair = next(p for p in plan.pairs if p.pair == "imu-lidar" and "lidar_front" in p.frames)
    assert imu_pair.reason_code != "frame_not_in_tree"


def test_vehicle_pair_keeps_the_prior_roll_and_is_marked_not_measured() -> None:
    """KITTI: pitch and yaw observed; roll and the lever arm are the prior's; vehicle root."""

    from calibrex.check.estimate import Relation, build_frame_tree

    prior = _pose((-0.85, 0.10, 0.04), (0.81, -0.31, 0.80))
    estimate = _pose((-0.85, 0.52, -0.31), (0.81, -0.31, 0.80))  # roll = the prior's
    relation = Relation(
        "lidar-vehicle",
        "base_link",
        "velo_link",
        estimate,
        frozenset({"pitch", "yaw"}),
        True,
        prior,
    )
    frames, edges = build_frame_tree([relation], [], [], "base_link")
    assert frames.root == "base_link"
    (entry,) = frames.entries
    assert (entry.parent, entry.frame) == ("base_link", "velo_link")
    assert entry.axes_from_prior == ["roll", "x", "y", "z"]
    rotvec = Rotation.from_quat(entry.transform.rotation_quat_xyzw).as_rotvec()
    assert np.degrees(rotvec) == pytest.approx([-0.85, 0.52, -0.31], abs=1e-6)
    assert entry.transform.translation_m == pytest.approx([0.81, -0.31, 0.80])
    assert len(edges) == 1


def test_vehicle_pair_without_a_prior_is_omitted_not_exported_as_measured() -> None:
    from calibrex.check.estimate import Relation, build_frame_tree

    relation = Relation(
        "lidar-vehicle",
        "base_link",
        "velo_link",
        _pose((0.0, 0.5, -0.3), (0.0, 0.0, 0.0)),
        frozenset({"pitch", "yaw"}),
        False,
    )
    frames, _ = build_frame_tree([relation], [], [], "base_link")
    assert frames.entries == []


def test_gnss_lever_arm_with_one_observed_axis_keeps_the_prior_for_the_rest() -> None:
    """RTK-SLAM: only y observed; the antenna orientation, x and z are the CAD prior's."""

    from calibrex.check.estimate import Relation, build_frame_tree

    prior = _pose((0.0, 0.0, 0.0), (0.023, -0.023, 0.090))
    estimate = _pose((0.0, 0.0, 0.0), (0.023, -0.0157, 0.090))
    relation = Relation("gnss-imu", "imu", "gnss_antenna", estimate, frozenset({"y"}), True, prior)
    frames, _ = build_frame_tree([relation], [], [], None, None)
    (entry,) = frames.entries
    assert entry.parent == "imu"
    assert entry.axes_from_prior == ["roll", "pitch", "yaw", "x", "z"]
    assert entry.transform.translation_m == pytest.approx([0.023, -0.0157, 0.090])
    assert entry.transform.rotation_quat_xyzw == pytest.approx([0.0, 0.0, 0.0, 1.0])
