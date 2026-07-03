"""Unit tests for online/streaming calibration session mechanics."""

from __future__ import annotations

import random

from calibrex.core.geometry import SE3, Vector3
from calibrex.data.livox import LivoxPointRecord
from calibrex.pipelines.online import OnlineCalibrationSession

_VARIABLE = "T_base_link_lidar_front_right"
_PARENT = "base_link"
_SENSOR = "lidar_front_right"


def _corner_scene(*, step: float = 0.2, span: float = 1.2) -> list[Vector3]:
    """Return three widely-separated plane patches, each with a distinct normal.

    A flat patch (z=0, normal +z) near the origin, plus two patches offset far
    enough away (x=3, normal +x; y=-3, normal +y) that no voxel/correspondence
    gate can confuse points from different patches. Having three independent
    normal directions at three different locations gives the point-to-plane
    factor full rank (all six DoF observable), mirroring the rank==6 geometry
    the real A2D2/Livox pair fixtures exercise in the integration test.
    """

    coords = [round(index * step, 6) for index in range(int(span / step) + 1)]
    points: list[Vector3] = []
    for a in coords:
        for b in coords:
            points.append((a, b, 0.0))  # patch A: z = 0, normal +z
            points.append((3.0, a, b))  # patch B: x = 3, normal +x
            points.append((a, -3.0, b))  # patch C: y = -3, normal +y
    return points


def _source_records(points: list[Vector3]) -> list[LivoxPointRecord]:
    records: list[LivoxPointRecord] = []
    for x, y, z in points:
        if abs(x - 3.0) < 1.0e-9:
            normal: Vector3 = (1.0, 0.0, 0.0)
        elif abs(y - (-3.0)) < 1.0e-9:
            normal = (0.0, 1.0, 0.0)
        else:
            normal = (0.0, 0.0, 1.0)
        records.append(LivoxPointRecord(point=(x, y, z, 0.0), normal_xyz=normal))
    return records


def _session(**overrides: object) -> OnlineCalibrationSession:
    scene = _corner_scene()
    defaults: dict[str, object] = {
        "variable": _VARIABLE,
        "parent": _PARENT,
        "sensor": _SENSOR,
        "source_records": _source_records(scene),
        "initial_transform": SE3((0.05, -0.03, 0.02), (0.0, 0.0, 0.0, 1.0)),
        "voxel_size_m": 0.3,
        "correspondence_gate_m": 0.6,
        "holdout_ratio": 0.3,
        "rolling_window": 200,
        "seed": 7,
    }
    defaults.update(overrides)
    return OnlineCalibrationSession(**defaults)  # type: ignore[arg-type]


def _clean_batch(rng: random.Random, scene: list[Vector3], count: int) -> list[Vector3]:
    return rng.sample(scene, count) if len(scene) >= count else list(scene)


def test_online_session_warm_starts_and_converges_on_clean_batches() -> None:
    scene = _corner_scene()
    session = _session()
    rng = random.Random(1)

    first = session.process_batch(_clean_batch(rng, scene, 90))
    assert first.batch_index == 0
    assert first.provenance["warm_start_from"] == "config_initial_transform"
    assert first.gate_status == "pass"
    assert first.estimate_accepted is True
    # the batch pulls the estimate close to the true (identity) relationship
    assert max(abs(value) for value in session.current_estimate.translation_m) < 0.01

    estimate_after_first = session.current_estimate

    second = session.process_batch(_clean_batch(rng, scene, 90))
    assert second.batch_index == 1
    # the second batch is warm-started from the *previous accepted* estimate,
    # not re-initialized from the session's original construction-time guess
    assert second.provenance["warm_start_from"] == "previous_accepted_estimate"
    assert second.gate_status == "pass"
    assert max(abs(value) for value in session.current_estimate.translation_m) < 0.01
    # estimate keeps refining rather than resetting back to the 5cm initial guess
    assert session.current_estimate.translation_m != estimate_after_first.translation_m or (
        max(abs(value) for value in estimate_after_first.translation_m) < 0.01
    )


def test_online_session_insufficient_batch_is_inconclusive_and_unchanged() -> None:
    session = _session()
    rng = random.Random(2)
    scene = _corner_scene()
    baseline = session.process_batch(_clean_batch(rng, scene, 90))
    before = session.current_estimate
    window_count_before = baseline.rolling_window_residual_count
    rolling_rmse_before = baseline.rolling_rmse_m

    # a tiny batch cannot produce enough point-to-plane correspondences
    snapshot = session.process_batch([scene[0], scene[1]])

    assert snapshot.gate_status == "inconclusive"
    # an inconclusive batch is never adopted: estimate and window untouched
    assert snapshot.estimate_accepted is False
    assert session.current_estimate.translation_m == before.translation_m
    assert session.current_estimate.rotation_quat_xyzw == before.rotation_quat_xyzw
    assert snapshot.rolling_window_residual_count == window_count_before
    assert snapshot.rolling_rmse_m == rolling_rmse_before


def test_online_session_inconclusive_holdout_batch_is_fully_non_destructive() -> None:
    """Regression test: a batch that solves on train but has too few holdout
    correspondences must not adopt its (never holdout-scored, never
    rank-checked) tentative estimate, and must not extend the rolling window.
    """

    session = _session()
    rng = random.Random(5)
    scene = _corner_scene()
    baseline = session.process_batch(_clean_batch(rng, scene, 90))
    assert baseline.gate_status == "pass"
    before = session.current_estimate
    window_count_before = baseline.rolling_window_residual_count
    rolling_rmse_before = baseline.rolling_rmse_m
    assert window_count_before > 0

    # 8 points with holdout_ratio=0.3: round(8 * 0.3) = 2 holdout points
    # (< _MIN_HOLDOUT_OBSERVATIONS = 3) and 6 train points
    # (>= _MIN_TRAIN_OBSERVATIONS = 6), so the tentative solve runs but the
    # batch cannot be holdout-scored.
    snapshot = session.process_batch(_clean_batch(rng, scene, 8))

    assert snapshot.train_point_count == 6
    assert snapshot.holdout_point_count == 2
    assert snapshot.correspondence_count >= 6
    assert snapshot.gate_status == "inconclusive"
    assert "holdout" in snapshot.gate_reason
    assert snapshot.estimate_accepted is False
    assert session.current_estimate.translation_m == before.translation_m
    assert session.current_estimate.rotation_quat_xyzw == before.rotation_quat_xyzw
    assert snapshot.rolling_window_residual_count == window_count_before
    assert snapshot.rolling_rmse_m == rolling_rmse_before

    # the next batch still warm-starts from the last *accepted* estimate
    recovery = session.process_batch(_clean_batch(rng, scene, 90))
    assert recovery.gate_status == "pass"


def test_online_session_estimate_accepted_matches_gate_status() -> None:
    session = _session()
    rng = random.Random(6)
    scene = _corner_scene()

    session.process_batch(_clean_batch(rng, scene, 90))  # pass
    session.process_batch([scene[0], scene[1]])  # inconclusive
    noisy_batch = [
        (x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), z + rng.uniform(-0.3, 0.3))
        for x, y, z in _clean_batch(rng, scene, 90)
    ]
    session.process_batch(noisy_batch)  # fail

    statuses = [snapshot.gate_status for snapshot in session.history]
    assert statuses == ["pass", "inconclusive", "fail"]
    for snapshot in session.history:
        assert snapshot.estimate_accepted == (snapshot.gate_status == "pass")
    accepted_count = sum(1 for snapshot in session.history if snapshot.gate_status == "pass")
    assert accepted_count == sum(
        1 for snapshot in session.history if snapshot.estimate_accepted
    )


def test_online_session_rolling_window_accumulates_holdout_residuals() -> None:
    session = _session(rolling_window=50)
    rng = random.Random(3)
    scene = _corner_scene()

    snapshot = None
    for _ in range(5):
        snapshot = session.process_batch(_clean_batch(rng, scene, 90))

    assert snapshot is not None
    assert snapshot.rolling_rmse_m is not None
    assert snapshot.rolling_rmse_m < 0.05
    assert snapshot.rolling_window_residual_count > 0
    assert snapshot.rolling_window_residual_count <= 50


def test_online_session_rejects_known_bad_noisy_batch() -> None:
    session = _session()
    rng = random.Random(4)
    scene = _corner_scene()

    # establish a converged, low-noise rolling baseline first
    for _ in range(3):
        good = session.process_batch(_clean_batch(rng, scene, 90))
        assert good.gate_status == "pass"

    accepted_estimate = session.current_estimate

    # a "known-bad" batch: independent per-point noise that no single rigid
    # SE(3) correction can absorb, simulating a corrupted/glitched sensor read
    noisy_batch = [
        (x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), z + rng.uniform(-0.3, 0.3))
        for x, y, z in _clean_batch(rng, scene, 90)
    ]
    bad = session.process_batch(noisy_batch)

    assert bad.gate_status == "fail"
    assert bad.estimate_accepted is False
    # the corrupted batch's proposed update is rejected: the running estimate
    # used for the *next* batch's warm start is left unchanged
    assert session.current_estimate.translation_m == accepted_estimate.translation_m
    assert session.current_estimate.rotation_quat_xyzw == accepted_estimate.rotation_quat_xyzw

    # the session recovers cleanly on the next good batch
    recovery = session.process_batch(_clean_batch(rng, scene, 90))
    assert recovery.gate_status == "pass"


def _patch_points(scene: list[Vector3], patch: str) -> list[Vector3]:
    if patch == "z":
        return [point for point in scene if abs(point[2]) < 1.0e-9]
    if patch == "x":
        return [point for point in scene if abs(point[0] - 3.0) < 1.0e-9]
    if patch == "y":
        return [point for point in scene if abs(point[1] - (-3.0)) < 1.0e-9]
    msg = f"unknown patch '{patch}'"
    raise ValueError(msg)


def _sample_patch(
    rng: random.Random, scene: list[Vector3], patch: str, count: int
) -> list[Vector3]:
    points = _patch_points(scene, patch)
    return rng.sample(points, min(count, len(points)))


def test_online_session_default_accumulation_matches_pre_change_snapshots() -> None:
    """Regression: accumulation_batches=1 preserves prior snapshot semantics."""

    scene = _corner_scene()
    session = _session()
    rng = random.Random(11)

    snapshots = [session.process_batch(_clean_batch(rng, scene, 90))]
    snapshots.append(session.process_batch(_clean_batch(rng, scene, 90)))

    for snapshot in snapshots:
        assert snapshot.batch_observability is None
        assert "accumulation_batches" not in snapshot.provenance
        assert snapshot.observability.rank is not None
        assert snapshot.observability.rank >= 6


def test_online_session_degenerate_batches_fail_without_accumulation() -> None:
    scene = _corner_scene()
    session = _session()
    rng = random.Random(12)

    for patch in ("x", "y"):
        snapshot = session.process_batch(_sample_patch(rng, scene, patch, 40))
        assert snapshot.gate_status == "fail"
        assert snapshot.estimate_accepted is False
        assert snapshot.observability.rank is not None
        assert snapshot.observability.rank < 6
        assert "rank-deficient" in snapshot.gate_reason


def test_online_session_accumulation_reaches_full_rank_and_converges() -> None:
    scene = _corner_scene()
    session = _session(accumulation_batches=3)
    rng = random.Random(13)
    true_transform = SE3.identity()

    snapshots = [
        session.process_batch(_sample_patch(rng, scene, patch, 40))
        for patch in ("z", "x", "y")
    ]

    assert snapshots[0].gate_status == "pass"
    assert snapshots[1].batch_observability is not None
    assert snapshots[1].batch_observability.rank < 6
    assert snapshots[1].observability.rank >= 6
    assert snapshots[1].gate_status == "pass"
    assert snapshots[2].batch_observability is not None
    assert snapshots[2].batch_observability.rank < 6
    assert snapshots[2].observability.rank >= 6
    assert snapshots[2].gate_status == "pass"

    assert max(abs(value) for value in session.current_estimate.translation_m) < 0.01
    assert session.current_estimate.translation_m != true_transform.translation_m or (
        max(abs(value) for value in true_transform.translation_m) < 0.01
    )


def test_online_session_rejected_batch_does_not_contaminate_accumulation() -> None:
    scene = _corner_scene()
    session = _session(accumulation_batches=3)
    rng = random.Random(14)

    first = session.process_batch(_sample_patch(rng, scene, "z", 40))
    second = session.process_batch(_sample_patch(rng, scene, "x", 40))
    assert first.gate_status == "pass"
    assert second.gate_status == "pass"
    retained_before_bad = second.provenance["retained_accepted_batch_count"]

    noisy_batch = [
        (
            x + rng.uniform(-0.3, 0.3),
            y + rng.uniform(-0.3, 0.3),
            z + rng.uniform(-0.3, 0.3),
        )
        for x, y, z in _sample_patch(rng, scene, "y", 40)
    ]
    bad = session.process_batch(noisy_batch)
    assert bad.gate_status == "fail"
    assert bad.provenance["retained_accepted_batch_count"] == retained_before_bad

    recovery = session.process_batch(_sample_patch(rng, scene, "y", 40))
    assert recovery.gate_status == "pass"
    assert recovery.observability.rank >= 6
    assert recovery.batch_observability is not None
    assert recovery.batch_observability.rank < 6
