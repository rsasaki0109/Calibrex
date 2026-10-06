"""Estimator cache of ``calibrex check``: keys, hits, candidate independence, scan reuse."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.check_fixtures import default_tf, write_check_bag
from tests.unit.test_imu_lidar_rotation import TRUE_ROTATION, _artifact, synthetic

from calibrex.check import cache as cache_module
from calibrex.check import estimators
from calibrex.check.cache import EstimatorCache, cached_artifact, default_cache_dir
from calibrex.check.estimators import (
    POINTCLOUD2_TYPE,
    EstimatorRun,
    PairContext,
    RunControls,
)
from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.cli.main import main
from calibrex.core.calibration_check import CheckPairRecord, CheckTransform
from calibrex.core.imu_lidar_rotation import ImuLidarRotationArtifact, load_imu_lidar_rotation
from calibrex.data import livox_ros2
from calibrex.data.livox_ros2 import LivoxStreamProfile, ScanStore
from calibrex.evaluation import imu_lidar_rotation as rotation_module
from calibrex.evaluation.imu_lidar_rotation import (
    ImuLidarRunOptions,
    evaluate_imu_lidar_rotation,
    rebase_rotation_reference,
)
from calibrex.evaluation.imu_lidar_translation import ImuLidarTranslationOptions
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions, preprocess_scan

BAG_SHA = "a" * 64


# ------------------------------------------------------------------ keys / storage


def test_key_changes_with_every_ingredient(tmp_path: Path) -> None:
    cache = EstimatorCache(tmp_path, code="1.0+abc")
    base = cache.key(BAG_SHA, "imu_lidar_rotation", {"max_seconds": 40.0, "topic": "/a"})

    assert base == cache.key(BAG_SHA, "imu_lidar_rotation", {"topic": "/a", "max_seconds": 40.0})
    assert base != cache.key("b" * 64, "imu_lidar_rotation", {"max_seconds": 40.0, "topic": "/a"})
    assert base != cache.key(BAG_SHA, "camera_imu_rotation", {"max_seconds": 40.0, "topic": "/a"})
    assert base != cache.key(BAG_SHA, "imu_lidar_rotation", {"max_seconds": 41.0, "topic": "/a"})
    assert base != cache.key(BAG_SHA, "imu_lidar_rotation", {"max_seconds": 40.0, "topic": "/b"})
    # A new calibrex version or source revision invalidates the entry.
    other_code = EstimatorCache(tmp_path, code="1.1+abc")
    assert base != other_code.key(
        BAG_SHA, "imu_lidar_rotation", {"max_seconds": 40.0, "topic": "/a"}
    )


def test_key_expands_dataclass_options(tmp_path: Path) -> None:
    cache = EstimatorCache(tmp_path, code="x")
    default = {"options": ImuLidarRunOptions()}
    changed = {"options": dataclasses.replace(ImuLidarRunOptions(), gyro_deskew_passes=2)}

    assert cache.key(BAG_SHA, "e", default) == cache.key(
        BAG_SHA, "e", {"options": ImuLidarRunOptions()}
    )
    assert cache.key(BAG_SHA, "e", default) != cache.key(BAG_SHA, "e", changed)


def test_default_cache_dir_respects_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "calibrex" / "check"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_cache_dir() == tmp_path / "home" / ".cache" / "calibrex" / "check"


def _evaluation(reference: np.ndarray | None) -> Any:
    gyro, windows = synthetic(planar=False, seed=5)
    return evaluate_imu_lidar_rotation(gyro, windows, reference_rotation=reference)


def _rotation_artifact(reference: np.ndarray | None) -> ImuLidarRotationArtifact:
    return _artifact(_evaluation(reference))


def test_hit_miss_and_corrupt_entry(tmp_path: Path) -> None:
    cache = EstimatorCache(tmp_path, code="v1")
    computed: list[int] = []

    def compute() -> ImuLidarRotationArtifact:
        computed.append(1)
        return _rotation_artifact(TRUE_ROTATION)

    def call(options: dict[str, Any]) -> tuple[ImuLidarRotationArtifact, bool]:
        return cached_artifact(
            cache,
            BAG_SHA,
            "imu_lidar_rotation",
            options,
            loader=load_imu_lidar_rotation,
            compute=compute,
            rebase=lambda artifact: rebase_rotation_reference(artifact, TRUE_ROTATION),
        )

    first, first_hit = call({"max_seconds": 40.0})
    second, second_hit = call({"max_seconds": 40.0})
    assert (first_hit, second_hit) == (False, True) and len(computed) == 1
    assert second.rotation_quat_xyzw == first.rotation_quat_xyzw
    assert [d.std_reported for d in second.dofs] == [d.std_reported for d in first.dofs]

    call({"max_seconds": 80.0})  # an option change is a miss
    assert len(computed) == 2

    key = cache.key(BAG_SHA, "imu_lidar_rotation", {"max_seconds": 40.0})
    (tmp_path / f"{key}.yaml").write_text("not: [valid", encoding="utf-8")
    _, hit = call({"max_seconds": 40.0})  # corrupt entry: recompute, not an error
    assert hit is False and len(computed) == 3


def test_version_change_misses_and_no_cache_always_computes(tmp_path: Path) -> None:
    computed: list[int] = []

    def run(cache: EstimatorCache | None) -> bool:
        def compute() -> ImuLidarRotationArtifact:
            computed.append(1)
            return _rotation_artifact(None)

        return cached_artifact(
            cache,
            BAG_SHA,
            "e",
            {},
            loader=load_imu_lidar_rotation,
            compute=compute,
            rebase=lambda artifact: artifact,
        )[1]

    assert [run(EstimatorCache(tmp_path, code="v1")) for _ in range(2)] == [False, True]
    assert run(EstimatorCache(tmp_path, code="v2")) is False  # new version: miss
    assert [run(None), run(None)] == [False, False]  # --no-cache
    assert len(computed) == 4
    assert len(list(tmp_path.glob("*.yaml"))) == 2  # no cache => nothing written


def test_code_fingerprint_names_version() -> None:
    from calibrex import __version__

    assert cache_module.code_fingerprint().startswith(__version__)


def _fake_package(root: Path) -> Path:
    pkg = root / "pkg"
    for relative in (
        "core/solver.py",
        "check/estimators.py",
        "check/hints.py",
        "cli/main.py",
        "visualization/report.py",
    ):
        (pkg / relative).parent.mkdir(parents=True, exist_ok=True)
        (pkg / relative).write_text("x = 1\n")
    return pkg


def _fingerprint_fresh(pkg: Path) -> str:
    cache_module._FINGERPRINT_MEMO.clear()
    return cache_module.code_fingerprint(pkg)


def test_code_fingerprint_is_stable_and_memoised(tmp_path: Path) -> None:
    pkg = _fake_package(tmp_path)
    first = _fingerprint_fresh(pkg)
    assert cache_module.code_fingerprint(pkg) == first
    assert _fingerprint_fresh(pkg) == first
    assert cache_module.code_fingerprint() == cache_module.code_fingerprint()


@pytest.mark.parametrize("relative", ["core/solver.py", "check/estimators.py"])
def test_code_fingerprint_changes_with_included_source(tmp_path: Path, relative: str) -> None:
    pkg = _fake_package(tmp_path)
    before = _fingerprint_fresh(pkg)
    with (pkg / relative).open("a") as handle:
        handle.write("# edit\n")
    assert _fingerprint_fresh(pkg) != before


def test_code_fingerprint_changes_with_new_included_file(tmp_path: Path) -> None:
    pkg = _fake_package(tmp_path)
    before = _fingerprint_fresh(pkg)
    (pkg / "core" / "new.py").write_text("y = 2\n")
    assert _fingerprint_fresh(pkg) != before


@pytest.mark.parametrize("relative", ["check/hints.py", "cli/main.py", "visualization/report.py"])
def test_code_fingerprint_ignores_excluded_source(tmp_path: Path, relative: str) -> None:
    pkg = _fake_package(tmp_path)
    before = _fingerprint_fresh(pkg)
    with (pkg / relative).open("a") as handle:
        handle.write("# edit\n")
    (pkg / "cli" / "extra.py").write_text("z = 3\n")
    assert _fingerprint_fresh(pkg) == before


def test_fingerprint_excluded_entries_exist() -> None:
    package_dir = Path(cache_module.__file__).resolve().parents[1]
    for entry in cache_module.FINGERPRINT_EXCLUDED:
        target = package_dir / entry
        assert target.is_dir() if entry.endswith("/") else target.is_file(), entry


# ------------------------------------------------------ candidate independence


def test_rebase_matches_a_fresh_evaluation_with_another_reference() -> None:
    other = Rotation.from_euler("xyz", [0.0, 0.0, 1.0], degrees=True).as_matrix() @ TRUE_ROTATION
    stale = _rotation_artifact(TRUE_ROTATION)
    fresh = _rotation_artifact(other)

    rebased = rebase_rotation_reference(stale, other)

    for got, want in zip(rebased.dofs, fresh.dofs, strict=True):
        assert got.name == want.name
        assert got.value == want.value and got.std_reported == want.std_reported
        assert got.known_bad_control == want.known_bad_control
        if want.reference_value is None:
            assert got.reference_value is None
        else:
            assert got.reference_value == pytest.approx(want.reference_value, abs=1e-9)
        if want.error_to_reference is None:
            assert got.error_to_reference is None
        else:
            assert got.error_to_reference == pytest.approx(want.error_to_reference, abs=1e-9)
    assert rebase_rotation_reference(stale, None).dofs[2].reference_value is None


@pytest.fixture
def fast_imu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake bags of these tests have no IMU topic: report a 200 Hz IMU to the rate gate."""

    times = np.arange(0.0, 1.0, 0.005)
    samples = livox_ros2.ImuSamples(times, np.zeros((len(times), 3)), np.zeros((len(times), 3)))
    monkeypatch.setattr(livox_ros2, "load_livox_imu", lambda *a, **k: samples)


def _imu_lidar_context(
    candidate_t_imu_lidar: np.ndarray, cache: EstimatorCache | None
) -> PairContext:
    record = CheckPairRecord(
        pair="imu-lidar",
        sensors=["imu", "lidar"],
        frames=["imu", "lidar"],
        status="planned",
        candidate_transform=CheckTransform(
            parent_frame="imu",
            child_frame="lidar",
            translation_m=[0.0, 0.0, 0.0],
            rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        ),
    )
    return PairContext(
        bag=Path("unused-bag"),
        pair=record,
        topic_types={"/lidar": POINTCLOUD2_TYPE},
        sensor_topics={"lidar": ("/lidar",), "imu": ("/imu",)},
        candidate=candidate_t_imu_lidar,
        sources=[],
        controls=RunControls(imu_lidar_translation=False, cache=cache, bag_sha256=BAG_SHA),
    )


def test_a_perturbed_candidate_reuses_the_cached_estimate(
    fast_imu: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[np.ndarray | None] = []

    def fake_rotation(*args: Any, **kwargs: Any) -> ImuLidarRotationArtifact:
        reference = kwargs["reference_rotation"]
        calls.append(reference)
        return _rotation_artifact(reference)

    monkeypatch.setattr(rotation_module, "run_livox_imu_lidar_rotation", fake_rotation)
    monkeypatch.setattr(estimators, "detect_point_time", lambda *a: ("t", "offset_s"))
    cache = EstimatorCache(tmp_path, code="v1")

    # T_imu_lidar candidates: the true calibration, then yawed by +1 deg (parent axes of
    # T_lidar_imu), which the estimator must see as a 1 deg yaw error.
    truth = np.eye(4)
    truth[:3, :3] = TRUE_ROTATION.T
    yawed_lidar_imu = Rotation.from_euler("z", 1.0, degrees=True).as_matrix() @ TRUE_ROTATION
    yawed = np.eye(4)
    yawed[:3, :3] = yawed_lidar_imu.T

    nominal = estimators.run_imu_lidar(_imu_lidar_context(truth, cache))
    perturbed = estimators.run_imu_lidar(_imu_lidar_context(yawed, cache))

    assert len(calls) == 1  # the second candidate did not re-run the estimator
    assert nominal.evidence_from_cache is False and perturbed.evidence_from_cache is True
    nominal_yaw = next(e for e in nominal.estimates if e.name == "yaw")
    perturbed_yaw = next(e for e in perturbed.estimates if e.name == "yaw")
    assert perturbed_yaw.std == nominal_yaw.std
    assert perturbed_yaw.candidate_error - nominal_yaw.candidate_error == pytest.approx(
        1.0, abs=1e-6
    )
    # The artifact written as evidence reports the *current* candidate's comparison.
    cached_yaw = next(d for d in perturbed.artifacts[0].artifact.dofs if d.name == "yaw")  # type: ignore[attr-defined]
    assert cached_yaw.error_to_reference == pytest.approx(-perturbed_yaw.candidate_error, abs=1e-6)

    # Without a cache both candidates run the estimator and nothing is flagged.
    calls.clear()
    plain = [estimators.run_imu_lidar(_imu_lidar_context(c, None)) for c in (truth, yawed)]
    assert len(calls) == 2 and all(run.evidence_from_cache is None for run in plain)


def test_check_artifact_records_cache_use_and_cli_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bag = write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])
    seen: list[RunControls] = []

    def adapter(ctx: PairContext) -> EstimatorRun:
        seen.append(ctx.controls)
        run = _stub_run()
        flag = ctx.controls.cache is not None
        return dataclasses.replace(
            run,
            artifacts=(dataclasses.replace(run.artifacts[0], from_cache=flag or None),),
        )

    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {"imu-lidar": adapter, "lidar-lidar": adapter, "camera-imu": adapter},
    )
    artifact = build_calibration_check(
        bag, run=CheckRunOptions(base_dir=tmp_path, cache_dir=tmp_path / "cache")
    )
    run_pairs = [p for p in artifact.pairs if p.status in {"pass", "warn", "fail"}]
    assert run_pairs and all(p.evidence_from_cache is True for p in run_pairs)
    assert all(p.evidence[0].from_cache is True for p in run_pairs)
    assert all(c.cache is not None and c.bag_sha256 == artifact.bag.input_sha256 for c in seen)

    seen.clear()
    off = build_calibration_check(bag, run=CheckRunOptions(base_dir=tmp_path))
    assert all(c.cache is None for c in seen)
    assert all(p.evidence_from_cache is None for p in off.pairs)

    seen.clear()
    out = tmp_path / "result.json"
    cache_dir = tmp_path / "cli-cache"
    assert (
        main(
            [
                "check",
                str(bag),
                "--output",
                str(out),
                "--cache-dir",
                str(cache_dir),
                "--fail-on",
                "never",
            ]
        )
        == 0
    )
    assert seen and all(c.cache is not None and c.cache.directory == cache_dir for c in seen)
    seen.clear()
    assert main(["check", str(bag), "--output", str(out), "--no-cache", "--fail-on", "never"]) == 0
    assert seen and all(c.cache is None for c in seen)
    seen.clear()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert main(["check", str(bag), "--output", str(out), "--fail-on", "never"]) == 0
    assert seen[0].cache is not None
    assert seen[0].cache.directory == tmp_path / "xdg" / "calibrex" / "check"


def _stub_run() -> EstimatorRun:
    from tests.unit.test_check_run import _run

    return _run()


# --------------------------------------------------------- scan reuse / defaults


def _fake_stream(monkeypatch: pytest.MonkeyPatch, count: int = 6) -> list[int]:
    rng = np.random.default_rng(1)
    clouds = [
        SimpleNamespace(
            timestamp_ns=1_700_000_000_000_000_000 + i * 100_000_000,
            xyz=rng.normal(size=(50, 3)).astype(np.float32),
            point_time_offsets_s=np.linspace(0.0, 0.09, 50),
        )
        for i in range(count)
    ]
    reads: list[int] = []

    def messages(bag: Any, topics: Any) -> Any:
        reads.append(1)
        return iter([(None, cloud.timestamp_ns, bytes([i])) for i, cloud in enumerate(clouds)])

    monkeypatch.setattr(livox_ros2, "iter_messages", messages)
    monkeypatch.setattr(
        livox_ros2.ros_cdr,
        "decode_ros2_pointcloud2",
        lambda topic, ns, payload, **k: clouds[payload[0]],
    )
    return reads


PROFILE = LivoxStreamProfile("p", "/lidar", "/imu", "t", "offset_s", "mps2")


def test_scan_store_replays_identical_scans_after_one_read(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = _fake_stream(monkeypatch)
    plain = list(livox_ros2.iter_livox_points("bag", PROFILE))
    reads.clear()
    store = ScanStore()

    passes = [list(store.scans("bag", PROFILE)) for _ in range(3)]

    assert len(reads) == 1 and store.hits == 2
    for replay in passes:
        assert len(replay) == len(plain)
        for (t0, xyz0, off0), (t1, xyz1, off1) in zip(plain, replay, strict=True):
            assert t0 == t1 and xyz1.dtype == np.float64 and np.array_equal(xyz0, xyz1)
            assert off0 is not None and off1 is not None and np.array_equal(off0, off1)


def test_scan_store_falls_back_to_rereading_beyond_its_budget_or_on_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = _fake_stream(monkeypatch)
    small = ScanStore(budget_bytes=100)  # smaller than one scan
    first = list(small.scans("bag", PROFILE))
    second = list(small.scans("bag", PROFILE))
    assert len(reads) == 2 and small.bytes_held == 0
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, second, strict=True))

    reads.clear()
    store = ScanStore()
    stream = store.scans("bag", PROFILE)
    next(stream)
    stream.close()  # a consumer that stops early must not leave a truncated cache
    assert len(list(store.scans("bag", PROFILE))) == 6
    assert len(list(store.scans("bag", PROFILE))) == 6
    assert len(reads) == 2 and store.hits == 1  # 1st partial, 2nd full read, 3rd replay


def test_scan_store_keeps_streams_with_different_limits_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_stream(monkeypatch)
    store = ScanStore()
    short = list(store.scans("bag", PROFILE, max_seconds=0.25))
    full = list(store.scans("bag", PROFILE))
    assert (len(short), len(full)) == (3, 6)
    assert len(list(store.scans("bag", PROFILE, max_seconds=0.25))) == 3


def _legacy_preprocess(points: np.ndarray, options: ScanOdometryOptions) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    ranges = np.linalg.norm(xyz, axis=1)
    xyz = xyz[(ranges >= options.min_range_m) & (ranges <= options.max_range_m)]
    keys = np.floor(xyz / options.voxel_size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.size, 3))
    np.add.at(sums, inverse.reshape(-1), xyz)
    return np.asarray(sums / counts[:, None])


@pytest.mark.parametrize("voxel", [0.1, 0.3, 0.5])
def test_fast_voxel_downsampling_is_bit_identical_to_the_row_unique(voxel: float) -> None:
    rng = np.random.default_rng(3)
    options = ScanOdometryOptions(voxel_size_m=voxel, min_range_m=1.0, max_range_m=60.0)
    for scale in (5.0, 25.0):
        points = rng.normal(size=(6000, 3)) * scale
        assert np.array_equal(preprocess_scan(points, options), _legacy_preprocess(points, options))
    # Negative voxel indices and a far outlier must keep the same order.
    points = np.vstack([rng.normal(size=(500, 3)) * 10.0, [[59.0, -59.0, 2.0]]])
    assert np.array_equal(preprocess_scan(points, options), _legacy_preprocess(points, options))


def test_default_estimator_options_are_unchanged() -> None:
    """Benchmarks and SOTA artifacts use these defaults; the check speed-ups must not move them."""

    run = ImuLidarRunOptions()
    assert (run.holdout_every, run.jackknife_groups, run.gyro_deskew_passes) == (3, 8, 5)
    assert (run.deskew_convergence_sigma, run.feedback_check_deg) == (0.25, 2.0)
    assert (run.rotation_control_deg, run.time_control_s, run.coverage_margin_s) == (1.0, 0.01, 0.2)
    odometry = run.windowing.odometry
    assert (odometry.voxel_size_m, odometry.min_range_m, odometry.max_range_m) == (0.3, 1.0, 60.0)
    assert (odometry.local_map_scans, odometry.deskew_iterations) == (5, 1)
    assert (run.windowing.window_duration_s, run.windowing.min_window_scans) == (10.0, 20)
    assert ImuLidarTranslationOptions().windowing == run.windowing


def test_gyro_rotation_model_with_a_shared_start_matches_the_per_point_computation() -> None:
    from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, gyro_rotation_model

    rng = np.random.default_rng(0)
    times = np.arange(0.0, 20.0, 0.0025)
    gyro = GyroSeries(times, rng.normal(size=(len(times), 3)) * 0.5)
    rotation = Rotation.from_euler("xyz", [1, 2, 3], degrees=True).as_matrix()
    bias = np.array([0.01, 0.02, -0.01])
    model = gyro_rotation_model(gyro, rotation, bias, 0.01)

    for count in (1, 7, 3000):
        offsets = np.sort(rng.uniform(0.0, 0.1, count))
        start = np.full(count, 5.0 + 0.01)
        delta, jacobian = gyro.increments(start, start + offsets)
        corrected = delta @ Rotation.from_rotvec(np.einsum("nij,j->ni", jacobian, bias)).as_matrix()
        assert np.array_equal(model(5.0, offsets), rotation @ corrected @ rotation.T)


def test_sqlite_topic_filter_returns_the_same_messages_as_filtering_afterwards(
    tmp_path: Path,
) -> None:
    from calibrex.data.rosbag2 import iter_messages

    bag = write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])
    everything = list(iter_messages(bag))
    topics = {conn.topic for conn, _, _ in everything}
    assert len(topics) > 2
    for wanted in (set(list(topics)[:1]), set(list(topics)[:2]), {"/missing"}):
        got = list(iter_messages(bag, topics=wanted))
        assert got == [item for item in everything if item[0].topic in wanted]


def _imu_at(rate_hz: float) -> livox_ros2.ImuSamples:
    times = np.arange(0.0, 5.0, 1.0 / rate_hz)
    return livox_ros2.ImuSamples(times, np.zeros((len(times), 3)), np.zeros((len(times), 3)))


@pytest.mark.parametrize("rate_hz", [10.0, 15.0])
def test_imu_lidar_skips_an_imu_too_slow_for_the_gyro_coverage_rule(
    monkeypatch: pytest.MonkeyPatch, rate_hz: float
) -> None:
    """KITTI's 10 Hz OXTS: no LiDAR interval has gap-free gyro coverage, so say so up front."""

    samples = _imu_at(rate_hz)
    monkeypatch.setattr(livox_ros2, "load_livox_imu", lambda *a, **k: samples)
    with pytest.raises(estimators.CheckSkipError) as skipped:
        estimators.run_imu_lidar(_imu_lidar_context(np.eye(4), None))
    assert skipped.value.code == "unsupported_sensor"
    assert f"{rate_hz:.1f} Hz" in skipped.value.reason
    assert "/imu" in skipped.value.reason and "20 Hz" in skipped.value.reason


def test_imu_rate_gate_passes_a_fast_enough_imu(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = _imu_at(25.0)
    monkeypatch.setattr(livox_ros2, "load_livox_imu", lambda *a, **k: samples)
    estimators._require_gyro_rate(_imu_lidar_context(np.eye(4), None), "/imu")


def test_the_rate_gate_matches_the_solvers_coverage_rule() -> None:
    """A series at the gate's rate covers nothing; just above it covers its span."""

    from calibrex.solvers.imu_lidar_rotation_solver import MAX_GYRO_GAP_S, GyroSeries

    for rate_hz, covered in ((10.0, False), (25.0, True)):
        times = np.arange(0.0, 5.0, 1.0 / rate_hz)
        gyro = GyroSeries(times, np.zeros((len(times), 3)))
        assert gyro.covers(1.0, 1.3) is covered
    assert pytest.approx(20.0) == 1.0 / MAX_GYRO_GAP_S
