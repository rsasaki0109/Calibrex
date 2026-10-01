"""The GNSS pairs of ``calibrex check`` (gnss-lidar, gnss-imu) with stubbed LiDAR odometry.

The NavSatFix topic, the planner, the runner, the memo that hands gnss-lidar and
imu-lidar evidence to gnss-imu, the cache and the verdicts run for real. Only what
needs scans is stubbed: the windowed LiDAR odometry and the fit, which are replaced
by the committed RTK-SLAM stadtgarten_seq2 evidence artifacts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tests.unit.check_fixtures import IDENTITY, NAVSAT_TYPE, Transform

from calibrex.check import estimators
from calibrex.check.cache import EstimatorCache
from calibrex.check.estimators import (
    CheckSkipError,
    EstimatorRun,
    EvidenceArtifact,
    PairContext,
    RunControls,
)
from calibrex.check.runner import CheckRunOptions, build_calibration_check, format_check_table
from calibrex.check.verdict import AxisEstimate
from calibrex.cli.main import main
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckPairRecord,
    CheckTransform,
)
from calibrex.core.gnss_lidar_lever_arm import (
    GnssLidarLeverArmArtifact,
    load_gnss_lidar_lever_arm,
)
from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
from calibrex.core.imu_lidar_translation import load_imu_lidar_translation
from calibrex.core.validation import validate_file
from calibrex.data.ros_cdr_writer import encode_header_only, encode_navsatfix, encode_tf_message
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.data.rtk_slam import GnssTrack

ASSETS = Path(__file__).resolve().parents[2] / "docs/assets"
GNSS_ASSET = ASSETS / "rtk_slam_gnss_lidar_stadtgarten_seq2.yaml"
ROTATION_ASSET = ASSETS / "mid360_imu_lidar_rotation_rtk_slam_seq2.yaml"
TRANSLATION_ASSET = ASSETS / "mid360_imu_lidar_translation_rtk_slam_seq2.yaml"
# RTK-SLAM semantics: edges are T_imu_lidar and T_imu_gnss (a CAD point offset).
LIDAR_IN_IMU = (-0.011, -0.023, 0.044)
ANTENNA_IN_IMU = (0.023, -0.023, 0.090)
ANTENNA_IN_LIDAR = tuple(a - b for a, b in zip(ANTENNA_IN_IMU, LIDAR_IN_IMU, strict=True))
BASE_NS = 1_700_000_000_000_000_000


def tf_edges(antenna: tuple[float, float, float] = ANTENNA_IN_IMU) -> list[Transform]:
    return [
        ("imu_link", "lidar_link", LIDAR_IN_IMU, IDENTITY),
        ("imu_link", "gnss_link", antenna, IDENTITY),
    ]


def write_bag(path: Path, *, antenna: tuple[float, float, float] = ANTENNA_IN_IMU) -> Path:
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/tf_static", "tf2_msgs/msg/TFMessage", latched=True)
        writer.write("/tf_static", BASE_NS, encode_tf_message(tf_edges(antenna)))
        writer.add_topic("/livox/points", "sensor_msgs/msg/PointCloud2")
        writer.write("/livox/points", BASE_NS, encode_header_only("lidar_link"))
        writer.add_topic("/livox/imu", "sensor_msgs/msg/Imu")
        writer.write("/livox/imu", BASE_NS, encode_header_only("imu_link"))
        writer.add_topic("/gnss/fix", NAVSAT_TYPE)
        for index in range(40):
            writer.write(
                "/gnss/fix",
                BASE_NS + index * 100_000_000,
                encode_navsatfix(
                    frame_id="gnss_link",
                    secs=BASE_NS // 10**9,
                    nsecs=index * 100_000_000,
                    status=0,
                    latitude=48.0 + index * 1e-7,
                    longitude=9.0,
                    altitude=300.0,
                    covariance=(1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 4e-4),
                    covariance_type=2,
                ),
            )
    return path


class Counters:
    def __init__(self) -> None:
        self.computed = 0
        self.tracks: list[tuple[Any, ...]] = []


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> Counters:
    """Replace the scan-dependent steps of gnss-lidar; the asset is the 'estimate'."""

    counters = Counters()
    asset = load_gnss_lidar_lever_arm(GNSS_ASSET)

    monkeypatch.setattr(
        estimators, "_point_time_or_skip", lambda bag, topic: ("offset_time", "offset_s")
    )
    monkeypatch.setattr(estimators, "_iter_scans_plain", lambda bag, profile, seconds: iter(()))
    import calibrex.evaluation.gnss_lidar_lever_arm as evaluation

    def collect(track: Any, scans: Any, options: Any, *, prefix: str = "") -> Any:
        counters.tracks.append((len(track.times_s), prefix))
        return SimpleNamespace(
            windows=[], scans_read=10, scans_covered=10, segments=1, unreliable=0
        )

    def evaluate(track: Any, windows: Any, options: Any, *, reference_lever_arm: Any = None) -> Any:
        counters.computed += 1
        return None

    def build(evaluation_result: Any, options: Any, **kwargs: Any) -> GnssLidarLeverArmArtifact:
        assert kwargs["provenance"].dataset_family == "calibrex-check"
        assert (
            "NavSatFix /gnss/fix: 40 of 40 fixes usable" in kwargs["provenance"].input_digest_scope
        )
        return asset

    monkeypatch.setattr(evaluation, "collect_windows", collect)
    monkeypatch.setattr(evaluation, "evaluate_gnss_lidar_lever_arm", evaluate)
    monkeypatch.setattr(evaluation, "build_gnss_lidar_artifact", build)
    return counters


def imu_lidar_stub(policy: str = "pass") -> estimators.Adapter:
    rotation = load_imu_lidar_rotation(ROTATION_ASSET)
    translation = load_imu_lidar_translation(TRANSLATION_ASSET)

    def adapter(ctx: PairContext) -> EstimatorRun:
        return EstimatorRun(
            estimator="stub/imu-lidar",
            artifacts=(
                EvidenceArtifact("rotation", rotation, rotation.policy_status),
                EvidenceArtifact("translation", translation, translation.policy_status),
            ),
            solved=policy != "unsolved",
            policy_status="fail" if policy == "fail" else "pass",
            policy_reasons=("stub",),
            estimates=(AxisEstimate("roll", "deg", 0.0, 0.1, True),),
            compared=np.eye(4),
        )

    return adapter


def pair(artifact: CalibrationCheckArtifact, name: str) -> CheckPairRecord:
    (record,) = [item for item in artifact.pairs if item.pair == name]
    return record


def run_check(
    bag: Path,
    out: Path,
    *,
    pairs: tuple[str, ...] = ("imu-lidar", "gnss-lidar", "gnss-imu"),
    cache: Path | None = None,
) -> CalibrationCheckArtifact:
    return build_calibration_check(
        bag,
        run=CheckRunOptions(pairs=pairs, evidence_dir=out / "ev", base_dir=out, cache_dir=cache),
    )


def test_gnss_pairs_pass_with_the_cad_antenna_and_leave_rotation_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed: Counters
) -> None:
    monkeypatch.setitem(estimators.ESTIMATORS, "imu-lidar", imu_lidar_stub())
    artifact = run_check(write_bag(tmp_path / "bag"), tmp_path)

    lidar = pair(artifact, "gnss-lidar")
    assert lidar.status == "pass", lidar.reason
    assert [axis.name for axis in lidar.axes] == ["x", "y"]  # z: 4 cm std, unobservable
    assert {item.name for item in lidar.unchecked_axes} == {"roll", "pitch", "yaw", "z"}
    rotation = next(item for item in lidar.unchecked_axes if item.name == "yaw")
    assert "orientation is not defined" in rotation.reason
    assert lidar.coverage == "partial"
    # the candidate is the antenna position in the LiDAR frame: parent lidar, child gnss
    assert (lidar.compared_transform.parent_frame, lidar.compared_transform.child_frame) == (
        "lidar_link",
        "gnss_link",
    )
    assert lidar.compared_transform.translation_m == pytest.approx(ANTENNA_IN_LIDAR)
    assert lidar.evidence[0].role == "lever_arm"

    imu = pair(artifact, "gnss-imu")
    assert imu.status == "pass", imu.reason
    assert imu.compared_transform.translation_m == pytest.approx(ANTENNA_IN_IMU)
    assert [axis.name for axis in imu.axes] == ["x", "y"]
    assert imu.evidence[0].role == "composed_lever_arm"
    assert validate_file(tmp_path / imu.evidence[0].path).kind == "gnss-imu-lever-arm"
    assert validate_file(tmp_path / lidar.evidence[0].path).kind == "gnss-lidar-lever-arm"
    assert stubbed.tracks == [(40, "bag/")]
    assert "gnss-imu" in format_check_table(artifact)


def test_a_wrong_antenna_offset_fails_both_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed: Counters
) -> None:
    monkeypatch.setitem(estimators.ESTIMATORS, "imu-lidar", imu_lidar_stub())
    wrong = (ANTENNA_IN_IMU[0] + 0.15, ANTENNA_IN_IMU[1], ANTENNA_IN_IMU[2])
    artifact = run_check(write_bag(tmp_path / "bag", antenna=wrong), tmp_path)

    for name in ("gnss-lidar", "gnss-imu"):
        record = pair(artifact, name)
        assert record.status == "fail", name
        x = next(axis for axis in record.axes if axis.name == "x")
        assert x.candidate_error == pytest.approx(0.15, abs=0.03)
        assert x.status == "fail"


def test_gnss_imu_is_skipped_without_its_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed: Counters
) -> None:
    monkeypatch.setitem(estimators.ESTIMATORS, "imu-lidar", imu_lidar_stub())
    bag = write_bag(tmp_path / "bag")

    only_imu = run_check(bag, tmp_path, pairs=("gnss-imu",))
    record = pair(only_imu, "gnss-imu")
    assert (record.status, record.reason_code) == ("skipped", "missing_dependency")
    assert "gnss-lidar and imu-lidar gave no result (not selected with --pairs, or skipped)" in (
        record.reason or ""
    )

    without_gnss = run_check(bag, tmp_path, pairs=("imu-lidar", "gnss-imu"))
    assert "gnss-lidar gave no result" in (pair(without_gnss, "gnss-imu").reason or "")

    monkeypatch.setitem(estimators.ESTIMATORS, "imu-lidar", imu_lidar_stub("fail"))
    failed = run_check(bag, tmp_path)
    record = pair(failed, "gnss-imu")
    assert (record.status, record.reason_code) == ("skipped", "missing_dependency")
    assert "imu-lidar has no usable rotation" in (record.reason or "")
    assert pair(failed, "gnss-lidar").status == "pass"


def test_cache_key_ignores_the_candidate_and_rebases_the_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed: Counters
) -> None:
    monkeypatch.setitem(estimators.ESTIMATORS, "imu-lidar", imu_lidar_stub())
    cache = tmp_path / "cache"
    first = run_check(
        write_bag(tmp_path / "a"), tmp_path / "o1", pairs=("gnss-lidar",), cache=cache
    )
    assert (pair(first, "gnss-lidar").evidence_from_cache, stubbed.computed) == (False, 1)

    # another candidate over the same bag content (same digest): no new fit, reference rebased
    shifted = write_bag(tmp_path / "b", antenna=(ANTENNA_IN_IMU[0] + 0.05, *ANTENNA_IN_IMU[1:]))
    monkeypatch.setattr(
        "calibrex.check.runner.bag_input_digest",
        lambda bag: ("0" * 64, "scope", None),
    )
    second = run_check(shifted, tmp_path / "o2", pairs=("gnss-lidar",), cache=cache)
    third = run_check(
        write_bag(tmp_path / "c"), tmp_path / "o3", pairs=("gnss-lidar",), cache=cache
    )
    assert pair(second, "gnss-lidar").evidence_from_cache is False  # digest "0"*64 first seen
    assert pair(third, "gnss-lidar").evidence_from_cache is True
    assert stubbed.computed == 2
    cached = load_gnss_lidar_lever_arm(tmp_path / "o3" / pair(third, "gnss-lidar").evidence[0].path)
    x = next(item for item in cached.dofs if item.name == "x")
    assert x.reference_value == pytest.approx(ANTENNA_IN_LIDAR[0])
    assert x.error_to_reference == pytest.approx(x.value - ANTENNA_IN_LIDAR[0])


def test_cache_key_covers_the_gnss_options_and_duration() -> None:
    from calibrex.data.navsatfix_track import NavSatFixTrackOptions
    from calibrex.evaluation.gnss_lidar_lever_arm import GnssLidarRunOptions

    cache = EstimatorCache(Path("/nonexistent"), code="x")
    base = {"track": NavSatFixTrackOptions(), "options": GnssLidarRunOptions(), "max_seconds": 60}
    key = cache.key("d", "gnss_lidar_lever_arm", base)
    for change in (
        {"track": NavSatFixTrackOptions(max_sigma_m=0.05)},
        {"options": GnssLidarRunOptions(window_duration_s=5.0)},
        {"max_seconds": 120},
    ):
        assert cache.key("d", "gnss_lidar_lever_arm", {**base, **change}) != key


def test_non_navsatfix_gnss_topic_is_skipped(tmp_path: Path) -> None:
    ctx = PairContext(
        bag=tmp_path,
        pair=CheckPairRecord(
            pair="gnss-lidar",
            sensors=["g", "l"],
            frames=["g", "l"],
            status="planned",
            candidate_transform=CheckTransform(
                parent_frame="g",
                child_frame="l",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
        ),
        topic_types={"/gps": "nav_msgs/msg/Odometry", "/pts": "sensor_msgs/msg/PointCloud2"},
        sensor_topics={"gnss": ("/gps",), "lidar": ("/pts",)},
        candidate=np.eye(4),
        sources=(),
        controls=RunControls(),
    )
    with pytest.raises(CheckSkipError, match="not sensor_msgs/NavSatFix"):
        estimators.run_gnss_lidar(ctx)


def test_unusable_gnss_track_skips_the_pair_and_its_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed: Counters
) -> None:
    bag = write_bag(tmp_path / "bag")

    def refuse(*args: Any, **kwargs: Any) -> tuple[GnssTrack, Any]:
        raise ValueError("/gnss/fix: fewer than two usable GNSS fixes of 40")

    monkeypatch.setattr("calibrex.data.navsatfix_track.read_navsatfix_track", refuse)
    artifact = run_check(bag, tmp_path, pairs=("gnss-lidar", "gnss-imu"))
    record = pair(artifact, "gnss-lidar")
    assert (record.status, record.reason_code) == ("skipped", "unsupported_sensor")
    assert "fewer than two usable GNSS fixes" in (record.reason or "")
    assert "RTK-grade" in (record.reason or "")
    assert pair(artifact, "gnss-imu").reason_code == "missing_dependency"


def test_cli_runs_the_gnss_pairs_on_a_synthetic_bag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stubbed: Counters,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(estimators.ESTIMATORS, "imu-lidar", imu_lidar_stub())
    bag = write_bag(tmp_path / "bag")
    output = tmp_path / "check.yaml"
    code = main(
        [
            "check", str(bag), "--pairs", "imu-lidar,gnss-lidar,gnss-imu", "--no-cache",
            "--evidence-dir", str(tmp_path / "ev"), "--output", str(output),
        ]
    )  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert "gnss-lidar" in printed and "gnss-imu" in printed
    assert validate_file(output).kind == "calibration-check"
