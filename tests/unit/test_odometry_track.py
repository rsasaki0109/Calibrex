"""Tests for explicit odometry pose-stream preprocessing."""

from __future__ import annotations

import pytest

from calibrex.core.geometry import SE3
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack


def _samples() -> list[OdometryPoseSample]:
    return [
        OdometryPoseSample(0, SE3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        OdometryPoseSample(10_000_000, SE3((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        OdometryPoseSample(10_050_000, SE3((1.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
        OdometryPoseSample(20_000_000, SE3((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))),
    ]


def test_odometry_track_preserves_bursts_by_default() -> None:
    track = OdometryTrack(_samples())

    assert track.message_count == 4
    assert track.raw_message_count == 4
    assert track.burst_policy == "preserve"
    assert track.burst_removed_count == 0


def test_odometry_track_keep_first_reduces_close_samples() -> None:
    track = OdometryTrack(_samples(), burst_policy="keep_first", min_interval_s=0.001)

    assert track.message_count == 3
    assert track.raw_message_count == 4
    assert track.burst_removed_count == 1
    assert [sample.timestamp_ns for sample in track.samples] == [0, 10_000_000, 20_000_000]


def test_odometry_track_keep_last_retains_latest_close_sample() -> None:
    track = OdometryTrack(_samples(), burst_policy="keep_last", min_interval_s=0.001)

    assert track.message_count == 3
    assert track.burst_removed_count == 1
    assert [sample.timestamp_ns for sample in track.samples] == [0, 10_050_000, 20_000_000]
    assert track.samples[1].pose.translation_m == (1.1, 0.0, 0.0)


def test_odometry_track_rejects_non_positive_burst_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        OdometryTrack(_samples(), burst_policy="keep_last", min_interval_s=0.0)
