"""NavSatFix messages into the GNSS track of the GNSS-LiDAR estimator."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from tests.unit.check_fixtures import NAVSAT_TYPE
from tests.unit.test_rosbag2 import _write_sqlite_bag

from calibrex.data.navsatfix_track import (
    NavSatFixTrackOptions,
    read_navsatfix_track,
    track_from_navsatfix,
)
from calibrex.data.ros_cdr import decode_ros2_navsatfix
from calibrex.data.ros_cdr_writer import encode_navsatfix
from calibrex.data.ros_messages import NavSatFixMessage
from calibrex.data.rtk_slam import geodetic_to_enu


def fix(
    stamp_s: float,
    *,
    status: int = 0,
    cov_diag: tuple[float, float, float] = (1e-4, 1e-4, 4e-4),
    cov_type: int = 2,
    lat: float = 48.0,
    lon: float = 9.0,
    alt: float = 300.0,
) -> NavSatFixMessage:
    payload = encode_navsatfix(
        secs=int(stamp_s),
        nsecs=round((stamp_s % 1) * 1e9),
        status=status,
        latitude=lat,
        longitude=lon,
        altitude=alt,
        covariance=(cov_diag[0], 0, 0, 0, cov_diag[1], 0, 0, 0, cov_diag[2]),
        covariance_type=cov_type,
    )
    return decode_ros2_navsatfix("/gnss/fix", 0, payload)


def test_status_and_covariance_mapping_and_header_stamps() -> None:
    messages = [
        fix(10.0),
        fix(10.1, status=-1),  # no fix
        fix(10.2, cov_diag=(1.0, 1.0, 4.0)),  # status 0 but metres of std: not RTK
        fix(10.3, cov_type=0, status=0),  # unknown covariance, not GBAS
        fix(10.4, cov_type=0, status=2),  # unknown covariance, GBAS: fallback std
        fix(10.5, lat=math.nan),
        fix(10.6),
    ]
    track, summary = track_from_navsatfix(messages, topic="/gnss/fix")

    assert track.times_s.tolist() == pytest.approx([10.0, 10.4, 10.6])
    # sigma = sqrt(trace(covariance)) like rtk.txt's blt_std; the fallback is recorded
    assert track.sigma_m[0] == pytest.approx(math.sqrt(1e-4 + 1e-4 + 4e-4))
    assert track.sigma_m[1] == pytest.approx(0.05)
    assert (summary.kept, summary.messages) == (3, 7)
    assert (summary.rejected_no_fix, summary.rejected_sigma) == (1, 1)
    assert (summary.rejected_unknown_quality, summary.rejected_invalid) == (1, 1)
    assert summary.fallback_sigma_epochs == 1
    assert summary.covariance_type_counts == {2: 5, 0: 2}
    assert track.rejected_epochs == 4


def test_enu_origin_is_first_kept_fix_and_altitude_is_ellipsoidal() -> None:
    first, second = fix(0.0, alt=300.0), fix(1.0, alt=301.0, lat=48.0001)
    track, _ = track_from_navsatfix([second, first])  # unsorted input is sorted by stamp
    assert track.origin_lat_lon_height == (48.0, 9.0, 300.0)
    expected = geodetic_to_enu(
        np.array([48.0, 48.0001]),
        np.array([9.0, 9.0]),
        np.array([300.0, 301.0]),
        (48.0, 9.0, 300.0),
    )
    np.testing.assert_allclose(track.enu_m, expected)
    assert track.enu_m[1, 2] == pytest.approx(1.0, abs=1e-3)  # +1 m altitude is +1 m up
    pooled, _ = track_from_navsatfix([first, second], origin=(48.0, 9.0, 299.0))
    assert pooled.enu_m[0, 2] == pytest.approx(1.0, abs=1e-6)


def test_too_few_usable_fixes_is_an_explained_error() -> None:
    with pytest.raises(ValueError, match=r"fewer than two usable GNSS fixes of 2.*no fix 2"):
        track_from_navsatfix([fix(0.0, status=-1), fix(1.0, status=-1)], topic="/gnss/fix")


def test_options_are_validated_and_relax_the_filter() -> None:
    with pytest.raises(ValueError):
        NavSatFixTrackOptions(max_sigma_m=0.0)
    loose = NavSatFixTrackOptions(max_sigma_m=10.0)
    track, _ = track_from_navsatfix([fix(0.0), fix(1.0, cov_diag=(1.0, 1.0, 4.0))], loose)
    assert len(track.times_s) == 2


def test_read_from_a_bag(tmp_path: Path) -> None:
    payloads = [
        encode_navsatfix(
            secs=100 + i,
            nsecs=0,
            status=0 if i % 3 else -1,
            covariance=(1e-4, 0, 0, 0, 1e-4, 0, 0, 0, 4e-4),
        )
        for i in range(9)
    ]
    bag = tmp_path / "gnss.db3"
    _write_sqlite_bag(
        bag,
        topics=[("/gnss/fix", NAVSAT_TYPE)],
        messages=[("/gnss/fix", 1_000 + i, payload) for i, payload in enumerate(payloads)],
    )
    track, summary = read_navsatfix_track(bag, "/gnss/fix")
    assert track.times_s.tolist() == pytest.approx([101, 102, 104, 105, 107, 108])
    assert summary.frame_id == "gps"
    assert (summary.first_stamp_s, summary.last_stamp_s) == (100.0, 108.0)
