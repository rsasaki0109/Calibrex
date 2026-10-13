"""Velodyne packet decoding against the clouds LidarView decoded (opt-in).

Needs ``python3 tools/download_public_dataset.py lidarview_velodyne`` (Kitware's
Apache-2.0 LidarView test recordings, about 210 MB) under ``data/public`` or at
``CALIBREX_LIDARVIEW_VELODYNE``. Every reference point must have a decoded point
nearby and the per-point time must agree with LidarView's to a microsecond.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial import cKDTree
from tools.validate_velodyne_decode import (
    compare_lidarview_frame,
    read_pcap_payloads,
    read_vtp_point_data,
)

from calibrex.core.exceptions import DatasetError
from calibrex.data import velodyne_packets
from calibrex.data.velodyne_packets import decode_velodyne_packet, load_velodyne_calibration

ROOT = Path(os.environ.get("CALIBREX_LIDARVIEW_VELODYNE", "data/public/lidarview_velodyne"))
DATA = ROOT / "Testing" / "Data"
CALIBRATIONS = ROOT / "VelodynePacketInterpreters" / "CalibrationFiles"
HDL64_CALIBRATION = CALIBRATIONS / "HDL_64.xml"
VLS128 = DATA / "HDL_128"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATA.is_dir(),
        reason="download lidarview_velodyne or set CALIBREX_LIDARVIEW_VELODYNE",
    ),
]


@pytest.mark.parametrize(
    ("recording", "packet_format", "max_median_mm"),
    [
        ("VLP_16_Single", "VLP-16 strongest return", 0.2),
        ("VLP_16_Dual", "VLP-16 dual return", 0.5),
        # Despite its name this recording is single (strongest) return.
        ("VLP_32c_Dual", "VLP-32C strongest return", 0.2),
        ("HDL_64_Single", "HDL-64E single return", 0.01),
        ("HDL_64_Dual", "HDL-64E dual return", 0.01),
    ],
)
def test_decoded_points_match_lidarview(
    recording: str, packet_format: str, max_median_mm: float
) -> None:
    calibration = load_velodyne_calibration(HDL64_CALIBRATION)
    payloads = read_pcap_payloads(DATA / recording / f"{recording}.pcap")

    result = compare_lidarview_frame(
        payloads, DATA / recording / "GroundTruth" / f"{recording}_1.vtp", calibration
    )

    assert result.packet_format.startswith(packet_format)
    assert result.median_m * 1e3 <= max_median_mm
    assert result.within_1mm >= 0.97
    assert abs(result.decoded_points - result.reference_points) <= 0.01 * result.reference_points
    assert result.time_spread_us <= 1.0


def test_vlp32ab_recording_is_refused_by_name() -> None:
    payloads = read_pcap_payloads(DATA / "VLP_32c_Single" / "VLP_32c_Single.pcap")

    with pytest.raises(DatasetError, match="VLP-32A/B"):
        decode_velodyne_packet(payloads[0])


def _reference_order_calibration(frame: Path, path: Path) -> Path:
    """The per-laser angles LidarView used for a frame, read back from that frame.

    LidarView's 2019 VLS-128 reference was decoded with a calibration that is no longer
    published: the same 128 (elevation, azimuth offset) pairs as ``Alpha_Prime.xml``, in
    another channel order. The pairs are read from frame 0; frames 1 and 2 then test
    everything else the decoder does (banks, units, azimuth advance, timing).
    """

    arrays = read_vtp_point_data(frame)
    laser = arrays["laser_id"].astype(int)
    azimuth_offset = (
        arrays["azimuth"] * 0.01 - np.degrees(np.arctan2(arrays["X"], arrays["Y"])) + 180.0
    ) % 360.0 - 180.0
    rows = []
    for index in range(128):
        mine = laser == index
        assert mine.any(), f"laser {index} has no point in {frame.name}"
        rows.append(
            {
                "laser_id": index,
                "rot_correction": math.radians(round(float(np.median(azimuth_offset[mine])), 3)),
                "vert_correction": math.radians(
                    round(float(np.median(arrays["vertical_angle"][mine])), 3)
                ),
                "dist_correction": 0.0,
            }
        )
    path.write_text(yaml.safe_dump({"lasers": rows, "distance_resolution": 0.004}))
    return path


@pytest.mark.skipif(not VLS128.is_dir(), reason="download the HDL_128 files of lidarview_velodyne")
def test_vls128_matches_lidarview_on_held_out_frames(tmp_path: Path) -> None:
    published = load_velodyne_calibration(CALIBRATIONS / "Alpha_Prime.xml")
    assert len(published.lasers) == 128 and published.distance_resolution_m == 0.004
    calibration = load_velodyne_calibration(
        _reference_order_calibration(VLS128 / "GroundTruth" / "HDL_128_0.vtp", tmp_path / "c.yaml")
    )
    pairs = {
        (round(item.vert_correction_deg, 3), round(item.rot_correction_deg, 3))
        for item in published.lasers
    }
    shared = sum(
        (round(item.vert_correction_deg, 3), round(item.rot_correction_deg, 3)) in pairs
        for item in calibration.lasers
    )
    assert shared == 127  # one laser (channel 38) is (0, 0) in the reference
    payloads = read_pcap_payloads(VLS128 / "HDL_128_2019-10-25_25Kpackets.pcap")

    for frame in ("HDL_128_1.vtp", "HDL_128_2.vtp"):
        result = compare_lidarview_frame(payloads, VLS128 / "GroundTruth" / frame, calibration)
        assert result.packet_format.startswith("VLS-128 strongest return")
        assert result.median_m < 1e-9 and result.p90_m < 1e-6
        assert result.within_1mm >= 0.98
        assert abs(result.decoded_points - result.reference_points) <= 5


@pytest.mark.skipif(not VLS128.is_dir(), reason="download the HDL_128 files of lidarview_velodyne")
def test_vls128_times_match_lidarview_with_the_firmware_5_1_cycle(tmp_path: Path) -> None:
    calibration = load_velodyne_calibration(
        _reference_order_calibration(VLS128 / "GroundTruth" / "HDL_128_0.vtp", tmp_path / "c.yaml")
    )
    arrays = read_vtp_point_data(VLS128 / "GroundTruth" / "HDL_128_1.vtp")
    reference = np.stack([arrays["X"], arrays["Y"], arrays["Z"]], axis=1)
    reference_time = arrays["timestamp"].astype(np.float64)
    xyz, times = [], []
    for payload in read_pcap_payloads(VLS128 / "HDL_128_2019-10-25_25Kpackets.pcap"):
        stamp_us = int.from_bytes(payload[1200:1204], "little")
        if not reference_time.min() <= stamp_us <= reference_time.max():
            continue
        # The recording predates firmware 5.2.3.0: LidarView stamped it with the
        # earlier 2.66666 us cycle, and rounds each adjustment to a microsecond.
        points = velodyne_packets._decode_vls128_packet(payload, calibration, cycle_us=2.66666)
        xyz.append(points.xyz)
        times.append(stamp_us + np.floor(points.time_s * 1e6 + 0.5))
    distance, index = cKDTree(np.concatenate(xyz)).query(reference)
    matched = distance < 1e-3
    assert matched.mean() >= 0.98
    delta = reference_time[matched] - np.concatenate(times)[index[matched]]
    assert np.all(delta == delta[0])  # one stamp convention, no per-point error
