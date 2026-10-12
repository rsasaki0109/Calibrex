"""Velodyne packet decoding against the clouds LidarView decoded (opt-in).

Needs ``python3 tools/download_public_dataset.py lidarview_velodyne`` (Kitware's
Apache-2.0 LidarView test recordings, about 160 MB) under ``data/public`` or at
``CALIBREX_LIDARVIEW_VELODYNE``. Every reference point must have a decoded point
nearby and the per-point time must agree with LidarView's to a microsecond.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from tools.validate_velodyne_decode import compare_lidarview_frame, read_pcap_payloads

from calibrex.core.exceptions import DatasetError
from calibrex.data.velodyne_packets import decode_velodyne_packet, load_velodyne_calibration

ROOT = Path(os.environ.get("CALIBREX_LIDARVIEW_VELODYNE", "data/public/lidarview_velodyne"))
DATA = ROOT / "Testing" / "Data"
HDL64_CALIBRATION = ROOT / "VelodynePacketInterpreters" / "CalibrationFiles" / "HDL_64.xml"

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
