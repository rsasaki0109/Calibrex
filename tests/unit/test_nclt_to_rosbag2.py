"""The NCLT converter's frames, odometry twist, hit packets and per-turn scans."""

from __future__ import annotations

import io
import math
import struct
from typing import ClassVar

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.nclt_to_rosbag2 import (
    MAGIC,
    hits_to_points,
    odometry_twist,
    packets,
    reference_base_velodyne,
    revolutions,
    ssc_to_matrix,
)


def test_ssc_is_heading_pitch_roll() -> None:
    matrix = ssc_to_matrix((1.0, 2.0, 3.0, 10.0, 20.0, 30.0))
    expected = (
        Rotation.from_euler("z", 30, degrees=True)
        * Rotation.from_euler("y", 20, degrees=True)
        * Rotation.from_euler("x", 10, degrees=True)
    )
    assert matrix[:3, :3] == pytest.approx(expected.as_matrix())
    assert matrix[:3, 3] == pytest.approx([1.0, 2.0, 3.0])


def test_the_upside_down_velodyne_sits_above_base_link() -> None:
    reference = reference_base_velodyne()
    # NCLT's body z points down: the LiDAR 0.957 m above the axle, its own z pointing down
    assert reference[2, 3] == pytest.approx(0.957)
    assert reference[2, 2] < -0.99
    # heading -90.7 deg in a z-down body is about +90.7 deg about base_link's z (up)
    assert reference[:3, 0] @ [0.0, 1.0, 0.0] > 0.99


def test_odometry_twist_of_a_left_turn() -> None:
    """A circle driven left at 1.5 m/s; NCLT's local frame and body are z down."""

    rate = 0.2  # rad/s, counter-clockwise seen from above
    t = np.arange(0.0, 10.0, 0.01)
    # z down: seen from above, a left turn is a negative heading rate
    heading = -rate * t
    radius = 1.5 / rate
    x = radius * np.sin(rate * t)
    y = -radius * (1.0 - np.cos(rate * t))
    rows = np.column_stack(
        [1.0e15 + t * 1.0e6, x, y, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), heading]
    )

    _utime, position, _quaternion, twist = odometry_twist(rows)

    assert np.median(twist[:, 0]) == pytest.approx(1.5, abs=1e-3)
    assert np.median(twist[:, 1]) == pytest.approx(rate, abs=1e-6)
    assert position[100, 1] > 0.0  # a left turn ends up to the left (y up in REP-103)


def _packet(utime: int, hits: np.ndarray) -> bytes:
    header = MAGIC + struct.pack("<IQ", len(hits), utime) + b"\0" * 4
    return header + hits.astype("<u2").tobytes()


def test_packets_and_hits() -> None:
    def raw(x: float, y: float, z: float) -> list[int]:
        return [round((v + 100.0) / 0.005) for v in (x, y, z)] + [7 * 256 + 3]

    hits = np.array([raw(1.0, -2.0, 0.5), [20000, 20000, 20000, 0]])
    stream = io.BytesIO(_packet(123, hits) + _packet(456, hits[:1]))

    read = list(packets(stream))

    assert [utime for utime, _hits in read] == [123, 456]
    points = hits_to_points(read[0][1])
    assert points.shape == (1, 3)  # the no-return hit is dropped
    assert points[0] == pytest.approx([1.0, -2.0, 0.5], abs=0.003)


def test_revolutions_are_whole_turns_of_the_head() -> None:
    """A head turning clockwise, one packet per 3.7 deg and 1.5 ms: turns of 97-98 packets."""

    step = math.radians(-3.7)
    items = []
    for k in range(1000):
        azimuth = 0.3 + k * step
        points = np.array([[10.0 * math.cos(azimuth), 10.0 * math.sin(azimuth), 0.0]] * 3)
        items.append((k * 1500, points))
        if k % 7 == 0:
            items.append((k * 1500 + 1, np.zeros((0, 3))))  # a packet without returns

    turns = list(revolutions(items))

    # 1000 packets of 3.7 deg are 10.3 turns: 10 whole ones, the partial last one dropped
    assert len(turns) == 10
    for turn in turns:
        assert len([p for _t, p in turn if len(p)]) in (97, 98)
    period_us = 360.0 / 3.7 * 1500
    assert np.diff([turn[0][0] for turn in turns]) == pytest.approx(period_us, abs=1500)


class _FakeCurl:
    """A curl that sends ``data`` from its Range start and stops after ``cut`` bytes."""

    calls: ClassVar[list[list[str]]] = []
    data: ClassVar[bytes] = b""
    cut: ClassVar[int] = 0

    def __init__(self, command: list[str], stdout: object = None) -> None:
        type(self).calls.append(command)
        start = int(command[command.index("-r") + 1].rstrip("-")) if "-r" in command else 0
        self._cut = len(type(self).data) if len(type(self).calls) > 1 else type(self).cut
        self.stdout = io.BufferedReader(io.BytesIO(type(self).data[start : self._cut]))
        self._code = 0 if self._cut == len(type(self).data) else 56

    def wait(self) -> int:
        return self._code

    def poll(self) -> int:
        return self._code

    def kill(self) -> None:
        pass


def test_a_cut_off_transfer_resumes_where_it_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    import gzip
    import tarfile

    from tools import nclt_to_rosbag2 as converter

    payload = np.random.default_rng(3).integers(0, 256, 300_000, dtype=np.uint8).tobytes()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("2012-01-08/velodyne_hits.bin")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    _FakeCurl.data = gzip.compress(buffer.getvalue())
    _FakeCurl.cut = len(_FakeCurl.data) // 3
    _FakeCurl.calls = []
    monkeypatch.setattr(converter.subprocess, "Popen", _FakeCurl)
    monkeypatch.setattr(converter.time, "sleep", lambda _s: None)

    download, archive = converter.velodyne_stream("2012-01-08", "abc-1")
    member = next(iter(archive))
    stream = archive.extractfile(member)
    assert stream is not None and stream.read() == payload
    download.close()

    assert len(_FakeCurl.calls) == 2 and download.failures == 1
    resumed = _FakeCurl.calls[1]
    assert resumed[resumed.index("-r") + 1] == f"{_FakeCurl.cut}-"
    assert 'If-Match: "abc-1"' in resumed
