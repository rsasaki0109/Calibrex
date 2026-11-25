"""The rotation input window uses image stamps and preserves mono16 byte order."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal
from pytest import MonkeyPatch

from calibrex.data import rosbag1
from calibrex.data.frame_sources import bag_frames


def test_ros1_bag_window_and_stride(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    bag = tmp_path / "capture.bag"
    bag.touch()
    stamp = 1_700_000_000_000_000_000

    def read_images(path: Path, *, topic: str, include_data: bool) -> list[SimpleNamespace]:
        assert (path, topic, include_data) == (bag, "/cam/image_raw", True)
        return [
            SimpleNamespace(
                timestamp_ns=stamp + i * 1_000_000_000,
                height=1,
                width=2,
                step=6,
                encoding="mono16",
                is_bigendian=True,
                data=bytes([i, 255, 200, 9, 3, 4]),
            )
            for i in range(5)
        ]

    monkeypatch.setattr(rosbag1, "read_image_messages", read_images)
    frames = list(bag_frames(bag, "/cam/image_raw", stride=2, start_seconds=1, max_seconds=2))
    assert len(frames) == 1
    assert frames[0][0] == 1_700_000_002
    assert_array_equal(frames[0][1], np.asarray([[2, 200]], dtype=np.uint8))
