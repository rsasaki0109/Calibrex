"""INS audits download only needed streams and reject over-budget payloads early."""

from __future__ import annotations

import zipfile

import pytest
from tools import kitti_drive_fetch as fetch

PREFIX = "2011_09_29/2011_09_29_drive_0071_sync/"


def entry(relative):
    return zipfile.ZipInfo(PREFIX + relative)


def test_ins_selection_excludes_images_and_keeps_both_required_timestamp_streams():
    infos = [
        entry(name)
        for name in (
            "image_02/timestamps.txt",
            "image_02/data/0000000000.png",
            "oxts/timestamps.txt",
            "oxts/dataformat.txt",
            "oxts/data/0000000000.txt",
            "velodyne_points/timestamps.txt",
            "velodyne_points/data/0000000000.bin",
            "velodyne_points/data/0000000002.bin",
        )
    ]
    selected = fetch.select(infos, PREFIX, 2, streams=("velodyne_points", "oxts"))
    names = [i.filename[len(PREFIX) :] for i in selected]
    assert names == [
        "oxts/timestamps.txt",
        "oxts/dataformat.txt",
        "oxts/data/0000000000.txt",
        "velodyne_points/timestamps.txt",
        "velodyne_points/data/0000000000.bin",
    ]
    assert len(fetch.select(infos, PREFIX, 2)) == len(selected) + 2


@pytest.mark.parametrize("streams", [(), ("oxts", "oxts"), ("unknown",)])
def test_invalid_streams_are_rejected_before_network(tmp_path, monkeypatch, streams):
    monkeypatch.setattr(fetch, "head", lambda *a: pytest.fail("must not fetch"))
    with pytest.raises(ValueError, match="streams"):
        fetch.fetch_drive("2011_09_29", "0071", 600, tmp_path, streams=streams)


def test_range_budget_is_checked_before_sensor_bytes_are_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "head", lambda _: (1000, "pin"))
    monkeypatch.setattr(fetch, "central_directory", lambda *a: [entry("oxts/data/0000000000.txt")])
    monkeypatch.setattr(fetch, "spans", lambda *a: [(0, 100)])
    monkeypatch.setattr(fetch, "fetch_range", lambda *a: pytest.fail("must not read sensor bytes"))
    with pytest.raises(ValueError, match="above 100 cap"):
        fetch.fetch_drive("2011_09_29", "0071", 600, tmp_path, max_fetch_bytes=100)
