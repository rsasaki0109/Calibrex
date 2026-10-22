"""Metadata readiness must never expose a candidate's sensor/reference contents."""

from __future__ import annotations

import hashlib
import urllib.parse
import xml.etree.ElementTree as ET

import jsonschema
import pytest
from tools.boreas_ins_lidar_metadata import (
    MAX_LISTING_BYTES,
    BoreasInsLidarMetadata,
    MetadataProvenance,
    _Inventory,
    fetch_listing,
    inventory,
)

NAMES = ["boreas-2020-11-26-13-58/", "boreas-2020-12-01-13-26/", "boreas-2020-12-04-14-00/"]
NS = "http://s3.amazonaws.com/doc/2006-03-01/"


def listing(prefix="", *, keys=(), prefixes=(), truncated=False, token=None):
    root = ET.Element("ListBucketResult", xmlns=NS)
    for key, value in [
        ("Name", "boreas"),
        ("Prefix", prefix),
        ("IsTruncated", "true" if truncated else "false"),
    ]:
        ET.SubElement(root, key).text = value
    for name in prefixes:
        ET.SubElement(ET.SubElement(root, "CommonPrefixes"), "Prefix").text = name
    for key in keys:
        node = ET.SubElement(root, "Contents")
        for field, value in [
            ("Key", key),
            ("Size", "100"),
            ("ETag", '"object-etag"'),
            ("LastModified", "2026-10-22T00:00:00Z"),
        ]:
            ET.SubElement(node, field).text = value
    if token:
        ET.SubElement(root, "NextContinuationToken").text = token
    return ET.tostring(root)


def provenance():
    return MetadataProvenance(
        created_at="2026-10-22T00:00:00Z",
        git_commit="test",
        source_sha256="a" * 64,
        command=["metadata test"],
    )


def transport(*, missing=None):
    requests = []

    def fetch(url):
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        assert parsed.path == "/" and query["list-type"] == ["2"]
        prefix = query["prefix"][0]
        requests.append(url)
        if not prefix:
            return listing(
                prefixes=[
                    *reversed(NAMES),
                    "boreas-2024-12-03-10-24/",
                    "boreas-objects-v1/",
                    "boreas-2020-11-26-calib/",
                ]
            )
        if prefix.endswith("lidar/"):
            return listing(
                prefix, keys=[prefix + "1000000000000000.bin", prefix + "1000000000100000.bin"]
            )
        if prefix.endswith("applanix/"):
            keys = [] if prefix == missing else [prefix + "gps_post_process.csv"]
            return listing(prefix, keys=keys)
        assert prefix.endswith("calib/")
        return listing(prefix, keys=[prefix + "T_applanix_lidar.txt"])

    return fetch, requests


def test_deterministic_metadata_only_roles_and_source_pins():
    fetch, calls = transport()
    result = inventory(provenance(), fetch=fetch, frame_limit=2)
    assert [r.recording_id for r in result.recordings] == [p[:-1] for p in NAMES]
    assert [r.role for r in result.recordings] == [
        "development_candidate",
        "evaluation_candidate",
        "evaluation_candidate",
    ]
    assert all(r.selected_lidar_bytes == 200 and not r.missing_required for r in result.recordings)
    assert len(calls) == len(result.requests) == 10
    assert all(r.reference_trajectory.etag == "object-etag" for r in result.recordings)
    assert result.sensor_object_download_bytes == result.calibration_object_download_bytes == 0
    assert not result.provenance.accuracy_claim and not result.provenance.exposure_audit_complete
    assert result.requests[0].sha256 == hashlib.sha256(fetch(calls[0])).hexdigest()
    jsonschema.validate(result.model_dump(mode="json"), BoreasInsLidarMetadata.model_json_schema())


def test_missing_reference_is_reported_without_replacing_the_recording():
    fetch, _ = transport(missing=NAMES[1] + "applanix/")
    result = inventory(provenance(), fetch=fetch, frame_limit=3)
    assert result.recordings[1].recording_id == NAMES[1][:-1]
    assert result.recordings[1].reference_trajectory is None
    assert result.recordings[1].missing_required == [
        NAMES[1] + "applanix/gps_post_process.csv",
        "3 LiDAR scans (found 2)",
    ]


def test_lidar_limit_stops_before_requesting_unneeded_pages():
    calls = []

    def fetch(url):
        calls.append(url)
        return listing("p/", keys=["p/1.bin", "p/2.bin"], truncated=True, token="next")

    values = _Inventory(fetch).objects("p/", limit=2)
    assert len(values) == 2 and len(calls) == 1


@pytest.mark.parametrize(
    "keys", [["p/2.bin", "p/1.bin"], ["p/1.bin", "p/1.bin"], ["outside/1.bin"]]
)
def test_unsorted_duplicate_or_outside_objects_are_rejected(keys):
    with pytest.raises(ValueError, match="unique, sorted and inside"):
        _Inventory(lambda _: listing("p/", keys=keys)).objects("p/")


@pytest.mark.parametrize("token", [None, "same"])
def test_broken_pagination_cannot_loop(token):
    with pytest.raises(ValueError, match="pagination token"):
        list(_Inventory(lambda _: listing(truncated=True, token=token)).pages(""))


def test_object_get_urls_are_refused_before_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network must not be reached")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    for url in [
        "https://boreas.s3.us-west-2.amazonaws.com/file.bin",
        "https://other.example/?list-type=2",
    ]:
        with pytest.raises(ValueError, match="only S3"):
            fetch_listing(url)


def test_oversized_listing_is_rejected_before_parsing():
    with pytest.raises(ValueError, match="byte bound"):
        list(_Inventory(lambda _: b"x" * (MAX_LISTING_BYTES + 1)).pages(""))


@pytest.mark.parametrize("limit", [0, 601])
def test_invalid_frame_limits_do_not_read_anything(limit):
    with pytest.raises(ValueError, match="frame_limit"):
        inventory(provenance(), fetch=lambda _: pytest.fail("must not fetch"), frame_limit=limit)
