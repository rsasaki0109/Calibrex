"""Request metadata reuse preserves fresh reads, independent values and lifetime."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import yaml

from calibrex.core.exceptions import DatasetError
from calibrex.data import rosbag2 as bag

TEXT = """rosbag2_bagfile_information:
  storage_identifier: sqlite3
  relative_file_paths: [data.db3]
  topics_with_message_count:
    - topic_metadata: {name: /lidar/points, type: sensor_msgs/msg/PointCloud2}
      message_count: 1
"""


def metadata(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.yaml"
    path.write_text(TEXT, encoding="utf-8")
    return path


def observe(monkeypatch):
    calls = []
    original = bag._parse_metadata

    def parse(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(bag, "_parse_metadata", parse)
    return calls


def test_only_one_parse_in_a_request_and_no_retained_data_after_it(tmp_path, monkeypatch):
    path = metadata(tmp_path)
    calls = observe(monkeypatch)
    reference = bag._load_metadata(path)
    assert bag._METADATA_READ_CACHE.get() is None
    with bag._metadata_read_scope():
        state = bag._METADATA_READ_CACHE.get()
        for _ in range(19):
            assert bag._load_metadata(path) == reference
        assert len(calls) == 2 and state.entry is not None
        assert bag._metadata_entry_fits(state.entry, state.max_bytes)
    assert state.entry is None and bag._METADATA_READ_CACHE.get() is None
    bag._load_metadata(path)
    with bag._metadata_read_scope():
        bag._load_metadata(path)
    assert len(calls) == 4


def test_same_size_edit_with_restored_mtime_is_seen_and_changes_digest(tmp_path, monkeypatch):
    from calibrex.check.runner import bag_input_digest

    path = metadata(tmp_path)
    (tmp_path / "data.db3").write_bytes(b"fixture storage bytes")
    calls = observe(monkeypatch)
    with bag._metadata_read_scope():
        assert bag._load_metadata(path)["topics_with_message_count"][0]["message_count"] == 1
        before = bag_input_digest(tmp_path)[0]
        stat = path.stat()
        path.write_text(TEXT.replace("message_count: 1", "message_count: 2"), encoding="utf-8")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size
        assert path.stat().st_mtime_ns == stat.st_mtime_ns
        assert bag._load_metadata(path)["topics_with_message_count"][0]["message_count"] == 2
        assert bag_input_digest(tmp_path)[0] != before
        assert len(calls) == 2


def test_first_return_and_cache_hits_are_independent_mutable_values(tmp_path, monkeypatch):
    path = metadata(tmp_path)
    calls = observe(monkeypatch)
    with bag._metadata_read_scope():
        first = bag._load_metadata(path)
        first["relative_file_paths"].append("wrong.db3")
        first["topics_with_message_count"][0]["topic_metadata"]["name"] = "/wrong"
        second = bag._load_metadata(path)
        assert second["relative_file_paths"] == ["data.db3"]
        assert second["topics_with_message_count"][0]["topic_metadata"]["name"] == "/lidar/points"
        second["topics_with_message_count"].clear()
        assert len(bag._load_metadata(path)["topics_with_message_count"]) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("invalid", ["[]", "{}", "rosbag2_bagfile_information: []", "[broken"])
def test_invalid_edits_and_error_paths_are_not_cached(tmp_path, monkeypatch, invalid):
    path = metadata(tmp_path)
    calls = observe(monkeypatch)
    with bag._metadata_read_scope():
        reference = bag._load_metadata(path)
        path.write_text(invalid, encoding="utf-8")
        errors = []
        for _ in range(2):
            with pytest.raises((DatasetError, yaml.YAMLError)) as error:
                bag._load_metadata(path)
            errors.append((type(error.value), str(error.value)))
        assert errors[0] == errors[1]
        assert bag._METADATA_READ_CACHE.get().entry is None
        path.write_text(TEXT, encoding="utf-8")
        assert bag._load_metadata(path) == reference
    assert len(calls) == 4


def test_missing_or_invalid_utf8_files_do_not_return_previous_metadata(tmp_path):
    path = metadata(tmp_path)
    with bag._metadata_read_scope():
        bag._load_metadata(path)
        path.unlink()
        with pytest.raises(FileNotFoundError):
            bag._load_metadata(path)
        path.write_bytes(b"\xff\xfe")
        with pytest.raises(UnicodeDecodeError):
            bag._load_metadata(path)


def test_nested_and_failed_scopes_restore_the_outer_owner(tmp_path, monkeypatch):
    path = metadata(tmp_path)
    calls = observe(monkeypatch)
    with bag._metadata_read_scope():
        outer = bag._METADATA_READ_CACHE.get()
        bag._load_metadata(path)
        with pytest.raises(RuntimeError, match="injected"), bag._metadata_read_scope():
            inner = bag._METADATA_READ_CACHE.get()
            assert inner is not outer
            bag._load_metadata(path)
            raise RuntimeError("injected")
        assert inner.entry is None and bag._METADATA_READ_CACHE.get() is outer
        bag._load_metadata(path)
        assert len(calls) == 2
    assert outer.entry is None and bag._METADATA_READ_CACHE.get() is None


def test_concurrent_requests_have_separate_owners(tmp_path, monkeypatch):
    path = metadata(tmp_path)
    calls = observe(monkeypatch)
    barrier = Barrier(2)

    def request():
        assert bag._METADATA_READ_CACHE.get() is None
        with bag._metadata_read_scope():
            state = bag._METADATA_READ_CACHE.get()
            first = bag._load_metadata(path)
            barrier.wait(timeout=10)
            first["relative_file_paths"].clear()
            assert bag._load_metadata(path)["relative_file_paths"] == ["data.db3"]
        assert state.entry is None and bag._METADATA_READ_CACHE.get() is None
        return state

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(executor.map(lambda _: request(), range(2)))
    assert first is not second and len(calls) == 2


def test_reuse_is_bounded_to_one_document_and_its_owned_graph(tmp_path, monkeypatch):
    path = metadata(tmp_path)
    other = tmp_path / "other.yaml"
    other.write_text(TEXT.replace("message_count: 1", "message_count: 2"), encoding="utf-8")
    calls = observe(monkeypatch)
    with bag._metadata_read_scope():
        state = bag._METADATA_READ_CACHE.get()
        bag._load_metadata(path)
        bag._load_metadata(other)
        assert state.entry[0] == other.read_text(encoding="utf-8")
        bag._load_metadata(path)
    assert len(calls) == 3
    monkeypatch.setattr(bag, "_METADATA_PARSE_MAX_BYTES", len(TEXT) + 1)
    with bag._metadata_read_scope():
        for _ in range(2):
            bag._load_metadata(path)
        assert bag._METADATA_READ_CACHE.get().entry is None
    assert len(calls) == 5


@pytest.mark.parametrize("reason", ["long_text", "nonordinary_value", "no_size_support"])
def test_unrepresentable_entries_fall_back_to_full_parsing(tmp_path, monkeypatch, reason):
    path = metadata(tmp_path)
    calls = observe(monkeypatch)
    if reason == "long_text":
        monkeypatch.setattr(bag, "_METADATA_PARSE_MAX_BYTES", 64)
    elif reason == "nonordinary_value":
        path.write_text(
            TEXT.replace("message_count: 1", "message_count: 2026-01-01"), encoding="utf-8"
        )
    else:

        def unavailable(value):
            raise TypeError("size unavailable")

        monkeypatch.setattr(bag.sys, "getsizeof", unavailable)
    with bag._metadata_read_scope():
        for _ in range(2):
            bag._load_metadata(path)
        assert bag._METADATA_READ_CACHE.get().entry is None
    assert len(calls) == 2


def test_yaml_shared_nodes_and_cycles_keep_their_identity_in_each_copy(tmp_path, monkeypatch):
    path = metadata(tmp_path)
    path.write_text(
        "rosbag2_bagfile_information:\n  topics_with_message_count: &cycle [*cycle]\n",
        encoding="utf-8",
    )
    calls = observe(monkeypatch)
    with bag._metadata_read_scope():
        first = bag._load_metadata(path)["topics_with_message_count"]
        second = bag._load_metadata(path)["topics_with_message_count"]
        assert first[0] is first and second[0] is second and first is not second
        first.append("caller mutation")
        assert len(bag._load_metadata(path)["topics_with_message_count"]) == 1
    assert len(calls) == 1


def test_deep_yaml_alias_extension_keeps_the_uncached_parser_behavior(tmp_path):
    path = metadata(tmp_path)
    nodes = ["extension_0: &node0 [0]"]
    for index in range(1, 651):
        nodes.append(f"extension_{index}: &node{index} [*node{index - 1}]")
    path.write_text(
        "\n".join(nodes) + "\n" + TEXT + "      extension: *node650\n", encoding="utf-8"
    )
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(1000)
        reference = bag._load_metadata(path)
        with bag._metadata_read_scope():
            actual = bag._load_metadata(path)
            assert bag._METADATA_READ_CACHE.get().entry is None
        for value in [reference, actual]:
            nested = value["topics_with_message_count"][0]["extension"]
            for _ in range(650):
                nested = nested[0]
            assert nested == [0]
    finally:
        sys.setrecursionlimit(original_limit)


@pytest.mark.parametrize("failure", [MemoryError, RecursionError])
@pytest.mark.parametrize("failed_copy", [1, 2])
def test_optional_copy_failure_falls_back_to_fresh_parsing(
    tmp_path, monkeypatch, failure, failed_copy
):
    path = metadata(tmp_path)
    calls = observe(monkeypatch)
    original = bag.deepcopy
    copies = []

    def copy(value):
        copies.append(1)
        if len(copies) == failed_copy:
            raise failure("injected optional cache copy failure")
        return original(value)

    monkeypatch.setattr(bag, "deepcopy", copy)
    with bag._metadata_read_scope():
        first = bag._load_metadata(path)
        first["relative_file_paths"].clear()
        assert bag._load_metadata(path)["relative_file_paths"] == ["data.db3"]
    assert len(calls) == 2 and bag._METADATA_READ_CACHE.get() is None
