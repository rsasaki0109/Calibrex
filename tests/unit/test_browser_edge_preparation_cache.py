"""The real browser check/estimate path retains fresh candidates and provenance."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import get_type_hints

import pytest
from tests.unit.camera_lidar_scene import perturbed, true_transform
from tests.unit.test_check_camera_lidar import write_bag

from calibrex.check.browser import plan_request_json, run_request_json
from calibrex.data import rosbag2
from calibrex.evaluation import camera_lidar_edge as edge
from calibrex.evaluation.edge_preparation_cache import EdgePreparationCache


@pytest.fixture(scope="module")
def candidate_bags(tmp_path_factory):
    root = tmp_path_factory.mktemp("edge-preparation-browser")
    truth = true_transform()
    return [
        write_bag(root / "original", truth),
        write_bag(root / "changed", truth, candidate=perturbed(truth, (0, 3.0, 0))),
    ]


@pytest.mark.parametrize("mode", ["check", "estimate"])
def test_browser_requests_reuse_only_edges_and_recheck_changed_candidates(
    candidate_bags, mode, monkeypatch
):
    results, received = [], []
    original = edge.run_ros2_camera_lidar_edge

    def capture(*args, **kwargs):
        received.append(kwargs.get("preparation_cache"))
        result = original(*args, **kwargs)
        results.append(result.model_dump(mode="json"))
        return result

    monkeypatch.setattr(edge, "run_ros2_camera_lidar_edge", capture)
    cache = EdgePreparationCache()
    starts, digests = [], []
    for bag in candidate_bags:
        request = json.dumps({"bag_dir": str(bag), "mode": mode, "pairs": ["camera-lidar"]})
        expected = json.loads(run_request_json(request))
        actual = json.loads(run_request_json(request, preparation_cache=cache))
        assert expected["ok"] and actual["ok"], (expected, actual)
        assert received[-2:] == [None, cache]
        left, right = results[-2:]
        assert left["provenance"].pop("created_at")
        assert right["provenance"].pop("created_at")
        assert left == right
        starts.append(right["start_transform"])
        digests.append(right["provenance"]["input_sha256"])
        assert "edge_preparation_cache" not in actual["artifact"]["options"]
        diagnostic = actual["edge_preparation_cache"]
        assert diagnostic["retained_bytes"] <= diagnostic["max_bytes"]
    assert starts[0] != starts[1]
    assert digests[0] != digests[1]
    assert cache.diagnostics.hits >= 24


@pytest.mark.parametrize("mode", ["check", "estimate"])
def test_request_metadata_scope_preserves_complete_calibration(candidate_bags, mode, monkeypatch):
    calls, artifacts = [], []
    original_parse = rosbag2._parse_metadata
    original_edge = edge.run_ros2_camera_lidar_edge

    def parse(*args):
        calls.append(args[0])
        return original_parse(*args)

    def capture(*args, **kwargs):
        result = original_edge(*args, **kwargs)
        assert result.samples.frames >= 8 and result.jackknife_fits == 6
        artifacts.append(result.model_dump(mode="json"))
        return result

    monkeypatch.setattr(rosbag2, "_parse_metadata", parse)
    monkeypatch.setattr(edge, "run_ros2_camera_lidar_edge", capture)
    request = json.dumps(
        {"bag_dir": str(candidate_bags[0]), "mode": mode, "pairs": ["camera-lidar"]}
    )
    assert json.loads(run_request_json(request))["ok"]
    assert len(calls) == 1 and rosbag2._METADATA_READ_CACHE.get() is None
    calls.clear()
    assert json.loads(inspect.unwrap(run_request_json)(request))["ok"]
    assert len(calls) > 1 and rosbag2._METADATA_READ_CACHE.get() is None
    assert len(artifacts) == 2
    for artifact in artifacts:
        assert artifact["provenance"].pop("created_at")
    assert artifacts[0] == artifacts[1]
    assert inspect.signature(run_request_json) == inspect.signature(
        inspect.unwrap(run_request_json)
    )
    assert get_type_hints(run_request_json) == get_type_hints(inspect.unwrap(run_request_json))


def test_failed_browser_request_releases_owned_metadata(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "docs/app/samples/check_sample/metadata.yaml"
    (tmp_path / "metadata.yaml").write_bytes(source.read_bytes())
    (tmp_path / "corrupt.db3").write_bytes(b"not a sqlite database" * 20)
    states = []
    original = rosbag2._parse_metadata

    def parse(*args):
        state = rosbag2._METADATA_READ_CACHE.get()
        assert state is not None
        states.append(state)
        return original(*args)

    monkeypatch.setattr(rosbag2, "_parse_metadata", parse)
    result = json.loads(run_request_json(json.dumps({"bag_dir": str(tmp_path)})))
    assert not result["ok"] and result["error"]
    assert states and all(state.entry is None for state in states)
    assert rosbag2._METADATA_READ_CACHE.get() is None


def test_browser_plan_uses_one_parse_and_keeps_its_complete_reply(monkeypatch):
    sample = Path(__file__).resolve().parents[2] / "docs/app/samples/check_sample"
    original = rosbag2._parse_metadata
    calls = []

    def parse(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(rosbag2, "_parse_metadata", parse)
    request = json.dumps({"bag_dir": str(sample), "bag_label": "public sample"})
    actual = json.loads(plan_request_json(request))
    assert actual["ok"] and len(calls) == 1
    assert rosbag2._METADATA_READ_CACHE.get() is None
    calls.clear()
    expected = json.loads(inspect.unwrap(plan_request_json)(request))
    assert expected["ok"] and len(calls) > 1
    assert actual["artifact"]["provenance"].pop("created_at")
    assert expected["artifact"]["provenance"].pop("created_at")
    assert actual == expected
    assert rosbag2._METADATA_READ_CACHE.get() is None
