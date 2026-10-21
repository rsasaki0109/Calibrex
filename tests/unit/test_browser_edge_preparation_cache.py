"""The real browser check/estimate path retains fresh candidates and provenance."""

from __future__ import annotations

import json

import pytest
from tests.unit.camera_lidar_scene import perturbed, true_transform
from tests.unit.test_check_camera_lidar import write_bag

from calibrex.check.browser import run_request_json
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
