"""Probe schema and negative controls; two frames make no speedup claim."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from tests.unit.test_camera_lidar_edge import CAMERA, _frames, true_transform
from tools import browser_edge_preparation_probe as probe

from calibrex.evaluation import camera_lidar_edge as edge


@pytest.fixture
def raw_fixture(tmp_path):
    truth = true_transform()
    frames = _frames(truth, 2)
    fields = {
        "frames": 2,
        "candidate": truth,
        "intrinsics": [CAMERA.fx, CAMERA.fy, CAMERA.cx, CAMERA.cy],
        "estimate_translation": False,
        "provenance_json": json.dumps({"kind": "synthetic-unit-control", "frames": 2}),
    }
    for i, frame in enumerate(frames):
        fields[f"image_{i}"] = frame.image
        fields[f"points_{i}"] = frame.points
    path = tmp_path / "raw.npz"
    np.savez_compressed(path, **fields)
    return path


def test_generated_schema_matches_the_report_model():
    root = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (root / "tools/schemas/browser_edge_preparation_probe.schema.json").read_text()
    )
    assert schema == probe.PreparationReuseBenchmark.model_json_schema()


def test_cpu_probe_retains_insufficient_artifacts_and_never_accesses_gpu(raw_fixture, tmp_path):
    baseline = tmp_path / "baseline.py"
    baseline.write_bytes(Path(edge.__file__).read_bytes())

    class NoGPU:
        def __getattr__(self, name):
            raise AssertionError("CPU preparation requested a GPU")

    # These are unit-control identities, not exported benchmark results.
    result = probe.PreparationReuseRun.model_validate(
        asyncio.run(
            probe.measure(
                str(raw_fixture),
                NoGPU(),
                source_head="a" * 40,
                baseline_source_head="b" * 40,
                baseline_path=str(baseline),
            )
        )
    )
    assert result.frames == 2 and result.every_score_and_decision_equal
    assert len(result.attempts) == 3
    assert result.baseline_replay.pose_counts == result.reuse_replay.pose_counts == [0, 0, 0]
    for attempt in result.attempts:
        assert attempt.reuse.diagnostics.hits == 8
        assert attempt.reuse.diagnostics.misses == 4
        assert attempt.reuse.diagnostics.retained_bytes <= attempt.reuse.diagnostics.max_bytes
        for call in attempt.reuse.calls:
            assert call.artifact.solver_status == "insufficient_samples"


def test_probe_rejects_a_changed_baseline_artifact(raw_fixture, tmp_path):
    source = Path(edge.__file__).read_text()
    original = "textureless or very distant scenes yield an inconclusive result."
    assert source.count(original) == 1
    baseline = tmp_path / "different.py"
    baseline.write_text(source.replace(original, "Different unit-test limitation."))
    with pytest.raises(AssertionError):
        asyncio.run(
            probe.measure(
                str(raw_fixture),
                source_head="a" * 40,
                baseline_source_head="b" * 40,
                baseline_path=str(baseline),
            )
        )


@pytest.mark.parametrize("repeats,head", [(2, "a" * 40), (3, "missing")])
def test_probe_rejects_missing_repeats_or_source_identity(repeats, head):
    with pytest.raises(ValueError):
        asyncio.run(
            probe.measure(
                "missing.npz",
                repetitions=repeats,
                source_head=head,
                baseline_source_head="b" * 40,
            )
        )
