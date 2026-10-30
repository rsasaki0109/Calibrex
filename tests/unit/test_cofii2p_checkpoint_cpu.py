import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("cv2")

from tools.cofii2p_checkpoint_cpu import (
    CheckpointRequest,
    export_pairs,
    load_strict_state,
    run_checkpoint,
    verify,
)
from tools.cofii2p_input_boundary import FileRef


def test_strict_weights_only_loader_restores_state_on_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    original = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    path = tmp_path / "state.pt"
    torch.save(original.state_dict(), path)
    assert len(load_strict_state(target, path)) == 64
    for name, value in original.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)
        assert target.state_dict()[name].device.type == "cpu"


@pytest.mark.parametrize("corruption", ["keys", "shape", "dtype", "nan", "wrapper", "nontensor"])
def test_checkpoint_rejected_before_mutating_model(tmp_path, corruption):
    torch = pytest.importorskip("torch")
    target = torch.nn.Linear(3, 2)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    state = {k: v.clone() for k, v in before.items()}
    if corruption == "keys":
        state["unexpected"] = torch.zeros(1)
    elif corruption == "shape":
        state["weight"] = torch.zeros(3, 2)
    elif corruption == "dtype":
        state["weight"] = state["weight"].double()
    elif corruption == "nan":
        state["weight"][0, 0] = float("nan")
    elif corruption == "wrapper":
        state = {"state_dict": state}
    else:
        state["weight"] = "not a tensor"
    path = tmp_path / "bad.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="checkpoint"):
        load_strict_state(target, path)
    for name, value in before.items():
        assert torch.equal(target.state_dict()[name], value)


def test_pair_export_preserves_centroids_pixels_duplicates_and_all_contributors():
    bridge = {
        "voxel_index_4": np.array([0, 1]),
        "voxel_point_lidar_m": np.array([[0.1, 0, 5], [2, 0, 6]]),
        "voxel_member_offsets": np.array([0, 2, 3]),
        "voxel_member_source_rows": np.array([2, 0, 1]),
        "source_point_id": np.array([9, 3, 7]),
        "source_to_input_pixel": np.array([[0.5, 0, -2.25], [0, 0.5, -1.25], [0, 0, 1]]),
    }
    pairs = export_pairs(
        np.tile([12, 20], (3, 1)),
        np.array([1, 1, 1]),
        np.array([1, 0, 1]),
        bridge,
        layout="row_major",
        frame_id="frame",
    )
    np.testing.assert_allclose(pairs["point_lidar_m"], [[2, 0, 6], [0.1, 0, 5], [2, 0, 6]])
    np.testing.assert_allclose(pairs["image_mean_px"], np.tile([48.5, 74.5], (3, 1)))
    np.testing.assert_array_equal(pairs["source_member_offsets"], [0, 1, 3, 4])
    np.testing.assert_array_equal(pairs["source_member_point_id"], [3, 7, 9, 3])
    assert len(np.unique(pairs["correspondence_id"])) == 3
    assert "image_covariance_px2" not in pairs  # no invented confidence


def fixture_request():
    ref = {"path": "missing", "sha256": "0" * 64}
    return {
        "schema_version": "calibrex.cofii2p_checkpoint_cpu/v0.1",
        "source_checkout": "unused",
        "bridge_manifest": ref,
        "checkpoint": ref,
        "checkpoint_kind": "synthetic_fixture",
        "declared_checkpoint_source": "test",
        "declared_training_provenance": "no training",
        "declared_terms": "fixture",
        "image_policy": "prepared_bgr",
        "pixel_layout": "row_major",
        "seed": 1,
    }


def test_reference_fields_and_implicit_pixel_policy_rejected():
    data = fixture_request()
    data["reference_problem_path"] = "forbidden.json"
    with pytest.raises(ValidationError):
        CheckpointRequest.model_validate(data)
    del data["reference_problem_path"]
    del data["pixel_layout"]
    with pytest.raises(ValidationError):
        CheckpointRequest.model_validate(data)


def test_fixture_cannot_silently_run_as_learned_checkpoint(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "request.json"
    path.write_text(CheckpointRequest.model_validate(fixture_request()).model_dump_json())
    with pytest.raises(ValueError, match="fixture"):
        run_checkpoint(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_digest_rejects_changed_checkpoint(tmp_path):
    path = tmp_path / "state.pt"
    path.write_bytes(b"original")
    ref = FileRef(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        verify(ref)
