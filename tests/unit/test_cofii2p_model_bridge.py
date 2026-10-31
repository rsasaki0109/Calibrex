import json
from pathlib import Path

import numpy as np
import pytest
from test_cofii2p_input_boundary import make_request
from tools.cofii2p_input_boundary import prepare
from tools.cofii2p_model_bridge import (
    BridgeOptions,
    build_bridge,
    count_indices,
    decode_fine_pixels,
    fine_to_source_pixels,
    neighbors,
    voxelize_with_lineage,
)


def test_voxel_means_keep_all_source_rows_and_unnormalized_features():
    points = np.array([[1.0, 0, 0], [0.02, 0, 0], [0, 0, 0]])
    features = np.array([[0.9, 1, 0, 0], [0.4, 1, 0, 0], [0.2, -1, 0, 0]])
    p, f, offsets, rows, inverse = voxelize_with_lineage(points, features, 0.1)
    np.testing.assert_allclose(p, [[0.01, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(f, [[0.3, 0, 0, 0], [0.9, 1, 0, 0]])
    np.testing.assert_array_equal(offsets, [0, 2, 3])
    np.testing.assert_array_equal(rows, [1, 2, 0])
    np.testing.assert_array_equal(inverse, [1, 0, 0])
    np.testing.assert_array_equal(np.sort(rows), np.arange(len(points)))


@pytest.mark.parametrize("source,target", [(9, 4), (3, 3), (3, 8), (3, 9), (1, 5)])
def test_count_rule_pins_duplicate_multiplicity_and_rng(source, target):
    rng = np.random.RandomState(17)
    if source >= target:
        expected = rng.choice(source, target, replace=False)
    else:
        fixed = np.arange(source)
        while source + len(fixed) < target:
            fixed = np.r_[fixed, np.arange(source)]
        expected = np.r_[fixed, rng.choice(source, target - len(fixed), replace=False)]
    actual = count_indices(source, target, np.random.RandomState(17))
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == target


def test_knn_direction_and_distances_against_exhaustive_search():
    support = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0], [8, 0, 0]])
    query = np.array([[2.7, 0, 0], [0.1, 0, 0]])
    actual = neighbors(support, query, 2)
    expected = np.argsort(np.linalg.norm(query[:, None] - support[None], axis=2), axis=1)[:, :2]
    np.testing.assert_array_equal(actual, expected)
    assert neighbors(support, query, 1).shape == (2, 1)


def test_all_patch_indices_follow_image_array_layout_and_explicit_legacy():
    centers = np.tile([12, 20], (16, 1))
    indices = np.arange(16)
    pixels = decode_fine_pixels(centers, indices, layout="row_major")
    grid = np.arange(80 * 256).reshape(80, 256)
    flat_patch = grid[18:22, 10:14].reshape(-1)
    np.testing.assert_array_equal(
        grid[pixels[:, 1].astype(int), pixels[:, 0].astype(int)], flat_patch
    )
    legacy = decode_fine_pixels(centers, indices, layout="official_eval_legacy")
    assert np.count_nonzero(np.any(pixels != legacy, axis=1)) == 12
    affine = np.array([[0.5, 0, -54.25], [0, 0.5, -25.25], [0, 0, 1]])
    source = fine_to_source_pixels(pixels, affine)
    expected = np.column_stack([4 * pixels[:, 0] + 108.5, 4 * pixels[:, 1] + 50.5])
    np.testing.assert_allclose(source, expected)
    with pytest.raises(ValueError, match="exceeds"):
        decode_fine_pixels(np.array([[1, 20]]), np.array([0]), layout="row_major")
    with pytest.raises(ValueError, match="indices"):
        decode_fine_pixels(np.array([[12, 20]]), np.array([16]), layout="row_major")


def test_bridge_multilevel_lineage_reproducibility_and_centroid_geometry(tmp_path):
    request, _, transform = make_request(tmp_path)
    prepared = prepare(request, tmp_path / "prepared")
    manifest = tmp_path / "prepared/manifest.json"
    options = BridgeOptions(seed=23, count=64, stages=5, neighbors=4)
    a = build_bridge(manifest, tmp_path / "a", options)
    b = build_bridge(manifest, tmp_path / "b", options)
    with np.load(a.artifact.path) as x, np.load(b.artifact.path) as y:
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])
        np.testing.assert_allclose(
            x["voxel_point_initial_camera_m"],
            x["voxel_point_lidar_m"] @ transform[:3, :3].T + transform[:3, 3],
        )
        np.testing.assert_array_equal(x["source_point_id"], [9, 3, 7])
        np.testing.assert_allclose(
            x["fine_intrinsics"],
            np.diag([0.5, 0.5, 1]) @ np.load(prepared.artifact.path)["input_intrinsics"],
        )
        for level in range(5):
            n = 64 // 2**level
            assert x[f"points_{level}"].shape == (n, 3)
            assert x[f"neighbors_{level}"].shape == (n, 4)
            voxel = x[f"voxel_index_{level}"]
            np.testing.assert_allclose(
                x[f"points_{level}"], x["voxel_point_initial_camera_m"][voxel], atol=1e-6
            )
            if level:
                parent = x[f"parent_{level}"]
                np.testing.assert_array_equal(voxel, x[f"voxel_index_{level - 1}"][parent])
                assert x[f"subsampling_{level - 1}"].max() < n * 2
                assert x[f"upsampling_{level - 1}"].max() < n
        assert len(np.unique(x["voxel_index_0"])) < 64
        assert len(np.unique(x["parent_1"])) < len(x["parent_1"])
    assert a.prepared_artifact == prepared.artifact
    assert "no model inference" in a.scope


def test_bad_digest_and_infeasible_knn_fail_before_output(tmp_path):
    request, _, _ = make_request(tmp_path)
    report = prepare(request, tmp_path / "prepared")
    manifest = tmp_path / "prepared/manifest.json"
    with pytest.raises(ValueError, match="smallest level"):
        build_bridge(manifest, tmp_path / "bad", BridgeOptions(seed=1, count=16))
    assert not (tmp_path / "bad").exists()
    Path(report.artifact.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        build_bridge(manifest, tmp_path / "bad", BridgeOptions(seed=1, count=64, neighbors=4))
    assert not (tmp_path / "bad").exists()
    assert json.loads(manifest.read_text())["artifact"]["sha256"] == report.artifact.sha256
