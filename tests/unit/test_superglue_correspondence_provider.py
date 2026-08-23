from __future__ import annotations

import numpy as np
import pytest
import tools.run_superglue_correspondence_provider as provider


def test_superglue_adapter_accepts_explicit_development_split() -> None:
    args = provider._parser().parse_args(
        [
            "sequence",
            "--depth-provider",
            "depth.json",
            "--superglue-repository",
            "superglue",
            "--output-directory",
            "exports",
            "--manifest-output",
            "manifest.yaml",
            "--dataset-family",
            "kitti_360",
            "--split-id",
            "development",
        ]
    )

    assert args.split_id == "development"
    assert args.max_keypoints == -1
    assert args.keypoint_threshold == pytest.approx(0.05)
    assert args.match_threshold == pytest.approx(0.01)
    assert args.equalize_camera_histogram is True
    assert args.lidar_intensity_normalization == "rank-histogram"
    assert args.lidar_integration_manifest is None
    assert provider.SUPERGLUE_ADAPTER_VERSION.endswith("/v0.5")


def test_integration_manifest_requires_explicit_lidar_directory() -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        provider.main(
            [
                "sequence",
                "--depth-provider",
                "depth.json",
                "--superglue-repository",
                "superglue",
                "--output-directory",
                "exports",
                "--manifest-output",
                "manifest.yaml",
                "--lidar-integration-manifest",
                "integration.yaml",
                "--dataset-family",
                "kitti_360",
            ]
        )


def test_superglue_score_is_not_double_counted_as_outlier_probability() -> None:
    points = np.asarray(
        [[1.0, 0.0, 2.0, 0.5], [2.0, 0.0, 3.0, 0.5]],
        dtype=np.float64,
    )
    index_image = np.full((4, 4), -1, dtype=np.int32)
    index_image[1, 1] = 0
    index_image[2, 2] = 1

    arrays = provider._matched_arrays(
        points,
        index_image,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
        np.asarray([0, 1]),
        np.asarray([0.2, 0.4]),
        min_confidence=0.0,
        top_k=4,
    )

    assert arrays["reliability"] == pytest.approx([0.4, 0.2])
    assert arrays["provider_confidence"] == pytest.approx([0.4, 0.2])
    assert arrays["outlier_probability"] == pytest.approx([0.0, 0.0])


def test_virtual_lidar_image_uses_rank_histogram_intensity() -> None:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0, 40.0],
            [0.0, 1.0, 0.0, 10.0],
            [-1.0, 0.0, 0.0, 30.0],
            [-0.1, -1.0, 0.0, 20.0],
        ],
        dtype=np.float64,
    )

    image, index_image = provider._virtual_lidar_image(
        points,
        intensity_normalization="rank-histogram",
    )

    expected = {0: 191, 1: 0, 2: 128, 3: 64}
    for point_index, intensity in expected.items():
        locations = np.argwhere(index_image == point_index)
        assert locations.shape == (1, 2)
        row, column = locations[0]
        assert image[row, column] == intensity


def test_virtual_lidar_image_depth_buffer_keeps_nearest_then_first() -> None:
    points = np.asarray(
        [
            [2.0, 0.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 30.0],
            [1.0, 0.0, 0.0, 20.0],
        ],
        dtype=np.float64,
    )

    image, index_image = provider._virtual_lidar_image(points)

    selected = np.argwhere(index_image == 1)
    assert selected.shape == (1, 2)
    row, column = selected[0]
    assert image[row, column] == 169
    assert not np.any(index_image == 0)
    assert not np.any(index_image == 2)
