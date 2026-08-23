from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from tools import run_i2pnet_correspondence_provider as provider

from calibrex.core.result import TransformEstimateProvenance, TransformResult


def _initial_transform(
    *,
    role: str = "initial",
    execution_mode: str = "imported",
) -> TransformResult:
    return TransformResult.model_validate(
        {
            "parent": "camera_0",
            "child": "lidar",
            "translation_m": [1.0, 2.0, 3.0],
            "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "estimate_id": "nominal-rig-initial-v1",
            "provenance": TransformEstimateProvenance(
                producer="factory",
                execution_mode=execution_mode,
                role_in_comparison=role,
                evidence_level="factory_provided",
                source="vehicle rig nominal CAD",
            ).model_dump(mode="json", exclude_none=True),
        }
    )


def test_parser_pins_official_kitti_large_preprocessing() -> None:
    args = provider._parser().parse_args(
        [
            "sequence",
            "--depth-provider",
            "depth.yaml",
            "--i2pnet-repository",
            "I2PNet",
            "--checkpoint",
            "ckpt.pt",
            "--initial-transform",
            "initial.yaml",
            "--output-directory",
            "exports",
            "--manifest-output",
            "manifest.yaml",
            "--dataset-family",
            "kitti_360",
        ]
    )

    assert args.crop_top_rows == 50
    assert args.image_scale == pytest.approx(0.5)
    assert (args.input_height, args.input_width) == (160, 512)
    assert args.model_point_count == 150_000
    assert args.split_id == "development"
    assert args.neighbor_backend == "torch"
    assert provider.I2PNET_ADAPTER_VERSION.endswith("/v0.2")
    assert provider.I2PNET_CORRESPONDENCE_RELIABILITY_GATE == 0.01


def test_explicit_frame_selection_is_ordered_and_closed() -> None:
    observations = [
        SimpleNamespace(frame_id="0000000000"),
        SimpleNamespace(frame_id="0000000115"),
    ]

    selected = provider._select_observations(
        observations,
        "0000000115,0000000000",
    )

    assert [item.frame_id for item in selected] == ["0000000115", "0000000000"]
    with pytest.raises(ValueError, match="absent from the depth provider"):
        provider._select_observations(observations, "0000000999")


def test_feature_grid_mapping_undoes_center_crop_scale_and_top_crop() -> None:
    mapping = provider._ImageMapping(
        source_width=1408,
        source_height=376,
        crop_top_rows=50,
        scale=0.5,
        crop_x_scaled_px=96,
        crop_y_scaled_px=1,
        output_width=512,
        output_height=160,
    )

    pixels = mapping.feature_pixels_to_source(
        np.asarray([[0, 15, 79]], dtype=np.int64),
        feature_height=5,
        feature_width=16,
    )

    np.testing.assert_allclose(
        pixels[0],
        np.asarray([[192.0, 52.0], [1152.0, 52.0], [1152.0, 308.0]]),
    )


def test_torch_neighbor_backend_wraps_width_and_copies_nearest() -> None:
    torch = pytest.importorskip("torch")
    xyz = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    xyz[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    xyz[0, 0, 3] = torch.tensor([1.2, 0.0, 0.0])

    batch, height, width, mask = provider._torch_neighbor_copy(
        xyz,
        xyz,
        torch.tensor([[[0, 0]]], dtype=torch.int32),
        [1, 3],
        3,
        distance=1.0,
    )

    assert batch.tolist() == [[[0, 0, 0]]]
    assert height.tolist() == [[[0, 0, 0]]]
    assert width.tolist() == [[[0, 3, 0]]]
    assert mask.squeeze(-1).tolist() == [[[1.0, 1.0, 1.0]]]


def test_torch_neighbor_backend_masks_invalid_queries_without_copying() -> None:
    torch = pytest.importorskip("torch")
    source = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    target = torch.ones((1, 2, 2, 3), dtype=torch.float32)

    batch, height, width, mask = provider._torch_neighbor_att(
        source,
        target,
        torch.tensor([[[0, 0]]], dtype=torch.int32),
        [1, 1],
        1,
    )

    assert batch.tolist() == [[[0]]]
    assert height.tolist() == [[[0]]]
    assert width.tolist() == [[[0]]]
    assert mask.squeeze(-1).tolist() == [[[0.0]]]


def test_soft_weights_become_mean_covariance_and_single_confidence() -> None:
    mapping = provider._ImageMapping(
        source_width=32,
        source_height=32,
        crop_top_rows=0,
        scale=1.0,
        crop_x_scaled_px=0,
        crop_y_scaled_px=0,
        output_width=32,
        output_height=32,
    )
    weights = np.asarray(
        [
            [[0.75, 0.75], [0.25, 0.25]],
            [[0.50, 0.50], [0.50, 0.50]],
        ],
        dtype=np.float64,
    )

    arrays = provider._soft_correspondence_arrays(
        np.asarray([[1.0, 0.0, 2.0], [2.0, 0.0, 3.0]], dtype=np.float64),
        np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        weights,
        feature_height=2,
        feature_width=2,
        mapping=mapping,
        min_reliability=0.0,
        top_k=2,
        covariance_floor_px2=1.0,
    )

    expected_reliability = 1.0 - (-0.75 * math.log(0.75) - 0.25 * math.log(0.25)) / math.log(2.0)
    assert arrays["i2pnet_p3_index"].tolist() == [0, 1]
    assert arrays["image_mean_px"][0] == pytest.approx([4.0, 0.0])
    assert arrays["image_covariance_px2"][0] == pytest.approx([49.0, 0.0, 0.0, 1.0])
    assert arrays["reliability"] == pytest.approx([expected_reliability, 0.0])
    assert arrays["provider_confidence"] == pytest.approx(arrays["reliability"])
    assert arrays["outlier_probability"] == pytest.approx([0.0, 0.0])
    assert arrays["channel_sharpness"] == pytest.approx([expected_reliability, 0.0])
    assert arrays["channel_agreement"] == pytest.approx([1.0, 1.0])


def test_reference_role_and_dataset_reference_inputs_are_forbidden() -> None:
    provider._verify_initial_transform(_initial_transform())

    with pytest.raises(ValueError, match="role must be 'initial'"):
        provider._verify_initial_transform(_initial_transform(role="selected_reference"))
    with pytest.raises(ValueError, match="dataset_reference transforms are forbidden"):
        provider._verify_initial_transform(_initial_transform(execution_mode="dataset_reference"))


def test_camera_stream_must_match_observation_frame() -> None:
    observations = [SimpleNamespace(frame_id="0", source_frame="camera_0")]
    initial = _initial_transform()

    provider._verify_observation_frames(
        observations,
        initial=initial,
        camera_stream="image_00",
    )
    with pytest.raises(ValueError, match="initial transform parent differs"):
        provider._verify_observation_frames(
            observations,
            initial=initial.model_copy(update={"parent": "camera_2"}),
            camera_stream="image_00",
        )
    with pytest.raises(ValueError, match="camera stream differs"):
        provider._verify_observation_frames(
            observations,
            initial=initial,
            camera_stream="image_02",
        )


def test_transform_matrix_uses_t_parent_child_convention() -> None:
    matrix = provider._transform_matrix(_initial_transform())

    assert matrix == pytest.approx(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )


def test_pose_correction_left_composes_with_initial_transform() -> None:
    correction = np.asarray(
        [
            math.sqrt(0.5),
            0.0,
            0.0,
            math.sqrt(0.5),
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    result = provider._aggregate_pose_result(
        initial=_initial_transform(),
        corrections=[correction, correction],
        frame_ids=["0000000000", "0000000001"],
        checkpoint=provider.Path("ckpt.pt"),
        checkpoint_digest="a" * 64,
        initial_digest="b" * 64,
        command=["python", "adapter.py"],
    )

    assert result.translation_m == pytest.approx([-1.0, 1.0, 3.0])
    assert result.rotation_quat_xyzw == pytest.approx([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    assert result.quality.std_translation_m == pytest.approx([0.0, 0.0, 0.0])
    assert result.provenance.role_in_comparison == "output"
    assert "T_correction @ T_camera_lidar_initial" in result.provenance.notes[0]
