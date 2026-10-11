from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from tools import run_featdepth_provider as runner

from calibrex.data.depth import load_depth_provider, verify_depth_provider_files
from calibrex.data.kitti_camera_lidar_problem import uniform_kitti_frame_ids


def test_uniform_frame_ids_freeze_borer_kitti_selection() -> None:
    frame_ids = uniform_kitti_frame_ids(2762, 25)

    assert len(frame_ids) == 25
    assert frame_ids[:3] == ["0000000000", "0000000115", "0000000230"]
    assert frame_ids[12] == "0000001381"
    assert frame_ids[-1] == "0000002761"
    assert len(set(frame_ids)) == len(frame_ids)


def test_uniform_frame_ids_reject_oversampling() -> None:
    with pytest.raises(ValueError, match="cannot select"):
        uniform_kitti_frame_ids(2, 3)


def test_image_digest_binds_decoded_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "0.png"
    Image.new("RGB", (11, 7), (10, 20, 30)).save(path)
    original = path.read_bytes()
    open_image = Image.open

    def replace_after_read(value):
        path.write_bytes(b"replacement after read")
        return open_image(value)

    monkeypatch.setattr(Image, "open", replace_after_read)
    actual = runner._read_image(tmp_path, "0")
    assert actual.reference.sha256 == hashlib.sha256(original).hexdigest()
    assert actual.reference.size_bytes == len(original)
    assert actual.rgb.shape == (7, 11, 3)
    np.testing.assert_array_equal(actual.rgb[0, 0], [10, 20, 30])


def test_prefetch_is_bounded_ordered_and_can_close(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted = []
    closed = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def submit(self, function, directory, frame_id):
            submitted.append(frame_id)
            return SimpleNamespace(result=lambda: frame_id)

    monkeypatch.setattr(runner, "ThreadPoolExecutor", Pool)
    iterator = runner._prepared_images(
        Path("unused"), [str(i) for i in range(100)], workers=2, batch_size=3
    )
    assert next(iterator) == "0"
    assert len(submitted) == 8  # window 7 plus replacement for consumed frame
    assert next(iterator) == "1"
    assert len(submitted) == 9
    iterator.close()
    assert closed == [True]


@pytest.mark.parametrize(
    "option,value", [("--batch-size", "0"), ("--preprocess-workers", "-1"), ("--cpu-threads", "0")]
)
def test_new_argument_validation_before_loading(option: str, value: str) -> None:
    with pytest.raises(ValueError, match="must be"):
        runner.main(
            [
                "missing",
                "--featdepth-repository",
                "missing",
                "--checkpoint",
                "missing",
                "--output-directory",
                "unused",
                "--artifact-output",
                "unused.json",
                option,
                value,
            ]
        )


def test_batched_export_retains_variable_grids_order_and_short_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class Encoder(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.num_ch_enc = [3]

        def forward(self, tensor):
            return tensor

    class Decoder(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()

        def forward(self, tensor):
            return {("disp", 0, 0): tensor[:, :1] * 0.8 + 0.1}

    for name, cls, attr in [
        ("depth_encoder", Encoder, "DepthEncoder"),
        ("depth_decoder", Decoder, "DepthDecoder"),
    ]:
        module = ModuleType(f"mono.model.mono_fm.{name}")
        setattr(module, attr, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runner, "_verify_repository", lambda path: None)
    monkeypatch.setattr(runner, "INPUT_HEIGHT", 8)
    monkeypatch.setattr(runner, "INPUT_WIDTH", 16)
    sequence = tmp_path / "date" / "drive"
    images = sequence / "image_02" / "data"
    images.mkdir(parents=True)
    (sequence / "image_02" / "timestamps.txt").write_text(
        "\n".join(f"2011-09-30 12:00:0{i}.000000000" for i in range(3)),
        encoding="utf-8",
    )
    (sequence.parent / "calib_cam_to_cam.txt").write_text(
        "P_rect_02: 10 0 5 0 0 11 4 0 0 0 1 0\n",
        encoding="utf-8",
    )
    for i, size in enumerate([(19, 13), (11, 15), (17, 9)]):
        Image.new("RGB", size, (40 + 50 * i, 20, 30)).save(images / f"{i:010d}.png")
    checkpoint = tmp_path / "weights.pth"
    torch.save({"state_dict": {}}, checkpoint)
    providers = []
    for mode, extra in [
        ("serial", []),
        ("batch", ["--batch-size", "2", "--preprocess-workers", "2"]),
    ]:
        provider = tmp_path / mode / "provider.json"
        assert (
            runner.main(
                [
                    str(sequence),
                    "--featdepth-repository",
                    str(tmp_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--frame-count",
                    "3",
                    "--output-directory",
                    str(provider.parent / "depth"),
                    "--artifact-output",
                    str(provider),
                    "--timing-output",
                    str(provider.parent / "timing.json"),
                    *extra,
                ]
            )
            == 0
        )
        artifact = load_depth_provider(provider)
        verify_depth_provider_files(artifact)
        providers.append(artifact)
    serial, batch = providers
    assert [o.frame_id for o in batch.observations] == [f"{i:010d}" for i in range(3)]
    assert batch.preprocessing[1].parameters["execution"]["batch_count"] == 2
    assert batch.preprocessing[1].parameters["execution"]["image_queue_limit"] == 6
    for left, right in zip(serial.observations, batch.observations, strict=True):
        assert left.intrinsics == right.intrinsics
        assert left.image_transform == right.image_transform
        assert left.image == right.image
        np.testing.assert_array_equal(np.load(left.depth.path), np.load(right.depth.path))


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--device", "cpu", "--precision", "float16"], "float16 requires"),
        (["--device", "cuda"], "CUDA-enabled PyTorch"),
        (["--device", "cuda", "--cpu-threads", "2"], "applies only"),
    ],
)
def test_device_settings_fail_before_checkpoint_loading(
    extra: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match=message):
        runner.main(
            [
                "missing",
                "--featdepth-repository",
                "missing",
                "--checkpoint",
                "missing",
                "--output-directory",
                "unused",
                "--artifact-output",
                "unused.json",
                *extra,
            ]
        )
