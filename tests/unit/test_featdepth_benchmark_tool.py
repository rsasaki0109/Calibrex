from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tools import benchmark_featdepth_provider as benchmark


def _provider(paths):
    return SimpleNamespace(
        provider="same pinned provider",
        observations=[
            SimpleNamespace(
                frame_id=str(i),
                image="same image bytes",
                capture_time_ns=i,
                intrinsics="same intrinsics",
                image_transform="same grid mapping",
                scale_convention="disparity",
                depth=SimpleNamespace(path=str(path)),
            )
            for i, path in enumerate(paths)
        ],
    )


def test_depth_difference_weights_pixels_and_rejects_identity_changes(tmp_path: Path) -> None:
    originals = [tmp_path / f"original-{i}.npy" for i in range(2)]
    candidates = [tmp_path / f"candidate-{i}.npy" for i in range(2)]
    for path, array in zip(originals, [np.ones((2, 2)), np.ones((2, 3))], strict=True):
        np.save(path, array)
    np.save(candidates[0], np.ones((2, 2)))
    np.save(candidates[1], np.ones((2, 3)) + 0.25)
    left, right = _provider(originals), _provider(candidates)
    assert benchmark._difference(left, right) == {
        "max_absolute": 0.25,
        "max_relative": 0.25,
        "rmse": (6 * 0.25**2 / 10) ** 0.5,
    }
    right.observations[0].capture_time_ns = 123
    with pytest.raises(ValueError, match="identity/grid"):
        benchmark._difference(left, right)


@pytest.mark.parametrize("candidate", [np.full((2, 2), np.nan), np.ones((3, 2))])
def test_depth_difference_rejects_invalid_values_or_grid(tmp_path: Path, candidate) -> None:
    original = tmp_path / "original.npy"
    actual = tmp_path / "actual.npy"
    np.save(original, np.ones((2, 2)))
    np.save(actual, candidate)
    with pytest.raises(ValueError, match="grid mismatch or non-finite"):
        benchmark._difference(_provider([original]), _provider([actual]))
