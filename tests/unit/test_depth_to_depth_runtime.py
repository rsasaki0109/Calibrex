import pytest

from calibrex.solvers.depth_to_depth_runtime import (
    create_depth_pair_projector,
    depth_pair_transfer_statistics,
)


@pytest.mark.parametrize("backend, cache_bytes, message", [
    ("cuda", -1, "non-negative"),
    ("numpy", 0, "require --projection-backend cuda"),
    ("numba_cpu", 1024, "require --projection-backend cuda"),
])
def test_cache_arguments_rejected_before_optional_imports(
    backend: str, cache_bytes: int, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_depth_pair_projector(backend, cuda_cache_bytes=cache_bytes)  # type: ignore[arg-type]


def test_cpu_projector_has_no_cuda_transfer_statistics() -> None:
    projector, _ = create_depth_pair_projector("numpy")
    assert depth_pair_transfer_statistics(projector) == {}
