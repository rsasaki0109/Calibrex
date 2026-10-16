from __future__ import annotations

import pytest
from tools.benchmark_continuous_camera_lidar_reuse import _verify_git_provenance

FULL = "a844dc00354f987da0e103c0dc5bb9610df2eb2a"


@pytest.mark.parametrize("short", ["a844", "a844dc00", FULL])
def test_benchmark_accepts_actual_core_abbreviated_identity(short: str) -> None:
    _verify_git_provenance(FULL, short, FULL, short)


@pytest.mark.parametrize(
    "short,recorded,expected",
    [
        ("bbbbbbbb", "bbbbbbbb", FULL),
        ("a844dc00", "deadbeef", FULL),
        ("a844dc00", None, FULL),
        ("a84", "a84", FULL),
        ("a844dc00", "a844dc00", "f" * 40),
    ],
)
def test_benchmark_rejects_wrong_or_missing_source_identity(
    short: str,
    recorded: str | None,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match="Git provenance differs"):
        _verify_git_provenance(FULL, short, expected, recorded)
