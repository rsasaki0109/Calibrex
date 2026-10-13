"""Instrumentation must restore the original CUDA handles after a failed call."""

from types import SimpleNamespace

import pytest
from tools.profile_d2d_cuda import _instrument


def test_profile_restores_all_handles_when_kernel_fails() -> None:
    recordings = []

    class Event:
        def record(self) -> None:
            recordings.append(self)

    def succeed(value: int) -> int:
        return value + 1

    def fail() -> None:
        raise RuntimeError("kernel failed")

    projector = SimpleNamespace(
        _initialize=succeed,
        _project=fail,
        _select=succeed,
        _histogram=succeed,
    )
    originals = vars(projector).copy()
    with (
        pytest.raises(RuntimeError, match="kernel failed"),
        _instrument(projector, SimpleNamespace(Event=Event)) as events,
    ):
        assert projector._initialize(2) == 3
        assert len(events["initialize_stream_seconds"]) == 1
        projector._project()
    assert vars(projector) == originals
    assert len(recordings) == 3
    assert events["project_stream_seconds"] == []
    assert projector._initialize(2) == 3


def test_profile_validation_precedes_optional_cuda_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from tools.profile_d2d_cuda import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--problem",
            "absent.json",
            "--output-dir",
            "unused",
            "--repeats",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="must be positive"):
        main()
