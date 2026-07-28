from __future__ import annotations

from pathlib import Path

from calibrex.core.benchmark import (
    load_benchmark,
    load_benchmark_definition,
    render_benchmark_markdown,
)

_ASSETS = Path("docs/assets")


def test_committed_ax_xb_benchmark_is_complete_and_regenerates_markdown() -> None:
    definition = load_benchmark_definition(
        _ASSETS / "ethz-hand-eye-ax-xb-benchmark.definition.json"
    )
    artifact = load_benchmark(_ASSETS / "ethz-hand-eye-ax-xb-benchmark.json")

    assert len(definition.methods) == 13
    assert len(definition.protocol.splits) == 5
    assert len(definition.trials) == 65
    assert all(trial.status == "success" for trial in definition.trials)
    assert {
        method.tool_version for method in definition.methods if method.tool_name == "opencv"
    } == {"4.13.0.92"}
    assert render_benchmark_markdown(artifact) == (
        _ASSETS / "ethz-hand-eye-ax-xb-benchmark.md"
    ).read_text(encoding="utf-8")


def test_committed_ax_yb_benchmark_is_separate_and_complete() -> None:
    definition = load_benchmark_definition(
        _ASSETS / "ethz-hand-eye-ax-yb-benchmark.definition.json"
    )
    artifact = load_benchmark(_ASSETS / "ethz-hand-eye-ax-yb-benchmark.json")

    assert len(definition.methods) == 7
    assert len(definition.protocol.splits) == 5
    assert len(definition.trials) == 35
    assert all(trial.status == "success" for trial in definition.trials)
    assert definition.protocol.protocol_id != ("ethz-real-hand-eye/ax-xb/absolute-pose-split/v0.1")
    assert render_benchmark_markdown(artifact) == (
        _ASSETS / "ethz-hand-eye-ax-yb-benchmark.md"
    ).read_text(encoding="utf-8")
