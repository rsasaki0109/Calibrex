from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from calibrex.core.io import read_mapping


def test_continuous_time_lidar_cli_dispatches_ros1_and_ros2_to_generic_pipeline(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cli = importlib.import_module("calibrex.cli.main")
    output = tmp_path / "continuous_time.yaml"
    calls: list[Path] = []

    class FakeArtifact:
        status = "converged"
        observability = SimpleNamespace(rank=6)
        final_holdout_rmse_m = 0.1
        holdout_correspondence_count = 6
        options = SimpleNamespace(min_correspondences=6)

        def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, str]:
            assert mode == "json"
            assert exclude_none is True
            return {"schema_version": "slac.continuous_time_lidar_pair_result/v0.1"}

    def _fake_evaluate(config: Path) -> FakeArtifact:
        calls.append(config)
        return FakeArtifact()

    monkeypatch.setattr(cli, "evaluate_continuous_time_lidar_pair", _fake_evaluate)

    assert (
        cli.main(
            [
                "continuous-time-lidar-pair",
                "examples/public_datasets/tiers_livox_lidars_cali/online_continuous_time_config.yaml",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )

    assert calls == [
        Path(
            "examples/public_datasets/tiers_livox_lidars_cali/"
            "online_continuous_time_config.yaml"
        )
    ]
    assert read_mapping(output)["schema_version"] == (
        "slac.continuous_time_lidar_pair_result/v0.1"
    )
    assert json.loads(capsys.readouterr().out)["schema_version"] == (
        "slac.continuous_time_lidar_pair_result/v0.1"
    )
