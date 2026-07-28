from pathlib import Path

import yaml


def test_composite_action_exposes_calibration_ci_contract() -> None:
    action = yaml.safe_load(Path("action.yml").read_text(encoding="utf-8"))

    assert action["name"] == "Calibrex Calibration CI"
    assert action["inputs"]["candidate"]["required"] is True
    assert action["inputs"]["enforce"]["default"] == "true"
    assert action["inputs"]["enforce_protocol"]["default"] == "true"
    assert set(action["outputs"]) == {
        "status",
        "artifact",
        "summary",
        "evidence_card",
        "comparison",
    }
    assert action["runs"]["using"] == "composite"
    run_step = action["runs"]["steps"][-1]
    assert run_step["id"] == "calibration_ci"
    assert "tools/run_calibration_ci_action.py" in run_step["run"]


def test_action_installs_the_checked_out_action_revision() -> None:
    action = yaml.safe_load(Path("action.yml").read_text(encoding="utf-8"))
    install_step = action["runs"]["steps"][1]

    assert install_step["name"] == "Install Calibrex action revision"
    assert "${{ github.action_path }}" in install_step["run"]
