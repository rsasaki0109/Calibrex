from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from calibrex.cli.main import main
from calibrex.core.koide_handoff import (
    KoideExecutionLock,
    koide_execution_lock_json_schema,
    load_koide_execution_lock,
)

LOCK = Path("examples/official/koide_execution_lock.yaml")


def test_checked_in_lock_is_digest_and_schema_valid() -> None:
    lock = load_koide_execution_lock(LOCK)
    payload = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    jsonschema.validate(payload, koide_execution_lock_json_schema())
    assert lock.profile == "commercial"
    assert lock.initial_guess_mode == "manual"
    assert lock.superglue_policy == "excluded"
    assert lock.source.commit == "02a0dc039f5509708f384be4ff3228e0ae09352d"
    assert lock.image.digest.endswith(
        "f7363cc3deadc2419f3527a7b9ec7bdabaa3031fd8e1b9c6cc46666ae1301bbb"
    )


def test_lock_rejects_superglue_stage() -> None:
    payload = yaml.safe_load(LOCK.read_text(encoding="utf-8"))
    payload["stages"][1]["argv"][-2] = "find_matches_superglue.py"
    with pytest.raises(ValueError, match="SuperGlue"):
        KoideExecutionLock.model_validate(payload)


def test_handoff_cli_is_non_executing_and_reports_runtime(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["camera-lidar", "koide-handoff", "--lock", str(LOCK), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "handoff_only"
    assert output["executes_external_process"] is False
    assert output["initial_guess_mode"] == "manual"
    assert output["superglue_policy"] == "excluded"
    assert len(output["next_actions"]) >= 5
