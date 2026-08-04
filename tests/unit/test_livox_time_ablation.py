"""Tests for public Livox time-ablation materialization helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "run_livox_time_ablation.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("run_livox_time_ablation", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_point_time_metadata_follows_estimated_target_sensor() -> None:
    tool = _load_tool_module()
    base = {
        "sensors": {
            "livox_avia": {
                "point_time_field": "t",
                "solid_state": {
                    "point_time_reference": "message_stamp",
                    "point_time_field": "t",
                },
            }
        },
        "frames": {
            "livox_avia": {"root": True},
            "livox_avia_copy": {
                "parent": "livox_avia",
                "transform": {"estimate": True},
            },
        },
    }

    assert tool._point_time_metadata(base) == ("t", "message_stamp")
