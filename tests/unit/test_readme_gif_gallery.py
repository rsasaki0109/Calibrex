from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
GIF_TOOL = ROOT / "tools" / "generate_calibration_evidence_gif.py"


def load_gif_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_calibration_evidence_gif", GIF_TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_readme_gallery_jobs_match_readme_gifs() -> None:
    tool = load_gif_tool()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    readme_gifs = set(re.findall(r'docs/assets/[^"]+\.gif', readme))
    gallery_gifs = {str(job.output) for job in tool.README_GIF_JOBS}

    assert gallery_gifs == readme_gifs


def test_readme_gallery_manifest_matches_assets() -> None:
    tool = load_gif_tool()
    manifest_path = ROOT / "docs" / "assets" / "readme-gif-gallery.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == tool.README_GIF_MANIFEST_SCHEMA_VERSION
    assert manifest["generator"] == "tools/generate_calibration_evidence_gif.py"
    assert manifest["fallback_metadata_allowed"] is False
    assert manifest["dimensions"] == {
        "width": tool.WIDTH,
        "height": tool.HEIGHT,
        "fps": tool.FPS,
        "frames": tool.FRAME_COUNT,
    }

    job_outputs = {str(job.output) for job in tool.README_GIF_JOBS}
    jobs_by_output = {str(job.output): job for job in tool.README_GIF_JOBS}
    asset_outputs = {asset["output"] for asset in manifest["assets"]}
    assert asset_outputs == job_outputs

    for asset in manifest["assets"]:
        asset_path = ROOT / asset["output"]
        job = jobs_by_output[asset["output"]]
        assert asset["sha256"] == hashlib.sha256(asset_path.read_bytes()).hexdigest()
        assert asset["size_bytes"] == asset_path.stat().st_size
        assert asset["visual"] == job.visual
        assert asset["visual"] in {"evidence", "online"}
        assert asset["uses_builtin_metadata_fallback"] is False
        assert asset["public_inputs"]


def test_readme_gallery_manifest_is_schema_valid() -> None:
    manifest = json.loads(
        (ROOT / "docs" / "assets" / "readme-gif-gallery.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (ROOT / "docs" / "assets" / "readme-gif-gallery.schema.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(manifest, schema)


def test_readme_gallery_has_multiple_public_sources() -> None:
    tool = load_gif_tool()

    sources = {job.source for job in tool.README_GIF_JOBS}
    visuals = {job.visual for job in tool.README_GIF_JOBS}
    a2d2_pairs = {
        (job.a2d2_source_id, job.a2d2_target_id)
        for job in tool.README_GIF_JOBS
        if job.source == "a2d2"
    }

    assert sources == {"livox-horizon-horizon", "a2d2"}
    assert visuals == {"evidence", "online"}
    assert len(a2d2_pairs) >= 2
    assert all(source_id != target_id for source_id, target_id in a2d2_pairs)


def test_online_gif_manifest_declares_real_pipeline_provenance() -> None:
    manifest = json.loads(
        (ROOT / "docs" / "assets" / "readme-gif-gallery.json").read_text(encoding="utf-8")
    )
    online_asset = next(
        asset
        for asset in manifest["assets"]
        if asset["output"] == "docs/assets/online-calibration-loop.gif"
    )
    assert online_asset["visual"] == "online"
    assert online_asset["pipeline"] == {
        "mode": "real_online",
        "source": "calibrex calibrate --online",
        "timeline_schema_version": "calibrex.online_timeline/v0.2",
    }
    assert online_asset["online_run"]["batch_count"] >= 1
    assert (
        online_asset["online_run"]["accepted_batch_count"]
        + online_asset["online_run"]["rejected_batch_count"]
        + online_asset["online_run"]["inconclusive_batch_count"]
        == online_asset["online_run"]["batch_count"]
    )
    assert online_asset["online_run"]["final_gate_status"] in {"pass", "fail", "inconclusive"}


def test_a2d2_metadata_fallback_is_explicit() -> None:
    tool = load_gif_tool()

    with pytest.raises(SystemExit, match="A2D2 public LiDAR metadata is required"):
        tool.load_a2d2_lidar_setup(
            None,
            allow_network=False,
            allow_fallback=False,
        )

    lidars, source = tool.load_a2d2_lidar_setup(
        None,
        allow_network=False,
        allow_fallback=True,
    )
    assert source == "fixed multi-LiDAR fallback"
    assert len(lidars) == 5
