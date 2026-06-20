from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

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


def test_readme_gallery_has_multiple_public_sources() -> None:
    tool = load_gif_tool()

    sources = {job.source for job in tool.README_GIF_JOBS}
    a2d2_pairs = {
        (job.a2d2_source_id, job.a2d2_target_id)
        for job in tool.README_GIF_JOBS
        if job.source == "a2d2"
    }

    assert sources == {"livox-horizon-horizon", "a2d2"}
    assert len(a2d2_pairs) >= 2
    assert all(source_id != target_id for source_id, target_id in a2d2_pairs)


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
