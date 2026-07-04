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
    sys.path.insert(0, str(ROOT / "tools"))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_readme_gallery_jobs_match_readme_gifs() -> None:
    tool = load_gif_tool()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    readme_gifs = set(re.findall(r'docs/assets/[^"]+\.gif', readme))
    gallery_gifs = {str(job.output) for job in tool.README_GIF_JOBS}

    assert readme_gifs <= gallery_gifs
    assert "docs/assets/slac-motion-calibration-loop.gif" in readme_gifs
    hero_jobs = [job for job in tool.README_GIF_JOBS if job.readme_role == "hero"]
    assert len(hero_jobs) == 1
    assert str(hero_jobs[0].output) in readme_gifs


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
        "fps": tool.README_HERO_FPS,
        "frames": tool.README_HERO_FRAME_COUNT,
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
        assert asset["visual"] in {"evidence", "online", "motion"}
        assert asset["uses_builtin_metadata_fallback"] is False
        assert asset["public_inputs"]
        if job.readme_role is not None:
            assert asset.get("readme_role") == job.readme_role


def test_readme_gif_tool_timeline_schema_version_matches_core() -> None:
    from slac.core.online_timeline import ONLINE_TIMELINE_SCHEMA_VERSION

    tool = load_gif_tool()
    assert tool.ONLINE_TIMELINE_SCHEMA_VERSION == ONLINE_TIMELINE_SCHEMA_VERSION


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

    assert sources == {
        "livox-horizon-horizon",
        "a2d2",
        "tiers-lidars-cali",
        "tiers-indoor02-kissicp",
    }
    assert visuals == {"evidence", "online", "motion"}
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
    assert online_asset["readme_role"] == "gallery"
    assert online_asset["source"] == "tiers-lidars-cali"
    assert online_asset["sensor_pair"] == {
        "source": "livox_horizon",
        "target": "livox_avia",
    }
    assert online_asset["pipeline"] == {
        "mode": "real_online",
        "source": "slac calibrate --online",
        "timeline_schema_version": "slac.online_timeline/v0.2",
    }
    assert online_asset["online_run"]["batch_count"] >= 1
    assert (
        online_asset["online_run"]["accepted_batch_count"]
        + online_asset["online_run"]["rejected_batch_count"]
        + online_asset["online_run"]["inconclusive_batch_count"]
        == online_asset["online_run"]["batch_count"]
    )
    assert online_asset["online_run"]["final_gate_status"] in {"pass", "fail", "inconclusive"}
    rosbag_input = online_asset["public_inputs"][0]
    assert rosbag_input["kind"] == "rosbag1_local_dataset"
    assert rosbag_input["bag_file"] == "LidarsCali.bag"
    assert "replay_budgets" in rosbag_input
    assert rosbag_input["replay_budgets"]["max_target_messages"] >= 12


def test_motion_hero_gif_manifest_declares_real_pipeline_provenance() -> None:
    manifest = json.loads(
        (ROOT / "docs" / "assets" / "readme-gif-gallery.json").read_text(encoding="utf-8")
    )
    hero_asset = next(
        asset
        for asset in manifest["assets"]
        if asset["output"] == "docs/assets/slac-motion-calibration-loop.gif"
    )
    assert hero_asset["visual"] == "motion"
    assert hero_asset["readme_role"] == "hero"
    assert hero_asset["source"] == "tiers-indoor02-kissicp"
    assert hero_asset["online_run"]["accepted_batch_count"] == 106
    assert hero_asset["online_run"]["batch_count"] == 108
    assert hero_asset["pipeline"]["config_path"].endswith("online_kissicp_config.yaml")
    assert hero_asset["pipeline"]["timeline_schema_version"] == "slac.online_timeline/v0.3"
    assert hero_asset["animation"]["frames"] == 50
    assert hero_asset["animation"]["fps"] == 10
    rosbag_input = hero_asset["public_inputs"][0]
    assert rosbag_input["kind"] == "rosbag2_local_dataset"
    assert rosbag_input["storage_file"] == "indoor02_rosbag2_kissicp.db3"
    assert rosbag_input["replay_budgets"]["max_target_messages"] == 36


def test_tiers_hero_gif_skips_when_bag_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_gif_tool()
    missing_bag = tmp_path / "missing" / "LidarsCali.bag"
    monkeypatch.setattr(tool, "TIERS_BAG_PATH", missing_bag)
    monkeypatch.setattr(tool, "tiers_bag_available", lambda: False)

    preserved = {
        "output": "docs/assets/online-calibration-loop.gif",
        "source": "tiers-lidars-cali",
        "visual": "online",
    }
    monkeypatch.setattr(
        tool,
        "load_readme_gallery_manifest_if_exists",
        lambda: {"assets": [preserved]},
    )

    generated: list[str] = []

    def _fake_generate_gif(**kwargs: object) -> None:
        generated.append(str(kwargs.get("output")))

    def _fake_load_gif_inputs(**kwargs: object) -> tuple[object, object, str]:
        if kwargs.get("source") == "tiers-lidars-cali":
            pytest.fail("should not load TIERS inputs when bag is absent")
        return (
            tool.LidarCloudPair(
                source_points=[],
                target_points=[],
                source_label="src",
                target_label="tgt",
                bar_labels=("a", "b", "c", "d"),
                bar_values=(1.0, 1.0, 1.0, 1.0),
                source_pose_name="src",
                target_pose_name="tgt",
                source_total=1,
                target_total=1,
                source_path=tmp_path / "sample",
                subtitle="test",
                scene_caption="test",
                legend="test",
                provenance="test",
                shared_voxel_count=1,
                source_recall=1.0,
                shared_centroid_rmse_m=0.1,
                holdout_plane_match_count=0,
                holdout_point_to_plane_p90_m=0.0,
                holdout_unmatched_fraction=0.0,
                known_bad_detectable_fraction=0.0,
                known_bad_max_rmse_delta_m=0.0,
                known_bad_max_point_to_plane_p90_delta_m=0.0,
                support_summary="test",
                holdout_summary="test",
                known_bad_summary="test",
                protocol_summary="test",
                case_summary="test",
            ),
            [],
            "test metadata",
        )

    monkeypatch.setattr(tool, "generate_gif", _fake_generate_gif)
    monkeypatch.setattr(tool, "load_gif_inputs", _fake_load_gif_inputs)
    monkeypatch.setattr(
        tool,
        "readme_gallery_manifest_asset",
        lambda **_kwargs: {"output": "docs/assets/calibration-evidence-demo.gif"},
    )
    monkeypatch.setattr(tool, "write_readme_gallery_manifest", lambda *_args, **_kwargs: None)

    tool.generate_readme_gallery(
        frames=tool.FRAME_COUNT,
        sensor_config=None,
        allow_network=False,
        allow_fallback=False,
    )

    assert "docs/assets/online-calibration-loop.gif" not in generated


def test_indoor02_motion_hero_gif_skips_when_bag_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = load_gif_tool()
    monkeypatch.setattr(tool, "indoor02_kissicp_bag_available", lambda: False)

    preserved = {
        "output": "docs/assets/slac-motion-calibration-loop.gif",
        "source": "tiers-indoor02-kissicp",
        "visual": "motion",
    }
    monkeypatch.setattr(
        tool,
        "load_readme_gallery_manifest_if_exists",
        lambda: {"assets": [preserved]},
    )

    generated: list[str] = []

    def _fake_generate_gif(**kwargs: object) -> None:
        generated.append(str(kwargs.get("output")))

    def _fake_load_gif_inputs(**kwargs: object) -> tuple[object, object, str]:
        if kwargs.get("source") == "tiers-indoor02-kissicp":
            pytest.fail("should not load Indoor02 inputs when bag is absent")
        return (
            tool.LidarCloudPair(
                source_points=[],
                target_points=[],
                source_label="src",
                target_label="tgt",
                bar_labels=("a", "b", "c", "d"),
                bar_values=(1.0, 1.0, 1.0, 1.0),
                source_pose_name="src",
                target_pose_name="tgt",
                source_total=1,
                target_total=1,
                source_path=tmp_path / "sample",
                subtitle="test",
                scene_caption="test",
                legend="test",
                provenance="test",
                shared_voxel_count=1,
                source_recall=1.0,
                shared_centroid_rmse_m=0.1,
                holdout_plane_match_count=0,
                holdout_point_to_plane_p90_m=0.0,
                holdout_unmatched_fraction=0.0,
                known_bad_detectable_fraction=0.0,
                known_bad_max_rmse_delta_m=0.0,
                known_bad_max_point_to_plane_p90_delta_m=0.0,
                support_summary="test",
                holdout_summary="test",
                known_bad_summary="test",
                protocol_summary="test",
                case_summary="test",
            ),
            [],
            "test metadata",
        )

    monkeypatch.setattr(tool, "generate_gif", _fake_generate_gif)
    monkeypatch.setattr(tool, "load_gif_inputs", _fake_load_gif_inputs)
    monkeypatch.setattr(
        tool,
        "readme_gallery_manifest_asset",
        lambda **_kwargs: {"output": "docs/assets/calibration-evidence-demo.gif"},
    )
    monkeypatch.setattr(tool, "write_readme_gallery_manifest", lambda *_args, **_kwargs: None)

    tool.generate_readme_gallery(
        frames=tool.FRAME_COUNT,
        sensor_config=None,
        allow_network=False,
        allow_fallback=False,
    )

    assert "docs/assets/slac-motion-calibration-loop.gif" not in generated


def test_tiers_gif_manifest_asset_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = load_gif_tool()
    bag = tmp_path / "LidarsCali.bag"
    bag.write_bytes(b"rosbag-fixture")
    monkeypatch.setattr(tool, "TIERS_BAG_PATH", bag)
    monkeypatch.setattr(
        tool,
        "load_tiers_download_url",
        lambda: "https://example.test/LidarsCali.bag",
    )

    cloud_pair = tool.LidarCloudPair(
        source_points=[],
        target_points=[],
        source_label="Livox Horizon points",
        target_label="Livox Avia points",
        bar_labels=("a", "b", "c", "d"),
        bar_values=(1.0, 1.0, 1.0, 1.0),
        source_pose_name="livox_horizon",
        target_pose_name="livox_avia",
        source_total=1,
        target_total=1,
        source_path=bag,
        subtitle="test",
        scene_caption="test",
        legend="test",
        provenance="provenance: TIERS LidarsCali rosbag1",
        shared_voxel_count=1,
        source_recall=1.0,
        shared_centroid_rmse_m=0.1,
        holdout_plane_match_count=0,
        holdout_point_to_plane_p90_m=0.0,
        holdout_unmatched_fraction=0.0,
        known_bad_detectable_fraction=0.0,
        known_bad_max_rmse_delta_m=0.0,
        known_bad_max_point_to_plane_p90_delta_m=0.0,
        support_summary="test",
        holdout_summary="test",
        known_bad_summary="test",
        protocol_summary="test",
        case_summary="test",
    )
    job = tool.ReadmeGifJob(
        source="tiers-lidars-cali",
        output=Path("docs/assets/online-calibration-loop.gif"),
        visual="online",
    )
    online_run = tool.OnlineGifRun(
        timeline_path=tmp_path / "timeline.json",
        frame_states=(),
        batch_count=14,
        accepted_batch_count=14,
        rejected_batch_count=0,
        inconclusive_batch_count=0,
        final_gate_status="pass",
        gate_thresholds=tool.OnlineGifGateThresholds(
            min_rank=6,
            max_holdout_rmse_m=0.4,
            max_rolling_regression_m=0.15,
        ),
    )

    asset = tool.readme_gallery_manifest_asset(
        job=job,
        cloud_pair=cloud_pair,
        metadata_source="TIERS LidarsCali static rig (Livox Horizon + Avia)",
        online_run=online_run,
    )

    assert asset["source"] == "tiers-lidars-cali"
    assert asset["sensor_pair"] == {"source": "livox_horizon", "target": "livox_avia"}
    rosbag_input = asset["public_inputs"][0]
    assert rosbag_input["kind"] == "rosbag1_local_dataset"
    assert rosbag_input["bag_file"] == "LidarsCali.bag"
    assert rosbag_input["size_bytes"] == bag.stat().st_size
    assert len(str(rosbag_input["sha256_first_mib"])) == 64
    assert (
        rosbag_input["replay_budgets"]["max_target_messages"]
        == tool.TIERS_GIF_MAX_TARGET_MESSAGES
    )


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
