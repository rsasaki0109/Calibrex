import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from calibrex.core.result import load_result
from calibrex.visualization.comparison_table import render_comparison_table


def _render_committed_example() -> str:
    left_path = Path("examples/precomputed/result.yaml")
    right_path = Path("examples/precomputed/known_bad_result.yaml")
    return render_comparison_table(
        load_result(left_path),
        load_result(right_path),
        left_label="accepted",
        right_label="known bad",
        left_source={
            "label": left_path.name,
            "sha256": hashlib.sha256(left_path.read_bytes()).hexdigest(),
        },
        right_source={
            "label": right_path.name,
            "sha256": hashlib.sha256(right_path.read_bytes()).hexdigest(),
        },
    )


def test_comparison_table_renders_metric_and_transform_deltas() -> None:
    svg = _render_committed_example()
    root = ET.fromstring(svg)

    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "LiDAR holdout RMSE" in svg
    assert "0.021 m" in svg
    assert "0.089 m" in svg
    assert "BASELINE BETTER" in svg
    assert "max translation 0.15 m" in svg
    assert "max rotation 5°" in svg
    assert "PROTOCOL · NOT COMPARABLE" in svg


def test_comparison_table_embeds_both_sources_and_producers() -> None:
    root = ET.fromstring(_render_committed_example())
    metadata = root.find("{http://www.w3.org/2000/svg}metadata")
    assert metadata is not None
    payload = json.loads(metadata.text or "")

    assert payload["artifact"] == "calibrex.evidence-comparison-table/v0.1"
    assert payload["comparison_schema"] == "slac.comparison/v0.1"
    assert payload["protocol_compatibility"] == "not_comparable"
    assert payload["sources"]["left"]["run_id"] == "precomputed_example"
    assert payload["sources"]["right"]["run_id"] == "known_bad_5deg_example"
    assert payload["estimate_producers"]["right"] == ["human"]


def test_committed_comparison_table_matches_validated_sources() -> None:
    committed = Path("docs/assets/readme-comparison-table.svg").read_text(encoding="utf-8")

    assert committed == _render_committed_example()
