"""Create explicitly synthetic full-model checkpoint for CPU loader integration tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.cofii2p_checkpoint_cpu import CheckpointRequest, initialize_model
from tools.cofii2p_input_boundary import FileRef
from tools.cofii2p_model_bridge import digest


def create_fixture(source: Path, bridge: Path, output: Path) -> Path:
    import torch

    output.mkdir(parents=True, exist_ok=False)
    model, _, _ = initialize_model(source, output, 20261030, 2)
    path = output / "SYNTHETIC_RANDOM_STATE.pt"
    torch.save(model.state_dict(), path)
    request = CheckpointRequest(
        schema_version="calibrex.cofii2p_checkpoint_cpu/v0.1",
        source_checkout=str(source),
        bridge_manifest=FileRef(path=str(bridge), sha256=digest(bridge)),
        checkpoint=FileRef(path=str(path), sha256=digest(path)),
        checkpoint_kind="synthetic_fixture",
        declared_checkpoint_source="LOCAL SYNTHETIC RANDOM STATE; NOT OFFICIAL LEARNED WEIGHTS",
        declared_training_provenance="No training; model initialization seed20261030",
        declared_terms="Local test fixture; external model code terms remain unresolved",
        image_policy="prepared_bgr",
        pixel_layout="row_major",
        seed=20261030,
        threads=2,
    )
    request_path = output / "fixture_request.json"
    request_path.write_text(request.model_dump_json(indent=2) + "\n")
    (output / "fixture_metadata.json").write_text(
        json.dumps(
            {"kind": "synthetic_fixture", "trained": False, "checkpoint_sha256": digest(path)},
            indent=2,
        )
        + "\n"
    )
    return request_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("bridge_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(create_fixture(args.source, args.bridge_manifest, args.output))
