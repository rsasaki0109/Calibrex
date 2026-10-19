"""Replay complete sparse screw evidence without importing calibration code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_screw_residual_evaluation as bench  # noqa: E402
from tools.audit_online_edge_cuda_archive import audit_archive  # noqa: E402


def record_schema() -> dict:
    """Closed measurement schemas; kept outside the unchanged runtime registry."""
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    commit = {"type": "string", "pattern": "^[0-9a-f]{40}$"}
    positive = {"type": "number", "exclusiveMinimum": 0}
    hashes = {"type": "object", "additionalProperties": digest}

    def obj(fields: dict) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(fields),
            "properties": fields,
        }

    origin = {
        "type": "object",
        "required": ["created_at"],
        "properties": {"created_at": {"type": "string"}},
    }
    variants = ("baseline", "candidate")
    definitions = {
        "protocol": obj(
            {
                "commits": obj(dict.fromkeys(variants, commit)),
                "manifests": obj(dict.fromkeys(variants, digest)),
                "tools": obj(dict.fromkeys(bench.TOOLS, digest)),
                "order": {"const": list(bench.ORDER)},
                "seeds": {"const": list(bench.SEEDS)},
                "geometry_sha256": digest,
                "result_schema_sha256": digest,
                "gate_api_ratio": {"const": 2.0},
                "blas_threads": {"const": 1},
                "host_reserved": {"const": False},
                "warmups": {"const": 0},
                "provenance": origin,
            }
        ),
        "measurement": obj(
            {
                "api_seconds": positive,
                "fit_seconds": positive,
                "fit_status": {"enum": ["converged", "max_iterations", "singular_system"]},
                "iterations": {"type": "integer", "minimum": 1},
                "point_to_plane_count": {"type": "integer", "minimum": 1},
                "result_quality_sha256": digest,
                "input_quality_sha256": digest,
                "fit_quality_sha256": digest,
                "imported": hashes,
                "provenance": origin,
            }
        ),
        "attempts": obj(
            {
                "observations": {
                    "type": "array",
                    "minItems": 12,
                    "maxItems": 12,
                    "items": obj(
                        {
                            "case": {"enum": ["joint", "geometry"]},
                            "variant": {"enum": list(variants)},
                            "seconds": positive,
                            "exit_code": {"const": 0},
                            "files": hashes,
                            "log_sha256": digest,
                        }
                    ),
                },
                "provenance": origin,
            }
        ),
    }
    pair = obj({"api_ratio": positive, "fresh_worker_ratio": positive, "quality_sha256": digest})
    case = obj(
        {
            "paired_median_api_ratio": positive,
            "paired_median_fresh_worker_ratio": positive,
            "passes_api_gate": {"type": "boolean"},
            "pairs": {"type": "array", "minItems": 3, "maxItems": 3, "items": pair},
            "medians_seconds": obj(
                dict.fromkeys(
                    variants,
                    obj(
                        {
                            "api_seconds": positive,
                            "fresh_worker_seconds": positive,
                        }
                    ),
                )
            ),
            "complete_input_and_result_bits_equal": {"const": True},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **obj({"cases": obj(dict.fromkeys(("joint", "geometry"), case)), "provenance": origin}),
        "$defs": definitions,
    }


def validate_records(folder: Path) -> None:
    """Validate protocols, all attempts and all twelve timing records."""
    import jsonschema

    schema = record_schema()
    for name in ("protocol", "attempts"):
        jsonschema.validate(
            json.loads((folder / f"{name}.json").read_bytes()),
            {"$defs": schema["$defs"], "$ref": f"#/$defs/{name}"},
        )
    for case in ("joint", "geometry"):
        for index, variant in enumerate(bench.ORDER):
            directory = folder / f"{case}-{index:02d}-{variant}"
            jsonschema.validate(
                json.loads((directory / "measurement.json").read_bytes()),
                {"$defs": schema["$defs"], "$ref": "#/$defs/measurement"},
            )
    jsonschema.validate(json.loads((folder / "summary.json").read_bytes()), schema)


def audit(path: Path, temporary_root: Path) -> dict:
    """Verify a bounded, digest-indexed archive and recompute every gate."""

    def check(folder: Path) -> dict:
        validate_records(folder)
        result = bench.audit(folder)
        saved = json.loads((folder / "summary.json").read_bytes())
        if result["cases"] != saved["cases"]:
            raise ValueError("published summary differs from complete observations")
        archived = json.loads((folder / "schemas/records.schema.json").read_bytes())
        if archived != record_schema():
            raise ValueError("archived record schema differs from maintained schema")
        return result

    result = audit_archive(path, temporary_root, _audit=check)
    result["provenance"]["screw_archive_auditor_sha256"] = bench.sha(Path(__file__))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--temporary-root", type=Path, default=Path("outputs/screw-archive-audits"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.archive, args.temporary_root)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if all(case["passes_api_gate"] for case in result["cases"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
