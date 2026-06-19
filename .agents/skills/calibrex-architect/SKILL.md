# Calibrex Architect

Use this skill when changing Calibrex architecture, schemas, CLI behavior,
calibration result semantics, evidence reports, or external adapter boundaries.

## Rules

- Preserve the ROS-independent core boundary.
- Preserve explicit frame conventions: `T_parent_child`, meters, quaternion `xyzw`.
- Keep GPL integrations as optional adapters or subprocess calls.
- Treat dataset-provided calibration as reference evidence, not ground truth.
- Keep `candidate_extrinsics`, `reference_extrinsics`, optimized `transforms`,
  and `comparison.json` semantics separate.
- Add or update tests for behavior changes.
- Update `schemas/` when pydantic config, result, comparison, manifest, or report
  sidecar models change.
- Calibration changes must emit quality metrics, diagnostics, evidence summaries,
  or tests proving no evidence semantics changed.

## Workflow

1. Read `AGENTS.md`.
2. Check whether the change affects config, result, comparison, dataset manifest,
   report sidecar schemas, or CLI output.
3. Implement the smallest stable API surface that supports the requested behavior.
4. If schema models changed, regenerate:

```bash
calibrex schema all --output-dir schemas
pytest tests/unit/test_schemas.py
```

5. Run `ruff check .`, `mypy src/calibrex`, and `pytest`.

## References

- `AGENTS.md`
- `CONTRIBUTING.md`
- `docs/concepts/frame_conventions.md`
- `docs/concepts/license_boundaries.md`
- `docs/tutorials/public_datasets.md`
- `docs/adr/0004-lidar-main-v0-1-evaluation-protocol.md`
