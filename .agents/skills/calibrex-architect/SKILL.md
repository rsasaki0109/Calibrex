# Calibrex Architect

Use this skill when changing Calibrex architecture, schemas, CLI behavior, or
calibration result semantics.

## Rules

- Preserve the ROS-independent core boundary.
- Preserve explicit frame conventions: `T_parent_child`, meters, quaternion `xyzw`.
- Keep GPL integrations as optional adapters or subprocess calls.
- Add or update tests for behavior changes.
- Update `schemas/` when pydantic config or result models change.
- Calibration changes must emit quality metrics or diagnostics.

## Workflow

1. Read `AGENTS.md`.
2. Check whether the change affects config schema, result schema, or CLI output.
3. Implement the smallest stable API surface that supports the requested behavior.
4. Run `pytest`, `ruff check .`, and `mypy src/calibrex`.
5. If schema models changed, regenerate:

```bash
calibrex schema config --output schemas/config.schema.json
calibrex schema result --output schemas/result.schema.json
```
