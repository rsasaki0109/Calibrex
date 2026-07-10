# ADR 0006: Rename to slac

## Status

Superseded by ADR 0009.

## Context

The project was originally named **Calibrex** (calibration evidence framework).
v0.2 made trajectory and odometry first-class elements: rosbag2 odometry
ingestion, motion compensation, and KISS-ICP rig-frame odometry. The scope
shifted from pure calibration evidence to simultaneous localization and
calibration (SLAC).

At that time, the GitHub repository was renamed from `rsasaki0109/Calibrex` to
`rsasaki0109/slac`. ADR 0009 later reversed this repository rename.

The name **slac** (lowercase) is deliberate. We are aware of the collision with
SLAC National Accelerator Laboratory; this project is unrelated robotics
software and uses lowercase branding to distinguish it.

The project is private, unpublished, and has no PyPI release or external users.
A wholesale breaking rename is acceptable with no compatibility shims.

## Decision

Rename the project to **slac** ("simultaneous localization and calibration"):

- Python package: `calibrex` → `slac` (`src/slac/`)
- CLI console script: `calibrex` → `slac`
- Schema IDs: `calibrex.<artifact>/vX.Y` → `slac.<artifact>/vX.Y`
- Provenance strings: `calibrex_native`, `calibrex_version`, and related keys
  → `slac_native`, `slac_version`, etc.
- Documentation, examples, CI, and issue templates updated to match.

No deprecation aliases, import shims, or schema-version translators are
provided. Configs, results, manifests, and cached artifacts must use the new
identifiers.

## Consequences

- All in-tree references to Calibrex as the product name are removed except
  historical mentions in this ADR and an optional one-line README note.
- Downstream forks or local clones must update imports, CLI invocations, and
  stored artifacts manually.
- Schema count remains 17; only the ID prefix changes.
