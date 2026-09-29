# Contributing

Calibrex is schema-first and evaluation-first.

Every PR that changes calibration behavior must include:

- tests
- docs or examples when user-facing behavior changes
- schema updates when config or result shape changes
- evaluation impact when a solver, factor, frontend, or metric changes

Core rules:

- Do not add ROS message types to `src/calibrex/core`.
- Do not copy GPL code into the core package.
- Keep external tools behind adapters or subprocess boundaries.
- Preserve `T_parent_child`, meters, nanoseconds, and quaternion `xyzw`.
- Follow [license boundary policy](docs/concepts/license_boundaries.md) for
  external tools and public datasets.

## Schema Changes

Every artifact kind is declared once in `src/calibrex/core/schema_registry.py`.
Validation, `calibrex schema`, and the committed schema ledger are derived from
it. If a PR changes config, result, comparison, dataset manifest, report
sidecar, or any other artifact shape, regenerate committed schemas and the
ledger:

```bash
calibrex schema all --output-dir schemas
pytest tests/unit/test_schemas.py tests/unit/test_schema_registry.py
python tools/check_schema_compat.py
```

Schema files are part of the public API, and a released schema version must
stay readable. Adding an optional field is compatible. Anything that makes a
previously valid artifact invalid (a new required field, a removed field, a
tighter pattern, enum, or type) needs a new `schema_version`, and the model
must keep accepting the old one. When an old version cannot be read as is:

- add a `SchemaMigration` only when the upgrade is lossless and invents no
  values (digest-bound kinds cannot be migrated); otherwise
- list it in `RETIRED_SCHEMA_VERSIONS` with the retiring release, the reason,
  and the remedy users should apply.

`tools/check_schema_compat.py` enforces this against the latest release tag in
CI. A deliberate exception must be recorded, with a reviewable reason, in
`tools/schema_compat_acknowledged.yaml`. Do not change a field name, unit,
transform convention, or timestamp convention without updating examples,
tests, docs, and migration notes.

## Evidence and Evaluation

Calibration outputs should be reproducible evidence artifacts, not only
matrices. When adding or changing metrics, factors, solvers, or adapters:

- keep `candidate_extrinsics`, `reference_extrinsics`, and optimized
  `transforms` semantically separate
- record producer, execution mode, evidence level, dataset source, and command
  provenance when available
- add or update train/holdout, perturbation, degeneracy, or protocol
  compatibility checks
- avoid calling dataset-provided calibration ground truth unless it is
  documented as synthetic truth or independent metrology
- keep raw public datasets out of the repository; use manifests, download
  helpers, hashes, or small metadata fixtures instead

## Local Checks

Run the standard checks before opening a PR:

```bash
ruff check .
uvx --isolated --with-requirements tools/typecheck-requirements.txt mypy src/calibrex
pytest
```

The strict mypy gate runs from pinned requirements without numpy or scipy;
see `tools/typecheck-requirements.txt` for why a plain `mypy src/calibrex` in a
development environment aborts. The pytest configuration disables plugin
autoloading, so the suite also runs in a shell where ROS is sourced; if
Calibrex itself misbehaves there, `calibrex doctor` reports ROS paths on
`sys.path` and any dependency resolved from outside the active environment.

## Releases

Release tags use `v*`, for example `v0.1.0-alpha.1`. A tag or manual release
workflow builds wheel/sdist artifacts, installs the wheel in a clean virtual
environment, runs `calibrex doctor --json`, regenerates schemas, and creates a
draft prerelease on GitHub.

Before tagging, run the local release smoke when GitHub Actions is unavailable
or when package metadata changed:

```bash
python3 tools/local_release_smoke.py
```

The tag workflow creates a draft GitHub prerelease. After reviewing the draft,
`CHANGELOG.md`, wheel smoke results, license boundaries, and public dataset
notes, publishing the GitHub release triggers the PyPI trusted-publishing
workflow. The `pypi` GitHub environment and PyPI trusted publisher must be
configured before publishing the first release.
