# Contributing

slac is schema-first and evaluation-first.

Every PR that changes calibration behavior must include:

- tests
- docs or examples when user-facing behavior changes
- schema updates when config or result shape changes
- evaluation impact when a solver, factor, frontend, or metric changes

Core rules:

- Do not add ROS message types to `src/slac/core`.
- Do not copy GPL code into the core package.
- Keep external tools behind adapters or subprocess boundaries.
- Preserve `T_parent_child`, meters, nanoseconds, and quaternion `xyzw`.
- Follow [license boundary policy](docs/concepts/license_boundaries.md) for
  external tools and public datasets.

## Schema Changes

If a PR changes config, result, comparison, dataset manifest, or report sidecar
shape, regenerate committed schemas:

```bash
slac schema all --output-dir schemas
pytest tests/unit/test_schemas.py
```

Schema files are part of the public API. Do not change a schema field name,
unit, transform convention, or timestamp convention without updating examples,
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
mypy src/slac
pytest
```

## Releases

Release tags use `v*`, for example `v0.1.0-alpha.1`. A tag or manual release
workflow builds wheel/sdist artifacts, installs the wheel in a clean virtual
environment, runs `slac doctor --json`, regenerates schemas, and creates a
draft prerelease on GitHub.

Before tagging, run the local release smoke when GitHub Actions is unavailable
or when package metadata changed:

```bash
python3 tools/local_release_smoke.py
```

The release workflow does not publish to PyPI. Publish release artifacts only
after reviewing the draft release, `CHANGELOG.md`, wheel smoke results, license
boundaries, and public dataset notes.
