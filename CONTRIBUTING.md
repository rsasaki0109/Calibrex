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

If a PR changes config, result, comparison, dataset manifest, or report sidecar
shape, regenerate committed schemas:

```bash
calibrex schema all --output-dir schemas
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
mypy src/calibrex
pytest
```

## Releases

Release tags use `v*`, for example `v0.1.0-alpha.1`. A tag or manual release
workflow builds wheel/sdist artifacts, installs the wheel in a clean virtual
environment, runs `calibrex doctor --json`, regenerates schemas, and creates a
draft prerelease on GitHub.

Before tagging, run the local release smoke when GitHub Actions is unavailable
or when package metadata changed:

```bash
rm -rf dist build src/calibrex.egg-info /tmp/calibrex-release-smoke
python -m build
python -m venv /tmp/calibrex-release-smoke
/tmp/calibrex-release-smoke/bin/python -m pip install dist/*.whl
/tmp/calibrex-release-smoke/bin/calibrex doctor --json
/tmp/calibrex-release-smoke/bin/calibrex schema all --output-dir /tmp/calibrex-release-schemas
test "$(find /tmp/calibrex-release-schemas -maxdepth 1 -name '*.schema.json' | wc -l)" -eq 9
/tmp/calibrex-release-smoke/bin/calibrex validate examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml --kind result --json
/tmp/calibrex-release-smoke/bin/calibrex report examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml --output-dir /tmp/calibrex-release-report --json
/tmp/calibrex-release-smoke/bin/calibrex validate /tmp/calibrex-release-report/evidence.json --kind report-evidence --json
```

The release workflow does not publish to PyPI. Publish release artifacts only
after reviewing the draft release, `CHANGELOG.md`, wheel smoke results, license
boundaries, and public dataset notes.
