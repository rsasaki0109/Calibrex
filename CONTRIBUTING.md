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

## Optional adapter CI coverage

`tests/unit/test_optional_adapter_ci_coverage.py` checks registration of optional
CPU test files in CI without importing the adapters. A unit test containing a
literal `pytest.importorskip` for `cv2`, `numba`, `lz4`, `zstandard` or `mcap`
must be selected by its corresponding extra-enabled job in
`.github/workflows/ci.yml`. Update that job when adding a dependent test file.
Whole-file paths, directories and globs are supported; individual node selectors
do not establish whole-file coverage. Missing paths also fail the check.

Files gated by a top-level `pytest.importorskip("cupy")` are GPU-only and excluded
from this CPU registration check. A per-test GPU skip does not exclude the whole
file. This is a static guard for literal imports, not proof that every optional
test executed; dynamic dependency checks and skip results still need review.

## Compressed bag format tests

The `compressed-bag-formats` CI job installs optional MCAP, LZ4 and Zstandard
dependencies so their interoperability tests run rather than skip. Run it locally:

```bash
python -m pip install -e ".[dev,mcap,rosbag1-lz4,rosbag2-compression]"
python -m pytest tests/unit/test_bag_formats.py tests/unit/test_rosbag2.py \
  tests/unit/test_rosbag2_writer.py tests/unit/test_compressed_bag_corruption.py \
  tests/unit/test_mcap_field_bounds.py tests/unit/test_rosbag1_record_bounds.py
```

These tests use synthetic fixtures and do not require ROS, GPU hardware or public
dataset downloads. Missing-codec error tests simulate unavailable imports and
therefore also run when the compression extras are installed.

## OpenCV adapter tests

The `opencv-adapter` CI job installs `.[dev,opencv]` and covers hand-eye
calibration, visual rotation, focal estimation, camera/LiDAR initialization,
probabilistic PnP, fisheye rectification and correspondence-provider plumbing.
It verifies that `cv2` imports before running the tests. Hilti calibration-file
checks and PyTorch-dependent provider cases remain conditional on those separate
resources; this job does not download recordings, model weights or PyTorch.

## Numba adapter tests

The `numba-adapter` CI job installs `.[dev,numba]` and requires a successful Numba
import before running its tests. It covers compiled interpolation and LiDAR
linearization, deskew, residual summaries, rig caching, ordered parallel execution,
depth-to-depth and online-edge processing. The selected tests use synthetic
fixtures and compare compiled results with reference paths; GPU hardware and
recording downloads are not required. See the job's test list to reproduce the
same coverage locally.

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

When `CUPY_CACHE_DIR` is unset, the test suite places CuPy's compiled kernels in
the pytest cache before collecting tests. This keeps optional CUDA tests inside
the checkout's writable area, including restricted Windows environments where
writing to the default home cache can stall. Use `-o cache_dir=outputs/test-cache`
to choose the pytest cache location. An explicit `CUPY_CACHE_DIR` takes priority.
Tools that choose a per-worker cache keep that setting. Ordinary Calibrex runs
keep CuPy's normal cache configuration.

For CUDA runs outside pytest, set a writable cache before starting Python:

```powershell
$env:CUPY_CACHE_DIR = Join-Path (Get-Location) 'outputs/cupy-cache'
python -c "import cupy as cp; print(cp.arange(4).sum().item())"
```

In a POSIX shell, use `export CUPY_CACHE_DIR="$PWD/outputs/cupy-cache"`.

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
